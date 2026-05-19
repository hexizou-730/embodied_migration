"""Run LMP code against a real ManiSkill environment.

The first supported real task is PickCube-v1. This command is expected to fail
gracefully on WSL machines where Vulkan/SAPIEN is unavailable, while remaining
ready for native Ubuntu execution.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from lmp.executor import execute_lmp

from .env_adapter import ManiSkillEnvAdapter
from .evaluation import TrialRecord, classify_failure
from .llm import gen_code
from .migration import MigrationRequest, build_migration_prompt, get_source_copy_code, norm_method
from .sim_check import diagnose_graphics_stack
from .skill_adapter import (
    ManiSkillPegInsertionRobot,
    ManiSkillPickCubeRobot,
    ManiSkillSceneAdapter,
    ManiSkillXArmPickCubePlannerRobot,
)
from .static_runner import _success_from_ret_val, build_oracle_code, build_static_report
from .tasks import get_task_spec


SUPPORTED_REAL_TASKS = ("PickCube-v1", "PegInsertionSide-v1")

DEFAULT_CONTROL_MODE: Dict[str, str] = {
    "PickCube-v1": "pd_ee_delta_pos",
    "PegInsertionSide-v1": "pd_ee_pose",
}


def _default_control_mode(task_id: str, robot_uid: str) -> str:
    if task_id == "PickCube-v1" and robot_uid.startswith("xarm6"):
        return "pd_joint_pos"
    return DEFAULT_CONTROL_MODE.get(task_id, "pd_ee_delta_pos")


def _build_robot_adapter(task_id: str, env: Any, control_mode: str, robot_uid: str) -> Any:
    if task_id == "PickCube-v1":
        if robot_uid.startswith("xarm6"):
            if control_mode in {"pd_joint_pos", "pd_joint_pos_vel"}:
                return ManiSkillXArmPickCubePlannerRobot(env, control_mode=control_mode)
            return ManiSkillPickCubeRobot(
                env,
                control_mode=control_mode,
                gripper_open=-1.0,
                gripper_close=1.0,
                move_steps=36,
                grip_steps=10,
                settle_steps=12,
            )
        return ManiSkillPickCubeRobot(env, control_mode=control_mode)
    if task_id == "PegInsertionSide-v1":
        return ManiSkillPegInsertionRobot(env, control_mode=control_mode)
    raise ValueError(f"No real skill adapter registered for task_id={task_id!r}")


def run_real_trial(
    *,
    task_id: str = "PickCube-v1",
    robot_uid: str = "panda",
    method: str = "source-copy",
    seed: int = 0,
    control_mode: Optional[str] = None,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 300,
    dry_run: bool = False,
) -> Dict[str, Any]:
    method = norm_method(method)
    task = get_task_spec(task_id)
    if control_mode is None:
        control_mode = _default_control_mode(task_id, robot_uid)
    result: Dict[str, Any] = {
        "task_id": task_id,
        "env_id": task.maniskill_env_id,
        "robot_uid": robot_uid,
        "method": method,
        "seed": seed,
        "real_runner": True,
    }
    if task_id not in SUPPORTED_REAL_TASKS:
        result.update(
            success=False,
            failure_type="execution failure",
            message=f"real runner currently supports only: {', '.join(SUPPORTED_REAL_TASKS)}",
        )
        return result

    request = MigrationRequest.from_ids(
        task_id=task_id,
        target_robot=robot_uid,
        method=method,
    )
    report_source_result = None
    report = None
    if method in {"llm_report_only", "llm_card_report"}:
        report_source_result = run_real_trial(
            task_id=task_id,
            robot_uid=robot_uid,
            method="source-copy",
            seed=seed,
            control_mode=control_mode,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
            max_episode_steps=max_episode_steps,
            dry_run=True,
        )
        if not bool(report_source_result.get("success", False)):
            report = build_static_report(
                task=request.task,
                target_profile=request.target_profile,
                failed_record=_result_to_report_record(report_source_result, source_robot=task.source_robot),
            )
            request = MigrationRequest(
                task=request.task,
                source_profile=request.source_profile,
                target_profile=request.target_profile,
                method=method,
                failure_report=report,
            )
    prompt = build_migration_prompt(request)
    if method == "source-copy":
        code = get_source_copy_code(task_id)
        llm_info: Dict[str, Any] = {}
    elif method == "oracle":
        code = build_oracle_code(task)
        llm_info = {}
    else:
        generated = gen_code(
            prompt=prompt,
            fallback_code=build_oracle_code(task),
            dry_run=dry_run,
        )
        code = generated.code
        llm_info = {
            "used_llm": generated.used_llm,
            "llm_model": generated.model,
            "llm_reason": generated.reason,
            "llm_raw_text": generated.raw_text,
        }

    adapter = ManiSkillEnvAdapter(
        task.maniskill_env_id,
        robot_uid=robot_uid,
        obs_mode=obs_mode,
        control_mode=control_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
    )
    robot = None
    try:
        env = adapter.make()
        obs, reset_info = adapter.reset(seed=seed)
        robot = _build_robot_adapter(task_id, env, control_mode, robot_uid)
        scene = ManiSkillSceneAdapter()
        code_ok, message, locals_dict = execute_lmp(
            code,
            {"scene": scene, "robot": robot},
            verbose=False,
        )
        ret_val = locals_dict.get("ret_val")
        success = bool(code_ok and _success_from_ret_val(ret_val))
        failure_message = message if success else _failure_message(robot, message)
        failure_type = classify_failure(
            success=success,
            code_ok=code_ok,
            message=failure_message,
            info=robot.last_info,
        )
        result.update(
            success=success,
            failure_type=failure_type,
            message=failure_message,
            generated_code=code,
            prompt=prompt,
            reset_info_keys=sorted(str(k) for k in getattr(reset_info, "keys", lambda: [])()),
            execution_log=robot.execution_log(),
            final_info=_jsonable(robot.last_info),
            **_report_info(report_source_result, report),
            **llm_info,
        )
    except Exception as exc:  # pragma: no cover - depends on local Vulkan/GPU stack
        result.update(
            success=False,
            failure_type="execution failure",
            message=repr(exc),
            generated_code=code,
            prompt=prompt,
            graphics_diagnosis=diagnose_graphics_stack(),
            **_report_info(report_source_result, report),
            **llm_info,
        )
    finally:
        if robot is not None and hasattr(robot, "close"):
            robot.close()
        adapter.close()
    return result


def _failure_message(robot: Any, fallback: str) -> str:
    for event in reversed(robot.execution_log()):
        if not event.get("ok"):
            return str(event.get("message") or fallback)
    return fallback


def _result_to_report_record(result: Dict[str, Any], *, source_robot: str) -> TrialRecord:
    return TrialRecord(
        task_id=str(result.get("task_id", "")),
        source_robot=source_robot,
        target_robot=str(result.get("robot_uid", "")),
        method=str(result.get("method", "source-copy")),
        seed=int(result.get("seed", 0)),
        generated_code=str(result.get("generated_code", "")),
        success=bool(result.get("success", False)),
        failure_type=str(result.get("failure_type", "unknown failure")),
        message=str(result.get("message", "")),
        prompt=str(result.get("prompt", "")),
        info={
            "execution_log": result.get("execution_log", []),
            "final_info": result.get("final_info", {}),
            "real_runner": True,
        },
    )


def _report_info(report_source_result: Optional[Dict[str, Any]], report: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    if report_source_result is not None:
        info.update(
            report_source_method=report_source_result.get("method"),
            report_source_failure_type=report_source_result.get("failure_type"),
            report_source_message=report_source_result.get("message"),
            report_source_log=report_source_result.get("execution_log", []),
        )
    if report is not None:
        info["failure_report"] = report.to_prompt_section()
    return info


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return repr(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real ManiSkill-backed LMP trials.")
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--robot", default="panda")
    parser.add_argument("--method", default="source-copy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control-mode",
        default=None,
        help="Override controller. If omitted, uses DEFAULT_CONTROL_MODE for the task.",
    )
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_real_trial(
        task_id=args.task,
        robot_uid=args.robot,
        method=args.method,
        seed=args.seed,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=repr))


if __name__ == "__main__":
    main()
