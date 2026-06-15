"""Run LMP code against a real ManiSkill environment."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from lmp.executor import execute_lmp

from .env_adapter import ManiSkillEnvAdapter
from .evaluation import TrialRecord, classify_failure, classify_failure_layer
from .llm import gen_code
from .migration import MigrationRequest, build_migration_prompt, get_source_copy_code, norm_method
from .reporting import build_oracle_code, build_real_failure_report, success_from_ret_val
from .sim_check import diagnose_graphics_stack
from .skill_adapter import (
    ManiSkillPickCubeRobot,
    ManiSkillPullCubeRobot,
    ManiSkillSceneAdapter,
)
from .tasks import get_task_spec


SUPPORTED_REAL_TASKS = ("pull_cube", "pick_cube")

DEFAULT_CONTROL_MODE: Dict[str, str] = {
    "pull_cube": "pd_ee_delta_pos",
    "pick_cube": "pd_ee_delta_pos",
}


def _default_control_mode(task_id: str, robot_uid: str) -> str:
    return DEFAULT_CONTROL_MODE.get(task_id, "pd_ee_delta_pos")


def _build_robot_adapter(task_id: str, env: Any, control_mode: str, robot_uid: str) -> Any:
    if task_id == "pull_cube":
        if robot_uid in {"panda", "fetch", "xarm6_robotiq"} and control_mode in {"pd_ee_delta_pos", "pd_ee_delta_pose"}:
            return ManiSkillPullCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
        raise ValueError(
            "PullCube real runner currently supports panda/fetch/xarm6_robotiq with "
            f"pd_ee_delta_* control, got robot={robot_uid!r}, control_mode={control_mode!r}."
        )
    if task_id == "pick_cube":
        if robot_uid in {"panda", "xarm6_robotiq"} and control_mode in {"pd_ee_delta_pos", "pd_ee_delta_pose"}:
            return ManiSkillPickCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
        raise ValueError(
            "PickCube real runner currently supports panda/xarm6_robotiq with "
            f"pd_ee_delta_* control, got robot={robot_uid!r}, control_mode={control_mode!r}."
        )
    raise ValueError(f"No real skill adapter registered for task_id={task_id!r}")


def _build_robot_adapter_from_module(adapter_module: str, env: Any, control_mode: str, robot_uid: str) -> Any:
    """Load a target-side adapter module generated for one migration case."""

    module = importlib.import_module(adapter_module)
    build_robot = getattr(module, "build_robot", None)
    if not callable(build_robot):
        raise ValueError(f"Adapter module {adapter_module!r} must define callable build_robot(...).")
    return build_robot(env, control_mode=control_mode, robot_uid=robot_uid)


def run_real_trial(
    *,
    task_id: str = "pull_cube",
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
    task_id = task.task_id
    requested_robot_uid = robot_uid
    robot_uid = _normalize_robot_uid(task_id, robot_uid)
    if control_mode is None:
        control_mode = _default_control_mode(task_id, robot_uid)
    result: Dict[str, Any] = {
        "task_id": task_id,
        "task_name": task.display_name,
        "task_name_cn": task.name_cn,
        "env_id": task.maniskill_env_id,
        "robot_uid": robot_uid,
        "method": method,
        "seed": seed,
        "control_mode": control_mode,
        "obs_mode": obs_mode,
        "sim_backend": sim_backend,
        "render_backend": render_backend,
        "max_episode_steps": max_episode_steps,
        "real_runner": True,
    }
    if requested_robot_uid != robot_uid:
        result["requested_robot_uid"] = requested_robot_uid
    if task_id not in SUPPORTED_REAL_TASKS:
        result.update(
            success=False,
            failure_type="execution failure",
            failure_layer="runtime_setup",
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
    if method == "llm_card_report":
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
            report = build_real_failure_report(
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

    return run_real_code_trial(
        task_id=task_id,
        robot_uid=robot_uid,
        method=method,
        code=code,
        prompt=prompt,
        seed=seed,
        control_mode=control_mode,
        obs_mode=obs_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
        extra_result={
            **_report_info(report_source_result, report),
            **llm_info,
        },
    )


def run_real_code_trial(
    *,
    task_id: str,
    robot_uid: str,
    method: str,
    code: str,
    prompt: str = "",
    seed: int = 0,
    control_mode: Optional[str] = None,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 300,
    extra_result: Optional[Dict[str, Any]] = None,
    adapter_module: str = "",
) -> Dict[str, Any]:
    """Execute a caller-provided LMP snippet in a real ManiSkill task."""

    task = get_task_spec(task_id)
    task_id = task.task_id
    requested_robot_uid = robot_uid
    robot_uid = _normalize_robot_uid(task_id, robot_uid)
    if control_mode is None:
        control_mode = _default_control_mode(task_id, robot_uid)
    result: Dict[str, Any] = {
        "task_id": task_id,
        "task_name": task.display_name,
        "task_name_cn": task.name_cn,
        "env_id": task.maniskill_env_id,
        "robot_uid": robot_uid,
        "method": method,
        "seed": seed,
        "control_mode": control_mode,
        "obs_mode": obs_mode,
        "sim_backend": sim_backend,
        "render_backend": render_backend,
        "max_episode_steps": max_episode_steps,
        "real_runner": True,
    }
    if adapter_module:
        result["adapter_module"] = adapter_module
    if requested_robot_uid != robot_uid:
        result["requested_robot_uid"] = requested_robot_uid
    if task_id not in SUPPORTED_REAL_TASKS:
        result.update(
            success=False,
            failure_type="execution failure",
            failure_layer="runtime_setup",
            message=f"real runner currently supports only: {', '.join(SUPPORTED_REAL_TASKS)}",
            generated_code=code,
            prompt=prompt,
        )
        if extra_result:
            result.update(extra_result)
        return result

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
        if adapter_module:
            robot = _build_robot_adapter_from_module(adapter_module, env, control_mode, robot_uid)
        else:
            robot = _build_robot_adapter(task_id, env, control_mode, robot_uid)
        initial_runtime_diagnostics = _runtime_diagnostics(task_id, robot, stage="initial")
        scene = ManiSkillSceneAdapter()
        code_ok, message, locals_dict = execute_lmp(
            code,
            {"scene": scene, "robot": robot},
            verbose=False,
        )
        ret_val = locals_dict.get("ret_val")
        success = bool(code_ok and success_from_ret_val(ret_val))
        failure_message = message if success else _failure_message(robot, message)
        failure_type = classify_failure(
            success=success,
            code_ok=code_ok,
            message=failure_message,
            info=robot.last_info,
        )
        execution_log = robot.execution_log()
        final_info = _jsonable(robot.last_info)
        failure_layer = classify_failure_layer(
            success=success,
            code_ok=code_ok,
            message=failure_message,
            info={
                "execution_log": execution_log,
                "final_info": final_info,
                "ret_val": ret_val,
            },
        )
        runtime_diagnostics = _runtime_diagnostics(
            task_id,
            robot,
            stage=_stage_from_message(failure_message),
        )
        result.update(
            success=success,
            failure_type=failure_type,
            failure_layer=failure_layer,
            message=failure_message,
            generated_code=code,
            prompt=prompt,
            reset_info_keys=sorted(str(k) for k in getattr(reset_info, "keys", lambda: [])()),
            execution_log=execution_log,
            final_info=final_info,
            initial_runtime_diagnostics=initial_runtime_diagnostics,
            runtime_diagnostics=runtime_diagnostics,
        )
        if extra_result:
            result.update(extra_result)
    except Exception as exc:  # pragma: no cover - depends on local Vulkan/GPU stack
        result.update(
            success=False,
            failure_type="execution failure",
            failure_layer="runtime_setup",
            message=repr(exc),
            generated_code=code,
            prompt=prompt,
            graphics_diagnosis=diagnose_graphics_stack(),
        )
        if extra_result:
            result.update(extra_result)
    finally:
        if robot is not None and hasattr(robot, "close"):
            robot.close()
        adapter.close()
    return result


def _runtime_diagnostics(task_id: str, robot: Any, *, stage: str) -> Dict[str, Any]:
    if task_id == "pull_cube":
        return _pull_cube_runtime_diagnostics(robot, stage=stage)
    return {}


def _stage_from_message(message: str) -> str:
    lower = str(message or "").lower()
    for stage in ("approach", "descent", "contact", "drag", "settle"):
        if stage in lower:
            return stage
    return "final"


def _pull_cube_runtime_diagnostics(robot: Any, *, stage: str) -> Dict[str, Any]:
    try:
        cube = _vector3(robot._actor_pos("cube"))
        goal = _vector3(robot._region_pos("goal"))
        tcp = _vector3(robot._tcp_pos())
    except Exception as exc:
        return {"stage": stage, "error": repr(exc)}

    x_offset = float(getattr(robot, "contact_x_offset_m", 0.07))
    z_offset = float(getattr(robot, "contact_z_offset_m", 0.02))
    drag_extra = 0.025
    nominal_contact = _add(cube, [x_offset, 0.0, z_offset])
    nominal_pre_contact = _add(nominal_contact, [0.0, 0.0, 0.075])
    nominal_drag_end = [goal[0] - drag_extra, cube[1], nominal_contact[2]]

    if stage == "approach":
        stage_target = nominal_pre_contact
    elif stage in {"descent", "contact"}:
        stage_target = nominal_contact
    elif stage == "drag":
        stage_target = nominal_drag_end
    else:
        stage_target = nominal_contact

    tcp_to_stage_target = _sub(stage_target, tcp)
    tcp_to_contact = _sub(nominal_contact, tcp)
    tcp_to_pre_contact = _sub(nominal_pre_contact, tcp)
    cube_to_goal_xy = [goal[0] - cube[0], goal[1] - cube[1]]
    tcp_to_cube_xy = [cube[0] - tcp[0], cube[1] - tcp[1]]

    return {
        "stage": stage,
        "cube_pos": _round_vec(cube),
        "goal_pos": _round_vec(goal),
        "tcp_pos": _round_vec(tcp),
        "contact_x_offset_m": round(x_offset, 5),
        "contact_z_offset_m": round(z_offset, 5),
        "nominal_contact_pos": _round_vec(nominal_contact),
        "nominal_pre_contact_pos": _round_vec(nominal_pre_contact),
        "nominal_drag_end_pos": _round_vec(nominal_drag_end),
        "stage_target_pos": _round_vec(stage_target),
        "tcp_stage_error_xyz": _round_vec(tcp_to_stage_target),
        "tcp_stage_error_norm": round(_norm(tcp_to_stage_target), 5),
        "tcp_contact_error_xyz": _round_vec(tcp_to_contact),
        "tcp_contact_error_norm": round(_norm(tcp_to_contact), 5),
        "tcp_pre_contact_error_xyz": _round_vec(tcp_to_pre_contact),
        "tcp_pre_contact_error_norm": round(_norm(tcp_to_pre_contact), 5),
        "cube_goal_xy": round(_norm(cube_to_goal_xy), 5),
        "tcp_cube_xy": round(_norm(tcp_to_cube_xy), 5),
    }


def _vector3(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flat.extend(float(x) for x in item)
        else:
            flat.append(float(item))
    return flat[:3]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b)]


def _norm(value: list[float]) -> float:
    return float(sum(float(x) * float(x) for x in value) ** 0.5)


def _round_vec(value: list[float]) -> list[float]:
    return [round(float(x), 5) for x in value]


def _failure_message(robot: Any, fallback: str) -> str:
    for event in reversed(robot.execution_log()):
        if not event.get("ok"):
            return str(event.get("message") or fallback)
    return fallback


def _normalize_robot_uid(task_id: str, robot_uid: str) -> str:
    return robot_uid


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
        failure_layer=str(result.get("failure_layer", "unknown")),
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
            report_source_failure_layer=report_source_result.get("failure_layer"),
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
    parser.add_argument("--task", default="pull_cube")
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
    parser.add_argument(
        "--code-file",
        default="",
        help="Execute target LMP code from a file instead of method-generated code.",
    )
    parser.add_argument(
        "--adapter-module",
        default="",
        help="Import a generated target adapter module with build_robot(...).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.code_file:
        code_path = Path(args.code_file)
        result = run_real_code_trial(
            task_id=args.task,
            robot_uid=args.robot,
            method=args.method,
            code=code_path.read_text(encoding="utf-8"),
            prompt=f"target code file: {code_path}",
            seed=args.seed,
            control_mode=args.control_mode,
            obs_mode=args.obs_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            max_episode_steps=args.max_episode_steps,
            adapter_module=args.adapter_module,
        )
        result["code_file"] = str(code_path)
    else:
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
