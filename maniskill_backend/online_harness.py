"""Online observe-decide-act harness for ManiSkill adapter migration.

The existing agent loop repairs adapters after a full episode finishes. This
module exposes a smaller online loop: observe current TCP/object/goal state,
choose one bounded primitive, execute a short segment with real ``env.step``,
then observe again before deciding the next segment.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.env_adapter import ManiSkillEnvAdapter
from maniskill_backend.llm import gen_text
from maniskill_backend.tasks import get_task_spec


ALLOWED_ONLINE_PRIMITIVES = (
    "move_to_pre_contact",
    "move_to_contact",
    "drag_toward_goal",
    "hold",
    "stop",
)


@dataclass(frozen=True)
class OnlinePullConfig:
    """Small set of online-control parameters for PullCube."""

    contact_x_offset_m: float = 0.12
    contact_z_offset_m: float = 0.015
    approach_height_m: float = 0.075
    max_delta_m: float = 0.04
    max_action: float = 0.85
    drag_strength: float = 0.55
    down_bias: float = -0.08
    gripper_close: float = -1.0
    approach_tolerance_m: float = 0.035
    contact_tolerance_m: float = 0.018
    goal_tolerance_m: float = 0.09
    settle_steps: int = 4


def build_pull_cube_online_observation(
    *,
    cube_pos: Iterable[float],
    goal_pos: Iterable[float],
    tcp_pos: Iterable[float],
    step_index: int = 0,
    stage: str = "observe",
    success: bool = False,
    config: OnlinePullConfig | None = None,
) -> Dict[str, Any]:
    """Build a JSON-friendly online PullCube observation from current vectors."""

    cfg = config or OnlinePullConfig()
    cube = np.asarray(list(cube_pos), dtype=np.float32)[:3]
    goal = np.asarray(list(goal_pos), dtype=np.float32)[:3]
    tcp = np.asarray(list(tcp_pos), dtype=np.float32)[:3]
    goal_vec_xy = goal[:2] - cube[:2]
    cube_goal_xy = float(np.linalg.norm(goal_vec_xy))
    if cube_goal_xy > 1e-8:
        goal_dir_xy = goal_vec_xy / cube_goal_xy
    else:
        goal_dir_xy = np.array([-1.0, 0.0], dtype=np.float32)

    # Contact must start from the far side of the cube, opposite the goal
    # direction. For the default seed this is the positive-x side.
    far_side_xy = -goal_dir_xy
    contact = np.array(
        [
            cube[0] + far_side_xy[0] * cfg.contact_x_offset_m,
            cube[1] + far_side_xy[1] * cfg.contact_x_offset_m,
            cube[2] + cfg.contact_z_offset_m,
        ],
        dtype=np.float32,
    )
    pre_contact = contact + np.array([0.0, 0.0, cfg.approach_height_m], dtype=np.float32)
    tcp_pre_error = pre_contact - tcp
    tcp_contact_error = contact - tcp
    tcp_cube_xy = float(np.linalg.norm(tcp[:2] - cube[:2]))

    return {
        "schema": "online_pull_cube_observation.v1",
        "step_index": int(step_index),
        "stage": stage,
        "success": bool(success),
        "positions": {
            "cube": _round_vec(cube),
            "goal": _round_vec(goal),
            "tcp": _round_vec(tcp),
        },
        "targets": {
            "pre_contact": _round_vec(pre_contact),
            "contact": _round_vec(contact),
        },
        "metrics": {
            "cube_goal_xy": round(cube_goal_xy, 5),
            "tcp_cube_xy": round(tcp_cube_xy, 5),
            "tcp_pre_contact_error_norm": round(float(np.linalg.norm(tcp_pre_error)), 5),
            "tcp_contact_error_norm": round(float(np.linalg.norm(tcp_contact_error)), 5),
            "tcp_pre_contact_error_xyz": _round_vec(tcp_pre_error),
            "tcp_contact_error_xyz": _round_vec(tcp_contact_error),
            "goal_dir_xy": _round_vec(goal_dir_xy),
            "far_side_xy": _round_vec(far_side_xy),
        },
        "allowed_primitives": list(ALLOWED_ONLINE_PRIMITIVES),
    }


def fallback_online_pull_action(
    observation: Mapping[str, Any],
    *,
    config: OnlinePullConfig | None = None,
) -> Dict[str, Any]:
    """Deterministic online policy used without an LLM or as validation fallback."""

    cfg = config or OnlinePullConfig()
    metrics = observation.get("metrics") or {}
    pre_contact_error = float(metrics.get("tcp_pre_contact_error_norm", 0.0))
    contact_error = float(metrics.get("tcp_contact_error_norm", 999.0))
    cube_goal_xy = float(metrics.get("cube_goal_xy", 999.0))
    if bool(observation.get("success")):
        return {"primitive": "stop", "args": {"reason": "success"}}
    if contact_error <= cfg.contact_tolerance_m and cube_goal_xy > cfg.goal_tolerance_m:
        return {
            "primitive": "drag_toward_goal",
            "args": {"drag_strength": cfg.drag_strength, "down_bias": cfg.down_bias},
        }
    if pre_contact_error > cfg.approach_tolerance_m:
        return {"primitive": "move_to_pre_contact", "args": {}}
    if contact_error > cfg.contact_tolerance_m:
        return {"primitive": "move_to_contact", "args": {}}
    if cube_goal_xy > cfg.goal_tolerance_m:
        return {
            "primitive": "drag_toward_goal",
            "args": {"drag_strength": cfg.drag_strength, "down_bias": cfg.down_bias},
        }
    return {"primitive": "hold", "args": {"steps": cfg.settle_steps}}


def plan_online_pull_action(
    observation: Mapping[str, Any],
    *,
    planner: str = "fallback",
    dry_run: bool = False,
    config: OnlinePullConfig | None = None,
) -> Dict[str, Any]:
    """Choose the next online primitive from current observation."""

    fallback = fallback_online_pull_action(observation, config=config)
    if planner == "fallback":
        return {"schema": "online_action_plan.v1", "used_llm": False, **fallback}

    generated = gen_text(
        prompt=build_online_planner_prompt(observation),
        system="You are an online robot-control planner. Return only JSON.",
        fallback_text=json.dumps(fallback),
        dry_run=dry_run,
    )
    raw = generated.text or generated.raw_text or ""
    try:
        parsed = json.loads(_strip_json(raw))
    except Exception:
        parsed = fallback
    validated = validate_online_action(parsed, observation)
    validated.update(
        schema="online_action_plan.v1",
        used_llm=bool(generated.used_llm),
        llm_model=generated.model,
        llm_reason=generated.reason,
    )
    return validated


def build_online_planner_prompt(observation: Mapping[str, Any]) -> str:
    """Prompt for a bounded online primitive planner."""

    return (
        "You are controlling a ManiSkill PullCube adapter online.\n"
        "You receive the latest simulator observation after real env.step calls.\n"
        "Choose exactly one bounded primitive from observation.allowed_primitives.\n"
        "Do not invent APIs and do not modify simulator state.\n\n"
        "Primitive meanings:\n"
        "- move_to_pre_contact: move TCP above the far-side contact pose.\n"
        "- move_to_contact: descend to the contact pose.\n"
        "- drag_toward_goal: apply short drag pulses toward the goal direction.\n"
        "- hold: send near-zero action for a few steps.\n"
        "- stop: stop when success is true or no useful primitive remains.\n\n"
        "Return only JSON:\n"
        '{"primitive": "move_to_pre_contact", "args": {}}\n\n'
        "Observation JSON:\n"
        f"{json.dumps(observation, indent=2, ensure_ascii=False)}"
    )


def validate_online_action(plan: Mapping[str, Any], observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an online primitive plan against the observation surface."""

    allowed = set(observation.get("allowed_primitives") or ALLOWED_ONLINE_PRIMITIVES)
    primitive = str(plan.get("primitive") or "").strip()
    if primitive not in allowed:
        return fallback_online_pull_action(observation)
    args = plan.get("args") or {}
    if not isinstance(args, Mapping):
        args = {}
    return {"primitive": primitive, "args": dict(args)}


class OnlinePullCubeHarness:
    """Online closed-loop PullCube runner using short observe/act segments."""

    def __init__(
        self,
        robot: Any,
        *,
        planner: str = "fallback",
        segment_steps: int = 8,
        config: OnlinePullConfig | None = None,
        dry_run: bool = False,
    ) -> None:
        self.robot = robot
        self.planner = planner
        self.segment_steps = max(1, int(segment_steps))
        self.config = config or OnlinePullConfig()
        self.dry_run = bool(dry_run)
        self.step_index = 0
        self.trace: List[Dict[str, Any]] = []

    def observe(self, *, stage: str) -> Dict[str, Any]:
        success = bool(self.robot._pull_cube_success())
        return build_pull_cube_online_observation(
            cube_pos=self.robot._actor_pos("cube"),
            goal_pos=self.robot._region_pos("goal"),
            tcp_pos=self.robot._tcp_pos(),
            step_index=self.step_index,
            stage=stage,
            success=success,
            config=self.config,
        )

    def run(self, *, max_online_steps: int) -> Dict[str, Any]:
        final_observation: Dict[str, Any] = {}
        while self.step_index < max_online_steps and not self.robot._early_stop():
            observation = self.observe(stage="decide")
            action = plan_online_pull_action(
                observation,
                planner=self.planner,
                dry_run=self.dry_run,
                config=self.config,
            )
            record = {
                "type": "decision",
                "step_index": self.step_index,
                "observation": observation,
                "action": action,
            }
            self.trace.append(record)
            primitive = str(action.get("primitive") or "stop")
            if primitive == "stop" or observation.get("success"):
                final_observation = observation
                break
            executed = self.execute_primitive(primitive, action.get("args") or {})
            self.trace.append(
                {
                    "type": "execution",
                    "primitive": primitive,
                    "steps_executed": executed,
                    "step_index": self.step_index,
                    "observation": self.observe(stage=primitive),
                }
            )
            final_observation = self.trace[-1]["observation"]

        if not final_observation:
            final_observation = self.observe(stage="final")
        success = bool(final_observation.get("success"))
        return {
            "schema": "online_harness_result.v1",
            "task_id": "pull_cube",
            "success": success,
            "elapsed_online_steps": self.step_index,
            "planner": self.planner,
            "final_observation": final_observation,
            "num_decisions": len([item for item in self.trace if item.get("type") == "decision"]),
        }

    def execute_primitive(self, primitive: str, args: Mapping[str, Any]) -> int:
        if primitive == "move_to_pre_contact":
            return self._execute_move_target("pre_contact")
        if primitive == "move_to_contact":
            return self._execute_move_target("contact")
        if primitive == "drag_toward_goal":
            return self._execute_drag(args)
        if primitive == "hold":
            return self._execute_hold(int(args.get("steps") or self.config.settle_steps))
        return 0

    def _execute_move_target(self, target_key: str) -> int:
        executed = 0
        for _ in range(self.segment_steps):
            if self.robot._early_stop():
                break
            obs = self.observe(stage=target_key)
            target = np.asarray(obs["targets"][target_key], dtype=np.float32)
            tcp = np.asarray(obs["positions"]["tcp"], dtype=np.float32)
            delta = target - tcp
            if float(np.linalg.norm(delta)) < self.config.contact_tolerance_m:
                break
            command = np.clip(delta / self.config.max_delta_m, -self.config.max_action, self.config.max_action)
            self._step(command)
            executed += 1
        return executed

    def _execute_drag(self, args: Mapping[str, Any]) -> int:
        executed = 0
        drag_strength = float(args.get("drag_strength") or self.config.drag_strength)
        down_bias = float(args.get("down_bias") or self.config.down_bias)
        for _ in range(self.segment_steps):
            if self.robot._early_stop():
                break
            obs = self.observe(stage="drag")
            if obs.get("success"):
                break
            goal_dir = np.asarray(obs["metrics"]["goal_dir_xy"], dtype=np.float32)
            command = np.array([goal_dir[0] * abs(drag_strength), goal_dir[1] * abs(drag_strength), down_bias], dtype=np.float32)
            command = np.clip(command, -self.config.max_action, self.config.max_action)
            self._step(command)
            executed += 1
        return executed

    def _execute_hold(self, steps: int) -> int:
        executed = 0
        for _ in range(max(1, steps)):
            if self.robot._early_stop():
                break
            self._step(np.zeros(3, dtype=np.float32))
            executed += 1
        return executed

    def _step(self, delta_xyz: np.ndarray) -> None:
        action = _make_robot_action(self.robot, delta_xyz, gripper=self.config.gripper_close)
        self.robot._step(action)
        self.step_index += 1


def run_online_pull_cube_case(
    *,
    case_id: str,
    seed: int = 0,
    planner: str = "fallback",
    segment_steps: int = 8,
    max_online_steps: int = 240,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 500,
    adapter_module: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run online PullCube harness for a registered migration case."""

    case = get_full_migration_case(case_id)
    if case.task_id != "pull_cube":
        raise ValueError("online harness currently supports PullCube cases only.")
    if dry_run:
        return _dry_run_online_payload(case, planner=planner, segment_steps=segment_steps, max_online_steps=max_online_steps)

    task = get_task_spec(case.task_id)
    env_adapter = ManiSkillEnvAdapter(
        task.maniskill_env_id,
        robot_uid=case.target_robot,
        obs_mode=obs_mode,
        control_mode=case.target_control_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
    )
    robot = None
    try:
        env = env_adapter.make()
        _, reset_info = env_adapter.reset(seed=seed)
        robot = _build_robot_from_module(
            adapter_module or case.target_adapter_module,
            env,
            control_mode=case.target_control_mode,
            robot_uid=case.target_robot,
        )
        harness = OnlinePullCubeHarness(
            robot,
            planner=planner,
            segment_steps=segment_steps,
            dry_run=False,
        )
        result = harness.run(max_online_steps=max_online_steps)
        result.update(
            case_id=case.case_id,
            source_robot=case.source_robot,
            target_robot=case.target_robot,
            seed=seed,
            reset_info_keys=sorted(str(k) for k in getattr(reset_info, "keys", lambda: [])()),
            trace=harness.trace,
            final_info=_jsonable(getattr(robot, "last_info", {})),
            execution_log=robot.execution_log(),
        )
        return result
    finally:
        if robot is not None and hasattr(robot, "close"):
            robot.close()
        env_adapter.close()


def write_online_outputs(output_dir: Path, payload: Mapping[str, Any]) -> Dict[str, str]:
    """Write online summary/trace artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "online_trace.jsonl"
    md_path = output_dir / "summary.md"
    trace = list(payload.get("trace") or [])
    summary = dict(payload)
    summary.pop("trace", None)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
    with trace_path.open("w", encoding="utf-8") as f:
        for item in trace:
            f.write(json.dumps(item, ensure_ascii=False, default=repr) + "\n")
    md_path.write_text(_online_markdown(summary, trace), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "trace_jsonl": str(trace_path),
        "markdown": str(md_path),
    }


def _online_markdown(summary: Mapping[str, Any], trace: List[Mapping[str, Any]]) -> str:
    lines = [
        "# Online Harness Run",
        "",
        f"- case: `{summary.get('case_id')}`",
        f"- planner: `{summary.get('planner')}`",
        f"- success: `{summary.get('success')}`",
        f"- elapsed_online_steps: `{summary.get('elapsed_online_steps')}`",
        f"- num_decisions: `{summary.get('num_decisions')}`",
        "",
        "## Decision Trace",
        "",
        "| decision | step | primitive | cube_goal_xy | tcp_contact_error |",
        "|---:|---:|---|---:|---:|",
    ]
    idx = 0
    for item in trace:
        if item.get("type") != "decision":
            continue
        idx += 1
        obs = item.get("observation") or {}
        metrics = obs.get("metrics") or {}
        action = item.get("action") or {}
        lines.append(
            f"| {idx} | {item.get('step_index')} | {action.get('primitive')} | "
            f"{metrics.get('cube_goal_xy')} | {metrics.get('tcp_contact_error_norm')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _dry_run_online_payload(
    case: FullMigrationCase,
    *,
    planner: str,
    segment_steps: int,
    max_online_steps: int,
) -> Dict[str, Any]:
    observation = build_pull_cube_online_observation(
        cube_pos=[0.0, 0.05, 0.02],
        goal_pos=[-0.2, 0.05, 0.001],
        tcp_pos=[0.0, 0.0, 0.16],
        step_index=0,
        stage="dry_run",
        success=False,
    )
    action = plan_online_pull_action(observation, planner="fallback", dry_run=True)
    return {
        "schema": "online_harness_result.v1",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "source_robot": case.source_robot,
        "target_robot": case.target_robot,
        "success": None,
        "dry_run": True,
        "planner": planner,
        "segment_steps": segment_steps,
        "max_online_steps": max_online_steps,
        "initial_observation": observation,
        "first_action": action,
        "message": "dry run: online harness would observe, choose one primitive, execute a short segment, then observe again",
        "trace": [{"type": "decision", "step_index": 0, "observation": observation, "action": action}],
    }


def _build_robot_from_module(adapter_module: str, env: Any, *, control_mode: str, robot_uid: str) -> Any:
    module = importlib.import_module(adapter_module)
    build_robot = getattr(module, "build_robot", None)
    if not callable(build_robot):
        raise ValueError(f"Adapter module {adapter_module!r} must define build_robot(...).")
    return build_robot(env, control_mode=control_mode, robot_uid=robot_uid)


def _make_robot_action(robot: Any, delta_xyz: np.ndarray, *, gripper: float) -> Any:
    make_action = getattr(robot, "_make_action")
    signature = inspect.signature(make_action)
    if "base" in signature.parameters:
        return make_action(delta_xyz, gripper=gripper, base=np.zeros(2, dtype=np.float32))
    return make_action(delta_xyz, gripper=gripper)


def _strip_json(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        import re

        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return stripped


def _round_vec(value: Any) -> List[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [round(float(x), 5) for x in arr.tolist()]


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
