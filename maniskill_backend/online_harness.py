"""Online observe-reason-act harness for ManiSkill tasks.

Unlike episode-level adapter repair, this harness keeps the same simulator
episode alive. After every short action segment it reads the current physical
state, asks a planner for one bounded semantic primitive, executes that
primitive through real ``env.step(action)`` calls, and observes again.

The high-level task, low-level controller, simulator, and success signal remain
fixed. The planner can only choose from the task's declared semantic tools.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.env_adapter import ManiSkillEnvAdapter
from maniskill_backend.llm import gen_text
from maniskill_backend.tasks import get_task_spec


PULL_ONLINE_PRIMITIVES = (
    "move_to_pre_contact",
    "move_to_contact",
    "drag_toward_goal",
    "hold",
    "stop",
)

PICK_ONLINE_PRIMITIVES = (
    "open_gripper",
    "move_to_pre_grasp",
    "move_to_grasp",
    "close_gripper",
    "lift_cube",
    "move_to_goal",
    "release_gripper",
    "hold",
    "stop",
)

# Backward-compatible name used by earlier tests and scripts.
ALLOWED_ONLINE_PRIMITIVES = PULL_ONLINE_PRIMITIVES


@dataclass(frozen=True)
class OnlinePullConfig:
    """Bounded task-level parameters for PullCube."""

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


@dataclass(frozen=True)
class OnlinePickConfig:
    """Bounded task-level parameters for PickCube."""

    grasp_z_offset_m: float = 0.0
    approach_height_m: float = 0.10
    lift_height_m: float = 0.10
    max_delta_m: float = 0.045
    max_action: float = 0.8
    gripper_open: float = 1.0
    gripper_close: float = -1.0
    approach_tolerance_m: float = 0.025
    grasp_tolerance_m: float = 0.015
    goal_tolerance_m: float = 0.05
    minimum_lift_m: float = 0.045
    settle_steps: int = 6
    close_steps: int = 12
    max_grasp_attempts: int = 3
    grasp_z_candidates: Sequence[float] = (0.0, 0.012, 0.016)


def build_pull_cube_online_observation(
    *,
    cube_pos: Iterable[float],
    goal_pos: Iterable[float],
    tcp_pos: Iterable[float],
    step_index: int = 0,
    stage: str = "observe",
    success: bool = False,
    config: OnlinePullConfig | None = None,
    last_action: Mapping[str, Any] | None = None,
    task_id: str = "pull_cube",
) -> Dict[str, Any]:
    """Build one machine-facing planar contact-task observation."""

    cfg = config or OnlinePullConfig()
    cube = _vec3(cube_pos)
    goal = _vec3(goal_pos)
    tcp = _vec3(tcp_pos)
    goal_vec_xy = goal[:2] - cube[:2]
    cube_goal_xy = float(np.linalg.norm(goal_vec_xy))
    goal_dir_xy = (
        goal_vec_xy / cube_goal_xy
        if cube_goal_xy > 1e-8
        else np.array([-1.0, 0.0], dtype=np.float32)
    )
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

    return {
        "schema": (
            "online_push_cube_observation.v1"
            if task_id == "push_cube"
            else "online_pull_cube_observation.v1"
        ),
        "task_id": task_id,
        "step_index": int(step_index),
        "stage": str(stage),
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
            "tcp_cube_xy": round(float(np.linalg.norm(tcp[:2] - cube[:2])), 5),
            "tcp_pre_contact_error_norm": round(float(np.linalg.norm(tcp_pre_error)), 5),
            "tcp_contact_error_norm": round(float(np.linalg.norm(tcp_contact_error)), 5),
            "tcp_pre_contact_error_xyz": _round_vec(tcp_pre_error),
            "tcp_contact_error_xyz": _round_vec(tcp_contact_error),
            "goal_dir_xy": _round_vec(goal_dir_xy),
            "far_side_xy": _round_vec(far_side_xy),
        },
        "tool_state": {
            "contact_x_offset_m": round(cfg.contact_x_offset_m, 5),
            "contact_z_offset_m": round(cfg.contact_z_offset_m, 5),
            "approach_height_m": round(cfg.approach_height_m, 5),
        },
        "last_action": dict(last_action or {}),
        "allowed_primitives": list(PULL_ONLINE_PRIMITIVES),
        "semantic_tools": _semantic_tool_specs(task_id),
    }


def build_pick_cube_online_observation(
    *,
    cube_pos: Iterable[float],
    goal_pos: Iterable[float],
    tcp_pos: Iterable[float],
    is_grasping: bool,
    step_index: int = 0,
    stage: str = "observe",
    phase: str = "start",
    success: bool = False,
    gripper_command: float = 1.0,
    grasp_attempt: int = 0,
    initial_cube_z: float | None = None,
    current_grasp_z_offset: float | None = None,
    config: OnlinePickConfig | None = None,
    last_action: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build one machine-facing PickCube observation."""

    cfg = config or OnlinePickConfig()
    cube = _vec3(cube_pos)
    goal = _vec3(goal_pos)
    tcp = _vec3(tcp_pos)
    z_offset = cfg.grasp_z_offset_m if current_grasp_z_offset is None else float(current_grasp_z_offset)
    grasp = cube + np.array([0.0, 0.0, z_offset], dtype=np.float32)
    pre_grasp = grasp + np.array([0.0, 0.0, cfg.approach_height_m], dtype=np.float32)
    transport_tcp = tcp + (goal - cube)
    initial_z = float(cube[2] if initial_cube_z is None else initial_cube_z)
    cube_goal_xyz = float(np.linalg.norm(goal - cube))
    tcp_grasp_error = grasp - tcp
    tcp_pre_error = pre_grasp - tcp

    return {
        "schema": "online_task_observation.v2",
        "task_id": "pick_cube",
        "step_index": int(step_index),
        "stage": str(stage),
        "phase": str(phase),
        "success": bool(success),
        "positions": {
            "cube": _round_vec(cube),
            "goal": _round_vec(goal),
            "tcp": _round_vec(tcp),
        },
        "targets": {
            "pre_grasp": _round_vec(pre_grasp),
            "grasp": _round_vec(grasp),
            "transport_tcp": _round_vec(transport_tcp),
        },
        "metrics": {
            "cube_goal_xyz": round(cube_goal_xyz, 5),
            "tcp_cube_xyz": round(float(np.linalg.norm(tcp - cube)), 5),
            "tcp_pre_grasp_error_norm": round(float(np.linalg.norm(tcp_pre_error)), 5),
            "tcp_grasp_error_norm": round(float(np.linalg.norm(tcp_grasp_error)), 5),
            "tcp_grasp_xy": round(float(np.linalg.norm(tcp[:2] - grasp[:2])), 5),
            "tcp_grasp_z": round(abs(float(tcp[2] - grasp[2])), 5),
            "cube_lift_delta_z": round(float(cube[2] - initial_z), 5),
            "is_grasping": bool(is_grasping),
        },
        "tool_state": {
            "gripper_command": round(float(gripper_command), 4),
            "grasp_attempt": int(grasp_attempt),
            "max_grasp_attempts": int(cfg.max_grasp_attempts),
            "current_grasp_z_offset": round(z_offset, 5),
        },
        "last_action": dict(last_action or {}),
        "allowed_primitives": list(PICK_ONLINE_PRIMITIVES),
        "semantic_tools": _semantic_tool_specs("pick_cube"),
    }


def fallback_online_pull_action(
    observation: Mapping[str, Any],
    *,
    config: OnlinePullConfig | None = None,
) -> Dict[str, Any]:
    """Deterministic PullCube planner used when no LLM is available."""

    cfg = config or OnlinePullConfig()
    metrics = observation.get("metrics") or {}
    pre_contact_error = float(metrics.get("tcp_pre_contact_error_norm", 999.0))
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


def fallback_online_pick_action(
    observation: Mapping[str, Any],
    *,
    config: OnlinePickConfig | None = None,
) -> Dict[str, Any]:
    """Deterministic PickCube planner used when no LLM is available."""

    cfg = config or OnlinePickConfig()
    metrics = observation.get("metrics") or {}
    state = observation.get("tool_state") or {}
    phase = str(observation.get("phase") or "start")
    grasping = bool(metrics.get("is_grasping"))
    attempt = int(state.get("grasp_attempt") or 0)
    gripper_command = float(state.get("gripper_command", cfg.gripper_open))

    if bool(observation.get("success")):
        return {"primitive": "stop", "args": {"reason": "success"}}
    if grasping:
        if float(metrics.get("cube_lift_delta_z", 0.0)) < cfg.minimum_lift_m:
            return {"primitive": "lift_cube", "args": {"lift_height": cfg.lift_height_m}}
        if float(metrics.get("cube_goal_xyz", 999.0)) > cfg.goal_tolerance_m:
            return {"primitive": "move_to_goal", "args": {}}
        return {"primitive": "hold", "args": {"steps": cfg.settle_steps}}

    if phase == "close_failed":
        if attempt >= cfg.max_grasp_attempts:
            return {"primitive": "stop", "args": {"reason": "grasp_attempt_budget_exhausted"}}
        if gripper_command < 0.0:
            return {"primitive": "open_gripper", "args": {"steps": cfg.settle_steps}}

    candidate_index = min(attempt, len(cfg.grasp_z_candidates) - 1)
    candidate_z = float(cfg.grasp_z_candidates[candidate_index])
    if (
        phase not in {"at_grasp"}
        and float(metrics.get("tcp_pre_grasp_error_norm", 999.0)) > cfg.approach_tolerance_m
    ):
        return {
            "primitive": "move_to_pre_grasp",
            "args": {"grasp_z_offset": candidate_z},
        }
    if float(metrics.get("tcp_grasp_error_norm", 999.0)) > cfg.grasp_tolerance_m:
        return {
            "primitive": "move_to_grasp",
            "args": {"grasp_z_offset": candidate_z},
        }
    return {
        "primitive": "close_gripper",
        "args": {"close_command": cfg.gripper_close, "steps": cfg.close_steps},
    }


def fallback_online_action(observation: Mapping[str, Any]) -> Dict[str, Any]:
    task_id = str(observation.get("task_id") or "pull_cube")
    if task_id == "pick_cube":
        return fallback_online_pick_action(observation)
    return fallback_online_pull_action(observation)


def plan_online_action(
    observation: Mapping[str, Any],
    *,
    planner: str = "llm",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Choose exactly one bounded semantic primitive from the latest state."""

    fallback = fallback_online_action(observation)
    if planner == "fallback":
        return {"schema": "online_action_plan.v2", "used_llm": False, **fallback}

    generated = gen_text(
        prompt=build_online_planner_prompt(observation),
        system=(
            "You are an embodied task agent operating through bounded semantic tools. "
            "Return only one valid JSON tool call."
        ),
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
        schema="online_action_plan.v2",
        used_llm=bool(generated.used_llm),
        llm_model=generated.model,
        llm_reason=generated.reason,
    )
    return validated


def plan_online_pull_action(
    observation: Mapping[str, Any],
    *,
    planner: str = "fallback",
    dry_run: bool = False,
    config: OnlinePullConfig | None = None,
) -> Dict[str, Any]:
    """Backward-compatible PullCube planner wrapper."""

    if planner == "fallback":
        fallback = fallback_online_pull_action(observation, config=config)
        return {"schema": "online_action_plan.v2", "used_llm": False, **fallback}
    return plan_online_action(observation, planner=planner, dry_run=dry_run)


def build_online_planner_prompt(observation: Mapping[str, Any]) -> str:
    """Build a compact GUAVA-style task prompt from one live observation."""

    task_id = str(observation.get("task_id") or "pull_cube")
    objective = (
        "Grasp the cube, preserve the grasp, and move the cube to the 3D goal."
        if task_id == "pick_cube"
        else "Establish contact on the far side and move the cube into the goal region."
    )
    return (
        "You are acting inside one live ManiSkill episode.\n"
        "The JSON below was measured after real env.step(action) calls.\n"
        "Choose one semantic tool, execute only a short segment, then wait for the next observation.\n\n"
        f"Task objective: {objective}\n\n"
        "Hard constraints:\n"
        "- Keep the high-level task, controller, simulator, and success signal unchanged.\n"
        "- Use exactly one tool listed in semantic_tools.\n"
        "- Do not invent APIs, joint commands, or direct object-state edits.\n"
        "- Use measured errors and grasp/contact state; do not assume the last action succeeded.\n"
        "- Return stop only on success or a measured exhausted budget.\n\n"
        "Return only JSON:\n"
        '{"primitive": "tool_name", "args": {}}\n\n'
        "Live observation:\n"
        f"{json.dumps(observation, indent=2, ensure_ascii=False)}"
    )


def validate_online_action(plan: Mapping[str, Any], observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and clamp a planner tool call."""

    allowed = set(observation.get("allowed_primitives") or ())
    primitive = str(plan.get("primitive") or "").strip()
    if primitive not in allowed:
        return fallback_online_action(observation)
    args = plan.get("args") or {}
    if not isinstance(args, Mapping):
        args = {}
    return {"primitive": primitive, "args": _sanitize_online_args(primitive, args)}


class _OnlineHarnessBase:
    """Shared short-segment decision loop."""

    task_id = ""

    def __init__(
        self,
        robot: Any,
        *,
        planner: str,
        segment_steps: int,
        dry_run: bool = False,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.robot = robot
        self.planner = planner
        self.segment_steps = max(1, int(segment_steps))
        self.dry_run = bool(dry_run)
        self.event_sink = event_sink
        self.step_index = 0
        self.trace: List[Dict[str, Any]] = []
        self.last_action: Dict[str, Any] = {}

    def observe(self, *, stage: str) -> Dict[str, Any]:
        raise NotImplementedError

    def execute_primitive(self, primitive: str, args: Mapping[str, Any]) -> int:
        raise NotImplementedError

    def run(self, *, max_online_steps: int) -> Dict[str, Any]:
        final_observation: Dict[str, Any] = {}
        while self.step_index < max_online_steps and not self.robot._early_stop():
            observation = self.observe(stage="decide")
            action = plan_online_action(
                observation,
                planner=self.planner,
                dry_run=self.dry_run,
            )
            decision = {
                "type": "decision",
                "step_index": self.step_index,
                "observation": observation,
                "action": action,
            }
            self._record(decision)
            primitive = str(action.get("primitive") or "stop")
            if primitive == "stop" or observation.get("success"):
                final_observation = observation
                break

            self.last_action = {"primitive": primitive, "args": dict(action.get("args") or {})}
            executed = self.execute_primitive(primitive, action.get("args") or {})
            post_observation = self.observe(stage=primitive)
            execution = {
                "type": "execution",
                "primitive": primitive,
                "steps_executed": executed,
                "step_index": self.step_index,
                "observation": post_observation,
            }
            self._record(execution)
            final_observation = post_observation

        if not final_observation:
            final_observation = self.observe(stage="final")
        return {
            "schema": "online_harness_result.v2",
            "task_id": self.task_id,
            "success": bool(final_observation.get("success")),
            "elapsed_online_steps": self.step_index,
            "planner": self.planner,
            "final_observation": final_observation,
            "num_decisions": sum(item.get("type") == "decision" for item in self.trace),
        }

    def _record(self, item: Dict[str, Any]) -> None:
        self.trace.append(item)
        if self.event_sink is not None:
            self.event_sink(item)

    def _execute_move(
        self,
        target: np.ndarray,
        *,
        gripper: float,
        max_delta_m: float,
        max_action: float,
        tolerance_m: float,
    ) -> int:
        executed = 0
        for _ in range(self.segment_steps):
            if self.robot._early_stop():
                break
            tcp = np.asarray(self.robot._tcp_pos(), dtype=np.float32)
            delta = np.asarray(target, dtype=np.float32) - tcp
            if float(np.linalg.norm(delta)) <= tolerance_m:
                break
            command = np.clip(delta / max_delta_m, -max_action, max_action)
            self._step(command, gripper=gripper)
            executed += 1
        return executed

    def _execute_hold(self, *, gripper: float, steps: int) -> int:
        executed = 0
        for _ in range(max(1, min(int(steps), 32))):
            if self.robot._early_stop():
                break
            self._step(np.zeros(3, dtype=np.float32), gripper=gripper)
            executed += 1
        return executed

    def _step(self, delta_xyz: np.ndarray, *, gripper: float) -> None:
        action = _make_robot_action(self.robot, delta_xyz, gripper=gripper)
        self.robot._step(action)
        self.step_index += 1


class OnlinePullCubeHarness(_OnlineHarnessBase):
    """Task-level online planar-contact agent for PullCube or PushCube."""

    task_id = "pull_cube"

    def __init__(
        self,
        robot: Any,
        *,
        planner: str = "llm",
        segment_steps: int = 8,
        config: OnlinePullConfig | None = None,
        dry_run: bool = False,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
        task_id: str = "pull_cube",
    ) -> None:
        if task_id not in {"pull_cube", "push_cube"}:
            raise ValueError(f"planar-contact harness does not support task {task_id!r}")
        self.task_id = task_id
        super().__init__(
            robot,
            planner=planner,
            segment_steps=segment_steps,
            dry_run=dry_run,
            event_sink=event_sink,
        )
        self.config = config or OnlinePullConfig()

    def observe(self, *, stage: str) -> Dict[str, Any]:
        return build_pull_cube_online_observation(
            cube_pos=self.robot._actor_pos("cube"),
            goal_pos=self.robot._region_pos("goal"),
            tcp_pos=self.robot._tcp_pos(),
            step_index=self.step_index,
            stage=stage,
            success=bool(
                self.robot._push_cube_success()
                if self.task_id == "push_cube"
                else self.robot._pull_cube_success()
            ),
            config=self.config,
            last_action=self.last_action,
            task_id=self.task_id,
        )

    def execute_primitive(self, primitive: str, args: Mapping[str, Any]) -> int:
        if primitive in {"move_to_pre_contact", "move_to_contact"}:
            self._update_contact_config(args)
            obs = self.observe(stage=primitive)
            key = "pre_contact" if primitive == "move_to_pre_contact" else "contact"
            return self._execute_move(
                np.asarray(obs["targets"][key], dtype=np.float32),
                gripper=self.config.gripper_close,
                max_delta_m=self.config.max_delta_m,
                max_action=self.config.max_action,
                tolerance_m=self.config.contact_tolerance_m,
            )
        if primitive == "drag_toward_goal":
            return self._execute_drag(args)
        if primitive == "hold":
            return self._execute_hold(
                gripper=self.config.gripper_close,
                steps=int(args.get("steps") or self.config.settle_steps),
            )
        return 0

    def _update_contact_config(self, args: Mapping[str, Any]) -> None:
        values = dict(self.config.__dict__)
        for name in ("contact_x_offset_m", "contact_z_offset_m", "approach_height_m"):
            if name in args:
                values[name] = float(args[name])
        self.config = OnlinePullConfig(**values)

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
            command = np.array(
                [goal_dir[0] * abs(drag_strength), goal_dir[1] * abs(drag_strength), down_bias],
                dtype=np.float32,
            )
            self._step(
                np.clip(command, -self.config.max_action, self.config.max_action),
                gripper=self.config.gripper_close,
            )
            executed += 1
        return executed


class OnlinePickCubeHarness(_OnlineHarnessBase):
    """Task-level online PickCube agent with grasp feedback after every segment."""

    task_id = "pick_cube"

    def __init__(
        self,
        robot: Any,
        *,
        planner: str = "llm",
        segment_steps: int = 8,
        config: OnlinePickConfig | None = None,
        dry_run: bool = False,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(
            robot,
            planner=planner,
            segment_steps=segment_steps,
            dry_run=dry_run,
            event_sink=event_sink,
        )
        self.config = config or OnlinePickConfig()
        self.phase = "start"
        self.gripper_command = self.config.gripper_open
        self.grasp_attempt = 0
        self.current_grasp_z_offset = self.config.grasp_z_offset_m
        self.initial_cube_z = float(self.robot._actor_pos("cube")[2])

    def observe(self, *, stage: str) -> Dict[str, Any]:
        return build_pick_cube_online_observation(
            cube_pos=self.robot._actor_pos("cube"),
            goal_pos=self.robot._region_pos("goal"),
            tcp_pos=self.robot._tcp_pos(),
            is_grasping=bool(self.robot._is_grasping("cube")),
            step_index=self.step_index,
            stage=stage,
            phase=self.phase,
            success=bool(self.robot._pick_cube_success()),
            gripper_command=self.gripper_command,
            grasp_attempt=self.grasp_attempt,
            initial_cube_z=self.initial_cube_z,
            current_grasp_z_offset=self.current_grasp_z_offset,
            config=self.config,
            last_action=self.last_action,
        )

    def execute_primitive(self, primitive: str, args: Mapping[str, Any]) -> int:
        if primitive == "open_gripper":
            self.gripper_command = self.config.gripper_open
            executed = self._execute_hold(
                gripper=self.gripper_command,
                steps=int(args.get("steps") or self.config.settle_steps),
            )
            self.phase = "open"
            return executed

        if primitive in {"move_to_pre_grasp", "move_to_grasp"}:
            if "grasp_z_offset" in args:
                self.current_grasp_z_offset = float(args["grasp_z_offset"])
            obs = self.observe(stage=primitive)
            key = "pre_grasp" if primitive == "move_to_pre_grasp" else "grasp"
            executed = self._execute_move(
                np.asarray(obs["targets"][key], dtype=np.float32),
                gripper=self.config.gripper_open,
                max_delta_m=self.config.max_delta_m,
                max_action=self.config.max_action,
                tolerance_m=self.config.grasp_tolerance_m,
            )
            self.gripper_command = self.config.gripper_open
            self.phase = "pre_grasp" if primitive == "move_to_pre_grasp" else "at_grasp"
            return executed

        if primitive == "close_gripper":
            self.gripper_command = float(args.get("close_command") or self.config.gripper_close)
            executed = self._execute_hold(
                gripper=self.gripper_command,
                steps=int(args.get("steps") or self.config.close_steps),
            )
            self.grasp_attempt += 1
            if self.robot._is_grasping("cube"):
                self.robot.held_object = "cube"
                self.phase = "grasped"
            else:
                self.phase = "close_failed"
            return executed

        if primitive == "lift_cube":
            lift_height = float(args.get("lift_height") or self.config.lift_height_m)
            target = np.asarray(self.robot._tcp_pos(), dtype=np.float32) + np.array(
                [0.0, 0.0, lift_height],
                dtype=np.float32,
            )
            executed = self._execute_move(
                target,
                gripper=self.config.gripper_close,
                max_delta_m=self.config.max_delta_m,
                max_action=self.config.max_action,
                tolerance_m=self.config.grasp_tolerance_m,
            )
            self.gripper_command = self.config.gripper_close
            self.phase = "lifted" if self.robot._is_grasping("cube") else "close_failed"
            return executed

        if primitive == "move_to_goal":
            obs = self.observe(stage=primitive)
            executed = self._execute_move(
                np.asarray(obs["targets"]["transport_tcp"], dtype=np.float32),
                gripper=self.config.gripper_close,
                max_delta_m=self.config.max_delta_m,
                max_action=self.config.max_action,
                tolerance_m=self.config.goal_tolerance_m,
            )
            self.gripper_command = self.config.gripper_close
            self.phase = "transport"
            return executed

        if primitive == "release_gripper":
            self.gripper_command = self.config.gripper_open
            executed = self._execute_hold(
                gripper=self.gripper_command,
                steps=int(args.get("steps") or self.config.settle_steps),
            )
            self.phase = "released"
            return executed

        if primitive == "hold":
            return self._execute_hold(
                gripper=self.gripper_command,
                steps=int(args.get("steps") or self.config.settle_steps),
            )
        return 0


def run_online_case(
    *,
    case_id: str,
    seed: int = 0,
    planner: str = "llm",
    segment_steps: int = 8,
    max_online_steps: int = 360,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 500,
    adapter_module: str = "",
    dry_run: bool = False,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    """Run a task-level online harness for a registered migration case."""

    case = get_full_migration_case(case_id)
    if case.task_id not in {"pull_cube", "pick_cube", "push_cube"}:
        raise ValueError(f"online harness does not support task {case.task_id!r}.")
    if dry_run:
        return _dry_run_online_payload(
            case,
            planner=planner,
            segment_steps=segment_steps,
            max_online_steps=max_online_steps,
        )

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
        harness: _OnlineHarnessBase
        if case.task_id == "pick_cube":
            harness = OnlinePickCubeHarness(
                robot,
                planner=planner,
                segment_steps=segment_steps,
                event_sink=event_sink,
            )
        else:
            harness = OnlinePullCubeHarness(
                robot,
                planner=planner,
                segment_steps=segment_steps,
                event_sink=event_sink,
                task_id=case.task_id,
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


def run_online_pull_cube_case(**kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible PullCube-only entrypoint."""

    case = get_full_migration_case(str(kwargs.get("case_id") or ""))
    if case.task_id != "pull_cube":
        raise ValueError("run_online_pull_cube_case requires a PullCube case.")
    return run_online_case(**kwargs)


def write_online_outputs(output_dir: Path, payload: Mapping[str, Any]) -> Dict[str, str]:
    """Write summary and the interleaved observation/action trace."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    trace_path = output_dir / "online_trace.jsonl"
    md_path = output_dir / "summary.md"
    trace = list(payload.get("trace") or [])
    summary = dict(payload)
    summary.pop("trace", None)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
    with trace_path.open("w", encoding="utf-8") as stream:
        for item in trace:
            stream.write(json.dumps(item, ensure_ascii=False, default=repr) + "\n")
    md_path.write_text(_online_markdown(summary, trace), encoding="utf-8")
    return {
        "summary": str(summary_path),
        "trace_jsonl": str(trace_path),
        "markdown": str(md_path),
    }


def _online_markdown(summary: Mapping[str, Any], trace: List[Mapping[str, Any]]) -> str:
    lines = [
        "# Online Task Harness Run",
        "",
        f"- case: `{summary.get('case_id')}`",
        f"- task: `{summary.get('task_id')}`",
        f"- planner: `{summary.get('planner')}`",
        f"- success: `{summary.get('success')}`",
        f"- elapsed_online_steps: `{summary.get('elapsed_online_steps')}`",
        f"- num_decisions: `{summary.get('num_decisions')}`",
        "",
        "## Observe - Act Trace",
        "",
        "| decision | step | stage | primitive | task_error | is_grasping |",
        "|---:|---:|---|---|---:|---|",
    ]
    decision_index = 0
    for item in trace:
        if item.get("type") != "decision":
            continue
        decision_index += 1
        obs = item.get("observation") or {}
        metrics = obs.get("metrics") or {}
        action = item.get("action") or {}
        error = metrics.get("cube_goal_xy", metrics.get("cube_goal_xyz"))
        lines.append(
            f"| {decision_index} | {item.get('step_index')} | {obs.get('stage')} | "
            f"{action.get('primitive')} | {error} | {metrics.get('is_grasping', '')} |"
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
    if case.task_id == "pick_cube":
        observation = build_pick_cube_online_observation(
            cube_pos=[0.0, 0.05, 0.02],
            goal_pos=[0.03, 0.0, 0.29],
            tcp_pos=[0.0, 0.0, 0.16],
            is_grasping=False,
            stage="dry_run",
        )
    else:
        observation = build_pull_cube_online_observation(
            cube_pos=[0.0, 0.05, 0.02],
            goal_pos=[0.2 if case.task_id == "push_cube" else -0.2, 0.05, 0.001],
            tcp_pos=[0.0, 0.0, 0.16],
            stage="dry_run",
            task_id=case.task_id,
        )
    action = plan_online_action(observation, planner="fallback", dry_run=True)
    return {
        "schema": "online_harness_result.v2",
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
        "message": (
            "dry run: the task agent would observe live state, choose one semantic "
            "primitive, execute a short segment, then observe again"
        ),
        "trace": [
            {
                "type": "decision",
                "step_index": 0,
                "observation": observation,
                "action": action,
            }
        ],
    }


def _semantic_tool_specs(task_id: str) -> List[Dict[str, Any]]:
    if task_id == "pick_cube":
        return [
            {"name": "open_gripper", "purpose": "open before a grasp retry", "args": {"steps": "1..32"}},
            {
                "name": "move_to_pre_grasp",
                "purpose": "move above the live cube pose",
                "args": {"grasp_z_offset": "-0.03..0.04 m"},
            },
            {
                "name": "move_to_grasp",
                "purpose": "descend toward the live grasp pose",
                "args": {"grasp_z_offset": "-0.03..0.04 m"},
            },
            {
                "name": "close_gripper",
                "purpose": "close while holding TCP position",
                "args": {"close_command": "-1.0..-0.1", "steps": "1..32"},
            },
            {"name": "lift_cube", "purpose": "lift only after measured grasp", "args": {"lift_height": "0.03..0.18 m"}},
            {"name": "move_to_goal", "purpose": "transport a grasped cube to the live goal", "args": {}},
            {"name": "release_gripper", "purpose": "release only when appropriate", "args": {"steps": "1..32"}},
            {"name": "hold", "purpose": "wait for settling and success update", "args": {"steps": "1..32"}},
            {"name": "stop", "purpose": "stop on success or exhausted measured budget", "args": {"reason": "text"}},
        ]
    return [
        {
            "name": "move_to_pre_contact",
            "purpose": "move above a far-side contact pose",
            "args": {
                "contact_x_offset_m": "0.02..0.16",
                "contact_z_offset_m": "0.003..0.06",
                "approach_height_m": "0.03..0.14",
            },
        },
        {
            "name": "move_to_contact",
            "purpose": "descend to the current contact pose",
            "args": {
                "contact_x_offset_m": "0.02..0.16",
                "contact_z_offset_m": "0.003..0.06",
            },
        },
        {
            "name": "drag_toward_goal",
            "purpose": "apply a short measured contact-motion segment toward the goal",
            "args": {"drag_strength": "0.1..0.9", "down_bias": "-0.3..0.0"},
        },
        {"name": "hold", "purpose": "wait for settling and success update", "args": {"steps": "1..32"}},
        {"name": "stop", "purpose": "stop on success or exhausted measured budget", "args": {"reason": "text"}},
    ]


def _sanitize_online_args(primitive: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    bounds = {
        "steps": (1.0, 32.0),
        "drag_strength": (0.1, 0.9),
        "down_bias": (-0.3, 0.0),
        "contact_x_offset_m": (0.02, 0.16),
        "contact_z_offset_m": (0.003, 0.06),
        "approach_height_m": (0.03, 0.14),
        "grasp_z_offset": (-0.03, 0.04),
        "close_command": (-1.0, -0.1),
        "lift_height": (0.03, 0.18),
    }
    for name, (low, high) in bounds.items():
        if name not in args:
            continue
        try:
            value = float(args[name])
        except (TypeError, ValueError):
            continue
        value = float(np.clip(value, low, high))
        cleaned[name] = int(value) if name == "steps" else value
    if primitive == "stop" and "reason" in args:
        cleaned["reason"] = str(args["reason"])[:160]
    return cleaned


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


def _vec3(value: Any) -> np.ndarray:
    return np.asarray(list(value), dtype=np.float32).reshape(-1)[:3]


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
