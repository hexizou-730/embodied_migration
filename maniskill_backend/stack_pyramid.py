"""Unvalidated StackPyramid source scaffold and measured stage observations.

Only env.step actions move the scene. Geometry gates below are adapter
preconditions; official env.evaluate() remains the task success authority.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .skill_adapter import ManiSkillDeltaEERobot, SkillTarget, _scalar_bool, _to_numpy


def stack_observation(robot: Any) -> dict[str, Any]:
    """Read all three objects, including after a generated adapter fails."""
    env = getattr(robot.env, "unwrapped", robot.env)
    cubes = {}
    for name in ("cubeA", "cubeB", "cubeC"):
        actor = getattr(env, name)
        cubes[name] = {
            "position": _to_numpy(actor.pose.p).tolist(),
            "quaternion": _to_numpy(actor.pose.q).tolist(),
            "is_grasping": _scalar_bool(env.agent.is_grasping(actor)),
            "is_static": _scalar_bool(actor.is_static(lin_thresh=0.01, ang_thresh=0.5)),
        }
    tcp = _to_numpy(env.agent.tcp.pose.p)
    a, b, c = (np.asarray(cubes[name]["position"]) for name in ("cubeA", "cubeB", "cubeC"))
    half = _to_numpy(env.cube_half_size)
    base_distance = float(np.linalg.norm(a[:2] - b[:2]))
    # An operational base gate, stricter than a loose final XY success flag.
    base_ready = bool(
        2 * min(half[:2]) - 0.005 <= base_distance <= np.linalg.norm(2 * half[:2]) + 0.005
        and abs(a[2] - b[2]) <= 0.008
        and abs(a[2] - half[2]) <= 0.01 and abs(b[2] - half[2]) <= 0.01
        and all(not cubes[n]["is_grasping"] and cubes[n]["is_static"] for n in ("cubeA", "cubeB"))
    )
    goal = (a + b) / 2
    goal[2] = max(a[2], b[2]) + 2 * half[2]
    target = getattr(robot, "stage_target", None)
    return {
        "stage": getattr(robot, "stage", "unknown"),
        "stage_object": getattr(robot, "stage_object", None),
        "steps_used": getattr(robot, "steps_used", None),
        "episode_ended": bool(getattr(robot, "terminated", False) or getattr(robot, "truncated", False)),
        "cubes": cubes,
        "tcp_pos": tcp.tolist(),
        "stage_target_pos": None if target is None else np.asarray(target).tolist(),
        "tcp_stage_error_norm": None if target is None else float(np.linalg.norm(np.asarray(target) - tcp)),
        "base_distance_xy": base_distance,
        "base_ready": base_ready,
        "top_target_pos": goal.tolist(),
        "top_error_xyz": (goal - c).tolist(),
        "official_success": _scalar_bool(env.evaluate()["success"]),
    }


class ManiSkillStackPyramidRobot(ManiSkillDeltaEERobot):
    """Source policy plus neutral target interface, not a verified oracle."""

    def __init__(self, env: Any, *, robot_uid: str, control_mode: str,
                 move_steps: int = 40, grasp_steps: int = 16,
                 settle_steps: int = 20, max_delta_m: float = 0.05,
                 approach_height: float = 0.12, tolerance: float = 0.008) -> None:
        self.robot_uid = robot_uid
        self.grasp_steps = grasp_steps
        self.approach_height = approach_height
        self.tolerance = tolerance
        self.stage = "initial"
        self.stage_object = None
        self.stage_target = None
        self.steps_used = 0
        self.stage_trace: list[dict[str, Any]] = []
        super().__init__(env, control_mode=control_mode, move_steps=move_steps,
                         settle_steps=settle_steps, max_delta_m=max_delta_m)

    def _validate_action_space(self) -> None:
        expected = {"panda": 4, "fetch": 9}.get(self.robot_uid)
        shape = getattr(self.env.action_space, "shape", ())
        if self.control_mode != "pd_ee_delta_pos" or expected is None:
            raise ValueError("StackPyramid scaffold supports Panda/Fetch pd_ee_delta_pos only.")
        if tuple(shape) not in {(expected,), (1, expected)}:
            raise ValueError(f"Expected one-env action space with {expected} channels, got {shape}.")

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        action = np.zeros(space.shape, dtype=space.dtype)
        action.reshape(-1)[:3] = np.asarray(delta_xyz).reshape(3)
        action.reshape(-1)[3] = gripper
        # Fetch body/base channels stay neutral; target LLM may adapt them.
        return np.clip(action, space.low, space.high)

    def _step(self, action: Any) -> None:
        if self._early_stop():
            return
        super()._step(action)
        self.steps_used += 1

    def _is_grasping(self, name: str) -> bool:
        return _scalar_bool(self._base_env().agent.is_grasping(self._actor(name)))

    def _stack_success(self) -> bool:
        return _scalar_bool(self._base_env().evaluate()["success"])

    def _begin_stage(self, stage: str, obj: str | None = None,
                     target: np.ndarray | None = None) -> None:
        self.stage, self.stage_object = stage, obj
        self.stage_target = None if target is None else np.asarray(target).copy()

    def _record_stage(self, ok: bool) -> None:
        self.stage_trace.append({**stack_observation(self), "stage_ok": bool(ok)})

    def _abort(self, api: str, reason: str) -> bool:
        state = stack_observation(self)
        return self._fail(api, {"stage": self.stage, "object": self.stage_object},
                          f"{reason}; stage={self.stage}, steps_used={self.steps_used}, "
                          f"tcp_stage_error_norm={state['tcp_stage_error_norm']}, "
                          f"base_ready={state['base_ready']}")

    def _move_stage(self, stage: str, target: np.ndarray, *, gripper: float,
                    obj: str, require_grasp: bool = False) -> bool:
        self._begin_stage(stage, obj, target)
        for _ in range(self.move_steps):
            if self._early_stop():
                break
            if require_grasp and not self._is_grasping(obj):
                self._record_stage(False)
                return False
            delta = np.asarray(target) - self._tcp_pos()
            if np.linalg.norm(delta) <= self.tolerance:
                break
            self._step(self._make_action(np.clip(delta / self.max_delta_m, -0.8, 0.8), gripper=gripper))
        ok = bool(np.linalg.norm(np.asarray(target) - self._tcp_pos()) <= self.tolerance
                  and (not require_grasp or self._is_grasping(obj)))
        self._record_stage(ok)
        return ok

    def _hold_stage(self, stage: str, *, obj: str, gripper: float, steps: int) -> None:
        target = self._tcp_pos().copy()
        self._begin_stage(stage, obj, target)
        for _ in range(steps):
            if self._early_stop():
                break
            command = np.clip((target - self._tcp_pos()) / self.max_delta_m, -0.3, 0.3)
            self._step(self._make_action(command, gripper=gripper))
        self._record_stage(not self._early_stop())

    def _grasp_object(self, name: str, prefix: str) -> bool:
        self._hold_stage(prefix + ".open", obj=name, gripper=self.gripper_open, steps=self.grasp_steps)
        center = self._actor_pos(name)
        above = center + np.array([0, 0, self.approach_height])
        if not self._move_stage(prefix + ".approach", above, gripper=self.gripper_open, obj=name):
            return False
        # Re-read the object after approach; do not close at a stale location.
        center = self._actor_pos(name)
        if not self._move_stage(prefix + ".descent", center, gripper=self.gripper_open, obj=name):
            return False
        self._hold_stage(prefix + ".close", obj=name, gripper=self.gripper_close, steps=self.grasp_steps)
        if not self._is_grasping(name):
            return False
        return self._move_stage(prefix + ".lift", self._tcp_pos() + np.array([0, 0, self.approach_height]),
                                gripper=self.gripper_close, obj=name, require_grasp=True)

    def _place_object(self, name: str, center: np.ndarray, prefix: str) -> bool:
        tcp_target = self._tcp_pos() + np.asarray(center) - self._actor_pos(name)
        above = tcp_target + np.array([0, 0, self.approach_height])
        for stage, target in (("transport", above), ("lower", tcp_target)):
            if not self._move_stage(prefix + "." + stage, target, gripper=self.gripper_close,
                                    obj=name, require_grasp=True):
                return False
        self._hold_stage(prefix + ".release", obj=name, gripper=self.gripper_open, steps=self.grasp_steps)
        if self._is_grasping(name):
            return False
        if not self._move_stage(prefix + ".retreat", above, gripper=self.gripper_open, obj=name):
            return False
        self._hold_stage(prefix + ".settle", obj=name, gripper=self.gripper_open, steps=self.settle_steps)
        return True

    def prepare_base(self, red: SkillTarget, green: SkillTarget) -> bool:
        if (red.name, green.name) != ("cubeA", "cubeB"):
            return self._abort("prepare_base", "invalid base objects")
        if stack_observation(self)["base_ready"]:
            return self._log("prepare_base", {}, True, True, "base already aligned")
        if self._early_stop():
            return self._abort("prepare_base", "episode ended")
        # This source scaffold uses one local placement, not a success trajectory.
        a, b = self._actor_pos(red.name), self._actor_pos(green.name)
        direction = a[:2] - b[:2]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-6 else np.array([1., 0.])
        half = _to_numpy(self._base_env().cube_half_size)
        center = b.copy()
        center[:2] += direction * (2 * np.linalg.norm(half[:2]) + 0.001)
        if not self._grasp_object(red.name, "base"):
            return self._abort("prepare_base", "base grasp/motion failed")
        if not self._place_object(red.name, center, "base"):
            return self._abort("prepare_base", "base placement failed")
        self._begin_stage("base.verify", red.name)
        ready = stack_observation(self)["base_ready"]
        self._record_stage(ready)
        if not ready:
            return self._abort("prepare_base", "base alignment/stability failed")
        return self._log("prepare_base", {}, True, True)

    def stack_on(self, blue: SkillTarget, red: SkillTarget, green: SkillTarget) -> bool:
        if (blue.name, red.name, green.name) != ("cubeC", "cubeA", "cubeB"):
            return self._abort("stack_on", "invalid stack objects")
        state = stack_observation(self)
        if not state["base_ready"]:
            self._begin_stage("base.verify")
            return self._abort("stack_on", "base changed before stacking")
        if self._early_stop():
            return self._abort("stack_on", "episode ended")
        if not self._grasp_object(blue.name, "top"):
            return self._abort("stack_on", "top grasp/motion failed")
        state = stack_observation(self)
        if not state["base_ready"]:
            return self._abort("stack_on", "base disturbed during top grasp")
        if not self._place_object(blue.name, np.asarray(state["top_target_pos"]), "top"):
            # Success may be reached exactly when the episode terminates.
            if not self._stack_success():
                return self._abort("stack_on", "top placement failed")
        self._begin_stage("top.verify", blue.name)
        ok = self._stack_success()
        self._record_stage(ok)
        if not ok:
            return self._abort("stack_on", "official stack success not reached")
        return self._log("stack_on", {}, True, True)
