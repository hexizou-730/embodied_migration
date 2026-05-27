"""Embodiment skill adapters backed by ManiSkill actions.

The active project scope is PullCube-v1 migration from Panda to Fetch. LMP code
calls a compact skill API, while each target embodiment still needs an adapter
that turns ``robot.pull(cube, goal)`` into real ``env.step(action)`` execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SkillTarget:
    name: str
    kind: str


class ManiSkillSceneAdapter:
    def get_object(self, name: str) -> SkillTarget:
        return SkillTarget(name=name, kind="object")

    def get_region(self, name: str) -> SkillTarget:
        return SkillTarget(name=name, kind="region")


class ManiSkillDeltaEERobot:
    """Shared delta end-effector action utilities for PullCube adapters."""

    def __init__(
        self,
        env: Any,
        *,
        move_steps: int = 14,
        settle_steps: int = 10,
        max_delta_m: float = 0.07,
        gripper_open: float = 1.0,
        gripper_close: float = -1.0,
        control_mode: Optional[str] = None,
    ) -> None:
        self.env = env
        self.move_steps = move_steps
        self.settle_steps = settle_steps
        self.max_delta_m = max_delta_m
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.control_mode = control_mode
        self.last_info: Dict[str, Any] = {}
        self.terminated: bool = False
        self.truncated: bool = False
        self.events: List[Dict[str, Any]] = []
        self._validate_action_space()

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "PullCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("PullCube adapter requires a Box-like action_space.")
        if shape[-1] not in (4, 7):
            raise RuntimeError(
                "PullCube adapter expects action_space last dim in {4, 7} "
                f"(pd_ee_delta_pos or pd_ee_delta_pose), got shape {tuple(shape)!r}."
            )

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def _move_towards(self, target_pos: np.ndarray, *, gripper: float, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            tcp = self._tcp_pos()
            delta = np.asarray(target_pos, dtype=np.float32) - tcp
            if np.linalg.norm(delta) < 0.01:
                break
            clipped = np.clip(delta / self.max_delta_m, -1.0, 1.0)
            self._step(self._make_action(clipped, gripper=gripper))

    def _repeat_action(self, delta_xyz: np.ndarray, *, gripper: float, steps: int) -> None:
        action = self._make_action(delta_xyz, gripper=gripper)
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(action)

    def _early_stop(self) -> bool:
        return bool(self.terminated or self.truncated)

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("PullCube adapter requires a Box-like action_space.")
        action = np.zeros(shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[: min(3, flat.size)] = np.asarray(delta_xyz, dtype=np.float32)[: min(3, flat.size)]
        if flat.size >= 4:
            flat[-1] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def _step(self, action: Any) -> None:
        _, _, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info or {})
        self.terminated = self.terminated or _scalar_bool(terminated)
        self.truncated = self.truncated or _scalar_bool(truncated)

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _actor_pos(self, name: str) -> np.ndarray:
        actor = getattr(self._base_env(), name)
        return _to_numpy(actor.pose.p)

    def _tcp_pos(self) -> np.ndarray:
        agent = self._base_env().agent
        tcp_pose = getattr(agent, "tcp_pose", None)
        if tcp_pose is not None:
            return _to_numpy(tcp_pose.p)
        tcp = getattr(agent, "tcp", None)
        tcp_pose = getattr(tcp, "pose", None)
        if tcp_pose is not None:
            return _to_numpy(tcp_pose.p)
        raise RuntimeError("Could not read ManiSkill agent TCP pose.")

    def _info_bool(self, key: str) -> bool:
        if key not in self.last_info:
            return False
        value = self.last_info[key]
        try:
            return bool(_to_numpy(value).reshape(-1)[0])
        except Exception:
            return bool(value)

    def _log(self, api: str, args: Dict[str, Any], result: Any, ok: bool, message: str = "") -> bool:
        self.events.append(
            {
                "step": len(self.events) + 1,
                "api": api,
                "args": dict(args),
                "result": bool(result),
                "ok": bool(ok),
                "message": message,
                "failure_type": "" if ok else "execution failure",
            }
        )
        return bool(ok)

    def _fail(self, api: str, args: Dict[str, Any], message: str) -> bool:
        return self._log(api, args, False, False, message)


class ManiSkillPullCubeRobot(ManiSkillDeltaEERobot):
    """PullCube-v1 wrapper using real end-effector delta actions.

    ``pull(cube, goal)`` moves the TCP to the far side of the cube, makes
    contact, and drags the cube toward the goal region. Panda and Fetch both
    expose the same high-level skill, but their geometry and controller response
    can still require different adapter choices.
    """

    def __init__(
        self,
        env: Any,
        *,
        robot_uid: str,
        control_mode: Optional[str] = None,
        move_steps: int = 14,
        contact_steps: int = 8,
        drag_steps: int = 42,
        settle_steps: int = 10,
        max_delta_m: float = 0.07,
        contact_x_offset_m: float = 0.07,
        contact_z_offset_m: float = 0.02,
        gripper_open: float = 1.0,
        gripper_close: float = -1.0,
    ) -> None:
        self.robot_uid = robot_uid
        self.contact_steps = contact_steps
        self.drag_steps = drag_steps
        self.contact_x_offset_m = contact_x_offset_m
        self.contact_z_offset_m = contact_z_offset_m
        super().__init__(
            env,
            move_steps=move_steps,
            settle_steps=settle_steps,
            max_delta_m=max_delta_m,
            gripper_open=gripper_open,
            gripper_close=gripper_close,
            control_mode=control_mode,
        )

    def pull(
        self,
        obj: SkillTarget,
        target: SkillTarget,
        *,
        contact_x_offset: Optional[float] = None,
        contact_z_offset: Optional[float] = None,
        drag_extra: float = 0.02,
        stages: int = 4,
    ) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        x_offset = self.contact_x_offset_m if contact_x_offset is None else float(contact_x_offset)
        z_offset = self.contact_z_offset_m if contact_z_offset is None else float(contact_z_offset)
        x_offset = float(np.clip(x_offset, 0.03, 0.14))
        z_offset = float(np.clip(z_offset, 0.005, 0.06))
        stages = int(np.clip(stages, 1, 8))

        cube_pos = self._actor_pos("cube")
        goal_pos = self._region_pos(target.name)
        contact_start = cube_pos + np.array([x_offset, 0.0, z_offset], dtype=np.float32)
        pre_contact = contact_start + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        drag_end = np.array(
            [
                goal_pos[0] - float(drag_extra),
                cube_pos[1],
                contact_start[2],
            ],
            dtype=np.float32,
        )

        self._move_towards(pre_contact, gripper=self.gripper_close, steps=self.move_steps)
        self._move_towards(contact_start, gripper=self.gripper_close, steps=self.move_steps)
        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.contact_steps)

        for stage in range(1, stages + 1):
            alpha = stage / stages
            waypoint = contact_start * (1.0 - alpha) + drag_end * alpha
            self._move_towards(waypoint, gripper=self.gripper_close, steps=max(1, self.drag_steps // stages))
            if self._pull_cube_success():
                return self._log(
                    "pull",
                    {
                        "obj": obj.name,
                        "target": target.name,
                        "contact_x_offset": round(x_offset, 4),
                        "contact_z_offset": round(z_offset, 4),
                        "stages": stages,
                    },
                    True,
                    True,
                    "",
                )

        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)
        ok = self._pull_cube_success()
        return self._log(
            "pull",
            {
                "obj": obj.name,
                "target": target.name,
                "contact_x_offset": round(x_offset, 4),
                "contact_z_offset": round(z_offset, 4),
                "stages": stages,
            },
            ok,
            ok,
            "" if ok else f"cube was not pulled to target; {self._pull_diagnostics(goal_pos)}",
        )

    def grasp(self, obj: SkillTarget) -> bool:
        return self._fail("grasp", {"obj": obj.name}, "PullCube-v1 uses contact pulling; call robot.pull(cube, goal).")

    def place(self, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "place",
            {"obj": obj.name, "target": target.name},
            "PullCube-v1 uses contact pulling; call robot.pull(cube, goal).",
        )

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        return self._fail(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            "PullCube-v1 exposes robot.pull(cube, goal), not a separate alignment skill.",
        )

    def insert(self, obj: SkillTarget, target: SkillTarget, speed: float) -> bool:
        return self._fail(
            "insert",
            {"obj": obj.name, "target": target.name, "speed": float(speed)},
            "PullCube-v1 is not an insertion task.",
        )

    def hook_object(self, tool: SkillTarget, obj: SkillTarget) -> bool:
        return self._fail("hook_object", {"tool": tool.name, "obj": obj.name}, "PullCube-v1 does not use a tool.")

    def pull_with_tool(self, tool: SkillTarget, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "pull_with_tool",
            {"tool": tool.name, "obj": obj.name, "target": target.name},
            "PullCube-v1 does not use a tool; call robot.pull(cube, goal).",
        )

    def _region_pos(self, name: str) -> np.ndarray:
        if name == "goal":
            name = "goal_region"
        return self._actor_pos(name)

    def _pull_cube_success(self) -> bool:
        if self._info_bool("success"):
            return True
        try:
            result = self._base_env().evaluate()
            self.last_info = dict(result or {})
            return _dict_bool(result, "success")
        except Exception:
            return False

    def _pull_diagnostics(self, goal_pos: np.ndarray) -> str:
        try:
            cube_pos = self._actor_pos("cube")
            tcp_pos = self._tcp_pos()
            cube_goal_xy = float(np.linalg.norm(goal_pos[:2] - cube_pos[:2]))
            tcp_cube_xy = float(np.linalg.norm(tcp_pos[:2] - cube_pos[:2]))
            return (
                f"cube_goal_xy={cube_goal_xy:.4f}, "
                f"tcp_cube_xy={tcp_cube_xy:.4f}, "
                f"cube_pos={np.round(cube_pos, 4).tolist()}, "
                f"goal_pos={np.round(goal_pos, 4).tolist()}"
            )
        except Exception:
            return "pull diagnostics unavailable"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape((-1, array.shape[-1]))[0]
    return array.reshape(-1)


def _scalar_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        array = _to_numpy(value)
        return bool(array[0]) if array.size else False
    except Exception:
        return bool(value)


def _dict_bool(value: Dict[str, Any], key: str) -> bool:
    if key not in value:
        return False
    try:
        array = _to_numpy(value[key])
        return bool(array[0]) if array.size else False
    except Exception:
        return bool(value[key])
