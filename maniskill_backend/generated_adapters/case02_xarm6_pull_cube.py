"""Initial xarm6_robotiq target adapter for PullCube migration.

This module is intentionally conservative. xarm6_robotiq is closer to Panda
than Fetch because it is also a fixed-base single-arm robot, so the first
migration attempt keeps the same high-level skill but retunes contact and
motion defaults for a less redundant arm.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedXArm6PullCubeRobot(ManiSkillPullCubeRobot):
    """Fixed-base xarm6 adapter for PullCube-v1."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=24,
            contact_steps=12,
            drag_steps=72,
            settle_steps=14,
            max_delta_m=0.045,
            contact_x_offset_m=0.055,
            contact_z_offset_m=0.012,
            gripper_open=1.0,
            gripper_close=-1.0,
        )

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "xarm6 PullCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("xarm6 PullCube adapter requires a Box-like action_space.")
        if shape[-1] < 4 or shape[-1] > 16:
            raise RuntimeError(
                "xarm6 PullCube adapter expects a compact fixed-arm action space "
                f"with at least xyz+gripper dims, got shape {tuple(shape)!r}."
            )

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("xarm6 PullCube adapter requires a Box-like action_space.")
        action = np.zeros(shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        if flat.size > 3:
            flat[3:] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def _move_towards(self, target_pos: np.ndarray, *, gripper: float, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            tcp = self._tcp_pos()
            delta = np.asarray(target_pos, dtype=np.float32) - tcp
            if np.linalg.norm(delta) < 0.008:
                break
            clipped = np.clip(delta / self.max_delta_m, -0.8, 0.8)
            self._step(self._make_action(clipped, gripper=gripper))

    def pull(self, obj, target, *, contact_x_offset=None, contact_z_offset=None, drag_extra=0.025, stages=5) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        x_offset = self.contact_x_offset_m if contact_x_offset is None else float(contact_x_offset)
        z_offset = self.contact_z_offset_m if contact_z_offset is None else float(contact_z_offset)
        candidates = (
            (x_offset, z_offset),
            (0.045, 0.010),
            (0.065, 0.014),
        )
        for x_off, z_off in candidates:
            if self._try_pull_once(
                target,
                x_offset=float(np.clip(x_off, 0.03, 0.09)),
                z_offset=float(np.clip(z_off, 0.006, 0.025)),
                drag_extra=drag_extra,
                stages=stages,
            ):
                return self._log(
                    "pull",
                    {
                        "obj": obj.name,
                        "target": target.name,
                        "contact_x_offset": round(float(x_off), 4),
                        "contact_z_offset": round(float(z_off), 4),
                        "stages": stages,
                    },
                    True,
                    True,
                    "",
                )
            self._move_towards(
                self._tcp_pos() + np.array([0.0, 0.0, 0.06], dtype=np.float32),
                gripper=self.gripper_close,
                steps=12,
            )

        goal_pos = self._region_pos(target.name)
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "attempts": len(candidates)},
            False,
            False,
            f"cube was not pulled to target; {self._pull_diagnostics(goal_pos)}",
        )

    def _try_pull_once(self, target, *, x_offset: float, z_offset: float, drag_extra: float, stages: int) -> bool:
        cube_pos = self._actor_pos("cube")
        goal_pos = self._region_pos(target.name)
        contact_start = cube_pos + np.array([x_offset, 0.0, z_offset], dtype=np.float32)
        pre_contact = contact_start + np.array([0.0, 0.0, 0.075], dtype=np.float32)
        drag_end = np.array(
            [
                goal_pos[0] - float(drag_extra),
                cube_pos[1],
                max(0.006, contact_start[2] - 0.004),
            ],
            dtype=np.float32,
        )

        self._move_towards(pre_contact, gripper=self.gripper_close, steps=self.move_steps)
        self._move_towards(contact_start, gripper=self.gripper_close, steps=self.move_steps)
        self._repeat_action(np.array([-0.08, 0.0, -0.04], dtype=np.float32), gripper=self.gripper_close, steps=self.contact_steps)

        stages = int(np.clip(stages, 1, 8))
        for stage in range(1, stages + 1):
            alpha = stage / stages
            waypoint = contact_start * (1.0 - alpha) + drag_end * alpha
            self._move_towards(waypoint, gripper=self.gripper_close, steps=max(1, self.drag_steps // stages))
            self._repeat_action(np.array([-0.12, 0.0, -0.025], dtype=np.float32), gripper=self.gripper_close, steps=2)
            if self._pull_cube_success():
                return True

        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)
        return self._pull_cube_success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6PullCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6PullCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
