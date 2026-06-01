"""Neutral xarm6_robotiq seed adapter for direct LLM module generation.

This module uses a conservative waypoint contact pull. It intentionally avoids
encoding the human-written oracle trajectory. The generation runner overwrites
this file with the LLM's complete target adapter module and evaluates it in the
real ManiSkill environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedXArm6PullCubeRobot(ManiSkillPullCubeRobot):
    """Non-oracle xarm6 seed adapter for failure-driven module generation."""

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
            raise RuntimeError("xarm6 seed adapter requires a Box-like action_space.")
        if shape[-1] != 4:
            raise RuntimeError(f"xarm6 seed adapter expects observed 4D action space, got shape {tuple(shape)!r}.")

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("xarm6 seed adapter requires a Box-like action_space.")
        action = np.zeros(shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        flat[3] = float(gripper)
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
            command = np.clip(delta / self.max_delta_m, -0.8, 0.8)
            self._step(self._make_action(command, gripper=gripper))

    def pull(self, obj, target, *, contact_x_offset=None, contact_z_offset=None, drag_extra=0.025, stages=5) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        x_offset = self.contact_x_offset_m if contact_x_offset is None else float(contact_x_offset)
        z_offset = self.contact_z_offset_m if contact_z_offset is None else float(contact_z_offset)
        cube_pos = self._actor_pos("cube")
        goal_pos = self._region_pos(target.name)
        contact = cube_pos + np.array([x_offset, 0.0, z_offset], dtype=np.float32)
        pre_contact = contact + np.array([0.0, 0.0, 0.075], dtype=np.float32)
        drag_end = np.array([goal_pos[0] - float(drag_extra), cube_pos[1], contact[2]], dtype=np.float32)

        self._move_towards(pre_contact, gripper=self.gripper_close, steps=self.move_steps)
        self._move_towards(contact, gripper=self.gripper_close, steps=self.move_steps)
        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.contact_steps)

        stages = int(np.clip(stages, 1, 8))
        for stage in range(1, stages + 1):
            waypoint = contact * (1.0 - stage / stages) + drag_end * (stage / stages)
            self._move_towards(waypoint, gripper=self.gripper_close, steps=max(1, self.drag_steps // stages))
            if self._pull_cube_success():
                return self._log("pull", {"obj": obj.name, "target": target.name, "seed": True}, True, True, "")

        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)
        ok = self._pull_cube_success()
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "seed": True},
            ok,
            ok,
            "" if ok else f"cube was not pulled to target; {self._pull_diagnostics(goal_pos)}",
        )


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6PullCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6PullCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
