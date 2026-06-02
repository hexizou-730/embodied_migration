"""Neutral xarm6_robotiq seed adapter for direct PickCube module generation.

This seed uses one conservative top-down grasp attempt. It intentionally avoids
encoding a hand-tuned target trajectory. The generation runner may overwrite
this module with a complete LLM-generated xarm6 target adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPickCubeRobot


class GeneratedXArm6PickCubeRobot(ManiSkillPickCubeRobot):
    """Non-oracle xarm6 seed adapter for grasp migration."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=28,
            grasp_steps=14,
            settle_steps=18,
            max_delta_m=0.045,
            approach_height_m=0.10,
            lift_height_m=0.10,
            grasp_z_offset_m=0.0,
            gripper_open=1.0,
            gripper_close=-1.0,
        )

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "xarm6 PickCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or shape[-1] != 4:
            raise RuntimeError(f"xarm6 PickCube adapter expects observed 4D action space, got {shape!r}.")

    def _move_towards(self, target_pos: np.ndarray, *, gripper: float, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            delta = np.asarray(target_pos, dtype=np.float32) - self._tcp_pos()
            if np.linalg.norm(delta) < 0.006:
                break
            command = np.clip(delta / self.max_delta_m, -0.8, 0.8)
            self._step(self._make_action(command, gripper=gripper))


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6PickCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6PickCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
