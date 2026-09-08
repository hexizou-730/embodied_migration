"""Neutral Fetch seed adapter for direct PickCube module generation.

The seed only maps the observed 9D controller and keeps the source top-down
grasp. It does not encode a base approach or a successful target trajectory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPickCubeRobot


class GeneratedFetchPickCubeRobot(ManiSkillPickCubeRobot):
    """Non-oracle Fetch seed used to expose mobile-manipulator failures."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=28,
            grasp_steps=14,
            settle_steps=18,
            max_delta_m=0.045,
            approach_height_m=0.12,
            lift_height_m=0.10,
            grasp_z_offset_m=0.0,
        )

    def _validate_action_space(self) -> None:
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or shape[-1] != 9:
            raise RuntimeError(f"Fetch PickCube seed expects observed 9D action space, got {shape!r}.")

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[0:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        flat[3] = float(gripper)
        return np.clip(action, space.low, space.high)


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedFetchPickCubeRobot:
    return GeneratedFetchPickCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
