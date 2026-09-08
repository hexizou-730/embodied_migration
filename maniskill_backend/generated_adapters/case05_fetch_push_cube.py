"""Neutral Fetch seed adapter for from-zero PushCube migration.

The seed implements Fetch's observed 9D action interface so it can execute as
a baseline, but it does not encode base motion or a successful contact path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPushCubeRobot


class GeneratedFetchPushCubeRobot(ManiSkillPushCubeRobot):
    """Source-like seed with interface compatibility but no Fetch strategy."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, robot_uid=robot_uid, control_mode=control_mode)

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "Fetch PushCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        shape = getattr(getattr(self.env, "action_space", None), "shape", None)
        if not shape or shape[-1] != 9:
            raise RuntimeError(
                "Fetch PushCube adapter expects the observed 9D action layout, "
                f"got shape {tuple(shape) if shape else None!r}."
            )

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[0:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        flat[3] = float(gripper)
        return np.clip(action, space.low, space.high)


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedFetchPushCubeRobot:
    return GeneratedFetchPushCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
