"""B1: robot-generic PickCube adapter with action-layout compatibility only."""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPickCubeRobot


class SharedPickCubeRobot(ManiSkillPickCubeRobot):
    """Keep the source grasp/place skill while normalizing action layouts."""

    def _validate_action_space(self) -> None:
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or shape[-1] not in (4, 7, 9):
            raise RuntimeError(f"shared PickCube adapter expects action dim 4, 7, or 9; got {shape!r}")

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[0:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        flat[3 if flat.size == 9 else flat.size - 1] = float(gripper)
        return np.clip(action, space.low, space.high)


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> SharedPickCubeRobot:
    return SharedPickCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
