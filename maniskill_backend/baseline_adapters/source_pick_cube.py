"""B0: use the unmodified source PickCube skill adapter on the target robot."""

from __future__ import annotations

from typing import Any

from maniskill_backend.skill_adapter import ManiSkillPickCubeRobot


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> ManiSkillPickCubeRobot:
    return ManiSkillPickCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
