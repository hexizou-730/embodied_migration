"""Neutral Fetch seed adapter for from-zero PushCube migration.

The seed deliberately keeps the source PushCube behavior and does not encode
Fetch base motion or a successful target-side contact trajectory.
"""

from __future__ import annotations

from typing import Any

from maniskill_backend.skill_adapter import ManiSkillPushCubeRobot


class GeneratedFetchPushCubeRobot(ManiSkillPushCubeRobot):
    """Source-like seed that exposes Fetch action-layout/reachability failures."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, robot_uid=robot_uid, control_mode=control_mode)


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedFetchPushCubeRobot:
    return GeneratedFetchPushCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
