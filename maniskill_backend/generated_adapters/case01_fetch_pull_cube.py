"""Case 01 target adapter surface for Fetch PullCube migration.

The module-generation runner may rewrite this file while keeping the source
Panda program and shared runner contracts stable.
"""

from __future__ import annotations

from typing import Any

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedFetchPullCubeRobot(ManiSkillPullCubeRobot):
    """Initial target adapter: reuse the shared PullCube contact primitive."""


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedFetchPullCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedFetchPullCubeRobot(
        env,
        robot_uid=robot_uid,
        control_mode=control_mode,
    )
