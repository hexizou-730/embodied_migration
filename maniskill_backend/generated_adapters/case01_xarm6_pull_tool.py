"""Case 01 target adapter surface for xarm6 PullCubeTool migration.

This file is intentionally a replaceable target module. The module-generation
runner asks the LLM to rewrite this complete module for the xarm6 target while
leaving the validated Panda source stack untouched.
"""

from __future__ import annotations

from typing import Any

from maniskill_backend.skill_adapter import ManiSkillPullCubeToolPlannerRobot


class GeneratedXArm6PullCubeToolRobot(ManiSkillPullCubeToolPlannerRobot):
    """Initial target adapter: inherit the current hand-written xarm6 wrapper."""


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6PullCubeToolRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6PullCubeToolRobot(
        env,
        robot_uid=robot_uid,
        control_mode=control_mode,
    )
