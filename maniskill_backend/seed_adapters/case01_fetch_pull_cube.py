"""Neutral Fetch seed adapter for from-zero PullCube migration.

This deliberately starts close to the Panda/source layout and does not solve
Fetch's 9D action mapping. The agent restores this file before a from-zero run
so the LLM must repair the target adapter through simulation feedback.
"""

from __future__ import annotations

from typing import Any

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedFetchPullCubeRobot(ManiSkillPullCubeRobot):
    """Source-like seed that exposes Fetch action-space mismatch."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
        )


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedFetchPullCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedFetchPullCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
