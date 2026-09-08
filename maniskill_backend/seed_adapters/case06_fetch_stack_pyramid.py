"""Neutral interface seed, not LLM-generated and not a verified solution."""

from maniskill_backend.stack_pyramid import ManiSkillStackPyramidRobot


class GeneratedFetchStackPyramidRobot(ManiSkillStackPyramidRobot):
    """The LLM replaces this module while the source scaffold stays frozen."""


def build_robot(env, *, control_mode: str, robot_uid: str):
    return GeneratedFetchStackPyramidRobot(env, control_mode=control_mode, robot_uid=robot_uid)
