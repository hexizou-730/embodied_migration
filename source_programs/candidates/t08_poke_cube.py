"""Executable bootstrap source program for T08; pending multi-seed validation."""

from source_programs.candidate_runtime import BootstrapSourceRobot

ENV_ID = "PokeCube-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"
TASK_PROGRAM = "ret_val = robot.solve_task()"


class SourceRobot(BootstrapSourceRobot):
    POLICY = "poke_cube"


def build_robot(env, *, control_mode, robot_uid):
    return SourceRobot(env, control_mode=control_mode, robot_uid=robot_uid)

