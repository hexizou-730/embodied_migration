"""Executable bootstrap source program for T09; pending multi-seed validation."""

from source_programs.candidate_runtime import BootstrapSourceRobot

ENV_ID = "PushT-v1"
SOURCE_ROBOT = "panda_stick"
CONTROL_MODE = "pd_ee_delta_pos"
TASK_PROGRAM = "ret_val = robot.solve_task()"


class SourceRobot(BootstrapSourceRobot):
    POLICY = "push_t"


def build_robot(env, *, control_mode, robot_uid):
    return SourceRobot(env, control_mode=control_mode, robot_uid=robot_uid)

