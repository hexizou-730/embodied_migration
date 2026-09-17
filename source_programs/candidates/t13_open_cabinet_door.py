"""Executable bootstrap source program for T13; pending multi-seed validation."""

from source_programs.candidate_runtime import BootstrapSourceRobot

ENV_ID = "OpenCabinetDoor-v1"
SOURCE_ROBOT = "fetch"
CONTROL_MODE = "pd_ee_delta_pos"
TASK_PROGRAM = "ret_val = robot.solve_task()"


class SourceRobot(BootstrapSourceRobot):
    POLICY = "open_door"


def build_robot(env, *, control_mode, robot_uid):
    return SourceRobot(env, control_mode=control_mode, robot_uid=robot_uid)

