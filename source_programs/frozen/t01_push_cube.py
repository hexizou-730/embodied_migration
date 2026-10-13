"""ManiSkill 3.0.1 reference source program for T01_PUSH_CUBE."""

ENV_ID = 'PushCube-v1'
SOURCE_ROBOT = 'panda'
CONTROL_MODE = 'pd_joint_pos'

import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from source_programs.vendor.mani_skill_motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def _move_line_in_segments(planner, target_pose, *, max_segment_m=0.04):
    """Follow a long contact motion through short, replanned screw segments."""

    last_result = -1
    for _ in range(20):
        current_pose = planner.base_env.agent.tcp.pose.sp
        delta = np.asarray(target_pose.p) - np.asarray(current_pose.p)
        distance = float(np.linalg.norm(delta))
        if distance <= 0.005:
            return last_result
        step = delta * min(1.0, max_segment_m / distance)
        waypoint = sapien.Pose(
            p=np.asarray(current_pose.p) + step,
            q=target_pose.q,
        )
        last_result = planner.move_to_pose_with_screw(waypoint)
        if isinstance(last_result, (int, np.integer)) and last_result == -1:
            return -1
    return -1


def run(env: PushCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env = env.unwrapped
    planner.close_gripper()
    reach_pose = sapien.Pose(p=env.obj.pose.sp.p + np.array([-0.05, 0, 0]), q=env.agent.tcp.pose.sp.q)
    reach_result = planner.move_to_pose_with_screw(reach_pose)
    if isinstance(reach_result, (int, np.integer)) and reach_result == -1:
        planner.close()
        return -1

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(p=env.goal_region.pose.sp.p + np.array([-0.12, 0, 0]),q=env.agent.tcp.pose.sp.q)
    res = _move_line_in_segments(planner, goal_pose)

    planner.close()
    return res
