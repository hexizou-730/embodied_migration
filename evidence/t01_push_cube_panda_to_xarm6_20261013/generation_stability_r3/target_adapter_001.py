"""Target adapter for PushCube-v1 with xarm6_robotiq pd_ee_delta_pos control.

This adapter closes the gripper, moves the end-effector behind the cube,
then pushes it toward the goal while monitoring success.
"""

import numpy as np

from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class PushCubeAdapter(ManiSkillDynamicRobot):
    """Replicates the source push-cube task using EE delta position commands."""

    def run_source_actions(self) -> bool:
        """Execute the push cube task: close gripper, move behind cube, push to goal.

        The push phase loops, stopping early when the official success condition is met.
        Returns the final success flag.
        """
        # ---- close gripper ----
        for _ in range(2):
            self._step(self._make_action(np.zeros(3), gripper=1.0))

        # ---- move to a point right behind the cube ----
        obj_pos = self._entity_pos("obj")          # measured state
        reach_target = obj_pos + np.array([-0.05, 0.0, 0.0])
        self._move_towards(reach_target, gripper=1.0, steps=30)

        # ---- push the cube toward the goal region ---
        goal_pos = self._entity_pos("goal_region")  # measured state
        push_target = goal_pos + np.array([-0.12, 0.0, 0.0])

        max_push_steps = 60
        for _ in range(max_push_steps):
            self._move_towards(push_target, gripper=1.0, steps=1)
            if self._official_success():  # measured runtime branch
                break

        return self._official_success()


def build_robot(env, *, control_mode, robot_uid):
    return PushCubeAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
