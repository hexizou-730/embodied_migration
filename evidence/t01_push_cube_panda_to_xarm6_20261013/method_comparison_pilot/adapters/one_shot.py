"""Target adapter for xarm6_robotiq with pd_ee_delta_pos control mode."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class TargetAdapter(ManiSkillDynamicRobot):
    """Replicates the source push-cube behaviour using delta end-effector commands."""

    def run_source_actions(self) -> bool:
        # Close the gripper and keep it closed throughout the task.
        self._close_gripper()

        # Move to a pose just behind the cube.
        obj_pos = self._entity_pos("obj")
        reach_target = obj_pos + np.array([-0.05, 0.0, 0.0])
        if not self._move_to(reach_target):
            return False

        # Push the cube towards the goal region in small segments.
        goal_pos = self._entity_pos("goal_region")
        push_target = goal_pos + np.array([-0.12, 0.0, 0.0])
        return self._push_in_segments(push_target)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _close_gripper(self, steps: int = 10) -> None:
        """Send a close-gripper command for a few steps."""
        for _ in range(steps):
            self._step(self._make_action(np.zeros(3), gripper=1.0))

    def _move_to(self, target: np.ndarray, max_steps: int = 200) -> bool:
        """Move TCP to a target world position using _move_towards."""
        for _ in range(max_steps):
            self._move_towards(target, gripper=1.0, steps=1)
            if np.linalg.norm(self._tcp_pos() - target) < 0.005:
                return True
        return False

    def _push_in_segments(self, target: np.ndarray, max_segment_m: float = 0.04) -> bool:
        """Follow a long contact motion through short, replanned segments."""
        for _ in range(200):  # safety bound
            current = self._tcp_pos()
            delta = target - current
            distance = float(np.linalg.norm(delta))
            if distance <= 0.005:
                return True
            step = delta * min(1.0, max_segment_m / distance)
            waypoint = current + step
            self._move_towards(waypoint, gripper=1.0, steps=1)
        return False


def build_robot(env, *, control_mode: str, robot_uid: str) -> TargetAdapter:
    """Factory function required by the dynamic adapter harness."""
    return TargetAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
