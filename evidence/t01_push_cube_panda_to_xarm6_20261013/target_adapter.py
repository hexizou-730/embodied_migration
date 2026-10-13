"""Target adapter for xarm6_robotiq with pd_ee_delta_pos control."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class XArm6RobotiqAdapter(ManiSkillDynamicRobot):
    """Adapter that replicates the source push-cube behaviour using delta EE commands."""

    def close_gripper(self, steps: int = 20) -> None:
        """Close the gripper without moving the arm, checking early stop."""
        for _ in range(steps):
            if self._early_stop():
                break
            # The gripper action range is [-1, 1]; 1.0 closes the Robotiq gripper.
            action = self._make_action(np.zeros(3), gripper=1.0)
            self._step(action)

    def _move_to_pose_segments(
        self, target_pos: np.ndarray, max_segment: float = 0.04, max_steps: int = 200
    ) -> bool:
        """Move TCP toward target_pos in small replanned segments."""
        for _ in range(max_steps):
            if self._early_stop():
                break
            current_pos = self._tcp_pos()
            delta = target_pos - current_pos
            dist = float(np.linalg.norm(delta))
            if dist < 0.005:
                return True
            step = delta * min(1.0, max_segment / dist)
            action = self._make_action(step, gripper=1.0)
            self._step(action)
        return False

    def run_source_actions(self) -> bool:
        """Execute the push-cube task using the target robot."""
        # Close gripper at the start
        self.close_gripper(steps=20)

        # Read object and goal positions (world frame)
        obj_pos = self._entity_pos("obj")
        goal_pos = self._entity_pos("goal_region")

        # Reach pose: slightly behind the cube, at the same height as the cube centre
        reach_pos = obj_pos + np.array([-0.05, 0.0, 0.0])
        if not self._move_to_pose_segments(reach_pos, max_segment=0.04, max_steps=300):
            return False

        # Push pose: behind the goal region, moving through the cube
        push_pos = goal_pos + np.array([-0.12, 0.0, 0.0])
        if not self._move_to_pose_segments(push_pos, max_segment=0.02, max_steps=400):
            return False

        # Check official success after the push
        return self._official_success()


def build_robot(env, *, control_mode: str, robot_uid: str) -> XArm6RobotiqAdapter:
    """Factory function required by the dynamic adapter harness."""
    return XArm6RobotiqAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
