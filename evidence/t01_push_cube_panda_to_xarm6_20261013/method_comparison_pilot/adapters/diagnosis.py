"""Target adapter for xarm6_robotiq with pd_ee_delta_pos control."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class XArm6RobotiqAdapter(ManiSkillDynamicRobot):
    """Adapter that executes the PushCube task using delta end-effector commands."""

    def _move_to_pose(self, target_pos: np.ndarray, gripper: float,
                      max_steps: int = 200, threshold: float = 0.01) -> None:
        """Move TCP to target_pos using measured feedback until close enough."""
        for _ in range(max_steps):
            current_pos = self._tcp_pos()
            error = target_pos - current_pos
            if np.linalg.norm(error) < threshold:
                break
            # Clip step size to avoid large jumps
            step_size = 0.02
            dist = np.linalg.norm(error)
            if dist > step_size:
                delta = error / dist * step_size
            else:
                delta = error
            self._step(self._make_action(delta, gripper=gripper))

    def run_source_actions(self) -> bool:
        """Replicate the source program: close gripper, move behind cube, push to goal."""
        # Close gripper while staying in place
        self._move_towards(self._tcp_pos(), gripper=-1.0, steps=10)

        # Move to a pose behind the cube at low height (z ≈ 0.015)
        obj_pos = self._entity_pos("obj")
        reach_pos = obj_pos + np.array([-0.05, 0.0, -0.005])  # z = 0.015
        self._move_to_pose(reach_pos, gripper=-1.0, max_steps=100)

        # Push the cube towards the goal region at low height
        goal_region_pos = self._entity_pos("goal_region")
        goal_pos = np.array([goal_region_pos[0] - 0.12, goal_region_pos[1], 0.015])
        self._move_to_pose(goal_pos, gripper=-1.0, max_steps=200)

        # Check official success after the motion
        return self._official_success()


def build_robot(env, *, control_mode: str, robot_uid: str) -> XArm6RobotiqAdapter:
    """Factory function required by the harness."""
    return XArm6RobotiqAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
