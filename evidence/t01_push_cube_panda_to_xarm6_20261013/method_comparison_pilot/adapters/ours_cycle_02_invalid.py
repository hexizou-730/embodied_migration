"""Target adapter for xarm6_robotiq with pd_ee_delta_pos control mode."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class XArm6RobotiqAdapter(ManiSkillDynamicRobot):
    """Adapter that mimics the source push-cube behaviour using delta EE commands."""

    def _move_to(self, target: np.ndarray, gripper: float, max_steps: int = 200, tol: float = 0.01) -> None:
        """Move TCP towards a world target position using measured feedback."""
        for _ in range(max_steps):
            current = self._tcp_pos()
            delta = target - current
            dist = float(np.linalg.norm(delta))
            if dist < tol:
                break
            # Limit step size to 0.02 m per action
            step = delta * min(1.0, 0.02 / dist)
            action = self._make_action(step, gripper=gripper)
            self._step(action)

    def run_source_actions(self) -> bool:
        """Execute the push-cube task: close gripper, reach behind cube, push to goal."""
        # Close gripper
        close_action = self._make_action(np.zeros(3), gripper=-1.0)
        for _ in range(10):
            self._step(close_action)

        # Get current cube and goal positions
        obj_pos = self._entity_pos("obj")
        goal_pos = self._entity_pos("goal_region")

        # Reach pose: behind the cube (offset -0.05 in x)
        reach_target = obj_pos + np.array([-0.05, 0.0, 0.0])
        self._move_to(reach_target, gripper=-1.0, max_steps=100)

        # Push pose: behind the goal region (offset -0.12 in x)
        push_target = goal_pos + np.array([-0.12, 0.0, 0.0])
        self._move_to(push_target, gripper=-1.0, max_steps=200)

        return True


def build_robot(env, *, control_mode: str, robot_uid: str) -> XArm6RobotiqAdapter:
    """Factory function required by the dynamic adapter harness."""
    return XArm6RobotiqAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
