"""Target adapter for xarm6_robotiq with pd_ee_delta_pos control mode."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class Xarm6RobotiqPushCubeAdapter(ManiSkillDynamicRobot):
    """Adapter that pushes the cube from behind into the goal region."""

    def run_source_actions(self) -> bool:
        """Execute a push strategy using measured state and delta EE commands."""
        # Close gripper to form a solid pusher
        for _ in range(10):
            self._step(self._make_action(np.zeros(3), gripper=1.0))
            if self._early_stop():
                return False

        cube_pos = self._entity_pos("obj")
        goal_pos = self._entity_pos("goal_region")
        tcp_z = self._tcp_pos()[2]

        # Position behind the cube
        behind = np.array([cube_pos[0] - 0.05, cube_pos[1], tcp_z])
        if not self._move_to_target(behind):
            return False

        # Verify contact with a short push; adjust height if needed
        for attempt in range(3):
            cube_before = self._entity_pos("obj")
            # Small forward push
            push_test = behind + np.array([0.02, 0.0, 0.0])
            if not self._move_to_target(push_test, max_steps=20):
                return False
            cube_after = self._entity_pos("obj")
            if np.linalg.norm(cube_after[:2] - cube_before[:2]) > 0.001:
                break
            # Lower TCP slightly and reposition behind
            tcp_z -= 0.01
            behind[2] = tcp_z
            if not self._move_to_target(behind):
                return False

        # Push to goal region
        push_target = np.array([goal_pos[0] - 0.12, goal_pos[1], tcp_z])
        if not self._move_to_target(push_target):
            return False

        return True

    def _move_to_target(self, target: np.ndarray, max_step: float = 0.02,
                        pos_threshold: float = 0.005, max_steps: int = 200,
                        gripper: float = 1.0) -> bool:
        """Move TCP to a world-frame target position using small delta steps."""
        for _ in range(max_steps):
            current = self._tcp_pos()
            delta = target - current
            dist = np.linalg.norm(delta)
            if dist <= pos_threshold:
                return True
            step = delta * min(1.0, max_step / dist)
            self._step(self._make_action(step, gripper=gripper))
            if self._early_stop():
                return False
        return False


def build_robot(env, *, control_mode: str, robot_uid: str):
    """Factory function required by the dynamic adapter harness."""
    return Xarm6RobotiqPushCubeAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
