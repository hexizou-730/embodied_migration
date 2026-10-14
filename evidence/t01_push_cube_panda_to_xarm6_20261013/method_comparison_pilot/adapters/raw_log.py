"""Target adapter for xarm6_robotiq with pd_ee_delta_pos control, replicating the push-cube source policy."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class TargetRobotAdapter(ManiSkillDynamicRobot):
    """Adapter that implements run_source_actions() using EE delta commands."""

    def run_source_actions(self) -> bool:
        """Execute the push-cube task: close gripper, move to pre-push pose, then push to goal."""
        # 1. Close the gripper
        for _ in range(10):
            self._step(self._make_action(np.zeros(3), gripper=1.0))
            if self._early_stop():
                break

        # 2. Move to a pose just behind the cube (reach pose)
        obj_pos = self._entity_pos("obj")
        reach_target = obj_pos + np.array([-0.05, 0.0, 0.0])
        self._move_towards(reach_target, gripper=1.0, steps=50)

        # 3. Push the cube towards the goal region in small segments
        goal_pos = self._entity_pos("goal_region")
        goal_target = goal_pos + np.array([-0.12, 0.0, 0.0])
        self._move_line_in_segments(goal_target)

        # 4. Report whether the official success condition is met
        return self._official_success()

    def _move_line_in_segments(
        self,
        target_pos: np.ndarray,
        max_segment: float = 0.04,
        steps_per_segment: int = 20,
    ) -> None:
        """Move the TCP towards target_pos in straight segments, replanning each segment."""
        while True:
            current_pos = self._tcp_pos()
            delta = target_pos - current_pos
            dist = float(np.linalg.norm(delta))
            if dist < 0.005:
                break
            step = delta * min(1.0, max_segment / dist)
            waypoint = current_pos + step
            self._move_towards(waypoint, gripper=1.0, steps=steps_per_segment)
            if self._early_stop():
                break


def build_robot(env, *, control_mode: str, robot_uid: str) -> TargetRobotAdapter:
    """Factory function required by the dynamic adapter harness."""
    return TargetRobotAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
