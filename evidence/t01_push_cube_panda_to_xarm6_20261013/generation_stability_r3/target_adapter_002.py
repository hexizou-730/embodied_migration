"""ManiSkill target adapter for PushCube-v1 with xarm6_robotiq (pd_ee_delta_pos)."""

import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class XArm6RobotiqTargetAdapter(ManiSkillDynamicRobot):
    def run_source_actions(self) -> bool:
        # 1. Close gripper
        for _ in range(20):
            self._step(self._make_action(np.zeros(3), gripper=1.0))

        # 2. Move TCP behind the cube
        behind_offset = np.array([-0.05, 0.0, 0.0])
        for _ in range(80):
            cube_pos = self._entity_pos("obj")
            target = cube_pos + behind_offset
            target[2] = cube_pos[2]  # lower the TCP to the table height
            if np.linalg.norm(target - self._tcp_pos()) < 0.005:
                break
            self._move_towards(target, gripper=1.0, steps=10)
            if self._official_success():
                return True

        # 3. Push cube towards goal region
        push_offset = np.array([-0.12, 0.0, 0.0])
        for _ in range(100):
            cube_pos = self._entity_pos("obj")
            goal_pos = self._entity_pos("goal_region")
            target = goal_pos + push_offset
            target[2] = cube_pos[2]  # keep TCP low to push the cube
            if np.linalg.norm(target - self._tcp_pos()) < 0.005:
                break
            self._move_towards(target, gripper=1.0, steps=10)
            if self._official_success():
                return True

        return self._official_success()


def build_robot(env, *, control_mode, robot_uid):
    return XArm6RobotiqTargetAdapter(env, control_mode=control_mode, robot_uid=robot_uid)
