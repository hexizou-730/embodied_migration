from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedSelectiveRobot(ManiSkillDynamicRobot):

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid, gripper_open=0.0, gripper_close=0.81, move_steps=30, settle_steps=12, max_delta_m=0.04)

    def _stage_approach(self) -> bool:
        cubeA = self._entity_pos('cubeA')
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.1], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=0.0, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - target) <= 0.02:
                break
        tcp = self._tcp_pos()
        return bool(np.linalg.norm(tcp - target) <= 0.03)

    def _stage_grasp(self) -> bool:
        cubeA = self._entity_pos('cubeA')
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=0.0, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - target) <= 0.015:
                break
        for _ in range(3):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)
            if self._is_grasping_entity('cubeA'):
                break
        return bool(self._is_grasping_entity('cubeA'))

    def _stage_lift(self) -> bool:
        cubeA = self._entity_pos('cubeA')
        start_z = float(cubeA[2])
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.1], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos('cubeA')
            if float(cubeA[2]) >= start_z + 0.06:
                break
        cubeA = self._entity_pos('cubeA')
        lifted = float(cubeA[2]) >= start_z + 0.05
        return bool(lifted and self._is_grasping_entity('cubeA'))

    def _stage_align(self) -> bool:
        cubeB = self._entity_pos('cubeB')
        goal = cubeB.copy()
        goal[2] += 0.04
        hover = goal.copy()
        hover[2] += 0.07
        for (target, limit) in ((hover, 120), (goal, 100)):
            for _ in range(limit):
                if self._early_stop():
                    return False
                cubeA = self._entity_pos('cubeA')
                error = target - cubeA
                if np.linalg.norm(error) < 0.003:
                    break
                command = np.clip(error / 0.1, -0.06, 0.06)
                self._step(self._make_action(command, gripper=self.gripper_close))
            self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=6)
        cubeA = self._entity_pos('cubeA')
        ev = self._official_evaluation()
        return bool(np.linalg.norm(cubeA[:2] - goal[:2]) < 0.008 and ev.get('is_cubeA_on_cubeB', False) and self._is_grasping_entity('cubeA'))

    def _stage_release(self) -> bool:
        cubeA = self._entity_pos('cubeA')
        for command in np.linspace(self.gripper_close, self.gripper_open, 9)[1:]:
            if self._early_stop():
                return self._official_success()
            self._repeat_action(np.zeros(3), gripper=float(command), steps=3)
        retreat = self._tcp_pos() + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        self._move_towards(retreat, gripper=self.gripper_open, steps=30)
        for _ in range(20):
            if self._early_stop():
                return self._official_success()
            self._repeat_action(np.zeros(3), gripper=self.gripper_open, steps=1)
            if self._official_success():
                return True
        return self._official_success()

    def run_source_actions(self) -> bool:
        if not self._run_stage('approach', self._stage_approach):
            return False
        if not self._run_stage('grasp', self._stage_grasp):
            return False
        if not self._run_stage('lift', self._stage_lift):
            return False
        if not self._run_stage('align', self._stage_align):
            return False
        if not self._run_stage('release', self._stage_release):
            return False
        return self._official_success()

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedSelectiveRobot:
    return GeneratedSelectiveRobot(env, control_mode=control_mode, robot_uid=robot_uid)
