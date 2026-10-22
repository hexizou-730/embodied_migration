from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedSelectiveRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    def _stage_approach(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        # Pregrasp pose: above cubeA, offset along approach direction.
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32)
        for _ in range(4):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=0.0, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - target) < 0.02:
                break
        tcp = self._tcp_pos()
        return bool(np.linalg.norm(tcp - target) < 0.03)

    def _stage_grasp(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float32)
        for _ in range(4):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=0.0, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - target) < 0.02:
                break
        # Close gripper at current position.
        for _ in range(2):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32), gripper=0.81, steps=self.settle_steps)
            if self._is_grasping_entity("cubeA"):
                break
        return bool(self._is_grasping_entity("cubeA"))

    def _stage_lift(self) -> bool:
        cubeA_start = self._entity_pos("cubeA")
        target = np.array([cubeA_start[0], cubeA_start[1], cubeA_start[2] + 0.10], dtype=np.float32)
        for _ in range(4):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=0.81, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            if cubeA[2] - cubeA_start[2] > 0.06:
                break
        cubeA = self._entity_pos("cubeA")
        lifted = bool(cubeA[2] - cubeA_start[2] > 0.05)
        return lifted and bool(self._is_grasping_entity("cubeA"))

    def _stage_align(self) -> bool:
        cubeB = self._entity_pos("cubeB")
        cubeA = self._entity_pos("cubeA")
        # Source stack geometry: cubeA center should sit one cube height above cubeB center.
        target_z = cubeB[2] + 0.04
        target = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32)
        # Move TCP so the held cubeA lands above cubeB; compensate for TCP-to-cubeA offset.
        for _ in range(8):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            offset = cubeA - tcp
            desired_tcp = target - offset
            self._move_towards(desired_tcp, gripper=0.81, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            xy_err = np.linalg.norm(cubeA[:2] - cubeB[:2])
            z_err = abs(cubeA[2] - target_z)
            if xy_err < 0.008 and z_err < 0.008:
                break
        cubeA = self._entity_pos("cubeA")
        xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) < 0.012)
        z_ok = bool(abs(cubeA[2] - target_z) < 0.012)
        return xy_ok and z_ok and bool(self._is_grasping_entity("cubeA"))

    def _stage_release(self) -> bool:
        for _ in range(3):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32), gripper=0.0, steps=self.settle_steps)
            ev = self._official_evaluation()
            if bool(ev.get("is_cubeA_on_cubeB", False)) and bool(ev.get("is_cubeA_static", False)) and not bool(ev.get("is_cubeA_grasped", False)):
                return True
        ev = self._official_evaluation()
        return bool(ev.get("is_cubeA_on_cubeB", False)) and bool(ev.get("is_cubeA_static", False)) and not bool(ev.get("is_cubeA_grasped", False))

    def run_source_actions(self) -> bool:
        if not self._run_stage("approach", self._stage_approach): return False
        if not self._run_stage("grasp", self._stage_grasp): return False
        if not self._run_stage("lift", self._stage_lift): return False
        if not self._run_stage("align", self._stage_align): return False
        if not self._run_stage("release", self._stage_release): return False
        return self._official_success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedSelectiveRobot:
    return GeneratedSelectiveRobot(env, control_mode=control_mode, robot_uid=robot_uid)
