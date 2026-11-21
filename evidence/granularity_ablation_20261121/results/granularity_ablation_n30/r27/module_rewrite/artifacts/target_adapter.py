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
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32)
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
        cubeA = self._entity_pos("cubeA")
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
            self._repeat_action(np.zeros(3, dtype=np.float32),
                                gripper=self.gripper_close, steps=self.settle_steps)
            if self._is_grasping_entity("cubeA"):
                break
        return bool(self._is_grasping_entity("cubeA"))

    def _stage_lift(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        start_z = float(cubeA[2])
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            if float(cubeA[2]) >= start_z + 0.06:
                break
        cubeA = self._entity_pos("cubeA")
        lifted = float(cubeA[2]) >= start_z + 0.05
        return bool(lifted and self._is_grasping_entity("cubeA"))

    def _stage_align(self) -> bool:
        # Source stack geometry: cubeA center must sit one cube height above cubeB center.
        # cube_half_size = 0.02, so target z offset = 0.04.
        # The held cubeA lags the TCP by a fixed offset; we drive the TCP so that
        # the measured cubeA lands on the target, correcting for the offset each step.
        cubeB = self._entity_pos("cubeB")
        stack_z = float(cubeB[2]) + 0.04
        target_xy = np.array([cubeB[0], cubeB[1]], dtype=np.float32)

        # Phase 1: hover above cubeB, correcting for held-object offset.
        for _ in range(8):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            offset = cubeA - tcp  # held-object offset from TCP
            desired_cubeA = np.array([target_xy[0], target_xy[1], stack_z + 0.06],
                                     dtype=np.float32)
            tcp_target = desired_cubeA - offset
            self._move_towards(tcp_target, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            if (np.linalg.norm(cubeA[:2] - target_xy) <= 0.01
                    and abs(float(cubeA[2]) - (stack_z + 0.06)) <= 0.02):
                break

        # Phase 2: descend so measured cubeA reaches stack height above cubeB.
        for _ in range(10):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            offset = cubeA - tcp
            desired_cubeA = np.array([target_xy[0], target_xy[1], stack_z],
                                     dtype=np.float32)
            tcp_target = desired_cubeA - offset
            self._move_towards(tcp_target, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            xy_err = float(np.linalg.norm(cubeA[:2] - target_xy))
            z_err = abs(float(cubeA[2]) - stack_z)
            if xy_err <= 0.008 and z_err <= 0.008:
                break

        cubeA = self._entity_pos("cubeA")
        xy_ok = bool(np.linalg.norm(cubeA[:2] - target_xy) <= 0.012)
        z_ok = bool(abs(float(cubeA[2]) - stack_z) <= 0.012)
        return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))

    def _stage_release(self) -> bool:
        for _ in range(8):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32),
                                gripper=self.gripper_open, steps=self.settle_steps)
            ev = self._official_evaluation()
            if bool(ev.get("is_cubeA_on_cubeB", False)) and bool(ev.get("is_cubeA_static", False)) \
                    and not bool(ev.get("is_cubeA_grasped", True)):
                return True
        ev = self._official_evaluation()
        return bool(ev.get("is_cubeA_on_cubeB", False)) and bool(ev.get("is_cubeA_static", False)) \
            and not bool(ev.get("is_cubeA_grasped", True))

    def run_source_actions(self) -> bool:
        if not self._run_stage("approach", self._stage_approach): return False
        if not self._run_stage("grasp", self._stage_grasp): return False
        if not self._run_stage("lift", self._stage_lift): return False
        if not self._run_stage("align", self._stage_align): return False
        if not self._run_stage("release", self._stage_release): return False
        return self._official_success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedSelectiveRobot:
    return GeneratedSelectiveRobot(env, control_mode=control_mode, robot_uid=robot_uid)
