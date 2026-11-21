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
        pregrasp = np.array(
            [cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32
        )
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(pregrasp, gripper=self.gripper_open, steps=8)
            tcp = self._tcp_pos()
            if float(np.linalg.norm(tcp - pregrasp)) <= 0.02:
                break
        tcp = self._tcp_pos()
        return bool(float(np.linalg.norm(tcp - pregrasp)) <= 0.05)

    def _stage_grasp(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        grasp_target = np.array(
            [cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float32
        )
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(grasp_target, gripper=self.gripper_open, steps=8)
            tcp = self._tcp_pos()
            if float(np.linalg.norm(tcp - grasp_target)) <= 0.015:
                break
        if self._early_stop():
            return False
        # Close gripper in place.
        self._repeat_action(
            np.zeros(3, dtype=np.float32),
            gripper=self.gripper_close,
            steps=self.settle_steps,
        )
        for _ in range(4):
            if self._early_stop():
                return False
            if self._is_grasping_entity("cubeA"):
                return True
            self._repeat_action(
                np.zeros(3, dtype=np.float32),
                gripper=self.gripper_close,
                steps=4,
            )
        return bool(self._is_grasping_entity("cubeA"))

    def _stage_lift(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        start_z = float(cubeA[2])
        lift_target = np.array(
            [cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32
        )
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(lift_target, gripper=self.gripper_close, steps=8)
            cubeA = self._entity_pos("cubeA")
            if float(cubeA[2]) >= start_z + 0.06:
                break
        cubeA = self._entity_pos("cubeA")
        lifted = float(cubeA[2]) >= start_z + 0.04
        grasped = bool(self._is_grasping_entity("cubeA"))
        return bool(lifted and grasped)

    def _stage_align(self) -> bool:
        cubeB = self._entity_pos("cubeB")
        cubeA = self._entity_pos("cubeA")
        # Use the measured TCP-to-held-object offset so the cube, not the TCP,
        # lands centered above cubeB at the official stack height.
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        target_z = float(cubeB[2]) + 0.04 + float(offset[2])
        align_target = np.array(
            [cubeB[0] - offset[0], cubeB[1] - offset[1], target_z],
            dtype=np.float32,
        )
        for _ in range(10):
            if self._early_stop():
                return False
            self._move_towards(align_target, gripper=self.gripper_close, steps=6)
            cubeA = self._entity_pos("cubeA")
            cubeB = self._entity_pos("cubeB")
            xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
            z_err = abs(float(cubeA[2] - cubeB[2]) - 0.04)
            if xy_err <= 0.008 and z_err <= 0.008:
                break
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_ok = float(np.linalg.norm(cubeA[:2] - cubeB[:2])) <= 0.012
        z_ok = abs(float(cubeA[2] - cubeB[2]) - 0.04) <= 0.012
        grasped = bool(self._is_grasping_entity("cubeA"))
        return bool(xy_ok and z_ok and grasped)

    def _stage_release(self) -> bool:
        if self._early_stop():
            return False
        self._repeat_action(
            np.zeros(3, dtype=np.float32),
            gripper=self.gripper_open,
            steps=self.settle_steps,
        )
        for _ in range(8):
            if self._early_stop():
                return False
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True
            self._repeat_action(
                np.zeros(3, dtype=np.float32),
                gripper=self.gripper_open,
                steps=4,
            )
        ev = self._official_evaluation()
        return bool(ev.get("success", False))

    def run_source_actions(self) -> bool:
        if not self._run_stage("approach", self._stage_approach): return False
        if not self._run_stage("grasp", self._stage_grasp): return False
        if not self._run_stage("lift", self._stage_lift): return False
        if not self._run_stage("align", self._stage_align): return False
        if not self._run_stage("release", self._stage_release): return False
        return self._official_success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedSelectiveRobot:
    return GeneratedSelectiveRobot(env, control_mode=control_mode, robot_uid=robot_uid)
