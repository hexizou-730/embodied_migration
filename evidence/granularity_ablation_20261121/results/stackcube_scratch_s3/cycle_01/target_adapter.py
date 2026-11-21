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
        # Pregrasp pose: above cubeA, offset along -z by a clearance distance.
        pregrasp = np.array(
            [cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32
        )
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(pregrasp, gripper=self.gripper_open, steps=self.move_steps)
            tcp = self._tcp_pos()
            if float(np.linalg.norm(tcp - pregrasp)) <= 0.02:
                break
        tcp = self._tcp_pos()
        return bool(float(np.linalg.norm(tcp - pregrasp)) <= 0.05)

    def _stage_grasp(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        # Descend to grasp height at cubeA center.
        grasp_target = np.array(
            [cubeA[0], cubeA[1], cubeA[2]], dtype=np.float32
        )
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(grasp_target, gripper=self.gripper_open, steps=self.move_steps)
            tcp = self._tcp_pos()
            if float(np.linalg.norm(tcp - grasp_target)) <= 0.02:
                break
        # Close gripper in place.
        for _ in range(3):
            if self._early_stop():
                return False
            self._repeat_action(
                np.zeros(3, dtype=np.float32),
                gripper=self.gripper_close,
                steps=self.settle_steps,
            )
            if self._is_grasping_entity("cubeA"):
                break
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
            self._move_towards(lift_target, gripper=self.gripper_close, steps=self.move_steps)
            if not self._is_grasping_entity("cubeA"):
                return False
            cubeA_now = self._entity_pos("cubeA")
            if float(cubeA_now[2]) >= start_z + 0.05:
                break
        cubeA_now = self._entity_pos("cubeA")
        return bool(
            self._is_grasping_entity("cubeA")
            and float(cubeA_now[2]) >= start_z + 0.03
        )

    def _stage_align(self) -> bool:
        cubeB = self._entity_pos("cubeB")
        cubeA = self._entity_pos("cubeA")
        # Place cubeA above cubeB using source object-relative stack geometry.
        align_target = np.array(
            [cubeB[0], cubeB[1], cubeB[2] + 0.04], dtype=np.float32
        )
        for _ in range(8):
            if self._early_stop():
                return False
            self._move_towards(align_target, gripper=self.gripper_close, steps=self.move_steps)
            if not self._is_grasping_entity("cubeA"):
                return False
            cubeA_now = self._entity_pos("cubeA")
            xy_err = float(np.linalg.norm(cubeA_now[:2] - cubeB[:2]))
            z_err = float(abs(cubeA_now[2] - (cubeB[2] + 0.04)))
            if xy_err <= 0.02 and z_err <= 0.02:
                break
        cubeA_now = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA_now[:2] - cubeB[:2]))
        z_err = float(abs(cubeA_now[2] - (cubeB[2] + 0.04)))
        return bool(
            self._is_grasping_entity("cubeA")
            and xy_err <= 0.04
            and z_err <= 0.04
        )

    def _stage_release(self) -> bool:
        # Open gripper in place and wait for official released/static conditions.
        for _ in range(6):
            if self._early_stop():
                return False
            self._repeat_action(
                np.zeros(3, dtype=np.float32),
                gripper=self.gripper_open,
                steps=self.settle_steps,
            )
            ev = self._official_evaluation()
            if (
                not bool(ev.get("is_cubeA_grasped", True))
                and bool(ev.get("is_cubeA_on_cubeB", False))
                and bool(ev.get("is_cubeA_static", False))
            ):
                break
        ev = self._official_evaluation()
        return bool(
            not bool(ev.get("is_cubeA_grasped", True))
            and bool(ev.get("is_cubeA_on_cubeB", False))
            and bool(ev.get("is_cubeA_static", False))
        )

    def run_source_actions(self) -> bool:
        if not self._run_stage("approach", self._stage_approach): return False
        if not self._run_stage("grasp", self._stage_grasp): return False
        if not self._run_stage("lift", self._stage_lift): return False
        if not self._run_stage("align", self._stage_align): return False
        if not self._run_stage("release", self._stage_release): return False
        return self._official_success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedSelectiveRobot:
    return GeneratedSelectiveRobot(env, control_mode=control_mode, robot_uid=robot_uid)
