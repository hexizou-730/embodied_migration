from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedSelectiveRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    def _stage_approach(self) -> bool:
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
        # Pregrasp pose: above cubeA, offset in world frame.
        pregrasp = cubeA.copy()
        pregrasp[2] = cubeA[2] + 0.10
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(pregrasp, gripper=0.0, steps=8)
            tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
            if float(np.linalg.norm(tcp - pregrasp)) <= 0.02:
                break
        tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
        return bool(float(np.linalg.norm(tcp - pregrasp)) <= 0.03)

    def _stage_grasp(self) -> bool:
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        grasp_target = cubeA.copy()
        grasp_target[2] = cubeA[2] + 0.005
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(grasp_target, gripper=0.0, steps=8)
            tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
            if float(np.linalg.norm(tcp - grasp_target)) <= 0.02:
                break
        # Close gripper at current position.
        for _ in range(3):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32), gripper=0.81, steps=4)
            if self._is_grasping_entity("cubeA"):
                break
        return bool(self._is_grasping_entity("cubeA"))

    def _stage_lift(self) -> bool:
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        start_z = float(cubeA[2])
        lift_target = cubeA.copy()
        lift_target[2] = cubeA[2] + 0.12
        for _ in range(8):
            if self._early_stop():
                return False
            self._move_towards(lift_target, gripper=0.81, steps=8)
            cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
            if float(cubeA[2]) >= start_z + 0.08:
                break
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        lifted = bool(float(cubeA[2]) >= start_z + 0.06)
        return bool(lifted and self._is_grasping_entity("cubeA"))

    def _stage_align(self) -> bool:
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float32).reshape(-1)
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
        cube_height = 0.04

        # Re-establish grasp if it was lost during lift/transit.
        if not self._is_grasping_entity("cubeA"):
            self._move_towards(cubeA, gripper=0.0, steps=8)
            self._repeat_action(np.zeros(3), gripper=0.81, steps=8)
            if self._early_stop():
                return False

        # Desired held-object center: directly above cubeB at one cube height.
        desired_A = cubeB.copy()
        desired_A[2] = cubeB[2] + cube_height

        for _ in range(20):
            if self._early_stop():
                return False
            cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
            tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
            xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
            z_err = float(abs(cubeA[2] - desired_A[2]))
            if xy_err <= 0.006 and z_err <= 0.006:
                break
            # Move TCP by the object's current offset from its desired pose.
            delta = desired_A - cubeA
            target = tcp + delta
            self._move_towards(target, gripper=0.81, steps=5)

        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = float(abs(cubeA[2] - (cubeB[2] + cube_height)))
        aligned = bool(xy_err <= 0.01 and z_err <= 0.01)
        return bool(aligned and self._is_grasping_entity("cubeA"))

    def _stage_release(self) -> bool:
        for _ in range(4):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32), gripper=0.0, steps=6)
            ev = self._official_evaluation()
            if bool(ev.get("is_cubeA_on_cubeB", False)) and bool(ev.get("is_cubeA_static", False)) and not bool(ev.get("is_cubeA_grasped", False)):
                break
        ev = self._official_evaluation()
        return bool(
            bool(ev.get("is_cubeA_on_cubeB", False))
            and bool(ev.get("is_cubeA_static", False))
            and not bool(ev.get("is_cubeA_grasped", False))
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
