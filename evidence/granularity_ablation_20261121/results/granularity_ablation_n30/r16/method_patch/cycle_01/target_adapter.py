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
        # Pregrasp pose: above cubeA, collision-clear.
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
        # Descend to grasp height around cubeA.
        target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(target, gripper=0.0, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - target) <= 0.015:
                break
        # Close gripper in place.
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
        cubeB = self._entity_pos("cubeB")
        cubeA = self._entity_pos("cubeA")
        # Target cubeA center: directly above cubeB at one cube height (2 * half_size).
        half = 0.02
        target_z = float(cubeB[2]) + 2.0 * half
        target_xy = np.array([cubeB[0], cubeB[1]], dtype=np.float32)

        # Compute TCP-to-held-cube offset so we can drive the cube, not just the TCP.
        offset = np.array(cubeA, dtype=np.float32) - np.array(self._tcp_pos(), dtype=np.float32)

        # Phase 1: raise to a safe hover height above the stack target.
        hover_z = target_z + 0.08
        for _ in range(8):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            desired_cube = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
            desired_tcp = desired_cube - offset
            delta = desired_tcp - np.array(tcp, dtype=np.float32)
            if np.linalg.norm(delta) <= 0.01:
                break
            self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)

        # Phase 2: align XY precisely over cubeB while staying at hover height.
        for _ in range(8):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            desired_cube = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
            desired_tcp = desired_cube - offset
            delta = desired_tcp - np.array(tcp, dtype=np.float32)
            if np.linalg.norm(delta[:2]) <= 0.005:
                break
            self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)

        # Phase 3: descend slowly to final stack height, re-reading cube pose each step.
        for _ in range(10):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            desired_cube = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
            desired_tcp = desired_cube - offset
            delta = desired_tcp - np.array(tcp, dtype=np.float32)
            if np.linalg.norm(delta) <= 0.008:
                break
            self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)

        # Settle in place to let the held cube stabilize.
        for _ in range(3):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)

        # Measure completion against the official object-relative predicate.
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(np.array(cubeA[:2]) - np.array(cubeB[:2])))
        z_err = float(abs((float(cubeA[2]) - float(cubeB[2])) - 2.0 * half))
        xy_ok = xy_err <= 0.01
        z_ok = z_err <= 0.01
        return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))

    def _stage_release(self) -> bool:
        # Open gripper in place and wait for official static/released conditions.
        for _ in range(6):
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
