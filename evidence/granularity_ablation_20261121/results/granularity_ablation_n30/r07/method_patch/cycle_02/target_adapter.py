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
        half = 0.02
        target_z = float(cubeB[2]) + 2.0 * half

        # Measure the TCP-to-held-cube offset so we can command the TCP to place cubeA.
        tcp = self._tcp_pos()
        offset = cubeA - tcp

        # Hover above cubeB first, keeping the held cube clear of the target.
        hover_cube = np.array([cubeB[0], cubeB[1], target_z + 0.10], dtype=np.float32)
        for _ in range(8):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            offset = cubeA - tcp
            hover_tcp = hover_cube - offset
            self._move_towards(hover_tcp, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            if np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.012:
                break

        # Descend and correct using measured cubeA error, commanding TCP via offset.
        for _ in range(14):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            offset = cubeA - tcp
            err = np.array([cubeB[0] - cubeA[0],
                            cubeB[1] - cubeA[1],
                            target_z - cubeA[2]], dtype=np.float32)
            norm = float(np.linalg.norm(err))
            if norm > 0.03:
                err = err * (0.03 / norm)
            cmd = err
            self._repeat_action(cmd, gripper=self.gripper_close, steps=1)
            cubeA = self._entity_pos("cubeA")
            if (np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.006
                    and abs(float(cubeA[2]) - target_z) <= 0.006):
                break

        cubeA = self._entity_pos("cubeA")
        xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008)
        z_ok = bool(abs(float(cubeA[2]) - target_z) <= 0.008)
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
