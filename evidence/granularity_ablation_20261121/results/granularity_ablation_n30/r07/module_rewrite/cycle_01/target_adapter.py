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
        # Source relation: place cubeA directly above cubeB at stack height.
        # cube_half_size = 0.02, so stack offset is 0.04 in z.
        # We must drive the *held cube* to the target, not the TCP, since the
        # TCP-to-held-object transform is unknown for xarm6_robotiq.
        # Strategy: iteratively correct using measured cubeA position error.
        stack_dz = 0.04
        for _ in range(14):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            cubeB = self._entity_pos("cubeB")
            # Desired cubeA position: directly above cubeB at stack height.
            desired = np.array([cubeB[0], cubeB[1], cubeB[2] + stack_dz],
                               dtype=np.float32)
            err = desired - cubeA
            # First lift clear above cubeB before lateral move to avoid collision.
            if float(cubeA[2]) < float(cubeB[2]) + stack_dz + 0.03:
                # Raise cubeA to a safe hover height above cubeB.
                hover = np.array([cubeA[0], cubeA[1],
                                  cubeB[2] + stack_dz + 0.06], dtype=np.float32)
                tcp = self._tcp_pos()
                delta = hover - cubeA
                tcp_target = tcp + delta
                self._move_towards(tcp_target, gripper=self.gripper_close,
                                   steps=self.move_steps)
                continue
            # Now correct lateral and vertical error using measured cube offset.
            if np.linalg.norm(err) <= 0.008:
                break
            tcp = self._tcp_pos()
            # Apply the measured cube error to the TCP target.
            tcp_target = tcp + err
            # Clamp per-step correction to avoid overshoot.
            step_vec = tcp_target - tcp
            norm = float(np.linalg.norm(step_vec))
            if norm > 0.05:
                step_vec = step_vec * (0.05 / norm)
                tcp_target = tcp + step_vec
            self._move_towards(tcp_target, gripper=self.gripper_close,
                               steps=self.move_steps)
        # Final verification against official geometry.
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        offset = cubeA - cubeB
        xy_ok = bool(np.linalg.norm(offset[:2]) <= 0.02)
        z_ok = bool(abs(float(offset[2]) - stack_dz) <= 0.012)
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
