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
        # Pregrasp pose: above cubeA, offset along world -z, keeping xy aligned.
        pregrasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(pregrasp, gripper=self.gripper_open, steps=self.move_steps)
            tcp = self._tcp_pos()
            if float(np.linalg.norm(tcp - pregrasp)) <= 0.02:
                break
        tcp = self._tcp_pos()
        cubeA = self._entity_pos("cubeA")
        # Completion: TCP is above cubeA and near the pregrasp xy.
        above = tcp[2] > cubeA[2] + 0.03
        near_xy = float(np.linalg.norm(tcp[:2] - cubeA[:2])) <= 0.05
        return bool(above and near_xy)

    def _stage_grasp(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        # Descend to grasp height around cubeA center.
        grasp_target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float32)
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
            self._repeat_action(np.zeros(3, dtype=np.float32),
                                gripper=self.gripper_close, steps=self.settle_steps)
            if self._is_grasping_entity("cubeA"):
                break
        return bool(self._is_grasping_entity("cubeA"))

    def _stage_lift(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        start_z = float(cubeA[2])
        lift_target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(lift_target, gripper=self.gripper_close, steps=self.move_steps)
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
        tcp = self._tcp_pos()
        # Account for TCP-to-held-object offset so cubeA lands above cubeB.
        offset = cubeA - tcp
        stack_z = cubeB[2] + 0.04
        align_target = np.array([cubeB[0] - offset[0], cubeB[1] - offset[1], stack_z - offset[2]], dtype=np.float32)
        for _ in range(10):
            if self._early_stop():
                return False
            self._move_towards(align_target, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            cubeB = self._entity_pos("cubeB")
            tcp = self._tcp_pos()
            offset = cubeA - tcp
            align_target = np.array([cubeB[0] - offset[0], cubeB[1] - offset[1], stack_z - offset[2]], dtype=np.float32)
            xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
            z_err = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04)
            if xy_err <= 0.008 and z_err <= 0.008:
                break
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04)
        grasped = bool(self._is_grasping_entity("cubeA"))
        return bool(xy_err <= 0.012 and z_err <= 0.012 and grasped)

    def _stage_release(self) -> bool:
        # First, open the gripper in place and let the cube settle.
        for _ in range(6):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32),
                                gripper=self.gripper_open, steps=self.settle_steps)
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True
            if (not bool(ev.get("is_cubeA_grasped", True))
                    and bool(ev.get("is_cubeA_static", False))
                    and bool(ev.get("is_cubeA_on_cubeB", False))):
                return True

        # If the cube is not on cubeB, re-grasp it and place it directly above cubeB.
        for _ in range(6):
            if self._early_stop():
                return False
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True
            if bool(ev.get("is_cubeA_on_cubeB", False)) and not bool(ev.get("is_cubeA_grasped", True)):
                return True

            pos_a = self._entity_pos("cubeA")
            pos_b = self._entity_pos("cubeB")

            # Move above cubeA and close the gripper to pick it up.
            above_a = np.array([pos_a[0], pos_a[1], pos_a[2] + 0.08], dtype=np.float32)
            self._move_towards(above_a, gripper=self.gripper_open, steps=self.move_steps)
            if self._early_stop():
                return False
            self._move_towards(np.array([pos_a[0], pos_a[1], pos_a[2] + 0.005], dtype=np.float32),
                               gripper=self.gripper_open, steps=self.move_steps)
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32),
                                gripper=self.gripper_close, steps=self.settle_steps)
            if self._early_stop():
                return False

            # Lift the cube.
            self._move_towards(np.array([pos_a[0], pos_a[1], pos_a[2] + 0.10], dtype=np.float32),
                               gripper=self.gripper_close, steps=self.move_steps)
            if self._early_stop():
                return False

            # Re-read cubeB position after lifting.
            pos_b = self._entity_pos("cubeB")
            target_xy = np.array([pos_b[0], pos_b[1]], dtype=np.float32)

            # Move above cubeB.
            self._move_towards(np.array([target_xy[0], target_xy[1], pos_b[2] + 0.10], dtype=np.float32),
                               gripper=self.gripper_close, steps=self.move_steps)
            if self._early_stop():
                return False

            # Lower to place cubeA on cubeB (cube half size ~0.02, so target z = pos_b[2] + 0.04).
            self._move_towards(np.array([target_xy[0], target_xy[1], pos_b[2] + 0.045], dtype=np.float32),
                               gripper=self.gripper_close, steps=self.move_steps)
            if self._early_stop():
                return False

            # Open gripper to release.
            for _ in range(4):
                if self._early_stop():
                    return False
                self._repeat_action(np.zeros(3, dtype=np.float32),
                                    gripper=self.gripper_open, steps=self.settle_steps)
                ev = self._official_evaluation()
                if bool(ev.get("success", False)):
                    return True
                if (not bool(ev.get("is_cubeA_grasped", True))
                        and bool(ev.get("is_cubeA_static", False))
                        and bool(ev.get("is_cubeA_on_cubeB", False))):
                    return True

            # Retreat upward to clear the cube.
            self._move_towards(np.array([target_xy[0], target_xy[1], pos_b[2] + 0.12], dtype=np.float32),
                               gripper=self.gripper_open, steps=self.move_steps)
            if self._early_stop():
                return False

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
