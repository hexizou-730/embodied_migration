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
        # Pregrasp pose: above cubeA, offset in world frame.
        pregrasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(pregrasp, gripper=self.gripper_open, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - pregrasp) < 0.02:
                break
        tcp = self._tcp_pos()
        return bool(np.linalg.norm(tcp - pregrasp) < 0.03)

    def _stage_grasp(self) -> bool:
        cubeA = self._entity_pos("cubeA")
        # Descend to grasp height just above cube center.
        grasp_target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float32)
        for _ in range(6):
            if self._early_stop():
                return False
            self._move_towards(grasp_target, gripper=self.gripper_open, steps=self.move_steps)
            tcp = self._tcp_pos()
            if np.linalg.norm(tcp - grasp_target) < 0.02:
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
            if float(cubeA[2]) - start_z > 0.06:
                break
        cubeA = self._entity_pos("cubeA")
        lifted = (float(cubeA[2]) - start_z) > 0.05
        return bool(lifted and self._is_grasping_entity("cubeA"))

    def _stage_align(self) -> bool:
        cubeB = self._entity_pos("cubeB")
        # Target: cubeA center directly above cubeB center at stack height (2 * half_size).
        target_z = float(cubeB[2]) + 0.04
        for _ in range(12):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            tcp = self._tcp_pos()
            # Compensate for TCP-to-held-object offset so cubeA lands over cubeB.
            offset = cubeA - tcp
            desired_tcp = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32) - offset
            self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)
            cubeA = self._entity_pos("cubeA")
            cubeB = self._entity_pos("cubeB")
            xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
            z_err = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04)
            if xy_err < 0.008 and z_err < 0.008:
                break
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_ok = float(np.linalg.norm(cubeA[:2] - cubeB[:2])) <= 0.01
        z_ok = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04) <= 0.01
        return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))

    def _stage_release(self) -> bool:
        # First, open the gripper in place to release the cube.
        for _ in range(3):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3, dtype=np.float32),
                                gripper=self.gripper_open, steps=self.settle_steps)
            ev = self._official_evaluation()
            if not bool(ev.get("is_cubeA_grasped", True)):
                break

        # Measure the actual object poses and correct the cube placement relative to cubeB.
        cubeB = self._entity_pos("cubeB")
        cubeA = self._entity_pos("cubeA")
        half = 0.02
        target_xy = cubeB[:2]
        target_z = cubeB[2] + 2.0 * half

        # Move TCP above the cube to nudge it toward cubeB if needed.
        for _ in range(6):
            if self._early_stop():
                return False
            ev = self._official_evaluation()
            if bool(ev.get("is_cubeA_on_cubeB", False)) and bool(ev.get("is_cubeA_static", False)) \
                    and not bool(ev.get("is_cubeA_grasped", True)):
                return True

            cubeA = self._entity_pos("cubeA")
            cubeB = self._entity_pos("cubeB")
            offset = cubeA - cubeB
            # Desired cubeA position: centered on cubeB in xy, stacked in z.
            desired = np.array([cubeB[0], cubeB[1], cubeB[2] + 2.0 * half], dtype=np.float32)
            delta = desired - cubeA
            # Approach from above the cube, then push gently toward desired xy.
            tcp = self._tcp_pos()
            above = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.06], dtype=np.float32)
            if np.linalg.norm(tcp[:2] - above[:2]) > 0.01 or tcp[2] < above[2] - 0.01:
                self._move_towards(above, gripper=self.gripper_open, steps=8)
            else:
                push = np.array([delta[0], delta[1], 0.0], dtype=np.float32)
                push = np.clip(push, -0.03, 0.03)
                self._repeat_action(push, gripper=self.gripper_open, steps=4)

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
