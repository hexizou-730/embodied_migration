from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # ---- helpers -------------------------------------------------------
    def _safe_move(self, target, gripper, steps, tol=0.008, max_iters=40):
        target = np.asarray(target, dtype=np.float64).reshape(3)
        for _ in range(max_iters):
            if self._early_stop():
                return False
            cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
            if np.linalg.norm(cur - target) <= tol:
                return True
            self._move_towards(target, gripper=gripper, steps=steps)
        return True

    def _hold(self, gripper, steps):
        for _ in range(max(1, steps // 6)):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3), gripper=gripper, steps=6)
        return True

    def _grasp_pose_for(self, cube_pos):
        # Top-down grasp: TCP directly above cube, gripper closing along y.
        cube_pos = np.asarray(cube_pos, dtype=np.float64).reshape(3)
        # TCP offset: approach from above, finger length ~0.025, cube half ~0.02
        grasp_z = cube_pos[2] + 0.025 + 0.02
        return np.array([cube_pos[0], cube_pos[1], grasp_z], dtype=np.float64)

    # ---- main policy ---------------------------------------------------
    def run_source_actions(self) -> bool:
        # Stage 0: read initial measured state
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)

        grasp_pose = self._grasp_pose_for(cubeA)
        pregrasp = grasp_pose.copy()
        pregrasp[2] += 0.06

        # ---- approach --------------------------------------------------
        if not self._safe_move(pregrasp, gripper=0.0, steps=self.move_steps):
            return False
        if not self._safe_move(grasp_pose, gripper=0.0, steps=self.move_steps):
            return False
        if not self._hold(0.0, self.settle_steps):
            return False

        # ---- grasp -----------------------------------------------------
        if not self._hold(0.81, self.settle_steps):
            return False
        # verify grasp; if not, nudge down and retry once
        if not self._is_grasping_entity("cubeA"):
            cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
            nudge = cur.copy()
            nudge[2] -= 0.01
            self._safe_move(nudge, gripper=0.81, steps=self.move_steps)
            self._hold(0.81, self.settle_steps)
        if not self._is_grasping_entity("cubeA"):
            return False

        # ---- lift ------------------------------------------------------
        cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        lift_target = cur.copy()
        lift_target[2] += 0.12
        if not self._safe_move(lift_target, gripper=0.81, steps=self.move_steps):
            return False
        if not self._hold(0.81, self.settle_steps):
            return False
        if not self._is_grasping_entity("cubeA"):
            return False

        # ---- align above cubeB ----------------------------------------
        # Re-measure cubeB and current held cubeA to compute placement offset.
        cubeB_now = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        cubeA_now = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        tcp_now = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)

        # Desired cubeA position: directly above cubeB by 2*half_size (~0.04)
        stack_height = 0.04
        desired_A = np.array([cubeB_now[0], cubeB_now[1], cubeB_now[2] + stack_height],
                             dtype=np.float64)
        # TCP-to-cubeA offset in world frame (preserve grasp geometry)
        offset = tcp_now - cubeA_now
        align_target = desired_A + offset
        # Slight overshoot in z to ensure contact, then settle
        align_target[2] += 0.005

        if not self._safe_move(align_target, gripper=0.81, steps=self.move_steps):
            return False
        if not self._hold(0.81, self.settle_steps):
            return False

        # ---- release ---------------------------------------------------
        if not self._hold(0.0, self.settle_steps):
            return False
        # retreat upward to clear the stack
        cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        retreat = cur.copy()
        retreat[2] += 0.08
        self._safe_move(retreat, gripper=0.0, steps=self.move_steps)
        self._hold(0.0, self.settle_steps)

        # Final verification loop: wait for official success (static + on cubeB + not grasped)
        for _ in range(60):
            if self._early_stop():
                break
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True
            self._repeat_action(np.zeros(3), gripper=0.0, steps=4)

        ev = self._official_evaluation()
        return bool(ev.get("success", False))


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
