from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    def _safe_move(self, target, gripper, steps):
        try:
            self._move_towards(np.asarray(target, dtype=np.float64),
                               gripper=float(gripper), steps=int(steps))
        except Exception:
            pass

    def _safe_repeat(self, delta, gripper, steps):
        try:
            self._repeat_action(np.asarray(delta, dtype=np.float64),
                                gripper=float(gripper), steps=int(steps))
        except Exception:
            pass

    def _stop(self):
        try:
            return bool(self._early_stop())
        except Exception:
            return False

    def _pos(self, name):
        try:
            return np.asarray(self._entity_pos(name), dtype=np.float64).reshape(-1)[:3]
        except Exception:
            return None

    def _tcp(self):
        try:
            return np.asarray(self._tcp_pos(), dtype=np.float64).reshape(-1)[:3]
        except Exception:
            return None

    def _grasped(self, name):
        try:
            return bool(self._is_grasping_entity(name))
        except Exception:
            return False

    def _success(self):
        try:
            return bool(self._official_evaluation().get("success", False))
        except Exception:
            return False

    def _align_xy(self, target_xy, gripper, steps, tol=0.006, max_iters=8):
        """Iteratively drive TCP xy toward target using measured feedback."""
        for _ in range(max_iters):
            if self._stop():
                return True
            tcp = self._tcp()
            if tcp is None:
                return False
            err = np.asarray(target_xy, dtype=np.float64) - tcp[:2]
            if float(np.linalg.norm(err)) <= tol:
                return False
            # bounded step toward target in xy, keep current z
            step = np.clip(err, -0.03, 0.03)
            goal = np.array([tcp[0] + step[0], tcp[1] + step[1], tcp[2]])
            self._safe_move(goal, gripper, steps)
        return False

    def _descend_to(self, target_z, gripper, steps, tol=0.004, max_iters=10):
        for _ in range(max_iters):
            if self._stop():
                return True
            tcp = self._tcp()
            if tcp is None:
                return False
            dz = float(target_z) - float(tcp[2])
            if abs(dz) <= tol:
                return False
            step = float(np.clip(dz, -0.03, 0.03))
            goal = np.array([tcp[0], tcp[1], tcp[2] + step])
            self._safe_move(goal, gripper, steps)
        return False

    def _grasp_sequence(self, pos_obj, grasp_z, lift_z):
        """Approach, descend, close, verify grasp, lift."""
        # Move above object
        above = np.array([pos_obj[0], pos_obj[1], pos_obj[2] + 0.12])
        self._safe_move(above, 0.0, 40)
        if self._stop():
            return True

        # Align xy precisely above object
        if self._align_xy(pos_obj[:2], 0.0, 20):
            return True

        # Descend to grasp height
        if self._descend_to(grasp_z, 0.0, 20):
            return True

        # Close gripper
        self._safe_repeat(np.zeros(3), 0.81, 25)
        if self._stop():
            return True

        # Verify grasp via detector; if not grasped, try small re-descend and close again
        if not self._grasped("cubeA"):
            tcp = self._tcp()
            if tcp is not None:
                # nudge down a bit more and re-close
                goal = np.array([tcp[0], tcp[1], tcp[2] - 0.01])
                self._safe_move(goal, 0.0, 15)
                self._safe_repeat(np.zeros(3), 0.81, 20)
                if self._stop():
                    return True

        # Lift
        lift = np.array([pos_obj[0], pos_obj[1], lift_z])
        self._safe_move(lift, 0.81, 40)
        if self._stop():
            return True
        return False

    def run_source_actions(self) -> bool:
        posA = self._pos("cubeA")
        posB = self._pos("cubeB")
        if posA is None or posB is None:
            return False

        # Cube half size ~0.02; grasp near top of cubeA
        grasp_z_A = float(posA[2]) + 0.005
        lift_z_A = float(posA[2]) + 0.16

        # Phase 1: grasp cubeA
        if self._grasp_sequence(posA, grasp_z_A, lift_z_A):
            return self._success()

        # Confirm grasp; if failed, retry once with slightly lower grasp
        if not self._grasped("cubeA"):
            posA = self._pos("cubeA")
            if posA is None:
                return False
            grasp_z_A = float(posA[2]) - 0.002
            lift_z_A = float(posA[2]) + 0.16
            if self._grasp_sequence(posA, grasp_z_A, lift_z_A):
                return self._success()

        # Phase 2: transport above cubeB
        posB = self._pos("cubeB")
        if posB is None:
            return False
        above_B = np.array([posB[0], posB[1], float(posB[2]) + 0.16])
        self._safe_move(above_B, 0.81, 60)
        if self._stop():
            return self._success()

        # Align xy over cubeB
        if self._align_xy(posB[:2], 0.81, 20):
            return self._success()

        # Phase 3: lower to stack height (cubeB top + cubeA half)
        stack_z = float(posB[2]) + 0.04 + 0.005
        if self._descend_to(stack_z, 0.81, 20):
            return self._success()

        # Release
        self._safe_repeat(np.zeros(3), 0.0, 25)
        if self._stop():
            return self._success()

        # Retreat upward to let cube settle
        retreat = np.array([posB[0], posB[1], stack_z + 0.12])
        self._safe_move(retreat, 0.0, 40)
        if self._stop():
            return self._success()

        # Settle in place
        self._safe_repeat(np.zeros(3), 0.0, 25)

        return self._success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
