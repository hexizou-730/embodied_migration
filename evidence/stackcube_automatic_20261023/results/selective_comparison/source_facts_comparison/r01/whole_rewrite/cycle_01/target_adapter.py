from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _safe_pos(self, name):
        try:
            p = np.asarray(self._entity_pos(name), dtype=np.float64).reshape(-1)
            if p.size >= 3 and np.all(np.isfinite(p[:3])):
                return p[:3].copy()
        except Exception:
            pass
        return None

    def _tcp(self):
        try:
            p = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(-1)
            if p.size >= 3 and np.all(np.isfinite(p[:3])):
                return p[:3].copy()
        except Exception:
            pass
        return np.zeros(3, dtype=np.float64)

    def _move_to(self, target, gripper, steps, tol=0.008, max_iters=40):
        target = np.asarray(target, dtype=np.float64).reshape(-1)[:3]
        for _ in range(max_iters):
            if self._early_stop():
                return False
            cur = self._tcp()
            if np.linalg.norm(cur - target) <= tol:
                return True
            self._move_towards(target, gripper=gripper, steps=steps)
        return True

    def _hold(self, gripper, steps):
        for _ in range(max(1, steps)):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3), gripper=gripper, steps=1)
        return True

    def _grasped(self):
        try:
            return bool(self._is_grasping_entity("cubeA"))
        except Exception:
            return False

    def _success(self):
        try:
            ev = self._official_evaluation()
            return bool(ev.get("success", False))
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # policy
    # ------------------------------------------------------------------ #
    def run_source_actions(self) -> bool:
        cubeA = self._safe_pos("cubeA")
        cubeB = self._safe_pos("cubeB")
        if cubeA is None or cubeB is None:
            return False

        # Estimate cube half-size from resting height (cube sits on table at z=half).
        half = float(max(cubeA[2], cubeB[2], 0.01))
        if half <= 0.0:
            half = 0.02

        # ---------------- approach ---------------- #
        # Pregrasp: above cubeA, offset down by ~5cm from grasp height.
        grasp_z = cubeA[2] + half * 0.6
        pregrasp = np.array([cubeA[0], cubeA[1], grasp_z + 0.06], dtype=np.float64)
        if not self._move_to(pregrasp, gripper=0.0, steps=self.move_steps):
            return False
        if self._early_stop():
            return False

        # ---------------- grasp ---------------- #
        grasp_target = np.array([cubeA[0], cubeA[1], grasp_z], dtype=np.float64)
        if not self._move_to(grasp_target, gripper=0.0, steps=self.move_steps):
            return False
        if self._early_stop():
            return False

        # Close gripper in place.
        self._hold(0.81, self.settle_steps)
        if self._early_stop():
            return False

        # If grasp not detected, try a small downward nudge and re-close.
        if not self._grasped():
            nudge = np.array([cubeA[0], cubeA[1], grasp_z - 0.01], dtype=np.float64)
            self._move_to(nudge, gripper=0.81, steps=self.move_steps)
            self._hold(0.81, self.settle_steps)
            if self._early_stop():
                return False

        # ---------------- lift ---------------- #
        lift_target = np.array([cubeA[0], cubeA[1], grasp_z + 0.12], dtype=np.float64)
        if not self._move_to(lift_target, gripper=0.81, steps=self.move_steps):
            return False
        if self._early_stop():
            return False
        self._hold(0.81, self.settle_steps)
        if self._early_stop():
            return False

        # ---------------- align ---------------- #
        # Goal: cubeA centered above cubeB at stack height (2*half above cubeB base).
        stack_z = cubeB[2] + 2.0 * half
        # TCP should be at stack_z + (grasp_z - cubeA[2]) offset so held cube lands at stack_z.
        tcp_offset_z = grasp_z - cubeA[2]
        align_target = np.array([cubeB[0], cubeB[1], stack_z + tcp_offset_z + 0.02],
                                dtype=np.float64)
        if not self._move_to(align_target, gripper=0.81, steps=self.move_steps):
            return False
        if self._early_stop():
            return False

        # Lower gently onto cubeB.
        place_target = np.array([cubeB[0], cubeB[1], stack_z + tcp_offset_z],
                                dtype=np.float64)
        if not self._move_to(place_target, gripper=0.81, steps=self.move_steps):
            return False
        if self._early_stop():
            return False
        self._hold(0.81, self.settle_steps)
        if self._early_stop():
            return False

        # ---------------- release ---------------- #
        self._hold(0.0, self.settle_steps)
        if self._early_stop():
            return False

        # Retreat upward to clear the stack while gripper stays open.
        retreat = np.array([cubeB[0], cubeB[1], stack_z + tcp_offset_z + 0.12],
                           dtype=np.float64)
        self._move_to(retreat, gripper=0.0, steps=self.move_steps)
        if self._early_stop():
            return False

        # Let physics settle so cubeA becomes static on cubeB.
        for _ in range(6):
            if self._early_stop():
                break
            self._hold(0.0, self.settle_steps)
            if self._success():
                return True

        return self._success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
