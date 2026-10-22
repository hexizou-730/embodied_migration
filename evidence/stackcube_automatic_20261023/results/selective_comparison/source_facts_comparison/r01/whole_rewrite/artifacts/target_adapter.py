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

    def _move_to(self, target, gripper, steps, tol=0.006, max_iters=60):
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

    def _eval(self):
        try:
            return self._official_evaluation()
        except Exception:
            return {}

    # ------------------------------------------------------------------ #
    # policy
    # ------------------------------------------------------------------ #
    def run_source_actions(self) -> bool:
        cubeA = self._safe_pos("cubeA")
        cubeB = self._safe_pos("cubeB")
        if cubeA is None or cubeB is None:
            return False

        # Cube half-size: cubes rest on table so z ~= half_size.
        half = float(max(cubeA[2], cubeB[2], 0.01))
        if half <= 0.0:
            half = 0.02

        # ---------------- approach ---------------- #
        pregrasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.12], dtype=np.float64)
        if not self._move_to(pregrasp, gripper=0.0, steps=self.move_steps):
            return False
        if self._early_stop():
            return False

        # ---------------- grasp ---------------- #
        grasp_z = cubeA[2] + half * 0.9
        grasp_target = np.array([cubeA[0], cubeA[1], grasp_z], dtype=np.float64)
        if not self._move_to(grasp_target, gripper=0.0, steps=self.move_steps):
            return False
        if self._early_stop():
            return False

        self._hold(0.81, self.settle_steps)
        if self._early_stop():
            return False

        if not self._grasped():
            for dz in (0.008, 0.016, 0.024):
                nudge = np.array([cubeA[0], cubeA[1], grasp_z - dz], dtype=np.float64)
                self._move_to(nudge, gripper=0.81, steps=self.move_steps)
                self._hold(0.81, self.settle_steps)
                if self._early_stop():
                    return False
                if self._grasped():
                    break

        if not self._grasped():
            back = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float64)
            self._move_to(back, gripper=0.0, steps=self.move_steps)
            self._hold(0.0, self.settle_steps)
            cubeA2 = self._safe_pos("cubeA")
            if cubeA2 is not None:
                cubeA = cubeA2
            grasp_z2 = cubeA[2] + half * 0.75
            retry = np.array([cubeA[0], cubeA[1], grasp_z2], dtype=np.float64)
            self._move_to(retry, gripper=0.0, steps=self.move_steps)
            self._hold(0.81, self.settle_steps)
            if self._early_stop():
                return False
            grasp_z = grasp_z2

        # ---------------- lift ---------------- #
        lift_target = np.array([cubeA[0], cubeA[1], grasp_z + 0.12], dtype=np.float64)
        if not self._move_to(lift_target, gripper=0.81, steps=self.move_steps):
            return False
        if self._early_stop():
            return False
        self._hold(0.81, self.settle_steps)
        if self._early_stop():
            return False

        lifted = self._safe_pos("cubeA")
        if lifted is not None and lifted[2] < cubeA[2] + 0.03:
            back = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float64)
            self._move_to(back, gripper=0.0, steps=self.move_steps)
            self._hold(0.0, self.settle_steps)
            cubeA_r = self._safe_pos("cubeA")
            if cubeA_r is not None:
                cubeA = cubeA_r
            grasp_z = cubeA[2] + half * 0.9
            self._move_to(np.array([cubeA[0], cubeA[1], grasp_z]), gripper=0.0,
                          steps=self.move_steps)
            self._hold(0.81, self.settle_steps)
            self._move_to(np.array([cubeA[0], cubeA[1], grasp_z + 0.12]),
                          gripper=0.81, steps=self.move_steps)
            self._hold(0.81, self.settle_steps)
            if self._early_stop():
                return False

        # ---------------- align ---------------- #
        cubeA_r = self._safe_pos("cubeA")
        if cubeA_r is not None:
            cubeA = cubeA_r
        cubeB_r = self._safe_pos("cubeB")
        if cubeB_r is not None:
            cubeB = cubeB_r

        stack_z = cubeB[2] + 2.0 * half
        tcp_offset_z = grasp_z - cubeA[2]

        above_stack = np.array([cubeB[0], cubeB[1], stack_z + tcp_offset_z + 0.08],
                               dtype=np.float64)
        if not self._move_to(above_stack, gripper=0.81, steps=self.move_steps):
            return False
        if self._early_stop():
            return False

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

        retreat = np.array([cubeB[0], cubeB[1], stack_z + tcp_offset_z + 0.12],
                           dtype=np.float64)
        self._move_to(retreat, gripper=0.0, steps=self.move_steps)
        if self._early_stop():
            return False

        for _ in range(8):
            if self._early_stop():
                break
            self._hold(0.0, self.settle_steps)
            if self._success():
                return True

        return self._success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
