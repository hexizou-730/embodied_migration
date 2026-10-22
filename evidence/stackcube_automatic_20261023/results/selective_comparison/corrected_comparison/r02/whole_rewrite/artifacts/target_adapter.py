from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # ---- helpers -------------------------------------------------------
    def _safe_move(self, target, gripper, steps, tol=0.006, max_iters=40):
        """Iteratively move toward target, checking early stop."""
        target = np.asarray(target, dtype=np.float64).reshape(3)
        for _ in range(max_iters):
            if self._early_stop():
                return False
            cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
            if np.linalg.norm(cur - target) < tol:
                return True
            self._move_towards(target, gripper=gripper, steps=steps)
        return True

    def _hold(self, gripper, steps):
        self._repeat_action(np.zeros(3), gripper=gripper, steps=steps)

    # ---- main policy ---------------------------------------------------
    def run_source_actions(self) -> bool:
        # Measured initial geometry
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        tcp = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)

        # Cube half-size estimate from resting height (cubeA sits on table at z ~ 0.02)
        half_z = max(float(cubeA[2]), 0.01)
        stack_height = 2.0 * half_z

        # ---- approach: pregrasp above cubeA ---------------------------
        pregrasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float64)
        self._safe_move(pregrasp, gripper=0.0, steps=self.move_steps)
        self._hold(0.0, self.settle_steps)

        # ---- grasp: descend to cubeA and close ------------------------
        grasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float64)
        self._safe_move(grasp, gripper=0.0, steps=self.move_steps)
        self._hold(0.0, self.settle_steps)

        # Close gripper in place
        self._hold(0.81, self.settle_steps)
        self._hold(0.81, self.settle_steps)

        # Verify grasp; if not grasped, retry a small descent
        if not self._is_grasping_entity("cubeA"):
            cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
            retry = np.array([cubeA[0], cubeA[1], cur[2] - 0.01], dtype=np.float64)
            self._safe_move(retry, gripper=0.81, steps=self.move_steps)
            self._hold(0.81, self.settle_steps)

        # ---- lift ------------------------------------------------------
        cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        lift = np.array([cur[0], cur[1], cur[2] + 0.12], dtype=np.float64)
        self._safe_move(lift, gripper=0.81, steps=self.move_steps)
        self._hold(0.81, self.settle_steps)

        # ---- align above cubeB ----------------------------------------
        # Re-measure cubeB (may have shifted) and current held cubeA offset
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        cubeA_now = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        tcp_now = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        # TCP-to-cubeA offset in world frame
        offset = cubeA_now - tcp_now
        # Desired cubeA position: directly above cubeB at stack height
        desired_cubeA = np.array([cubeB[0], cubeB[1], cubeB[2] + stack_height],
                                 dtype=np.float64)
        desired_tcp = desired_cubeA - offset
        # Approach in two steps: first XY, then Z, to keep grasp stable
        xy_target = np.array([desired_tcp[0], desired_tcp[1], tcp_now[2]],
                             dtype=np.float64)
        self._safe_move(xy_target, gripper=0.81, steps=self.move_steps)
        self._hold(0.81, self.settle_steps)

        # Re-measure and refine XY once more
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        cubeA_now = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        tcp_now = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        offset = cubeA_now - tcp_now
        desired_cubeA = np.array([cubeB[0], cubeB[1], cubeB[2] + stack_height],
                                 dtype=np.float64)
        desired_tcp = desired_cubeA - offset
        xy_target = np.array([desired_tcp[0], desired_tcp[1], tcp_now[2]],
                             dtype=np.float64)
        self._safe_move(xy_target, gripper=0.81, steps=self.move_steps)
        self._hold(0.81, self.settle_steps)

        # Descend to place cubeA on cubeB
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        cubeA_now = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        tcp_now = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        offset = cubeA_now - tcp_now
        desired_cubeA = np.array([cubeB[0], cubeB[1], cubeB[2] + stack_height],
                                 dtype=np.float64)
        desired_tcp = desired_cubeA - offset
        self._safe_move(desired_tcp, gripper=0.81, steps=self.move_steps)
        self._hold(0.81, self.settle_steps)

        # ---- release ---------------------------------------------------
        self._hold(0.0, self.settle_steps)
        self._hold(0.0, self.settle_steps)

        # Retreat upward to clear the stack
        cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
        retreat = np.array([cur[0], cur[1], cur[2] + 0.10], dtype=np.float64)
        self._safe_move(retreat, gripper=0.0, steps=self.move_steps)
        self._hold(0.0, self.settle_steps)

        # Allow settling for static check
        for _ in range(6):
            if self._early_stop():
                break
            self._hold(0.0, self.settle_steps)

        ev = self._official_evaluation()
        try:
            return bool(ev.get("success", False))
        except Exception:
            return False


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
