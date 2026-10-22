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
    def _safe_move(self, target, gripper, steps, tol=0.008, max_iters=40):
        """Iteratively move towards a world target, checking early stop."""
        target = np.asarray(target, dtype=np.float64).reshape(3)
        for _ in range(max_iters):
            if self._early_stop():
                return False
            cur = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)
            if np.linalg.norm(cur - target) <= tol:
                return True
            self._move_towards(target, gripper=gripper, steps=steps)
        return True

    def _hold_gripper(self, value, steps):
        for _ in range(max(1, steps // 6)):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3), gripper=value, steps=6)
        return True

    # ------------------------------------------------------------------ #
    # main policy
    # ------------------------------------------------------------------ #
    def run_source_actions(self) -> bool:
        # ---- measured initial geometry -------------------------------- #
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        tcp0 = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)

        # Cube half-size estimate from resting height (cubeA sits on table).
        half_z = max(0.005, float(cubeA[2]))
        stack_height = 2.0 * half_z

        # ---- approach: pregrasp above cubeA --------------------------- #
        pregrasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.10], dtype=np.float64)
        # First lift to a safe transit height above both cubes.
        transit_z = max(tcp0[2], cubeA[2], cubeB[2]) + 0.15
        transit = np.array([tcp0[0], tcp0[1], transit_z], dtype=np.float64)
        self._safe_move(transit, gripper=self.gripper_open, steps=self.move_steps)
        self._safe_move(pregrasp, gripper=self.gripper_open, steps=self.move_steps)

        # ---- grasp ---------------------------------------------------- #
        grasp = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float64)
        self._safe_move(grasp, gripper=self.gripper_open, steps=self.move_steps,
                        tol=0.006)
        self._hold_gripper(self.gripper_close, steps=self.settle_steps)
        # small settle to let fingers close
        self._hold_gripper(self.gripper_close, steps=self.settle_steps)

        if not self._is_grasping_entity("cubeA"):
            # retry once with a slightly deeper grasp
            self._safe_move(grasp, gripper=self.gripper_open, steps=self.move_steps,
                            tol=0.005)
            self._hold_gripper(self.gripper_close, steps=self.settle_steps)
            self._hold_gripper(self.gripper_close, steps=self.settle_steps)

        # ---- lift ----------------------------------------------------- #
        lift = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.12], dtype=np.float64)
        self._safe_move(lift, gripper=self.gripper_close, steps=self.move_steps)

        # ---- align above cubeB ---------------------------------------- #
        # Re-measure cubeA (it moved with the gripper) and cubeB.
        cubeA_now = np.asarray(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        cubeB_now = np.asarray(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)
        tcp_now = np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)

        # TCP-to-held-cube offset in world frame (preserve grasp geometry).
        offset = cubeA_now - tcp_now

        # Desired cubeA position: directly above cubeB at stack height.
        target_cubeA = np.array([cubeB_now[0], cubeB_now[1],
                                 cubeB_now[2] + stack_height], dtype=np.float64)
        target_tcp = target_cubeA - offset

        # Move to a safe height above the target first, then descend.
        above_target = np.array([target_tcp[0], target_tcp[1],
                                 max(target_tcp[2], tcp_now[2]) + 0.08],
                                dtype=np.float64)
        self._safe_move(above_target, gripper=self.gripper_close, steps=self.move_steps)
        self._safe_move(target_tcp, gripper=self.gripper_close, steps=self.move_steps,
                        tol=0.006)

        # ---- release -------------------------------------------------- #
        self._hold_gripper(self.gripper_open, steps=self.settle_steps)
        # retreat upward to clear the stack
        retreat = np.array([target_tcp[0], target_tcp[1],
                            target_tcp[2] + 0.12], dtype=np.float64)
        self._safe_move(retreat, gripper=self.gripper_open, steps=self.move_steps)

        # ---- wait for official static + released success -------------- #
        for _ in range(60):
            if self._early_stop():
                break
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True
            self._repeat_action(np.zeros(3), gripper=self.gripper_open, steps=6)

        ev = self._official_evaluation()
        return bool(ev.get("success", False))


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
