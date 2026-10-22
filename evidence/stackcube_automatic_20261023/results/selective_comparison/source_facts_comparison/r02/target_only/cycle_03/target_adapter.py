from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # ---- helpers -------------------------------------------------------
    def _safe_move(self, target, gripper, steps):
        self._move_towards(np.asarray(target, dtype=np.float64),
                           gripper=float(gripper), steps=int(steps))
        return bool(self._early_stop())

    def _safe_repeat(self, delta, gripper, steps):
        self._repeat_action(np.asarray(delta, dtype=np.float64),
                            gripper=float(gripper), steps=int(steps))
        return bool(self._early_stop())

    def _entity_xyz(self, name):
        return np.array(self._entity_pos(name), dtype=np.float64).reshape(3)

    def _tcp_xyz(self):
        return np.array(self._tcp_pos(), dtype=np.float64).reshape(3)

    def _success(self):
        try:
            ev = self._official_evaluation()
            return bool(ev.get("success", False))
        except Exception:
            return False

    def _grasped(self):
        try:
            return bool(self._is_grasping_entity("cubeA"))
        except Exception:
            return False

    def _move_xy_then_z(self, target, gripper, steps_xy=40, steps_z=40):
        """Move in XY first at safe height, then descend in Z."""
        t = np.array(target, dtype=np.float64).reshape(3)
        cur = self._tcp_xyz()
        # XY move at current Z (or slightly above target Z if lower)
        mid_z = max(cur[2], t[2] + 0.02)
        mid = np.array([t[0], t[1], mid_z], dtype=np.float64)
        if self._safe_move(mid, gripper, steps_xy):
            return True
        if self._safe_move(t, gripper, steps_z):
            return True
        return False

    # ---- policy --------------------------------------------------------
    def run_source_actions(self) -> bool:
        open_g = 0.0
        close_g = 0.81

        # Measure initial positions
        posA = self._entity_xyz("cubeA")
        posB = self._entity_xyz("cubeB")

        # 1. Move above cubeA (open gripper) - approach from above
        approach_A = np.array([posA[0], posA[1], posA[2] + 0.12], dtype=np.float64)
        if self._safe_move(approach_A, open_g, 40):
            return False

        # 2. Descend to grasp height. Cube half size 0.02, so grasp center at posA.z + 0.02
        # Use a slightly lower target to ensure fingers straddle the cube
        grasp_A = np.array([posA[0], posA[1], posA[2] + 0.018], dtype=np.float64)
        if self._safe_move(grasp_A, open_g, 40):
            return False

        # 3. Close gripper to grasp
        if self._safe_repeat(np.zeros(3), close_g, 30):
            return False

        # 4. Verify grasp; if not grasped, try slight descent and re-close
        if not self._grasped():
            grasp_A2 = np.array([posA[0], posA[1], posA[2] + 0.010], dtype=np.float64)
            if self._safe_move(grasp_A2, open_g, 20):
                return False
            if self._safe_repeat(np.zeros(3), close_g, 30):
                return False

        # 5. Lift cubeA straight up
        lift_A = np.array([posA[0], posA[1], posA[2] + 0.18], dtype=np.float64)
        if self._safe_move(lift_A, close_g, 40):
            return False

        # Re-measure positions after lift
        posB = self._entity_xyz("cubeB")

        # 6. Move above cubeB (XY first, then descend)
        approach_B = np.array([posB[0], posB[1], posB[2] + 0.18], dtype=np.float64)
        if self._safe_move(approach_B, close_g, 60):
            return False

        # 7. Descend to place height: cubeB top + cubeA half height + clearance
        # cubeB top = posB.z + 0.02; cubeA bottom should rest at posB.z + 0.04
        # cubeA center at posB.z + 0.04 + 0.02 = posB.z + 0.06
        place_B = np.array([posB[0], posB[1], posB[2] + 0.055], dtype=np.float64)
        if self._safe_move(place_B, close_g, 40):
            return False

        # 8. Release gripper
        if self._safe_repeat(np.zeros(3), open_g, 30):
            return False

        # 9. Retreat upward to clear the stack
        retreat = np.array([posB[0], posB[1], posB[2] + 0.25], dtype=np.float64)
        if self._safe_move(retreat, open_g, 40):
            return False

        # 10. Settle and verify official success
        for _ in range(10):
            if self._safe_repeat(np.zeros(3), open_g, 10):
                return False
            if self._success():
                return True

        return self._success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
