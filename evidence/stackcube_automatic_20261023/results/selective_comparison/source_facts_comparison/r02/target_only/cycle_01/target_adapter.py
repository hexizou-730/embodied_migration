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
        """Move towards target, checking early stop."""
        self._move_towards(np.asarray(target, dtype=np.float64),
                           gripper=float(gripper), steps=int(steps))
        return bool(self._early_stop())

    def _safe_repeat(self, delta, gripper, steps):
        self._repeat_action(np.asarray(delta, dtype=np.float64),
                            gripper=float(gripper), steps=int(steps))
        return bool(self._early_stop())

    def _above(self, pos, dz):
        p = np.array(pos, dtype=np.float64).reshape(3)
        return np.array([p[0], p[1], p[2] + dz], dtype=np.float64)

    # ---- policy --------------------------------------------------------
    def run_source_actions(self) -> bool:
        # Target-only baseline: grasp cubeA, lift, place on cubeB, release.
        open_g = 0.0
        close_g = 0.81

        # 1. Approach above cubeA
        posA = np.array(self._entity_pos("cubeA"), dtype=np.float64).reshape(3)
        posB = np.array(self._entity_pos("cubeB"), dtype=np.float64).reshape(3)

        approach_A = self._above(posA, 0.12)
        if self._safe_move(approach_A, open_g, 40):
            return False

        # 2. Descend to grasp height above cubeA
        grasp_A = self._above(posA, 0.015)
        if self._safe_move(grasp_A, open_g, 40):
            return False

        # 3. Close gripper (stationary)
        if self._safe_repeat(np.zeros(3), close_g, 20):
            return False

        # 4. Lift cubeA
        lift_A = self._above(posA, 0.15)
        if self._safe_move(lift_A, close_g, 40):
            return False

        # 5. Move above cubeB
        approach_B = self._above(posB, 0.15)
        if self._safe_move(approach_B, close_g, 60):
            return False

        # 6. Descend to place height (cubeB top + cubeA half height)
        place_B = self._above(posB, 0.045)
        if self._safe_move(place_B, close_g, 40):
            return False

        # 7. Release gripper (stationary)
        if self._safe_repeat(np.zeros(3), open_g, 20):
            return False

        # 8. Retreat upward to clear the stack
        retreat = self._above(posB, 0.20)
        if self._safe_move(retreat, open_g, 40):
            return False

        # 9. Settle and verify official success
        for _ in range(6):
            if self._safe_repeat(np.zeros(3), open_g, 10):
                return False
            ev = self._official_evaluation()
            try:
                if bool(ev.get("success", False)):
                    return True
            except Exception:
                pass

        ev = self._official_evaluation()
        try:
            return bool(ev.get("success", False))
        except Exception:
            return False


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
