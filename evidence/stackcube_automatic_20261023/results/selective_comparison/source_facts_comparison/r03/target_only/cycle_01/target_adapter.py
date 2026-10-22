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
        """Move towards target, checking early stop between chunks."""
        target = np.asarray(target, dtype=np.float64).reshape(3)
        remaining = int(steps)
        chunk = 8
        while remaining > 0:
            if self._early_stop():
                return True
            n = min(chunk, remaining)
            self._move_towards(target, gripper=gripper, steps=n)
            remaining -= n
        return self._early_stop()

    def _safe_repeat(self, delta, gripper, steps):
        delta = np.asarray(delta, dtype=np.float64).reshape(3)
        remaining = int(steps)
        chunk = 8
        while remaining > 0:
            if self._early_stop():
                return True
            n = min(chunk, remaining)
            self._repeat_action(delta, gripper=gripper, steps=n)
            remaining -= n
        return self._early_stop()

    def _grasp_pose(self, cube_pos):
        """TCP target for grasping: above cube, aligned in xy."""
        cube_pos = np.asarray(cube_pos, dtype=np.float64).reshape(3)
        return np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.10], dtype=np.float64)

    def _lift_pose(self, cube_pos):
        cube_pos = np.asarray(cube_pos, dtype=np.float64).reshape(3)
        return np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.20], dtype=np.float64)

    def _place_pose(self, cubeB_pos):
        cubeB_pos = np.asarray(cubeB_pos, dtype=np.float64).reshape(3)
        # place cubeA on top of cubeB (cube half size ~0.02 -> stack height ~0.04)
        return np.array([cubeB_pos[0], cubeB_pos[1], cubeB_pos[2] + 0.10], dtype=np.float64)

    # ---- main policy ---------------------------------------------------
    def run_source_actions(self) -> bool:
        # 1. Open gripper fully and settle.
        if self._safe_repeat(np.zeros(3), gripper=0.0, steps=12):
            return True

        # 2. Move above cubeA.
        cubeA = self._entity_pos("cubeA")
        if cubeA is None:
            return False
        cubeA = np.asarray(cubeA, dtype=np.float64).reshape(3)
        above_A = self._grasp_pose(cubeA)
        if self._safe_move(above_A, gripper=0.0, steps=40):
            return True

        # 3. Descend to grasp height.
        grasp_A = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.005], dtype=np.float64)
        if self._safe_move(grasp_A, gripper=0.0, steps=40):
            return True

        # 4. Close gripper.
        if self._safe_repeat(np.zeros(3), gripper=0.81, steps=20):
            return True

        # 5. Lift cubeA.
        if self._safe_move(above_A, gripper=0.81, steps=40):
            return True

        # 6. Move above cubeB.
        cubeB = self._entity_pos("cubeB")
        if cubeB is None:
            return False
        cubeB = np.asarray(cubeB, dtype=np.float64).reshape(3)
        above_B = self._place_pose(cubeB)
        if self._safe_move(above_B, gripper=0.81, steps=60):
            return True

        # 7. Descend to place height (cubeA bottom just above cubeB top).
        place_B = np.array([cubeB[0], cubeB[1], cubeB[2] + 0.045], dtype=np.float64)
        if self._safe_move(place_B, gripper=0.81, steps=40):
            return True

        # 8. Open gripper to release.
        if self._safe_repeat(np.zeros(3), gripper=0.0, steps=20):
            return True

        # 9. Retreat upward to leave cubeA static and ungrasped.
        retreat = np.array([cubeB[0], cubeB[1], cubeB[2] + 0.20], dtype=np.float64)
        if self._safe_move(retreat, gripper=0.0, steps=40):
            return True

        # 10. Settle and verify.
        self._safe_repeat(np.zeros(3), gripper=0.0, steps=20)

        ev = self._official_evaluation()
        try:
            return bool(ev.get("success", False))
        except Exception:
            return False


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
