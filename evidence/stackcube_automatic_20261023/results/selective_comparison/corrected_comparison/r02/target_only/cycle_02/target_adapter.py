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
        return self._early_stop()

    def _safe_repeat(self, delta, gripper, steps):
        self._repeat_action(np.asarray(delta, dtype=np.float64),
                            gripper=float(gripper), steps=int(steps))
        return self._early_stop()

    def _above(self, pos, dz):
        p = np.array(pos, dtype=np.float64).reshape(3)
        p[2] += float(dz)
        return p

    def _entity_xy(self, name):
        p = np.array(self._entity_pos(name), dtype=np.float64).reshape(3)
        return p

    def _tcp_xy(self):
        return np.array(self._tcp_pos(), dtype=np.float64).reshape(3)

    def _pulse_and_check(self, name, gripper, steps=6):
        """Small action pulse; return displacement of entity."""
        before = self._entity_xy(name)
        self._safe_repeat(np.zeros(3), gripper=gripper, steps=steps)
        after = self._entity_xy(name)
        return float(np.linalg.norm(after - before))

    # ---- policy --------------------------------------------------------
    def run_source_actions(self) -> bool:
        # Open gripper and settle.
        if self._safe_repeat(np.zeros(3), gripper=0.0, steps=8):
            return True

        cubeA = self._entity_xy("cubeA")
        cubeB = self._entity_xy("cubeB")

        # Approach cubeA from above with a taller pre-grasp clearance.
        pre = self._above(cubeA, 0.16)
        if self._safe_move(pre, gripper=0.0, steps=40):
            return True

        # Re-read cubeA in case it moved (it shouldn't, but be safe).
        cubeA = self._entity_xy("cubeA")

        # Descend to a grasp height that puts the fingers around the cube.
        # Cube half-size is 0.02; grasp center at cube top + small offset.
        grasp = self._above(cubeA, 0.005)
        if self._safe_move(grasp, gripper=0.0, steps=40):
            return True

        # Close gripper to grasp.
        if self._safe_repeat(np.zeros(3), gripper=0.81, steps=20):
            return True

        # Verify grasp by lifting a small amount and checking cube motion.
        lift_small = self._above(cubeA, 0.06)
        if self._safe_move(lift_small, gripper=0.81, steps=25):
            return True

        moved = float(np.linalg.norm(self._entity_xy("cubeA") - cubeA))
        if moved < 0.005:
            # Grasp failed; retry with a slightly lower grasp and re-close.
            cubeA = self._entity_xy("cubeA")
            retry_grasp = self._above(cubeA, -0.005)
            if self._safe_move(retry_grasp, gripper=0.0, steps=30):
                return True
            if self._safe_repeat(np.zeros(3), gripper=0.81, steps=20):
                return True
            if self._safe_move(self._above(cubeA, 0.06), gripper=0.81, steps=25):
                return True
            moved = float(np.linalg.norm(self._entity_xy("cubeA") - cubeA))
            if moved < 0.003:
                # Give up on grasp; still attempt a place to avoid dead-end.
                pass

        # Lift cubeA to a safe transport height.
        cubeA = self._entity_xy("cubeA")
        lift = self._above(cubeA, 0.18)
        if self._safe_move(lift, gripper=0.81, steps=40):
            return True

        # Move above cubeB.
        cubeB = self._entity_xy("cubeB")
        over_B = self._above(cubeB, 0.18)
        if self._safe_move(over_B, gripper=0.81, steps=60):
            return True

        # Lower onto cubeB: target stack height = cubeB top + cubeA half-size.
        # cubeB top ~ 0.04; cubeA center should sit at ~0.06 above ground.
        stack = self._above(cubeB, 0.045)
        if self._safe_move(stack, gripper=0.81, steps=40):
            return True

        # Release.
        if self._safe_repeat(np.zeros(3), gripper=0.0, steps=20):
            return True

        # Retreat upward to let cube settle.
        retreat = self._above(cubeB, 0.22)
        if self._safe_move(retreat, gripper=0.0, steps=40):
            return True

        # Settle and verify.
        for _ in range(8):
            if self._safe_repeat(np.zeros(3), gripper=0.0, steps=10):
                return True
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True

        return bool(self._official_evaluation().get("success", False))


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
