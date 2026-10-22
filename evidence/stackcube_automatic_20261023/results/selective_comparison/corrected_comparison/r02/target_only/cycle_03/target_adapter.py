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

    def _grasped(self, name):
        try:
            return bool(self._is_grasping_entity(name))
        except Exception:
            return False

    def _pulse(self, gripper, steps=6):
        return self._safe_repeat(np.zeros(3), gripper=gripper, steps=steps)

    def _descend_to(self, target_xyz, gripper, max_iters=8, tol=0.006):
        """Iteratively move toward target, re-reading TCP each time."""
        for _ in range(max_iters):
            if self._early_stop():
                return True
            tcp = self._tcp_xyz()
            err = np.asarray(target_xyz, dtype=np.float64) - tcp
            if float(np.linalg.norm(err)) < tol:
                return False
            if self._safe_move(target_xyz, gripper=gripper, steps=20):
                return True
        return False

    def _grasp_attempt(self, cube_name, grasp_dz):
        """Approach, descend, close, lift, verify. Returns (ok, cube_pos_before)."""
        cube = self._entity_xyz(cube_name)
        pre = self._above(cube, 0.12)
        if self._safe_move(pre, gripper=0.0, steps=40):
            return True, cube
        # Re-read after approach
        cube = self._entity_xyz(cube_name)
        grasp = self._above(cube, grasp_dz)
        if self._descend_to(grasp, gripper=0.0):
            return True, cube
        # Close gripper firmly
        if self._pulse(0.81, steps=20):
            return True, cube
        # Lift and verify
        cube = self._entity_xyz(cube_name)
        lift = self._above(cube, 0.08)
        if self._safe_move(lift, gripper=0.81, steps=25):
            return True, cube
        moved = float(np.linalg.norm(self._entity_xyz(cube_name) - cube))
        ok = moved > 0.01 or self._grasped(cube_name)
        return ok, cube

    # ---- policy --------------------------------------------------------
    def run_source_actions(self) -> bool:
        # Open gripper and settle.
        if self._pulse(0.0, steps=8):
            return self._success()

        # Attempt grasp with a couple of height offsets.
        grasp_offsets = [0.0, -0.008, 0.008]
        grasped = False
        for dz in grasp_offsets:
            ok, _ = self._grasp_attempt("cubeA", dz)
            if ok:
                grasped = True
                break
            # If failed, retreat and try next offset
            cube = self._entity_xyz("cubeA")
            if self._safe_move(self._above(cube, 0.12), gripper=0.0, steps=25):
                return self._success()

        if not grasped:
            # Last resort: try once more with a deeper grasp
            ok, _ = self._grasp_attempt("cubeA", -0.012)
            grasped = ok

        # Lift to transport height regardless (best effort).
        cubeA = self._entity_xyz("cubeA")
        lift = self._above(cubeA, 0.20)
        if self._safe_move(lift, gripper=0.81, steps=40):
            return self._success()

        # Move above cubeB.
        cubeB = self._entity_xyz("cubeB")
        over_B = self._above(cubeB, 0.20)
        if self._safe_move(over_B, gripper=0.81, steps=60):
            return self._success()

        # Align XY precisely over cubeB, then descend to stack height.
        # cubeB top ~ 0.04; cubeA center should sit ~0.06 above ground.
        stack = np.array([cubeB[0], cubeB[1], cubeB[2] + 0.045], dtype=np.float64)
        if self._descend_to(stack, gripper=0.81, max_iters=10, tol=0.005):
            return self._success()

        # Small settle before release.
        if self._pulse(0.81, steps=10):
            return self._success()

        # Release.
        if self._pulse(0.0, steps=20):
            return self._success()

        # Retreat upward to let cube settle.
        retreat = self._above(cubeB, 0.22)
        if self._safe_move(retreat, gripper=0.0, steps=40):
            return self._success()

        # Settle and verify.
        for _ in range(10):
            if self._pulse(0.0, steps=10):
                return self._success()
            if self._success():
                return True

        return self._success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
