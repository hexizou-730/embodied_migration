from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot


class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # ---- helpers -------------------------------------------------------
    def _tcp(self):
        return np.asarray(self._tcp_pos(), dtype=np.float64).reshape(3)

    def _pos(self, name):
        return np.asarray(self._entity_pos(name), dtype=np.float64).reshape(3)

    def _goto(self, target, gripper, tol=0.006, max_iters=60, steps=None):
        target = np.asarray(target, dtype=np.float64).reshape(3)
        steps = steps if steps is not None else self.move_steps
        for _ in range(max_iters):
            if self._early_stop():
                return False
            cur = self._tcp()
            if np.linalg.norm(cur - target) <= tol:
                return True
            self._move_towards(target, gripper=gripper, steps=steps)
        return True

    def _hold(self, gripper, steps):
        for _ in range(steps):
            if self._early_stop():
                return False
            self._repeat_action(np.zeros(3), gripper=gripper, steps=1)
        return True

    def _pulse_and_check(self, target, gripper, steps=6):
        """Move a short pulse toward target and report object motion."""
        before = self._pos("cubeA").copy()
        self._move_towards(np.asarray(target, dtype=np.float64).reshape(3),
                           gripper=gripper, steps=steps)
        after = self._pos("cubeA")
        return float(np.linalg.norm(after - before))

    # ---- main policy ---------------------------------------------------
    def run_source_actions(self) -> bool:
        # Measure initial geometry
        cubeA = self._pos("cubeA")
        cubeB = self._pos("cubeB")
        tcp0 = self._tcp()

        # Stage 1: approach — hover above cubeA at safe height
        hover = cubeA.copy()
        hover[2] = cubeA[2] + 0.14
        if not self._goto(hover, gripper=self.gripper_open):
            return False

        # Stage 2: descend to grasp height (top-down, just above cube top)
        # Cube half size ~0.02; grasp TCP slightly above cube center so fingers close on sides.
        grasp_z = cubeA[2] + 0.012
        grasp_target = np.array([cubeA[0], cubeA[1], grasp_z], dtype=np.float64)
        if not self._goto(grasp_target, gripper=self.gripper_open, tol=0.004):
            return False

        # Verify we are actually near the cube in xy before closing
        cur = self._tcp()
        if np.linalg.norm(cur[:2] - cubeA[:2]) > 0.02:
            # Re-approach with a fresh measurement of cubeA
            cubeA = self._pos("cubeA")
            grasp_target = np.array([cubeA[0], cubeA[1], cubeA[2] + 0.012], dtype=np.float64)
            if not self._goto(grasp_target, gripper=self.gripper_open, tol=0.004):
                return False

        # Close gripper and settle
        if not self._hold(self.gripper_close, self.settle_steps):
            return False
        if not self._hold(self.gripper_close, self.settle_steps):
            return False

        # Confirm grasp; if not, try a small downward nudge and re-close
        if not bool(self._is_grasping_entity("cubeA")):
            cur = self._tcp()
            nudge = cur.copy()
            nudge[2] = cur[2] - 0.01
            self._move_towards(nudge, gripper=self.gripper_close, steps=8)
            if not self._hold(self.gripper_close, self.settle_steps):
                return False
            if not bool(self._is_grasping_entity("cubeA")):
                # Last resort: try closing again after slight lift
                lift_probe = cur.copy()
                lift_probe[2] = cur[2] + 0.02
                self._move_towards(lift_probe, gripper=self.gripper_close, steps=10)
                if not self._hold(self.gripper_close, self.settle_steps):
                    return False
                if not bool(self._is_grasping_entity("cubeA")):
                    return False

        # Stage 3: lift — raise while keeping gripper closed
        cubeA_now = self._pos("cubeA")
        lift_target = np.array([cubeA_now[0], cubeA_now[1], cubeA_now[2] + 0.12],
                               dtype=np.float64)
        if not self._goto(lift_target, gripper=self.gripper_close):
            return False
        if not self._hold(self.gripper_close, self.settle_steps):
            return False

        # Re-measure to compute stack alignment relative to cubeB
        cubeA_now = self._pos("cubeA")
        cubeB_now = self._pos("cubeB")
        tcp_now = self._tcp()

        # Stage 4: align — move above cubeB at stack height
        # Stack height: cubeB top + cubeA half size + clearance
        stack_z = cubeB_now[2] + 0.04 + 0.006
        # Preserve TCP-to-held-object xy offset so cubeA lands on cubeB
        offset_xy = tcp_now[:2] - cubeA_now[:2]
        align_target = np.array([cubeB_now[0] + offset_xy[0],
                                 cubeB_now[1] + offset_xy[1],
                                 stack_z], dtype=np.float64)
        if not self._goto(align_target, gripper=self.gripper_close):
            return False
        if not self._hold(self.gripper_close, self.settle_steps):
            return False

        # Fine-tune xy alignment using fresh measurements
        for _ in range(6):
            if self._early_stop():
                return False
            cubeA_now = self._pos("cubeA")
            cubeB_now = self._pos("cubeB")
            tcp_now = self._tcp()
            err_xy = cubeB_now[:2] - cubeA_now[:2]
            if np.linalg.norm(err_xy) < 0.004:
                break
            new_target = tcp_now.copy()
            new_target[:2] = new_target[:2] + err_xy
            new_target[2] = stack_z
            self._move_towards(new_target, gripper=self.gripper_close, steps=8)

        # Stage 5: release — open gripper and wait for static success
        if not self._hold(self.gripper_open, self.settle_steps):
            return False
        if not self._hold(self.gripper_open, self.settle_steps):
            return False

        # Allow physics to settle; check official success in loop
        for _ in range(80):
            if self._early_stop():
                break
            ev = self._official_evaluation()
            if bool(ev.get("success", False)):
                return True
            self._repeat_action(np.zeros(3), gripper=self.gripper_open, steps=2)

        ev = self._official_evaluation()
        return bool(ev.get("success", False))


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
