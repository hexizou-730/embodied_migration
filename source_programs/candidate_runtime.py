"""Conservative bootstrap policies for tasks without packaged reference solvers.

These policies provide executable, state-dependent starting points. They are
not frozen evidence and may be replaced by source synthesis after real trials.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot, _to_numpy


class BootstrapSourceRobot(ManiSkillDynamicRobot):
    POLICY = ""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            control_mode=control_mode,
            robot_uid=robot_uid,
            move_steps=40,
            settle_steps=16,
            max_delta_m=0.035,
        )

    def solve_task(self) -> bool:
        self._snapshot()
        if self.POLICY == "roll_ball":
            return self._push_to_goal("ball", "goal_region", clearance=0.075)
        if self.POLICY == "poke_cube":
            return self._push_to_goal("cube", "goal_region", clearance=0.055)
        if self.POLICY == "push_t":
            return self._push_to_goal("tee", "goal_region", clearance=0.08)
        if self.POLICY == "turn_faucet":
            return self._operate_handle("target_switch_link", motion=[0.0, 0.55, 0.15])
        if self.POLICY == "open_drawer":
            return self._operate_handle("handle_link", motion=[-0.65, 0.0, 0.0])
        if self.POLICY == "open_door":
            return self._operate_handle("handle_link", motion=[-0.45, 0.45, 0.0])
        if self.POLICY == "pick_ycb":
            return self._pick_place("obj", "goal_site")
        if self.POLICY == "assembling_kits":
            return self._pick_place("obj", "goal_pos")
        if self.POLICY == "fmb_assembly":
            return self._pick_place("bridge", "goal_bridge_pose")
        return self._fail("solve_task", {}, f"unknown bootstrap policy {self.POLICY!r}")

    def _push_to_goal(self, object_name: str, goal_name: str, *, clearance: float) -> bool:
        obj = self._position(object_name)
        goal = self._position(goal_name)
        planar = goal[:2] - obj[:2]
        distance = float(np.linalg.norm(planar))
        if distance < 1e-6:
            return self._settle_and_check("push")
        direction = planar / distance
        contact = obj.copy()
        contact[:2] -= direction * clearance
        contact[2] += 0.012
        approach = contact + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        self._move_towards(approach, gripper=self.gripper_close, steps=self.move_steps)
        self._move_towards(contact, gripper=self.gripper_close, steps=self.move_steps)
        stages = max(3, min(10, int(distance / 0.035) + 1))
        for stage in range(1, stages + 1):
            if self._early_stop():
                break
            waypoint = contact.copy()
            waypoint[:2] += direction * (distance * stage / stages)
            self._move_towards(waypoint, gripper=self.gripper_close, steps=20)
            if self._official_success():
                return self._log("solve_task", {"policy": self.POLICY}, True, True, "")
        return self._settle_and_check("push")

    def _pick_place(self, object_name: str, goal_name: str) -> bool:
        obj = self._position(object_name)
        goal = self._position(goal_name)
        approach = obj + np.array([0.0, 0.0, 0.10], dtype=np.float32)
        grasp = obj + np.array([0.0, 0.0, 0.01], dtype=np.float32)
        self._move_towards(approach, gripper=self.gripper_open, steps=self.move_steps)
        self._move_towards(grasp, gripper=self.gripper_open, steps=self.move_steps)
        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=18)
        if not self._is_grasping_entity(object_name):
            return self._settle_and_check("grasp")
        lift = self._tcp_pos() + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        self._move_towards(lift, gripper=self.gripper_close, steps=self.move_steps)
        place = goal + np.array([0.0, 0.0, 0.06], dtype=np.float32)
        self._move_towards(place, gripper=self.gripper_close, steps=self.move_steps * 2)
        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_open, steps=12)
        return self._settle_and_check("place")

    def _operate_handle(self, handle_name: str, *, motion: list[float]) -> bool:
        handle = self._position(handle_name)
        approach = handle + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        self._move_towards(approach, gripper=self.gripper_open, steps=self.move_steps * 2)
        self._move_towards(handle, gripper=self.gripper_open, steps=self.move_steps)
        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=18)
        command = np.asarray(motion, dtype=np.float32)
        for _ in range(80):
            if self._early_stop() or self._official_success():
                break
            self._step(self._make_action(command, gripper=self.gripper_close))
        return self._settle_and_check("handle")

    def _position(self, name: str) -> np.ndarray:
        try:
            value = self._entity(name)
        except Exception:
            value = getattr(self._base_env(), name)
        pose = getattr(value, "pose", value)
        position = getattr(pose, "p", pose)
        array = np.asarray(_to_numpy(position), dtype=np.float32)
        return array.reshape(-1, array.shape[-1])[0, :3].copy()

    def _settle_and_check(self, phase: str) -> bool:
        self._repeat_action(
            np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps
        )
        ok = self._official_success()
        return self._log(
            "solve_task",
            {"policy": self.POLICY, "phase": phase},
            ok,
            ok,
            "" if ok else f"bootstrap {self.POLICY} did not reach official success",
        )

