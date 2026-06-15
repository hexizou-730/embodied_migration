"""Success-preserving adaptive xarm6_robotiq adapter for PullCube-v1.

This module keeps the seed-0 successful far-side strategy as the first attempt,
then adds bounded fallback contact offsets for multi-seed diagnosis. It is meant
for A/B testing against the saved seed-0 adapter, not as an LLM generation seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedXArm6PullCubeAdaptiveRobot(ManiSkillPullCubeRobot):
    """xarm6 adapter with far-side-first contact candidates."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=30,
            contact_steps=10,
            drag_steps=90,
            settle_steps=15,
            max_delta_m=0.04,
            contact_x_offset_m=0.12,
            contact_z_offset_m=0.015,
            gripper_open=1.0,
            gripper_close=-1.0,
        )

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "xarm6 PullCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("xarm6 adaptive adapter requires a Box-like action_space.")
        if shape[-1] != 4:
            raise RuntimeError(
                f"xarm6 adaptive adapter expects observed 4D action space, got shape {tuple(shape)!r}."
            )

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("xarm6 adaptive adapter requires a Box-like action_space.")
        action = np.zeros(shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[:3] = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]
        flat[3] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def _move_towards(self, target_pos: np.ndarray, *, gripper: float, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            tcp = self._tcp_pos()
            delta = np.asarray(target_pos, dtype=np.float32) - tcp
            if np.linalg.norm(delta) < 0.005:
                break
            command = np.clip(delta / self.max_delta_m, -0.9, 0.9)
            self._step(self._make_action(command, gripper=gripper))

    def _drag_pulse(self, direction: np.ndarray, *, magnitude: float, steps: int) -> None:
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return
        command = direction / norm * float(np.clip(magnitude, 0.1, 0.9))
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(self._make_action(command, gripper=self.gripper_close))

    def _attempt_contact(
        self,
        *,
        obj_name: str,
        target_name: str,
        x_offset: float,
        z_offset: float,
        drag_extra: float,
        stages: int,
        require_far_side: bool,
    ) -> bool:
        cube_pos = self._actor_pos("cube")
        goal_pos = self._region_pos(target_name)
        contact = cube_pos + np.array([x_offset, 0.0, z_offset], dtype=np.float32)
        pre_contact = contact + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        drag_end = np.array([goal_pos[0] - float(drag_extra), cube_pos[1], contact[2]], dtype=np.float32)

        self._move_towards(pre_contact, gripper=self.gripper_close, steps=self.move_steps)
        if self._early_stop():
            return False

        self._move_towards(contact, gripper=self.gripper_close, steps=self.move_steps)
        if self._early_stop():
            return False

        tcp = self._tcp_pos()
        if require_far_side and tcp[0] <= cube_pos[0] + 0.03:
            return False

        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.contact_steps)

        drag_dir = np.array([-1.0, 0.0, -0.15], dtype=np.float32)
        pulse_steps = max(4, self.drag_steps // max(1, stages * 3))
        max_pulses = max(10, stages * 3)
        prev_cube = self._actor_pos("cube")
        prev_goal_dist = float(np.linalg.norm((goal_pos - prev_cube)[:2]))

        for pulse_idx in range(max_pulses):
            if self._pull_cube_success():
                return self._log(
                    "pull",
                    {
                        "obj": obj_name,
                        "target": target_name,
                        "contact_x_offset": round(float(x_offset), 4),
                        "contact_z_offset": round(float(z_offset), 4),
                        "adaptive": True,
                    },
                    True,
                    True,
                    "",
                )
            self._drag_pulse(drag_dir, magnitude=0.6, steps=pulse_steps)
            cube_now = self._actor_pos("cube")
            goal_dist = float(np.linalg.norm((goal_pos - cube_now)[:2]))

            if cube_now[0] > prev_cube[0] + 0.006 or goal_dist > prev_goal_dist + 0.008:
                break
            if pulse_idx > 3 and abs(float(cube_now[0] - prev_cube[0])) < 0.002:
                self._drag_pulse(np.array([-1.0, 0.0, -0.25], dtype=np.float32), magnitude=0.78, steps=pulse_steps)
                cube_now = self._actor_pos("cube")
                goal_dist = float(np.linalg.norm((goal_pos - cube_now)[:2]))
                if cube_now[0] > prev_cube[0] + 0.006 or goal_dist > prev_goal_dist + 0.008:
                    break

            prev_cube = cube_now
            prev_goal_dist = goal_dist

        return self._pull_cube_success()

    def pull(self, obj, target, *, contact_x_offset=None, contact_z_offset=None, drag_extra=0.03, stages=5) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        if contact_x_offset is not None:
            x_offsets = (float(contact_x_offset),)
        else:
            # Keep the seed-0 successful far-side offsets first. Smaller offsets
            # are only fallback candidates because they broke seed 0.
            x_offsets = (0.14, 0.12, 0.10, 0.08, 0.06)
        if contact_z_offset is not None:
            z_offsets = (float(contact_z_offset),)
        else:
            z_offsets = (0.018, 0.015, 0.012, 0.008)

        attempts = 0
        for x_offset in x_offsets:
            for z_offset in z_offsets:
                if self._early_stop():
                    goal_pos = self._region_pos(target.name)
                    return self._fail(
                        "pull",
                        {"obj": obj.name, "target": target.name, "attempts": attempts},
                        f"episode ended during adaptive far-side contact search; {self._pull_diagnostics(goal_pos)}",
                    )
                attempts += 1
                ok = self._attempt_contact(
                    obj_name=obj.name,
                    target_name=target.name,
                    x_offset=float(x_offset),
                    z_offset=float(z_offset),
                    drag_extra=float(drag_extra),
                    stages=int(np.clip(stages, 1, 8)),
                    require_far_side=float(x_offset) >= 0.08,
                )
                if ok:
                    return True

                # Recover upward before the next bounded candidate, but do not
                # spend much budget chasing a bad contact pose.
                self._repeat_action(np.array([0.0, 0.0, 0.25], dtype=np.float32), gripper=self.gripper_close, steps=4)

        goal_pos = self._region_pos(target.name)
        ok = self._pull_cube_success()
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "adaptive": True, "attempts": attempts},
            ok,
            ok,
            "" if ok else f"adaptive far-side contact search failed; {self._pull_diagnostics(goal_pos)}",
        )


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6PullCubeAdaptiveRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6PullCubeAdaptiveRobot(env, robot_uid=robot_uid, control_mode=control_mode)
