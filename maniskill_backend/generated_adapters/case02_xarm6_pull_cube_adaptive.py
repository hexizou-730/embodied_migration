"""Adaptive xarm6_robotiq target adapter for PullCube-v1.

This module is an experimental follow-up to the seed-0 successful adapter. The
multi-seed diagnostics showed that a fixed far-side contact point can be
unreachable for many seeds. This adapter tries a small bounded set of contact
offsets and rejects candidates whose approach/descent error stops improving.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedXArm6PullCubeAdaptiveRobot(ManiSkillPullCubeRobot):
    """xarm6 adapter with adaptive contact-offset search."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=34,
            contact_steps=10,
            drag_steps=96,
            settle_steps=12,
            max_delta_m=0.045,
            contact_x_offset_m=0.06,
            contact_z_offset_m=0.014,
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

    def _move_towards(
        self,
        target_pos: np.ndarray,
        *,
        gripper: float,
        steps: int,
    ) -> None:
        self._move_towards_checked(target_pos, gripper=gripper, steps=steps)

    def _move_towards_checked(
        self,
        target_pos: np.ndarray,
        *,
        gripper: float,
        steps: int,
        tolerance: float = 0.01,
        command_clip: float = 0.85,
        down_bias: float | None = None,
        stall_window: int = 18,
    ) -> dict[str, Any]:
        target = np.asarray(target_pos, dtype=np.float32)
        best_error = float("inf")
        stale_steps = 0
        final_error = float("inf")
        for step_idx in range(max(1, steps)):
            if self._early_stop():
                break
            tcp = self._tcp_pos()
            delta = target - tcp
            if down_bias is not None:
                delta[2] = min(float(delta[2]), float(down_bias))
            error = float(np.linalg.norm(target - tcp))
            final_error = error
            if error < tolerance:
                return {
                    "ok": True,
                    "error": round(error, 5),
                    "steps": step_idx,
                    "target": np.round(target, 5).tolist(),
                    "tcp": np.round(tcp, 5).tolist(),
                }
            if error < best_error - 0.002:
                best_error = error
                stale_steps = 0
            else:
                stale_steps += 1
            if stale_steps >= stall_window and error > max(tolerance * 2.0, best_error + 0.01):
                break
            command = np.clip(delta / self.max_delta_m, -command_clip, command_clip)
            self._step(self._make_action(command, gripper=gripper))
        tcp = self._tcp_pos()
        return {
            "ok": bool(final_error < tolerance),
            "error": round(float(np.linalg.norm(target - tcp)), 5),
            "best_error": round(float(best_error), 5) if np.isfinite(best_error) else None,
            "target": np.round(target, 5).tolist(),
            "tcp": np.round(tcp, 5).tolist(),
        }

    def _drag_pulse(self, *, steps: int, x_command: float = -0.55, down_command: float = -0.04) -> None:
        action = self._make_action(
            np.array([x_command, 0.0, down_command], dtype=np.float32),
            gripper=self.gripper_close,
        )
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(action)

    def _candidate_summary(
        self,
        *,
        x_offset: float,
        z_offset: float,
        phase: str,
        target_pos: np.ndarray,
    ) -> dict[str, Any]:
        tcp = self._tcp_pos()
        cube = self._actor_pos("cube")
        goal = self._region_pos("goal")
        return {
            "x_offset": round(float(x_offset), 4),
            "z_offset": round(float(z_offset), 4),
            "phase": phase,
            "target": np.round(target_pos, 4).tolist(),
            "tcp": np.round(tcp, 4).tolist(),
            "cube": np.round(cube, 4).tolist(),
            "tcp_target_error": round(float(np.linalg.norm(target_pos - tcp)), 4),
            "tcp_cube_xy": round(float(np.linalg.norm((tcp - cube)[:2])), 4),
            "cube_goal_xy": round(float(np.linalg.norm((goal - cube)[:2])), 4),
        }

    def pull(
        self,
        obj,
        target,
        *,
        contact_x_offset=None,
        contact_z_offset=None,
        drag_extra=0.025,
        stages=5,
    ) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        if contact_x_offset is None:
            x_offsets = (0.04, 0.055, 0.07, 0.09, 0.12)
        else:
            x_offsets = (float(contact_x_offset),)
        if contact_z_offset is None:
            z_offsets = (0.012, 0.016)
        else:
            z_offsets = (float(contact_z_offset),)

        stages = int(np.clip(stages, 1, 8))
        goal_pos = self._region_pos(target.name)
        best: dict[str, Any] | None = None

        for x_offset in x_offsets:
            for z_offset in z_offsets:
                if self._early_stop():
                    return self._fail(
                        "pull",
                        {"obj": obj.name, "target": target.name, "best": best},
                        "episode ended before trying remaining adaptive contact candidates",
                    )

                cube_pos = self._actor_pos("cube")
                contact = cube_pos + np.array([float(x_offset), 0.0, float(z_offset)], dtype=np.float32)
                pre_contact = contact + np.array([0.0, 0.0, 0.075], dtype=np.float32)
                drag_end = np.array(
                    [goal_pos[0] - float(drag_extra), cube_pos[1], contact[2]],
                    dtype=np.float32,
                )

                approach = self._move_towards_checked(
                    pre_contact,
                    gripper=self.gripper_close,
                    steps=self.move_steps,
                    tolerance=0.035,
                    command_clip=0.9,
                )
                best = self._candidate_summary(
                    x_offset=float(x_offset),
                    z_offset=float(z_offset),
                    phase="approach",
                    target_pos=pre_contact,
                )
                best["approach"] = approach
                if approach["error"] > 0.09:
                    continue

                descent = self._move_towards_checked(
                    contact,
                    gripper=self.gripper_close,
                    steps=self.move_steps + 10,
                    tolerance=0.035,
                    command_clip=0.75,
                )
                best = self._candidate_summary(
                    x_offset=float(x_offset),
                    z_offset=float(z_offset),
                    phase="descent",
                    target_pos=contact,
                )
                best["descent"] = descent
                if descent["error"] > 0.075:
                    self._repeat_action(np.array([0.0, 0.0, 0.25], dtype=np.float32), gripper=self.gripper_close, steps=6)
                    continue

                self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.contact_steps)
                self._drag_pulse(steps=8, x_command=-0.18, down_command=-0.05)

                drag_start = self._tcp_pos()
                previous_cube = self._actor_pos("cube")
                previous_goal_dist = float(np.linalg.norm((goal_pos - previous_cube)[:2]))
                for stage in range(1, stages + 1):
                    alpha = stage / stages
                    waypoint = drag_start * (1.0 - alpha) + drag_end * alpha
                    self._move_towards_checked(
                        waypoint,
                        gripper=self.gripper_close,
                        steps=max(1, self.drag_steps // stages),
                        tolerance=0.04,
                        command_clip=0.8,
                        down_bias=-0.004,
                    )
                    self._drag_pulse(steps=5, x_command=-0.6, down_command=-0.04)

                    if self._pull_cube_success():
                        return self._log(
                            "pull",
                            {
                                "obj": obj.name,
                                "target": target.name,
                                "contact_x_offset": round(float(x_offset), 4),
                                "contact_z_offset": round(float(z_offset), 4),
                                "stages": stages,
                                "adaptive": True,
                            },
                            True,
                            True,
                            "",
                        )

                    cube_now = self._actor_pos("cube")
                    goal_dist = float(np.linalg.norm((goal_pos - cube_now)[:2]))
                    if cube_now[0] > previous_cube[0] + 0.025 or goal_dist > previous_goal_dist + 0.03:
                        break
                    previous_cube = cube_now
                    previous_goal_dist = goal_dist

                if self._pull_cube_success():
                    return self._log(
                        "pull",
                        {
                            "obj": obj.name,
                            "target": target.name,
                            "contact_x_offset": round(float(x_offset), 4),
                            "contact_z_offset": round(float(z_offset), 4),
                            "stages": stages,
                            "adaptive": True,
                        },
                        True,
                        True,
                        "",
                    )

                self._repeat_action(np.array([0.0, 0.0, 0.25], dtype=np.float32), gripper=self.gripper_close, steps=6)

        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)
        ok = self._pull_cube_success()
        message = "" if ok else f"adaptive contact search failed; best={best}; {self._pull_diagnostics(goal_pos)}"
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "adaptive": True},
            ok,
            ok,
            message,
        )


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6PullCubeAdaptiveRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6PullCubeAdaptiveRobot(env, robot_uid=robot_uid, control_mode=control_mode)
