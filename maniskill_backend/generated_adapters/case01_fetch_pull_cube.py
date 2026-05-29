"""Hand-written oracle adapter for Fetch PullCube migration.

This module is intentionally small and auditable. It establishes a deterministic
upper-bound route for the current case:

1. map Fetch's 9D action layout;
2. use a short positive base-forward pulse to bring the TCP closer;
3. stop the base;
4. use arm-only contact and drag.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot, SkillTarget


class GeneratedFetchPullCubeRobot(ManiSkillPullCubeRobot):
    """Minimal Fetch oracle adapter for PullCube-v1."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=30,
            contact_steps=12,
            drag_steps=90,
            settle_steps=12,
            max_delta_m=0.05,
            contact_x_offset_m=0.025,
            contact_z_offset_m=0.008,
        )

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "Fetch PullCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("Fetch PullCube adapter requires a Box-like action_space.")
        if shape[-1] not in (4, 7, 9):
            raise RuntimeError(
                "Fetch PullCube adapter expects action_space last dim in {4, 7, 9}, "
                f"got shape {tuple(shape)!r}."
            )

    def _make_action(
        self,
        delta_xyz: np.ndarray,
        *,
        gripper: float,
        base: Optional[np.ndarray] = None,
    ) -> Any:
        space = self.env.action_space
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("Fetch PullCube adapter requires a Box-like action_space.")

        action = np.zeros(shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        delta = np.asarray(delta_xyz, dtype=np.float32).reshape(-1)[:3]

        if flat.size == 9:
            flat[0:3] = delta
            flat[3] = float(gripper)
            if base is not None:
                flat[7:9] = np.asarray(base, dtype=np.float32).reshape(-1)[:2]
        else:
            flat[: min(3, flat.size)] = delta[: min(3, flat.size)]
            if flat.size >= 4:
                flat[-1] = float(gripper)

        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def pull(
        self,
        obj: SkillTarget,
        target: SkillTarget,
        *,
        contact_x_offset: Optional[float] = None,
        contact_z_offset: Optional[float] = None,
        drag_extra: float = 0.04,
        stages: int = 6,
    ) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        self._drive_base_forward_guarded()
        self._stop_base(steps=6)

        offsets = (
            (
                self.contact_x_offset_m if contact_x_offset is None else float(contact_x_offset),
                self.contact_z_offset_m if contact_z_offset is None else float(contact_z_offset),
            ),
            (0.015, 0.006),
            (0.035, 0.006),
        )

        for x_offset, z_offset in offsets:
            if self._try_contact_drag(
                target,
                x_offset=float(np.clip(x_offset, 0.005, 0.06)),
                z_offset=float(np.clip(z_offset, 0.003, 0.02)),
                drag_extra=drag_extra,
                stages=stages,
            ):
                return self._log(
                    "pull",
                    {"obj": obj.name, "target": target.name, "oracle": True},
                    True,
                    True,
                    "",
                )
            self._move_towards(self._tcp_pos() + np.array([0.0, 0.0, 0.06], dtype=np.float32), gripper=self.gripper_close, steps=12)

        goal_pos = self._region_pos(target.name)
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "oracle": True, "attempts": len(offsets)},
            False,
            False,
            f"cube was not pulled to target; {self._pull_diagnostics(goal_pos)}",
        )

    def _drive_base_forward_guarded(self) -> None:
        """Use the empirically good Fetch base direction, then stop before arm contact."""

        best = self._tcp_cube_xy()
        stagnant_checks = 0
        total_steps = 0
        while best > 0.145 and total_steps < 40 and not self._early_stop():
            speed = 0.3 if best > 0.16 else 0.15
            for _ in range(8):
                if self._early_stop():
                    return
                self._step(
                    self._make_action(
                        np.zeros(3, dtype=np.float32),
                        gripper=self.gripper_close,
                        base=np.array([speed, 0.0], dtype=np.float32),
                    )
                )
                total_steps += 1
            current = self._tcp_cube_xy()
            if current < best - 0.005:
                best = current
                stagnant_checks = 0
            else:
                stagnant_checks += 1
            if stagnant_checks >= 2:
                break

    def _stop_base(self, *, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(
                self._make_action(
                    np.zeros(3, dtype=np.float32),
                    gripper=self.gripper_close,
                    base=np.zeros(2, dtype=np.float32),
                )
            )

    def _try_contact_drag(
        self,
        target: SkillTarget,
        *,
        x_offset: float,
        z_offset: float,
        drag_extra: float,
        stages: int,
    ) -> bool:
        cube_pos = self._actor_pos("cube")
        goal_pos = self._region_pos(target.name)
        contact = cube_pos + np.array([x_offset, 0.0, z_offset], dtype=np.float32)
        pre_contact = contact + np.array([0.0, 0.0, 0.07], dtype=np.float32)

        self._move_towards(pre_contact, gripper=self.gripper_close, steps=42)
        self._move_towards(contact, gripper=self.gripper_close, steps=42)
        self._repeat_action(np.array([-0.15, 0.0, -0.08], dtype=np.float32), gripper=self.gripper_close, steps=self.contact_steps)

        drag_end = np.array(
            [
                goal_pos[0] - float(drag_extra),
                cube_pos[1],
                max(0.006, contact[2] - 0.004),
            ],
            dtype=np.float32,
        )
        stages = int(np.clip(stages, 1, 8))
        for stage in range(1, stages + 1):
            alpha = stage / stages
            waypoint = contact * (1.0 - alpha) + drag_end * alpha
            self._move_towards(waypoint, gripper=self.gripper_close, steps=max(1, self.drag_steps // stages))
            self._repeat_action(np.array([-0.20, 0.0, -0.04], dtype=np.float32), gripper=self.gripper_close, steps=3)
            if self._pull_cube_success():
                return True

        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)
        return self._pull_cube_success()

    def _tcp_cube_xy(self) -> float:
        cube_pos = self._actor_pos("cube")
        tcp_pos = self._tcp_pos()
        return float(np.linalg.norm((tcp_pos - cube_pos)[:2]))


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedFetchPullCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedFetchPullCubeRobot(
        env,
        robot_uid=robot_uid,
        control_mode=control_mode,
    )
