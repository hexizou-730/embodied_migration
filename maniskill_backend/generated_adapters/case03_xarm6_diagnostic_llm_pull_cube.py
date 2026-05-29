"""Seed adapter for xarm6 diagnostic-feedback LLM generation.

This file is intentionally *not* the successful oracle. It gives the LLM a
simple failing starting point for case03, while the prompt supplies the measured
successful contact sequence discovered by diagnostics.

Expected case03 flow:
1. this seed adapter fails in real ManiSkill execution;
2. Opus receives the failure plus diagnostic trace;
3. Opus rewrites this complete module;
4. the rewritten module is evaluated as LLM-generated migration code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


class GeneratedXArm6DiagnosticLLMPullCubeRobot(ManiSkillPullCubeRobot):
    """Failing seed adapter used only to trigger diagnostic-feedback generation."""

    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(
            env,
            robot_uid=robot_uid,
            control_mode=control_mode,
            move_steps=24,
            contact_steps=12,
            drag_steps=72,
            settle_steps=14,
            max_delta_m=0.045,
            contact_x_offset_m=0.055,
            contact_z_offset_m=0.012,
            gripper_open=1.0,
            gripper_close=-1.0,
        )

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "xarm6 PullCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        shape = getattr(getattr(self.env, "action_space", None), "shape", None)
        if not shape or shape[-1] != 4:
            raise RuntimeError(f"xarm6 diagnostic seed expects observed 4D action space, got {shape!r}.")

    def pull(self, obj, target, *, contact_x_offset=None, contact_z_offset=None, drag_extra=0.025, stages=5) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube adapter only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        # Deliberately incomplete: approach the correct side and descend, but
        # omit the measured negative-x drag phase. The LLM should add it.
        self._repeat_action(np.array([0.8, 0.0, 0.0], dtype=np.float32), gripper=self.gripper_close, steps=100)
        self._repeat_action(np.array([0.0, 0.0, -0.8], dtype=np.float32), gripper=self.gripper_close, steps=80)
        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)

        goal_pos = self._region_pos(target.name)
        ok = self._pull_cube_success()
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "diagnostic_seed": True},
            ok,
            ok,
            "" if ok else f"seed adapter omitted negative-x drag; {self._pull_diagnostics(goal_pos)}",
        )


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedXArm6DiagnosticLLMPullCubeRobot:
    """Factory used by ``real_runner --adapter-module``."""

    return GeneratedXArm6DiagnosticLLMPullCubeRobot(env, robot_uid=robot_uid, control_mode=control_mode)
