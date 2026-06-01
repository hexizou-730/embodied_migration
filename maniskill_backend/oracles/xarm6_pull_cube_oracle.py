"""Human-written xarm6_robotiq feasibility oracle for PullCube.

This reference is intentionally stored outside ``generated_adapters``. The LLM
module-generation runner does not read it. It proves that the target task is
physically feasible and provides an internal upper bound for later evaluation.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np

from maniskill_backend.skill_adapter import ManiSkillPullCubeRobot


ArmCommand = Tuple[float, float, float]
Phase = Tuple[str, ArmCommand, int]


class XArm6PullCubeOracleRobot(ManiSkillPullCubeRobot):
    """Fixed-base xarm6 feasibility oracle using a measured contact sequence."""

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
                "xarm6 PullCube oracle requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or shape[-1] != 4:
            raise RuntimeError(f"xarm6 PullCube oracle expects observed 4D action space, got {shape!r}.")

    def pull(self, obj, target, *, contact_x_offset=None, contact_z_offset=None, drag_extra=0.025, stages=5) -> bool:
        if obj.name != "cube":
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube oracle only supports cube.")
        if target.name not in {"goal", "goal_region"}:
            return self._fail("pull", {"obj": obj.name, "target": target.name}, "PullCube target must be goal.")

        primary = (
            ("approach_positive_x_side", (0.8, 0.0, 0.0), 100),
            ("descend_to_contact_height", (0.0, 0.0, -0.8), 80),
            ("drag_toward_goal_negative_x", (-0.8, 0.0, -0.05), 160),
        )
        if self._run_phases(primary):
            return self._log("pull", {"obj": obj.name, "target": target.name, "oracle": True}, True, True, "")

        fallback = (
            ("reclose_contact", (0.25, 0.0, -0.3), 30),
            ("drag_toward_goal_negative_x_retry", (-0.75, 0.0, -0.05), 120),
        )
        if self._run_phases(fallback):
            return self._log(
                "pull",
                {"obj": obj.name, "target": target.name, "oracle": True, "retry": True},
                True,
                True,
                "",
            )

        goal_pos = self._region_pos(target.name)
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name, "oracle": True},
            False,
            False,
            f"cube was not pulled to target; {self._pull_diagnostics(goal_pos)}",
        )

    def _run_phases(self, phases: Iterable[Phase]) -> bool:
        for _, command, steps in phases:
            self._repeat_action(np.asarray(command, dtype=np.float32), gripper=self.gripper_close, steps=steps)
            if self._pull_cube_success():
                return True
            if self._early_stop():
                break
        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=self.gripper_close, steps=self.settle_steps)
        return self._pull_cube_success()


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> XArm6PullCubeOracleRobot:
    """Build the internal xarm6 feasibility oracle."""

    return XArm6PullCubeOracleRobot(env, robot_uid=robot_uid, control_mode=control_mode)
