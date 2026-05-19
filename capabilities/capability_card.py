"""Capability Card for cross-embodiment migration prompts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CapabilityCard:
    # ---- Official / nominal robot specs ----
    reach_m: float = 0.85
    payload_kg: float = 3.0
    repeatability_m: float = 0.001
    dof: int = 7
    gripper_type: str = "unknown"
    mobile_base: bool = False

    # ---- Derived task priors used by migration code ----
    workspace_radius_m: float = 0.85
    ik_accuracy_m: float = 0.03
    insertion_speed_limit_mps: float = 0.015
    recommended_alignment_tolerance_m: float = 0.01
    refusal_conditions: List[str] = field(default_factory=list)

    # ---- Backward-compatible and task-specific metadata ----
    grasp_mechanism: str = "unknown"
    stable_when_stacked: bool = False
    release_must_be_low: bool = False
    recommended_release_height_m: float = 0.05
    can_rotate_object: bool = False
    max_payload_kg: float = 3.0
    has_mobile_base: bool = False
    global_reachable: bool = False
    nav_min_clearance_m: float = 0.4
    has_dual_arms: bool = False
    can_bimanual: bool = False
    can_hold_object: bool = False
    can_coordinate_arms: bool = False
    left_workspace_radius_m: float = 0.75
    right_workspace_radius_m: float = 0.75

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_section(self) -> str:
        lines = ["# Capability Card"]
        lines.append("")
        lines.append("## Official Specs")
        for key in _OFFICIAL_SPEC_FIELDS:
            lines.append(_format_field(key, getattr(self, key)))

        lines.append("")
        lines.append("## Derived Task Priors")
        for key in _DERIVED_PRIOR_FIELDS:
            lines.append(_format_field(key, getattr(self, key)))

        lines.append("")
        lines.append("# Implications for code generation:")
        for hint in self._implications():
            lines.append(f"  - {hint}")
        return "\n".join(lines)

    def _implications(self) -> list:
        hints = []
        if self.release_must_be_low:
            hints.append(
                f"When placing, descend to within {self.recommended_release_height_m:.3f}m "
                f"above the target before releasing. High-altitude release will cause bounce."
            )
        if not self.stable_when_stacked:
            hints.append(
                "Objects tend to roll off after release during stacking. "
                "Consider using a wider-base target, or hold longer before releasing."
            )
        gripper_type = self.gripper_type or self.grasp_mechanism
        if gripper_type == "suction":
            hints.append("Suction grasps the TOP of objects. Approach from directly above.")
        elif gripper_type == "parallel_jaw":
            hints.append("Parallel jaws grasp from the SIDES. Object must fit between fingers.")
        if self.ik_accuracy_m > 0.02:
            hints.append(
                f"IK typical error is ~{self.ik_accuracy_m:.3f}m; "
                f"do not rely on sub-cm precision."
            )

        has_mobile_base = self.mobile_base or self.has_mobile_base
        if has_mobile_base:
            hints.append(
                f"You have a MOBILE BASE. Arm single-point reach is only "
                f"{self.workspace_radius_m:.2f}m, but you can navigate anywhere on the floor."
            )
            hints.append(
                f"BEFORE picking/placing, ALWAYS check `mobile.is_reachable(target)`. "
                f"If False, call `mobile.navigate_to(x, y)` first, positioning the base "
                f"at a table-side standoff about {self.nav_min_clearance_m:.2f}m away "
                f"from the target. "
                f"Do NOT navigate to the target's exact (x, y); the arm cannot work if "
                f"the base is parked on top of the object."
            )
            hints.append(
                "After navigating, re-query object positions if needed; "
                "your arm's effective workspace has shifted with the base."
            )
        if self.has_dual_arms:
            hints.append(
                "You have TWO ARMS: `robot.left` and `robot.right`. "
                "Use `robot.is_reachable_by('left', target)` and "
                "`robot.is_reachable_by('right', target)` to choose an arm."
            )
            hints.append(
                "Use `robot.pick_with_arm(arm_name, src)` and "
                "`robot.place_with_arm(arm_name, dst)` for explicit arm assignment. "
                "`robot.pick_and_place(src, dst)` is still available and auto-selects one arm."
            )
            if self.can_hold_object:
                hints.append(
                    "One arm can hold or stabilize an object while the other arm manipulates. "
                    "This enables sequential dual-arm tasks such as hold-then-place."
                )
            if self.can_coordinate_arms:
                hints.append(
                    "For instructions requiring two objects to be lifted at the same time, "
                    "prefer the coordinated API `robot.lift_two_objects(pos_a, pos_b)` "
                    "instead of two separate sequential `pick_with_arm` calls. "
                    "Use `robot.place_two_objects(target_a, target_b)` for coordinated placement."
                )
        if not has_mobile_base and not self.has_dual_arms:
            hints.append(
                f"You have a FIXED base. Targets outside the {self.workspace_radius_m:.2f}m "
                f"radius from the base are physically UNREACHABLE - refuse such tasks."
            )

        return hints


_OFFICIAL_SPEC_FIELDS = (
    "reach_m",
    "payload_kg",
    "repeatability_m",
    "dof",
    "gripper_type",
    "mobile_base",
)

_DERIVED_PRIOR_FIELDS = (
    "workspace_radius_m",
    "ik_accuracy_m",
    "insertion_speed_limit_mps",
    "recommended_alignment_tolerance_m",
    "refusal_conditions",
)


def _format_field(key: str, value: Any) -> str:
    comment = _COMMENTS.get(key, "")
    comment = f"   # {comment}" if comment else ""
    return f"  {key}: {value!r}{comment}"


_COMMENTS = {
    "reach_m": "nominal maximum reach from official specs (meters)",
    "payload_kg": "nominal maximum payload from official specs (kg)",
    "repeatability_m": "nominal repeatability from official specs (meters)",
    "dof": "degrees of freedom",
    "gripper_type": "end-effector / gripper type",
    "mobile_base": "whether the platform includes a mobile base",
    "recommended_alignment_tolerance_m": "recommended control tolerance, not the task success requirement",
    "insertion_speed_limit_mps": "task-level safe insertion speed limit",
    "refusal_conditions": "conditions where generated code should refuse or avoid the task",
    "grasp_mechanism": "how the gripper holds objects",
    "stable_when_stacked": "whether released objects stay put on stacks",
    "release_must_be_low": "whether release height must be near the target",
    "recommended_release_height_m": "safe release altitude (meters)",
    "workspace_radius_m": "single-point end-effector reach (meters)",
    "can_rotate_object": "can rotate held object in-hand",
    "max_payload_kg": "max payload in kg",
    "ik_accuracy_m": "typical IK positioning error (meters)",
    "has_mobile_base": "robot can navigate around the scene",
    "global_reachable": "robot can reach anywhere on the floor (via navigation)",
    "nav_min_clearance_m": "preferred distance to keep from target after navigating",
    "has_dual_arms": "robot has independently controlled left/right arms",
    "can_bimanual": "robot can use both arms in one task",
    "can_hold_object": "one arm can hold or stabilize while the other manipulates",
    "can_coordinate_arms": "robot can execute coordinated two-arm motions",
    "left_workspace_radius_m": "left arm approximate end-effector reach (meters)",
    "right_workspace_radius_m": "right arm approximate end-effector reach (meters)",
}
