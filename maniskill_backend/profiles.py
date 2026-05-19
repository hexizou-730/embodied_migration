"""Robot capability profiles for cross-embodiment migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from capabilities.capability_card import CapabilityCard


@dataclass(frozen=True)
class RobotProfile:
    name: str
    role: str
    card: CapabilityCard
    notes: str = ""

    def to_prompt_section(self) -> str:
        lines = [f"# Robot Profile: {self.name}", f"role: {self.role}"]
        if self.notes:
            lines.append(f"notes: {self.notes}")
        lines.append("")
        lines.append(self.card.to_prompt_section())
        return "\n".join(lines)


def _card(**extra: object) -> CapabilityCard:
    extra = _fill_compat_aliases(dict(extra))
    known_fields = {
        "reach_m",
        "payload_kg",
        "repeatability_m",
        "dof",
        "gripper_type",
        "mobile_base",
        "grasp_mechanism",
        "stable_when_stacked",
        "release_must_be_low",
        "recommended_release_height_m",
        "can_rotate_object",
        "max_payload_kg",
        "workspace_radius_m",
        "ik_accuracy_m",
        "insertion_speed_limit_mps",
        "recommended_alignment_tolerance_m",
        "refusal_conditions",
        "has_mobile_base",
        "global_reachable",
        "nav_min_clearance_m",
        "has_dual_arms",
        "can_bimanual",
        "can_hold_object",
        "can_coordinate_arms",
        "left_workspace_radius_m",
        "right_workspace_radius_m",
    }
    direct = {k: v for k, v in extra.items() if k in known_fields}
    rest = {k: v for k, v in extra.items() if k not in known_fields}
    return CapabilityCard(**direct, extra=rest)


def _fill_compat_aliases(values: Dict[str, object]) -> Dict[str, object]:
    if "gripper_type" not in values and "grasp_mechanism" in values:
        values["gripper_type"] = values["grasp_mechanism"]
    if "grasp_mechanism" not in values and "gripper_type" in values:
        values["grasp_mechanism"] = values["gripper_type"]
    if "payload_kg" not in values and "max_payload_kg" in values:
        values["payload_kg"] = values["max_payload_kg"]
    if "max_payload_kg" not in values and "payload_kg" in values:
        values["max_payload_kg"] = values["payload_kg"]
    if "reach_m" not in values and "workspace_radius_m" in values:
        values["reach_m"] = values["workspace_radius_m"]
    if "workspace_radius_m" not in values and "reach_m" in values:
        values["workspace_radius_m"] = values["reach_m"]
    if "mobile_base" not in values and "has_mobile_base" in values:
        values["mobile_base"] = values["has_mobile_base"]
    if "has_mobile_base" not in values and "mobile_base" in values:
        values["has_mobile_base"] = values["mobile_base"]
    if "recommended_alignment_tolerance_m" not in values and "alignment_tolerance_m" in values:
        values["recommended_alignment_tolerance_m"] = values.pop("alignment_tolerance_m")
    elif "alignment_tolerance_m" in values:
        values.pop("alignment_tolerance_m")
    return values


ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "panda": RobotProfile(
        name="panda",
        role="source robot / standard single arm",
        notes="Baseline source embodiment for source programs.",
        card=_card(
            reach_m=0.85,
            payload_kg=3.0,
            repeatability_m=0.001,
            dof=7,
            gripper_type="parallel_jaw",
            mobile_base=False,
            workspace_radius_m=0.85,
            ik_accuracy_m=0.01,
            recommended_alignment_tolerance_m=0.01,
            insertion_speed_limit_mps=0.02,
            available_skills=[
                "grasp",
                "align_to_target",
                "insert",
                "hook_object",
                "pull_with_tool",
                "move_to_pose",
            ],
        ),
    ),
    "fetch": RobotProfile(
        name="fetch",
        role="target robot / mobile manipulator",
        notes="Tests reachability and mobile-base-aware code adaptation.",
        card=_card(
            reach_m=0.75,
            payload_kg=6.0,
            repeatability_m=0.005,
            dof=7,
            gripper_type="parallel_jaw",
            mobile_base=True,
            workspace_radius_m=0.75,
            ik_accuracy_m=0.025,
            global_reachable=True,
            nav_min_clearance_m=0.45,
            recommended_alignment_tolerance_m=0.02,
            insertion_speed_limit_mps=0.015,
            tool_hook_requires_alignment=True,
            tool_workspace_required_m=0.42,
            available_skills=[
                "mobile.is_reachable",
                "mobile.navigate_to",
                "grasp",
                "align_to_target",
                "insert",
                "hook_object",
                "pull_with_tool",
            ],
        ),
    ),
    "xarm6_robotiq": RobotProfile(
        name="xarm6_robotiq",
        role="target robot / different arm and gripper",
        notes="Tests API, grasp, and precision differences.",
        card=_card(
            reach_m=0.85,
            payload_kg=5.0,
            repeatability_m=0.0001,
            dof=6,
            gripper_type="robotiq_parallel_jaw",
            mobile_base=False,
            workspace_radius_m=0.78,
            ik_accuracy_m=0.018,
            recommended_alignment_tolerance_m=0.015,
            insertion_speed_limit_mps=0.015,
            tool_hook_requires_alignment=True,
            tool_workspace_required_m=0.42,
            available_skills=[
                "grasp",
                "align_to_target",
                "insert",
                "hook_object",
                "pull_with_tool",
                "move_to_pose",
            ],
        ),
    ),
    "so100": RobotProfile(
        name="so100",
        role="target robot / small low-cost arm",
        notes="Tests workspace, payload, and precision limits.",
        card=_card(
            reach_m=0.45,
            payload_kg=0.5,
            repeatability_m=0.01,
            dof=6,
            gripper_type="parallel_jaw",
            mobile_base=False,
            workspace_radius_m=0.45,
            ik_accuracy_m=0.035,
            recommended_alignment_tolerance_m=0.03,
            insertion_speed_limit_mps=0.008,
            tool_hook_requires_alignment=True,
            tool_workspace_required_m=0.42,
            refusal_conditions=[
                "target pose outside fixed-base workspace",
                "requires sub-centimeter insertion precision",
            ],
            available_skills=[
                "grasp",
                "align_to_target",
                "insert",
                "hook_object",
                "pull_with_tool",
            ],
        ),
    ),
    "widowxai": RobotProfile(
        name="widowxai",
        role="target robot / compact arm",
        notes="Tests small-workspace and precision-sensitive failures.",
        card=_card(
            reach_m=0.55,
            payload_kg=0.8,
            repeatability_m=0.005,
            dof=5,
            gripper_type="parallel_jaw",
            mobile_base=False,
            workspace_radius_m=0.55,
            ik_accuracy_m=0.028,
            recommended_alignment_tolerance_m=0.025,
            insertion_speed_limit_mps=0.01,
            tool_hook_requires_alignment=True,
            tool_workspace_required_m=0.42,
            available_skills=[
                "grasp",
                "align_to_target",
                "insert",
                "hook_object",
                "pull_with_tool",
                "move_to_pose",
            ],
        ),
    ),
}


ROBOT_PROFILE_ALIASES: Dict[str, str] = {
    # ManiSkill ships per-env Panda variants (panda_wristcam etc.) whose
    # capability profile is the same as panda for migration purposes.
    "panda_wristcam": "panda",
}


def get_robot_profile(name: str) -> RobotProfile:
    canonical = ROBOT_PROFILE_ALIASES.get(name, name)
    try:
        return ROBOT_PROFILES[canonical]
    except KeyError as exc:
        available = ", ".join(sorted(ROBOT_PROFILES))
        raise KeyError(f"Unknown robot profile {name!r}. Available: {available}") from exc


def iter_robot_profiles() -> Iterable[RobotProfile]:
    return ROBOT_PROFILES.values()
