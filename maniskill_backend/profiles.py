"""Robot capability profiles for the current two-robot migration study."""

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
        "max_payload_kg",
        "workspace_radius_m",
        "ik_accuracy_m",
        "recommended_alignment_tolerance_m",
        "refusal_conditions",
        "has_mobile_base",
        "global_reachable",
        "nav_min_clearance_m",
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
    return values


ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "panda": RobotProfile(
        name="panda",
        role="source robot / fixed single arm",
        notes="Baseline source embodiment for PullCube-v1.",
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
            available_skills=[
                "pull",
                "move_to_contact",
                "drag_contact",
            ],
        ),
    ),
    "fetch": RobotProfile(
        name="fetch",
        role="target robot / mobile manipulator",
        notes="Target embodiment for PullCube-v1 migration.",
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
            available_skills=[
                "pull",
                "move_to_contact",
                "drag_contact",
                "base_aware_contact",
            ],
        ),
    ),
}


def get_robot_profile(name: str) -> RobotProfile:
    try:
        return ROBOT_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(ROBOT_PROFILES))
        raise KeyError(f"Unknown robot profile {name!r}. Available: {available}") from exc


def iter_robot_profiles() -> Iterable[RobotProfile]:
    return ROBOT_PROFILES.values()
