"""Robot profiles for the optional robosuite migration backend.

The project still uses PyBullet for the original tabletop experiments. These
profiles describe more complex robosuite embodiments and the high-level skill
constraints that the LLM should consider when migrating source programs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from capabilities import CapabilityCard


@dataclass(frozen=True)
class RobosuiteProfile:
    name: str
    display_name: str
    robosuite_robots: Sequence[str]
    env_configuration: str = "parallel"
    arm_names: Sequence[str] = ("left", "right")
    gripper_type: str = "parallel_jaw"
    dof: int = 14
    has_mobile_base: bool = False
    requires_navigation: bool = False
    required_grip_force: float = 0.0
    handover_clearance_m: float = 0.06
    peg_alignment_tolerance_m: float = 0.02
    peg_insert_speed_limit: float = 0.04
    notes: Sequence[str] = field(default_factory=tuple)

    def capability_card(self) -> CapabilityCard:
        return CapabilityCard(
            grasp_mechanism=self.gripper_type,
            stable_when_stacked=False,
            release_must_be_low=True,
            recommended_release_height_m=0.02,
            can_rotate_object=False,
            max_payload_kg=3.0,
            workspace_radius_m=0.85,
            ik_accuracy_m=0.025,
            has_mobile_base=self.has_mobile_base,
            global_reachable=self.has_mobile_base,
            nav_min_clearance_m=0.6,
            has_dual_arms=len(self.arm_names) >= 2,
            can_bimanual=len(self.arm_names) >= 2,
            can_hold_object=len(self.arm_names) >= 2,
            can_coordinate_arms=len(self.arm_names) >= 2,
            left_workspace_radius_m=0.75,
            right_workspace_radius_m=0.75,
            extra={
                "backend": "robosuite_mujoco",
                "robosuite_robots": list(self.robosuite_robots),
                "env_configuration": self.env_configuration,
                "arm_names": list(self.arm_names),
                "requires_navigation": self.requires_navigation,
                "required_grip_force": self.required_grip_force,
                "handover_clearance_m": self.handover_clearance_m,
                "peg_alignment_tolerance_m": self.peg_alignment_tolerance_m,
                "peg_insert_speed_limit": self.peg_insert_speed_limit,
                "notes": list(self.notes),
            },
        )

    def describe(self) -> str:
        mobile = "mobile base + " if self.has_mobile_base else ""
        return (
            f"Embodiment: {self.display_name} | Backend: robosuite/MuJoCo | "
            f"Robots: {list(self.robosuite_robots)} | DoF: {self.dof} | "
            f"{mobile}Arms: {list(self.arm_names)} | Gripper: {self.gripper_type}"
        )

    def to_prompt_section(self) -> str:
        return self.capability_card().to_prompt_section()


PROFILES: Dict[str, RobosuiteProfile] = {
    "rs_dual_panda": RobosuiteProfile(
        name="rs_dual_panda",
        display_name="Dual Panda Manipulator",
        robosuite_robots=("Panda", "Panda"),
        env_configuration="parallel",
        gripper_type="parallel_jaw",
        dof=14,
        notes=(
            "Good baseline dual-arm robot.",
            "Can lift pot handles, hand over objects, and perform peg insertion.",
        ),
    ),
    "rs_dual_iiwa": RobosuiteProfile(
        name="rs_dual_iiwa",
        display_name="Dual KUKA IIWA Manipulator",
        robosuite_robots=("IIWA", "IIWA"),
        env_configuration="parallel",
        gripper_type="parallel_jaw",
        dof=14,
        required_grip_force=0.75,
        peg_alignment_tolerance_m=0.015,
        peg_insert_speed_limit=0.03,
        notes=(
            "Needs explicit grip force before lifting heavier objects.",
            "Peg insertion should use slower insertion and tighter alignment.",
        ),
    ),
    "rs_baxter": RobosuiteProfile(
        name="rs_baxter",
        display_name="Baxter Bimanual Robot",
        robosuite_robots=("Baxter",),
        env_configuration="single-robot",
        gripper_type="parallel_jaw",
        dof=14,
        handover_clearance_m=0.10,
        peg_alignment_tolerance_m=0.025,
        peg_insert_speed_limit=0.025,
        notes=(
            "Bimanual single robot with wider arm spacing.",
            "Handover needs an explicit clearance pose before transfer.",
        ),
    ),
    "rs_mobile_tiago": RobosuiteProfile(
        name="rs_mobile_tiago",
        display_name="Tiago Mobile Manipulator",
        robosuite_robots=("Tiago",),
        env_configuration="mobile",
        arm_names=("right",),
        gripper_type="parallel_jaw",
        dof=20,
        has_mobile_base=True,
        requires_navigation=True,
        notes=(
            "Mobile single-arm embodiment.",
            "Can navigate to stations but cannot execute true simultaneous bimanual tasks.",
        ),
    ),
}


def get_profile(name: str) -> RobosuiteProfile:
    key = name.lower()
    if key not in PROFILES:
        raise ValueError(f"Unknown robosuite profile '{name}'. Available: {sorted(PROFILES)}")
    return PROFILES[key]


def profile_names() -> List[str]:
    return sorted(PROFILES)
