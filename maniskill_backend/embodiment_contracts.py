"""Machine-readable embodiment contracts for adapter synthesis.

Robot profiles describe broad capabilities for people and prompts. Contracts
describe the execution boundary that generated adapters must obey: action
layout, frozen interfaces, mobility, and runtime checks that require measured
simulation evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class ActionChannel:
    """One contiguous channel in a normalized ManiSkill action vector."""

    name: str
    start: int
    stop: int
    semantics: str


@dataclass(frozen=True)
class EmbodimentContract:
    """Frozen interface and physical assumptions for one robot/control mode."""

    robot_uid: str
    control_mode: str
    action_dim: int
    action_channels: Tuple[ActionChannel, ...]
    mobile_base: bool
    arm_dof: int
    tcp_frame: str
    gripper_type: str
    gripper_action_semantics: str
    workspace_model: str
    frozen_interfaces: Tuple[str, ...]
    runtime_constraints: Tuple[str, ...]
    guarded_adapter_requirements: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_section(self) -> str:
        lines = [
            f"# Embodiment contract: {self.robot_uid}",
            f"control_mode: {self.control_mode}",
            f"action_dim: {self.action_dim}",
            f"mobile_base: {self.mobile_base}",
            f"arm_dof: {self.arm_dof}",
            f"tcp_frame: {self.tcp_frame}",
            f"gripper_type: {self.gripper_type}",
            f"workspace_model: {self.workspace_model}",
            "action_channels:",
        ]
        for channel in self.action_channels:
            lines.append(
                f"- {channel.name}: action[{channel.start}:{channel.stop}] "
                f"({channel.semantics})"
            )
        lines.append("runtime_constraints:")
        lines.extend(f"- {item}" for item in self.runtime_constraints)
        lines.append("guarded_adapter_requirements:")
        lines.extend(f"- {item}" for item in self.guarded_adapter_requirements)
        return "\n".join(lines)

    def validate_action_shape(self, shape: Any) -> Dict[str, Any]:
        observed_dim = None
        if shape:
            try:
                observed_dim = int(shape[-1])
            except (TypeError, ValueError, IndexError):
                observed_dim = None
        return {
            "valid": observed_dim == self.action_dim,
            "expected_action_dim": self.action_dim,
            "observed_action_dim": observed_dim,
            "constraint_id": "action_layout_matches_contract",
        }


COMMON_FROZEN_INTERFACES = (
    "high_level_program",
    "low_level_controller",
    "simulator",
    "task_success_signal",
)


EMBODIMENT_CONTRACTS: Dict[Tuple[str, str], EmbodimentContract] = {
    ("panda", "pd_ee_delta_pos"): EmbodimentContract(
        robot_uid="panda",
        control_mode="pd_ee_delta_pos",
        action_dim=4,
        action_channels=(
            ActionChannel("arm_delta_xyz", 0, 3, "normalized TCP position delta"),
            ActionChannel("gripper", 3, 4, "single parallel-jaw command"),
        ),
        mobile_base=False,
        arm_dof=7,
        tcp_frame="tcp",
        gripper_type="parallel_jaw",
        gripper_action_semantics="one normalized open/close command",
        workspace_model="runtime_measurement_required",
        frozen_interfaces=COMMON_FROZEN_INTERFACES,
        runtime_constraints=(
            "Clip every action to env.action_space.low/high.",
            "Treat TCP reachability and contact as runtime measurements, not constants.",
            "Do not add base or body channels.",
        ),
        guarded_adapter_requirements=(
            "Check progress after contact-rich action segments.",
            "Return real failure when an episode or reachability guard is exhausted.",
        ),
    ),
    ("xarm6_robotiq", "pd_ee_delta_pos"): EmbodimentContract(
        robot_uid="xarm6_robotiq",
        control_mode="pd_ee_delta_pos",
        action_dim=4,
        action_channels=(
            ActionChannel("arm_delta_xyz", 0, 3, "normalized TCP position delta"),
            ActionChannel("gripper_active", 3, 4, "Robotiq active mimic-joint command"),
        ),
        mobile_base=False,
        arm_dof=6,
        tcp_frame="eef",
        gripper_type="robotiq_parallel_jaw",
        gripper_action_semantics="one active mimic command; passive joints are not action channels",
        workspace_model="runtime_measurement_required",
        frozen_interfaces=COMMON_FROZEN_INTERFACES,
        runtime_constraints=(
            "Observed pd_ee_delta_pos action shape must be exactly (4,).",
            "Do not invent Fetch-style base or body channels.",
            "Use measured TCP/object residuals before contact or gripper close.",
            "Abort or switch strategy when object progress reverses.",
        ),
        guarded_adapter_requirements=(
            "Select contact/grasp geometry from current TCP, object, and goal state.",
            "Separate approach failure from contact or gripper-envelope failure.",
            "Keep bounded progress and displacement guards in contact-rich phases.",
        ),
    ),
    ("fetch", "pd_ee_delta_pos"): EmbodimentContract(
        robot_uid="fetch",
        control_mode="pd_ee_delta_pos",
        action_dim=9,
        action_channels=(
            ActionChannel("arm_delta_xyz", 0, 3, "normalized TCP position delta"),
            ActionChannel("gripper", 3, 4, "single mimic-gripper command"),
            ActionChannel("body", 4, 7, "head pan, head tilt, torso lift"),
            ActionChannel("base", 7, 9, "forward velocity and angular velocity"),
        ),
        mobile_base=True,
        arm_dof=7,
        tcp_frame="gripper_link",
        gripper_type="parallel_jaw",
        gripper_action_semantics="one active mimic command at action[3]",
        workspace_model="base_arm_runtime_coordination_required",
        frozen_interfaces=COMMON_FROZEN_INTERFACES,
        runtime_constraints=(
            "Observed pd_ee_delta_pos action shape must be exactly (9,).",
            "Keep body channels zero unless a measured torso/body policy requires them.",
            "Stop base motion before arm contact and drag.",
            "Accept base motion only while measured TCP/object distance improves.",
        ),
        guarded_adapter_requirements=(
            "Use a bounded base-approach branch only when arm-only contact is unreachable.",
            "Switch to arm-only control after base convergence.",
            "Declare infeasible when neither bounded base approach nor arm contact improves reachability.",
        ),
    ),
}


ROBOT_ALIASES = {
    "franka": "panda",
    "franka_panda": "panda",
    "xarm6": "xarm6_robotiq",
}


def normalize_robot_uid(robot_uid: str) -> str:
    value = str(robot_uid or "").strip().lower().replace("-", "_")
    return ROBOT_ALIASES.get(value, value)


def get_embodiment_contract(
    robot_uid: str,
    control_mode: str = "pd_ee_delta_pos",
) -> EmbodimentContract:
    key = (normalize_robot_uid(robot_uid), str(control_mode or "").strip())
    try:
        return EMBODIMENT_CONTRACTS[key]
    except KeyError as exc:
        available = ", ".join(f"{robot}/{mode}" for robot, mode in sorted(EMBODIMENT_CONTRACTS))
        raise KeyError(f"No embodiment contract for {key!r}. Available: {available}") from exc


def iter_embodiment_contracts() -> Iterable[EmbodimentContract]:
    return EMBODIMENT_CONTRACTS.values()


def compare_embodiments(
    source_robot: str,
    target_robot: str,
    *,
    source_control_mode: str = "pd_ee_delta_pos",
    target_control_mode: str = "pd_ee_delta_pos",
) -> Dict[str, Any]:
    source = get_embodiment_contract(source_robot, source_control_mode)
    target = get_embodiment_contract(target_robot, target_control_mode)
    differences = []
    for field in (
        "action_dim",
        "mobile_base",
        "arm_dof",
        "tcp_frame",
        "gripper_type",
        "gripper_action_semantics",
        "workspace_model",
    ):
        source_value = getattr(source, field)
        target_value = getattr(target, field)
        if source_value != target_value:
            differences.append(
                {
                    "field": field,
                    "source": source_value,
                    "target": target_value,
                }
            )
    return {
        "schema": "embodiment_contract_diff.v1",
        "source": source.to_dict(),
        "target": target.to_dict(),
        "differences": differences,
    }


def contract_from_case(case: Any) -> EmbodimentContract:
    return get_embodiment_contract(case.target_robot, case.target_control_mode)


def contract_prompt_from_case(case: Any) -> str:
    target = contract_from_case(case)
    diff = compare_embodiments(
        case.source_robot,
        case.target_robot,
        source_control_mode=case.source_control_mode,
        target_control_mode=case.target_control_mode,
    )
    lines = [
        target.to_prompt_section(),
        "",
        "# Source-target contract differences",
    ]
    if diff["differences"]:
        for item in diff["differences"]:
            lines.append(
                f"- {item['field']}: source={item['source']!r}, target={item['target']!r}"
            )
    else:
        lines.append("- No declared interface differences; runtime geometry still requires validation.")
    return "\n".join(lines)
