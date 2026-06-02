"""Canonical migration cases for the current ManiSkill experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class FullMigrationCase:
    """A fixed source-target protocol for one complete migration study."""

    case_id: str
    case_number: int
    title: str
    task_id: str
    source_robot: str
    target_robot: str
    source_control_mode: str
    target_control_mode: str
    target_program_path: str
    target_adapter_module: str
    target_adapter_path: str
    seed: int
    max_attempts: int
    max_episode_steps: int
    migration_layers: Tuple[str, ...]
    required_evidence: Tuple[str, ...]
    notes: str = ""


CASE01_PULL_CUBE = FullMigrationCase(
    case_id="case01_pull_cube_panda_to_fetch",
    case_number=1,
    title="PullCube Panda to Fetch",
    task_id="pull_cube",
    source_robot="panda",
    target_robot="fetch",
    source_control_mode="pd_ee_delta_pos",
    target_control_mode="pd_ee_delta_pos",
    target_program_path="maniskill_backend/case_programs/case01_pull_cube.py",
    target_adapter_module="maniskill_backend.generated_adapters.case01_fetch_pull_cube",
    target_adapter_path="maniskill_backend/generated_adapters/case01_fetch_pull_cube.py",
    seed=0,
    max_attempts=3,
    max_episode_steps=100,
    migration_layers=(
        "program",
        "skill_adapter",
        "controller_primitive",
        "contact_primitive",
    ),
    required_evidence=(
        "Panda source task stack succeeds in real ManiSkill simulation.",
        "Fetch source-copy exposes target portability evidence.",
        "LLM-generated target adapter modules and real failure feedback are saved.",
        "Final Fetch success is evaluated with real success state and logs.",
    ),
    notes=(
        "Primary clean migration case. PullCube-v1 is a contact-rich pulling "
        "task supported by both active embodiments."
    ),
)

CASE02_PULL_CUBE_XARM6 = FullMigrationCase(
    case_id="case02_pull_cube_panda_to_xarm6",
    case_number=2,
    title="PullCube Panda to xarm6_robotiq",
    task_id="pull_cube",
    source_robot="panda",
    target_robot="xarm6_robotiq",
    source_control_mode="pd_ee_delta_pos",
    target_control_mode="pd_ee_delta_pos",
    target_program_path="maniskill_backend/case_programs/case01_pull_cube.py",
    target_adapter_module="maniskill_backend.generated_adapters.case02_xarm6_pull_cube",
    target_adapter_path="maniskill_backend/generated_adapters/case02_xarm6_pull_cube.py",
    seed=0,
    max_attempts=3,
    max_episode_steps=500,
    migration_layers=(
        "program",
        "skill_adapter",
        "controller_primitive",
        "contact_primitive",
    ),
    required_evidence=(
        "Panda source task stack succeeds in real ManiSkill simulation.",
        "xarm6_robotiq source-copy or initial adapter execution exposes target portability evidence.",
        "Target adapter changes are evaluated through real ManiSkill execution.",
        "Final success or failure is evaluated with real success state and logs.",
    ),
    notes=(
        "Primary success-candidate migration case. xarm6_robotiq is a fixed-base "
        "single-arm target, so this case should isolate controller/contact "
        "migration without Fetch-style mobile-base reachability failures."
    ),
)

CASE03_PICK_CUBE_XARM6 = FullMigrationCase(
    case_id="case03_pick_cube_panda_to_xarm6",
    case_number=3,
    title="PickCube Panda to xarm6_robotiq",
    task_id="pick_cube",
    source_robot="panda",
    target_robot="xarm6_robotiq",
    source_control_mode="pd_ee_delta_pos",
    target_control_mode="pd_ee_delta_pos",
    target_program_path="maniskill_backend/case_programs/case03_pick_cube.py",
    target_adapter_module="maniskill_backend.generated_adapters.case03_xarm6_pick_cube",
    target_adapter_path="maniskill_backend/generated_adapters/case03_xarm6_pick_cube.py",
    seed=0,
    max_attempts=3,
    max_episode_steps=500,
    migration_layers=(
        "program",
        "skill_adapter",
        "controller_primitive",
        "grasp_geometry",
    ),
    required_evidence=(
        "Panda source task stack succeeds in real ManiSkill simulation.",
        "xarm6_robotiq target execution uses real grasp validation.",
        "Target adapter changes are evaluated through real ManiSkill execution.",
        "Final success or failure is evaluated with real success state and logs.",
    ),
    notes=(
        "Primary grasp-migration case. PickCube-v1 requires the xarm6 target "
        "adapter to establish a real gripper grasp, lift the cube, and move it "
        "to a 3D goal while the low-level controller remains frozen."
    ),
)

FULL_MIGRATION_CASES: Dict[str, FullMigrationCase] = {
    CASE01_PULL_CUBE.case_id: CASE01_PULL_CUBE,
    CASE02_PULL_CUBE_XARM6.case_id: CASE02_PULL_CUBE_XARM6,
    CASE03_PICK_CUBE_XARM6.case_id: CASE03_PICK_CUBE_XARM6,
}

PRIMARY_FULL_MIGRATION_CASE_ID = CASE03_PICK_CUBE_XARM6.case_id
PRIMARY_FULL_MIGRATION_CASE = CASE03_PICK_CUBE_XARM6


def get_full_migration_case(case_id: str) -> FullMigrationCase:
    try:
        return FULL_MIGRATION_CASES[case_id]
    except KeyError as exc:
        available = ", ".join(sorted(FULL_MIGRATION_CASES))
        raise KeyError(f"Unknown migration case {case_id!r}. Available: {available}") from exc


def iter_full_migration_cases() -> Iterable[FullMigrationCase]:
    return FULL_MIGRATION_CASES.values()
