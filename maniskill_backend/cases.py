"""Canonical full-stack migration cases for the ManiSkill experiments."""

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


CASE01_PULL_CUBE_TOOL = FullMigrationCase(
    case_id="case01_pull_cube_tool_panda_to_xarm6",
    case_number=1,
    title="PullCubeTool Panda to xarm6",
    task_id="pull_cube_tool",
    source_robot="panda",
    target_robot="xarm6_robotiq",
    source_control_mode="pd_joint_pos",
    target_control_mode="pd_joint_pos",
    target_program_path="maniskill_backend/case_programs/case01_pull_cube_tool.py",
    target_adapter_module="maniskill_backend.generated_adapters.case01_xarm6_pull_tool",
    target_adapter_path="maniskill_backend/generated_adapters/case01_xarm6_pull_tool.py",
    seed=0,
    max_attempts=3,
    max_episode_steps=300,
    migration_layers=(
        "program",
        "skill_adapter",
        "controller_primitive",
    ),
    required_evidence=(
        "Panda source task stack succeeds in real ManiSkill simulation.",
        "xarm6 source-copy exposes target portability failures.",
        "LLM-generated target adapter modules and real failure feedback are saved.",
        "LLM target-module changes are recorded with physical evidence and migration analysis.",
        "Final xarm6 success is evaluated with real success state and logs.",
    ),
    notes=(
        "First complete migration case. Tool use forces target-specific grasp, "
        "held-tool compensation, contact correction, and pull-frame choices."
    ),
)


FULL_MIGRATION_CASES: Dict[str, FullMigrationCase] = {
    CASE01_PULL_CUBE_TOOL.case_id: CASE01_PULL_CUBE_TOOL,
}

PRIMARY_FULL_MIGRATION_CASE_ID = CASE01_PULL_CUBE_TOOL.case_id
PRIMARY_FULL_MIGRATION_CASE = CASE01_PULL_CUBE_TOOL


def get_full_migration_case(case_id: str) -> FullMigrationCase:
    try:
        return FULL_MIGRATION_CASES[case_id]
    except KeyError as exc:
        available = ", ".join(sorted(FULL_MIGRATION_CASES))
        raise KeyError(f"Unknown full migration case {case_id!r}. Available: {available}") from exc


def iter_full_migration_cases() -> Iterable[FullMigrationCase]:
    return FULL_MIGRATION_CASES.values()
