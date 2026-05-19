"""Prompt construction and method definitions for code migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lmp.failure_report import FailureReport

from .profiles import RobotProfile, get_robot_profile
from .tasks import TaskSpec, get_task_spec


METHODS = (
    "source-copy",
    "llm_no_card",
    "llm_card_only",
    "llm_report_only",
    "llm_card_report",
    "llm_static_feedback",
    "oracle",
)

METHOD_ALIASES = {
    "llm_failure_only": "llm_report_only",
    "llm_card_failure": "llm_card_report",
}


def norm_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


@dataclass(frozen=True)
class MigrationRequest:
    task: TaskSpec
    source_profile: RobotProfile
    target_profile: RobotProfile
    method: str
    failure_report: Optional[FailureReport] = None

    @classmethod
    def from_ids(
        cls,
        *,
        task_id: str,
        target_robot: str,
        method: str,
        failure_report: Optional[FailureReport] = None,
    ) -> "MigrationRequest":
        task = get_task_spec(task_id)
        return cls(
            task=task,
            source_profile=get_robot_profile(task.source_robot),
            target_profile=get_robot_profile(target_robot),
            method=norm_method(method),
            failure_report=failure_report,
        )


def build_migration_prompt(request: MigrationRequest) -> str:
    method = norm_method(request.method)
    if method not in METHODS:
        allowed = ", ".join(METHODS)
        raise ValueError(f"Unknown method {request.method!r}. Allowed: {allowed}")

    lines = [
        "You are adapting LMP robot code across robot embodiments.",
        "Output only executable Python code using the provided high-level skill API.",
        "",
        "# Allowed API",
        "- scene.get_object(name)",
        "- scene.get_region(name)",
        "- robot.grasp(obj)",
        "- robot.place(obj, target)",
        "- robot.align_to_target(obj, target, tolerance)",
        "- robot.insert(obj, target, speed)",
        "- robot.hook_object(tool, obj)",
        "- robot.pull_with_tool(tool, obj, target)",
        "",
        "# Hard constraints",
        "- Objects returned by scene are opaque handles. Do not call methods on them.",
        "- Do not use obj.get_position(), obj.pose, obj.position, or distance math.",
        "- Do not import packages.",
        "- Choose alignment tolerance and insertion speed values appropriate for the target robot.",
        "- Use explicit numeric values from the prompt; do not read hidden oracle fields such as robot.recommended_alignment_tolerance or robot.safe_insertion_speed.",
        "- Set ret_val to the final success/failure value.",
        "",
        request.task.to_prompt_section(),
        "",
        request.source_profile.to_prompt_section(),
        "",
        f"# Target Robot: {request.target_profile.name}",
    ]

    if method in {"llm_card_only", "llm_card_report", "llm_static_feedback"}:
        lines.append(request.target_profile.to_prompt_section())
    else:
        lines.append("No target Capability Card is provided for this baseline.")

    if method in {"llm_report_only", "llm_card_report"}:
        if request.failure_report is None:
            lines.append("")
            lines.append("# Failure Report")
            lines.append("No Failure Report was provided yet.")
        else:
            lines.append("")
            lines.append(request.failure_report.to_prompt_section())

    if method == "llm_static_feedback":
        lines.extend(
            [
                "",
                "# Static feedback instruction",
                "Before writing corrected code, inspect whether the source code calls",
                "unavailable skills, ignores workspace limits, exceeds precision limits,",
                "or uses unsafe insertion speed. Then output corrected code only.",
            ]
        )

    lines.extend(
        [
            "",
            "# Required output",
            "Return a Python snippet that sets ret_val to the final success/failure value.",
        ]
    )
    return "\n".join(lines)


def get_source_copy_code(task_id: str) -> str:
    return get_task_spec(task_id).source_program.strip()
