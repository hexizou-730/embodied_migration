"""Prompt construction for the LMP-program layer of ManiSkill migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lmp.failure_report import FailureReport

from .profiles import RobotProfile, get_robot_profile
from .tasks import TaskSpec, get_task_spec


METHODS = (
    "source-copy",
    "llm_card_report",
    "oracle",
)

METHOD_ALIASES = {}


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
        "- robot.pull(obj, target)",
        "",
        "# Safety constraints",
        "- Do not fake success, bypass task outcomes, or directly modify simulator state.",
        "- If the target cannot realize the task with the exposed API, set ret_val to a string beginning `infeasible:` and briefly state the reason.",
        "",
        "# Code-generation constraints",
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

    if request.task.task_id == "pull_cube":
        lines.extend(
            [
                "",
                "# Task-specific API note",
                "- For PullCube-v1, use robot.pull(cube, goal).",
                "- robot.pull(cube, goal, contact_x_offset=0.07, contact_z_offset=0.02, stages=4) can tune the contact primitive.",
        "- Do not invent grasp/place/tool APIs for PullCube-v1; this is a contact-pulling task.",
                "- If all exposed contact parameter choices are infeasible for the target, set ret_val to `infeasible: ...` rather than adding unsupported low-level object access.",
            ]
        )

    if method == "llm_card_report":
        lines.append(request.target_profile.to_prompt_section())
    else:
        lines.append("No target Capability Card is provided for this baseline.")

    if method == "llm_card_report":
        if request.failure_report is None:
            lines.append("")
            lines.append("# Failure Report")
            lines.append("No Failure Report was provided yet.")
        else:
            lines.append("")
            lines.append(request.failure_report.to_prompt_section())

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
