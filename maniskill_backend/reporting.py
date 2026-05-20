"""Reporting helpers for real ManiSkill trials."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from lmp.failure_report import FailureReport

from .evaluation import TrialRecord
from .profiles import RobotProfile
from .tasks import TaskSpec


def success_from_ret_val(ret_val: Any) -> bool:
    if ret_val is True:
        return True
    if ret_val is None:
        return False
    if isinstance(ret_val, str):
        return not ret_val.lower().startswith("failure")
    return bool(ret_val)


def build_oracle_code(task: TaskSpec) -> str:
    """Return the hand-written real-simulation baseline for a task.

    The real benchmark currently exposes only task programs that can be executed
    against ManiSkill skill adapters, so the oracle is the validated source
    program rather than a hidden static shortcut.
    """

    return task.source_program.strip()


def build_real_failure_report(
    *,
    task: TaskSpec,
    target_profile: RobotProfile,
    failed_record: TrialRecord,
) -> FailureReport:
    execution_log = failed_record.info.get("execution_log", [])
    failed_event = _first_failed_event(execution_log)
    failed_step = _format_failed_step(failed_event)
    message = _sanitize_failure_message(
        str(
            (failed_event or {}).get("message")
            or failed_record.message
            or failed_record.failure_type
        )
    )

    if task.task_id == "pick_cube":
        failed_api = str((failed_event or {}).get("api", "unknown skill"))
        return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "grasp_cube": "robot.grasp(cube) returns True",
                "place_cube": "robot.place(cube, goal) returns True after grasp succeeds",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
            },
            diagnosis=[
                f"Execution log failed at {failed_step}.",
                message,
                (
                    "The failing operation was a real ManiSkill-backed skill "
                    f"wrapper call: {failed_api}."
                ),
            ],
            suggestions=[
                "Keep the grasp guard: only call place after robot.grasp(cube) returns True.",
                "If grasp fails, set ret_val to a clear failure string.",
                "Use only the allowed high-level skill API; do not invent object pose APIs.",
                "If the same code succeeds under a planner control mode, report this as a controller/skill-wrapper portability issue.",
            ],
        )

    if task.task_id == "peg_insertion":
        return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "grasp_peg": "robot.grasp(peg) returns True",
                "align_peg": "robot.align_to_target(peg, hole, tolerance=...) returns True",
                "insert_peg": "robot.insert(peg, hole, speed=...) returns True",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
            },
            diagnosis=[
                f"Execution log failed at {failed_step}.",
                message,
                "Peg insertion is contact-sensitive, so alignment and insertion speed must be conservative.",
            ],
            suggestions=[
                "Call align_to_target before insert and check its return value.",
                "Choose tolerance no tighter than the target Capability Card recommends.",
                "Choose insertion speed no faster than the target Capability Card limit.",
                "Use explicit numeric values from the prompt instead of hidden robot fields.",
            ],
        )

    if task.task_id == "stack_cube":
        return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "grasp_cubeA": "robot.grasp(cubeA) returns True",
                "place_cubeA_on_cubeB": "robot.place(cubeA, cubeB) returns True",
                "stack_state": "cubeA is on cubeB and static after release",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
            },
            diagnosis=[
                f"Execution log failed at {failed_step}.",
                message,
                "Stacking requires both placement accuracy and post-release stability.",
            ],
            suggestions=[
                "Keep the grasp guard: only call place after robot.grasp(cubeA) returns True.",
                "Place cubeA on cubeB using the high-level place API, not raw object pose access.",
                "If placement fails after release, treat it as a skill-wrapper/controller stability issue.",
                "Use only the allowed high-level skill API.",
            ],
        )

    return FailureReport(
        task_name=task.task_id,
        instruction=task.instruction,
        robot_name=target_profile.name,
        expected={"execution_result": "success"},
        actual={
            "execution_result": "failure",
            "failure_type": failed_record.failure_type,
            "message": message,
            "failed_skill_call": failed_step,
        },
        diagnosis=[f"Execution log failed at {failed_step}.", message],
        suggestions=[
            "Use only the allowed high-level skill API.",
            "Check each skill return value before calling the next skill.",
        ],
    )


def _sanitize_failure_message(message: str) -> str:
    text = re.sub(
        r"than [\w_]+ can reliably achieve \([0-9.]+\)",
        "than the target robot can reliably achieve",
        message,
    )
    text = re.sub(
        r"exceeds [\w_]+ limit [0-9.]+",
        "exceeds the target robot insertion speed limit",
        text,
    )
    return text


def _first_failed_event(execution_log: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(execution_log, list):
        return None
    for event in execution_log:
        if isinstance(event, dict) and event.get("ok") is False:
            return event
    return None


def _format_failed_step(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "unknown skill call"
    step = event.get("step", "?")
    api = event.get("api", "unknown_api")
    args = event.get("args", {})
    if isinstance(args, dict) and args:
        arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(args.items()))
        return f"step {step}: {api}({arg_text})"
    return f"step {step}: {api}(...)"
