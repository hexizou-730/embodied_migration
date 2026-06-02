"""Reporting helpers for ManiSkill trials."""

from __future__ import annotations

import re
from typing import Any

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
        lowered = ret_val.lower()
        return not (lowered.startswith("failure") or lowered.startswith("infeasible"))
    return bool(ret_val)


def build_oracle_code(task: TaskSpec) -> str:
    """Return the current source program as the deterministic baseline."""

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
        expected = {
            "execution_result": "success",
            "pick_cube": "robot.grasp(cube) and robot.place(cube, goal) return True",
            "target_state": "cube is grasped, lifted, and moved to the 3D goal position",
        }
        diagnosis = [
            f"Execution log failed at {failed_step}.",
            message,
            "PickCube-v1 requires a real gripper grasp before lift and transport.",
        ]
        suggestions = [
            "Call robot.grasp(cube) before robot.place(cube, goal).",
            "Tune bounded grasp offsets, approach height, lift height, and gripper settle timing.",
            "Do not replace grasping with pushing or directly modify cube state.",
            "Use only the allowed high-level skill API.",
        ]
    else:
        expected = {
            "execution_result": "success",
            "pull_cube": "robot.pull(cube, goal) returns True",
            "target_state": "cube is pulled onto the goal region",
        }
        diagnosis = [
            f"Execution log failed at {failed_step}.",
            message,
            "PullCube-v1 is a contact task, so failure is contact/controller migration evidence.",
        ]
        suggestions = [
            "Use robot.pull(cube, goal) and tune only exposed contact parameters.",
            "Do not add robot.grasp(cube); PullCube-v1 is solved by contact pulling.",
            "If contact parameters cannot solve the failure, report `infeasible: target contact/controller migration required`.",
            "Use only the allowed high-level skill API.",
        ]

    return FailureReport(
        task_name=task.task_id,
        instruction=task.instruction,
        robot_name=target_profile.name,
        expected=expected,
        actual={
            "execution_result": "failure",
            "failure_type": failed_record.failure_type,
            "failure_layer": failed_record.failure_layer,
            "message": message,
            "failed_skill_call": failed_step,
        },
        diagnosis=diagnosis,
        suggestions=suggestions,
    )


def _first_failed_event(execution_log: Any) -> dict[str, Any] | None:
    if not isinstance(execution_log, list):
        return None
    for event in execution_log:
        if isinstance(event, dict) and event.get("ok") is False:
            return event
    return None


def _format_failed_step(event: dict[str, Any] | None) -> str:
    if not event:
        return "unknown"
    step = event.get("step", "?")
    api = event.get("api", "unknown")
    args = event.get("args", {})
    return f"step {step}: {api}({args})"


def _sanitize_failure_message(message: str) -> str:
    text = re.sub(
        r"than [\w_]+ can reliably achieve \([0-9.]+\)",
        "than the target robot can reliably achieve",
        message,
    )
    text = re.sub(
        r"exceeds [\w_]+ limit [0-9.]+",
        "exceeds the target robot limit",
        text,
    )
    return text
