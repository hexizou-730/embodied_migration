"""Evaluation records and lightweight failure classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


FAILURE_TYPES = (
    "success",
    "api mismatch",
    "reachability failure",
    "gripper/force failure",
    "alignment failure",
    "placement stability failure",
    "contact execution failure",
    "insertion failure",
    "insertion speed failure",
    "tool-use ordering failure",
    "tool-use execution failure",
    "impossible-task refusal failure",
    "invalid generated code",
    "execution failure",
    "unknown failure",
)

FAILURE_LAYERS = (
    "success",
    "program",
    "skill_adapter",
    "controller_primitive",
    "contact_geometry",
    "infeasibility",
    "task_outcome",
    "runtime_setup",
    "unknown",
)


@dataclass
class TrialRecord:
    task_id: str
    source_robot: str
    target_robot: str
    method: str
    seed: int
    generated_code: str
    success: bool
    failure_type: str
    failure_layer: str = "unknown"
    attempts: int = 1
    message: str = ""
    prompt: str = ""
    failure_report: str = ""
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.success and self.failure_layer == "unknown":
            self.failure_layer = "success"


def classify_failure(
    *,
    success: bool,
    code_ok: bool = True,
    message: str = "",
    info: Optional[Dict[str, Any]] = None,
) -> str:
    if success:
        return "success"
    if not code_ok:
        if "SafetyError" in message or "forbidden" in message.lower():
            return "invalid generated code"
        if "attributeerror" in message.lower() or "nameerror" in message.lower():
            return "api mismatch"
        return "execution failure"

    info = info or {}
    candidates = [
        str(info.get("failure_type", "")),
        str(info.get("diagnosis", "")),
        message,
    ]
    text = " ".join(candidates).lower()

    if _looks_like_infeasible_failure(text):
        return "impossible-task refusal failure"
    if "unreachable" in text or "reach" in text:
        return "reachability failure"
    if (
        "tool pull failed" in text
        or "not pulled into workspace" in text
        or "cube_progress" in text
        or "cube_distance" in text
    ):
        return "tool-use execution failure"
    if "not pulled" in text or "pull" in text and "target" in text or "contact" in text:
        return "contact execution failure"
    if "grasp" in text or "gripper" in text or "force" in text or "slip" in text:
        return "gripper/force failure"
    if "ordering" in text or "called before" in text:
        return "tool-use ordering failure"
    if "align" in text or "misalign" in text or "pose error" in text:
        return "alignment failure"
    if "stack" in text or "on cubeb" in text or "cubea_static" in text or "stability" in text:
        return "placement stability failure"
    if "not inserted" in text or "insertion failure" in text or "peg_head_pos_at_hole" in text:
        return "insertion failure"
    if "speed" in text or "too fast" in text:
        return "insertion speed failure"
    if "not placed" in text or "not at goal" in text or "not moved to goal" in text or "place" in text:
        return "execution failure"
    ret_val_text = str(info.get("ret_val", "")).lower()
    if "failure" in ret_val_text:
        log = info.get("execution_log") or []
        skill_failed = any(
            isinstance(event, dict) and event.get("ok") is False for event in log
        )
        if not skill_failed:
            return "impossible-task refusal failure"

    return "unknown failure"


def _looks_like_infeasible_failure(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "impossible",
            "unsafe",
            "refuse",
            "sub-centimeter",
            "sub-cm",
            "requires sub",
            "outside workspace",
            "outside fixed-base",
            "infeasible",
            "not feasible",
            "cannot perform",
            "exceeds the robot",
            "beyond the capability",
        )
    )


def classify_failure_layer(
    *,
    success: bool,
    code_ok: bool = True,
    message: str = "",
    info: Optional[Dict[str, Any]] = None,
) -> str:
    """Locate the migration layer that most directly exposed a failure."""

    if success:
        return "success"
    if not code_ok:
        return "program"

    info = info or {}
    text = _failure_layer_text(message=message, info=info)
    if _looks_like_runtime_setup_failure(text, info):
        return "runtime_setup"

    failed_events = _failed_execution_events(info.get("execution_log"))
    if failed_events:
        failed_text = " ".join(
            _event_text(event) for event in failed_events
        ).lower()
        last_info = info.get("final_info") or info.get("last_info") or info
        if _looks_like_controller_failure(failed_text, last_info):
            return "controller_primitive"
        return "skill_adapter"

    ret_val = str(info.get("ret_val", "")).lower()
    if ret_val.startswith("failure") or ret_val in {"false", "none"}:
        return "program"
    if _looks_like_task_outcome_failure(text, info):
        return "task_outcome"
    return "unknown"


def _failure_layer_text(*, message: str, info: Dict[str, Any]) -> str:
    pieces = [
        message,
        str(info.get("failure_type", "")),
        str(info.get("diagnosis", "")),
        str(info.get("final_info", "")),
        str(info.get("last_info", "")),
    ]
    return " ".join(pieces).lower()


def _failed_execution_events(execution_log: Any) -> list[Dict[str, Any]]:
    if not isinstance(execution_log, list):
        return []
    return [
        event
        for event in execution_log
        if isinstance(event, dict) and event.get("ok") is False
    ]


def _event_text(event: Dict[str, Any]) -> str:
    return " ".join(
        str(event.get(key, ""))
        for key in ("api", "failure_type", "message", "result", "args")
    )


def _looks_like_runtime_setup_failure(text: str, info: Dict[str, Any]) -> bool:
    if info.get("graphics_diagnosis"):
        return True
    return any(
        marker in text
        for marker in (
            "graphics",
            "render",
            "vulkan",
            "cuda",
            "importerror",
            "modulenotfounderror",
            "no real skill adapter registered",
            "real runner currently supports only",
            "failed to create environment",
        )
    )


def _looks_like_controller_failure(text: str, info: Any) -> bool:
    info_text = str(info).lower()
    combined = f"{text} {info_text}"
    return any(
        marker in combined
        for marker in (
            "planner",
            "controller",
            "control_mode",
            "trajectory",
            "rrt",
            "screw",
            "ik",
            "joint limit",
            "move_to_pose",
            "pose plan",
        )
    )


def _looks_like_task_outcome_failure(text: str, info: Dict[str, Any]) -> bool:
    outcome_text = f"{text} {info.get('final_info', '')}".lower()
    return any(
        marker in outcome_text
        for marker in (
            "not at goal",
            "not placed",
            "not inserted",
            "not pulled into workspace",
            "success': false",
            '"success": false',
            "is_obj_placed",
            "peg_head_pos_at_hole",
        )
    )
