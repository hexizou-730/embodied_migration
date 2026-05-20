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
    "insertion failure",
    "insertion speed failure",
    "tool-use ordering failure",
    "impossible-task refusal failure",
    "invalid generated code",
    "execution failure",
    "unknown failure",
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

    if "unreachable" in text or "reach" in text:
        return "reachability failure"
    if "grasp" in text or "gripper" in text or "force" in text:
        return "gripper/force failure"
    if "tool" in text or "hook" in text or "ordering" in text:
        return "tool-use ordering failure"
    if "align" in text or "misalign" in text or "pose error" in text:
        return "alignment failure"
    if "stack" in text or "on cubeb" in text or "cubea_static" in text or "stability" in text:
        return "placement stability failure"
    if "not inserted" in text or "insertion failure" in text or "peg_head_pos_at_hole" in text:
        return "insertion failure"
    if "speed" in text or "too fast" in text:
        return "insertion speed failure"
    if "not placed" in text or "not at goal" in text or "place" in text:
        return "execution failure"
    if (
        "impossible" in text
        or "unsafe" in text
        or "refuse" in text
        or "sub-centimeter" in text
        or "sub-cm" in text
        or "requires sub" in text
        or "outside workspace" in text
        or "outside fixed-base" in text
        or "not feasible" in text
        or "cannot perform" in text
        or "exceeds the robot" in text
        or "beyond the capability" in text
    ):
        return "impossible-task refusal failure"

    ret_val_text = str(info.get("ret_val", "")).lower()
    if "failure" in ret_val_text:
        log = info.get("execution_log") or []
        skill_failed = any(
            isinstance(event, dict) and event.get("ok") is False for event in log
        )
        if not skill_failed:
            return "impossible-task refusal failure"

    return "unknown failure"
