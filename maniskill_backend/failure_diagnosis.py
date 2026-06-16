"""Structured failure diagnosis for real ManiSkill migration trials.

The lightweight classifiers in ``evaluation.py`` keep compatibility with the
benchmark result schema. This module adds a more research-facing diagnosis:
which migration layer failed, what evidence supports that decision, and what
repair direction should be tried next.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping


NUMERIC_FIELD_RE = re.compile(
    r"\b(?P<key>tcp_grasp_xy|tcp_grasp_z|cube_disp_xy|tcp_cube_xy|cube_goal_xy|tcp_stage_error_norm)"
    r"\s*=\s*(?P<value>unknown|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def diagnose_failure(
    *,
    task_id: str,
    success: bool,
    code_ok: bool = True,
    message: str = "",
    failure_type: str = "",
    failure_layer: str = "",
    execution_log: Any = None,
    final_info: Any = None,
    runtime_diagnostics: Mapping[str, Any] | None = None,
    initial_runtime_diagnostics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a structured diagnosis for a trial result.

    The output is intentionally plain JSON so it can be written into JSONL
    results, Markdown tables, and LLM retry prompts.
    """

    if success:
        return _diagnosis(
            layer="success",
            reason="task_succeeded",
            repair_hint="No repair needed.",
            confidence=1.0,
            evidence={"message": message or "ret_val=True"},
        )

    context = _context_text(message, failure_type, failure_layer, execution_log, final_info)
    if not code_ok or _looks_like_program_failure(context):
        return _diagnosis(
            layer="program",
            reason="program_or_api_mismatch",
            repair_hint="Fix the high-level program/API call before changing robot motion logic.",
            confidence=0.95,
            evidence={"message": message, "failure_type": failure_type},
        )

    if _looks_like_runtime_setup_failure(context):
        return _diagnosis(
            layer="runtime_setup",
            reason="environment_or_import_setup_failure",
            repair_hint="Fix environment setup, imports, renderer, or adapter loading before evaluating migration.",
            confidence=0.95,
            evidence={"message": message, "failure_type": failure_type},
        )

    if _looks_like_controller_failure(context):
        return _diagnosis(
            layer="controller_primitive",
            reason="controller_or_action_interface_failure",
            repair_hint="Check control_mode, action_space layout, IK/planner status, and action mapping.",
            confidence=0.9,
            evidence={"message": message, "failure_type": failure_type},
        )

    if _looks_like_infeasible(context):
        return _diagnosis(
            layer="infeasibility",
            reason="declared_or_detected_infeasible_condition",
            repair_hint="Add an explicit infeasibility/refusal path or change the task setup.",
            confidence=0.85,
            evidence={"message": message, "failure_type": failure_type},
        )

    if task_id == "pull_cube":
        return _diagnose_pull_cube(
            message=message,
            failure_type=failure_type,
            execution_log=execution_log,
            runtime_diagnostics=runtime_diagnostics or {},
            initial_runtime_diagnostics=initial_runtime_diagnostics or {},
        )
    if task_id == "pick_cube":
        return _diagnose_pick_cube(
            message=message,
            failure_type=failure_type,
            execution_log=execution_log,
        )

    return _diagnosis(
        layer=failure_layer or "unknown",
        reason="unclassified_failure",
        repair_hint="Add task-specific runtime diagnostics for this failure mode.",
        confidence=0.2,
        evidence={"message": message, "failure_type": failure_type},
    )


def _diagnose_pull_cube(
    *,
    message: str,
    failure_type: str,
    execution_log: Any,
    runtime_diagnostics: Mapping[str, Any],
    initial_runtime_diagnostics: Mapping[str, Any],
) -> Dict[str, Any]:
    stage = str(runtime_diagnostics.get("stage") or _stage_from_text(message) or "final")
    stage_error = _float_or_none(runtime_diagnostics.get("tcp_stage_error_norm"))
    contact_error = _float_or_none(runtime_diagnostics.get("tcp_contact_error_norm"))
    tcp_cube_xy = _float_or_none(runtime_diagnostics.get("tcp_cube_xy"))
    cube_goal_xy = _float_or_none(runtime_diagnostics.get("cube_goal_xy"))
    elapsed = _elapsed_steps_from_log(execution_log)

    evidence = _compact_evidence(
        {
            "stage": stage,
            "tcp_stage_error_norm": stage_error,
            "tcp_contact_error_norm": contact_error,
            "tcp_cube_xy": tcp_cube_xy,
            "cube_goal_xy": cube_goal_xy,
            "tcp_stage_error_xyz": runtime_diagnostics.get("tcp_stage_error_xyz"),
            "stage_target_pos": runtime_diagnostics.get("stage_target_pos"),
            "tcp_pos": runtime_diagnostics.get("tcp_pos"),
            "cube_pos": runtime_diagnostics.get("cube_pos"),
            "initial_tcp_cube_xy": initial_runtime_diagnostics.get("tcp_cube_xy"),
            "message": message,
        }
    )

    if stage in {"approach", "descent", "contact"} and (stage_error or 0.0) >= 0.08:
        return _diagnosis(
            layer="contact_geometry",
            reason="contact_side_reachability_failure",
            repair_hint=(
                "Do not only add more steps. Choose contact pose adaptively from current cube/TCP geometry, "
                "or run a reachability precheck before descent."
            ),
            confidence=0.9,
            evidence=evidence,
        )
    if (tcp_cube_xy or 0.0) >= 0.10 and (cube_goal_xy or 0.0) >= 0.05:
        return _diagnosis(
            layer="contact_geometry",
            reason="tcp_never_established_effective_contact",
            repair_hint="Move the target contact point closer to the reachable side of the cube before dragging.",
            confidence=0.82,
            evidence=evidence,
        )
    if (tcp_cube_xy or 9.0) <= 0.06 and (cube_goal_xy or 0.0) >= 0.05:
        return _diagnosis(
            layer="contact_geometry",
            reason="contact_established_but_drag_progress_insufficient",
            repair_hint="Change drag direction, down-force, or contact surface; the TCP is near the cube but progress is weak.",
            confidence=0.78,
            evidence=evidence,
        )
    if "episode ended" in message.lower() or elapsed is not None:
        return _diagnosis(
            layer="infeasibility",
            reason="episode_budget_exhausted_before_success",
            repair_hint="Add progress guards and stop early when contact target is not reachable; do not spend the full episode on a fixed pose.",
            confidence=0.65,
            evidence=evidence,
        )
    return _diagnosis(
        layer="skill_adapter",
        reason="pull_skill_execution_failed",
        repair_hint="Inspect contact candidate selection and drag progress guards.",
        confidence=0.45,
        evidence=evidence,
    )


def _diagnose_pick_cube(
    *,
    message: str,
    failure_type: str,
    execution_log: Any,
) -> Dict[str, Any]:
    metrics = _parse_numeric_fields(message)
    lowered = message.lower()
    is_grasping = _parse_bool_field(lowered, "is_grasping")
    tcp_xy = _float_or_none(metrics.get("tcp_grasp_xy"))
    tcp_z = _float_or_none(metrics.get("tcp_grasp_z"))
    cube_disp = _float_or_none(metrics.get("cube_disp_xy"))
    evidence = _compact_evidence(
        {
            "tcp_grasp_xy": tcp_xy,
            "tcp_grasp_z": tcp_z,
            "cube_disp_xy": cube_disp,
            "is_grasping": is_grasping,
            "message": message,
            "failed_api": _first_failed_api(execution_log),
        }
    )

    if is_grasping is True:
        return _diagnosis(
            layer="skill_adapter",
            reason="grasp_detected_but_not_preserved_or_placed",
            repair_hint="If self._is_grasping('cube') is true, preserve the grasp and transition to place instead of reporting grasp failure.",
            confidence=0.9,
            evidence=evidence,
        )
    if tcp_z is not None and tcp_z >= 0.03:
        return _diagnosis(
            layer="contact_geometry",
            reason="approach_descent_alignment_failure",
            repair_hint="Continue bounded vertical descent/refinement before close; do not diagnose gripper force while TCP is still high.",
            confidence=0.9,
            evidence=evidence,
        )
    if tcp_xy is not None and tcp_xy >= 0.02:
        return _diagnosis(
            layer="contact_geometry",
            reason="xy_alignment_failure_before_close",
            repair_hint="Finish XY alignment above the cube before the vertical close phase.",
            confidence=0.82,
            evidence=evidence,
        )
    if cube_disp is not None and cube_disp >= 0.03:
        return _diagnosis(
            layer="contact_geometry",
            reason="gripper_envelope_side_push",
            repair_hint="Keep XY fixed and change close envelope: higher/lower close height, slower close, or smaller close command; abort on displacement.",
            confidence=0.9,
            evidence=evidence,
        )
    if (
        tcp_xy is not None
        and tcp_z is not None
        and cube_disp is not None
        and tcp_xy <= 0.005
        and tcp_z <= 0.005
        and cube_disp <= 0.01
        and is_grasping is False
    ):
        return _diagnosis(
            layer="contact_geometry",
            reason="good_alignment_no_displacement_no_grasp",
            repair_hint="Do not add horizontal candidates. Change gripper close timing, close command, or grasp height envelope.",
            confidence=0.92,
            evidence=evidence,
        )
    if _missing_close_time_diagnostics(message):
        return _diagnosis(
            layer="skill_adapter",
            reason="insufficient_close_time_diagnostics",
            repair_hint="Every failed grasp message must include numeric tcp_grasp_xy, tcp_grasp_z, and cube_disp_xy.",
            confidence=0.8,
            evidence=evidence,
        )
    return _diagnosis(
        layer="skill_adapter",
        reason="pick_skill_execution_failed",
        repair_hint="Add close-time diagnostics or run the fixed-XY grasp probe to identify the failed subphase.",
        confidence=0.45,
        evidence=evidence,
    )


def _diagnosis(
    *,
    layer: str,
    reason: str,
    repair_hint: str,
    confidence: float,
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "layer": layer,
        "reason": reason,
        "repair_hint": repair_hint,
        "confidence": round(float(confidence), 3),
        "evidence": _compact_evidence(evidence),
    }


def _compact_evidence(values: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value not in (None, "", [], {})}


def _context_text(*pieces: Any) -> str:
    return " ".join(_flatten_text(piece) for piece in pieces).lower()


def _flatten_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key}={_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


def _looks_like_program_failure(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "attributeerror",
            "nameerror",
            "syntaxerror",
            "api mismatch",
            "invalid generated code",
            "forbidden text",
            "object has no attribute",
        )
    )


def _looks_like_runtime_setup_failure(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "runtime_setup",
            "modulenotfounderror",
            "importerror",
            "vulkan",
            "cuda",
            "renderer",
            "create window failed",
            "no real skill adapter registered",
        )
    )


def _looks_like_controller_failure(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "action_space",
            "action dim",
            "control_mode",
            "controller",
            "planner",
            "rrt",
            "screw",
            "ik",
            "joint limit",
            "move_to_pose",
        )
    )


def _looks_like_infeasible(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "infeasible",
            "not feasible",
            "outside workspace",
            "unreachable",
            "impossible",
            "cannot perform",
            "beyond the capability",
        )
    )


def _stage_from_text(text: str) -> str:
    lower = str(text or "").lower()
    for stage in ("approach", "descent", "contact", "drag", "settle"):
        if stage in lower:
            return stage
    return ""


def _elapsed_steps_from_log(execution_log: Any) -> int | None:
    for event in _iter_dicts(execution_log):
        args = event.get("args")
        if isinstance(args, Mapping):
            raw = args.get("elapsed_steps")
            if isinstance(raw, (int, float)):
                return int(raw)
    return None


def _iter_dicts(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                yield item


def _parse_numeric_fields(message: str) -> Dict[str, float | str]:
    values: Dict[str, float | str] = {}
    for match in NUMERIC_FIELD_RE.finditer(str(message or "")):
        key = match.group("key")
        raw = match.group("value")
        values[key] = raw if raw == "unknown" else float(raw)
    displaced = re.search(r"cube displaced(?: laterally)?(?: by)?\s*(?P<value>[-+]?\d+(?:\.\d+)?)m", str(message or "").lower())
    if displaced and "cube_disp_xy" not in values:
        values["cube_disp_xy"] = float(displaced.group("value"))
    return values


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "unknown":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool_field(text: str, key: str) -> bool | None:
    match = re.search(rf"\b{re.escape(key)}\s*=\s*(true|false)\b", text)
    if not match:
        return None
    return match.group(1) == "true"


def _first_failed_api(execution_log: Any) -> str:
    for event in _iter_dicts(execution_log):
        if event.get("ok") is False:
            return str(event.get("api") or "")
    return ""


def _missing_close_time_diagnostics(message: str) -> bool:
    metrics = _parse_numeric_fields(message)
    return not all(key in metrics and metrics[key] != "unknown" for key in ("tcp_grasp_xy", "tcp_grasp_z", "cube_disp_xy"))
