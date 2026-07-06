"""Structured probing utilities for simulation-in-the-loop adapter repair.

The probe layer sits between a failed real ManiSkill run and the next LLM
adapter-generation prompt. It does not solve the task directly. Instead, it
turns a small bounded parameter sweep into standardized evidence: what was
varied, which metrics were measured, which trials were least destructive, and
what the next repair prompt should know.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from maniskill_backend.cases import FullMigrationCase


ProbeCase = Dict[str, Any]


@dataclass(frozen=True)
class ProbeSpec:
    """Static description of a bounded probe family."""

    probe_id: str
    case_id: str
    task_id: str
    robot_uid: str
    description: str
    parameter_grid: Mapping[str, Sequence[Any]]
    primary_metrics: Tuple[str, ...]
    success_keys: Tuple[str, ...]
    score_key: str
    output_slug: str
    no_success_guidance: str
    success_guidance: str


def get_probe_spec(case: FullMigrationCase, *, diagnosis: Optional[Mapping[str, Any]] = None) -> ProbeSpec:
    """Return the default structured probe for a migration case.

    The first implemented probe is the xarm6 PickCube close-envelope sweep.
    It is intentionally narrow: fixed XY alignment, a small Z-height sweep,
    close duration, close command, and settle duration.
    """

    reason = str((diagnosis or {}).get("reason") or "")
    if case.case_id == "case03_pick_cube_panda_to_xarm6":
        description = (
            "Fixed-XY xarm6 PickCube close-envelope sweep. Varies grasp height, "
            "close duration, close command, and settle duration while recording "
            "force-closure and displacement metrics."
        )
        if reason:
            description += f" Triggered by diagnosis reason: {reason}."
        return ProbeSpec(
            probe_id="pick_cube_xarm6_close_envelope",
            case_id=case.case_id,
            task_id=case.task_id,
            robot_uid=case.target_robot,
            description=description,
            parameter_grid={
                "grasp_z_offset": (0.004, 0.008, 0.012, 0.016),
                "close_steps": (12, 24),
                "close_command": (-0.6, -1.0),
                "settle_steps": (8, 16),
            },
            primary_metrics=(
                "is_grasping_after_close",
                "is_grasping_after_lift",
                "cube_disp_xy",
                "tcp_grasp_xy",
                "tcp_grasp_z",
                "cube_lift_delta_z",
            ),
            success_keys=("task_success", "is_grasping_after_lift", "is_grasping_after_close"),
            score_key="score",
            output_slug="pick_cube_xarm6_close_envelope",
            no_success_guidance=(
                "No probed close-envelope case formed a grasp. Do not keep "
                "guessing more horizontal candidates. Use the least destructive "
                "high-score case as evidence that fixed-XY close tuning is "
                "insufficient, then change the grasp primitive structure."
            ),
            success_guidance=(
                "At least one probe case formed a grasp. Prefer the successful "
                "case with the lowest cube displacement and reuse its close "
                "envelope in the next adapter."
            ),
        )

    if case.case_id == "case02_pull_cube_panda_to_xarm6":
        description = (
            "xarm6 PullCube contact-geometry sweep. Varies far-side contact "
            "offset, contact height, approach height, drag strength, down-bias, "
            "and stage count while recording contact reachability and cube "
            "progress toward the goal."
        )
        if reason:
            description += f" Triggered by diagnosis reason: {reason}."
        return ProbeSpec(
            probe_id="pull_cube_xarm6_contact_geometry",
            case_id=case.case_id,
            task_id=case.task_id,
            robot_uid=case.target_robot,
            description=description,
            parameter_grid={
                "contact_x_offset": (0.08, 0.10, 0.12, 0.14),
                "contact_z_offset": (0.010, 0.014),
                "approach_height": (0.06, 0.09),
                "drag_strength": (-0.6, -0.8),
                "down_bias": (-0.02,),
                "stages": (5,),
            },
            primary_metrics=(
                "task_success",
                "cube_goal_xy",
                "cube_goal_improvement",
                "cube_delta_x",
                "tcp_contact_xy",
                "tcp_contact_z",
                "tcp_cube_xy",
            ),
            success_keys=("task_success",),
            score_key="score",
            output_slug="pull_cube_xarm6_contact_geometry",
            no_success_guidance=(
                "No probed contact geometry solved PullCube. Use the best "
                "progress case to decide whether the bottleneck is far-side "
                "reachability, contact establishment, or drag progress; do not "
                "only add more episode steps."
            ),
            success_guidance=(
                "At least one contact probe solved PullCube. Prefer the "
                "successful case with the lowest cube_goal_xy and reuse its "
                "contact geometry/drag parameters in the next adapter."
            ),
        )

    raise KeyError(f"No structured probe is registered for case {case.case_id!r}.")


def probe_grid(spec: ProbeSpec, *, max_cases: int = 0) -> List[ProbeCase]:
    """Expand a probe parameter grid into ordered dictionaries."""

    keys = list(spec.parameter_grid)
    rows: List[ProbeCase] = []
    for index, values in enumerate(product(*(spec.parameter_grid[key] for key in keys)), start=1):
        if max_cases and index > max_cases:
            break
        row = {key: value for key, value in zip(keys, values)}
        row["case_index"] = index
        rows.append(row)
    return rows


def is_successful_probe_case(case: Mapping[str, Any], spec: ProbeSpec) -> bool:
    return any(bool(case.get(key)) for key in spec.success_keys)


def _case_score(case: Mapping[str, Any], spec: ProbeSpec) -> float:
    raw_score = case.get(spec.score_key)
    if raw_score is None:
        # Fall back to a conservative score for dry-run or externally imported
        # rows that may not have been generated by the task-specific script.
        success_bonus = 100.0 if is_successful_probe_case(case, spec) else 0.0
        disp_penalty = float(case.get("cube_disp_xy") or 0.0) * 1000.0
        tcp_xy_penalty = float(case.get("tcp_grasp_xy") or 0.0) * 200.0
        tcp_z_penalty = float(case.get("tcp_grasp_z") or 0.0) * 200.0
        return success_bonus - disp_penalty - tcp_xy_penalty - tcp_z_penalty
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 0.0


def rank_probe_cases(results: Sequence[Mapping[str, Any]], spec: ProbeSpec) -> List[ProbeCase]:
    """Return probe rows sorted from most promising to least promising."""

    normalized = [dict(item) for item in results]
    return sorted(normalized, key=lambda item: _case_score(item, spec), reverse=True)


def _parameter_key(case: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(case.get(key) for key in keys)


def _numeric_grid_step(values: Sequence[Any]) -> float:
    numeric = sorted({float(value) for value in values})
    diffs = [right - left for left, right in zip(numeric, numeric[1:]) if right > left]
    return min(diffs) if diffs else 1.0


def _local_values_for_parameter(key: str, anchor_value: Any, grid_values: Sequence[Any]) -> List[Any]:
    """Return small local refinements around one high-scoring parameter."""

    if not isinstance(anchor_value, (int, float)):
        return [anchor_value]
    step = _numeric_grid_step(grid_values)
    is_integer = all(isinstance(value, int) and not isinstance(value, bool) for value in grid_values)
    if is_integer:
        delta = max(1, int(round(step / 2.0)))
        lower = max(1, int(anchor_value) - delta)
        upper = int(anchor_value) + delta
        return sorted({lower, int(anchor_value), upper})

    anchor = float(anchor_value)
    delta = step / 2.0
    lower_bound = min(float(value) for value in grid_values) - delta
    upper_bound = max(float(value) for value in grid_values) + delta
    if "z" in key or "offset" in key or "height" in key:
        lower_bound = max(0.0, lower_bound)
    if "command" in key or "gripper" in key or "close" in key or "strength" in key or "bias" in key:
        lower_bound = max(-1.0, lower_bound)
        upper_bound = min(1.0, upper_bound)
    values = [
        max(lower_bound, min(upper_bound, anchor - delta)),
        anchor,
        max(lower_bound, min(upper_bound, anchor + delta)),
    ]
    return sorted({round(value, 5) for value in values})


def suggest_next_probe_cases(
    spec: ProbeSpec,
    results: Sequence[Mapping[str, Any]],
    *,
    budget: int = 8,
    anchor_count: int = 3,
) -> List[ProbeCase]:
    """Suggest a small score-guided local search plan from previous probe rows.

    This is deliberately lightweight and deterministic: rank previous rows by
    score, take the best few anchors, and perturb one parameter at a time by a
    half-grid step. It is a learning-guided optimizer in the narrow sense that
    the next experiments are selected from measured rewards, not hand-written
    guesses or another full Cartesian grid.
    """

    if budget <= 0:
        return []
    grid_keys = list(spec.parameter_grid)
    ranked = rank_probe_cases(results, spec)
    if not ranked:
        rows = probe_grid(spec, max_cases=budget)
        for row in rows:
            row["suggestion_reason"] = "initial_grid_no_previous_results"
        return rows

    tried = {_parameter_key(item, grid_keys) for item in results}
    suggestions: List[ProbeCase] = []
    seen: set[Tuple[Any, ...]] = set()
    anchors = ranked[: max(1, anchor_count)]
    for anchor_rank, anchor in enumerate(anchors, start=1):
        base = {key: anchor.get(key) for key in grid_keys}
        for key in grid_keys:
            for value in _local_values_for_parameter(key, base[key], spec.parameter_grid[key]):
                candidate = dict(base)
                candidate[key] = value
                key_tuple = _parameter_key(candidate, grid_keys)
                if key_tuple in tried or key_tuple in seen:
                    continue
                candidate["source_anchor_rank"] = anchor_rank
                candidate["source_anchor_score"] = round(_case_score(anchor, spec), 5)
                candidate["suggestion_reason"] = f"score_guided_local_refinement_of_{key}"
                seen.add(key_tuple)
                suggestions.append(candidate)
                if len(suggestions) >= budget:
                    return suggestions

    if len(suggestions) < budget:
        for row in probe_grid(spec):
            key_tuple = _parameter_key(row, grid_keys)
            if key_tuple in tried or key_tuple in seen:
                continue
            row["suggestion_reason"] = "untried_grid_fallback"
            seen.add(key_tuple)
            suggestions.append(row)
            if len(suggestions) >= budget:
                break
    return suggestions


def build_probe_feedback(payload: Mapping[str, Any], *, top_k: int = 8) -> str:
    """Build compact text that can be inserted into a future LLM prompt."""

    best = payload.get("best_probe_case") or {}
    top_cases = list(payload.get("top_probe_cases") or [])[:top_k]
    suggestions = list(payload.get("next_probe_suggestions") or [])[:top_k]
    metrics = ", ".join(payload.get("primary_metrics") or [])
    lines = [
        "Structured probe feedback.",
        f"probe_id={payload.get('probe_id')}",
        f"case_id={payload.get('case_id')}",
        f"task_id={payload.get('task_id')}",
        f"robot_uid={payload.get('robot_uid')}",
        f"total_probe_cases={payload.get('num_cases')}",
        f"successful_probe_cases={payload.get('num_success')}",
    ]
    if metrics:
        lines.append(f"primary_metrics={metrics}")

    if best:
        lines.extend(["", "best_probe_case:"])
        for key in payload.get("grid_keys") or []:
            lines.append(f"  {key}={best.get(key)}")
        for key in payload.get("primary_metrics") or []:
            lines.append(f"  {key}={best.get(key)}")
        if "score" in best:
            lines.append(f"  score={best.get('score')}")

    lines.extend(["", "top_probe_cases:"])
    for item in top_cases:
        params = ", ".join(f"{key}={item.get(key)}" for key in payload.get("grid_keys") or [])
        metric_text = ", ".join(f"{key}={item.get(key)}" for key in payload.get("primary_metrics") or [])
        score = item.get("score")
        if score is not None:
            lines.append(f"- {params}; {metric_text}; score={score}")
        else:
            lines.append(f"- {params}; {metric_text}")

    if suggestions:
        lines.extend(["", "next_probe_suggestions:"])
        for item in suggestions:
            params = ", ".join(f"{key}={item.get(key)}" for key in payload.get("grid_keys") or [])
            reason = item.get("suggestion_reason") or "score_guided_suggestion"
            lines.append(f"- {params}; reason={reason}")

    recommendation = str(payload.get("recommendation") or "").strip()
    if recommendation:
        lines.extend(["", "probe_repair_guidance:", recommendation])
    lines.extend(
        [
            "",
            "Use this as measured physical evidence, not as a success oracle.",
            "Keep the high-level program unchanged; only repair the target-side adapter.",
        ]
    )
    return "\n".join(lines)


def summarize_probe_results(
    spec: ProbeSpec,
    results: Sequence[Mapping[str, Any]],
    *,
    diagnosis: Optional[Mapping[str, Any]] = None,
    top_k: int = 8,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Normalize task-specific probe rows into a stable result schema."""

    ranked = rank_probe_cases(results, spec)
    successes = [item for item in ranked if is_successful_probe_case(item, spec)]
    best = ranked[0] if ranked else None
    next_suggestions = suggest_next_probe_cases(spec, ranked, budget=8) if results else []
    payload: Dict[str, Any] = {
        "schema": "structured_probe_result.v1",
        "probe_id": spec.probe_id,
        "case_id": spec.case_id,
        "task_id": spec.task_id,
        "robot_uid": spec.robot_uid,
        "description": spec.description,
        "dry_run": bool(dry_run),
        "diagnosis": dict(diagnosis or {}),
        "parameter_grid": {key: list(value) for key, value in spec.parameter_grid.items()},
        "grid_keys": list(spec.parameter_grid),
        "primary_metrics": list(spec.primary_metrics),
        "success_keys": list(spec.success_keys),
        "score_key": spec.score_key,
        "num_cases": len(results),
        "num_success": len(successes),
        "best_probe_case": best,
        "top_probe_cases": ranked[:top_k],
        "all_probe_cases": ranked,
        "next_probe_suggestions": next_suggestions,
        "recommendation": spec.success_guidance if successes else spec.no_success_guidance,
    }
    payload["prompt_feedback"] = build_probe_feedback(payload, top_k=top_k)
    return payload


def write_probe_outputs(output_dir: Path, payload: Mapping[str, Any]) -> Dict[str, str]:
    """Write JSON, Markdown, and prompt-text artifacts for a probe payload."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(payload.get("probe_id") or "structured_probe")
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    prompt_path = output_dir / f"{stem}_prompt.txt"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_probe_markdown(payload), encoding="utf-8")
    prompt_path.write_text(str(payload.get("prompt_feedback") or "") + "\n", encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "prompt_feedback": str(prompt_path),
    }


def _probe_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# Structured Probe: {payload.get('probe_id')}",
        "",
        f"- case_id: `{payload.get('case_id')}`",
        f"- task_id: `{payload.get('task_id')}`",
        f"- robot_uid: `{payload.get('robot_uid')}`",
        f"- total cases: `{payload.get('num_cases')}`",
        f"- successful cases: `{payload.get('num_success')}`",
        "",
        "## Recommendation",
        "",
        str(payload.get("recommendation") or ""),
        "",
        "## Top Cases",
        "",
    ]
    grid_keys = list(payload.get("grid_keys") or [])
    metrics = list(payload.get("primary_metrics") or [])
    header = ["rank", "score", *grid_keys, *metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---:" if index < 2 else "---" for index, _ in enumerate(header)) + "|")
    for rank, item in enumerate(payload.get("top_probe_cases") or [], start=1):
        row = [str(rank), str(item.get("score", ""))]
        row.extend(str(item.get(key, "")) for key in grid_keys)
        row.extend(str(item.get(key, "")) for key in metrics)
        lines.append("| " + " | ".join(row) + " |")
    suggestions = list(payload.get("next_probe_suggestions") or [])
    if suggestions:
        lines.extend(["", "## Next Probe Suggestions", ""])
        suggestion_header = ["rank", *grid_keys, "reason"]
        lines.append("| " + " | ".join(suggestion_header) + " |")
        lines.append("|" + "|".join("---:" if index == 0 else "---" for index, _ in enumerate(suggestion_header)) + "|")
        for rank, item in enumerate(suggestions, start=1):
            row = [str(rank)]
            row.extend(str(item.get(key, "")) for key in grid_keys)
            row.append(str(item.get("suggestion_reason") or ""))
            lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", "## Prompt Feedback", "", "```text", str(payload.get("prompt_feedback") or ""), "```", ""])
    return "\n".join(lines)
