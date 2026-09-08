"""Constraint-guided active probe selection.

The selector narrows a structured probe to parameters relevant to the selected
counterexample. Previous measured probe rows are used as local-search anchors;
otherwise the selector chooses a small one-factor-at-a-time design around the
default probe center.
"""

from __future__ import annotations

import json
from itertools import product
from math import exp, sqrt
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from maniskill_backend.cases import FullMigrationCase
from maniskill_backend.structured_probe import (
    ProbeSpec,
    rank_probe_cases,
    suggest_next_probe_cases,
)


REASON_PARAMETER_FOCUS: Dict[str, Sequence[str]] = {
    "contact_side_reachability_failure": (
        "base_speed",
        "base_steps",
        "contact_x_offset",
        "contact_z_offset",
        "approach_height",
    ),
    "tcp_never_established_effective_contact": (
        "base_speed",
        "base_steps",
        "contact_x_offset",
        "contact_z_offset",
        "approach_height",
        "down_bias",
    ),
    "contact_established_but_drag_progress_insufficient": (
        "drag_strength",
        "down_bias",
        "stages",
    ),
    "episode_budget_exhausted_before_success": (
        "base_speed",
        "base_steps",
        "contact_x_offset",
        "approach_height",
        "stages",
    ),
    "approach_descent_alignment_failure": (
        "base_speed",
        "base_steps",
        "grasp_z_offset",
    ),
    "xy_alignment_failure_before_close": (
        "base_speed",
        "base_steps",
        "grasp_z_offset",
    ),
    "gripper_envelope_side_push": (
        "grasp_z_offset",
        "close_steps",
        "close_command",
        "settle_steps",
    ),
    "good_alignment_no_displacement_no_grasp": (
        "grasp_z_offset",
        "close_steps",
        "close_command",
        "settle_steps",
    ),
}


CONSTRAINT_PARAMETER_FOCUS: Dict[str, Sequence[str]] = {
    "contact_pose_reachable_before_descent": (
        "base_speed",
        "base_steps",
        "contact_x_offset",
        "contact_z_offset",
        "approach_height",
    ),
    "contact_side_selected_from_runtime_geometry": (
        "contact_x_offset",
        "contact_z_offset",
        "approach_height",
    ),
    "tcp_object_contact_established_before_drag": (
        "contact_x_offset",
        "contact_z_offset",
        "approach_height",
        "down_bias",
    ),
    "object_goal_error_decreases_during_drag": ("drag_strength", "stages", "down_bias"),
    "contact_preserved_during_drag": ("contact_z_offset", "down_bias", "drag_strength"),
    "bounded_phase_budget": ("base_steps", "approach_height", "stages"),
    "early_stop_on_no_progress": ("drag_strength", "stages"),
    "tcp_aligned_before_gripper_close": ("base_speed", "base_steps", "grasp_z_offset"),
    "tcp_xy_aligned_before_gripper_close": ("base_speed", "base_steps", "grasp_z_offset"),
    "cube_displacement_bounded_during_close": (
        "grasp_z_offset",
        "close_steps",
        "close_command",
    ),
    "gripper_envelope_matches_object": (
        "grasp_z_offset",
        "close_steps",
        "close_command",
        "settle_steps",
    ),
    "gripper_envelope_forms_force_closure": (
        "grasp_z_offset",
        "close_steps",
        "close_command",
        "settle_steps",
    ),
    "grasp_preserved_until_place": ("close_steps", "close_command", "settle_steps"),
}


def _middle(values: Sequence[Any]) -> Any:
    return list(values)[len(values) // 2]


def _baseline(spec: ProbeSpec) -> Dict[str, Any]:
    return {key: _middle(values) for key, values in spec.parameter_grid.items()}


def _one_factor_candidates(
    spec: ProbeSpec,
    focused: Sequence[str],
    *,
    budget: int,
) -> List[Dict[str, Any]]:
    baseline = _baseline(spec)
    candidates: List[Dict[str, Any]] = []
    for key in focused:
        if key not in spec.parameter_grid:
            continue
        for value in spec.parameter_grid[key]:
            candidate = dict(baseline)
            candidate[key] = value
            candidate["active_parameter"] = key
            candidate["suggestion_reason"] = f"constraint_guided_variation_of_{key}"
            candidates.append(candidate)
            if len(candidates) >= budget:
                return candidates
    if not candidates:
        keys = list(spec.parameter_grid)
        for values in product(*(spec.parameter_grid[key] for key in keys)):
            candidate = {key: value for key, value in zip(keys, values)}
            candidate["suggestion_reason"] = "bounded_grid_fallback"
            candidates.append(candidate)
            if len(candidates) >= budget:
                break
    return candidates


def _deduplicate(candidates: Sequence[Mapping[str, Any]], spec: ProbeSpec) -> List[Dict[str, Any]]:
    keys = list(spec.parameter_grid)
    seen = set()
    result = []
    for item in candidates:
        signature = tuple(item.get(key) for key in keys)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(dict(item))
    return result


def _numeric_value(key: str, value: Any, spec: ProbeSpec) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    values = list(spec.parameter_grid[key])
    try:
        return float(values.index(value))
    except ValueError:
        return 0.0


def _normalized_vector(item: Mapping[str, Any], spec: ProbeSpec) -> tuple[float, ...]:
    vector = []
    for key, values in spec.parameter_grid.items():
        numeric = [_numeric_value(key, value, spec) for value in values]
        low, high = min(numeric), max(numeric)
        value = _numeric_value(key, item.get(key), spec)
        vector.append(0.0 if high == low else (value - low) / (high - low))
    return tuple(vector)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _focused_grid_candidates(spec: ProbeSpec, focused: Sequence[str]) -> List[Dict[str, Any]]:
    baseline = _baseline(spec)
    active = [key for key in focused if key in spec.parameter_grid]
    if not active:
        active = list(spec.parameter_grid)
    candidates = []
    for values in product(*(spec.parameter_grid[key] for key in active)):
        candidate = dict(baseline)
        candidate.update({key: value for key, value in zip(active, values)})
        candidates.append(candidate)
    return candidates


def _parameter_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    spec: ProbeSpec,
) -> Dict[str, float]:
    """Estimate one-dimensional score sensitivity from measured probe rows."""

    raw: Dict[str, float] = {}
    for key in spec.parameter_grid:
        groups: Dict[Any, List[float]] = {}
        for row in rows:
            if key not in row:
                continue
            try:
                score = float(row.get(spec.score_key))
            except (TypeError, ValueError):
                continue
            groups.setdefault(row[key], []).append(score)
        means = [sum(values) / len(values) for values in groups.values() if values]
        raw[key] = max(means) - min(means) if len(means) >= 2 else 0.0
    scale = max(raw.values(), default=0.0)
    return {
        key: round(value / scale, 5) if scale > 1e-12 else 0.0
        for key, value in raw.items()
    }


def _constraint_focused_parameters(
    spec: ProbeSpec,
    *,
    reason: str,
    violated_constraints: Sequence[str],
) -> List[str]:
    candidates: List[str] = []
    for constraint in violated_constraints:
        candidates.extend(CONSTRAINT_PARAMETER_FOCUS.get(str(constraint), ()))
    candidates.extend(REASON_PARAMETER_FOCUS.get(reason, ()))
    if not candidates:
        candidates.extend(spec.parameter_grid)
    return list(dict.fromkeys(key for key in candidates if key in spec.parameter_grid))


def _kernel_ucb_candidates(
    spec: ProbeSpec,
    prior_rows: Sequence[Mapping[str, Any]],
    focused: Sequence[str],
    *,
    budget: int,
    beta: float = 0.75,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select untried probes with a deterministic kernel-UCB acquisition."""

    keys = list(spec.parameter_grid)
    tried = {tuple(item.get(key) for key in keys) for item in prior_rows}
    local = suggest_next_probe_cases(
        spec,
        rank_probe_cases(prior_rows, spec),
        budget=max(4 * budget, budget),
    )
    local = [
        item
        for item in local
        if not item.get("suggestion_reason")
        or any(str(item.get("suggestion_reason")).endswith(f"_{key}") for key in focused)
    ]
    pool = _deduplicate([*_focused_grid_candidates(spec, focused), *local], spec)
    pool = [item for item in pool if tuple(item.get(key) for key in keys) not in tried]
    if len(pool) < budget:
        pool = _deduplicate([*pool, *_focused_grid_candidates(spec, list(spec.parameter_grid))], spec)
        pool = [item for item in pool if tuple(item.get(key) for key in keys) not in tried]

    observations = []
    for row in prior_rows:
        try:
            score = float(row.get(spec.score_key))
        except (TypeError, ValueError):
            continue
        observations.append((_normalized_vector(row, spec), score))
    if not observations:
        return _one_factor_candidates(spec, focused, budget=budget), {
            "name": "constraint_guided_initial_design",
            "observations": 0,
        }

    scores = [score for _, score in observations]
    score_low, score_high = min(scores), max(scores)
    global_mean = sum(scores) / len(scores)
    dimension_scale = sqrt(max(1, len(keys)))
    bandwidth = 0.35
    scored: List[Dict[str, Any]] = []
    for item in pool:
        vector = _normalized_vector(item, spec)
        distances = [_distance(vector, observed) for observed, _ in observations]
        weights = [exp(-(distance ** 2) / (2.0 * bandwidth ** 2)) for distance in distances]
        weight_sum = sum(weights)
        predicted = (
            sum(weight * score for weight, (_, score) in zip(weights, observations)) / weight_sum
            if weight_sum > 1e-12
            else global_mean
        )
        predicted_unit = (
            0.5 if score_high == score_low else (predicted - score_low) / (score_high - score_low)
        )
        uncertainty = min(1.0, min(distances) / dimension_scale)
        candidate = dict(item)
        candidate["surrogate_predicted_score"] = round(predicted, 5)
        candidate["surrogate_uncertainty"] = round(uncertainty, 5)
        candidate["acquisition_score"] = round(predicted_unit + beta * uncertainty, 5)
        candidate["suggestion_reason"] = "kernel_ucb_exploitation_exploration"
        scored.append(candidate)

    selected: List[Dict[str, Any]] = []
    while scored and len(selected) < budget:
        def batch_score(item: Mapping[str, Any]) -> tuple[float, str]:
            if not selected:
                diversity = 1.0
            else:
                vector = _normalized_vector(item, spec)
                diversity = min(
                    _distance(vector, _normalized_vector(chosen, spec)) for chosen in selected
                ) / dimension_scale
            value = float(item.get("acquisition_score") or 0.0) + 0.2 * min(1.0, diversity)
            signature = repr(tuple(item.get(key) for key in keys))
            return value, signature

        best = max(scored, key=batch_score)
        best["selection_rank"] = len(selected) + 1
        selected.append(best)
        scored.remove(best)
    return selected, {
        "name": "kernel_ucb",
        "beta": beta,
        "bandwidth": bandwidth,
        "observations": len(observations),
        "candidate_pool_size": len(pool),
        "acquisition": "normalized_kernel_mean + beta * nearest_observation_distance",
        "batch_diversity_bonus": 0.2,
    }


def select_active_probe_plan(
    case: FullMigrationCase,
    spec: ProbeSpec,
    counterexample: Mapping[str, Any],
    *,
    previous_probe: Mapping[str, Any] | None = None,
    budget: int = 8,
) -> Dict[str, Any]:
    reason = str(
        counterexample.get("failure_reason")
        or counterexample.get("reason")
        or (counterexample.get("failure_diagnosis") or {}).get("reason")
        or ""
    )
    violated_constraints = list(counterexample.get("violated_constraints") or [])
    focused = _constraint_focused_parameters(
        spec,
        reason=reason,
        violated_constraints=violated_constraints,
    )
    prior_rows = list(
        (previous_probe or {}).get("all_probe_cases")
        or (previous_probe or {}).get("results")
        or []
    )
    sensitivity = _parameter_sensitivity(prior_rows, spec)
    focus_order = {key: index for index, key in enumerate(focused)}
    focused.sort(key=lambda key: (-sensitivity.get(key, 0.0), focus_order[key]))
    mode = "constraint_guided_initial_design"
    optimizer: Dict[str, Any] = {
        "name": "constraint_guided_initial_design",
        "observations": 0,
    }
    if prior_rows:
        candidates, optimizer = _kernel_ucb_candidates(
            spec,
            prior_rows,
            focused,
            budget=budget,
        )
        mode = "measurement_guided_kernel_ucb"
    else:
        candidates = _one_factor_candidates(spec, focused, budget=budget)
    candidates = _deduplicate(candidates, spec)[: max(0, budget)]
    frozen = [key for key in spec.parameter_grid if key not in focused]
    return {
        "schema": "active_probe_plan.v1",
        "case_id": case.case_id,
        "probe_id": spec.probe_id,
        "counterexample_seed": counterexample.get("seed"),
        "failure_reason": reason,
        "violated_constraints": violated_constraints,
        "selection_mode": mode,
        "budget": budget,
        "controlled_parameters": focused,
        "frozen_parameters": {
            key: _middle(spec.parameter_grid[key])
            for key in frozen
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "optimizer": optimizer,
        "measured_parameter_sensitivity": sensitivity,
        "selection_rationale": (
            "Map machine-readable violated constraints to a bounded parameter subspace; "
            "order that subspace by measured score sensitivity, then balance predicted "
            "score and uncertainty with a deterministic kernel-UCB acquisition."
        ),
    }


def write_active_probe_plan(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
