"""Constraint-guided active probe selection.

The selector narrows a structured probe to parameters relevant to the selected
counterexample. Previous measured probe rows are used as local-search anchors;
otherwise the selector chooses a small one-factor-at-a-time design around the
default probe center.
"""

from __future__ import annotations

import json
from itertools import product
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
        "contact_x_offset",
        "contact_z_offset",
        "approach_height",
    ),
    "tcp_never_established_effective_contact": (
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
        "contact_x_offset",
        "approach_height",
        "stages",
    ),
    "approach_descent_alignment_failure": (
        "grasp_z_offset",
    ),
    "xy_alignment_failure_before_close": (
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
    focused = [
        key
        for key in REASON_PARAMETER_FOCUS.get(reason, tuple(spec.parameter_grid))
        if key in spec.parameter_grid
    ]
    prior_rows = list(
        (previous_probe or {}).get("all_probe_cases")
        or (previous_probe or {}).get("results")
        or []
    )
    mode = "constraint_guided_initial_design"
    if prior_rows:
        candidates = suggest_next_probe_cases(
            spec,
            rank_probe_cases(prior_rows, spec),
            budget=max(budget * 2, budget),
        )
        candidates = [
            item
            for item in candidates
            if any(
                str(item.get("suggestion_reason") or "").endswith(f"_{key}")
                for key in focused
            )
        ] or candidates
        mode = "measurement_guided_local_refinement"
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
        "violated_constraints": list(counterexample.get("violated_constraints") or []),
        "selection_mode": mode,
        "budget": budget,
        "controlled_parameters": focused,
        "frozen_parameters": {
            key: _middle(spec.parameter_grid[key])
            for key in frozen
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_rationale": (
            "Vary only parameters connected to the diagnosed violated constraint; "
            "use measured high-score rows as anchors when available."
        ),
    }


def write_active_probe_plan(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
