"""Counterexample extraction for simulation-guided adapter synthesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from maniskill_backend.cases import FullMigrationCase
from maniskill_backend.embodiment_contracts import contract_from_case


REASON_CONSTRAINTS: Dict[str, Tuple[str, ...]] = {
    "program_or_api_mismatch": ("high_level_program_api_compatible",),
    "controller_or_action_interface_failure": ("action_layout_matches_contract",),
    "contact_side_reachability_failure": (
        "contact_pose_reachable_before_descent",
        "contact_side_selected_from_runtime_geometry",
    ),
    "tcp_never_established_effective_contact": (
        "tcp_object_contact_established_before_drag",
    ),
    "contact_established_but_drag_progress_insufficient": (
        "object_goal_error_decreases_during_drag",
        "contact_preserved_during_drag",
    ),
    "episode_budget_exhausted_before_success": (
        "bounded_phase_budget",
        "early_stop_on_no_progress",
    ),
    "approach_descent_alignment_failure": (
        "tcp_aligned_before_gripper_close",
    ),
    "xy_alignment_failure_before_close": (
        "tcp_xy_aligned_before_gripper_close",
    ),
    "gripper_envelope_side_push": (
        "cube_displacement_bounded_during_close",
        "gripper_envelope_matches_object",
    ),
    "good_alignment_no_displacement_no_grasp": (
        "gripper_envelope_forms_force_closure",
    ),
    "grasp_detected_but_not_preserved_or_placed": (
        "grasp_preserved_until_place",
    ),
    "declared_or_detected_infeasible_condition": (
        "explicit_infeasibility_supported_by_measurement",
    ),
}


@dataclass(frozen=True)
class Counterexample:
    schema: str
    case_id: str
    task_id: str
    source_robot: str
    target_robot: str
    seed: int
    failure_layer: str
    failure_reason: str
    stage: str
    confidence: float
    message: str
    violated_constraints: Tuple[str, ...]
    evidence: Mapping[str, Any]
    adapter_sha256: str
    information_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_section(self) -> str:
        payload = self.to_dict()
        return "\n".join(
            [
                "# Selected simulation counterexample",
                "This is a failed physical execution, not a hand-written answer.",
                "```json",
                json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
                "```",
            ]
        )


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_evidence(row: Mapping[str, Any]) -> Dict[str, Any]:
    diagnosis = row.get("failure_diagnosis") or {}
    runtime = row.get("runtime_diagnostics") or {}
    initial = row.get("initial_runtime_diagnostics") or {}
    evidence: Dict[str, Any] = dict(diagnosis.get("evidence") or {})
    for key in (
        "stage",
        "tcp_stage_error_norm",
        "tcp_stage_error_xyz",
        "stage_target_pos",
        "tcp_contact_error_norm",
        "tcp_cube_xy",
        "cube_goal_xy",
        "tcp_pos",
        "cube_pos",
        "goal_pos",
    ):
        value = runtime.get(key)
        if value not in (None, "", [], {}):
            evidence[key] = value
    if initial:
        evidence["initial_runtime_diagnostics"] = dict(initial)
    final_info = row.get("final_info")
    if final_info:
        evidence["final_info"] = final_info
    return evidence


def _diagnostic_completeness(evidence: Mapping[str, Any]) -> float:
    useful = (
        "stage",
        "tcp_stage_error_norm",
        "tcp_cube_xy",
        "cube_goal_xy",
        "tcp_grasp_xy",
        "tcp_grasp_z",
        "cube_disp_xy",
        "cube_pos",
        "tcp_pos",
        "goal_pos",
    )
    return sum(1.0 for key in useful if evidence.get(key) not in (None, "", [], {}))


def _information_score(
    *,
    confidence: float,
    evidence: Mapping[str, Any],
    reason: str,
) -> float:
    score = 10.0 * max(0.0, min(1.0, confidence))
    score += _diagnostic_completeness(evidence)
    if reason in REASON_CONSTRAINTS:
        score += 2.0
    if evidence.get("stage"):
        score += 1.0
    return round(score, 4)


def counterexample_from_trial(
    case: FullMigrationCase,
    trial: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> Counterexample:
    diagnosis = trial.get("failure_diagnosis") or {}
    reason = str(diagnosis.get("reason") or trial.get("failure_type") or "unclassified_failure")
    layer = str(diagnosis.get("layer") or trial.get("failure_layer") or "unknown")
    runtime = trial.get("runtime_diagnostics") or {}
    stage = str(runtime.get("stage") or (diagnosis.get("evidence") or {}).get("stage") or "")
    confidence = float(diagnosis.get("confidence") or 0.0)
    evidence = _runtime_evidence(trial)
    root = repo_root or Path(__file__).resolve().parents[1]
    adapter_hash = file_sha256(root / case.target_adapter_path)
    contract = contract_from_case(case)
    violated = list(
        diagnosis.get("violated_constraints")
        or REASON_CONSTRAINTS.get(reason, (f"unclassified:{layer}",))
    )
    if reason == "controller_or_action_interface_failure":
        violated.append(f"expected_action_dim:{contract.action_dim}")
    return Counterexample(
        schema="embodiment_counterexample.v1",
        case_id=case.case_id,
        task_id=case.task_id,
        source_robot=case.source_robot,
        target_robot=case.target_robot,
        seed=int(trial.get("seed") if trial.get("seed") is not None else case.seed),
        failure_layer=layer,
        failure_reason=reason,
        stage=stage,
        confidence=round(confidence, 3),
        message=str(trial.get("message") or ""),
        violated_constraints=tuple(dict.fromkeys(violated)),
        evidence=evidence,
        adapter_sha256=adapter_hash,
        information_score=_information_score(
            confidence=confidence,
            evidence=evidence,
            reason=reason,
        ),
    )


def rank_counterexamples(
    case: FullMigrationCase,
    trials: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
) -> List[Counterexample]:
    counterexamples = [
        counterexample_from_trial(case, trial, repo_root=repo_root)
        for trial in trials
        if not bool(trial.get("success"))
    ]
    return sorted(
        counterexamples,
        key=lambda item: (item.information_score, item.confidence, -item.seed),
        reverse=True,
    )


def select_counterexample(
    case: FullMigrationCase,
    trials: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
) -> Counterexample | None:
    ranked = rank_counterexamples(case, trials, repo_root=repo_root)
    return ranked[0] if ranked else None


def counterexample_prompt(counterexample: Mapping[str, Any] | Counterexample | None) -> str:
    if counterexample is None:
        return ""
    if isinstance(counterexample, Counterexample):
        return counterexample.to_prompt_section()
    payload = dict(counterexample)
    supporting = payload.get("supporting_counterexamples") or []
    return "\n".join(
        [
            "# Selected simulation counterexample",
            "This is a failed physical execution, not a hand-written answer.",
            (
                "The primary failure is accompanied by distinct development-seed failures. "
                "Repair the shared constraint and preserve guarded branches; do not overfit one seed."
                if supporting
                else "Repair this measured failure without changing frozen interfaces."
            ),
            "```json",
            json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
            "```",
        ]
    )


def _selection_novelty(
    item: Counterexample,
    history: Sequence[Mapping[str, Any]],
) -> float:
    if not history:
        return 0.0
    reasons = {str(previous.get("failure_reason") or "") for previous in history}
    seeds = {int(previous.get("seed", -1)) for previous in history}
    signatures = {
        (
            int(previous.get("seed", -1)),
            str(previous.get("failure_reason") or ""),
            str(previous.get("stage") or ""),
        )
        for previous in history
    }
    signature = (item.seed, item.failure_reason, item.stage)
    bonus = 0.0
    if item.failure_reason not in reasons:
        bonus += 2.0
    if item.seed not in seeds:
        bonus += 1.0
    if signature in signatures:
        bonus -= 4.0
    return bonus


def write_counterexample_set(
    output_path: Path,
    case: FullMigrationCase,
    trials: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
    selection_history: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    ranked = rank_counterexamples(case, trials, repo_root=repo_root)
    scored = []
    for item in ranked:
        row = item.to_dict()
        novelty = _selection_novelty(item, selection_history)
        row["novelty_bonus"] = novelty
        row["selection_score"] = round(item.information_score + novelty, 4)
        scored.append(row)
    scored.sort(
        key=lambda item: (
            float(item.get("selection_score") or 0.0),
            float(item.get("information_score") or 0.0),
            -int(item.get("seed") or 0),
        ),
        reverse=True,
    )
    selected = dict(scored[0]) if scored else None
    supporting = []
    if selected:
        primary_signature = (
            str(selected.get("failure_reason") or ""),
            str(selected.get("stage") or ""),
        )
        seen_signatures = {primary_signature}
        for item in scored[1:]:
            signature = (
                str(item.get("failure_reason") or ""),
                str(item.get("stage") or ""),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            supporting.append(
                {
                    key: item.get(key)
                    for key in (
                        "seed",
                        "failure_layer",
                        "failure_reason",
                        "stage",
                        "confidence",
                        "violated_constraints",
                        "evidence",
                        "information_score",
                    )
                }
            )
            if len(supporting) >= 3:
                break
        selected["portfolio_policy"] = "primary_plus_distinct_reason_stage_representatives"
        selected["supporting_counterexamples"] = supporting
    payload = {
        "schema": "embodiment_counterexample_set.v1",
        "case_id": case.case_id,
        "num_failures": len(ranked),
        "selection_policy": "information_score_plus_history_novelty",
        "selection_history_size": len(selection_history),
        "selected": selected,
        "portfolio_size": (1 + len(supporting)) if selected else 0,
        "counterexamples": scored,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
    return payload
