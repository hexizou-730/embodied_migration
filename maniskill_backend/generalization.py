"""Multi-seed generalization summaries and automatic strategy selection."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _elapsed_steps(row: Mapping[str, Any]) -> int | None:
    final_info = row.get("final_info") or {}
    raw = final_info.get("elapsed_steps")
    if isinstance(raw, list) and raw:
        return int(raw[0])
    if isinstance(raw, (int, float)):
        return int(raw)
    return None


def _dominant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def _reason_from_row(row: Mapping[str, Any]) -> str:
    diagnosis = row.get("failure_diagnosis") or {}
    return str(diagnosis.get("reason") or row.get("failure_type") or "unknown")


def _stage_from_row(row: Mapping[str, Any]) -> str:
    diagnostics = row.get("runtime_diagnostics") or {}
    return str(diagnostics.get("stage") or "")


def _strategy_for_failure_cluster(reason: str, stage: str) -> Dict[str, str]:
    if reason == "contact_side_reachability_failure" or stage in {"approach", "descent"}:
        return {
            "selected_strategy": "reachability_aware_contact_selection",
            "next_action": (
                "Cluster failures by seed geometry, add a pre-drag reachability/contact-side check, "
                "and choose contact waypoints before committing episode budget."
            ),
            "strategy_reason": "Most failures happen before stable contact, during approach/descent.",
        }
    if reason == "tcp_never_established_effective_contact":
        return {
            "selected_strategy": "contact_establishment_repair",
            "next_action": "Repair approach/contact establishment before tuning drag pulses.",
            "strategy_reason": "TCP does not reliably establish physical contact with the object.",
        }
    if reason == "contact_established_but_drag_progress_insufficient":
        return {
            "selected_strategy": "drag_progress_repair",
            "next_action": "Keep the successful contact side and tune drag pulse direction, strength, and down-bias.",
            "strategy_reason": "Contact exists, but object progress toward the goal is insufficient.",
        }
    if reason in {"good_alignment_no_displacement_no_grasp", "gripper_envelope_side_push"}:
        return {
            "selected_strategy": "gripper_envelope_probe_and_repair",
            "next_action": "Use structured close-envelope probing and then repair grasp geometry/timing.",
            "strategy_reason": "TCP alignment is not the bottleneck; force-closure/contact geometry is.",
        }
    return {
        "selected_strategy": "diagnosis_first_repair",
        "next_action": "Improve runtime diagnostics for the dominant failure cluster before changing the adapter.",
        "strategy_reason": "The dominant failure does not map cleanly to a known repair policy.",
    }


def build_generalization_report(
    summary: Mapping[str, Any],
    *,
    success_threshold: float = 0.8,
    min_trials_for_accept: int = 5,
) -> Dict[str, Any]:
    """Summarize multi-seed robustness and choose the next repair strategy."""

    rows = list(summary.get("rows") or [])
    successes = [row for row in rows if bool(row.get("success"))]
    failures = [row for row in rows if not bool(row.get("success"))]
    num_trials = len(rows)
    success_rate = round(len(successes) / num_trials, 4) if num_trials else 0.0

    failure_type_counts = Counter(str(row.get("failure_type") or "unknown") for row in failures)
    failure_layer_counts = Counter(str(row.get("failure_layer") or "unknown") for row in failures)
    reason_counts = Counter(_reason_from_row(row) for row in failures)
    stage_counts = Counter(_stage_from_row(row) for row in failures if _stage_from_row(row))
    success_seeds = [row.get("seed") for row in successes]
    failure_seeds = [row.get("seed") for row in failures]
    steps = [_elapsed_steps(row) for row in rows]
    good_steps = [step for step in steps if step is not None]

    dominant_reason = _dominant(reason_counts)
    dominant_stage = _dominant(stage_counts)
    threshold_met = success_rate >= success_threshold and num_trials >= min_trials_for_accept
    if threshold_met:
        selected = {
            "selected_strategy": "accept_current_adapter",
            "next_action": (
                "Promote this adapter as the current robust baseline, then continue regression testing "
                "on a held-out seed set."
            ),
            "strategy_reason": (
                f"Success rate {success_rate} meets threshold {success_threshold} over {num_trials} trials."
            ),
        }
        status = "accepted"
    else:
        selected = _strategy_for_failure_cluster(dominant_reason, dominant_stage)
        status = "needs_repair"

    seed_clusters: Dict[str, List[Any]] = defaultdict(list)
    for row in failures:
        key = _reason_from_row(row)
        stage = _stage_from_row(row)
        if stage:
            key = f"{key}:{stage}"
        seed_clusters[key].append(row.get("seed"))

    return {
        "schema": "generalization_strategy.v1",
        "task_id": summary.get("task_id"),
        "robot_uid": summary.get("robot_uid"),
        "adapter_module": summary.get("adapter_module"),
        "adapter_sha256": summary.get("adapter_sha256"),
        "num_trials": num_trials,
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": success_rate,
        "success_threshold": success_threshold,
        "min_trials_for_accept": min_trials_for_accept,
        "status": status,
        "success_seeds": success_seeds,
        "failure_seeds": failure_seeds,
        "mean_elapsed_steps": round(sum(good_steps) / len(good_steps), 2) if good_steps else None,
        "failure_type_counts": dict(failure_type_counts),
        "failure_layer_counts": dict(failure_layer_counts),
        "failure_reason_counts": dict(reason_counts),
        "failure_stage_counts": dict(stage_counts),
        "failure_seed_clusters": dict(seed_clusters),
        "dominant_failure_reason": dominant_reason,
        "dominant_failure_stage": dominant_stage,
        **selected,
    }


def generalization_report_to_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "## Generalization Strategy Selection",
        "",
        f"- status: `{report.get('status')}`",
        f"- success rate: `{report.get('success_rate')}`",
        f"- selected strategy: `{report.get('selected_strategy')}`",
        f"- reason: {report.get('strategy_reason')}",
        f"- next action: {report.get('next_action')}",
        "",
        "### Seed Split",
        "",
        f"- success seeds: `{report.get('success_seeds')}`",
        f"- failure seeds: `{report.get('failure_seeds')}`",
        "",
        "### Failure Distribution",
        "",
        "| type | counts |",
        "|---|---|",
        f"| failure_type | `{report.get('failure_type_counts')}` |",
        f"| failure_layer | `{report.get('failure_layer_counts')}` |",
        f"| diagnosis_reason | `{report.get('failure_reason_counts')}` |",
        f"| runtime_stage | `{report.get('failure_stage_counts')}` |",
        "",
        "### Failure Seed Clusters",
        "",
        "| cluster | seeds |",
        "|---|---|",
    ]
    clusters = report.get("failure_seed_clusters") or {}
    if clusters:
        for cluster, seeds in clusters.items():
            lines.append(f"| `{cluster}` | `{seeds}` |")
    else:
        lines.append("| none | `[]` |")
    lines.append("")
    return "\n".join(lines)
