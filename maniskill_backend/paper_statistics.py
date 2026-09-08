"""Paired held-out comparisons for the frozen paper benchmark."""

from __future__ import annotations

import hashlib
import random
from math import comb
from typing import Any, Dict, Iterable, Mapping, Sequence


def _held_out_by_seed(run: Mapping[str, Any]) -> Dict[int, bool]:
    rows = (((run.get("details") or {}).get("held_out") or {}).get("rows") or [])
    result: Dict[int, bool] = {}
    for row in rows:
        if row.get("seed") is None:
            continue
        result[int(row["seed"])] = bool(row.get("success"))
    return result


def _mcnemar_exact(reference_only: int, comparator_only: int) -> float:
    """Two-sided exact McNemar p-value for discordant binary pairs."""

    discordant = reference_only + comparator_only
    if discordant == 0:
        return 1.0
    lower = min(reference_only, comparator_only)
    tail = sum(comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return round(min(1.0, 2.0 * tail), 6)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _hierarchical_bootstrap_interval(
    records: Sequence[tuple[str, int, int, bool, bool]],
    *,
    key: str,
    iterations: int = 2000,
) -> tuple[float | None, float | None]:
    """Bootstrap paired success differences over case, repetition, and seed.

    Reusing the same held-out seeds across independent LLM generations creates
    clustered observations. Resampling the three coordinates hierarchically
    avoids presenting every adapter-seed pair as an unrelated Bernoulli trial.
    """

    hierarchy: Dict[str, Dict[int, list[tuple[int, bool, bool]]]] = {}
    for case_id, repetition, seed, reference, comparator in records:
        hierarchy.setdefault(case_id, {}).setdefault(repetition, []).append(
            (seed, reference, comparator)
        )
    cases = sorted(hierarchy)
    if not cases:
        return None, None
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    samples: list[float] = []
    for _ in range(iterations):
        differences: list[int] = []
        for case_id in rng.choices(cases, k=len(cases)):
            repetitions = sorted(hierarchy[case_id])
            for repetition in rng.choices(repetitions, k=len(repetitions)):
                rows = hierarchy[case_id][repetition]
                for _seed, reference, comparator in rng.choices(rows, k=len(rows)):
                    differences.append(int(reference) - int(comparator))
        if differences:
            samples.append(sum(differences) / len(differences))
    low = _percentile(samples, 0.025)
    high = _percentile(samples, 0.975)
    return (
        round(low, 4) if low is not None else None,
        round(high, 4) if high is not None else None,
    )


def _comparison_row(
    *,
    case_id: str,
    reference_method: str,
    comparator_method: str,
    pairs: Sequence[tuple[bool, bool]],
    records: Sequence[tuple[str, int, int, bool, bool]],
    support_status: str,
) -> Dict[str, Any]:
    both_success = sum(reference and comparator for reference, comparator in pairs)
    reference_only = sum(reference and not comparator for reference, comparator in pairs)
    comparator_only = sum(not reference and comparator for reference, comparator in pairs)
    both_failure = sum(not reference and not comparator for reference, comparator in pairs)
    total = len(pairs)
    bootstrap_low, bootstrap_high = _hierarchical_bootstrap_interval(
        records,
        key=f"{reference_method}:{comparator_method}:{case_id}",
    )
    return {
        "case_id": case_id,
        "reference_method": reference_method,
        "comparator_method": comparator_method,
        "support_status": support_status,
        "num_paired_trials": total,
        "both_success": both_success,
        "reference_only_success": reference_only,
        "comparator_only_success": comparator_only,
        "both_failure": both_failure,
        "reference_success_rate": round((both_success + reference_only) / total, 4) if total else None,
        "comparator_success_rate": round((both_success + comparator_only) / total, 4) if total else None,
        "paired_success_rate_difference": round((reference_only - comparator_only) / total, 4) if total else None,
        "mcnemar_exact_p": _mcnemar_exact(reference_only, comparator_only) if total else None,
        "hierarchical_bootstrap_95_low": bootstrap_low,
        "hierarchical_bootstrap_95_high": bootstrap_high,
        "hierarchical_bootstrap_iterations": 2000,
    }


def _holm_adjust(rows: Sequence[Dict[str, Any]]) -> None:
    indexed = [
        (index, float(row["mcnemar_exact_p"]))
        for index, row in enumerate(rows)
        if row.get("mcnemar_exact_p") is not None
        and not str(row.get("case_id") or "").startswith("__")
    ]
    ordered = sorted(indexed, key=lambda item: item[1])
    running = 0.0
    count = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * p_value)
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = round(running, 6)
    for row in rows:
        row["holm_family"] = (
            "planned_case_level_comparisons"
            if not str(row.get("case_id") or "").startswith("__")
            else "descriptive_aggregate_not_adjusted"
        )


def paired_method_comparisons(
    completed_runs: Iterable[Mapping[str, Any]],
    *,
    reference_method: str = "Ours",
) -> list[Dict[str, Any]]:
    """Pair held-out outcomes by case, repetition, and seed.

    Missing methods or seeds are never imputed. A comparison is reported only
    for outcomes that both methods actually evaluated under the same frozen
    run coordinates.
    """

    indexed: Dict[tuple[str, str, int], Dict[int, bool]] = {}
    methods = set()
    cases = set()
    support_by_case: Dict[str, str] = {}
    for run in completed_runs:
        method_id = str(run.get("method_id") or "")
        case_id = str(run.get("case_id") or "")
        repetition = int(run.get("repetition") or 1)
        outcomes = _held_out_by_seed(run)
        if not method_id or not case_id or not outcomes:
            continue
        indexed[(method_id, case_id, repetition)] = outcomes
        methods.add(method_id)
        cases.add(case_id)
        support_by_case.setdefault(case_id, str(run.get("support_status") or "unknown"))

    rows: list[Dict[str, Any]] = []
    for comparator in sorted(methods - {reference_method}):
        aggregate: list[tuple[bool, bool]] = []
        aggregate_records: list[tuple[str, int, int, bool, bool]] = []
        aggregate_by_support: Dict[str, list[tuple[bool, bool]]] = {}
        records_by_support: Dict[str, list[tuple[str, int, int, bool, bool]]] = {}
        for case_id in sorted(cases):
            pairs: list[tuple[bool, bool]] = []
            case_records: list[tuple[str, int, int, bool, bool]] = []
            repetitions = sorted(
                repetition
                for method, case, repetition in indexed
                if method == reference_method and case == case_id
            )
            for repetition in repetitions:
                reference = indexed.get((reference_method, case_id, repetition), {})
                baseline = indexed.get((comparator, case_id, repetition), {})
                for seed in sorted(set(reference) & set(baseline)):
                    pairs.append((reference[seed], baseline[seed]))
                    case_records.append(
                        (case_id, repetition, seed, reference[seed], baseline[seed])
                    )
            if pairs:
                rows.append(
                    _comparison_row(
                        case_id=case_id,
                        reference_method=reference_method,
                        comparator_method=comparator,
                        pairs=pairs,
                        records=case_records,
                        support_status=support_by_case.get(case_id, "unknown"),
                    )
                )
                aggregate.extend(pairs)
                aggregate_records.extend(case_records)
                support = support_by_case.get(case_id, "unknown")
                aggregate_by_support.setdefault(support, []).extend(pairs)
                records_by_support.setdefault(support, []).extend(case_records)
        for support, pairs in sorted(aggregate_by_support.items()):
            if support == "unknown":
                continue
            rows.append(
                _comparison_row(
                    case_id=f"__support__:{support}",
                    reference_method=reference_method,
                    comparator_method=comparator,
                    pairs=pairs,
                    records=records_by_support[support],
                    support_status=support,
                )
            )
        if aggregate:
            rows.append(
                _comparison_row(
                    case_id="__all__",
                    reference_method=reference_method,
                    comparator_method=comparator,
                    pairs=aggregate,
                    records=aggregate_records,
                    support_status="mixed",
                )
            )
    _holm_adjust(rows)
    return rows
