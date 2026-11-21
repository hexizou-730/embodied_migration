"""Offline recomputation and statistics for repair-granularity ablation runs.

Reads the raw per-cycle evidence under ``results/repair_granularity/<run>/``,
recomputes every aggregate instead of trusting derived summaries, flags drift,
and reports paired statistics: Wilson intervals on success, the exact sign test
on discordant paired outcomes, and bootstrap intervals on cost differences.
All statistics are descriptive; sample sizes stay attached to every number.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from .repair_granularity import ARMS, _arm_metrics

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 0
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""

    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def sign_test_p(discordant_a: int, discordant_b: int) -> float | None:
    """Two-sided exact sign test on paired binary outcomes."""

    total = discordant_a + discordant_b
    if total == 0:
        return None
    tail = sum(math.comb(total, k) for k in range(0, min(discordant_a, discordant_b) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval around the mean, deterministic for fixed seed."""

    if not values:
        return None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    low = stats[int(0.025 * len(stats))]
    high = stats[min(len(stats) - 1, int(0.975 * len(stats)))]
    return low, high


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cycle_records(arm_dir: Path) -> list[dict[str, Any]]:
    records = []
    for cycle_dir in sorted(arm_dir.glob("cycle_*")):
        record_path = cycle_dir / "cycle_record.json"
        if record_path.is_file():
            records.append(_read_json(record_path))
    return records


def load_ablation_run(root: Path) -> dict[str, Any]:
    """Read raw evidence and recompute per-run metrics; never trust derived rows."""

    root = root.resolve()
    manifest_path = root / "repetition_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No repetition_manifest.json under {root}.")
    manifest = _read_json(manifest_path)
    checks: list[dict[str, Any]] = []

    def flag(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    repetitions: list[dict[str, Any]] = []
    rep_dirs = sorted(
        (path for path in root.glob("r*") if path.is_dir() and path.name[1:].isdigit()),
        key=lambda path: int(path.name[1:]),
    )
    flag(
        "manifest/repetition_count",
        len(rep_dirs) == len(manifest.get("sampling_seeds", [])),
        f"directories={len(rep_dirs)} manifest_seeds={len(manifest.get('sampling_seeds', []))}",
    )
    pinned_sha = (manifest.get("failed_candidate") or {}).get("sha256")
    for index, rep_dir in enumerate(rep_dirs, start=1):
        arms: dict[str, Any] = {}
        for arm_id in ARMS:
            arm_dir = rep_dir / arm_id
            summary_path = arm_dir / "summary.json"
            if not summary_path.is_file():
                flag(f"r{index:02d}/{arm_id}/summary_missing", False, str(arm_dir))
                continue
            summary = _read_json(summary_path)
            records = _cycle_records(arm_dir)
            recomputed = _recomputed_metrics(records)
            arms[arm_id] = {"summary": summary, "records": records, "recomputed": recomputed}
            flag(
                f"r{index:02d}/{arm_id}/cycle_count_matches_summary",
                len(records) == len(summary.get("cycles") or []),
                f"records={len(records)} summary_cycles={len(summary.get('cycles') or [])}",
            )
            recorded = _arm_metrics(arm_id, summary)
            for key in ("generation_cycles", "target_action_steps", "total_tokens", "valid_candidates"):
                if int(recorded[key]) != int(recomputed[key]):
                    flag(
                        f"r{index:02d}/{arm_id}/{key}_drift",
                        False,
                        f"summary={recorded[key]} recomputed={recomputed[key]}",
                    )
            flag(
                f"r{index:02d}/{arm_id}/success_consistent_with_cycles",
                bool(summary.get("success")) == bool(recomputed["any_cycle_success"]),
                f"summary={summary.get('success')} cycles={recomputed['any_cycle_success']}",
            )
            if summary.get("failed_candidate_sha256") and pinned_sha:
                flag(
                    f"r{index:02d}/{arm_id}/candidate_sha_matches_manifest",
                    summary["failed_candidate_sha256"] == pinned_sha,
                )
        if ARMS[0] in arms and ARMS[1] in arms:
            seed_a = (arms[ARMS[0]]["summary"].get("generation_settings") or {}).get("sampling_seed")
            seed_b = (arms[ARMS[1]]["summary"].get("generation_settings") or {}).get("sampling_seed")
            flag(
                f"r{index:02d}/paired_sampling_seed",
                seed_a == seed_b,
                f"{ARMS[0]}={seed_a} {ARMS[1]}={seed_b}",
            )
        repetitions.append({"repetition": index, "arms": arms})

    aggregate_path = root / "summary.json"
    aggregate = _read_json(aggregate_path) if aggregate_path.is_file() else None
    if aggregate:
        rows = {row.get("arm_id"): row for row in aggregate.get("arms") or []}
        for arm_id in ARMS:
            recorded = rows.get(arm_id) or {}
            summed = _summed_metrics(repetitions, arm_id)
            for key in ("num_runs", "num_success", "target_action_steps", "total_tokens"):
                if key in recorded and int(recorded[key]) != int(summed[key]):
                    flag(
                        f"aggregate/{arm_id}/{key}_drift",
                        False,
                        f"aggregate={recorded[key]} recomputed={summed[key]}",
                    )
    return {
        "schema": "granularity_analysis_input.v1",
        "root": str(root),
        "manifest": manifest,
        "repetitions": repetitions,
        "checks": checks,
    }


def _recomputed_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if row.get("valid") is True]
    return {
        "generation_cycles": len(records),
        "llm_calls": sum(row.get("used_llm") is True for row in records),
        "valid_candidates": len(valid),
        "target_action_steps": sum(int(row.get("action_steps") or 0) for row in records),
        "total_tokens": sum(int((row.get("usage") or {}).get("total_tokens") or 0) for row in records),
        "llm_cost_usd": round(
            sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in records), 8
        ),
        "cycles_to_success": next(
            (index for index, row in enumerate(records, start=1) if row.get("success")), None
        ),
        "any_cycle_success": any(row.get("success") for row in records),
    }


def _summed_metrics(repetitions: Sequence[Mapping[str, Any]], arm_id: str) -> dict[str, Any]:
    metrics = [
        (rep["arms"][arm_id]["recomputed"])
        for rep in repetitions
        if arm_id in rep["arms"] and "recomputed" in rep["arms"][arm_id]
    ]
    successes = sum(1 for row in metrics if row["cycles_to_success"] is not None)
    return {
        "num_runs": len(metrics),
        "num_success": successes,
        "generation_cycles": sum(row["generation_cycles"] for row in metrics),
        "llm_calls": sum(row["llm_calls"] for row in metrics),
        "valid_candidates": sum(row["valid_candidates"] for row in metrics),
        "target_action_steps": sum(row["target_action_steps"] for row in metrics),
        "total_tokens": sum(row["total_tokens"] for row in metrics),
        "llm_cost_usd": round(sum(row["llm_cost_usd"] for row in metrics), 8),
        "replay_action_steps_excluded": sum(
            int((rep["arms"][arm_id]["summary"].get("resume_replay") or {}).get("action_steps") or 0)
            for rep in repetitions
            if arm_id in rep["arms"]
        ),
    }


def analyze_ablation_run(root: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    """Recompute evidence, run paired statistics, and write derived analysis files."""

    loaded = load_ablation_run(root)
    output_dir = (output_dir or root).resolve()
    failed_checks = [check for check in loaded["checks"] if not check["pass"]]

    arm_rows = []
    for arm_id in ARMS:
        summed = _summed_metrics(loaded["repetitions"], arm_id)
        interval = wilson_interval(summed["num_success"], summed["num_runs"])
        cycles = [
            rep["arms"][arm_id]["recomputed"]["cycles_to_success"]
            for rep in loaded["repetitions"]
            if arm_id in rep["arms"]
            and rep["arms"][arm_id]["recomputed"]["cycles_to_success"] is not None
        ]
        arm_rows.append({
            "arm_id": arm_id,
            **summed,
            "success_rate": summed["num_success"] / summed["num_runs"] if summed["num_runs"] else None,
            "success_wilson_95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
            "cycles_to_success": cycles,
            "median_cycles_to_success": sorted(cycles)[len(cycles) // 2] if cycles else None,
        })

    paired = []
    for rep in loaded["repetitions"]:
        if not all(arm in rep["arms"] for arm in ARMS):
            continue
        method_row = rep["arms"][ARMS[0]]["recomputed"]
        module_row = rep["arms"][ARMS[1]]["recomputed"]
        method_ok = method_row["cycles_to_success"] is not None
        module_ok = module_row["cycles_to_success"] is not None
        paired.append({
            "method_patch_success": method_ok,
            "module_rewrite_success": module_ok,
            "action_steps": [method_row["target_action_steps"], module_row["target_action_steps"]],
            "total_tokens": [method_row["total_tokens"], module_row["total_tokens"]],
        })
    discordant_method_only = sum(1 for row in paired if row["method_patch_success"] and not row["module_rewrite_success"])
    discordant_module_only = sum(1 for row in paired if row["module_rewrite_success"] and not row["method_patch_success"])
    sign_p = sign_test_p(discordant_method_only, discordant_module_only)

    action_diffs = [float(row["action_steps"][1] - row["action_steps"][0]) for row in paired]
    token_diffs = [float(row["total_tokens"][1] - row["total_tokens"][0]) for row in paired]
    action_ci = bootstrap_mean_ci(action_diffs)
    token_ci = bootstrap_mean_ci(token_diffs)
    cost_guidance = []
    if action_ci:
        if action_ci[0] > 0:
            cost_guidance.append("module_rewrite consumed more actions per run (paired CI above zero)")
        elif action_ci[1] < 0:
            cost_guidance.append("module_rewrite consumed fewer actions per run (paired CI below zero)")
        else:
            cost_guidance.append("paired action difference is not separated from zero at this sample size")
    if token_ci:
        if token_ci[0] > 0:
            cost_guidance.append("module_rewrite consumed more tokens per run (paired CI above zero)")
        elif token_ci[1] < 0:
            cost_guidance.append("module_rewrite consumed fewer tokens per run (paired CI below zero)")
        else:
            cost_guidance.append("paired token difference is not separated from zero at this sample size")

    payload = {
        "schema": "granularity_analysis.v1",
        "root": loaded["root"],
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "method": "percentile"},
        "failed_candidate_sha256": (loaded["manifest"].get("failed_candidate") or {}).get("sha256"),
        "evidence_checks": {
            "total": len(loaded["checks"]),
            "failed": len(failed_checks),
            "failed_details": failed_checks,
        },
        "arms": arm_rows,
        "paired": {
            "num_pairs": len(paired),
            "per_repetition": paired,
            "discordant_method_only": discordant_method_only,
            "discordant_module_only": discordant_module_only,
            "sign_test_p_two_sided": sign_p,
            "mean_action_difference_module_minus_method": (
                sum(action_diffs) / len(action_diffs) if action_diffs else None
            ),
            "action_difference_bootstrap_95": [round(action_ci[0], 2), round(action_ci[1], 2)] if action_ci else None,
            "mean_token_difference_module_minus_method": (
                sum(token_diffs) / len(token_diffs) if token_diffs else None
            ),
            "token_difference_bootstrap_95": [round(token_ci[0], 2), round(token_ci[1], 2)] if token_ci else None,
            "cost_guidance": cost_guidance,
        },
        "interpretation_limits": [
            f"n={len(paired)} paired repair runs from one shared failed candidate in one fixed scene",
            "resume-based repair evidence; never merge with from-scratch generation comparisons",
            "descriptive statistics only; no held-out-scene or cross-task generalization claim",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "granularity_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "granularity_analysis.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Repair-granularity ablation analysis",
        "",
        f"Derived offline from raw cycle records under `{payload['root']}`; "
        "original summaries and trials are never modified.",
        "",
        "## Per-arm results",
        "",
        "| arm | n | success | rate (Wilson 95%) | median cycles | actions | tokens |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["arms"]:
        interval = row["success_wilson_95"]
        rate = "-" if interval is None else f"{row['success_rate']:.2f} [{interval[0]:.2f}, {interval[1]:.2f}]"
        cycles = "-" if row["median_cycles_to_success"] is None else row["median_cycles_to_success"]
        lines.append(
            f"| {row['arm_id']} | {row['num_runs']} | {row['num_success']} | {rate} | {cycles} "
            f"| {row['target_action_steps']} | {row['total_tokens']} |"
        )
    paired = payload["paired"]
    lines += [
        "",
        "## Paired comparison",
        "",
        f"- pairs: {paired['num_pairs']}; discordant method-only / module-only: "
        f"{paired['discordant_method_only']} / {paired['discordant_module_only']}; "
        f"two-sided exact sign test p = {paired['sign_test_p_two_sided']}",
        f"- mean action difference (module - method): "
        f"{paired['mean_action_difference_module_minus_method']} "
        f"(bootstrap 95% {paired['action_difference_bootstrap_95']})",
        f"- mean token difference (module - method): "
        f"{paired['mean_token_difference_module_minus_method']} "
        f"(bootstrap 95% {paired['token_difference_bootstrap_95']})",
        "",
    ]
    for note in paired["cost_guidance"]:
        lines.append(f"- {note}")
    lines += ["", "## Evidence checks", ""]
    checks = payload["evidence_checks"]
    lines.append(f"- {checks['total']} recomputation checks, {checks['failed']} failed")
    for failure in checks["failed_details"]:
        lines.append(f"- FAILED: {failure['name']} ({failure.get('detail', '')})")
    lines += ["", "## Limits", ""]
    for limit in payload["interpretation_limits"]:
        lines.append(f"- {limit}")
    return "\n".join(lines) + "\n"
