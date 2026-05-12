"""
Audit a benchmark run for completeness and reproducibility.

Usage:
    python -m benchmark.audit_run results/runs/stage5_mobile_dual_seeded
    python -m benchmark.audit_run results/runs/stage5_mobile_dual_seeded --fail-on-missing

Outputs:
    results/runs/<run_id>/audit/audit_summary.json
    results/runs/<run_id>/audit/audit_report.md
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Benchmark run directory.")
    ap.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero if expected trials are missing or incomplete.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    audit = audit_run(run_dir)
    out_dir = run_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "audit_report.md").write_text(render_report(audit), encoding="utf-8")

    print(f"Wrote audit to: {out_dir}")
    print(
        f"expected={audit['expected_trial_count']} "
        f"found={audit['found_trial_count']} "
        f"missing={len(audit['missing_trials'])} "
        f"incomplete={len(audit['incomplete_trials'])}"
    )
    if args.fail_on_missing and (audit["missing_trials"] or audit["incomplete_trials"]):
        raise SystemExit(1)


def audit_run(run_dir: Path) -> Dict[str, Any]:
    metadata = read_json_if_exists(run_dir / "metadata.json")
    records = load_trial_records(run_dir)
    summary_rows = load_summary_rows(run_dir / "summary.csv")
    expected_ids = expected_trial_ids(metadata)
    records_by_id = {record.get("trial_id", ""): record for record in records}

    if expected_ids:
        missing = [trial_id for trial_id in expected_ids if trial_id not in records_by_id]
    else:
        missing = []

    incomplete = [
        record.get("trial_id", "")
        for record in records
        if not is_complete_record(record)
    ]
    llm_errors = [
        record.get("trial_id", "")
        for record in records
        if record.get("llm_error")
    ]
    cache_hit_attempts = sum(
        1
        for record in records
        for attempt in record.get("attempts", []) or []
        if attempt.get("llm_cache_hit")
    )
    total_attempts = sum(len(record.get("attempts", []) or []) for record in records)

    return {
        "run_dir": str(run_dir),
        "metadata_present": bool(metadata),
        "metadata": metadata,
        "expected_trial_count": len(expected_ids) if expected_ids else "",
        "found_trial_count": len(records),
        "summary_row_count": len(summary_rows),
        "total_attempts": total_attempts,
        "cache_hit_attempts": cache_hit_attempts,
        "cache_hit_rate": pct(cache_hit_attempts, total_attempts),
        "success_count": sum(1 for record in records if record.get("success")),
        "failure_count": sum(1 for record in records if not record.get("success")),
        "missing_trials": missing,
        "incomplete_trials": incomplete,
        "llm_error_trials": llm_errors,
        "modes": sorted({record.get("canonical_mode", "") for record in records}),
        "robots": sorted({record.get("robot", "") for record in records}),
        "tasks": sorted({record.get("task", "") for record in records}),
        "scene_variants": sorted({str(record.get("scene_variant", "")) for record in records}),
        "scene_seeds": sorted(
            {str(record.get("scene_seed", "")) for record in records},
            key=natural_sort_value,
        ),
    }


def expected_trial_ids(metadata: Dict[str, Any]) -> List[str]:
    if not metadata:
        return []
    modes = metadata.get("modes") or []
    robots = metadata.get("robots") or []
    tasks = metadata.get("task_names") or []
    n_trials = int(metadata.get("n_trials") or 0)
    expected = []
    trial_seq = 0
    for mode in modes:
        for robot in robots:
            for task in tasks:
                for trial in range(n_trials):
                    trial_seq += 1
                    expected.append(
                        f"{trial_seq:05d}_{mode}_{robot}_{task}_trial{trial + 1:03d}"
                    )
    return expected


def load_trial_records(run_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted((run_dir / "trials").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path)
        records.append(record)
    return records


def load_summary_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_complete_record(record: Dict[str, Any]) -> bool:
    return bool(
        record.get("trial_id")
        and record.get("final_reason")
        and isinstance(record.get("attempts"), list)
        and record.get("attempt_count", len(record.get("attempts") or [])) == len(record.get("attempts") or [])
    )


def render_report(audit: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Run Audit",
        "",
        f"Run directory: `{audit['run_dir']}`",
        "",
        "## Summary",
        "",
        f"- Metadata present: `{audit['metadata_present']}`",
        f"- Expected trials: `{audit['expected_trial_count']}`",
        f"- Found trial JSON files: `{audit['found_trial_count']}`",
        f"- Summary CSV rows: `{audit['summary_row_count']}`",
        f"- Success / failure: `{audit['success_count']}` / `{audit['failure_count']}`",
        f"- Total LLM attempts: `{audit['total_attempts']}`",
        f"- Cache hit attempts: `{audit['cache_hit_attempts']}` ({audit['cache_hit_rate']})",
        "",
        "## Coverage",
        "",
        f"- Modes: `{audit['modes']}`",
        f"- Robots: `{audit['robots']}`",
        f"- Tasks: `{audit['tasks']}`",
        f"- Scene variants: `{audit['scene_variants']}`",
        f"- Scene seeds: `{audit['scene_seeds']}`",
        "",
        "## Missing Trials",
        "",
        bullet_list(audit["missing_trials"]),
        "",
        "## Incomplete Trials",
        "",
        bullet_list(audit["incomplete_trials"]),
        "",
        "## LLM Error Trials",
        "",
        bullet_list(audit["llm_error_trials"]),
        "",
    ]
    return "\n".join(lines)


def bullet_list(items: List[str], limit: int = 50) -> str:
    if not items:
        return "_None._"
    shown = items[:limit]
    lines = [f"- `{item}`" for item in shown]
    if len(items) > limit:
        lines.append(f"- ... {len(items) - limit} more")
    return "\n".join(lines)


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def natural_sort_value(value: Any) -> tuple:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


if __name__ == "__main__":
    main()
