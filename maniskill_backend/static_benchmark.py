"""Batch runner for the static migration benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .evaluation import TrialRecord
from .static_runner import append_jsonl, run_static_trial
from .tasks import get_task_spec


DEFAULT_TASKS = (
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "PlugMulti-v1",
    "PullCubeTool-v1",
    "PegMulti-v1",
)
DEFAULT_METHODS = (
    "source-copy",
    "llm_no_card",
    "llm_card_only",
    "llm_report_only",
    "llm_card_report",
    "oracle",
)


def run_static_benchmark(
    *,
    tasks: Sequence[str] = DEFAULT_TASKS,
    methods: Sequence[str] = DEFAULT_METHODS,
    seeds: Sequence[int] = (0,),
    dry_run: bool = False,
) -> List[TrialRecord]:
    records: List[TrialRecord] = []
    for task_id in tasks:
        task = get_task_spec(task_id)
        for target_robot in task.target_robots:
            for method in methods:
                for seed in seeds:
                    records.append(
                        run_static_trial(
                            task_id=task_id,
                            target_robot=target_robot,
                            method=method,
                            seed=seed,
                            dry_run=dry_run,
                        )
                    )
    return records


def summarize_records(records: Iterable[TrialRecord]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str, str], List[TrialRecord]] = defaultdict(list)
    for record in records:
        buckets[(record.task_id, record.target_robot, record.method)].append(record)

    rows: List[Dict[str, object]] = []
    for (task_id, target_robot, method), group in sorted(buckets.items()):
        total = len(group)
        successes = sum(1 for item in group if item.success)
        failure_counts: Dict[str, int] = defaultdict(int)
        for item in group:
            failure_counts[item.failure_type] += 1
        dominant_failure = max(failure_counts.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "task_id": task_id,
                "target_robot": target_robot,
                "method": method,
                "trials": total,
                "successes": successes,
                "success_rate": round(successes / total, 4) if total else 0.0,
                "dominant_failure_type": dominant_failure,
            }
        )
    return rows


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "target_robot",
        "method",
        "trials",
        "successes",
        "success_rate",
        "dominant_failure_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_arg(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_seed_arg(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static migration benchmark.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--jsonl", default="results/trials.jsonl")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = run_static_benchmark(
        tasks=_parse_csv_arg(args.tasks),
        methods=_parse_csv_arg(args.methods),
        seeds=_parse_seed_arg(args.seeds),
        dry_run=args.dry_run,
    )
    rows = summarize_records(records)

    append_jsonl(Path(args.jsonl), records)
    write_summary_csv(Path(args.summary), rows)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nWrote: {args.jsonl}")
    print(f"Wrote: {args.summary}")


if __name__ == "__main__":
    main()
