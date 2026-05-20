"""Result logging and summary helpers for real ManiSkill trials."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .evaluation import TrialRecord


def append_jsonl(path: Path, records: Iterable[TrialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


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
