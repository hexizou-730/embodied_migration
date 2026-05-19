"""Convenience command for the current static experiment workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .static_benchmark import DEFAULT_TASKS, summarize_records, write_summary_csv
from .static_runner import append_jsonl, run_static_trial
from .view import records_to_md


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and format static trials.")
    parser.add_argument("--task", default="", help="Single task shortcut.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--target", default="", help="Single target shortcut.")
    parser.add_argument("--targets", default="so100")
    parser.add_argument(
        "--methods",
        default="source-copy,llm_no_card,llm_card_only,llm_report_only,llm_card_report,oracle",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl", default="results/trials.jsonl")
    parser.add_argument("--md", default="results/trials.md")
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _items(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    args = _parse_args()
    tasks = (args.task,) if args.task else _items(args.tasks)
    targets = (args.target,) if args.target else _items(args.targets)
    methods = _items(args.methods)
    records = [
        run_static_trial(
            task_id=task,
            target_robot=target,
            method=method,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        for task in tasks
        for target in targets
        for method in methods
    ]
    rows = summarize_records(records)

    jsonl_path = Path(args.jsonl)
    md_path = Path(args.md)
    summary_path = Path(args.summary)

    append_jsonl(jsonl_path, records)
    write_summary_csv(summary_path, rows)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(records_to_md([record.to_dict() for record in records]), encoding="utf-8")

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
