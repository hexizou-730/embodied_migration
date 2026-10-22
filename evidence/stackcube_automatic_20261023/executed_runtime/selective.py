#!/usr/bin/env python3
"""Run the StackCube stage-preserving migration pilot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from maniskill_backend.selective_migration import run_selective_migration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("benchmark_tasks/t07_stack_cube/selective_migration_study.json"),
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-adapter", type=Path, help="Replay a saved candidate, then repair from its measured failure.")
    args = parser.parse_args()
    name = args.run_name or datetime.now(timezone.utc).strftime("stackcube_%Y%m%d_%H%M%S")
    run_dir = Path("results/selective_migration") / name
    result = run_selective_migration(
        study_path=args.study,
        run_dir=run_dir,
        max_cycles=args.max_cycles,
        seed=args.seed,
        dry_run=args.dry_run,
        resume_adapter=args.resume_adapter,
    )
    print(json.dumps({
        "status": result["status"],
        "success": result["success"],
        "cycles": len(result["cycles"]),
        "locked_stages": result["locked_stages"],
        "next_repair_stage": result["next_repair_stage"],
        "summary": str(run_dir / "summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
