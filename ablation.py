#!/usr/bin/env python3
"""Run the paired repair-granularity ablation from one shared failed candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from maniskill_backend.repair_granularity import (
    resolve_failed_candidate,
    run_granularity_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("benchmark_tasks/t07_stack_cube/repair_granularity_ablation.json"),
    )
    parser.add_argument(
        "--resume-adapter",
        type=Path,
        help="Shared failed candidate; defaults to the candidate pinned in the ablation study.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Paired runs per arm; the environment seed stays fixed, sampling seeds advance.",
    )
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_cycles < 1:
        parser.error("repetitions and max-cycles must be positive")
    resume_adapter = args.resume_adapter or resolve_failed_candidate(args.study)
    name = args.run_name or datetime.now(timezone.utc).strftime("granularity_%Y%m%d_%H%M%S")
    run_dir = Path("results/repair_granularity") / name
    if run_dir.exists():
        parser.error("Use a new run-name to preserve the existing ablation evidence.")
    summary = run_granularity_ablation(
        study_path=args.study,
        run_dir=run_dir,
        resume_adapter=resume_adapter,
        max_cycles=args.max_cycles,
        seed=args.seed,
        repetitions=args.repetitions,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
