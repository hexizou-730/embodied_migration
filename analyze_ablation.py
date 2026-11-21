#!/usr/bin/env python3
"""Recompute and analyze a repair-granularity ablation run from raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maniskill_backend.granularity_analysis import analyze_ablation_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Ablation run directory, e.g. results/repair_granularity/granularity_ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write granularity_analysis.json/.md; defaults to the run directory.",
    )
    args = parser.parse_args()
    payload = analyze_ablation_run(args.run_dir, output_dir=args.output_dir)
    print(json.dumps({
        "failed_checks": payload["evidence_checks"]["failed"],
        "num_pairs": payload["paired"]["num_pairs"],
        "arms": [
            {
                "arm_id": row["arm_id"],
                "num_success": row["num_success"],
                "num_runs": row["num_runs"],
                "success_wilson_95": row["success_wilson_95"],
                "target_action_steps": row["target_action_steps"],
                "total_tokens": row["total_tokens"],
            }
            for row in payload["arms"]
        ],
        "sign_test_p_two_sided": payload["paired"]["sign_test_p_two_sided"],
        "action_difference_bootstrap_95": payload["paired"]["action_difference_bootstrap_95"],
        "token_difference_bootstrap_95": payload["paired"]["token_difference_bootstrap_95"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
