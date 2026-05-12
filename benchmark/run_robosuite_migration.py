"""Small benchmark for complex robosuite source-to-target code migration.

This benchmark is intentionally separate from `benchmark.run_benchmark` because
the robosuite backend evaluates source-program migration on complex task-level
skills rather than early tabletop object-position checks.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv

from llm_client import make_client
from robosuite_backend.migration import run_migration_trial
from robosuite_backend.profiles import profile_names
from robosuite_backend.tasks import get_task, task_names


def analyze_robosuite_code(code: str) -> Dict[str, object]:
    return {
        "lines_of_code": len([ln for ln in code.splitlines() if ln.strip()]),
        "used_navigate": "navigate_to_station" in code,
        "used_grip_force": "set_grip_force" in code,
        "used_pot_lift": "lift_pot" in code,
        "used_handover_pose": "move_to_handover_pose" in code,
        "used_handover": "handover_object" in code,
        "used_peg_alignment": "align_peg_to_hole" in code,
        "used_insert_peg": "insert_peg" in code,
        "used_refusal": "refuse_" in code,
        "checked_return": "if " in code or " and " in code,
    }


def run_suite(
    tasks: Iterable[str],
    targets: Iterable[str],
    planners: Iterable[str],
    attempts: int,
    out_dir: Path,
    use_card: bool = True,
    use_failure_report: bool = True,
) -> List[Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    rows: List[Dict[str, object]] = []
    for planner in planners:
        if planner == "llm" and client is None:
            client = make_client()
        for task_name in tasks:
            task = get_task(task_name)
            for target in targets:
                if target == task.source_robot:
                    continue
                print(f"\n[{planner}] {task_name}: {task.source_robot} -> {target}")
                result = run_migration_trial(
                    task_name=task_name,
                    target_name=target,
                    client=client,
                    planner=planner,
                    max_attempts=attempts,
                    use_card=use_card,
                    use_failure_report=use_failure_report,
                    verbose=False,
                )
                record = asdict(result)
                trial_path = out_dir / f"{planner}_{task_name}_{target}.json"
                trial_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                final_code = result.attempts[-1].code if result.attempts else ""
                row = {
                    "planner": planner,
                    "task": task_name,
                    "source": result.source,
                    "target": target,
                    "success": result.success,
                    "final_reason": result.final_reason,
                    "attempts": len(result.attempts),
                    **analyze_robosuite_code(final_code),
                    "trial_json": str(trial_path),
                }
                rows.append(row)
                print(f"  -> success={result.success} reason={result.final_reason}")

    if rows:
        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {summary_path}")
    return rows


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", choices=task_names(), default=task_names())
    ap.add_argument("--targets", nargs="+", choices=profile_names(),
                    default=["rs_dual_iiwa", "rs_baxter", "rs_mobile_tiago"])
    ap.add_argument("--planners", nargs="+", choices=["source-copy", "oracle", "llm"],
                    default=["source-copy", "oracle"],
                    help="Use llm only when OPENROUTER_API_KEY is configured.")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--no-card", action="store_true",
                    help="For planner=llm, omit source/target capability cards.")
    ap.add_argument("--no-retry", action="store_true",
                    help="For planner=llm, disable Failure Report retry.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default="results/robosuite_runs")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_id
    run_suite(
        args.tasks,
        args.targets,
        args.planners,
        args.attempts,
        out_dir,
        use_card=not args.no_card,
        use_failure_report=not args.no_retry,
    )


if __name__ == "__main__":
    main()
