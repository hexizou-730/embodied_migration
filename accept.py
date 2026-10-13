"""Run and verify one real source-to-target migration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from maniskill_backend.acceptance import verify_acceptance
from maniskill_backend.source_programs import REPO_ROOT, load_source_program_catalog, source_program_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one end-to-end migration acceptance test.")
    parser.add_argument("target_robot", help="ManiSkill target robot UID, e.g. xarm6 or xarm6_robotiq")
    parser.add_argument("--task", default="t01_push_cube", help="One valid frozen task ID")
    parser.add_argument("--run-name", help="Defaults to acceptance_<task>_<target>")
    parser.add_argument("--output-root", default="results/target_migration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Only verify an existing run")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target = args.target_robot.strip().lower().replace("-", "_")
    if target == "xarm6":
        target = "xarm6_robotiq"
    run_name = args.run_name or f"acceptance_{args.task}_{target}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise SystemExit("--run-name must be a simple directory name")
    run_dir = REPO_ROOT / args.output_root / run_name

    if not args.verify_only:
        _, specs = load_source_program_catalog()
        rows = {row["task_id"]: row for row in source_program_status(specs)}
        row = rows.get(args.task)
        if row is None:
            raise SystemExit(f"Unknown task: {args.task}")
        if not row["frozen"]:
            raise SystemExit(
                f"{args.task} is not ready ({row['freeze_status']}). "
                "Run `python source.py status --run-name phase4_refresh`; "
                "use `python source.py refresh --tasks " + args.task + "` only for stale evidence."
            )
        command = [
            sys.executable,
            str(REPO_ROOT / "target.py"),
            "migrate",
            target,
            "--tasks",
            args.task,
            "--run-name",
            run_name,
            "--output-root",
            args.output_root,
            "--seed",
            str(args.seed),
            "--max-cycles",
            str(args.max_cycles),
            "--max-episode-steps",
            str(args.max_episode_steps),
        ]
        if args.retry_failed:
            command.append("--retry-failed")
        if args.no_download:
            command.append("--no-download")
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode and not (run_dir / "summary.json").is_file():
            raise SystemExit(completed.returncode)

    report = verify_acceptance(run_dir, task_id=args.task, target_robot=target)
    print(f"acceptance = {'PASS' if report['passed'] else 'FAIL'}")
    print(f"report = {run_dir / 'acceptance_report.json'}")
    if not report["passed"]:
        print("failed_checks = " + ", ".join(report["failed_checks"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
