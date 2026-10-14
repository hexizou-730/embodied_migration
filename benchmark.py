"""Phase 8 command: compare adapter-generation and repair methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from maniskill_backend.dynamic_harness import REPAIR_METHODS
from maniskill_backend.experiment_environment import load_experiment_contract
from maniskill_backend.method_benchmark import run_method_benchmark
from maniskill_backend.source_preparation import use_contract_llm
from maniskill_backend.source_programs import (
    REPO_ROOT,
    _sha256_file,
    load_source_program_catalog,
    source_program_status,
)


CONTRACT = load_experiment_contract()
DEFAULTS = CONTRACT["migration_defaults"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare target-adapter repair methods.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run a controlled method comparison.")
    status = subparsers.add_parser("status", help="Show benchmark progress and results.")
    for command in (run, status):
        command.add_argument("--run-name")
        command.add_argument("--output-root", default="results/method_benchmark")
    run.add_argument("target_robot", help="ManiSkill target robot UID, e.g. xarm6")
    run.add_argument("--task", default="t01_push_cube")
    run.add_argument("--methods", default=",".join(REPAIR_METHODS))
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--max-cycles", type=int, default=8)
    run.add_argument("--max-episode-steps", type=int, default=1000)
    run.add_argument("--task-timeout", type=int, default=3600)
    run.add_argument("--asset-timeout", type=int, default=900)
    run.add_argument("--target-control-mode", default=DEFAULTS["target_control_mode"])
    run.add_argument("--obs-mode", default=DEFAULTS["obs_mode"])
    run.add_argument("--sim-backend", default=DEFAULTS["sim_backend"])
    run.add_argument("--render-backend", default=DEFAULTS["render_backend"])
    run.add_argument("--no-download", action="store_true")
    run.add_argument("--background", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "status":
        _status(args, parser)
        return

    target = _normalize_robot(args.target_robot)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [item for item in methods if item not in REPAIR_METHODS]
    if unknown:
        parser.error(f"Unknown methods: {', '.join(unknown)}")
    run_name = args.run_name or datetime.now().strftime(
        f"phase8_{args.task}_{target}_%Y%m%d_%H%M%S"
    )
    _validate_run_name(run_name, parser)
    output_dir = REPO_ROOT / args.output_root / run_name
    if min(args.repetitions, args.max_cycles, args.max_episode_steps, args.task_timeout) <= 0:
        parser.error("Repetition, cycle, episode, and timeout budgets must be positive.")
    if args.seed not in CONTRACT["seed_policy"]["development_seeds"]:
        parser.error("Method development comparison may use only development seeds.")

    _, specs = load_source_program_catalog()
    by_id = {spec.task_id: spec for spec in specs}
    spec = by_id.get(args.task)
    if spec is None:
        parser.error(f"Unknown task: {args.task}")
    status = source_program_status([spec])[0]
    if not status["frozen"]:
        parser.error(f"{args.task} is not a valid frozen source ({status['freeze_status']}).")
    use_contract_llm()
    plan = {
        "schema": "method_benchmark_plan.v1",
        "task_id": spec.task_id,
        "source_robot": spec.source_robot,
        "target_robot": target,
        "target_control_mode": args.target_control_mode,
        "seed": args.seed,
        "methods": methods,
        "repetitions": args.repetitions,
        "max_cycles": args.max_cycles,
        "max_episode_steps": args.max_episode_steps,
        "task_timeout": args.task_timeout,
        "asset_timeout": args.asset_timeout,
        "download_assets": not args.no_download,
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "source_hashes": {spec.task_id: _sha256_file(spec.frozen_path)},
        "contract_sha256": _sha256_file(REPO_ROOT / "experiment_config.json"),
    }
    if args.background:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_dir.parent / f"{run_name}.log"
        command = [arg for arg in sys.argv[1:] if arg != "--background"]
        if "--run-name" not in command:
            command.extend(["--run-name", run_name])
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(REPO_ROOT / "benchmark.py"), *command],
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Submitted: pid={process.pid}")
        print(f"log = {log_path}")
        print(f"Check: python benchmark.py status --run-name {run_name}")
        return

    result = run_method_benchmark(plan, output_dir)
    print(f"state = {result['state']}")
    for method_id, row in result.get("method_summary", {}).items():
        print(f"{method_id}: {row['successes']}/{row['completed']}")
    print(f"summary = {output_dir / 'summary.md'}")
    if result["state"] != "complete":
        raise SystemExit(1)


def _status(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.run_name:
        parser.error("status requires --run-name")
    _validate_run_name(args.run_name, parser)
    output_dir = REPO_ROOT / args.output_root / args.run_name
    path = output_dir / "summary.json"
    if not path.is_file():
        print(f"{args.run_name}: not started")
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    pid = summary.get("pid")
    try:
        os.kill(pid, 0)
        alive = True
    except (TypeError, ProcessLookupError):
        alive = False
    state = summary.get("state", "unknown")
    if state == "running" and not alive:
        state = "interrupted"
    print(
        f"{args.run_name}: {state}; completed={summary.get('completed_runs', 0)}/"
        f"{summary.get('total_runs', 0)}; current={summary.get('current_run') or '-'}"
    )
    for method_id in summary.get("methods", []):
        row = summary.get("method_summary", {}).get(method_id, {})
        print(
            f"  {method_id}: success={row.get('successes', 0)}/"
            f"{row.get('completed', 0)} cost=${float(row.get('total_llm_cost_usd') or 0.0):.4f}"
        )


def _normalize_robot(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return CONTRACT["robot_uid_policy"]["normalize_aliases"].get(text, text)


def _validate_run_name(value: str, parser: argparse.ArgumentParser) -> None:
    if Path(value).name != value or value in {".", ".."}:
        parser.error("--run-name must be a simple directory name")


if __name__ == "__main__":
    main()
