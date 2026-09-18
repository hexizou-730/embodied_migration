"""Migrate validated frozen source programs to one ManiSkill target robot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from maniskill_backend.experiment_environment import load_experiment_contract
from maniskill_backend.source_preparation import use_contract_llm
from maniskill_backend.source_programs import REPO_ROOT, load_source_program_catalog
from maniskill_backend.target_preparation import (
    frozen_source_specs,
    launch_background,
    migration_plan,
    run_migrations,
)


CONTRACT = load_experiment_contract()
DEFAULTS = CONTRACT["migration_defaults"]


def _normalize_robot(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return CONTRACT["robot_uid_policy"]["normalize_aliases"].get(text, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5: migrate frozen source programs to a target robot.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate", help="Run target-interface checks and bounded LLM repair.")
    status = subparsers.add_parser("status", help="Show one target migration batch.")
    for command in (migrate, status):
        command.add_argument("target_robot", help="ManiSkill robot UID, e.g. xarm6_robotiq.")
        command.add_argument("--run-name")
        command.add_argument("--output-root", default="results/target_migration")
    migrate.add_argument("--tasks", default="frozen", help="frozen or comma-separated frozen task IDs")
    migrate.add_argument("--seed", type=int, default=CONTRACT["seed_policy"]["simulation_default"])
    migrate.add_argument("--max-cycles", type=int, default=5)
    migrate.add_argument(
        "--max-episode-steps",
        type=int,
        default=CONTRACT["source_program_validation"]["max_episode_steps"],
    )
    migrate.add_argument("--target-control-mode", default=DEFAULTS["target_control_mode"])
    migrate.add_argument("--obs-mode", default=DEFAULTS["obs_mode"])
    migrate.add_argument("--sim-backend", default=DEFAULTS["sim_backend"])
    migrate.add_argument("--render-backend", default=DEFAULTS["render_backend"])
    migrate.add_argument("--task-timeout", type=int, default=3600)
    migrate.add_argument("--asset-timeout", type=int, default=900)
    migrate.add_argument("--no-download", action="store_true")
    migrate.add_argument("--background", action="store_true")
    migrate.add_argument("--retry-failed", action="store_true")
    migrate.add_argument("--keep-llm-settings", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    target = _normalize_robot(args.target_robot)
    run_name = args.run_name or f"phase5_{target}"
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("--run-name must be a simple directory name")
    output_dir = REPO_ROOT / args.output_root / run_name

    if args.command == "status":
        path = output_dir / "summary.json"
        if not path.exists():
            print(f"{run_name}: not started")
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
            state = "interrupted (rerun migrate to resume)"
        print(
            f"{run_name}: {state}; success={summary.get('succeeded', 0)}/"
            f"{summary.get('tasks', 0)}; current={summary.get('current_task') or '-'}"
        )
        for row in summary.get("results", []):
            metrics = row.get("metrics") or {}
            print(
                f"  {row['task_id']}: {row.get('status')} "
                f"cycles={metrics.get('generation_cycles', 0)} "
                f"steps={metrics.get('target_action_steps', 0)} "
                f"cost=${metrics.get('llm_cost_usd', 0):.4f}"
            )
        return

    if min(args.max_cycles, args.max_episode_steps, args.task_timeout, args.asset_timeout) <= 0:
        parser.error("Cycle, episode, and timeout budgets must be positive.")
    if args.seed not in CONTRACT["seed_policy"]["development_seeds"]:
        parser.error("Migration repair may use only configured development seeds.")
    _, all_specs = load_source_program_catalog()
    try:
        specs = frozen_source_specs(all_specs, args.tasks)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.keep_llm_settings:
        use_contract_llm()
    plan = migration_plan(args, specs, target)
    if args.background:
        pid = launch_background([arg for arg in sys.argv[1:] if arg != "--background"], output_dir)
        print(
            f"Submitted in background: pid={pid}\nlog = {output_dir / 'run.log'}\n"
            f"Check: python target.py status {target} --run-name {run_name}"
        )
        return
    result = run_migrations(plan, output_dir, retry_failed=args.retry_failed)
    print(f"summary = {output_dir / 'summary.json'}")
    print(f"success = {result.get('succeeded', 0)}/{result.get('tasks', 0)}")
    if result.get("succeeded") != result.get("tasks"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
