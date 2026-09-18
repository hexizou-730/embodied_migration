"""Prepare, validate, and freeze benchmark source programs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from maniskill_backend.experiment_environment import load_experiment_contract
from maniskill_backend.source_programs import (
    REPO_ROOT,
    freeze_source_program,
    load_source_program_catalog,
    select_source_programs,
    source_program_status,
    synthesize_source_program,
    validate_source_program,
)
from maniskill_backend.source_preparation import (
    launch_background, preparation_options, run_preparation, use_contract_llm,
)


CONTRACT = load_experiment_contract()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate source programs over multiple seeds and freeze only 100% passes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show candidate and frozen status.")
    status.add_argument("--json", action="store_true")
    status.add_argument("--run-name", default="phase4")
    status.add_argument("--output-root", default="results/source_preparation")

    validate = subparsers.add_parser("validate", help="Run source programs in ManiSkill.")
    validate.add_argument("--tasks", default="all", help="all or comma-separated task IDs")
    validate.add_argument(
        "--seeds",
        default=",".join(map(str, CONTRACT["seed_policy"]["development_seeds"])),
    )
    validate.add_argument("--freeze", action="store_true")
    validate.add_argument("--dry-run", action="store_true")
    validate.add_argument("--obs-mode", default="state")
    validate.add_argument("--sim-backend", default="auto")
    validate.add_argument("--render-backend", default="gpu")
    validate.add_argument("--output-root", default="results/source_validation")
    validate.add_argument("--run-name")

    synthesize = subparsers.add_parser(
        "synthesize", help="Generate and repair missing source programs in simulation."
    )
    synthesize.add_argument("--tasks", required=True, help="comma-separated task IDs")
    synthesize.add_argument(
        "--seeds",
        default=",".join(map(str, CONTRACT["seed_policy"]["development_seeds"])),
    )
    synthesize.add_argument("--max-cycles", type=int, default=5)
    synthesize.add_argument("--freeze", action="store_true")
    synthesize.add_argument("--obs-mode", default="state")
    synthesize.add_argument("--sim-backend", default="auto")
    synthesize.add_argument("--render-backend", default="gpu")
    synthesize.add_argument("--output-root", default="results/source_synthesis")
    synthesize.add_argument("--run-name")

    prepare = subparsers.add_parser(
        "prepare",
        help="Validate, repair failures with the LLM, and freeze full multi-seed passes.",
    )
    prepare.add_argument("--tasks", default="all", help="all or comma-separated task IDs")
    prepare.add_argument(
        "--seeds",
        default=",".join(map(str, CONTRACT["seed_policy"]["development_seeds"])),
    )
    prepare.add_argument("--max-cycles", type=int, default=5)
    prepare.add_argument("--obs-mode", default="state")
    prepare.add_argument("--sim-backend", default="auto")
    prepare.add_argument("--render-backend", default="gpu")
    prepare.add_argument("--output-root", default="results/source_preparation")
    prepare.add_argument("--run-name")
    prepare.add_argument("--background", action="store_true", help="Detach safely from SSH; progress is saved to run.log.")
    prepare.add_argument("--retry-failed", action="store_true", help="Recheck blocked tasks; does not reset the LLM cycle budget.")
    prepare.add_argument("--task-timeout", type=int, default=3600, help="Wall-clock seconds per task, including asset setup.")
    prepare.add_argument("--asset-timeout", type=int, default=900)
    prepare.add_argument("--no-download", action="store_true", help="Mark missing assets blocked instead of downloading them.")
    prepare.add_argument("--keep-llm-settings", action="store_true", help="Check existing environment settings instead of applying the frozen LLM contract.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog, specs = load_source_program_catalog()
    if args.command == "status":
        rows = source_program_status(specs)
        progress_path = REPO_ROOT / args.output_root / args.run_name / "summary.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
        if args.json:
            print(json.dumps({"preparation": progress, "sources": rows}, indent=2, ensure_ascii=False))
        else:
            if progress:
                pid = progress.get("pid")
                try:
                    os.kill(pid, 0)
                    alive = True
                except (TypeError, ProcessLookupError):
                    alive = False
                state = progress.get("state", "legacy_result")
                if state == "running" and not alive:
                    state = "interrupted (rerun prepare to resume)"
                print(f"{args.run_name}: {state}; frozen={progress['frozen']}/{progress['tasks']}; current={progress.get('current_task') or '-'}")
                for item in progress.get("results", []):
                    if not item.get("success"):
                        print(f"  {item['task_id']}: {item['status']}")
            print("task  source          candidate               frozen  seeds")
            for row in rows:
                seeds = ",".join(map(str, row["validated_seeds"])) or "-"
                print(
                    f"{row['task_id']:<22} {row['source_robot']:<15} "
                    f"{row['candidate_status']:<23} {str(row['frozen']):<7} {seeds}"
                )
        return

    selected = select_source_programs(specs, args.tasks)
    seeds = _parse_seeds(args.seeds)
    allowed = set(CONTRACT["seed_policy"]["development_seeds"])
    if not set(seeds) <= allowed:
        raise SystemExit(
            "Source preparation may use only development seeds: "
            + ",".join(map(str, sorted(allowed)))
        )
    run_name = args.run_name or ("phase4" if args.command == "prepare" else datetime.now().strftime("source_%Y%m%d_%H%M%S"))
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise SystemExit("--run-name must be a simple directory name")
    output_dir = REPO_ROOT / args.output_root / run_name
    if args.command == "prepare":
        if sorted(seeds) != sorted(CONTRACT["source_program_validation"]["default_seeds"]):
            raise SystemExit("prepare requires all development seeds; use validate for a partial smoke test.")
        if not selected or min(args.max_cycles, args.task_timeout, args.asset_timeout) <= 0:
            raise SystemExit("Choose tasks and positive cycle/time budgets.")
        if not args.keep_llm_settings:
            use_contract_llm()
        if args.background:
            pid = launch_background([arg for arg in sys.argv[1:] if arg != "--background"], output_dir)
            print(f"Submitted in background: pid={pid}\nlog = {output_dir / 'run.log'}\nCheck: python source.py status --run-name {run_name}")
            return
        print(f"LLM settings: {CONTRACT['llm']['model']} (contract checked before execution)", flush=True)
        batch = run_preparation(preparation_options(args, selected, seeds), output_dir, retry_failed=args.retry_failed)
        print(f"summary = {output_dir / 'summary.json'}")
        if not batch["complete"]:
            raise SystemExit(1)
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.command == "synthesize":
        failed = False
        for spec in selected:
            print(f"[{spec.task_id}] synthesizing {spec.env_id}", flush=True)
            result = synthesize_source_program(
                spec,
                seeds=seeds,
                output_dir=output_dir,
                max_cycles=args.max_cycles,
                obs_mode=args.obs_mode,
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
            )
            if result.get("success") and args.freeze:
                freeze_source_program(
                    spec,
                    result["validation"],
                    minimum_seed_count=int(
                        catalog["freeze_policy"]["minimum_seed_count"]
                    ),
                )
            failed = failed or not result.get("success")
            print(f"  success={result.get('success')}", flush=True)
        print(f"results = {output_dir}")
        if failed:
            raise SystemExit(1)
        return
    if args.dry_run:
        plan = {
            "tasks": [spec.task_id for spec in selected],
            "seeds": seeds,
            "freeze": args.freeze,
            "output": str(output_dir),
        }
        (output_dir / "plan.json").write_text(
            json.dumps(plan, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(plan, indent=2))
        return

    results = []
    for spec in selected:
        print(f"[{spec.task_id}] {spec.env_id} source={spec.source_robot}", flush=True)
        result = validate_source_program(
            spec,
            seeds=seeds,
            output_dir=output_dir,
            obs_mode=args.obs_mode,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            enforce_environment=True,
        )
        if args.freeze and result.get("success"):
            freeze_source_program(
                spec,
                result,
                minimum_seed_count=int(catalog["freeze_policy"]["minimum_seed_count"]),
            )
            result["frozen"] = True
        results.append(result)
        print(
            f"  success={result.get('success')} "
            f"rate={result.get('success_rate', 0):.2f} frozen={result.get('frozen', False)}",
            flush=True,
        )

    summary = {
        "schema": "source_program_batch.v1",
        "tasks": len(results),
        "validated": sum(int(bool(item.get("success"))) for item in results),
        "frozen": sum(int(bool(item.get("frozen"))) for item in results),
        "results": [
            {
                "task_id": item["task"]["task_id"],
                "success": item.get("success"),
                "success_rate": item.get("success_rate", 0.0),
                "frozen": item.get("frozen", False),
                "status": item.get("status", item["candidate"]["status"]),
            }
            for item in results
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"summary = {output_dir / 'summary.json'}")
    if any(not item.get("success") for item in results):
        raise SystemExit(1)


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--seeds must be comma-separated integers") from exc
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise SystemExit("--seeds must contain at least two unique values")
    return seeds


if __name__ == "__main__":
    main()
