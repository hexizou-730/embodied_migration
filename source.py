"""Prepare, validate, and freeze benchmark source programs."""

from __future__ import annotations

import argparse
import json
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


CONTRACT = load_experiment_contract()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate source programs over multiple seeds and freeze only 100% passes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show candidate and frozen status.")
    status.add_argument("--json", action="store_true")

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog, specs = load_source_program_catalog()
    if args.command == "status":
        rows = source_program_status(specs)
        if args.json:
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        else:
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
    run_name = args.run_name or datetime.now().strftime("source_%Y%m%d_%H%M%S")
    output_dir = REPO_ROOT / args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    if args.command == "prepare":
        rows = []
        for spec in selected:
            print(f"[{spec.task_id}] validating candidate", flush=True)
            validation = validate_source_program(
                spec,
                seeds=seeds,
                output_dir=output_dir / "initial_validation",
                obs_mode=args.obs_mode,
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
                enforce_environment=True,
            )
            if validation.get("status") == "environment_contract_mismatch":
                row = {
                    "task_id": spec.task_id,
                    "success": False,
                    "path": "environment_contract_mismatch",
                }
            elif validation.get("success"):
                freeze_source_program(
                    spec,
                    validation,
                    minimum_seed_count=int(
                        catalog["freeze_policy"]["minimum_seed_count"]
                    ),
                )
                row = {
                    "task_id": spec.task_id,
                    "success": True,
                    "path": "validated_existing_candidate",
                }
            else:
                print(f"[{spec.task_id}] repairing failed candidate", flush=True)
                synthesis = synthesize_source_program(
                    spec,
                    seeds=seeds,
                    output_dir=output_dir / "repair",
                    max_cycles=args.max_cycles,
                    obs_mode=args.obs_mode,
                    sim_backend=args.sim_backend,
                    render_backend=args.render_backend,
                    initial_validation=validation,
                )
                if synthesis.get("success"):
                    freeze_source_program(
                        spec,
                        synthesis["validation"],
                        minimum_seed_count=int(
                            catalog["freeze_policy"]["minimum_seed_count"]
                        ),
                    )
                row = {
                    "task_id": spec.task_id,
                    "success": bool(synthesis.get("success")),
                    "path": "llm_repair",
                    "cycles": len(synthesis.get("cycles") or []),
                }
            rows.append(row)
            print(f"  frozen={row['success']} path={row['path']}", flush=True)
        batch = {
            "schema": "source_program_preparation.v1",
            "tasks": len(rows),
            "frozen": sum(int(row["success"]) for row in rows),
            "complete": all(row["success"] for row in rows),
            "results": rows,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(batch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"summary = {output_dir / 'summary.json'}")
        if not batch["complete"]:
            raise SystemExit(1)
        return
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
