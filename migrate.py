"""Single entrypoint for source-code-to-robot migration.

Example:

    python migrate.py examples/panda_pull.py xarm6_robotiq
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

from maniskill_backend.dynamic_harness import (
    DynamicMigrationSpec,
    read_source_actions,
    run_dynamic_agent_migration,
)
from maniskill_backend.experiment_environment import load_experiment_contract


REPO_ROOT = Path(__file__).resolve().parent
EXPERIMENT_CONTRACT = load_experiment_contract()
MIGRATION_DEFAULTS = EXPERIMENT_CONTRACT["migration_defaults"]
SEED_POLICY = EXPERIMENT_CONTRACT["seed_policy"]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_robot(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = EXPERIMENT_CONTRACT["robot_uid_policy"]["normalize_aliases"]
    return aliases.get(text, text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify existing source actions, discover a ManiSkill target robot, "
            "and synthesize a fresh target adapter through execution feedback."
        )
    )
    parser.add_argument("source_file", help="Python file containing source actions.")
    parser.add_argument("target_robot", help="ManiSkill robot UID, for example xarm6_robotiq.")
    parser.add_argument("--env", dest="env_id", help="Override ENV_ID from the source file.")
    parser.add_argument("--source", help="Override SOURCE_ROBOT from the source file.")
    parser.add_argument("--source-control-mode", help="Override source CONTROL_MODE.")
    parser.add_argument(
        "--target-control-mode", default=MIGRATION_DEFAULTS["target_control_mode"]
    )
    parser.add_argument("--seed", type=int, default=SEED_POLICY["simulation_default"])
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument(
        "--max-episode-steps", type=int, default=MIGRATION_DEFAULTS["max_episode_steps"]
    )
    parser.add_argument("--obs-mode", default=MIGRATION_DEFAULTS["obs_mode"])
    parser.add_argument("--sim-backend", default=MIGRATION_DEFAULTS["sim_backend"])
    parser.add_argument("--render-backend", default=MIGRATION_DEFAULTS["render_backend"])
    parser.add_argument("--output-root", default="results/migrations")
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        source = read_source_actions(Path(args.source_file).expanduser())
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))

    env_id = args.env_id or source.env_id
    source_robot = args.source or source.robot_uid
    if not env_id:
        parser.error("Specify ENV_ID in the source file or pass --env.")
    if not source_robot:
        parser.error("Specify SOURCE_ROBOT in the source file or pass --source.")
    if args.max_cycles < 1 or args.max_episode_steps < 1:
        parser.error("--max-cycles and --max-episode-steps must be positive.")
    if args.seed not in SEED_POLICY["development_seeds"]:
        parser.error(
            "Migration repair may use only development seeds from experiment_config.json; "
            f"got --seed={args.seed}."
        )

    spec = DynamicMigrationSpec(
        env_id=env_id,
        task_label=env_id,
        source_robot=_normalize_robot(source_robot),
        target_robot=_normalize_robot(args.target_robot),
        source_control_mode=(
            args.source_control_mode
            or source.control_mode
            or MIGRATION_DEFAULTS["source_control_mode"]
        ),
        target_control_mode=args.target_control_mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        max_source_cycles=0,
        max_target_cycles=args.max_cycles,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    run_name = args.run_name or f"{Path(source.path).stem}_to_{spec.target_robot}_{_timestamp()}"
    run_dir = REPO_ROOT / args.output_root / run_name
    if run_dir.exists():
        parser.error(f"Run directory already exists: {run_dir}. Choose a new --run-name.")
    run_dir.mkdir(parents=True)

    print(f"{spec.env_id}: {spec.source_robot} -> {spec.target_robot}", flush=True)
    print(f"Log: {run_dir / 'run.log'}", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = run_dynamic_agent_migration(
                spec=spec,
                run_dir=run_dir,
                dry_run=args.dry_run,
                source_actions=source,
                progress=lambda message: print(message, file=sys.__stdout__, flush=True),
                enforce_environment=True,
            )

    print(f"status = {result['status']}")
    print(f"success = {result['success']}")
    if result.get("message"):
        print(f"message = {result['message']}")
    if result.get("target_adapter"):
        print(f"target_code = {result['target_adapter']}")
    if result.get("runtime_environment"):
        print(f"environment = {result['runtime_environment']}")
    print(f"details = {run_dir / 'dynamic_summary.json'}")
    if not args.dry_run and result.get("success") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
