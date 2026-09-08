"""One-command counterexample-guided target-adapter synthesis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.cases import find_full_migration_case, get_full_migration_case
from maniskill_backend.counterexample_loop import CEGISConfig, run_counterexample_loop


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="")
    parser.add_argument("--task", default="pull_cube")
    parser.add_argument("--source", default="panda")
    parser.add_argument("--target", default="xarm6_robotiq")
    parser.add_argument("--development-seeds", default="0-4")
    parser.add_argument("--held-out-seeds", default="100-109")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--attempts-per-cycle", type=int, default=1)
    parser.add_argument("--probe-budget", type=int, default=8)
    parser.add_argument("--probe-strategy", choices=("active", "fixed_grid"), default="active")
    parser.add_argument("--prompt-policy", default="full")
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--min-trials-for-accept", type=int, default=5)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--trial-timeout-s", type=int, default=900)
    parser.add_argument("--test-timeout-s", type=int, default=240)
    parser.add_argument("--output-root", default="results/cegis")
    parser.add_argument("--evidence-root", default="evidence/runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--keep-current-adapter", action="store_true")
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    case = (
        get_full_migration_case(args.case)
        if args.case
        else find_full_migration_case(args.task, args.source, args.target)
    )
    config = CEGISConfig(
        case_id=case.case_id,
        development_seeds=args.development_seeds,
        held_out_seeds=args.held_out_seeds,
        max_cycles=args.max_cycles,
        attempts_per_cycle=args.attempts_per_cycle,
        probe_budget=args.probe_budget,
        probe_strategy=args.probe_strategy,
        prompt_policy=args.prompt_policy,
        success_threshold=args.success_threshold,
        min_trials_for_accept=args.min_trials_for_accept,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        trial_timeout_s=args.trial_timeout_s,
        test_timeout_s=args.test_timeout_s,
        from_zero=not args.keep_current_adapter,
        source_check=not args.no_source_check,
        dry_run=args.dry_run,
        evidence_root=args.evidence_root,
    )
    payload = run_counterexample_loop(
        config,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
    if not args.dry_run and not payload.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
