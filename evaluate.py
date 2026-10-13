"""Run Phase 6: fixed-adapter multi-seed evaluation without LLM repair."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from maniskill_backend.adapter_evaluation import evaluate_fixed_adapter
from maniskill_backend.experiment_environment import load_experiment_contract
from maniskill_backend.source_programs import REPO_ROOT


CONTRACT = load_experiment_contract()
DEFAULT_EVIDENCE = "evidence/t01_push_cube_panda_to_xarm6_20261013"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen generated adapter over multiple ManiSkill seeds."
    )
    parser.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--seeds",
        default="heldout",
        help="heldout, development, or a comma-separated integer list",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--output-root", default="results/adapter_evaluation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evidence = (REPO_ROOT / args.evidence).resolve()
    seeds = _parse_seeds(args.seeds)
    run_name = args.run_name or datetime.now().strftime("t01_xarm6_%Y%m%d_%H%M%S")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise SystemExit("--run-name must be a simple directory name")
    output = REPO_ROOT / args.output_root / run_name
    result = evaluate_fixed_adapter(
        evidence_dir=evidence,
        output_dir=output,
        seeds=seeds,
    )
    print(f"state = {result['state']}")
    print(f"success = {result.get('num_success', 0)}/{result.get('num_trials', 0)}")
    print(f"success_rate = {float(result.get('success_rate') or 0.0):.3f}")
    print(f"summary = {output / 'summary.md'}")
    if result["state"] != "complete":
        raise SystemExit(1)


def _parse_seeds(value: str) -> tuple[int, ...]:
    normalized = str(value).strip().lower()
    policy = CONTRACT["seed_policy"]
    if normalized == "heldout":
        return tuple(int(seed) for seed in policy["held_out_seeds"])
    if normalized == "development":
        return tuple(int(seed) for seed in policy["development_seeds"])
    try:
        seeds = tuple(int(item.strip()) for item in normalized.split(",") if item.strip())
    except ValueError as exc:
        raise SystemExit("--seeds must be heldout, development, or comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must contain unique integers")
    return seeds


if __name__ == "__main__":
    main()
