"""Build an autonomous harness plan for simulation-in-the-loop repair.

The harness does not give an LLM raw simulator access. It exposes bounded
commands: run the real ManiSkill trial, run multi-seed evaluation, run a
structured probe, and then run adapter generation with the measured feedback.

Example:

python scripts/autonomous_harness_runner.py \
  --case case02_pull_cube_panda_to_xarm6 \
  --multiseed-jsonl results/pullcube_xarm6_multiseed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.autonomous_harness import build_harness_plan, write_harness_plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Migration case id.")
    parser.add_argument(
        "--multiseed-jsonl",
        default="",
        help="Latest multi-seed JSONL result. If omitted, the harness starts from a single-seed run.",
    )
    parser.add_argument(
        "--probe-json",
        default="",
        help="Optional structured probe JSON to expose to the agent.",
    )
    parser.add_argument(
        "--no-existing-probe",
        action="store_true",
        help="Ignore default results/structured_probes/<case>/*.json files.",
    )
    parser.add_argument(
        "--seed-policy",
        choices=("auto", "near_contact", "severe_reachability", "first"),
        default="auto",
        help="How to choose the next failed seed for probing.",
    )
    parser.add_argument("--seeds", default="0-9", help="Seed range shown in the multi-seed tool command.")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="results/autonomous_harness")
    parser.add_argument("--print-agent-observation", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle = build_harness_plan(
        case_id=args.case,
        multiseed_jsonl=args.multiseed_jsonl or None,
        probe_json=args.probe_json or None,
        include_existing_probe=not args.no_existing_probe,
        seed_policy=args.seed_policy,
        seeds=args.seeds,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
    )
    wrote = write_harness_plan(args.output_dir, bundle)
    human_report = bundle.get("human_report") or {}
    payload = {"wrote": wrote, "human_suggested_next_action": human_report.get("suggested_next_action")}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.print_agent_observation:
        print("\n--- agent observation ---")
        print(json.dumps(bundle.get("agent_observation") or {}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
