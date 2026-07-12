"""Run an online observe-decide-act harness in ManiSkill.

Example:

python scripts/online_harness_runner.py \
  --case case02_pull_cube_panda_to_xarm6 \
  --planner fallback \
  --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.online_harness import run_online_pull_cube_case, write_online_outputs


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case02_pull_cube_panda_to_xarm6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner", choices=("fallback", "llm"), default="fallback")
    parser.add_argument("--segment-steps", type=int, default=8)
    parser.add_argument("--max-online-steps", type=int, default=240)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--adapter-module", default="")
    parser.add_argument("--output-root", default="results/online_harness")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_name = args.run_name or f"{args.case}_{_timestamp()}"
    output_dir = REPO_ROOT / args.output_root / run_name
    result = run_online_pull_cube_case(
        case_id=args.case,
        seed=args.seed,
        planner=args.planner,
        segment_steps=args.segment_steps,
        max_online_steps=args.max_online_steps,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        adapter_module=args.adapter_module,
        dry_run=args.dry_run,
    )
    result["wrote"] = write_online_outputs(output_dir, result)
    latest_path = REPO_ROOT / args.output_root / "latest.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(output_dir), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=repr))
    if not args.dry_run and result.get("success") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
