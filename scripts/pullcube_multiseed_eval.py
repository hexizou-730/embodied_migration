"""Evaluate a PullCube target adapter across multiple random seeds.

This script does not call the LLM and does not regenerate code. It runs the
current target adapter module in the real ManiSkill environment, then writes a
seed-by-seed JSONL log and a compact Markdown summary.

Example:

python scripts/pullcube_multiseed_eval.py \
  --seeds 0-9 \
  --robot xarm6_robotiq \
  --adapter-module maniskill_backend.generated_adapters.case02_xarm6_pull_cube \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.real_runner import run_real_code_trial


DEFAULT_CODE_FILE = "maniskill_backend/case_programs/case01_pull_cube.py"
DEFAULT_ADAPTER_MODULE = "maniskill_backend.generated_adapters.case02_xarm6_pull_cube"


def parse_seeds(text: str) -> List[int]:
    """Parse comma/range seed specs such as '0-4,10,12'."""

    seeds: List[int] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def git_rev() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def file_sha256(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def adapter_module_to_path(module_name: str) -> Path:
    return REPO_ROOT / (module_name.replace(".", "/") + ".py")


def result_digest(result: Dict[str, Any]) -> Dict[str, Any]:
    final_info = result.get("final_info") or {}
    return {
        "seed": result.get("seed"),
        "success": bool(result.get("success")),
        "failure_type": result.get("failure_type"),
        "failure_layer": result.get("failure_layer"),
        "message": result.get("message"),
        "final_info": final_info,
    }


def elapsed_steps(result: Dict[str, Any]) -> int | None:
    final_info = result.get("final_info") or {}
    raw = final_info.get("elapsed_steps")
    if isinstance(raw, list) and raw:
        return int(raw[0])
    if isinstance(raw, (int, float)):
        return int(raw)
    return None


def run(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")

    code_path = REPO_ROOT / args.code_file
    code = code_path.read_text(encoding="utf-8")
    adapter_path = adapter_module_to_path(args.adapter_module)

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / args.jsonl_name
    md_path = output_dir / args.md_name

    metadata = {
        "task_id": args.task,
        "robot_uid": args.robot,
        "method": args.method,
        "adapter_module": args.adapter_module,
        "adapter_path": str(adapter_path.relative_to(REPO_ROOT)),
        "adapter_sha256": file_sha256(adapter_path) if adapter_path.exists() else "missing",
        "code_file": args.code_file,
        "git_rev": git_rev(),
        "control_mode": args.control_mode,
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "max_episode_steps": args.max_episode_steps,
        "seeds": seeds,
    }

    results: List[Dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")
        for seed in seeds:
            print(f"[pullcube-multiseed] seed={seed}", flush=True)
            result = run_real_code_trial(
                task_id=args.task,
                robot_uid=args.robot,
                method=args.method,
                code=code,
                prompt=f"target code file: {args.code_file}",
                seed=seed,
                control_mode=args.control_mode,
                obs_mode=args.obs_mode,
                sim_backend=args.sim_backend,
                render_backend=args.render_backend,
                max_episode_steps=args.max_episode_steps,
                adapter_module=args.adapter_module,
            )
            results.append(result)
            f.write(json.dumps({"type": "trial", **result}, ensure_ascii=False, default=repr) + "\n")
            print(
                f"  success={result.get('success')} message={result.get('message')}",
                flush=True,
            )

    summary = build_summary(metadata, results)
    md_path.write_text(summary_to_markdown(summary), encoding="utf-8")

    payload = {
        "metadata": metadata,
        "summary": summary,
        "wrote": {
            "jsonl": str(jsonl_path.relative_to(REPO_ROOT)),
            "markdown": str(md_path.relative_to(REPO_ROOT)),
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_summary(metadata: Dict[str, Any], results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [result_digest(item) for item in results]
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    steps = [elapsed_steps(item) for item in results]
    good_steps = [step for step in steps if step is not None]
    return {
        **metadata,
        "num_trials": len(rows),
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": round(len(successes) / len(rows), 4) if rows else 0.0,
        "mean_elapsed_steps": round(mean(good_steps), 2) if good_steps else None,
        "rows": rows,
    }


def summary_to_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# PullCube Multi-Seed Evaluation",
        "",
        "This evaluation runs the current target adapter directly in ManiSkill. It does not call the LLM.",
        "",
        "## Setup",
        "",
        f"- task: `{summary['task_id']}`",
        f"- robot: `{summary['robot_uid']}`",
        f"- adapter: `{summary['adapter_module']}`",
        f"- adapter sha256: `{summary['adapter_sha256']}`",
        f"- code file: `{summary['code_file']}`",
        f"- git rev: `{summary['git_rev']}`",
        f"- control mode: `{summary['control_mode']}`",
        f"- max episode steps: `{summary['max_episode_steps']}`",
        "",
        "## Result",
        "",
        f"- trials: `{summary['num_trials']}`",
        f"- successes: `{summary['num_success']}`",
        f"- failures: `{summary['num_failure']}`",
        f"- success rate: `{summary['success_rate']}`",
        f"- mean elapsed steps: `{summary['mean_elapsed_steps']}`",
        "",
        "## Seeds",
        "",
        "| seed | success | failure_type | failure_layer | elapsed_steps | message |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in summary["rows"]:
        final_info = row.get("final_info") or {}
        raw_steps = final_info.get("elapsed_steps")
        if isinstance(raw_steps, list) and raw_steps:
            steps = raw_steps[0]
        else:
            steps = raw_steps if raw_steps is not None else ""
        message = str(row.get("message") or "").replace("|", "\\|")
        lines.append(
            f"| {row['seed']} | {row['success']} | {row.get('failure_type')} | "
            f"{row.get('failure_layer')} | {steps} | {message} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0-9", help="Comma/range list, e.g. '0-9' or '0,1,5'.")
    parser.add_argument("--task", default="pull_cube")
    parser.add_argument("--robot", default="xarm6_robotiq")
    parser.add_argument("--method", default="target-module-generation")
    parser.add_argument("--code-file", default=DEFAULT_CODE_FILE)
    parser.add_argument("--adapter-module", default=DEFAULT_ADAPTER_MODULE)
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--jsonl-name", default="pullcube_xarm6_multiseed.jsonl")
    parser.add_argument("--md-name", default="pullcube_xarm6_multiseed.md")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
