"""Real-physics benchmark runner for ManiSkill trials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .evaluation import TrialRecord
from .real_runner import run_real_trial
from .results import append_jsonl, summarize_records, write_summary_csv
from .tasks import get_task_spec
from .view import records_to_md


def _result_to_record(result: Dict[str, Any], *, source_robot: str, seed: int) -> TrialRecord:
    """Convert the dict that `run_real_trial` returns into a TrialRecord."""
    info_keys = (
        "real_runner",
        "env_id",
        "task_name",
        "task_name_cn",
        "reset_info_keys",
        "execution_log",
        "final_info",
        "used_llm",
        "llm_model",
        "llm_reason",
        "llm_raw_text",
        "graphics_diagnosis",
        "report_source_method",
        "report_source_failure_type",
        "report_source_message",
        "report_source_log",
    )
    info: Dict[str, Any] = {k: result[k] for k in info_keys if k in result}
    return TrialRecord(
        task_id=result["task_id"],
        source_robot=source_robot,
        target_robot=result["robot_uid"],
        method=result["method"],
        seed=seed,
        generated_code=result.get("generated_code", ""),
        success=bool(result.get("success", False)),
        failure_type=result.get("failure_type", "unknown failure"),
        message=str(result.get("message", "")),
        prompt=result.get("prompt", ""),
        info=info,
        failure_report=result.get("failure_report", ""),
    )


def run_real_matrix(
    *,
    task: str,
    robot: str,
    methods: Sequence[str],
    seed: int = 0,
    control_mode: str | None = None,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 500,
    dry_run: bool = False,
) -> List[TrialRecord]:
    spec = get_task_spec(task)
    records: List[TrialRecord] = []
    for method in methods:
        result = run_real_trial(
            task_id=task,
            robot_uid=robot,
            method=method,
            seed=seed,
            control_mode=control_mode,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
            max_episode_steps=max_episode_steps,
            dry_run=dry_run,
        )
        records.append(_result_to_record(result, source_robot=spec.source_robot, seed=seed))
    return records


def _items(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-physics migration matrix.")
    parser.add_argument("--task", default="pick_cube")
    parser.add_argument("--robot", default="panda")
    parser.add_argument(
        "--methods",
        default="source-copy,llm_card_report,oracle",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--jsonl", default="results/real_trials.jsonl")
    parser.add_argument("--md", default="results/real_trials.md")
    parser.add_argument("--summary", default="results/real_summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = run_real_matrix(
        task=args.task,
        robot=args.robot,
        methods=_items(args.methods),
        seed=args.seed,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        dry_run=args.dry_run,
    )
    rows = summarize_records(records)

    jsonl_path = Path(args.jsonl)
    md_path = Path(args.md)
    summary_path = Path(args.summary)

    append_jsonl(jsonl_path, records)
    write_summary_csv(summary_path, rows)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(records_to_md([r.to_dict() for r in records]), encoding="utf-8")

    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
