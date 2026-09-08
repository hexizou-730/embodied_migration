"""Validate source tasks on the frozen paper seed splits before migration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.paper_benchmark import PAPER_TIERS


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_TASKS: Dict[str, Dict[str, str]] = {
    "pull_cube": {
        "program_path": "maniskill_backend/case_programs/case01_pull_cube.py",
        "adapter_module": "maniskill_backend.baseline_adapters.source_pull_cube",
        "adapter_path": "maniskill_backend/baseline_adapters/source_pull_cube.py",
    },
    "pick_cube": {
        "program_path": "maniskill_backend/case_programs/case03_pick_cube.py",
        "adapter_module": "maniskill_backend.baseline_adapters.source_pick_cube",
        "adapter_path": "maniskill_backend/baseline_adapters/source_pick_cube.py",
    },
    "push_cube": {
        "program_path": "maniskill_backend/case_programs/case05_push_cube.py",
        "adapter_module": "maniskill_backend.baseline_adapters.source_push_cube",
        "adapter_path": "maniskill_backend/baseline_adapters/source_push_cube.py",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_cases(case_ids: Iterable[str]) -> list[FullMigrationCase]:
    """Return one representative case per unique source task."""

    selected: Dict[tuple[str, str], FullMigrationCase] = {}
    for case_id in case_ids:
        case = get_full_migration_case(case_id)
        key = (case.task_id, case.source_robot)
        existing = selected.get(key)
        if existing is not None and existing.source_control_mode != case.source_control_mode:
            raise ValueError(f"Conflicting source control modes for {key!r}.")
        selected.setdefault(key, case)
    return [selected[key] for key in sorted(selected)]


def _request_signature(
    cases: Iterable[FullMigrationCase],
    *,
    tier_id: str,
    sim_backend: str,
    render_backend: str,
) -> Dict[str, Any]:
    tier = PAPER_TIERS[tier_id]
    tasks = []
    for case in cases:
        config = SOURCE_TASKS[case.task_id]
        program = REPO_ROOT / config["program_path"]
        adapter = REPO_ROOT / config["adapter_path"]
        tasks.append(
            {
                "task_id": case.task_id,
                "source_robot": case.source_robot,
                "source_control_mode": case.source_control_mode,
                "program_path": config["program_path"],
                "program_sha256": _sha256(program),
                "adapter_module": config["adapter_module"],
                "adapter_sha256": _sha256(adapter),
                "max_episode_steps": case.max_episode_steps,
            }
        )
    return {
        "tier_id": tier_id,
        "development_seeds": tier.development_seeds,
        "held_out_seeds": tier.held_out_seeds,
        "success_threshold": tier.success_threshold,
        "min_trials_for_accept": tier.min_trials_for_accept,
        "sim_backend": sim_backend,
        "render_backend": render_backend,
        "tasks": tasks,
    }


def _cached_summary(path: Path, request: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != "source_baseline_gate.v1":
        return None
    if payload.get("request") != request:
        return None
    return payload


def _default_runner(args: argparse.Namespace) -> Dict[str, Any]:
    from scripts.pullcube_multiseed_eval import run

    return run(args)


def _runner_output_dir(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_source_baseline_gate(
    case_ids: Iterable[str],
    *,
    tier_id: str,
    output_dir: Path,
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    resume: bool = True,
    runner: Callable[[argparse.Namespace], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Run each unique source task on development and held-out seeds."""

    cases = _task_cases(case_ids)
    if not cases:
        raise ValueError("At least one migration case is required.")
    unknown_tasks = [case.task_id for case in cases if case.task_id not in SOURCE_TASKS]
    if unknown_tasks:
        raise KeyError(f"No source baseline adapter is registered for tasks {unknown_tasks!r}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    request = _request_signature(
        cases,
        tier_id=tier_id,
        sim_backend=sim_backend,
        render_backend=render_backend,
    )
    cached = _cached_summary(summary_path, request) if resume else None
    if cached is not None:
        return {**cached, "reused": True}

    tier = PAPER_TIERS[tier_id]
    run_fn = runner or _default_runner
    rows = []
    for case in cases:
        config = SOURCE_TASKS[case.task_id]
        task_dir = output_dir / case.task_id
        for split, seeds in (
            ("development", tier.development_seeds),
            ("held_out", tier.held_out_seeds),
        ):
            args = argparse.Namespace(
                seeds=seeds,
                task=case.task_id,
                robot=case.source_robot,
                method="source-task-validation",
                code_file=config["program_path"],
                adapter_module=config["adapter_module"],
                control_mode=case.source_control_mode,
                obs_mode="state",
                sim_backend=sim_backend,
                render_backend=render_backend,
                max_episode_steps=case.max_episode_steps,
                output_dir=_runner_output_dir(task_dir),
                jsonl_name=f"{split}.jsonl",
                md_name=f"{split}.md",
                success_threshold=tier.success_threshold,
                min_trials_for_accept=tier.min_trials_for_accept,
            )
            result = run_fn(args)
            summary = result.get("summary") or {}
            strategy = summary.get("generalization_strategy") or {}
            rows.append(
                {
                    "task_id": case.task_id,
                    "source_robot": case.source_robot,
                    "split": split,
                    "seeds": seeds,
                    "num_trials": summary.get("num_trials"),
                    "num_success": summary.get("num_success"),
                    "success_rate": summary.get("success_rate"),
                    "accepted": strategy.get("status") == "accepted",
                    "jsonl": result.get("wrote", {}).get("jsonl"),
                    "markdown": result.get("wrote", {}).get("markdown"),
                }
            )

    payload = {
        "schema": "source_baseline_gate.v1",
        "ready": all(row["accepted"] for row in rows),
        "reused": False,
        "request": request,
        "rows": rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(source_baseline_markdown(payload), encoding="utf-8")
    return payload


def source_baseline_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Source Baseline Gate",
        "",
        "Migration results are accepted only when the frozen source task succeeds on the same seed splits.",
        "",
        f"- ready: `{payload.get('ready')}`",
        f"- reused: `{payload.get('reused')}`",
        "",
        "| Task | Source robot | Split | Seeds | Success | Trials | Accepted |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in payload.get("rows") or []:
        lines.append(
            f"| `{row.get('task_id')}` | `{row.get('source_robot')}` | {row.get('split')} | "
            f"`{row.get('seeds')}` | {row.get('success_rate')} | {row.get('num_trials')} | "
            f"{row.get('accepted')} |"
        )
    lines.append("")
    return "\n".join(lines)
