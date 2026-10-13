"""Repeat a frozen-source migration from scratch and summarize generation stability."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .experiment_environment import capture_experiment_environment
from .source_preparation import run_worker
from .source_programs import REPO_ROOT, _write_json


def run_generation_stability(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run independent target generations; never carry candidates between repetitions."""

    repetitions = int(plan["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "plan.json", dict(plan))
    runtime = capture_experiment_environment(
        {
            "phase": "generation_stability",
            "task_id": plan["task_id"],
            "source_robot": plan["source_robot"],
            "target_robot": plan["target_robot"],
            "seed": plan["seed"],
            "repetitions": repetitions,
        }
    )
    _write_json(output_dir / "runtime_environment.json", runtime)
    summary: dict[str, Any] = {
        "schema": "generation_stability.v1",
        "state": "running",
        "complete": False,
        "pid": os.getpid(),
        "task_id": plan["task_id"],
        "source_robot": plan["source_robot"],
        "target_robot": plan["target_robot"],
        "repetitions": repetitions,
        "seed": plan["seed"],
        "max_cycles": plan["max_cycles"],
        "results": [],
    }
    _checkpoint(output_dir, summary)
    if not runtime.get("validation", {}).get("ok"):
        summary.update(
            state="environment_contract_mismatch",
            environment_mismatches=runtime.get("validation", {}).get("mismatches", []),
        )
        _checkpoint(output_dir, summary)
        return summary

    for repetition in range(1, repetitions + 1):
        repetition_dir = output_dir / f"repetition_{repetition:03d}"
        repetition_dir.mkdir()
        worker = {
            "task_id": plan["task_id"],
            "plan": dict(plan),
            "output_dir": str(repetition_dir.resolve()),
        }
        _write_json(repetition_dir / "worker.json", worker)
        summary["current_repetition"] = repetition
        _checkpoint(output_dir, summary)
        print(f"[repetition {repetition}/{repetitions}] fresh target generation", flush=True)
        result_path = repetition_dir / "worker_result.json"
        with (repetition_dir / "worker.log").open("a", encoding="utf-8") as log:
            returncode, timed_out = run_worker(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "maniskill_backend.target_preparation",
                    str(repetition_dir / "worker.json"),
                ],
                log=log,
                timeout=int(plan["task_timeout"]),
            )
        if result_path.is_file() and returncode == 0:
            worker_result = _read_json(result_path)
        else:
            worker_result = {
                "success": False,
                "status": "worker_timeout" if timed_out else "worker_crashed",
                "returncode": returncode,
            }
        metrics = worker_result.get("metrics")
        metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
        row = {
            "repetition": repetition,
            "success": bool(worker_result.get("success")),
            "status": worker_result.get("status", "unknown"),
            "message": worker_result.get("message", ""),
            "generation_cycles": int(metrics.get("generation_cycles") or 0),
            "repair_cycles": int(metrics.get("repair_cycles") or 0),
            "target_action_steps": int(metrics.get("target_action_steps") or 0),
            "total_tokens": int(metrics.get("total_tokens") or 0),
            "llm_cost_usd": float(metrics.get("llm_cost_usd") or 0.0),
            "result": str(result_path),
            "details": worker_result.get("details"),
            "target_adapter": worker_result.get("target_adapter"),
        }
        summary["results"].append(row)
        summary["current_repetition"] = None
        _checkpoint(output_dir, summary)
        print(
            f"  status={row['status']} success={row['success']} "
            f"cycles={row['generation_cycles']} cost=${row['llm_cost_usd']:.4f}",
            flush=True,
        )

    summary.update(state="complete", complete=True, current_repetition=None)
    _checkpoint(output_dir, summary)
    return summary


def _checkpoint(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = summary["results"]
    successes = [row for row in rows if row["success"]]
    completed = len(rows)
    summary.update(
        completed_repetitions=completed,
        successful_repetitions=len(successes),
        generation_success_rate=(len(successes) / completed) if completed else 0.0,
        total_generation_cycles=sum(row["generation_cycles"] for row in rows),
        total_target_action_steps=sum(row["target_action_steps"] for row in rows),
        total_tokens=sum(row["total_tokens"] for row in rows),
        total_llm_cost_usd=round(sum(row["llm_cost_usd"] for row in rows), 8),
        mean_cycles_to_success=(
            round(mean(row["generation_cycles"] for row in successes), 3)
            if successes
            else None
        ),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", summary)


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Generation Stability",
        "",
        f"- state: `{summary['state']}`",
        f"- task: `{summary['task_id']}`",
        f"- migration: `{summary['source_robot']} -> {summary['target_robot']}`",
        f"- fixed simulation seed: `{summary['seed']}`",
        f"- success: `{summary.get('successful_repetitions', 0)}/{summary['repetitions']}`",
        f"- generation success rate: `{float(summary.get('generation_success_rate') or 0.0):.3f}`",
        f"- total tokens: `{summary.get('total_tokens', 0)}`",
        f"- total LLM cost: `${float(summary.get('total_llm_cost_usd') or 0.0):.6f}`",
        "",
        "| repetition | success | status | cycles | repairs | actions | tokens | cost USD |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("results", []):
        lines.append(
            f"| {row['repetition']} | {row['success']} | {row['status']} | "
            f"{row['generation_cycles']} | {row['repair_cycles']} | "
            f"{row['target_action_steps']} | {row['total_tokens']} | "
            f"{row['llm_cost_usd']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}

