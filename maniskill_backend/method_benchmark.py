"""Run a controlled comparison of target-adapter repair methods."""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .dynamic_harness import REPAIR_METHODS
from .experiment_environment import capture_experiment_environment
from .source_preparation import run_worker
from .source_programs import _write_json


def run_method_benchmark(plan: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run every method from scratch under one frozen experimental plan."""

    methods = [str(item) for item in plan["methods"]]
    unknown = [item for item in methods if item not in REPAIR_METHODS]
    if unknown:
        raise ValueError(f"Unknown benchmark methods: {', '.join(unknown)}")
    repetitions = int(plan["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "plan.json", dict(plan))
    runtime = capture_experiment_environment(
        {
            "phase": "method_benchmark",
            "task_id": plan["task_id"],
            "source_robot": plan["source_robot"],
            "target_robot": plan["target_robot"],
            "seed": plan["seed"],
            "methods": methods,
            "repetitions": repetitions,
        }
    )
    _write_json(output_dir / "runtime_environment.json", runtime)
    summary: dict[str, Any] = {
        "schema": "method_benchmark.v1",
        "state": "running",
        "complete": False,
        "pid": os.getpid(),
        "task_id": plan["task_id"],
        "source_robot": plan["source_robot"],
        "target_robot": plan["target_robot"],
        "seed": plan["seed"],
        "methods": methods,
        "repetitions": repetitions,
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

    total = repetitions * len(methods)
    completed = 0
    for repetition in range(1, repetitions + 1):
        for method_id in methods:
            completed += 1
            run_id = f"{method_id}__r{repetition:03d}"
            run_dir = output_dir / run_id
            run_dir.mkdir()
            worker_plan = dict(plan)
            worker_plan["method_id"] = method_id
            worker = {
                "task_id": plan["task_id"],
                "plan": worker_plan,
                "output_dir": str(run_dir.resolve()),
            }
            _write_json(run_dir / "worker.json", worker)
            summary["current_run"] = run_id
            _checkpoint(output_dir, summary)
            print(f"[{completed}/{total}] {method_id}, repetition {repetition}", flush=True)
            result_path = run_dir / "worker_result.json"
            with (run_dir / "worker.log").open("a", encoding="utf-8") as log:
                returncode, timed_out = run_worker(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "maniskill_backend.target_preparation",
                        str(run_dir / "worker.json"),
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
                "method_id": method_id,
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
            row.update(_evidence_fields(worker_result))
            summary["results"].append(row)
            summary["current_run"] = None
            _checkpoint(output_dir, summary)
            print(
                f"  success={row['success']} cycles={row['generation_cycles']} "
                f"actions={row['target_action_steps']} cost=${row['llm_cost_usd']:.4f}",
                flush=True,
            )

    summary.update(state="complete", complete=True, current_run=None)
    _checkpoint(output_dir, summary)
    return summary


def _checkpoint(output_dir: Path, summary: dict[str, Any]) -> None:
    rows = summary["results"]
    methods: dict[str, Any] = {}
    for method_id in summary["methods"]:
        selected = [row for row in rows if row["method_id"] == method_id]
        successes = [row for row in selected if row["success"]]
        methods[method_id] = {
            "completed": len(selected),
            "successes": len(successes),
            "success_rate": len(successes) / len(selected) if selected else 0.0,
            "mean_generation_cycles": round(
                mean(row["generation_cycles"] for row in selected), 3
            ) if selected else None,
            "mean_target_action_steps": round(
                mean(row["target_action_steps"] for row in selected), 3
            ) if selected else None,
            "total_tokens": sum(row["total_tokens"] for row in selected),
            "total_llm_cost_usd": round(
                sum(row["llm_cost_usd"] for row in selected), 8
            ),
        }
    summary.update(
        completed_runs=len(rows),
        total_runs=len(summary["methods"]) * int(summary["repetitions"]),
        method_summary=methods,
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(output_dir / "summary.json", summary)
    _write_markdown(output_dir / "summary.md", summary)


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Method Comparison",
        "",
        f"- state: `{summary['state']}`",
        f"- task: `{summary['task_id']}`",
        f"- migration: `{summary['source_robot']} -> {summary['target_robot']}`",
        f"- fixed seed: `{summary['seed']}`",
        f"- completed: `{summary.get('completed_runs', 0)}/{summary.get('total_runs', 0)}`",
        "",
        "| method | success | rate | mean cycles | mean actions | tokens | cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method_id in summary["methods"]:
        row = summary.get("method_summary", {}).get(method_id, {})
        lines.append(
            f"| {method_id} | {row.get('successes', 0)}/{row.get('completed', 0)} | "
            f"{float(row.get('success_rate') or 0.0):.3f} | "
            f"{row.get('mean_generation_cycles')} | {row.get('mean_target_action_steps')} | "
            f"{row.get('total_tokens', 0)} | {float(row.get('total_llm_cost_usd') or 0.0):.6f} |"
        )
    lines.extend([
        "",
        "## Runs",
        "",
        "| method | repetition | success | official | cycles | actions | tokens | cost USD |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ])
    for row in summary.get("results", []):
        lines.append(
            f"| {row['method_id']} | {row['repetition']} | {row['success']} | "
            f"{row.get('official_success')} | {row['generation_cycles']} | "
            f"{row['target_action_steps']} | {row['total_tokens']} | "
            f"{row['llm_cost_usd']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _evidence_fields(worker_result: Mapping[str, Any]) -> dict[str, Any]:
    details_path = Path(str(worker_result.get("details") or ""))
    details = _read_json(details_path) if details_path.is_file() else {}
    target_result = details.get("target_result")
    target_result = target_result if isinstance(target_result, Mapping) else {}
    adapter_path = Path(str(worker_result.get("target_adapter") or ""))
    adapter_sha256 = (
        hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        if adapter_path.is_file()
        else None
    )
    diagnosis = target_result.get("failure_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
    return {
        "official_success": bool(target_result.get("official_success")),
        "adapter_sha256": adapter_sha256,
        "failure_reason": None if worker_result.get("success") else diagnosis.get("reason"),
    }


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
