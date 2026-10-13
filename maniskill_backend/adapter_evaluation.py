"""Evaluate one immutable generated adapter on a fixed set of simulation seeds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .dynamic_harness import DynamicMigrationSpec, run_dynamic_trial
from .experiment_environment import capture_experiment_environment


def evaluate_fixed_adapter(
    *,
    evidence_dir: Path,
    output_dir: Path,
    seeds: Sequence[int],
    enforce_environment: bool = True,
) -> dict[str, Any]:
    """Replay an unchanged adapter; never call the LLM or repair from test feedback."""

    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be a non-empty unique sequence.")
    evidence_dir = evidence_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    replay = _read_json(evidence_dir / "replay_result.json")
    generation = _read_json(evidence_dir / "generation_cycle.json")
    adapter_path = evidence_dir / "target_adapter.py"
    if replay.get("schema") != "generated_adapter_replay.v1":
        raise ValueError("Evidence needs replay_result.json with generated_adapter_replay.v1.")
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    adapter_sha256 = _sha256(adapter_path)
    expected_sha256 = generation.get("adapter_sha256")
    if expected_sha256 and adapter_sha256 != expected_sha256:
        raise ValueError("Target adapter hash differs from its generation record.")
    spec_payload = replay.get("spec")
    if not isinstance(spec_payload, Mapping):
        raise ValueError("Replay evidence is missing its migration spec.")
    base_spec = DynamicMigrationSpec(**dict(spec_payload))
    program = str(replay.get("program") or "").strip()
    if not program:
        raise ValueError("Replay evidence is missing the frozen high-level program.")

    runtime = capture_experiment_environment(
        {
            "source_robot": base_spec.source_robot,
            "target_robot": base_spec.target_robot,
            "seed": seeds[0],
            "evaluation_seeds": list(seeds),
            "adapter_sha256": adapter_sha256,
        }
    )
    _write_json(output_dir / "runtime_environment.json", runtime)
    summary: dict[str, Any] = {
        "schema": "fixed_adapter_multiseed.v1",
        "state": "running",
        "complete": False,
        "evidence_dir": str(evidence_dir),
        "adapter": str(adapter_path),
        "adapter_sha256": adapter_sha256,
        "source_generation_cycle": replay.get("source_generation_cycle"),
        "spec": dict(spec_payload),
        "seeds": list(seeds),
        "llm_calls": 0,
        "code_repairs": 0,
        "results": [],
    }
    _write_summary(output_dir, summary)
    if enforce_environment and not runtime.get("validation", {}).get("ok"):
        summary.update(
            state="environment_contract_mismatch",
            complete=False,
            environment_mismatches=runtime["validation"]["mismatches"],
        )
        _write_summary(output_dir, summary)
        return summary

    for seed in seeds:
        trial_spec = replace(
            base_spec,
            seed=seed,
            max_source_cycles=0,
            max_target_cycles=0,
        )
        trial = run_dynamic_trial(
            spec=trial_spec,
            robot_uid=trial_spec.target_robot,
            control_mode=trial_spec.target_control_mode,
            program=program,
            adapter_path=adapter_path,
        )
        trial.pop("implementation_context", None)
        trial_path = output_dir / f"seed_{seed:03d}.json"
        _write_json(trial_path, trial)
        diagnosis = trial.get("failure_diagnosis")
        diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
        succeeded = bool(trial.get("success"))
        row = {
            "seed": seed,
            "success": succeeded,
            "official_success": bool(trial.get("official_success")),
            "action_steps": int(trial.get("action_steps") or 0),
            "terminated": bool(trial.get("terminated")),
            "truncated": bool(trial.get("truncated")),
            "failure_layer": "success" if succeeded else diagnosis.get("layer"),
            "failure_reason": "success" if succeeded else diagnosis.get("reason"),
            "message": trial.get("message"),
            "trial": str(trial_path),
        }
        summary["results"].append(row)
        _update_aggregates(summary)
        _write_summary(output_dir, summary)

    summary.update(state="complete", complete=True)
    _update_aggregates(summary)
    _write_summary(output_dir, summary)
    return summary


def _update_aggregates(summary: dict[str, Any]) -> None:
    rows = summary["results"]
    successes = [row for row in rows if row["success"]]
    reasons = Counter(str(row["failure_reason"] or "unknown") for row in rows if not row["success"])
    summary.update(
        num_trials=len(rows),
        num_success=len(successes),
        num_failure=len(rows) - len(successes),
        success_rate=(len(successes) / len(rows)) if rows else 0.0,
        mean_success_action_steps=(
            round(mean(row["action_steps"] for row in successes), 3) if successes else None
        ),
        total_action_steps=sum(row["action_steps"] for row in rows),
        failure_reasons=dict(sorted(reasons.items())),
    )


def _write_summary(output_dir: Path, summary: Mapping[str, Any]) -> None:
    _write_json(output_dir / "summary.json", summary)
    lines = [
        "# Fixed Adapter Multi-Seed Evaluation",
        "",
        f"- state: `{summary.get('state')}`",
        f"- adapter SHA-256: `{summary.get('adapter_sha256')}`",
        f"- success: `{summary.get('num_success', 0)}/{summary.get('num_trials', 0)}`",
        f"- success rate: `{float(summary.get('success_rate') or 0.0):.3f}`",
        "- LLM calls: `0`",
        "- code repairs: `0`",
        "",
        "| seed | success | official | actions | failure reason | message |",
        "|---:|---|---|---:|---|---|",
    ]
    for row in summary.get("results") or []:
        message = str(row.get("message") or "").replace("|", "/")
        lines.append(
            f"| {row['seed']} | {row['success']} | {row['official_success']} | "
            f"{row['action_steps']} | {row.get('failure_reason') or ''} | {message} |"
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
