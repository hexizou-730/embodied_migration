"""One entrypoint for planning and running the frozen paper benchmark."""

from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import os
import subprocess
import sys
from math import sqrt
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, Mapping, Sequence

from maniskill_backend.cases import iter_full_migration_cases
from maniskill_backend.paper_benchmark import (
    PAPER_METHODS,
    PAPER_TIERS,
    build_paper_plan,
    normalize_method_id,
    write_paper_plan,
)
from maniskill_backend.paper_preflight import run_paper_preflight, write_paper_preflight
from maniskill_backend.paper_verify import verify_paper_plan, write_verification
from maniskill_backend.llm_preflight import run_llm_preflight, write_llm_preflight
from maniskill_backend.protocol_audit import run_protocol_audit, write_protocol_audit
from maniskill_backend.source_baseline import run_source_baseline_gate
from maniskill_backend.paper_package import (
    build_remote_package,
    build_evidence_export,
    verify_remote_package_manifest,
)
from maniskill_backend.paper_statistics import paired_method_comparisons
from maniskill_backend.diagnosis_eval import (
    build_annotation_template,
    diagnosis_evaluation_markdown,
    evaluate_annotations,
)
from maniskill_backend.motivating_examples import (
    build_motivating_examples,
    write_motivating_examples,
)


REPO_ROOT = Path(__file__).resolve().parent


def _selection(text: str, available: Iterable[str], *, normalize=None) -> list[str]:
    values = list(available)
    if str(text or "all").strip().lower() == "all":
        return values
    selected = [item.strip() for item in text.split(",") if item.strip()]
    if normalize:
        selected = [normalize(item) for item in selected]
    unknown = [item for item in selected if item not in values]
    if unknown:
        raise KeyError(f"Unknown selection {unknown!r}. Available: {', '.join(values)}")
    return list(dict.fromkeys(selected))


def _load_completed_summary(
    path: Path,
    run: Mapping[str, Any],
    plan: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "run_id": run.get("run_id"),
        "method_id": run.get("method_id"),
        "case_id": run.get("case_id"),
        "repetition": run.get("repetition"),
        "development_seeds": run.get("development_seeds"),
        "held_out_seeds": run.get("held_out_seeds"),
    }
    if payload.get("schema") != "paper_method_run.v1":
        return None
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    protocol = plan or {}
    runtime_config = protocol.get("runtime_config") or {}
    for key in ("sim_backend", "render_backend"):
        if runtime_config.get(key) is not None and payload.get(key) != runtime_config.get(key):
            return None
    method_id = str(run.get("method_id") or "")
    if PAPER_METHODS.get(method_id) and PAPER_METHODS[method_id].uses_llm:
        if _llm_protocol_snapshot(payload.get("runtime") or {}) != (protocol.get("llm_config") or {}):
            return None
    return payload


def _run_plan(
    plan: Mapping[str, Any],
    plan_dir: Path,
    *,
    resume: bool = True,
) -> Dict[str, Any]:
    command_log = plan_dir / "benchmark_commands.log"
    progress_path = plan_dir / "progress.json"
    completed_runs = []
    completed_run_ids: list[str] = []
    infrastructure_failures = []
    skipped_existing = 0
    attempted_current = 0
    ready_runs = [run for run in plan.get("runs") or [] if run.get("ready")]

    def publish_progress(state: str, current_run: Mapping[str, Any] | None = None) -> None:
        current = None
        if current_run is not None:
            current = {
                key: current_run.get(key)
                for key in ("run_id", "method_id", "case_id", "repetition")
            }
        _write_json_atomic(
            progress_path,
            {
                "schema": "paper_benchmark_progress.v1",
                "plan_name": plan.get("plan_name"),
                "tier": (plan.get("tier") or {}).get("tier_id"),
                "state": state,
                "total_ready_runs": len(ready_runs),
                "completed_runs": len(completed_runs),
                "pending_runs": max(0, len(ready_runs) - len(completed_runs)),
                "skipped_existing": skipped_existing,
                "attempted_current": attempted_current,
                "infrastructure_failures": len(infrastructure_failures),
                "current_run": current,
                "completed_run_ids": completed_run_ids,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "command_log": str(command_log),
            },
        )

    publish_progress("starting")
    for run in ready_runs:
        summary_path = plan_dir / "runs" / str(run.get("run_id")) / "summary.json"
        existing = _load_completed_summary(summary_path, run, plan) if resume else None
        if existing is not None:
            print(f"[paper] {run.get('run_id')} [resume: already complete]", flush=True)
            completed_runs.append(existing)
            completed_run_ids.append(str(run.get("run_id")))
            skipped_existing += 1
            publish_progress("running")
            continue
        command = list(run.get("command") or [])
        print(f"[paper] {run.get('run_id')}", flush=True)
        attempted_current += 1
        publish_progress("running", run)
        try:
            returncode, output_tail = _stream_benchmark_command(
                command,
                command_log=command_log,
                run_id=str(run.get("run_id")),
            )
        except OSError as exc:
            returncode = -1
            output_tail = repr(exc)
        if summary_path.exists():
            loaded = _load_completed_summary(summary_path, run, plan)
            if loaded is not None:
                completed_runs.append(loaded)
                completed_run_ids.append(str(run.get("run_id")))
                publish_progress("running")
                continue
            infrastructure_failures.append(
                {
                    "run_id": run.get("run_id"),
                    "returncode": int(returncode),
                    "output_tail": "summary.json exists but does not match the frozen run specification",
                }
            )
        else:
            infrastructure_failures.append(
                {
                    "run_id": run.get("run_id"),
                    "returncode": int(returncode),
                    "output_tail": output_tail,
                }
            )
        publish_progress("running")
    final_state = "completed" if not infrastructure_failures else "completed_with_infrastructure_failures"
    publish_progress(final_state)
    summary = _summarize(
        plan,
        completed_runs,
        infrastructure_failures,
        skipped_existing=skipped_existing,
        attempted_current=attempted_current,
    )
    summary["progress_file"] = str(progress_path)
    return summary


def _command_flag_value(command: Sequence[str], flag: str) -> str | None:
    try:
        index = list(command).index(flag)
    except ValueError:
        return None
    return str(command[index + 1]) if index + 1 < len(command) else ""


def _smoke_nested_preview_audit(
    plan: Mapping[str, Any],
    plan_dir: Path,
) -> Dict[str, Any]:
    """Verify the dry-run command chain for the two controlled CEGIS methods."""

    rows = []
    command_previews: Dict[tuple[str, int, str], Dict[str, list[str]]] = {}
    for run in plan.get("runs") or []:
        method_id = str(run.get("method_id") or "")
        if not run.get("ready") or method_id not in {"B5", "Ours"}:
            continue
        errors = []
        summary_path = plan_dir / "runs" / str(run.get("run_id")) / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary = {}
            errors.append(f"missing or invalid summary: {exc!r}")

        cegis = ((summary.get("details") or {}).get("cegis_summary") or {})
        preview = cegis.get("nested_command_preview") or {}
        cycles = list(cegis.get("cycles") or [])
        initial_generation = (
            ((cycles[0].get("generation") or {}).get("command") or [])
            if cycles
            else []
        )
        development = (
            ((cycles[0].get("development_evaluation") or {}).get("command") or [])
            if cycles
            else []
        )
        probe = ((preview.get("structured_probe") or {}).get("command") or [])
        repair = ((preview.get("repair_generation") or {}).get("command") or [])
        held_out = ((preview.get("held_out_evaluation") or {}).get("command") or [])
        outer = (((summary.get("details") or {}).get("cegis_run") or {}).get("command") or [])
        expected_strategy = PAPER_METHODS[method_id].probe_strategy

        required_commands = {
            "outer": outer,
            "initial_generation": initial_generation,
            "development_evaluation": development,
            "structured_probe": probe,
            "repair_generation": repair,
            "held_out_evaluation": held_out,
        }
        for name, command in required_commands.items():
            if not command:
                errors.append(f"missing {name} command")
        if cegis.get("status") != "dry_run_planned":
            errors.append("CEGIS summary is not a dry-run plan")
        if _command_flag_value(initial_generation, "--prompt-policy") != "full":
            errors.append("initial generation prompt policy is not full")
        if "--no-implicit-probe-feedback" not in initial_generation:
            errors.append("initial generation permits implicit historical probe feedback")
        if _command_flag_value(development, "--seeds") != run.get("development_seeds"):
            errors.append("development split drifted from the frozen plan")
        if _command_flag_value(probe, "--probe-selection") != expected_strategy:
            errors.append(f"probe selector is not {expected_strategy}")
        if "--dry-run" not in probe:
            errors.append("probe preview is not marked dry-run")
        if _command_flag_value(repair, "--prompt-policy") != "full":
            errors.append("repair generation prompt policy is not full")
        for flag in ("--counterexample-json", "--probe-feedback-json", "--dry-run"):
            if flag not in repair:
                errors.append(f"repair generation missing {flag}")
        if _command_flag_value(held_out, "--seeds") != run.get("held_out_seeds"):
            errors.append("held-out split drifted from the frozen plan")
        if (preview.get("counterexample_selection") or {}).get("held_out_used") is not False:
            errors.append("counterexample preview does not explicitly exclude held-out data")
        if method_id == "B5":
            if "--fixed-grid-total-budget" not in probe or "--fixed-grid-offset" not in probe:
                errors.append("B5 fixed-grid schedule flags are missing")
        elif "--fixed-grid-total-budget" in probe or "--fixed-grid-offset" in probe:
            errors.append("Ours unexpectedly contains fixed-grid schedule flags")

        rows.append(
            {
                "run_id": run.get("run_id"),
                "method_id": method_id,
                "case_id": run.get("case_id"),
                "expected_probe_strategy": expected_strategy,
                "valid": not errors,
                "errors": errors,
            }
        )

        command_previews[
            (str(run.get("case_id") or ""), int(run.get("repetition") or 0), method_id)
        ] = {
            "outer": list(outer),
            "initial_generation": list(initial_generation),
            "development": list(development),
            "probe": list(probe),
            "repair_generation": list(repair),
            "held_out": list(held_out),
        }

    pair_rows = []
    pair_keys = sorted({
        (case_id, repetition)
        for case_id, repetition, method_id in command_previews
        if method_id == "B5" and (case_id, repetition, "Ours") in command_previews
    })
    shared_flags = {
        "outer": (
            "--case", "--development-seeds", "--held-out-seeds", "--max-cycles",
            "--attempts-per-cycle", "--probe-budget", "--prompt-policy",
            "--success-threshold", "--min-trials-for-accept", "--obs-mode",
            "--sim-backend", "--render-backend", "--max-episode-steps",
        ),
        "initial_generation": (
            "--case", "--max-attempts", "--seed", "--obs-mode", "--sim-backend",
            "--render-backend", "--trial-timeout-s", "--test-timeout-s", "--prompt-policy",
        ),
        "development": (
            "--seeds", "--task", "--robot", "--adapter-module", "--control-mode",
            "--obs-mode", "--sim-backend", "--render-backend", "--max-episode-steps",
            "--success-threshold", "--min-trials-for-accept",
        ),
        "probe": (
            "--seed", "--suggestion-budget", "--max-cases", "--obs-mode",
            "--sim-backend", "--render-backend", "--max-episode-steps",
        ),
        "repair_generation": (
            "--case", "--max-attempts", "--seed", "--obs-mode", "--sim-backend",
            "--render-backend", "--trial-timeout-s", "--test-timeout-s", "--prompt-policy",
        ),
        "held_out": (
            "--seeds", "--task", "--robot", "--adapter-module", "--control-mode",
            "--obs-mode", "--sim-backend", "--render-backend", "--max-episode-steps",
            "--success-threshold", "--min-trials-for-accept",
        ),
    }
    for case_id, repetition in pair_keys:
        b5 = command_previews.get((case_id, repetition, "B5"))
        ours = command_previews.get((case_id, repetition, "Ours"))
        errors = []
        if b5 is None or ours is None:
            errors.append("B5/Ours controlled pair is incomplete")
        else:
            for command_name, flags in shared_flags.items():
                for flag in flags:
                    left = _command_flag_value(b5[command_name], flag)
                    right = _command_flag_value(ours[command_name], flag)
                    if left is None or right is None:
                        errors.append(f"{command_name} missing required shared flag {flag}")
                    elif left != right:
                        errors.append(
                            f"{command_name} differs at {flag}: B5={left!r}, Ours={right!r}"
                        )
            if _command_flag_value(b5["outer"], "--probe-strategy") != "fixed_grid":
                errors.append("B5 outer loop does not select fixed_grid")
            if _command_flag_value(ours["outer"], "--probe-strategy") != "active":
                errors.append("Ours outer loop does not select active")
        pair_rows.append(
            {
                "case_id": case_id,
                "repetition": repetition,
                "valid": not errors,
                "allowed_difference": "probe selector and fixed-grid scheduling flags only",
                "errors": errors,
            }
        )

    return {
        "schema": "paper_smoke_nested_preview_audit.v1",
        "valid": all(row["valid"] for row in rows) and all(row["valid"] for row in pair_rows),
        "num_expected": len(rows),
        "num_valid": sum(bool(row["valid"]) for row in rows),
        "num_controlled_pairs_expected": len(pair_rows),
        "num_controlled_pairs_valid": sum(bool(row["valid"]) for row in pair_rows),
        "rows": rows,
        "controlled_pairs": pair_rows,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stream_benchmark_command(
    command: Sequence[str],
    *,
    command_log: Path,
    run_id: str,
) -> tuple[int, str]:
    """Stream one run to the terminal and durable log while retaining a short tail."""

    command_log.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=200)
    with command_log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## {run_id}\n$ {' '.join(command)}\n")
        stream.flush()
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
                tail.append(line)
        returncode = int(process.wait())
        stream.write(f"[paper] returncode={returncode}\n")
        stream.flush()
    return returncode, "".join(tail)[-2000:]


def _summarize(
    plan: Mapping[str, Any],
    completed_runs: list[Mapping[str, Any]],
    infrastructure_failures: list[Mapping[str, Any]],
    *,
    skipped_existing: int = 0,
    attempted_current: int = 0,
) -> Dict[str, Any]:
    case_support = {
        str(item.get("case_id")): str(item.get("support_status") or "unknown")
        for item in plan.get("case_definitions") or []
    }
    groups: Dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in completed_runs:
        key = (str(run.get("method_id")), str(run.get("case_id")))
        groups.setdefault(key, []).append(run)
    rows = []
    for (method_id, case_id), runs in sorted(groups.items()):
        held_out = [
            float(run["held_out_success_rate"])
            for run in runs
            if run.get("held_out_success_rate") is not None
        ]
        development = [
            float(run["development_success_rate"])
            for run in runs
            if run.get("development_success_rate") is not None
        ]
        metrics = [run.get("metrics") or {} for run in runs]
        held_out_trial_rows = [
            trial
            for run in runs
            for trial in (((run.get("details") or {}).get("held_out") or {}).get("rows") or [])
        ]
        held_out_successes = sum(bool(item.get("success")) for item in held_out_trial_rows)
        held_out_trials = len(held_out_trial_rows)
        pooled_rate = held_out_successes / held_out_trials if held_out_trials else None
        ci_low, ci_high = _wilson_interval(held_out_successes, held_out_trials)
        unique_adapter_hashes = {
            str(run.get("adapter_sha256"))
            for run in runs
            if str(run.get("adapter_sha256") or "")
        }
        rows.append(
            {
                "method_id": method_id,
                "case_id": case_id,
                "support_status": str(runs[0].get("support_status") or case_support.get(case_id, "unknown")),
                "num_repetitions": len(runs),
                "num_accepted": sum(run.get("success") is True for run in runs),
                "mean_development_success_rate": round(mean(development), 4) if development else None,
                "mean_held_out_success_rate": round(mean(held_out), 4) if held_out else None,
                "held_out_success_rate_std": round(stdev(held_out), 4) if len(held_out) > 1 else 0.0 if held_out else None,
                "held_out_total_trials": held_out_trials,
                "held_out_total_successes": held_out_successes,
                "held_out_pooled_success_rate": round(pooled_rate, 4) if pooled_rate is not None else None,
                "held_out_wilson_95_low": ci_low,
                "held_out_wilson_95_high": ci_high,
                "mean_simulator_calls": round(
                    mean(float(item.get("simulator_calls") or 0) for item in metrics), 2
                ),
                "mean_environment_steps": round(
                    mean(float(item.get("environment_steps") or 0) for item in metrics), 2
                ),
                "mean_wall_time_seconds": round(
                    mean(float(item.get("wall_time_seconds") or 0) for item in metrics), 2
                ),
                "mean_llm_api_calls": round(
                    mean(float(item.get("llm_api_calls") or 0) for item in metrics), 2
                ),
                "mean_total_tokens": round(
                    mean(float(item.get("total_tokens") or 0) for item in metrics), 2
                ),
                "mean_probe_cases": round(
                    mean(float(item.get("probe_cases") or 0) for item in metrics), 2
                ),
                "mean_cegis_cycles": round(
                    mean(float(item.get("cegis_cycles") or 0) for item in metrics), 2
                ),
                "mean_unique_counterexample_reasons": round(
                    mean(float(item.get("unique_counterexample_reasons") or 0) for item in metrics), 2
                ),
                "mean_kernel_ucb_batches": round(
                    mean(float(item.get("kernel_ucb_batches") or 0) for item in metrics), 2
                ),
                "mean_adapter_branch_count": round(
                    mean(float(item.get("adapter_branch_count") or 0) for item in metrics), 2
                ),
                "adapter_provenance": sorted({str(run.get("adapter_provenance")) for run in runs}),
                "num_unique_adapter_sha256": len(unique_adapter_hashes),
                "duplicate_generation_warning": bool(
                    PAPER_METHODS.get(method_id)
                    and PAPER_METHODS[method_id].uses_llm
                    and len(runs) > 1
                    and unique_adapter_hashes
                    and len(unique_adapter_hashes) < len(runs)
                ),
            }
        )
    support_subgroups = _support_subgroup_summary(completed_runs, case_support)
    paired_comparisons = paired_method_comparisons(completed_runs)
    return {
        "schema": "embodied_migration_paper_summary.v1",
        "plan_name": plan.get("plan_name"),
        "tier": plan.get("tier"),
        "num_planned_runs": plan.get("num_runs"),
        "num_ready_runs": plan.get("num_ready_runs"),
        "num_blocked_runs": plan.get("num_blocked_runs"),
        "num_completed_runs": len(completed_runs),
        "num_pending_ready_runs": max(0, int(plan.get("num_ready_runs") or 0) - len(completed_runs)),
        "num_skipped_existing": skipped_existing,
        "num_attempted_current": attempted_current,
        "num_infrastructure_failures": len(infrastructure_failures),
        "rows": rows,
        "support_subgroups": support_subgroups,
        "paired_method_comparisons": paired_comparisons,
        "infrastructure_failures": infrastructure_failures,
    }


def _support_subgroup_summary(
    completed_runs: Sequence[Mapping[str, Any]],
    case_support: Mapping[str, str],
) -> list[Dict[str, Any]]:
    """Pool held-out trials by method and declared task-support status."""

    groups: Dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for run in completed_runs:
        case_id = str(run.get("case_id") or "")
        support_status = str(run.get("support_status") or case_support.get(case_id, "unknown"))
        groups.setdefault((str(run.get("method_id")), support_status), []).append(run)
    rows: list[Dict[str, Any]] = []
    for (method_id, support_status), runs in sorted(groups.items()):
        trials = [
            trial
            for run in runs
            for trial in (((run.get("details") or {}).get("held_out") or {}).get("rows") or [])
        ]
        successes = sum(bool(item.get("success")) for item in trials)
        total = len(trials)
        low, high = _wilson_interval(successes, total)
        rows.append(
            {
                "method_id": method_id,
                "support_status": support_status,
                "num_runs": len(runs),
                "num_cases": len({str(run.get("case_id")) for run in runs}),
                "held_out_total_trials": total,
                "held_out_total_successes": successes,
                "held_out_pooled_success_rate": round(successes / total, 4) if total else None,
                "held_out_wilson_95_low": low,
                "held_out_wilson_95_high": high,
            }
        )
    return rows


def _wilson_interval(successes: int, trials: int, *, z: float = 1.96) -> tuple[float | None, float | None]:
    """Return a Wilson score interval for a binomial success rate."""

    if trials <= 0:
        return None, None
    rate = successes / trials
    denominator = 1.0 + (z * z / trials)
    center = (rate + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * sqrt((rate * (1.0 - rate) / trials) + (z * z / (4.0 * trials * trials)))
        / denominator
    )
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Paper Benchmark Summary",
        "",
        f"- completed executable runs: `{summary.get('num_completed_runs')}/{summary.get('num_ready_runs')}`",
        f"- blocked by protocol: `{summary.get('num_blocked_runs')}`",
        f"- pending executable runs: `{summary.get('num_pending_ready_runs')}`",
        f"- reused from previous invocation: `{summary.get('num_skipped_existing')}`",
        f"- infrastructure failures: `{summary.get('num_infrastructure_failures')}`",
        "",
        "| Method | Case | Support | Repetitions | Unique adapters | Duplicate warning | Accepted | Dev success | Held-out mean +/- std | Pooled 95% CI | Trials | Sim calls | Time (s) | Probe cases | CEGIS cycles | LLM calls | Tokens | Branches | Provenance |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("rows") or []:
        lines.append(
            f"| {row.get('method_id')} | `{row.get('case_id')}` | `{row.get('support_status')}` | "
            f"{row.get('num_repetitions')} | "
            f"{row.get('num_unique_adapter_sha256')} | {row.get('duplicate_generation_warning')} | "
            f"{row.get('num_accepted')} | {row.get('mean_development_success_rate')} | "
            f"{row.get('mean_held_out_success_rate')} +/- {row.get('held_out_success_rate_std')} | "
            f"[{row.get('held_out_wilson_95_low')}, {row.get('held_out_wilson_95_high')}] | "
            f"{row.get('held_out_total_trials')} | {row.get('mean_simulator_calls')} | "
            f"{row.get('mean_wall_time_seconds')} | "
            f"{row.get('mean_probe_cases')} | {row.get('mean_cegis_cycles')} | "
            f"{row.get('mean_llm_api_calls')} | {row.get('mean_total_tokens')} | "
            f"{row.get('mean_adapter_branch_count')} | {', '.join(row.get('adapter_provenance') or [])} |"
        )
    lines.extend(
        [
            "",
            "## Support Subgroups",
            "",
            "| Method | Support | Cases | Trials | Successes | Pooled success | 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("support_subgroups") or []:
        lines.append(
            f"| {row.get('method_id')} | `{row.get('support_status')}` | {row.get('num_cases')} | "
            f"{row.get('held_out_total_trials')} | {row.get('held_out_total_successes')} | "
            f"{row.get('held_out_pooled_success_rate')} | "
            f"[{row.get('held_out_wilson_95_low')}, {row.get('held_out_wilson_95_high')}] |"
        )
    lines.extend(
        [
            "",
            "## Paired Held-Out Comparisons",
            "",
            "Outcomes are paired by case, repetition, and seed. P-values use the exact McNemar test; "
            "Holm adjustment covers the planned case-level comparisons. A hierarchical bootstrap "
            "resamples case, generation repetition, and seed for the effect-size interval. Pooled rows are descriptive.",
            "",
            "| Reference | Baseline | Case | Support | Pairs | Reference only | Baseline only | Difference | Bootstrap 95% CI | Exact p | Holm p |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("paired_method_comparisons") or []:
        lines.append(
            f"| {row.get('reference_method')} | {row.get('comparator_method')} | "
            f"`{row.get('case_id')}` | `{row.get('support_status')}` | {row.get('num_paired_trials')} | "
            f"{row.get('reference_only_success')} | {row.get('comparator_only_success')} | "
            f"{row.get('paired_success_rate_difference')} | "
            f"[{row.get('hierarchical_bootstrap_95_low')}, {row.get('hierarchical_bootstrap_95_high')}] | "
            f"{row.get('mcnemar_exact_p')} | "
            f"{row.get('holm_adjusted_p')} |"
        )
    source = summary.get("source_baseline_gate") or {}
    lines.extend(
        [
            "",
            "## Source Baseline Gate",
            "",
            f"- ready: `{source.get('ready')}`",
            f"- reused: `{source.get('reused')}`",
            "",
            "| Task | Source | Split | Seeds | Success | Trials | Accepted |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in source.get("rows") or []:
        lines.append(
            f"| `{row.get('task_id')}` | `{row.get('source_robot')}` | {row.get('split')} | "
            f"`{row.get('seeds')}` | {row.get('success_rate')} | {row.get('num_trials')} | "
            f"{row.get('accepted')} |"
        )
    lines.append("")
    return "\n".join(lines)


SUMMARY_COLUMNS: Sequence[str] = (
    "method_id",
    "case_id",
    "support_status",
    "num_repetitions",
    "num_accepted",
    "mean_development_success_rate",
    "mean_held_out_success_rate",
    "held_out_success_rate_std",
    "held_out_total_trials",
    "held_out_total_successes",
    "held_out_pooled_success_rate",
    "held_out_wilson_95_low",
    "held_out_wilson_95_high",
    "mean_simulator_calls",
    "mean_environment_steps",
    "mean_wall_time_seconds",
    "mean_llm_api_calls",
    "mean_total_tokens",
    "mean_probe_cases",
    "mean_cegis_cycles",
    "mean_unique_counterexample_reasons",
    "mean_kernel_ucb_batches",
    "mean_adapter_branch_count",
    "adapter_provenance",
    "num_unique_adapter_sha256",
    "duplicate_generation_warning",
)


SUPPORT_SUMMARY_COLUMNS: Sequence[str] = (
    "method_id",
    "support_status",
    "num_runs",
    "num_cases",
    "held_out_total_trials",
    "held_out_total_successes",
    "held_out_pooled_success_rate",
    "held_out_wilson_95_low",
    "held_out_wilson_95_high",
)


PAIRED_COMPARISON_COLUMNS: Sequence[str] = (
    "case_id",
    "reference_method",
    "comparator_method",
    "support_status",
    "num_paired_trials",
    "both_success",
    "reference_only_success",
    "comparator_only_success",
    "both_failure",
    "reference_success_rate",
    "comparator_success_rate",
    "paired_success_rate_difference",
    "hierarchical_bootstrap_95_low",
    "hierarchical_bootstrap_95_high",
    "hierarchical_bootstrap_iterations",
    "mcnemar_exact_p",
    "holm_adjusted_p",
    "holm_family",
)


def _write_support_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUPPORT_SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUPPORT_SUMMARY_COLUMNS})


def _write_paired_comparisons_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PAIRED_COMPARISON_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in PAIRED_COMPARISON_COLUMNS})


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key) for key in SUMMARY_COLUMNS}
            normalized["adapter_provenance"] = ";".join(row.get("adapter_provenance") or [])
            writer.writerow(normalized)


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    for source, target in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(source, target)
    return text


def _write_summary_latex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Method & Case & Reps & Accepted & Dev SR & Test SR & 95\% CI \\",
        r"\hline",
    ]
    for row in rows:
        values = (
            row.get("method_id"),
            row.get("case_id"),
            row.get("num_repetitions"),
            row.get("num_accepted"),
            row.get("mean_development_success_rate"),
            row.get("mean_held_out_success_rate"),
            f"[{row.get('held_out_wilson_95_low')}, {row.get('held_out_wilson_95_high')}]",
        )
        lines.append(" & ".join(_latex_escape(value) for value in values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _list_protocol() -> None:
    print("Methods:")
    for method in PAPER_METHODS.values():
        print(f"  {method.method_id:6} {method.label}")
    print("\nCases:")
    for case in iter_full_migration_cases():
        print(
            f"  {case.case_id}: {case.source_robot} -> {case.target_robot} "
            f"({case.task_id}, {case.support_status}, benchmark_enabled={case.benchmark_enabled})"
        )


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _process_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
    except (TypeError, ValueError, OSError):
        return False
    return True


def _start_frozen_remote_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Launch a verified packaged plan in a detached process."""

    plan_path = _repo_path(args.plan_file)
    manifest_path = _repo_path(args.package_manifest)
    package_verification = verify_remote_package_manifest(manifest_path)
    raw_plan = _read_json_file(plan_path)
    if raw_plan:
        plan, plan_errors = _validate_and_rebind_frozen_plan(raw_plan)
    else:
        plan, plan_errors = {}, ["cannot read frozen plan"]
    payload: Dict[str, Any] = {
        "schema": "frozen_remote_plan_launcher.v1",
        "started": False,
        "package_verification": package_verification,
        "plan_valid": not plan_errors,
        "plan_errors": plan_errors,
    }
    if not package_verification.get("valid") or plan_errors:
        payload["valid"] = False
        payload["blocked_reason"] = "package or frozen-plan integrity check failed"
        return payload

    method_ids = [str(item.get("method_id")) for item in plan.get("method_definitions") or []]
    uses_llm = any(PAPER_METHODS[item].uses_llm for item in method_ids)
    frozen_environment = _apply_frozen_llm_environment(
        plan.get("llm_config") or {},
        required=uses_llm,
    )
    llm_preflight = run_llm_preflight(
        method_ids,
        require_stochastic_repetitions=int((plan.get("tier") or {}).get("repetitions") or 0) > 1,
    )
    expected_llm_config = plan.get("llm_config") or {}
    observed_llm_config = _llm_protocol_snapshot(llm_preflight)
    llm_config_matches = not uses_llm or observed_llm_config == expected_llm_config
    llm_preflight = {
        **llm_preflight,
        "frozen_config": expected_llm_config,
        "observed_config": observed_llm_config,
        "matches_frozen_plan": llm_config_matches,
        "ready": bool(llm_preflight.get("ready")) and llm_config_matches,
        "errors": list(llm_preflight.get("errors") or [])
        + ([] if llm_config_matches else ["runtime LLM configuration does not match frozen plan"]),
    }
    payload["frozen_llm_environment"] = frozen_environment
    payload["llm_preflight"] = llm_preflight
    if not llm_preflight.get("ready"):
        payload["valid"] = False
        payload["blocked_reason"] = "LLM configuration or API key preflight failed"
        return payload

    output_root = str(plan.get("output_root") or "results/paper")
    plan_name = str(plan.get("plan_name") or "")
    plan_dir = REPO_ROOT / output_root / plan_name
    plan_dir.mkdir(parents=True, exist_ok=True)
    launcher_path = plan_dir / "launcher.json"
    existing = _read_json_file(launcher_path)
    if _process_alive(existing.get("pid")):
        return {
            **payload,
            "valid": True,
            "already_running": True,
            "pid": existing.get("pid"),
            "execute_log": existing.get("execute_log"),
            "launcher": str(launcher_path),
        }

    execute_log = plan_dir / "execute.log"
    command = [
        sys.executable,
        str(REPO_ROOT / "paper.py"),
        "execute",
        "--plan-file",
        str(plan_path),
        "--package-manifest",
        str(manifest_path),
    ]
    if args.rerun_completed:
        command.append("--rerun-completed")
    with execute_log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[paper-start] {datetime.now(timezone.utc).isoformat()}\n")
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    launcher = {
        "schema": "frozen_remote_plan_launcher_state.v1",
        "plan_name": plan_name,
        "pid": int(process.pid),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "execute_log": str(execute_log),
        "command": command,
    }
    _write_json_atomic(launcher_path, launcher)
    return {
        **payload,
        "valid": True,
        "started": True,
        "already_running": False,
        "plan_name": plan_name,
        "pid": int(process.pid),
        "execute_log": str(execute_log),
        "launcher": str(launcher_path),
        "next_action": "python paper.py status",
    }


def _paper_status(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a compact status view for a local or packaged benchmark plan."""

    frozen_plan_path = _repo_path(args.plan_file)
    frozen_plan = _read_json_file(frozen_plan_path)
    plan_name = str(args.plan_name or frozen_plan.get("plan_name") or "")
    output_root = str(frozen_plan.get("output_root") or args.output_root)
    plan_dir = REPO_ROOT / output_root / plan_name if plan_name else Path()
    plan = _read_json_file(plan_dir / "plan.json") if plan_name else {}
    if not plan and frozen_plan.get("plan_name") == plan_name:
        plan = frozen_plan
    progress = _read_json_file(plan_dir / "progress.json") if plan_name else {}
    summary = _read_json_file(plan_dir / "summary.json") if plan_name else {}
    verification = _read_json_file(plan_dir / "verification.json") if plan_name else {}
    launcher = _read_json_file(plan_dir / "launcher.json") if plan_name else {}
    launcher_alive = _process_alive(launcher.get("pid")) if launcher else False
    export_manifest_path = (
        REPO_ROOT
        / output_root
        / "exports"
        / f"embodied_migration_evidence_{plan_name}.manifest.json"
        if plan_name
        else Path()
    )
    export_manifest = _read_json_file(export_manifest_path) if plan_name else {}

    if not plan_name:
        state = "unknown_plan"
        next_action = "pass --plan-name or run inside an unpacked remote package"
    elif verification.get("valid") and verification.get("complete"):
        state = "verified_complete"
        next_action = "return the evidence archive"
    elif progress.get("state") == "running":
        if launcher and not launcher_alive:
            state = "interrupted"
            next_action = "rerun python paper.py start"
        else:
            state = "running"
            next_action = "wait or rerun paper.py status"
    elif progress.get("state") == "completed_with_infrastructure_failures":
        state = "needs_resume"
        next_action = "rerun python paper.py execute"
    elif summary:
        state = "finished_unverified"
        next_action = "rerun python paper.py start"
    elif launcher and launcher_alive:
        state = "launching"
        next_action = "wait or rerun paper.py status"
    elif launcher:
        state = "launcher_exited"
        next_action = "inspect execute_log, then rerun python paper.py start"
    else:
        state = str(progress.get("state") or "not_started")
        next_action = "run python paper.py execute"

    total_ready_runs = progress.get("total_ready_runs", plan.get("num_ready_runs"))
    completed_runs = progress.get("completed_runs", summary.get("num_completed_runs", 0))
    pending_runs = progress.get("pending_runs")
    if pending_runs is None:
        pending_runs = summary.get("num_pending_ready_runs")
    if pending_runs is None and total_ready_runs is not None:
        pending_runs = max(0, int(total_ready_runs) - int(completed_runs or 0))
    command_log = progress.get("command_log")
    if not command_log and plan_name:
        command_log = str(plan_dir / "benchmark_commands.log")

    return {
        "schema": "paper_benchmark_status.v1",
        "plan_name": plan_name or None,
        "plan_dir": str(plan_dir) if plan_name else None,
        "state": state,
        "total_ready_runs": total_ready_runs,
        "completed_runs": completed_runs,
        "pending_runs": pending_runs,
        "current_run": progress.get("current_run"),
        "infrastructure_failures": progress.get(
            "infrastructure_failures", summary.get("num_infrastructure_failures", 0)
        ),
        "updated_utc": progress.get("updated_utc"),
        "command_log": command_log,
        "launcher_pid": launcher.get("pid"),
        "launcher_alive": launcher_alive if launcher else None,
        "execute_log": launcher.get("execute_log"),
        "verification_valid": verification.get("valid"),
        "verification_complete": verification.get("complete"),
        "evidence_archive": export_manifest.get("archive"),
        "next_action": next_action,
    }


def _command_option(command: Sequence[str], option: str) -> str:
    values = list(command)
    try:
        return values[values.index(option) + 1]
    except (ValueError, IndexError):
        return ""


def _llm_protocol_snapshot(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep reproducibility settings while excluding key names and values."""

    return {
        "provider": preflight.get("provider", preflight.get("llm_provider")),
        "model": preflight.get("model", preflight.get("llm_model")),
        "max_tokens": preflight.get("max_tokens", preflight.get("llm_max_tokens")),
        "temperature": preflight.get("temperature", preflight.get("llm_temperature")),
        "deepseek_thinking": preflight.get("deepseek_thinking"),
    }


def _apply_frozen_llm_environment(
    llm_config: Mapping[str, Any],
    *,
    required: bool,
) -> Dict[str, Any]:
    """Restore checksummed non-secret LLM settings for this execute process."""

    if not required:
        return {
            "required": False,
            "applied": False,
            "variables": [],
            "overridden_variables": [],
        }
    values = {
        "EM_LLM_PROVIDER": llm_config.get("provider"),
        "EM_MODEL": llm_config.get("model"),
        "EM_MAX_TOKENS": llm_config.get("max_tokens"),
        "EM_TEMPERATURE": llm_config.get("temperature"),
    }
    if llm_config.get("provider") == "deepseek":
        values["EM_DEEPSEEK_THINKING"] = llm_config.get("deepseek_thinking")
    missing = [name for name, value in values.items() if value is None or str(value) == ""]
    if missing:
        raise ValueError(f"frozen LLM configuration is missing: {', '.join(missing)}")

    overridden = [
        name
        for name, value in values.items()
        if name in os.environ and os.environ[name] != str(value)
    ]
    for name, value in values.items():
        os.environ[name] = str(value)
    return {
        "required": True,
        "applied": True,
        "variables": sorted(values),
        "overridden_variables": sorted(overridden),
        "source": "checksummed_frozen_plan",
    }


def _validate_and_rebind_frozen_plan(plan: Mapping[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Validate a packaged plan against local code and bind it to this Python."""

    errors = []
    if plan.get("schema") != "embodied_migration_paper_plan.v1":
        errors.append("invalid frozen plan schema")
        return dict(plan), errors
    tier_id = str((plan.get("tier") or {}).get("tier_id") or "")
    plan_name = str(plan.get("plan_name") or "")
    method_ids = [str(item.get("method_id")) for item in plan.get("method_definitions") or []]
    case_ids = [str(item.get("case_id")) for item in plan.get("case_definitions") or []]
    runs = list(plan.get("runs") or [])
    output_root = str(plan.get("output_root") or "")
    if not output_root and runs:
        output_root = _command_option(runs[0].get("command") or [], "--output-root")
    if tier_id not in PAPER_TIERS:
        errors.append(f"unknown frozen tier: {tier_id!r}")
    if not plan_name or Path(plan_name).name != plan_name:
        errors.append("frozen plan_name must be one safe path component")
    output_path = Path(output_root)
    if not output_root or output_path.is_absolute() or ".." in output_path.parts:
        errors.append("frozen output_root must be a safe relative path")
    llm_config = plan.get("llm_config") or {}
    if any(PAPER_METHODS.get(item) and PAPER_METHODS[item].uses_llm for item in method_ids):
        required_llm_fields = {
            "provider",
            "model",
            "max_tokens",
            "temperature",
            "deepseek_thinking",
        }
        if set(llm_config) != required_llm_fields or not llm_config.get("model"):
            errors.append("frozen LLM configuration is incomplete")
    if errors:
        return dict(plan), errors

    expected = build_paper_plan(
        tier_id=tier_id,
        method_ids=method_ids,
        case_ids=case_ids,
        output_root=output_root,
        plan_name=plan_name,
        sim_backend=str((plan.get("runtime_config") or {}).get("sim_backend") or "auto"),
        render_backend=str((plan.get("runtime_config") or {}).get("render_backend") or "gpu"),
        llm_config=plan.get("llm_config") or {},
    )

    def run_signature(item: Mapping[str, Any]) -> Dict[str, Any]:
        command = list(item.get("command") or [])
        if command:
            command[0] = "<python>"
        return {
            key: item.get(key)
            for key in (
                "run_id",
                "tier_id",
                "method_id",
                "case_id",
                "repetition",
                "development_seeds",
                "held_out_seeds",
                "ready",
                "blocked_reason",
            )
        } | {"command": command}

    observed_runs = [run_signature(item) for item in runs]
    expected_runs = [run_signature(item) for item in expected.get("runs") or []]
    if plan.get("tier") != expected.get("tier"):
        errors.append("frozen tier parameters do not match local benchmark definition")
    if plan.get("llm_config") != expected.get("llm_config"):
        errors.append("frozen LLM configuration is malformed")
    if observed_runs != expected_runs:
        errors.append("frozen run matrix or command arguments do not match local benchmark code")

    rebound = json.loads(json.dumps(plan))
    rebound["output_root"] = output_root
    for run in rebound.get("runs") or []:
        command = list(run.get("command") or [])
        if command:
            command[0] = sys.executable
        run["command"] = command
        run["command_text"] = " ".join(command)
    return rebound, errors


def _write_benchmark_summary(summary: Mapping[str, Any], plan_dir: Path) -> Dict[str, str]:
    paths = {
        "json": plan_dir / "summary.json",
        "markdown": plan_dir / "summary.md",
        "csv": plan_dir / "summary.csv",
        "support_subgroups_csv": plan_dir / "summary_support_subgroups.csv",
        "paired_method_comparisons_csv": plan_dir / "paired_method_comparisons.csv",
        "latex": plan_dir / "summary.tex",
    }
    paths["json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(_summary_markdown(summary), encoding="utf-8")
    _write_summary_csv(paths["csv"], summary.get("rows") or [])
    _write_support_summary_csv(paths["support_subgroups_csv"], summary.get("support_subgroups") or [])
    _write_paired_comparisons_csv(
        paths["paired_method_comparisons_csv"],
        summary.get("paired_method_comparisons") or [],
    )
    _write_summary_latex(paths["latex"], summary.get("rows") or [])
    return {key: str(value) for key, value in paths.items()}


def _build_motivation_evidence_plan(
    *,
    python_executable: str,
    pull_results: str | Path,
    pick_probe: str | Path,
    seeds: str,
    max_episode_steps: int,
    probe_max_episode_steps: int,
    probe_cases: int,
    sim_backend: str,
    render_backend: str,
) -> Dict[str, Any]:
    """Build the two real-simulation commands required by the motivating examples."""

    pull_path = _repo_path(pull_results)
    pick_path = _repo_path(pick_probe)
    expected_case = "case03_pick_cube_panda_to_xarm6"
    expected_probe_name = "pick_cube_xarm6_close_envelope.json"
    if pull_path.suffix != ".jsonl":
        raise ValueError("PullCube motivating evidence must be written as JSONL.")
    if pick_path.name != expected_probe_name or pick_path.parent.name != expected_case:
        raise ValueError(
            "PickCube motivating evidence path must end with "
            f"{expected_case}/{expected_probe_name}."
        )
    probe_output_root = pick_path.parent.parent
    commands = [
        {
            "id": "pullcube_xarm6_multiseed",
            "command": [
                python_executable,
                "scripts/pullcube_multiseed_eval.py",
                "--seeds",
                seeds,
                "--sim-backend",
                sim_backend,
                "--render-backend",
                render_backend,
                "--max-episode-steps",
                str(max_episode_steps),
                "--output-dir",
                str(pull_path.parent),
                "--jsonl-name",
                pull_path.name,
                "--md-name",
                pull_path.with_suffix(".md").name,
            ],
            "expected_artifact": str(pull_path),
        },
        {
            "id": "pickcube_xarm6_structured_probe",
            "command": [
                python_executable,
                "scripts/structured_probe_runner.py",
                "--case",
                expected_case,
                "--seed",
                "0",
                "--max-cases",
                str(probe_cases),
                "--sim-backend",
                sim_backend,
                "--render-backend",
                render_backend,
                "--max-episode-steps",
                str(probe_max_episode_steps),
                "--output-dir",
                str(probe_output_root),
            ],
            "expected_artifact": str(pick_path),
        },
    ]
    return {
        "schema": "motivating_evidence_collection_plan.v1",
        "pull_results": str(pull_path),
        "pick_probe": str(pick_path),
        "commands": commands,
    }


def _execute_frozen_remote_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Verify and execute the exact benchmark plan shipped in a remote package."""

    plan_path = _repo_path(args.plan_file)
    manifest_path = _repo_path(args.package_manifest)
    package_verification = verify_remote_package_manifest(manifest_path)
    try:
        raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raw_plan = {}
        plan_errors = [f"cannot read frozen plan: {exc!r}"]
        plan = {}
    else:
        plan, plan_errors = _validate_and_rebind_frozen_plan(raw_plan)

    launch = {
        "schema": "frozen_remote_plan_execution.v1",
        "plan_file": str(plan_path),
        "package_verification": package_verification,
        "plan_valid": not plan_errors,
        "plan_errors": plan_errors,
        "executed": False,
    }
    if not package_verification.get("valid") or plan_errors:
        launch["blocked_reason"] = "package or frozen-plan integrity check failed"
        return launch

    tier_id = str((plan.get("tier") or {}).get("tier_id"))
    runtime_config = plan.get("runtime_config") or {}
    sim_backend = str(runtime_config.get("sim_backend") or "auto")
    render_backend = str(runtime_config.get("render_backend") or "gpu")
    case_ids = [str(item.get("case_id")) for item in plan.get("case_definitions") or []]
    method_ids = [str(item.get("method_id")) for item in plan.get("method_definitions") or []]
    uses_llm = any(PAPER_METHODS[item].uses_llm for item in method_ids)
    output_root = str(plan.get("output_root"))
    plan_name = str(plan.get("plan_name"))
    plan_dir = REPO_ROOT / output_root / plan_name
    wrote = write_paper_plan(plan, plan_dir)

    frozen_llm_environment = _apply_frozen_llm_environment(
        plan.get("llm_config") or {},
        required=uses_llm,
    )

    audit = run_protocol_audit()
    preflight = run_paper_preflight(
        case_ids,
        seed=args.seed,
        sim_backend=sim_backend,
        render_backend=render_backend,
        static_only=False,
    )
    llm_preflight = run_llm_preflight(
        method_ids,
        require_stochastic_repetitions=int((plan.get("tier") or {}).get("repetitions") or 0) > 1,
    )
    expected_llm_config = plan.get("llm_config") or {}
    observed_llm_config = _llm_protocol_snapshot(llm_preflight)
    llm_config_matches = not uses_llm or (
        observed_llm_config == expected_llm_config
    )
    llm_preflight = {
        **llm_preflight,
        "frozen_config": expected_llm_config,
        "observed_config": observed_llm_config,
        "matches_frozen_plan": llm_config_matches,
        "ready": bool(llm_preflight.get("ready")) and llm_config_matches,
        "errors": list(llm_preflight.get("errors") or [])
        + ([] if llm_config_matches else ["runtime LLM configuration does not match frozen plan"]),
    }
    launch.update(
        plan_name=plan_name,
        tier=tier_id,
        wrote=wrote,
        frozen_llm_environment=frozen_llm_environment,
        protocol_audit=audit,
        preflight=preflight,
        llm_preflight=llm_preflight,
    )
    if not audit.get("valid") or not preflight.get("ready") or not llm_preflight.get("ready"):
        launch["blocked_reason"] = "protocol, simulator, or LLM preflight failed"
        return launch

    source = run_source_baseline_gate(
        case_ids,
        tier_id=tier_id,
        output_dir=plan_dir / "source_baselines",
        sim_backend=sim_backend,
        render_backend=render_backend,
        resume=not args.rerun_completed,
    )
    launch["source_baseline_gate"] = source
    if not source.get("ready"):
        launch["blocked_reason"] = "source baseline gate failed"
        return launch

    summary = _run_plan(plan, plan_dir, resume=not args.rerun_completed)
    summary["source_baseline_gate"] = source
    launch["summary_files"] = _write_benchmark_summary(summary, plan_dir)
    verification = verify_paper_plan(plan_dir)
    verification_files = write_verification(verification, plan_dir)
    evidence_export = None
    if (
        verification.get("valid") is True
        and verification.get("complete") is True
        and not summary.get("num_infrastructure_failures")
    ):
        evidence_export = build_evidence_export(
            plan_dir=plan_dir,
            output_dir=REPO_ROOT / output_root / "exports",
            verification=verification,
        )
    launch.update(
        executed=True,
        summary=summary,
        evidence_verification=verification,
        verification_files=verification_files,
        evidence_export=evidence_export,
        valid=bool(verification.get("valid"))
        and not summary.get("num_infrastructure_failures"),
    )
    return launch


def _run_logged_command(command: Sequence[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return int(process.wait())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=(
            "plan", "run", "pilot", "start", "execute", "status", "list", "preflight", "audit", "source", "verify", "export", "package",
            "diagnosis-template", "diagnosis-eval", "motivation", "evidence", "smoke",
        ),
        default="plan",
    )
    parser.add_argument("--tier", choices=tuple(PAPER_TIERS), default="pilot")
    parser.add_argument("--methods", default="all", help="Comma list such as B0,B2,Ours or all.")
    parser.add_argument("--cases", default="all", help="Comma list of registered case ids or all.")
    parser.add_argument("--output-root", default="results/paper")
    parser.add_argument("--plan-name", default="")
    parser.add_argument(
        "--plan-file",
        default="remote_plan/plan.json",
        help="Frozen packaged plan used by the execute action.",
    )
    parser.add_argument(
        "--package-manifest",
        default="remote_plan/package_manifest.json",
        help="Checksummed package manifest used by the execute action.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used by the preflight reset.")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument(
        "--trial-jsonl",
        action="append",
        default=[],
        help="Trial JSONL input for diagnosis-template; may be passed more than once.",
    )
    parser.add_argument(
        "--annotations",
        default="results/paper/diagnosis_annotations.jsonl",
        help="Human annotation JSONL used by diagnosis-template/eval.",
    )
    parser.add_argument(
        "--annotations-secondary",
        default="",
        help="Optional independently labeled JSONL for inter-annotator agreement.",
    )
    parser.add_argument(
        "--pull-results",
        default="results/pullcube_xarm6_multiseed.jsonl",
        help="Tracked real PullCube multi-seed JSON/JSONL used by motivation.",
    )
    parser.add_argument(
        "--pick-probe",
        default=(
            "results/structured_probes/case03_pick_cube_panda_to_xarm6/"
            "pick_cube_xarm6_close_envelope.json"
        ),
        help="Tracked real PickCube structured-probe JSON used by motivation.",
    )
    parser.add_argument(
        "--motivation-output",
        default="results/paper/motivating_examples",
        help="Output directory for machine-generated motivating examples.",
    )
    parser.add_argument("--motivation-seeds", default="0-9")
    parser.add_argument("--motivation-probe-cases", type=int, default=32)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--probe-max-episode-steps", type=int, default=220)
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Check files and Python interfaces without creating ManiSkill environments.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run completed matching items again. By default an existing plan resumes safely.",
    )
    args = parser.parse_args()

    if args.action == "list":
        _list_protocol()
        return

    if args.action == "execute":
        payload = _execute_frozen_remote_plan(args)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
        if not payload.get("valid"):
            raise SystemExit(2)
        return

    if args.action == "start":
        payload = _start_frozen_remote_plan(args)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
        if not payload.get("valid"):
            raise SystemExit(2)
        return

    if args.action == "status":
        print(json.dumps(_paper_status(args), indent=2, ensure_ascii=False, default=repr))
        return

    if args.action == "diagnosis-template":
        if not args.trial_jsonl:
            parser.error("diagnosis-template requires at least one --trial-jsonl")
        payload = build_annotation_template(
            [REPO_ROOT / path for path in args.trial_jsonl],
            REPO_ROOT / args.annotations,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.action == "diagnosis-eval":
        annotation_path = REPO_ROOT / args.annotations
        payload = evaluate_annotations(
            annotation_path,
            secondary_path=(REPO_ROOT / args.annotations_secondary) if args.annotations_secondary else None,
        )
        output_dir = annotation_path.parent
        json_path = output_dir / "diagnosis_evaluation.json"
        md_path = output_dir / "diagnosis_evaluation.md"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(diagnosis_evaluation_markdown(payload), encoding="utf-8")
        print(json.dumps({**payload, "wrote": {"json": str(json_path), "markdown": str(md_path)}}, indent=2, ensure_ascii=False))
        if not payload.get("ready"):
            raise SystemExit(2)
        return

    if args.action == "motivation":
        payload = build_motivating_examples(
            pull_results=REPO_ROOT / args.pull_results,
            pick_probe=REPO_ROOT / args.pick_probe,
        )
        wrote = write_motivating_examples(payload, REPO_ROOT / args.motivation_output)
        print(json.dumps({**payload, "wrote": wrote}, indent=2, ensure_ascii=False))
        if not payload.get("paper_ready"):
            raise SystemExit(2)
        return

    if args.action == "evidence":
        plan = _build_motivation_evidence_plan(
            python_executable=sys.executable,
            pull_results=args.pull_results,
            pick_probe=args.pick_probe,
            seeds=args.motivation_seeds,
            max_episode_steps=args.max_episode_steps,
            probe_max_episode_steps=args.probe_max_episode_steps,
            probe_cases=args.motivation_probe_cases,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
        )
        output_dir = _repo_path(args.motivation_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "evidence_collection.json"
        if args.static_only:
            manifest = {**plan, "executed": False, "ready": False}
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(json.dumps({**manifest, "wrote": str(manifest_path)}, indent=2, ensure_ascii=False))
            return

        log_path = output_dir / "evidence_collection.log"
        command_results = []
        for item in plan["commands"]:
            print(f"[evidence] {item['id']}", flush=True)
            returncode = _run_logged_command(item["command"], log_path=log_path)
            artifact = Path(item["expected_artifact"])
            command_results.append(
                {
                    "id": item["id"],
                    "returncode": returncode,
                    "artifact_exists": artifact.exists(),
                    "expected_artifact": str(artifact),
                }
            )
        examples = build_motivating_examples(
            pull_results=Path(plan["pull_results"]),
            pick_probe=Path(plan["pick_probe"]),
        )
        wrote = write_motivating_examples(examples, output_dir)
        ready = bool(examples.get("paper_ready")) and all(
            item["returncode"] == 0 and item["artifact_exists"] for item in command_results
        )
        manifest = {
            **plan,
            "executed": True,
            "ready": ready,
            "command_results": command_results,
            "motivating_examples": examples,
            "wrote": {**wrote, "log": str(log_path), "manifest": str(manifest_path)},
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        if not ready:
            raise SystemExit(2)
        return

    if args.action == "audit":
        payload = run_protocol_audit()
        wrote = write_protocol_audit(payload, REPO_ROOT / args.output_root)
        print(json.dumps({**payload, "wrote": wrote}, indent=2, ensure_ascii=False, default=repr))
        return

    if args.action in {"verify", "export"}:
        if not args.plan_name:
            parser.error(f"{args.action} requires --plan-name")
        plan_dir = REPO_ROOT / args.output_root / args.plan_name
        verification = verify_paper_plan(plan_dir)
        wrote = write_verification(verification, plan_dir)
        if not verification.get("valid"):
            print(json.dumps({**verification, "wrote": wrote}, indent=2, ensure_ascii=False, default=repr))
            raise SystemExit(2)
        if args.action == "export":
            exported = build_evidence_export(
                plan_dir=plan_dir,
                output_dir=REPO_ROOT / args.output_root / "exports",
                verification=verification,
            )
            print(json.dumps({**exported, "verification": verification}, indent=2, ensure_ascii=False, default=repr))
            return
        print(json.dumps({**verification, "wrote": wrote}, indent=2, ensure_ascii=False, default=repr))
        return

    case_ids = _selection(
        args.cases or ",".join(case.case_id for case in iter_full_migration_cases() if case.benchmark_enabled),
        [case.case_id for case in iter_full_migration_cases()],
    )
    pilot_checks: Dict[str, Any] = {}
    pilot_plan_name = ""
    if args.action == "pilot":
        args.tier = "pilot"
        pilot_plan_name = args.plan_name or "pilot_v1"
        launch_dir = REPO_ROOT / args.output_root / pilot_plan_name
        audit = run_protocol_audit()
        preflight = run_paper_preflight(
            case_ids,
            seed=args.seed,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            static_only=False,
        )
        selected_methods = _selection(args.methods, PAPER_METHODS, normalize=normalize_method_id)
        llm_preflight = run_llm_preflight(selected_methods)
        pilot_checks = {
            "protocol_audit": audit,
            "preflight": preflight,
            "llm_preflight": llm_preflight,
            "wrote": {
                "audit": write_protocol_audit(audit, launch_dir),
                "preflight": write_paper_preflight(preflight, launch_dir),
                "llm_preflight": write_llm_preflight(llm_preflight, launch_dir),
            },
        }
        if not audit.get("valid") or not preflight.get("ready") or not llm_preflight.get("ready"):
            print(
                json.dumps(
                    {
                        "executed": False,
                        "reason": "pilot launch blocked by protocol, runtime, or LLM configuration preflight",
                        "plan_name": pilot_plan_name,
                        "checks": pilot_checks,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=repr,
                )
            )
            raise SystemExit(2)
        args.action = "run"
    if args.action == "preflight":
        payload = run_paper_preflight(
            case_ids,
            seed=args.seed,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            static_only=args.static_only,
        )
        wrote = write_paper_preflight(payload, REPO_ROOT / args.output_root)
        print(json.dumps({**payload, "wrote": wrote}, indent=2, ensure_ascii=False, default=repr))
        return

    if args.action == "source":
        source_plan_name = args.plan_name or f"{args.tier}_source_baseline"
        source = run_source_baseline_gate(
            case_ids,
            tier_id=args.tier,
            output_dir=REPO_ROOT / args.output_root / source_plan_name / "source_baselines",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            resume=not args.rerun_completed,
        )
        print(json.dumps(source, indent=2, ensure_ascii=False, default=repr))
        if not source.get("ready"):
            raise SystemExit(2)
        return

    method_ids = _selection(args.methods, PAPER_METHODS, normalize=normalize_method_id)
    planning_llm_preflight = (
        pilot_checks.get("llm_preflight")
        if pilot_checks
        else run_llm_preflight(
            method_ids,
            require_stochastic_repetitions=PAPER_TIERS[args.tier].repetitions > 1,
        )
    )
    llm_config = _llm_protocol_snapshot(planning_llm_preflight)
    plan_name = pilot_plan_name or args.plan_name or f"{args.tier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plan = build_paper_plan(
        tier_id=args.tier,
        method_ids=method_ids,
        case_ids=case_ids,
        output_root=args.output_root,
        plan_name=plan_name,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        llm_config=llm_config,
        dry_run_commands=args.action == "smoke",
    )
    plan_dir = REPO_ROOT / args.output_root / plan_name
    wrote = write_paper_plan(plan, plan_dir)
    if args.action == "package":
        package_audit = run_protocol_audit()
        package_static_preflight = run_paper_preflight(
            case_ids,
            seed=args.seed,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            static_only=True,
        )
        preparation_checks = {
            "protocol_audit_valid": package_audit.get("valid") is True,
            "protocol_checks_passed": package_audit.get("num_passed"),
            "protocol_checks_total": package_audit.get("num_checks"),
            "static_preflight_ready": package_static_preflight.get("ready") is True,
            "llm_config_frozen": (
                not any(PAPER_METHODS[item].uses_llm for item in method_ids)
                or bool(llm_config.get("provider") and llm_config.get("model") and llm_config.get("max_tokens"))
            ),
            "llm_config_explicit": (
                not any(PAPER_METHODS[item].uses_llm for item in method_ids)
                or all(
                    (planning_llm_preflight.get("explicit_configuration") or {}).get(key) is True
                    for key in (
                        "provider",
                        "model",
                        "max_tokens",
                        "temperature",
                        "deepseek_thinking",
                    )
                )
            ),
            "stochastic_repetitions_configured": (
                PAPER_TIERS[args.tier].repetitions <= 1
                or not any(PAPER_METHODS[item].uses_llm for item in method_ids)
                or float(llm_config.get("temperature") or 0.0) > 0.0
            ),
        }
        if not all(
            preparation_checks[key]
            for key in (
                "protocol_audit_valid",
                "static_preflight_ready",
                "llm_config_frozen",
                "llm_config_explicit",
                "stochastic_repetitions_configured",
            )
        ):
            print(
                json.dumps(
                    {
                        "packaged": False,
                        "reason": "remote-package preparation check failed",
                        "preparation_checks": preparation_checks,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            raise SystemExit(2)
        package = build_remote_package(
            output_dir=REPO_ROOT / args.output_root / "packages",
            plan=plan,
            plan_files=wrote,
            preparation_checks=preparation_checks,
        )
        print(json.dumps(package, indent=2, ensure_ascii=False, default=repr))
        return
    if args.action == "smoke":
        summary = _run_plan(plan, plan_dir, resume=False)
        nested_preview_audit = _smoke_nested_preview_audit(plan, plan_dir)
        summary_path = plan_dir / "smoke_summary.json"
        summary["nested_preview_audit"] = nested_preview_audit
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=repr),
            encoding="utf-8",
        )
        payload = {
            "schema": "paper_benchmark_smoke.v1",
            "plan_name": plan_name,
            "tier": args.tier,
            "executed": True,
            "num_ready_runs": plan["num_ready_runs"],
            "num_blocked_runs": plan["num_blocked_runs"],
            "num_smoke_completed": summary["num_completed_runs"],
            "num_infrastructure_failures": summary["num_infrastructure_failures"],
            "num_nested_previews_valid": nested_preview_audit["num_valid"],
            "num_nested_previews_expected": nested_preview_audit["num_expected"],
            "num_controlled_pairs_valid": nested_preview_audit["num_controlled_pairs_valid"],
            "num_controlled_pairs_expected": nested_preview_audit["num_controlled_pairs_expected"],
            "nested_preview_audit_valid": nested_preview_audit["valid"],
            "wrote": {**wrote, "smoke_summary": str(summary_path)},
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
        if summary["num_infrastructure_failures"] or not nested_preview_audit["valid"]:
            raise SystemExit(2)
        return
    payload: Dict[str, Any] = {
        "plan_name": plan_name,
        "tier": args.tier,
        "num_runs": plan["num_runs"],
        "num_ready_runs": plan["num_ready_runs"],
        "num_blocked_runs": plan["num_blocked_runs"],
        "wrote": wrote,
        "executed": False,
    }
    if pilot_checks:
        payload["launch_checks"] = pilot_checks

    if args.action == "run":
        llm_preflight = (pilot_checks.get("llm_preflight") if pilot_checks else None) or run_llm_preflight(
            method_ids,
            require_stochastic_repetitions=PAPER_TIERS[args.tier].repetitions > 1,
        )
        payload["llm_preflight"] = llm_preflight
        if not pilot_checks:
            payload.setdefault("wrote", {})["llm_preflight"] = write_llm_preflight(llm_preflight, plan_dir)
        if not llm_preflight.get("ready"):
            payload["blocked_reason"] = "LLM configuration preflight failed"
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
            raise SystemExit(2)
        source = run_source_baseline_gate(
            case_ids,
            tier_id=args.tier,
            output_dir=plan_dir / "source_baselines",
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            resume=not args.rerun_completed,
        )
        payload["source_baseline_gate"] = source
        if not source.get("ready"):
            payload["blocked_reason"] = "source task did not meet the frozen seed-split acceptance threshold"
            print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
            raise SystemExit(2)
        summary = _run_plan(plan, plan_dir, resume=not args.rerun_completed)
        summary["source_baseline_gate"] = source
        summary_files = _write_benchmark_summary(summary, plan_dir)
        payload.update(
            executed=True,
            summary=summary,
            summary_files=summary_files,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))


if __name__ == "__main__":
    main()
