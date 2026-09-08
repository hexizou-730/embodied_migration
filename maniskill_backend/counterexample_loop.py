"""Counterexample-guided embodiment adapter synthesis (CEGIS).

The loop keeps the high-level program, controller, simulator, and task success
signal frozen. It alternates between target-adapter generation, development
seed validation, counterexample selection, active probing, and guarded repair.
Held-out seeds are evaluated only after the development acceptance threshold is
met and are never fed back into the repair loop.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.counterexamples import (
    select_counterexample,
    write_counterexample_set,
)
from maniskill_backend.evidence import archive_cegis_evidence
from maniskill_backend.structured_probe import get_probe_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_BUDGET_SCOPE = "per_run_total"
HELD_OUT_FEEDBACK_POLICY = "evaluation_only_no_repair"
PROBE_STRATEGIES = {"active", "fixed_grid"}


@dataclass(frozen=True)
class CEGISConfig:
    case_id: str
    development_seeds: str = "0-4"
    held_out_seeds: str = "100-109"
    max_cycles: int = 3
    attempts_per_cycle: int = 1
    probe_budget: int = 8
    probe_strategy: str = "active"
    prompt_policy: str = "full"
    success_threshold: float = 0.8
    min_trials_for_accept: int = 5
    obs_mode: str = "state"
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    max_episode_steps: int = 0
    trial_timeout_s: int = 900
    test_timeout_s: int = 240
    from_zero: bool = True
    source_check: bool = True
    dry_run: bool = False
    evidence_root: str = "evidence/runs"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in command)
    if dry_run:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"$ {printable}\n[dry-run]\n")
        return {"returncode": 0, "command": list(command), "dry_run": True}
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {printable}\n")
        stream.write(completed.stdout or "")
        if completed.stdout and not completed.stdout.endswith("\n"):
            stream.write("\n")
    return {
        "returncode": int(completed.returncode),
        "command": list(command),
        "output_tail": (completed.stdout or "")[-4000:],
        "dry_run": False,
    }


def _read_multiseed_jsonl(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("type") == "metadata":
            metadata = item
        elif item.get("type") == "trial":
            rows.append(item)
    successes = [row for row in rows if bool(row.get("success"))]
    return {
        **metadata,
        "rows": rows,
        "num_trials": len(rows),
        "num_success": len(successes),
        "num_failure": len(rows) - len(successes),
        "success_rate": round(len(successes) / len(rows), 4) if rows else 0.0,
    }


def _accepted(summary: Mapping[str, Any], config: CEGISConfig) -> bool:
    return (
        int(summary.get("num_trials") or 0) >= config.min_trials_for_accept
        and float(summary.get("success_rate") or 0.0) >= config.success_threshold
    )


def _restore_seed_adapter(
    case: FullMigrationCase,
    run_dir: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    target = REPO_ROOT / case.target_adapter_path
    seed = REPO_ROOT / case.seed_adapter_path
    backup = run_dir / "adapter_before_run.py"
    details = {
        "target": case.target_adapter_path,
        "seed": case.seed_adapter_path,
        "backup": str(backup),
        "restored": False,
    }
    if not case.seed_adapter_path:
        details["message"] = "No seed adapter is registered."
        return details
    if not seed.exists():
        raise FileNotFoundError(f"Missing seed adapter: {seed}")
    if dry_run:
        details["message"] = "Dry run: seed adapter would be restored."
        return details
    if target.exists():
        shutil.copy2(target, backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed, target)
    details["restored"] = True
    details["message"] = "Neutral seed adapter restored."
    return details


def _module_generation_command(
    case: FullMigrationCase,
    config: CEGISConfig,
    cycle_dir: Path,
    *,
    seed: int,
    counterexample_path: Path | None,
    probe_path: Path | None,
    cycle: int,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "maniskill_backend.module_generation_runner",
        "--case",
        case.case_id,
        "--max-attempts",
        str(config.attempts_per_cycle),
        "--seed",
        str(seed),
        "--obs-mode",
        config.obs_mode,
        "--sim-backend",
        config.sim_backend,
        "--render-backend",
        config.render_backend,
        "--trial-timeout-s",
        str(config.trial_timeout_s),
        "--test-timeout-s",
        str(config.test_timeout_s),
        "--jsonl",
        str(cycle_dir / "module_generation.jsonl"),
        "--md",
        str(cycle_dir / "module_generation.md"),
        "--no-analysis",
        "--no-implicit-probe-feedback",
        "--force-regeneration",
        "--prompt-policy",
        config.prompt_policy,
    ]
    if not config.source_check or cycle > 1:
        command.append("--no-source-check")
    if counterexample_path:
        command.extend(["--counterexample-json", str(counterexample_path)])
    if probe_path:
        command.extend(["--probe-feedback-json", str(probe_path)])
    if config.dry_run:
        command.append("--dry-run")
    return command


def _multiseed_command(
    case: FullMigrationCase,
    config: CEGISConfig,
    output_dir: Path,
    *,
    seeds: str,
    name: str,
) -> List[str]:
    return [
        sys.executable,
        "scripts/multiseed_eval.py",
        "--seeds",
        seeds,
        "--task",
        case.task_id,
        "--robot",
        case.target_robot,
        "--code-file",
        case.target_program_path,
        "--adapter-module",
        case.target_adapter_module,
        "--control-mode",
        case.target_control_mode,
        "--obs-mode",
        config.obs_mode,
        "--sim-backend",
        config.sim_backend,
        "--render-backend",
        config.render_backend,
        "--max-episode-steps",
        str(config.max_episode_steps or case.max_episode_steps),
        "--success-threshold",
        str(config.success_threshold),
        "--min-trials-for-accept",
        str(config.min_trials_for_accept),
        "--output-dir",
        str(output_dir),
        "--jsonl-name",
        f"{name}.jsonl",
        "--md-name",
        f"{name}.md",
    ]


def _probe_command(
    case: FullMigrationCase,
    config: CEGISConfig,
    cycle_dir: Path,
    *,
    counterexample_path: Path,
    previous_probe_path: Path | None,
    seed: int,
    budget: int,
    fixed_grid_offset: int = 0,
) -> List[str]:
    command = [
        sys.executable,
        "scripts/structured_probe_runner.py",
        "--case",
        case.case_id,
        "--seed",
        str(seed),
        "--active-counterexample-json",
        str(counterexample_path),
        "--probe-selection",
        config.probe_strategy,
        "--suggestion-budget",
        str(budget),
        "--max-cases",
        str(budget),
        "--obs-mode",
        config.obs_mode,
        "--sim-backend",
        config.sim_backend,
        "--render-backend",
        config.render_backend,
        "--max-episode-steps",
        str(config.max_episode_steps or case.max_episode_steps),
        "--output-dir",
        str(cycle_dir / "structured_probe"),
    ]
    if config.probe_strategy == "fixed_grid":
        command.extend(
            [
                "--fixed-grid-total-budget",
                str(config.probe_budget),
                "--fixed-grid-offset",
                str(fixed_grid_offset),
            ]
        )
    if previous_probe_path:
        command.extend(["--adaptive-from", str(previous_probe_path)])
    if config.dry_run:
        command.append("--dry-run")
    return command


def _probe_result_path(case: FullMigrationCase, cycle_dir: Path) -> Path | None:
    try:
        spec = get_probe_spec(case)
    except KeyError:
        return None
    return (
        cycle_dir
        / "structured_probe"
        / case.case_id
        / f"{spec.probe_id}.json"
    )


def _probe_case_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if payload.get("num_new_cases") is not None:
        return max(0, int(payload.get("num_new_cases") or 0))
    rows = payload.get("all_probe_cases") or payload.get("results") or []
    return max(0, int(payload.get("num_cases") or len(rows)))


def _probe_batch_budget(remaining: int, repair_opportunities: int) -> int:
    """Reserve a fair share of the total budget for later repair cycles."""

    if remaining <= 0 or repair_opportunities <= 0:
        return 0
    return min(remaining, max(1, (remaining + repair_opportunities - 1) // repair_opportunities))


def _dry_run_nested_command_preview(
    case: FullMigrationCase,
    config: CEGISConfig,
    cycle_dir: Path,
) -> Dict[str, Any]:
    """Preview the first repair path without inventing measured evidence."""

    counterexample_path = cycle_dir / "counterexamples.json"
    later_cycles = max(0, config.max_cycles - 1)
    allocated_probe_budget = _probe_batch_budget(config.probe_budget, later_cycles)
    probe_path = _probe_result_path(case, cycle_dir)
    probe_command: List[str] = []
    if _has_probe(case) and allocated_probe_budget > 0:
        probe_command = _probe_command(
            case,
            config,
            cycle_dir,
            counterexample_path=counterexample_path,
            previous_probe_path=None,
            seed=case.seed,
            budget=allocated_probe_budget,
            fixed_grid_offset=0,
        )

    repair_generation_command: List[str] = []
    if later_cycles > 0:
        repair_generation_command = _module_generation_command(
            case,
            config,
            cycle_dir.parent / "cycle_02",
            seed=case.seed,
            counterexample_path=counterexample_path,
            probe_path=probe_path if probe_command else None,
            cycle=2,
        )

    return {
        "schema": "cegis_nested_command_preview.v1",
        "assumption": (
            "Static preview assumes the development split yields one counterexample. "
            "No outcome or measurement is fabricated."
        ),
        "counterexample_selection": {
            "source": "development_only",
            "output_path": str(counterexample_path),
            "held_out_used": False,
        },
        "structured_probe": {
            "strategy": config.probe_strategy,
            "budget_allocated": allocated_probe_budget,
            "expected_result_path": str(probe_path) if probe_path else "",
            "command": probe_command,
        },
        "repair_generation": {
            "prompt_policy": config.prompt_policy,
            "command": repair_generation_command,
        },
        "held_out_evaluation": {
            "usage": "acceptance_only_no_repair",
            "command": _multiseed_command(
                case,
                config,
                cycle_dir.parent / "cycle_02",
                seeds=config.held_out_seeds,
                name="held_out",
            ),
        },
    }


def _adapter_provenance_from_cycles(run_dir: Path, *, from_zero: bool) -> str:
    for path in sorted(run_dir.glob("cycle_*/module_generation.jsonl")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            continue
        for attempt in payload.get("attempts") or []:
            if (
                attempt.get("used_llm")
                and attempt.get("module_applied")
                and attempt.get("module_kept")
            ):
                return "llm_generated_cegis"
    return "neutral_seed" if from_zero else "preexisting_adapter_unverified"


def _has_probe(case: FullMigrationCase) -> bool:
    try:
        get_probe_spec(case)
        return True
    except KeyError:
        return False


def run_counterexample_loop(
    config: CEGISConfig,
    *,
    output_root: Path | str = "results/cegis",
    run_name: str = "",
) -> Dict[str, Any]:
    if config.probe_strategy not in PROBE_STRATEGIES:
        raise ValueError(
            f"Unknown probe strategy {config.probe_strategy!r}; "
            f"expected one of {sorted(PROBE_STRATEGIES)!r}."
        )
    case = get_full_migration_case(config.case_id)
    name = run_name or f"{case.case_id}_{_timestamp()}"
    root = Path(output_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_dir / "commands.log"
    target_adapter = REPO_ROOT / case.target_adapter_path

    seed_restore = {
        "restored": False,
        "message": "Current adapter kept.",
    }
    if config.from_zero:
        seed_restore = _restore_seed_adapter(case, run_dir, dry_run=config.dry_run)

    cycles: List[Dict[str, Any]] = []
    selected_counterexample_path: Path | None = None
    latest_probe_path: Path | None = None
    counterexample_history: List[Dict[str, Any]] = []
    status = "cycle_budget_exhausted"
    held_out: Dict[str, Any] = {}
    final_cycle_dir: Path | None = None
    remaining_probe_budget = max(0, int(config.probe_budget))
    probe_cases_used = 0

    for cycle in range(1, config.max_cycles + 1):
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        final_cycle_dir = cycle_dir
        cycle_dir.mkdir(parents=True, exist_ok=True)
        counterexample: Dict[str, Any] = {}
        if selected_counterexample_path and selected_counterexample_path.exists():
            counterexample_payload = json.loads(
                selected_counterexample_path.read_text(encoding="utf-8")
            )
            counterexample = dict(counterexample_payload.get("selected") or {})
        repair_seed = int(counterexample.get("seed", case.seed))

        generation = _run_command(
            _module_generation_command(
                case,
                config,
                cycle_dir,
                seed=repair_seed,
                counterexample_path=selected_counterexample_path,
                probe_path=latest_probe_path,
                cycle=cycle,
            ),
            log_path=command_log,
            dry_run=config.dry_run,
        )
        cycle_record: Dict[str, Any] = {
            "cycle": cycle,
            "repair_seed": repair_seed,
            "repair_feedback": {
                "source_split": (
                    "development_counterexample"
                    if selected_counterexample_path
                    else "initial_case_seed"
                ),
                "counterexample_path": (
                    str(selected_counterexample_path)
                    if selected_counterexample_path
                    else ""
                ),
                "probe_path": str(latest_probe_path) if latest_probe_path else "",
                "held_out_used": False,
            },
            "generation": generation,
            "adapter_sha256_after_generation": _file_sha256(target_adapter),
        }
        if config.dry_run:
            cycle_record["development_evaluation"] = {
                "command": _multiseed_command(
                    case,
                    config,
                    cycle_dir,
                    seeds=config.development_seeds,
                    name="development",
                )
            }
            cycle_record["nested_command_preview"] = _dry_run_nested_command_preview(
                case,
                config,
                cycle_dir,
            )
            cycle_record["status"] = "dry_run_planned"
            cycles.append(cycle_record)
            status = "dry_run_planned"
            break
        if generation["returncode"] != 0:
            cycle_record["status"] = "generation_command_failed"
            cycles.append(cycle_record)
            status = "command_failed"
            break

        development = _run_command(
            _multiseed_command(
                case,
                config,
                cycle_dir,
                seeds=config.development_seeds,
                name="development",
            ),
            log_path=command_log,
            dry_run=False,
        )
        development_path = cycle_dir / "development.jsonl"
        development_summary = _read_multiseed_jsonl(development_path)
        cycle_record["development_evaluation"] = development
        cycle_record["development_summary"] = development_summary
        if development["returncode"] != 0 or not development_summary:
            cycle_record["status"] = "development_evaluation_failed"
            cycles.append(cycle_record)
            status = "command_failed"
            break

        if _accepted(development_summary, config):
            held_out_run = _run_command(
                _multiseed_command(
                    case,
                    config,
                    cycle_dir,
                    seeds=config.held_out_seeds,
                    name="held_out",
                ),
                log_path=command_log,
                dry_run=False,
            )
            held_out_summary = _read_multiseed_jsonl(cycle_dir / "held_out.jsonl")
            held_out = {
                "run": held_out_run,
                "summary": held_out_summary,
                "used_for_repair": False,
            }
            cycle_record["held_out_evaluation"] = held_out
            if held_out_run["returncode"] != 0:
                cycle_record["status"] = "held_out_evaluation_failed"
                status = "command_failed"
            elif _accepted(held_out_summary, config):
                cycle_record["status"] = "accepted_on_held_out"
                status = "accepted"
            else:
                cycle_record["status"] = "rejected_on_held_out"
                status = "held_out_rejected"
            cycles.append(cycle_record)
            break

        selected_counterexample_path = cycle_dir / "counterexamples.json"
        counterexample_set = write_counterexample_set(
            selected_counterexample_path,
            case,
            development_summary.get("rows") or [],
            repo_root=REPO_ROOT,
            selection_history=counterexample_history,
        )
        selected = counterexample_set.get("selected") or {}
        cycle_record["selected_counterexample"] = selected
        cycle_record["counterexample_path"] = str(selected_counterexample_path)
        if not selected:
            cycle_record["status"] = "no_counterexample_available"
            cycles.append(cycle_record)
            status = "diagnosis_failed"
            break
        counterexample_history.append(
            {
                "seed": selected.get("seed"),
                "failure_reason": selected.get("failure_reason"),
                "stage": selected.get("stage"),
                "selection_score": selected.get("selection_score"),
            }
        )

        repair_opportunities = max(0, config.max_cycles - cycle)
        allocated_probe_budget = _probe_batch_budget(
            remaining_probe_budget,
            repair_opportunities,
        )
        if _has_probe(case) and allocated_probe_budget > 0:
            probe_run = _run_command(
                _probe_command(
                    case,
                    config,
                    cycle_dir,
                    counterexample_path=selected_counterexample_path,
                    previous_probe_path=latest_probe_path,
                    seed=int(selected.get("seed", case.seed)),
                    budget=allocated_probe_budget,
                    fixed_grid_offset=probe_cases_used,
                ),
                log_path=command_log,
                dry_run=False,
            )
            candidate_probe_path = _probe_result_path(case, cycle_dir)
            if candidate_probe_path and candidate_probe_path.exists():
                latest_probe_path = candidate_probe_path
            measured_probe_cases = min(
                remaining_probe_budget,
                _probe_case_count(candidate_probe_path),
            )
            probe_cases_used += measured_probe_cases
            remaining_probe_budget -= measured_probe_cases
            probe_record = {
                "run": probe_run,
                "result_path": str(latest_probe_path) if latest_probe_path else "",
                "strategy": config.probe_strategy,
                "budget_scope": PROBE_BUDGET_SCOPE,
                "budget_allocated": allocated_probe_budget,
                "cases_measured": measured_probe_cases,
                "budget_remaining": remaining_probe_budget,
            }
            cycle_record["structured_probe"] = probe_record
            cycle_record[
                "active_probe" if config.probe_strategy == "active" else "fixed_grid_probe"
            ] = probe_record
            if probe_run["returncode"] != 0:
                cycle_record["status"] = "structured_probe_failed"
                cycles.append(cycle_record)
                status = "command_failed"
                break
        elif _has_probe(case) and remaining_probe_budget <= 0:
            cycle_record["structured_probe"] = {
                "skipped": True,
                "reason": "Per-run probe budget exhausted; reuse prior measurements.",
                "budget_scope": PROBE_BUDGET_SCOPE,
                "budget_remaining": 0,
            }
        elif _has_probe(case):
            cycle_record["structured_probe"] = {
                "skipped": True,
                "reason": "No later repair cycle remains; do not spend unused probe budget.",
                "budget_scope": PROBE_BUDGET_SCOPE,
                "budget_remaining": remaining_probe_budget,
            }
        else:
            cycle_record["structured_probe"] = {
                "skipped": True,
                "reason": "No executable structured probe is registered for this case.",
            }
        cycle_record["status"] = "counterexample_ready_for_next_cycle"
        cycles.append(cycle_record)

    payload = {
        "schema": "counterexample_guided_adapter_synthesis.v1",
        "method": (
            "counterexample_guided_embodiment_adapter_synthesis"
            if config.probe_strategy == "active"
            else "fixed_grid_counterexample_guided_adapter_synthesis"
        ),
        "case_id": case.case_id,
        "task_id": case.task_id,
        "source_robot": case.source_robot,
        "target_robot": case.target_robot,
        "frozen_interfaces": [
            "high_level_program",
            "low_level_controller",
            "simulator",
            "task_success_signal",
        ],
        "config": asdict(config),
        "run_dir": str(run_dir),
        "seed_restore": seed_restore,
        "status": status,
        "success": status == "accepted",
        "cycles": cycles,
        "nested_command_preview": (
            cycles[0].get("nested_command_preview", {}) if cycles else {}
        ),
        "counterexample_history": counterexample_history,
        "held_out": held_out,
        "probe_budget": {
            "scope": PROBE_BUDGET_SCOPE,
            "total": int(config.probe_budget),
            "cases_used": probe_cases_used,
            "remaining": remaining_probe_budget,
        },
        "final_adapter_sha256": _file_sha256(target_adapter),
        "evidence_policy": {
            "development_seeds_used_for_repair": True,
            "held_out_seeds_used_for_repair": False,
            "held_out_feedback_policy": HELD_OUT_FEEDBACK_POLICY,
            "allowed_repair_feedback": [
                "initial_case_seed",
                "development_counterexample",
                f"{config.probe_strategy}_probe_on_development_counterexample",
            ],
            "success_requires_held_out_threshold": True,
        },
    }
    if status == "accepted" and final_cycle_dir is not None and not config.dry_run:
        evidence_root = Path(config.evidence_root)
        if not evidence_root.is_absolute():
            evidence_root = REPO_ROOT / evidence_root
        payload["evidence_bundle"] = archive_cegis_evidence(
            evidence_root=evidence_root,
            run_name=name,
            case=case,
            adapter_path=target_adapter,
            cycle_dir=final_cycle_dir,
            command_log=command_log,
            summary=payload,
            adapter_provenance=_adapter_provenance_from_cycles(
                run_dir,
                from_zero=config.from_zero,
            ),
            repo_root=REPO_ROOT,
        )
    _write_json(run_dir / "summary.json", payload)
    (run_dir / "summary.md").write_text(
        counterexample_loop_markdown(payload),
        encoding="utf-8",
    )
    return payload


def counterexample_loop_markdown(payload: Mapping[str, Any]) -> str:
    config = payload.get("config") or {}
    lines = [
        "# Counterexample-Guided Adapter Synthesis",
        "",
        f"- case: `{payload.get('case_id')}`",
        f"- source -> target: `{payload.get('source_robot')} -> {payload.get('target_robot')}`",
        f"- status: `{payload.get('status')}`",
        f"- success: `{payload.get('success')}`",
        f"- development seeds: `{config.get('development_seeds')}`",
        f"- held-out seeds: `{config.get('held_out_seeds')}`",
        f"- probe budget: `{(payload.get('probe_budget') or {}).get('cases_used')}/"
        f"{(payload.get('probe_budget') or {}).get('total')}` cases, scope=`{PROBE_BUDGET_SCOPE}`",
        "",
        "## Cycles",
        "",
        "| cycle | dev success rate | counterexample | probe | status |",
        "|---:|---:|---|---|---|",
    ]
    for item in payload.get("cycles") or []:
        development = item.get("development_summary") or {}
        counterexample = item.get("selected_counterexample") or {}
        probe = item.get("structured_probe") or item.get("active_probe") or {}
        lines.append(
            f"| {item.get('cycle')} | {development.get('success_rate', '')} | "
            f"{counterexample.get('failure_reason', '')}@seed={counterexample.get('seed', '')} | "
            f"{probe.get('result_path') or ('skipped' if probe.get('skipped') else '')} | "
            f"{item.get('status')} |"
        )
    held_out = (payload.get("held_out") or {}).get("summary") or {}
    if held_out:
        lines.extend(
            [
                "",
                "## Held-Out Evaluation",
                "",
                f"- trials: `{held_out.get('num_trials')}`",
                f"- success rate: `{held_out.get('success_rate')}`",
                "- held-out results were not fed back into repair.",
            ]
        )
    lines.append("")
    return "\n".join(lines)
