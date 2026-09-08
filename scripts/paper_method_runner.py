"""Execute one frozen method/case/repetition from the paper benchmark plan."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_client import (
    completion_token_limit,
    current_provider,
    deepseek_thinking_mode,
    default_model,
    generation_temperature,
)
from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.evidence import detect_adapter_provenance
from maniskill_backend.paper_benchmark import (
    PAPER_METHODS,
    PAPER_TIERS,
    method_llm_call_budget,
    normalize_method_id,
)


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_interface_hashes(case: FullMigrationCase) -> Dict[str, str]:
    """Hash repository interfaces that no generation method may edit."""

    paths = {
        "high_level_program": REPO_ROOT / case.target_program_path,
        "neutral_seed_adapter": REPO_ROOT / case.seed_adapter_path,
        "controller_bridge": REPO_ROOT / "maniskill_backend/skill_adapter.py",
        "task_definition": REPO_ROOT / "maniskill_backend/tasks.py",
        "success_evaluation": REPO_ROOT / "maniskill_backend/evaluation.py",
        "environment_adapter": REPO_ROOT / "maniskill_backend/env_adapter.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _frozen_integrity(before: Mapping[str, str], after: Mapping[str, str]) -> Dict[str, Any]:
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    return {
        "valid": not changed,
        "changed_interfaces": changed,
        "before": dict(before),
        "after": dict(after),
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def _run_command(command: Sequence[str], *, log_path: Path, dry_run: bool) -> Dict[str, Any]:
    printable = " ".join(str(item) for item in command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"$ {printable}\n[dry-run]\n")
        return {
            "command": list(command),
            "returncode": 0,
            "dry_run": True,
            "wall_time_seconds": 0.0,
        }
    started = perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    wall_time_seconds = round(perf_counter() - started, 6)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {printable}\n")
        stream.write(output)
        if output and not output.endswith("\n"):
            stream.write("\n")
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "output_tail": output[-4000:],
        "dry_run": False,
        "wall_time_seconds": wall_time_seconds,
    }


def _read_last_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(rows[-1]) if rows else {}


def _read_multiseed(path: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    rows = []
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
    successes = sum(bool(row.get("success")) for row in rows)
    return {
        **metadata,
        "num_trials": len(rows),
        "num_success": successes,
        "num_failure": len(rows) - successes,
        "success_rate": round(successes / len(rows), 4) if rows else 0.0,
        "rows": rows,
    }


def _module_path(module_name: str) -> Path:
    return REPO_ROOT / (module_name.replace(".", "/") + ".py")


def _baseline_module(case: FullMigrationCase, method_id: str) -> str:
    suffixes = {
        "pull_cube": "pull_cube",
        "pick_cube": "pick_cube",
        "push_cube": "push_cube",
    }
    try:
        suffix = suffixes[case.task_id]
    except KeyError as exc:
        raise KeyError(f"No frozen baseline adapter for task {case.task_id!r}.") from exc
    if method_id == "B0":
        return f"maniskill_backend.baseline_adapters.source_{suffix}"
    if method_id == "B1":
        return f"maniskill_backend.baseline_adapters.shared_{suffix}"
    raise KeyError(method_id)


def _oracle_module(case: FullMigrationCase) -> str:
    if case.case_id == "case02_pull_cube_panda_to_xarm6":
        return "maniskill_backend.oracles.xarm6_pull_cube_oracle"
    if case.case_id == "case01_pull_cube_panda_to_fetch":
        path = REPO_ROOT / case.target_adapter_path
        seed = REPO_ROOT / case.seed_adapter_path
        provenance = detect_adapter_provenance(path, seed_path=seed)
        if provenance != "hand_written_oracle":
            raise RuntimeError(
                "Fetch Oracle requires the committed hand-written oracle adapter; "
                f"observed provenance={provenance!r}."
            )
        return case.target_adapter_module
    raise RuntimeError(f"No hand-written oracle is registered for {case.case_id}.")


def _multiseed_command(
    case: FullMigrationCase,
    *,
    adapter_module: str,
    seeds: str,
    output_dir: Path,
    name: str,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "scripts/multiseed_eval.py",
        "--seeds",
        seeds,
        "--task",
        case.task_id,
        "--robot",
        case.target_robot,
        "--method",
        f"paper-{args.method.lower()}",
        "--code-file",
        case.target_program_path,
        "--adapter-module",
        adapter_module,
        "--control-mode",
        case.target_control_mode,
        "--obs-mode",
        args.obs_mode,
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--success-threshold",
        str(args.success_threshold),
        "--min-trials-for-accept",
        str(args.min_trials_for_accept),
        "--output-dir",
        str(output_dir),
        "--jsonl-name",
        f"{name}.jsonl",
        "--md-name",
        f"{name}.md",
    ]


def _evaluate_adapter(
    case: FullMigrationCase,
    adapter_module: str,
    args: argparse.Namespace,
    run_dir: Path,
    command_log: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], list[Dict[str, Any]]]:
    commands = []
    for name, seeds in (
        ("development", args.development_seeds),
        ("held_out", args.held_out_seeds),
    ):
        result = _run_command(
            _multiseed_command(
                case,
                adapter_module=adapter_module,
                seeds=seeds,
                output_dir=run_dir,
                name=name,
                args=args,
            ),
            log_path=command_log,
            dry_run=args.dry_run,
        )
        commands.append(result)
    return (
        _read_multiseed(run_dir / "development.jsonl"),
        _read_multiseed(run_dir / "held_out.jsonl"),
        commands,
    )


def _generated_candidate_integrity(result: Mapping[str, Any]) -> Dict[str, Any]:
    attempts = list(result.get("attempts") or [])
    kept = [
        attempt
        for attempt in attempts
        if attempt.get("used_llm") and attempt.get("module_applied") and attempt.get("module_kept")
    ]
    usage_rows = [attempt.get("llm_usage") or {} for attempt in attempts]
    return {
        "used_llm": any(bool(attempt.get("used_llm")) for attempt in attempts),
        "num_attempts": len(attempts),
        "num_valid_modules": sum(bool(item.get("module_valid")) for item in attempts),
        "num_kept_llm_modules": len(kept),
        "valid_generated_candidate": bool(kept),
        "models": sorted({str(item.get("llm_model")) for item in attempts if item.get("llm_model")}),
        "api_calls": sum(bool(attempt.get("used_llm")) for attempt in attempts),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage_rows),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in usage_rows),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage_rows),
    }


def _run_direct_generation(
    case: FullMigrationCase,
    args: argparse.Namespace,
    run_dir: Path,
    command_log: Path,
) -> Dict[str, Any]:
    method = PAPER_METHODS[args.method]
    target_adapter = REPO_ROOT / case.target_adapter_path
    seed_adapter = REPO_ROOT / case.seed_adapter_path
    if not seed_adapter.exists():
        raise FileNotFoundError(f"Missing neutral seed adapter: {seed_adapter}")
    if not args.dry_run:
        shutil.copy2(seed_adapter, target_adapter)

    command = [
        sys.executable,
        "-m",
        "maniskill_backend.module_generation_runner",
        "--case",
        case.case_id,
        "--max-attempts",
        str(args.llm_call_budget),
        "--seed",
        str(case.seed),
        "--obs-mode",
        args.obs_mode,
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--trial-timeout-s",
        str(args.trial_timeout_s),
        "--test-timeout-s",
        str(args.test_timeout_s),
        "--prompt-policy",
        method.prompt_policy,
        "--force-regeneration",
        "--no-implicit-probe-feedback",
        "--no-analysis",
        "--jsonl",
        str(run_dir / "module_generation.jsonl"),
        "--md",
        str(run_dir / "module_generation.md"),
    ]
    if args.dry_run:
        command.append("--dry-run")
    generation_run = _run_command(command, log_path=command_log, dry_run=args.dry_run)
    generation_result = _read_last_jsonl(run_dir / "module_generation.jsonl")
    integrity = _generated_candidate_integrity(generation_result)
    if not args.dry_run and target_adapter.exists():
        shutil.copy2(target_adapter, run_dir / "final_adapter.py")

    development, held_out, evaluation_runs = _evaluate_adapter(
        case,
        case.target_adapter_module,
        args,
        run_dir,
        command_log,
    )
    return {
        "adapter_module": case.target_adapter_module,
        "adapter_path": str(target_adapter),
        "adapter_sha256": _sha256(target_adapter),
        "adapter_provenance": (
            "dry_run"
            if args.dry_run
            else "llm_generated_direct"
            if integrity["valid_generated_candidate"]
            else "generation_invalid"
        ),
        "generation_run": generation_run,
        "generation_result": generation_result,
        "generation_integrity": integrity,
        "development": development,
        "held_out": held_out,
        "evaluation_runs": evaluation_runs,
    }


def _run_cegis(
    case: FullMigrationCase,
    args: argparse.Namespace,
    run_dir: Path,
    command_log: Path,
) -> Dict[str, Any]:
    method = PAPER_METHODS[args.method]
    command = [
        sys.executable,
        "scripts/cegis_migration_runner.py",
        "--case",
        case.case_id,
        "--development-seeds",
        args.development_seeds,
        "--held-out-seeds",
        args.held_out_seeds,
        "--max-cycles",
        str(args.llm_call_budget),
        "--attempts-per-cycle",
        "1",
        "--probe-budget",
        str(args.probe_budget),
        "--probe-strategy",
        method.probe_strategy,
        "--prompt-policy",
        method.prompt_policy,
        "--success-threshold",
        str(args.success_threshold),
        "--min-trials-for-accept",
        str(args.min_trials_for_accept),
        "--obs-mode",
        args.obs_mode,
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--trial-timeout-s",
        str(args.trial_timeout_s),
        "--test-timeout-s",
        str(args.test_timeout_s),
        "--output-root",
        str(run_dir),
        "--run-name",
        "cegis",
        "--evidence-root",
        str(run_dir / "accepted_evidence"),
    ]
    if args.dry_run:
        command.append("--dry-run")
    cegis_run = _run_command(command, log_path=command_log, dry_run=False)
    summary_path = run_dir / "cegis" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    cycles = list(summary.get("cycles") or [])
    latest_cycle = cycles[-1] if cycles else {}
    development = latest_cycle.get("development_summary") or {}
    held_out = (summary.get("held_out") or {}).get("summary") or {}
    if cycles:
        cycle_dir = run_dir / "cegis" / f"cycle_{int(latest_cycle.get('cycle') or len(cycles)):02d}"
        for name in (
            "module_generation.jsonl",
            "module_generation.md",
            "development.jsonl",
            "development.md",
            "held_out.jsonl",
            "held_out.md",
        ):
            source = cycle_dir / name
            if source.exists():
                shutil.copy2(source, run_dir / name)
    target_adapter = REPO_ROOT / case.target_adapter_path
    if not args.dry_run and target_adapter.exists():
        shutil.copy2(target_adapter, run_dir / "final_adapter.py")

    generation_results = []
    for path in sorted((run_dir / "cegis").glob("cycle_*/module_generation.jsonl")):
        generation_results.append(_read_last_jsonl(path))
    integrity_rows = [_generated_candidate_integrity(item) for item in generation_results]
    integrity = {
        "used_llm": any(item["used_llm"] for item in integrity_rows),
        "num_cycles_with_generation": len(generation_results),
        "num_kept_llm_modules": sum(item["num_kept_llm_modules"] for item in integrity_rows),
        "valid_generated_candidate": any(item["valid_generated_candidate"] for item in integrity_rows),
        "models": sorted({model for item in integrity_rows for model in item["models"]}),
        "api_calls": sum(item.get("api_calls", 0) for item in integrity_rows),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in integrity_rows),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in integrity_rows),
        "total_tokens": sum(item.get("total_tokens", 0) for item in integrity_rows),
    }
    return {
        "adapter_module": case.target_adapter_module,
        "adapter_path": str(target_adapter),
        "adapter_sha256": _sha256(target_adapter),
        "adapter_provenance": (
            "dry_run"
            if args.dry_run
            else "llm_generated_cegis"
            if integrity["valid_generated_candidate"]
            else "generation_invalid"
        ),
        "cegis_run": cegis_run,
        "cegis_summary": summary,
        "generation_integrity": integrity,
        "development": development,
        "held_out": held_out,
    }


def _accepted(summary: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return (
        int(summary.get("num_trials") or 0) >= args.min_trials_for_accept
        and float(summary.get("success_rate") or 0.0) >= args.success_threshold
    )


def _runtime_metadata() -> Dict[str, Any]:
    provider = current_provider()
    packages = {}
    for package in ("mani-skill", "gymnasium", "sapien", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not_installed"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git("rev-parse", "HEAD") or "unknown",
        "git_status_short": _git("status", "--short"),
        "python": sys.version,
        "platform": platform.platform(),
        "llm_provider": provider,
        "llm_model": default_model(provider),
        "llm_max_tokens": completion_token_limit(),
        "llm_temperature": generation_temperature(),
        "deepseek_thinking": deepseek_thinking_mode() if provider == "deepseek" else "not_applicable",
        "packages": packages,
    }


def _elapsed_steps(result: Mapping[str, Any]) -> int:
    raw = (result.get("final_info") or {}).get("elapsed_steps")
    if isinstance(raw, list) and raw:
        return int(raw[0])
    if isinstance(raw, (int, float)):
        return int(raw)
    return 0


def _generation_trial_metrics(result: Mapping[str, Any]) -> tuple[int, int]:
    trials = []
    if result.get("source_result"):
        trials.append(result["source_result"])
    if result.get("initial_target_result"):
        trials.append(result["initial_target_result"])
    trials.extend(
        attempt["target_result"]
        for attempt in result.get("attempts") or []
        if attempt.get("target_result")
    )
    return len(trials), sum(_elapsed_steps(item) for item in trials)


def _command_wall_time(value: Any) -> float:
    """Sum measured subprocess time without counting scalar fields twice."""

    if isinstance(value, Mapping):
        own = (
            float(value.get("wall_time_seconds") or 0.0)
            if "command" in value
            else 0.0
        )
        return own + sum(
            _command_wall_time(item)
            for key, item in value.items()
            if key != "wall_time_seconds"
        )
    if isinstance(value, (list, tuple)):
        return sum(_command_wall_time(item) for item in value)
    return 0.0


def _paper_metrics(details: Mapping[str, Any], run_dir: Path) -> Dict[str, Any]:
    generation_results = []
    if details.get("generation_result"):
        generation_results.append(details["generation_result"])
    generation_results.extend(
        _read_last_jsonl(path)
        for path in sorted((run_dir / "cegis").glob("cycle_*/module_generation.jsonl"))
    )
    simulator_calls = 0
    environment_steps = 0
    for result in generation_results:
        calls, steps = _generation_trial_metrics(result)
        simulator_calls += calls
        environment_steps += steps

    multiseed_paths = (
        sorted((run_dir / "cegis").glob("cycle_*/*.jsonl"))
        if (run_dir / "cegis").exists()
        else [run_dir / "development.jsonl", run_dir / "held_out.jsonl"]
    )
    for path in multiseed_paths:
        if path.name == "module_generation.jsonl" or not path.exists():
            continue
        summary = _read_multiseed(path)
        rows = summary.get("rows") or []
        simulator_calls += len(rows)
        environment_steps += sum(_elapsed_steps(row) for row in rows)

    probe_cases = 0
    active_probe_batches = 0
    kernel_ucb_batches = 0
    probe_paths = list(run_dir.glob("fixed_probe/**/*.json")) + list(
        (run_dir / "cegis").glob("cycle_*/**/*.json")
    )
    for path in probe_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") == "structured_probe_result.v1":
            probe_cases += int(
                payload.get("num_new_cases")
                if payload.get("num_new_cases") is not None
                else payload.get("num_cases") or 0
            )
        if payload.get("schema") == "active_probe_plan.v1":
            active_probe_batches += 1
            if (payload.get("optimizer") or {}).get("name") == "kernel_ucb":
                kernel_ucb_batches += 1
    simulator_calls += probe_cases

    integrity = details.get("generation_integrity") or {}
    adapter_path = run_dir / "final_adapter.py"
    code_lines = 0
    branch_count = 0
    if adapter_path.exists():
        source = adapter_path.read_text(encoding="utf-8")
        code_lines = len(source.splitlines())
        try:
            branch_count = sum(
                isinstance(node, (ast.If, ast.IfExp))
                for node in ast.walk(ast.parse(source))
            )
        except SyntaxError:
            branch_count = 0
    cegis_summary = details.get("cegis_summary") or {}
    counterexample_history = list(cegis_summary.get("counterexample_history") or [])
    return {
        "simulator_calls": simulator_calls,
        "environment_steps": environment_steps,
        "wall_time_seconds": round(_command_wall_time(details), 6),
        "probe_cases": probe_cases,
        "active_probe_batches": active_probe_batches,
        "kernel_ucb_batches": kernel_ucb_batches,
        "cegis_cycles": len(cegis_summary.get("cycles") or []),
        "unique_counterexample_reasons": len(
            {str(item.get("failure_reason")) for item in counterexample_history if item.get("failure_reason")}
        ),
        "unique_counterexample_seeds": len(
            {int(item.get("seed")) for item in counterexample_history if item.get("seed") is not None}
        ),
        "llm_api_calls": int(integrity.get("api_calls") or 0),
        "prompt_tokens": int(integrity.get("prompt_tokens") or 0),
        "completion_tokens": int(integrity.get("completion_tokens") or 0),
        "total_tokens": int(integrity.get("total_tokens") or 0),
        "human_adapter_edits": 0,
        "adapter_code_lines": code_lines,
        "adapter_branch_count": branch_count,
    }


def _copy_if_exists(source: Path, target: Path) -> str:
    if not source.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target.relative_to(REPO_ROOT)) if target.is_relative_to(REPO_ROOT) else str(target)


def _archive_run(
    payload: Mapping[str, Any],
    *,
    run_dir: Path,
    evidence_root: Path,
) -> Dict[str, Any]:
    bundle = evidence_root / str(payload["run_id"])
    bundle.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    sources = {
        "final_adapter": run_dir / "final_adapter.py",
        "module_generation_jsonl": run_dir / "module_generation.jsonl",
        "module_generation_md": run_dir / "module_generation.md",
        "development_jsonl": run_dir / "development.jsonl",
        "development_md": run_dir / "development.md",
        "held_out_jsonl": run_dir / "held_out.jsonl",
        "held_out_md": run_dir / "held_out.md",
        "commands_log": run_dir / "commands.log",
    }
    for key, source in sources.items():
        suffix = source.suffix or ".txt"
        target = bundle / ("final_adapter.py" if key == "final_adapter" else f"{key}{suffix}")
        copied = _copy_if_exists(source, target)
        if copied:
            artifacts[key] = copied
    manifest = {
        "schema": "paper_run_evidence.v1",
        **{key: payload.get(key) for key in (
            "run_id",
            "tier",
            "method_id",
            "method_label",
            "case_id",
            "task_id",
            "source_robot",
            "target_robot",
            "repetition",
            "development_seeds",
            "held_out_seeds",
            "sim_backend",
            "render_backend",
            "success",
            "adapter_sha256",
            "adapter_provenance",
            "development_success_rate",
            "held_out_success_rate",
            "runtime",
            "generation_integrity",
            "frozen_interface_integrity",
            "metrics",
        )},
        "held_out_used_for_repair": False,
        "artifacts": artifacts,
    }
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
    return {"bundle_dir": str(bundle), "manifest": str(manifest_path), "artifacts": artifacts}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    args.method = normalize_method_id(args.method)
    method = PAPER_METHODS[args.method]
    tier = PAPER_TIERS[args.tier]
    if args.max_cycles != tier.max_cycles:
        raise ValueError(
            f"max_cycles must match frozen {args.tier} tier: "
            f"expected {tier.max_cycles}, got {args.max_cycles}."
        )
    expected_llm_budget = method_llm_call_budget(method, tier)
    if args.llm_call_budget < 0:
        args.llm_call_budget = expected_llm_budget
    if args.llm_call_budget != expected_llm_budget:
        raise ValueError(
            f"{args.method} LLM-call budget must match frozen {args.tier} tier: "
            f"expected {expected_llm_budget}, got {args.llm_call_budget}."
        )
    if args.method in {"B5", "Ours"} and args.probe_budget != tier.probe_budget:
        raise ValueError(
            f"{args.method} probe budget must match frozen {args.tier} tier: "
            f"expected {tier.probe_budget}, got {args.probe_budget}."
        )
    case = get_full_migration_case(args.case)
    run_dir = REPO_ROOT / args.output_root / args.plan_name / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_dir / "commands.log"
    target_adapter = REPO_ROOT / case.target_adapter_path
    lock_dir = REPO_ROOT / ".tmp" / "paper_adapter_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{case.case_id}.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
    original_adapter = target_adapter.read_bytes() if target_adapter.exists() else None
    frozen_before = _frozen_interface_hashes(case)

    details: Dict[str, Any]
    try:
        if method.runner_kind == "direct_generation":
            details = _run_direct_generation(case, args, run_dir, command_log)
        elif method.runner_kind == "cegis":
            details = _run_cegis(case, args, run_dir, command_log)
        else:
            adapter_module = (
                _oracle_module(case)
                if method.runner_kind == "evaluate_oracle"
                else _baseline_module(case, args.method)
            )
            adapter_path = _module_path(adapter_module)
            development, held_out, evaluation_runs = _evaluate_adapter(
                case,
                adapter_module,
                args,
                run_dir,
                command_log,
            )
            if adapter_path.exists() and not args.dry_run:
                shutil.copy2(adapter_path, run_dir / "final_adapter.py")
            details = {
                "adapter_module": adapter_module,
                "adapter_path": str(adapter_path),
                "adapter_sha256": _sha256(adapter_path),
                "adapter_provenance": (
                    "dry_run"
                    if args.dry_run
                    else "hand_written_oracle"
                    if method.runner_kind == "evaluate_oracle"
                    else "source_copy"
                    if method.runner_kind == "evaluate_source_copy"
                    else "shared_generic"
                ),
                "development": development,
                "held_out": held_out,
                "evaluation_runs": evaluation_runs,
                "generation_integrity": {"not_applicable": True},
            }

        development = details.get("development") or {}
        held_out = details.get("held_out") or {}
        integrity = details.get("generation_integrity") or {}
        integrity_ok = (
            bool(integrity.get("valid_generated_candidate"))
            if method.uses_llm
            else True
        )
        frozen_integrity = _frozen_integrity(frozen_before, _frozen_interface_hashes(case))
        success = (
            None
            if args.dry_run
            else _accepted(held_out, args) and integrity_ok and frozen_integrity["valid"]
        )
        payload: Dict[str, Any] = {
            "schema": "paper_method_run.v1",
            "run_id": args.run_id,
            "tier": args.tier,
            "method_id": method.method_id,
            "method_label": method.label,
            "prompt_policy": method.prompt_policy,
            "probe_strategy": method.probe_strategy,
            "case_id": case.case_id,
            "support_status": case.support_status,
            "support_source_url": case.support_source_url,
            "task_id": case.task_id,
            "source_robot": case.source_robot,
            "target_robot": case.target_robot,
            "repetition": args.repetition,
            "development_seeds": args.development_seeds,
            "held_out_seeds": args.held_out_seeds,
            "sim_backend": args.sim_backend,
            "render_backend": args.render_backend,
            "success": success,
            "development_success_rate": development.get("success_rate"),
            "held_out_success_rate": held_out.get("success_rate"),
            "held_out_used_for_repair": False,
            "budget": {
                "scope": "per_run_maximum",
                "llm_api_call_budget": args.llm_call_budget,
                "explicit_probe_case_budget": (
                    args.probe_budget if args.method in {"B5", "Ours"} else 0
                ),
                "posthoc_analysis_llm_enabled": False,
            },
            "adapter_sha256": details.get("adapter_sha256"),
            "adapter_provenance": details.get("adapter_provenance"),
            "generation_integrity": integrity,
            "frozen_interface_integrity": frozen_integrity,
            "metrics": _paper_metrics(details, run_dir),
            "runtime": _runtime_metadata(),
            "details": details,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
        if not args.dry_run:
            evidence_root = REPO_ROOT / args.evidence_root
            payload["evidence_bundle"] = _archive_run(payload, run_dir=run_dir, evidence_root=evidence_root)
            summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
        return payload
    finally:
        if not args.dry_run:
            if original_adapter is None:
                target_adapter.unlink(missing_ok=True)
            else:
                target_adapter.write_bytes(original_adapter)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(PAPER_METHODS))
    parser.add_argument("--case", required=True)
    parser.add_argument("--tier", default="pilot", choices=tuple(PAPER_TIERS))
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--development-seeds", default="0-4")
    parser.add_argument("--held-out-seeds", default="100-104")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--llm-call-budget", type=int, default=-1)
    parser.add_argument("--probe-budget", type=int, default=8)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--min-trials-for-accept", type=int, default=5)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--trial-timeout-s", type=int, default=900)
    parser.add_argument("--test-timeout-s", type=int, default=240)
    parser.add_argument("--output-root", default="results/paper")
    parser.add_argument("--evidence-root", default="evidence/paper_runs")
    parser.add_argument("--plan-name", default="paper_benchmark")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run(args)
    if args.dry_run:
        cegis = ((payload.get("details") or {}).get("cegis_summary") or {})
        preview = cegis.get("nested_command_preview") or {}
        display = {
            "run_id": payload.get("run_id"),
            "method_id": payload.get("method_id"),
            "case_id": payload.get("case_id"),
            "dry_run": True,
            "prompt_policy": payload.get("prompt_policy"),
            "probe_strategy": payload.get("probe_strategy"),
            "nested_repair_preview": bool(preview),
            "summary": str(
                REPO_ROOT
                / args.output_root
                / args.plan_name
                / "runs"
                / args.run_id
                / "summary.json"
            ),
        }
        print(json.dumps(display, ensure_ascii=False, default=repr))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))


if __name__ == "__main__":
    main()
