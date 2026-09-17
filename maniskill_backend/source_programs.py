"""Validate and freeze one reliable source program per benchmark task."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .code_validation import extract_python_module
from .dynamic_harness import (
    DynamicMigrationSpec,
    _dynamic_system_prompt,
    _source_prompt,
    discover_environment,
    extract_task_program,
    read_source_actions,
    run_dynamic_trial,
    validate_dynamic_adapter,
)
from .experiment_environment import capture_experiment_environment, load_experiment_contract
from .llm import gen_text
from .task_catalog import load_task_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source_programs"
SOURCE_CATALOG_PATH = SOURCE_ROOT / "programs.json"


@dataclass(frozen=True)
class SourceProgramSpec:
    task_id: str
    env_id: str
    source_robot: str
    control_mode: str
    candidate_path: Path | None
    frozen_path: Path
    origin: str
    episode_limit: int


def load_source_program_catalog(
    source_root: Path = SOURCE_ROOT,
    *,
    task_root: Path | None = None,
) -> tuple[dict[str, Any], list[SourceProgramSpec]]:
    payload = _read_json(source_root / "programs.json")
    tasks_payload = load_task_catalog(task_root) if task_root is not None else load_task_catalog()
    tasks = {task["task_id"]: task for task in tasks_payload["tasks"]}
    validate_source_program_catalog(payload, tasks)
    frozen_policy = load_experiment_contract()["source_program_validation"]
    policy = payload["freeze_policy"]
    if int(policy["minimum_seed_count"]) != int(frozen_policy["minimum_seed_count"]):
        raise ValueError("Source minimum seed count differs from experiment_config.json")
    if float(policy["required_success_rate"]) != float(
        frozen_policy["required_success_rate"]
    ):
        raise ValueError("Source success rate differs from experiment_config.json")
    if int(policy["max_episode_steps"]) != int(frozen_policy["max_episode_steps"]):
        raise ValueError("Source episode budget differs from experiment_config.json")

    specs = []
    for entry in payload["programs"]:
        task = tasks[entry["task_id"]]
        candidate = entry.get("candidate")
        specs.append(
            SourceProgramSpec(
                task_id=entry["task_id"],
                env_id=task["env_id"],
                source_robot=task["source_robot"],
                control_mode=entry["control_mode"],
                candidate_path=(source_root / candidate if candidate else None),
                frozen_path=source_root / "frozen" / f"{entry['task_id']}.py",
                origin=entry["origin"],
                episode_limit=int(payload["freeze_policy"]["max_episode_steps"]),
            )
        )
    return payload, specs


def validate_source_program_catalog(
    payload: Mapping[str, Any], tasks: Mapping[str, Mapping[str, Any]]
) -> None:
    if payload.get("schema") != "source_program_catalog.v1":
        raise ValueError("Source programs must use schema source_program_catalog.v1")
    entries = payload.get("programs")
    if not isinstance(entries, list) or len(entries) != 20:
        raise ValueError("Source program catalog must contain exactly 20 entries")
    ids = [entry.get("task_id") for entry in entries]
    if len(set(ids)) != 20 or set(ids) != set(tasks):
        raise ValueError("Source program task IDs must match the 20-task catalog exactly")
    policy = payload.get("freeze_policy") or {}
    if int(policy.get("minimum_seed_count", 0)) < 2:
        raise ValueError("Freezing must require at least two seeds")
    if float(policy.get("required_success_rate", 0.0)) != 1.0:
        raise ValueError("Freezing must require a 100% validation success rate")
    if int(policy.get("max_episode_steps", 0)) <= 0:
        raise ValueError("Source validation max_episode_steps must be positive")

    for entry in entries:
        if not entry.get("control_mode") or not entry.get("origin"):
            raise ValueError(f"{entry.get('task_id')} lacks source-program metadata")
        candidate = entry.get("candidate")
        if candidate is not None and (
            not isinstance(candidate, str) or not candidate.startswith("candidates/")
        ):
            raise ValueError(f"{entry['task_id']} has an invalid candidate path")
        source_robot = str(tasks[entry["task_id"]]["source_robot"])
        if entry["task_id"] in {"t12_open_cabinet_drawer", "t13_open_cabinet_door"}:
            if source_robot != "fetch":
                raise ValueError(f"{entry['task_id']} must use Fetch as its source")
        elif not source_robot.startswith("panda"):
            raise ValueError(f"{entry['task_id']} must use a Panda-family source")


def select_source_programs(
    specs: Iterable[SourceProgramSpec], selection: str
) -> list[SourceProgramSpec]:
    by_id = {spec.task_id: spec for spec in specs}
    requested = list(by_id) if selection.strip().lower() == "all" else [
        item.strip() for item in selection.split(",") if item.strip()
    ]
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown source task IDs: {', '.join(unknown)}")
    return [by_id[task_id] for task_id in requested]


def statically_check_candidate(spec: SourceProgramSpec) -> dict[str, Any]:
    if spec.candidate_path is None:
        return {
            "ok": False,
            "status": "candidate_missing",
            "message": "No source candidate yet; synthesize or author it before validation.",
        }
    if not spec.candidate_path.is_file():
        return {
            "ok": False,
            "status": "candidate_file_missing",
            "message": str(spec.candidate_path),
        }
    try:
        source = read_source_actions(spec.candidate_path)
        mismatches = []
        if source.env_id != spec.env_id:
            mismatches.append(f"ENV_ID={source.env_id!r}, expected {spec.env_id!r}")
        if source.robot_uid != spec.source_robot:
            mismatches.append(
                f"SOURCE_ROBOT={source.robot_uid!r}, expected {spec.source_robot!r}"
            )
        if source.control_mode != spec.control_mode:
            mismatches.append(
                f"CONTROL_MODE={source.control_mode!r}, expected {spec.control_mode!r}"
            )
        if mismatches:
            raise ValueError("; ".join(mismatches))
    except Exception as exc:
        return {"ok": False, "status": "candidate_invalid", "message": repr(exc)}
    return {
        "ok": True,
        "status": "candidate_ready",
        "entrypoint": source.entrypoint,
        "sha256": _sha256_file(spec.candidate_path),
    }


def validate_source_program(
    spec: SourceProgramSpec,
    *,
    seeds: list[int],
    output_dir: Path,
    obs_mode: str,
    sim_backend: str,
    render_backend: str,
    enforce_environment: bool,
) -> dict[str, Any]:
    if len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Source validation requires at least two unique seeds")
    static = statically_check_candidate(spec)
    summary: dict[str, Any] = {
        "schema": "source_program_validation.v1",
        "task": asdict(spec),
        "candidate": static,
        "seeds": seeds,
        "trials": [],
        "success": False,
        "frozen": False,
    }
    summary["task"]["candidate_path"] = (
        str(spec.candidate_path) if spec.candidate_path is not None else None
    )
    summary["task"]["frozen_path"] = str(spec.frozen_path)
    if not static["ok"]:
        _write_json(output_dir / spec.task_id / "summary.json", summary)
        return summary

    source = read_source_actions(spec.candidate_path)
    runtime = capture_experiment_environment(
        {
            "phase": "source_program_validation",
            "task_id": spec.task_id,
            "source_robot": spec.source_robot,
            "target_robot": spec.source_robot,
            "source_control_mode": spec.control_mode,
            "target_control_mode": spec.control_mode,
            "seed": seeds[0],
        }
    )
    summary["runtime_environment"] = str(output_dir / spec.task_id / "runtime_environment.json")
    summary["environment_contract_sha256"] = runtime["contract_sha256"]
    _write_json(output_dir / spec.task_id / "runtime_environment.json", runtime)
    if enforce_environment and not runtime["validation"]["ok"]:
        summary["status"] = "environment_contract_mismatch"
        summary["environment_mismatches"] = runtime["validation"]["mismatches"]
        _write_json(output_dir / spec.task_id / "summary.json", summary)
        return summary

    for seed in seeds:
        trial_spec = DynamicMigrationSpec(
            env_id=spec.env_id,
            task_label=spec.task_id,
            source_robot=spec.source_robot,
            target_robot=spec.source_robot,
            source_control_mode=spec.control_mode,
            target_control_mode=spec.control_mode,
            seed=seed,
            max_episode_steps=spec.episode_limit,
            max_source_cycles=0,
            max_target_cycles=0,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
        )
        trial = run_dynamic_trial(
            spec=trial_spec,
            robot_uid=spec.source_robot,
            control_mode=spec.control_mode,
            program=source.program,
            adapter_path=spec.candidate_path,
            source_entrypoint=source.entrypoint,
        )
        compact = {
            "seed": seed,
            "success": bool(trial.get("success")),
            "message": trial.get("message"),
            "action_steps": trial.get("action_steps", 0),
            "official_success": trial.get("official_success", False),
            "terminated": trial.get("terminated", False),
            "truncated": trial.get("truncated", False),
            "failure_diagnosis": trial.get("failure_diagnosis"),
        }
        summary["trials"].append(compact)
        _write_json(output_dir / spec.task_id / f"seed_{seed:03d}.json", trial)

    successes = sum(int(row["success"]) for row in summary["trials"])
    summary.update(
        status="validated" if successes == len(seeds) else "validation_failed",
        num_success=successes,
        num_trials=len(seeds),
        success_rate=successes / len(seeds),
        success=successes == len(seeds),
    )
    _write_json(output_dir / spec.task_id / "summary.json", summary)
    return summary


def synthesize_source_program(
    spec: SourceProgramSpec,
    *,
    seeds: list[int],
    output_dir: Path,
    max_cycles: int,
    obs_mode: str,
    sim_backend: str,
    render_backend: str,
    initial_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and repair one source program against real multi-seed trials."""

    if spec.candidate_path is None:
        raise ValueError(f"{spec.task_id} has no candidate destination")
    if max_cycles < 1:
        raise ValueError("max_cycles must be positive")
    runtime = capture_experiment_environment(
        {
            "phase": "source_program_synthesis",
            "task_id": spec.task_id,
            "source_robot": spec.source_robot,
            "target_robot": spec.source_robot,
            "source_control_mode": spec.control_mode,
            "target_control_mode": spec.control_mode,
            "seed": seeds[0],
        }
    )
    _write_json(output_dir / spec.task_id / "runtime_environment.json", runtime)
    if not runtime["validation"]["ok"]:
        result = {
            "schema": "source_program_synthesis.v1",
            "task_id": spec.task_id,
            "success": False,
            "status": "environment_contract_mismatch",
            "environment_mismatches": runtime["validation"]["mismatches"],
            "cycles": [],
            "candidate_path": str(spec.candidate_path),
        }
        _write_json(output_dir / spec.task_id / "synthesis_summary.json", result)
        return result
    base_spec = DynamicMigrationSpec(
        env_id=spec.env_id,
        task_label=spec.task_id,
        source_robot=spec.source_robot,
        target_robot=spec.source_robot,
        source_control_mode=spec.control_mode,
        target_control_mode=spec.control_mode,
        seed=seeds[0],
        max_episode_steps=spec.episode_limit,
        max_source_cycles=max_cycles,
        max_target_cycles=0,
        obs_mode=obs_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
    )
    observation = discover_environment(base_spec, spec.source_robot, spec.control_mode)
    current_code = (
        spec.candidate_path.read_text(encoding="utf-8")
        if spec.candidate_path.is_file()
        else ""
    )
    latest_result: dict[str, Any] = (
        _multi_seed_feedback(initial_validation) if initial_validation else {}
    )
    cycles = []
    for cycle in range(1, max_cycles + 1):
        cycle_dir = output_dir / spec.task_id / f"synthesis_cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        system = _dynamic_system_prompt(source=True)
        prompt = _source_prompt(base_spec, observation, current_code, latest_result)
        prompt += (
            "\nMulti-seed requirement:\n"
            f"- Define ENV_ID = {spec.env_id!r}, SOURCE_ROBOT = {spec.source_robot!r}, "
            f"and CONTROL_MODE = {spec.control_mode!r} as module string constants.\n"
            f"- The same module must pass seeds {seeds}.\n"
            "- Do not hard-code an initial pose from one seed. Re-read object, goal, TCP, "
            "joint and official evaluation state while acting.\n"
            "- Repair the earliest shared failure across the reported seeds.\n"
        )
        (cycle_dir / "system_prompt.txt").write_text(system, encoding="utf-8")
        (cycle_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
        generated = gen_text(system=system, prompt=prompt, fallback_text="")
        (cycle_dir / "llm_response.txt").write_text(
            generated.raw_text or generated.text, encoding="utf-8"
        )
        record: dict[str, Any] = {
            "cycle": cycle,
            "model": generated.model,
            "used_llm": generated.used_llm,
            "usage": generated.usage,
            "reason": generated.reason,
        }
        if not generated.used_llm:
            record["error"] = generated.reason or "LLM unavailable"
            cycles.append(record)
            break
        try:
            candidate = extract_python_module(generated.text).rstrip() + "\n"
            candidate_path = cycle_dir / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            # Preserve even a statically invalid generation so the next cycle repairs
            # the concrete failed module instead of restarting from the bootstrap code.
            current_code = candidate
            record["candidate_sha256"] = _sha256_file(candidate_path)
            program = extract_task_program(candidate)
            validate_dynamic_adapter(
                candidate, task_program=program, require_program_constant=True
            )
            trial_spec = replace(spec, candidate_path=candidate_path)
            validation = validate_source_program(
                trial_spec,
                seeds=seeds,
                output_dir=cycle_dir / "validation",
                obs_mode=obs_mode,
                sim_backend=sim_backend,
                render_backend=render_backend,
                enforce_environment=True,
            )
            record.update(
                valid=True,
                candidate_sha256=_sha256_file(candidate_path),
                success=validation.get("success", False),
                success_rate=validation.get("success_rate", 0.0),
                validation=str(cycle_dir / "validation" / spec.task_id / "summary.json"),
            )
            current_code = candidate
            latest_result = _multi_seed_feedback(validation)
            if validation.get("success"):
                spec.candidate_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, spec.candidate_path)
                record["installed_candidate"] = str(spec.candidate_path)
                cycles.append(record)
                result = {
                    "schema": "source_program_synthesis.v1",
                    "task_id": spec.task_id,
                    "success": True,
                    "cycles": cycles,
                    "validation": validation,
                    "candidate_path": str(spec.candidate_path),
                }
                _write_json(output_dir / spec.task_id / "synthesis_summary.json", result)
                return result
        except Exception as exc:
            record.update(valid=False, success=False, error=repr(exc))
            latest_result = {
                "success": False,
                "message": repr(exc),
                "action_steps": 0,
                "failure_diagnosis": {
                    "layer": "program",
                    "reason": "generated_source_module_invalid",
                    "repair_hint": (
                        "Repair the concrete module shown in the prompt before changing motion "
                        "parameters. For state-feedback validation, call a documented helper such "
                        "as self._snapshot(), self._tcp_pos(), self._entity_pos(name), "
                        "self._region_pos(name), or self._official_evaluation(), then use the "
                        "measured value to decide an action or runtime branch."
                    ),
                },
            }
        cycles.append(record)
        _write_json(cycle_dir / "cycle.json", record)

    result = {
        "schema": "source_program_synthesis.v1",
        "task_id": spec.task_id,
        "success": False,
        "cycles": cycles,
        "candidate_path": str(spec.candidate_path),
    }
    _write_json(output_dir / spec.task_id / "synthesis_summary.json", result)
    return result


def freeze_source_program(
    spec: SourceProgramSpec,
    validation: Mapping[str, Any],
    *,
    minimum_seed_count: int,
) -> dict[str, Any]:
    trials = validation.get("trials") or []
    if not validation.get("success") or len(trials) < minimum_seed_count:
        raise ValueError(
            f"{spec.task_id} cannot be frozen before all required seeds succeed"
        )
    if spec.candidate_path is None:
        raise ValueError(f"{spec.task_id} has no candidate to freeze")

    spec.frozen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.candidate_path, spec.frozen_path)
    record = {
        "schema": "frozen_source_program.v1",
        "task_id": spec.task_id,
        "env_id": spec.env_id,
        "source_robot": spec.source_robot,
        "control_mode": spec.control_mode,
        "origin": spec.origin,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_path": _display_path(spec.candidate_path),
        "frozen_path": _display_path(spec.frozen_path),
        "source_sha256": _sha256_file(spec.frozen_path),
        "support_code_sha256": _support_code_sha256(spec),
        "validated_seeds": [int(row["seed"]) for row in trials],
        "success_rate": 1.0,
        "max_episode_steps": spec.episode_limit,
        "success_authority": "env.unwrapped.evaluate()['success']",
        "environment_contract_sha256": validation.get("environment_contract_sha256"),
    }
    _write_json(spec.frozen_path.with_suffix(".json"), record)
    return record


def source_program_status(specs: Iterable[SourceProgramSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        static = statically_check_candidate(spec)
        record_path = spec.frozen_path.with_suffix(".json")
        frozen_valid = False
        frozen_record = None
        if spec.frozen_path.is_file() and record_path.is_file():
            frozen_record = _read_json(record_path)
            frozen_valid = (
                frozen_record.get("source_sha256") == _sha256_file(spec.frozen_path)
                and frozen_record.get("support_code_sha256")
                == _support_code_sha256(spec)
            )
        rows.append(
            {
                "task_id": spec.task_id,
                "env_id": spec.env_id,
                "source_robot": spec.source_robot,
                "candidate_status": static["status"],
                "frozen": frozen_valid,
                "validated_seeds": (
                    frozen_record.get("validated_seeds", []) if frozen_valid else []
                ),
            }
        )
    return rows


def _multi_seed_feedback(validation: Mapping[str, Any]) -> dict[str, Any]:
    trials = validation.get("trials") or []
    failed = [row for row in trials if not row.get("success")]
    return {
        "success": not failed,
        "message": (
            "all validation seeds succeeded"
            if not failed
            else f"{len(failed)}/{len(trials)} validation seeds failed"
        ),
        "official_success": not failed,
        "action_steps": sum(int(row.get("action_steps") or 0) for row in trials),
        "multi_seed_failures": failed,
        "failure_diagnosis": (
            failed[0].get("failure_diagnosis")
            if failed
            else {"reason": "none"}
        ),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(SOURCE_ROOT))
    except ValueError:
        return str(path)


def _support_code_sha256(spec: SourceProgramSpec) -> str:
    paths = (
        [SOURCE_ROOT / "candidate_runtime.py"]
        if spec.origin.startswith("bootstrap_")
        else sorted((SOURCE_ROOT / "vendor" / "mani_skill_motionplanning").rglob("*.py"))
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(SOURCE_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
