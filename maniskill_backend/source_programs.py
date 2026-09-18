"""Validate and freeze one reliable source program per benchmark task."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .code_validation import extract_python_module, validate_generated_adapter_module
from .dynamic_harness import (
    DynamicMigrationSpec,
    _dynamic_system_prompt,
    _prompt_observation,
    _prompt_trial,
    _semantic_code_sha256,
    _source_prompt,
    discover_environment,
    extract_task_program,
    read_source_actions,
    run_dynamic_trial,
    validate_dynamic_adapter,
)
from .experiment_environment import capture_experiment_environment, load_experiment_contract
from .llm import LLMTextResult, gen_text
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
    summary["provenance"] = {"kind": "existing_candidate", "catalog_origin": spec.origin}
    provenance_path = spec.candidate_path.with_suffix(".provenance.json")
    if provenance_path.exists():
        provenance = _read_json(provenance_path)
        if provenance.get("candidate_sha256") == static["sha256"]:
            summary["provenance"] = provenance["provenance"]

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
    summary["environment_valid"] = runtime["validation"]["ok"]
    summary["support_code_sha256"] = _support_code_sha256(spec)
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
            "trial_path": str((output_dir / spec.task_id / f"seed_{seed:03d}.json").resolve()),
        }
        summary["trials"].append(compact)
        _write_json(output_dir / spec.task_id / f"seed_{seed:03d}.json", trial)
        summary["status"] = "validating"
        _write_json(output_dir / spec.task_id / "summary.json", summary)
        if _infrastructure_failure(trial):
            summary.update(status="infrastructure_failure", message=trial.get("message"))
            _write_json(output_dir / spec.task_id / "summary.json", summary)
            return summary

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
    retry_unavailable: bool = False,
) -> dict[str, Any]:
    """Repair sources in their native controller contract, with durable cycle records."""

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
    task_dir = output_dir / spec.task_id
    previous_summary = task_dir / "synthesis_summary.json"
    if previous_summary.exists():
        previous = _read_json(previous_summary)
        evidence = previous.get("validation", {})
        if (previous.get("success") and spec.candidate_path.is_file()
                and evidence.get("candidate", {}).get("sha256") == _sha256_file(spec.candidate_path)
                and evidence.get("environment_contract_sha256") == runtime["contract_sha256"]
                and evidence.get("support_code_sha256") == _support_code_sha256(spec)
                and evidence.get("seeds") == seeds):
            return previous
    context = {
        "task_id": spec.task_id,
        "seeds": seeds,
        "control_mode": spec.control_mode,
        "contract_sha256": runtime["contract_sha256"],
        "support_code_sha256": _support_code_sha256(spec),
        "candidate_sha256": _sha256_file(spec.candidate_path) if spec.candidate_path.is_file() else None,
        "obs_mode": obs_mode, "sim_backend": sim_backend, "render_backend": render_backend,
    }
    context_path = task_dir / "context.json"
    if context_path.exists() and _read_json(context_path) != context:
        raise ValueError("Source repair inputs changed; start a new --run-name.")
    _write_json(context_path, context)
    observation = discover_environment(base_spec, spec.source_robot, spec.control_mode)
    _write_json(task_dir / "observation.json", observation)
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
        record_path = cycle_dir / "cycle.json"
        if record_path.exists():
            saved = _read_json(record_path)
            if retry_unavailable and saved.get("blocked") == "llm_unavailable":
                suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                cycle_dir.rename(cycle_dir.with_name(cycle_dir.name + "_unavailable_" + suffix))
                cycle_dir.mkdir()
            else:
                cycles.append(saved)
                candidate_path = cycle_dir / "candidate.py"
                if candidate_path.exists():
                    current_code = candidate_path.read_text(encoding="utf-8")
                latest_result = saved.get("feedback", latest_result)
                if saved.get("blocked"):
                    break
                continue
        system, prompt = _repair_prompt(spec, base_spec, observation, current_code, latest_result)
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
        generation_path = cycle_dir / "generation.json"
        if generation_path.exists():
            generated = LLMTextResult(**_read_json(generation_path))
        else:
            generated = gen_text(system=system, prompt=prompt, fallback_text="")
            _write_json(generation_path, asdict(generated))
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
            record["blocked"] = "llm_unavailable"
            cycles.append(record)
            _write_json(record_path, record)
            break
        try:
            candidate = extract_python_module(generated.text).rstrip() + "\n"
            candidate_path = cycle_dir / "candidate.py"
            candidate_path.write_text(candidate, encoding="utf-8")
            # Preserve even a statically invalid generation so the next cycle repairs
            # the concrete failed module instead of restarting from the bootstrap code.
            previous_code = current_code
            current_code = candidate
            record["candidate_sha256"] = _sha256_file(candidate_path)
            _validate_source_repair(candidate, native_run=spec.control_mode == "pd_joint_pos")
            try:
                previous_hash = _semantic_code_sha256(previous_code) if previous_code else None
            except SyntaxError:
                previous_hash = None
            if previous_hash == _semantic_code_sha256(candidate):
                raise ValueError("Generated source is semantically unchanged. Change the failed decision or action sequence, not comments.")
            trial_spec = replace(spec, candidate_path=candidate_path)
            smoke_seeds = seeds[:2]
            validation = validate_source_program(
                trial_spec,
                seeds=smoke_seeds,
                output_dir=cycle_dir / "smoke",
                obs_mode=obs_mode,
                sim_backend=sim_backend,
                render_backend=render_backend,
                enforce_environment=True,
            )
            validation_path = cycle_dir / "smoke" / spec.task_id / "summary.json"
            if validation.get("success") and len(seeds) > len(smoke_seeds):
                validation = validate_source_program(
                    trial_spec, seeds=seeds, output_dir=cycle_dir / "validation",
                    obs_mode=obs_mode, sim_backend=sim_backend,
                    render_backend=render_backend, enforce_environment=True,
                )
                validation_path = cycle_dir / "validation" / spec.task_id / "summary.json"
            record.update(
                valid=True,
                candidate_sha256=_sha256_file(candidate_path),
                success=validation.get("success", False),
                success_rate=validation.get("success_rate", 0.0),
                validation=str(validation_path),
                evaluated_seeds=validation.get("seeds", []),
                num_success=validation.get("num_success", 0),
            )
            current_code = candidate
            latest_result = _multi_seed_feedback(validation)
            if validation.get("status") in {"environment_contract_mismatch", "infrastructure_failure"}:
                record["blocked"] = validation["status"]
            if validation.get("success"):
                spec.candidate_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_path, spec.candidate_path)
                record["installed_candidate"] = str(spec.candidate_path)
                cycles.append(record)
                _write_json(record_path, record)
                validation["provenance"] = {
                    "kind": "llm_repair", "base_origin": spec.origin,
                    "model": generated.model, "cycle": cycle,
                    "generation_path": str(generation_path),
                }
                _write_json(spec.candidate_path.with_suffix(".provenance.json"), {
                    "candidate_sha256": record["candidate_sha256"],
                    "provenance": validation["provenance"],
                })
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
                "multi_seed_failures": latest_result.get("multi_seed_failures", []),
                "failure_diagnosis": {
                    "layer": "program",
                    "reason": "generated_source_module_invalid",
                    "repair_hint": (
                        "Repair the reported exception while preserving the documented "
                        "entrypoint and control mode. Use the state API in the module contract "
                        "and feed the measured values into bounded action decisions."
                    ),
                },
            }
        cycles.append(record)
        record["feedback"] = latest_result
        _write_json(record_path, record)
        _write_json(task_dir / "synthesis_summary.json", {
            "schema": "source_program_synthesis.v1", "task_id": spec.task_id,
            "success": False, "status": "repairing", "cycles": cycles,
        })
        if record.get("blocked"):
            break

    result = {
        "schema": "source_program_synthesis.v1",
        "task_id": spec.task_id,
        "success": False,
        "status": cycles[-1].get("blocked", "repair_budget_exhausted") if cycles else "not_started",
        "cycles": cycles,
        "candidate_path": str(spec.candidate_path),
    }
    _write_json(output_dir / spec.task_id / "synthesis_summary.json", result)
    return result


def _repair_prompt(spec, base_spec, observation, current_code, latest_result):
    if spec.control_mode != "pd_joint_pos":
        prompt = _source_prompt(base_spec, observation, current_code, latest_result)
        prompt += "\nRead-only task definition (do not copy simulator mutations):\n" + str(
            observation.get("environment_source", "")
        )[:18000]
        if "BootstrapSourceRobot" in current_code:
            prompt += "\nInherited bootstrap implementation (read-only context):\n" + (
                SOURCE_ROOT / "candidate_runtime.py"
            ).read_text(encoding="utf-8")
        return _dynamic_system_prompt(source=True), prompt
    system = (
        "Repair a complete ManiSkill source motion-planning program. Return only Python. "
        "Keep run(env, seed=None, debug=False, vis=False), pd_joint_pos, and the provided "
        "PandaArmMotionPlanningSolver API. Never mutate simulator state or task success."
    )
    prompt = f"""Task: {spec.task_id}. Environment: {spec.env_id}. Robot: {spec.source_robot}.
Observed environment:
{json.dumps(_prompt_observation(observation), indent=2, default=str)}
Required source contract:
- Keep the run(env, seed=None, debug=False, vis=False) entrypoint, not build_robot.
- Construct the planner with the supplied wrapped env BEFORE obtaining env.unwrapped.
  Physical execution must go through that planner or the original wrapped env.step.
- Read poses/joints/evaluate from env.unwrapped. Tensor positions may be (1,3);
  flatten explicitly and do not index a batch dimension as xyz.
- pd_joint_pos actions are joint positions, NOT normalized xyz commands.
- Do not call reset, set_pose, set_state, set_qpos, or edit any environment fields.
  The harness resets each trial and checks official success independently.
- Use bounded waypoints, inspect planning failures, and re-read state between moves.
Current failed source:
```python
{current_code}
```
Measured validation feedback:
{json.dumps(_prompt_trial(latest_result), indent=2, default=str)}
"""
    planner_root = SOURCE_ROOT / "vendor" / "mani_skill_motionplanning"
    for part in ("panda", "base_motionplanner", "two_finger_gripper"):
        path = planner_root / part / "motionplanner.py"
        prompt += f"\nRead-only planner API ({part}):\n" + path.read_text(encoding="utf-8")
    return system, prompt


def _validate_source_repair(code: str, *, native_run: bool) -> None:
    if not native_run:
        validate_dynamic_adapter(code, task_program=extract_task_program(code), require_program_constant=True)
        return
    validate_generated_adapter_module(
        code, entrypoint="run",
        extra_imports=("source_programs.vendor.mani_skill_motionplanning",),
    )
    tree = ast.parse(code)
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    if any(name.startswith("set_") or name in {"reset", "exec", "eval"} for name in calls):
        raise ValueError("Source repair may not reset or directly mutate the simulator.")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if any(isinstance(item, ast.Attribute) and item.attr in {
                    "pose", "success", "qpos", "qvel", "controller", "agent", "robot", "scene", "unwrapped",
                } for item in ast.walk(target)):
                    raise ValueError("Source repair may not assign environment or controller fields.")
    if not calls.intersection({"step", "move_to_pose_with_screw", "move_to_pose_with_RRTConnect", "move_to_pose_with_RRTStar"}):
        raise ValueError("Source repair must execute physical actions through the wrapped env/planner.")
    if not any(isinstance(node, ast.Attribute) and node.attr in {"pose", "tcp", "qpos", "evaluate"}
               for node in ast.walk(tree)):
        raise ValueError("Source repair must read physical state before planning.")


def freeze_source_program(
    spec: SourceProgramSpec,
    validation: Mapping[str, Any],
    *,
    minimum_seed_count: int,
) -> dict[str, Any]:
    trials = validation.get("trials") or []
    required = load_experiment_contract()["source_program_validation"]["default_seeds"]
    actual_seeds = [row.get("seed") for row in trials]
    if (
        not validation.get("success") or len(trials) < minimum_seed_count
        or sorted(actual_seeds) != sorted(required)
        or any(not row.get("success") or not row.get("official_success")
               or int(row.get("action_steps", 0)) <= 0 for row in trials)
    ):
        raise ValueError(
            f"{spec.task_id} cannot be frozen before all required seeds succeed"
        )
    if spec.candidate_path is None:
        raise ValueError(f"{spec.task_id} has no candidate to freeze")
    if (
        not validation.get("environment_valid")
        or validation.get("environment_contract_sha256") != _sha256_file(REPO_ROOT / "experiment_config.json")
        or validation.get("candidate", {}).get("sha256") != _sha256_file(spec.candidate_path)
        or validation.get("support_code_sha256") != _support_code_sha256(spec)
        or validation.get("task", {}).get("task_id") != spec.task_id
    ):
        raise ValueError("Cannot freeze: validation evidence does not match the current code/environment.")

    spec.frozen_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.candidate_path, spec.frozen_path)
    record = {
        "schema": "frozen_source_program.v2",
        "task_id": spec.task_id,
        "env_id": spec.env_id,
        "source_robot": spec.source_robot,
        "control_mode": spec.control_mode,
        "origin": spec.origin,
        "provenance": validation.get("provenance", {"kind": "existing_candidate", "catalog_origin": spec.origin}),
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
        "trials": trials,
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
            try:
                frozen_record = _read_json(record_path)
            except (OSError, ValueError):
                frozen_record = {}
            evidence = frozen_record.get("trials", [])
            frozen_valid = (
                frozen_record.get("schema") == "frozen_source_program.v2"
                and frozen_record.get("source_sha256") == _sha256_file(spec.frozen_path)
                and frozen_record.get("support_code_sha256")
                == _support_code_sha256(spec)
                and frozen_record.get("environment_contract_sha256")
                == _sha256_file(REPO_ROOT / "experiment_config.json")
                and sorted(frozen_record.get("validated_seeds", []))
                == sorted(load_experiment_contract()["source_program_validation"]["default_seeds"])
                and [row.get("seed") for row in evidence] == frozen_record.get("validated_seeds")
                and all(row.get("success") and row.get("official_success")
                        and int(row.get("action_steps", 0)) > 0 for row in evidence)
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
    enriched = []
    for index, row in enumerate(failed):
        full = dict(row)
        path = row.get("trial_path")
        if index < 3 and path and Path(path).is_file():
            full.update(_read_json(Path(path)))
            full["seed"] = row["seed"]
            filtered = _prompt_trial(full)
            # Keep the canonical input key so repeated prompt construction remains lossless.
            filtered["state_trace"] = filtered.pop("state_trace_sample", [])
            full = filtered
        enriched.append(full)
    passed = bool(validation.get("success")) and bool(trials) and not failed
    return {
        "success": passed,
        "message": (
            "all validation seeds succeeded"
            if passed
            else f"{len(failed)}/{len(trials)} validation seeds failed; status={validation.get('status')}"
        ),
        "official_success": passed,
        "action_steps": sum(int(row.get("action_steps") or 0) for row in trials),
        "multi_seed_failures": enriched,
        "failure_diagnosis": (
            failed[0].get("failure_diagnosis")
            if failed
            else {"reason": "none"}
        ),
    }


def _infrastructure_failure(trial: Mapping[str, Any]) -> bool:
    if int(trial.get("action_steps") or 0) > 0:
        return False
    text = str(trial.get("message", "")) + str(trial.get("execution_error", ""))
    return any(token in text for token in (
        "URLError", "EOFError", "No module named", "Vulkan", "CUDA error",
        "VK_ERROR", "requires asset", "could not be found", "FileNotFoundError",
    ))


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
    for name in ("dynamic_adapter.py", "dynamic_harness.py", "env_adapter.py"):
        digest.update(name.encode("utf-8"))
        digest.update((REPO_ROOT / "maniskill_backend" / name).read_bytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
