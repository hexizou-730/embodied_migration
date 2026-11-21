"""Paired ablation isolating repair granularity from one shared failed candidate.

Both arms start from the same replayed failed adapter, receive the same measured
state feedback and repair localization, and run under the same budget. The only
treatment difference is the allowed repair output:

- ``method_patch``   : the LLM returns one diagnosed ``_stage_*`` method and the
                       harness splices it in, AST-locking every other method
                       (the existing ``selective.py --resume-adapter`` path).
- ``module_rewrite``: the LLM returns a complete module under the same stage
                       contract; any stage method may change, and a module
                       semantically identical to a tested candidate is rejected.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .code_validation import extract_python_module
from .dynamic_harness import (
    DynamicMigrationSpec,
    discover_environment,
    run_dynamic_trial,
)
from .llm import gen_text
from .selective_comparison import _semantic_hash
from .selective_migration import (
    FROZEN_PROGRAM,
    apply_stage_semantics,
    completed_stages,
    failed_stage,
    generation_settings,
    locate_repair_stage,
    run_selective_migration,
    stage_boundary_states,
    validate_selective_adapter,
)


ARMS = ("method_patch", "module_rewrite")
_TERMINAL_STATUSES = {"success", "unlocalized_runtime_failure", "passed_stage_replay_instability"}
_MODULE_SYSTEM_PROMPT = (
    "Repair one ManiSkill policy by rewriting the complete target adapter. "
    "Return only one complete executable Python module implementing the fixed stage contract. "
    "Any _stage_* method may change; the stage order and inherited runtime are fixed. "
    "Read measured state and execute bounded actions."
)


def run_granularity_arm(
    *,
    arm_id: str,
    study_path: Path,
    run_dir: Path,
    resume_adapter: Path,
    max_cycles: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Dispatch one ablation arm over the shared failed candidate."""

    if arm_id == "method_patch":
        return run_selective_migration(
            study_path=study_path,
            run_dir=run_dir,
            max_cycles=max_cycles,
            seed=seed,
            dry_run=False,
            resume_adapter=resume_adapter,
        )
    if arm_id == "module_rewrite":
        return run_module_rewrite_repair(
            study_path=study_path,
            run_dir=run_dir,
            max_cycles=max_cycles,
            seed=seed,
            resume_adapter=resume_adapter,
        )
    raise ValueError(f"Unknown repair-granularity arm {arm_id!r}. Allowed: {', '.join(ARMS)}")


def run_module_rewrite_repair(
    *,
    study_path: Path,
    run_dir: Path,
    max_cycles: int = 3,
    seed: int = 0,
    resume_adapter: Path,
) -> dict[str, Any]:
    """Replay one shared failed candidate, then repair by whole-module rewrite."""

    if resume_adapter is None:
        raise ValueError("module_rewrite requires the shared failed candidate to resume from.")
    study_path = study_path.resolve()
    resume_adapter = resume_adapter.resolve()
    root = study_path.parents[2]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    stage_path = root / study["source"]["stage_representation"]
    stage_program = apply_stage_semantics(json.loads(stage_path.read_text(encoding="utf-8")), study)
    stage_order = [str(value) for value in stage_program["stage_order"]]
    target = study["target"]
    task = study["task"]
    source = study["source"]
    spec = DynamicMigrationSpec(
        env_id=task["env_id"],
        task_label=task["task_id"],
        source_robot=source["robot_uid"],
        target_robot=target["robot_uid"],
        source_control_mode=source["control_mode"],
        target_control_mode=target["control_mode"],
        seed=int(seed),
        max_episode_steps=1000,
        max_target_cycles=int(max_cycles),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    target_observation = discover_environment(spec, spec.target_robot, spec.target_control_mode)
    _write_json(run_dir / "target_observation.json", target_observation)
    _write_json(run_dir / "stage_program.json", stage_program)

    incumbent_code = resume_adapter.read_text(encoding="utf-8")
    incumbent_hashes = validate_selective_adapter(incumbent_code, stage_order=stage_order)
    replay_dir = run_dir / "resume_replay"
    replay_dir.mkdir(exist_ok=True)
    replay_path = replay_dir / "target_adapter.py"
    replay_path.write_text(incumbent_code, encoding="utf-8")
    last_result = run_dynamic_trial(
        spec=spec, robot_uid=spec.target_robot, control_mode=spec.target_control_mode,
        program=FROZEN_PROGRAM, adapter_path=replay_path,
    )
    last_result["repair_localization"] = locate_repair_stage(last_result, stage_program)
    _write_json(replay_dir / "trial_result.json", last_result)
    repair_stage = last_result["repair_localization"]["repair_stage"]
    locked_stages = set(completed_stages(last_result))
    if repair_stage in stage_order:
        locked_stages.difference_update(stage_order[stage_order.index(repair_stage):])
    resume_record = {
        "adapter": str(resume_adapter),
        "failed_candidate_sha256": hashlib.sha256(resume_adapter.read_bytes()).hexdigest(),
        "method_hashes": incumbent_hashes,
        "success": bool(last_result.get("success")),
        "action_steps": int(last_result.get("action_steps") or 0),
        "repair_localization": last_result["repair_localization"],
    }
    seen = {_semantic_hash(incumbent_code)}
    if last_result.get("success"):
        status = "success"
    elif repair_stage is None:
        status = "unlocalized_runtime_failure"
    else:
        status = "target_budget_exhausted"

    cycles: list[dict[str, Any]] = []
    latest_error = ""
    for cycle in range(1, max(1, int(max_cycles)) + 1):
        if status in _TERMINAL_STATUSES:
            break
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        prompt = _module_repair_prompt(
            spec=spec,
            stage_program=stage_program,
            target_observation=target_observation,
            incumbent_code=incumbent_code,
            previous_result=last_result,
            repair_stage=repair_stage,
            locked_stages=sorted(locked_stages),
            latest_error=latest_error,
        )
        (cycle_dir / "system_prompt.txt").write_text(_MODULE_SYSTEM_PROMPT, encoding="utf-8")
        (cycle_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
        generated = gen_text(system=_MODULE_SYSTEM_PROMPT, prompt=prompt, fallback_text="")
        record: dict[str, Any] = {
            "cycle": cycle,
            "model": generated.model,
            "used_llm": generated.used_llm,
            "usage": generated.usage,
            "diagnosed_stage": repair_stage,
            "generation_unit": "complete_module",
            "locked_stages_before": sorted(locked_stages),
        }
        (cycle_dir / "llm_response.txt").write_text(
            generated.raw_text or generated.text,
            encoding="utf-8",
        )
        if not generated.used_llm:
            record.update(valid=False, error=generated.reason or "LLM unavailable")
            _write_json(cycle_dir / "cycle_record.json", record)
            cycles.append(record)
            status = "llm_unavailable"
            break
        try:
            candidate = extract_python_module(generated.text)
            candidate_path = cycle_dir / "target_adapter.py"
            candidate_path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
            semantic_hash = _semantic_hash(candidate)
            if semantic_hash in seen:
                raise ValueError("Generated adapter is semantically unchanged from a tested candidate.")
            seen.add(semantic_hash)
            hashes = validate_selective_adapter(candidate, stage_order=stage_order)
            changed_methods = sorted(
                name for name, value in hashes.items() if incumbent_hashes.get(name) != value
            )
            record["changed_methods"] = changed_methods
            trial = run_dynamic_trial(
                spec=spec,
                robot_uid=spec.target_robot,
                control_mode=spec.target_control_mode,
                program=FROZEN_PROGRAM,
                adapter_path=candidate_path,
            )
            trial_path = cycle_dir / "trial_result.json"
            localization = locate_repair_stage(trial, stage_program)
            trial["repair_localization"] = localization
            _write_json(trial_path, trial)
            selected_repair_stage = localization["repair_stage"]
            record.update(
                valid=True,
                adapter=str(candidate_path),
                semantic_sha256=semantic_hash,
                method_hashes=hashes,
                trial_result=str(trial_path),
                success=bool(trial.get("success")),
                action_steps=int(trial.get("action_steps") or 0),
                failed_stage=failed_stage(trial),
                repair_localization=localization,
                completed_stages=completed_stages(trial),
                message=str(trial.get("message") or ""),
            )
            incumbent_code = candidate
            incumbent_hashes = hashes
            last_result = trial
            latest_error = ""
            locked_stages.update(completed_stages(trial))
            if localization["reason"] == "source_stage_postcondition_not_reached":
                locked_stages.difference_update(stage_order[stage_order.index(selected_repair_stage):])
            if trial.get("success"):
                status = "success"
                repair_stage = None
            elif selected_repair_stage is None:
                status = "unlocalized_runtime_failure"
                repair_stage = None
            elif selected_repair_stage in locked_stages:
                status = "passed_stage_replay_instability"
                repair_stage = None
            else:
                repair_stage = selected_repair_stage
        except Exception as exc:
            latest_error = repr(exc)
            record.update(valid=False, error=latest_error)
        _write_json(cycle_dir / "cycle_record.json", record)
        cycles.append(record)
        if status in _TERMINAL_STATUSES:
            break

    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    if incumbent_code:
        (artifacts / "target_adapter.py").write_text(incumbent_code.rstrip() + "\n", encoding="utf-8")
    result = {
        "schema": "repair_granularity_module_rewrite.v1",
        "arm_id": "module_rewrite",
        "status": status,
        "success": status == "success",
        "study": str(study_path),
        "spec": asdict(spec),
        "runtime_contract": {"gripper_open": 0.0, "gripper_close": 0.81},
        "generation_settings": generation_settings(),
        "source_program_sha256": stage_program.get("source_sha256"),
        "failed_candidate": str(resume_adapter),
        "failed_candidate_sha256": resume_record["failed_candidate_sha256"],
        "repair_output_format": "complete_module",
        "source_metadata_filter": "extracted_source_facts_only",
        "stage_semantics_source": stage_program.get("stage_semantics_source"),
        "stage_order": stage_order,
        "locked_stages": sorted(locked_stages),
        "next_repair_stage": repair_stage,
        "incumbent_method_hashes": incumbent_hashes,
        "cycles": cycles,
        "resume_replay": resume_record,
        "target_adapter": str(artifacts / "target_adapter.py") if incumbent_code else "",
    }
    _write_json(run_dir / "summary.json", result)
    return result


def _module_repair_prompt(
    *,
    spec: DynamicMigrationSpec,
    stage_program: Mapping[str, Any],
    target_observation: Mapping[str, Any],
    incumbent_code: str,
    previous_result: Mapping[str, Any],
    repair_stage: str,
    locked_stages: Sequence[str],
    latest_error: str,
) -> str:
    """Same measured evidence as the method arm; only the output scope differs."""

    name = f"_stage_{repair_stage}"
    source_stage = next(stage for stage in stage_program["stages"] if stage["stage_id"] == repair_stage)
    tree = ast.parse(incumbent_code)
    class_name = next((node.name for node in tree.body if isinstance(node, ast.ClassDef)), "GeneratedRobot")
    methods = {
        node.name: node
        for cls in tree.body if isinstance(cls, ast.ClassDef)
        for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    initialization = ast.get_source_segment(incumbent_code, methods["__init__"]) if "__init__" in methods else "inherited defaults"
    snapshot = previous_result.get("runtime_snapshot") or {}
    evidence = {
        key: previous_result.get(key) for key in
        ("success", "message", "execution_error", "action_steps", "official_evaluation", "repair_localization")
    }
    evidence["stage_boundary_states"] = stage_boundary_states(previous_result)
    evidence["final_state"] = {
        "tcp": snapshot.get("tcp"), "task_state": snapshot.get("task_state"),
        "entities": {key: value for key, value in (snapshot.get("entities") or {}).items()
                     if not key.startswith("segmentation_id_map")},
    }
    module_contract = f"""The module contract is literal, not optional:
```python
from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class {class_name}(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # Keep, improve, or replace any of the five required _stage_* methods.

    def run_source_actions(self) -> bool:
        if not self._run_stage("approach", self._stage_approach): return False
        if not self._run_stage("grasp", self._stage_grasp): return False
        if not self._run_stage("lift", self._stage_lift): return False
        if not self._run_stage("align", self._stage_align): return False
        if not self._run_stage("release", self._stage_release): return False
        return self._official_success()

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> {class_name}:
    return {class_name}(env, control_mode=control_mode, robot_uid=robot_uid)
```
"""
    return f"""Repair one failed target policy for {spec.env_id}, {spec.source_robot} -> {spec.target_robot}
by rewriting the complete adapter. Return exactly one complete Python module and nothing else.
This is a whole-module repair: any of the five _stage_* methods and __init__ may change; the stage
order and the inherited runtime stay fixed. Retained stages: {list(locked_stages)}.
The diagnosed repair stage from measured evidence is "{repair_stage}"; focus the substantive change there.
A module semantically identical to a previously tested candidate is rejected; make an executable
change, not comment edits.

Source-derived relation and successful source-stage observation:
{json.dumps(source_stage, ensure_ascii=False, indent=2)}

Target control mode: {spec.target_control_mode}
Action bounds: {json.dumps(target_observation.get('action_space'), ensure_ascii=False)}
Official evaluation implementation:
{target_observation.get('evaluate_source', '')}

Available inherited API (np is already imported):
- self._entity_pos(name) and self._tcp_pos(): flat xyz numpy arrays, in world coordinates.
- self._official_evaluation(): official boolean fields; self._official_success(): task success bool.
- self._is_grasping_entity(name): actual target grasp detector.
- self._make_action(normalized_world_xyz, gripper=value): maps to the controller frame, leaves rotation zero.
- self._step(action): sends a real simulator action and updates termination.
- self._move_towards(world_target, gripper=value, steps=count): translates TCP, stops within 1 cm.
- self._repeat_action(normalized_world_xyz, gripper=value, steps=count): repeats real commands.
- self._early_stop(): episode ended. Check it in explicit bounded loops.
Motion helper arguments gripper and steps are keyword-only. No tol argument exists.
For opening/closing in place use _repeat_action(np.zeros(3), gripper=..., steps=...).
Never modify the simulator, object poses, controller, official evaluator, runtime methods, or official success.

Current initialization (read-only context):
{initialization}

Current complete candidate module:
```python
{incumbent_code}
```

{module_contract}
Use only Python standard-library modules and numpy. Do not import torch, sapien, mani_skill, gymnasium,
or controller implementations. Do not override _step, _run_stage, _snapshot, _official_evaluation,
_official_success, _entity_catalog, _robot_root_quat, or _world_delta_to_action_delta. Do not define
helper methods outside the five required _stage_* methods.

Real execution evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Each candidate restarts from reset and replays retained stages. The final failed snapshot is not the
entry state of this repair: use the measured preceding-stage boundary and re-read state at runtime.
Use the source object-relative relation to meet the official predicates; TCP arrival alone does not
establish the object's pose. Return True only after measuring each stage's completion.
Latest validation error: {latest_error}
Return only the complete module.
"""


def resolve_failed_candidate(study_path: Path) -> Path:
    """Read the pinned shared failed candidate from the ablation study config."""

    study_path = study_path.resolve()
    root = study_path.parents[2]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    section = study.get("repair_granularity_ablation") or {}
    relative = section.get("failed_candidate")
    if not relative:
        raise ValueError(
            "The study config does not define repair_granularity_ablation.failed_candidate; "
            "pass --resume-adapter explicitly."
        )
    candidate = root / relative
    if not candidate.is_file():
        raise FileNotFoundError(f"Shared failed candidate is missing: {candidate}")
    expected_sha = section.get("failed_candidate_sha256")
    if expected_sha:
        observed_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed_sha != expected_sha:
            raise ValueError(
                "Pinned failed candidate bytes changed "
                f"(expected sha256 {expected_sha}, observed {observed_sha}); "
                "update the ablation study config deliberately, never silently."
            )
    return candidate


def run_granularity_ablation(
    *,
    study_path: Path,
    run_dir: Path,
    resume_adapter: Path,
    max_cycles: int = 3,
    seed: int = 0,
    repetitions: int = 3,
) -> dict[str, Any]:
    """Run paired repair arms from one shared failed candidate with advancing sampling seeds."""

    if int(repetitions) < 1 or int(max_cycles) < 1:
        raise ValueError("repetitions and max_cycles must be positive.")
    study_path = study_path.resolve()
    resume_adapter = resume_adapter.resolve()
    if not resume_adapter.is_file():
        raise FileNotFoundError(f"Shared failed candidate is missing: {resume_adapter}")
    run_dir.mkdir(parents=True, exist_ok=True)
    base_seed = generation_settings()["sampling_seed"]
    candidate_sha = hashlib.sha256(resume_adapter.read_bytes()).hexdigest()
    manifest = {
        "schema": "repair_granularity_ablation_plan.v1",
        "study": str(study_path),
        "failed_candidate": {"path": str(resume_adapter), "sha256": candidate_sha},
        "environment_seed": int(seed),
        "max_cycles_per_run": int(max_cycles),
        "sampling_seeds": [base_seed + index for index in range(int(repetitions))],
        "max_generation_calls": int(repetitions) * len(ARMS) * int(max_cycles),
        "generation_settings": generation_settings(),
        "arms": {
            "method_patch": (
                "LLM returns only the diagnosed _stage_* method; the harness splices it in and "
                "AST-locks every other method (existing selective.py --resume-adapter path)."
            ),
            "module_rewrite": (
                "LLM returns a complete module under the same stage contract with the same measured "
                "evidence; any stage method may change; semantically unchanged modules are rejected."
            ),
        },
        "shared_controls": [
            "same failed candidate bytes and same replay before repair",
            "same repair localization and measured state feedback",
            "same environment seed, model, temperature, token limit, and max cycles",
            "paired LLM sampling seeds across arms",
            "official success only; failed runs stay in the denominator",
        ],
        "note": (
            "Resume-based repair evidence. Never merge these runs into from-scratch "
            "generation comparisons."
        ),
    }
    _write_json(run_dir / "repetition_manifest.json", manifest)
    per_repetition: list[dict[str, Any]] = []
    previous_seed = os.environ.get("EM_LLM_SEED")
    try:
        for index, sampling_seed in enumerate(manifest["sampling_seeds"], start=1):
            os.environ["EM_LLM_SEED"] = str(sampling_seed)
            arms: dict[str, Any] = {}
            for arm_id in ARMS:
                arm_dir = run_dir / f"r{index:02d}" / arm_id
                print(f"[r{index:02d}/{arm_id}] running", flush=True)
                arms[arm_id] = run_granularity_arm(
                    arm_id=arm_id,
                    study_path=study_path,
                    run_dir=arm_dir,
                    resume_adapter=resume_adapter,
                    max_cycles=max_cycles,
                    seed=seed,
                )
                print(
                    f"[r{index:02d}/{arm_id}] success={arms[arm_id]['success']} "
                    f"status={arms[arm_id]['status']}",
                    flush=True,
                )
            per_repetition.append({
                "repetition": index,
                "sampling_seed": sampling_seed,
                "failed_candidate_sha256": candidate_sha,
                "arms": arms,
            })
            summarize_granularity_repetitions(per_repetition, output_path=run_dir / "summary.json")
    finally:
        if previous_seed is None:
            os.environ.pop("EM_LLM_SEED", None)
        else:
            os.environ["EM_LLM_SEED"] = previous_seed
    return summarize_granularity_repetitions(per_repetition, output_path=run_dir / "summary.json")


def _arm_metrics(arm_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    cycles = list(result.get("cycles") or [])
    usages = [row.get("usage") or {} for row in cycles]
    success = bool(result.get("success"))
    cycles_to_success = None
    if success:
        cycles_to_success = next(
            (index for index, row in enumerate(cycles, start=1) if row.get("success")), 0
        )
    return {
        "arm_id": arm_id,
        "success": success,
        "status": result.get("status"),
        "generation_unit": result.get("repair_output_format"),
        "generation_cycles": len(cycles),
        "valid_candidates": sum(row.get("valid") is True for row in cycles),
        "executed_trials": sum(bool(row.get("trial_result")) for row in cycles),
        "cycles_to_success": cycles_to_success,
        "target_action_steps": sum(int(row.get("action_steps") or 0) for row in cycles),
        "replay_action_steps": int((result.get("resume_replay") or {}).get("action_steps") or 0),
        "total_tokens": sum(int(usage.get("total_tokens") or 0) for usage in usages),
        "llm_cost_usd": round(sum(float(usage.get("cost") or 0.0) for usage in usages), 8),
        "infrastructure_failure": result.get("status") == "llm_unavailable",
    }


def summarize_granularity_repetitions(
    per_repetition: Sequence[Mapping[str, Any]],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate paired arms; failed runs always stay in the denominator."""

    rows = []
    for arm_id in ARMS:
        runs = [repetition["arms"][arm_id] for repetition in per_repetition]
        metrics = [_arm_metrics(arm_id, run) for run in runs]
        successes = sum(int(row["success"]) for row in metrics)
        rows.append({
            "arm_id": arm_id,
            "num_runs": len(metrics),
            "num_success": successes,
            "success_rate": successes / len(metrics) if metrics else None,
            "infrastructure_failures": sum(int(row["infrastructure_failure"]) for row in metrics),
            "cycles_to_success": [row["cycles_to_success"] for row in metrics if row["cycles_to_success"]],
            "generation_unit": metrics[0]["generation_unit"] if metrics else None,
            **{key: sum(int(row[key]) for row in metrics) for key in (
                "generation_cycles", "valid_candidates", "executed_trials",
                "target_action_steps", "total_tokens",
            )},
            "replay_action_steps_excluded": sum(int(row["replay_action_steps"]) for row in metrics),
            "llm_cost_usd": round(sum(float(row["llm_cost_usd"]) for row in metrics), 8),
        })
    paired = []
    for repetition in per_repetition:
        arm_metrics = {arm_id: _arm_metrics(arm_id, repetition["arms"][arm_id]) for arm_id in ARMS}
        paired.append({
            "sampling_seed": repetition.get("sampling_seed"),
            **{
                f"{arm_id}_{key}": arm_metrics[arm_id][key]
                for arm_id in ARMS
                for key in ("success", "generation_cycles", "target_action_steps", "total_tokens")
            },
        })
    payload = {
        "schema": "repair_granularity_ablation.v1",
        "failed_candidate_sha256": (
            per_repetition[0].get("failed_candidate_sha256") if per_repetition else None
        ),
        "arms": rows,
        "paired_by_sampling_seed": paired,
        "experimental_unit": "one repair run from one shared failed candidate per arm and repetition",
        "interpretation_limit": (
            "Resume-based paired repair ablation in one fixed scene; not from-scratch generation "
            "success and not a held-out-scene generalization estimate."
        ),
    }
    if output_path is not None:
        _write_json(output_path, payload)
    return payload


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=repr) + "\n", encoding="utf-8")
