"""Controlled baselines for the StackCube selective-migration study."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .code_validation import extract_python_module
from .dynamic_harness import DynamicMigrationSpec, discover_environment, run_dynamic_trial, validate_dynamic_adapter
from .llm import gen_text
from .selective_migration import FROZEN_PROGRAM, _compact_observation, _compact_trial, generation_settings, prompt_stage_program


BASELINE_METHODS = ("target_only", "whole_rewrite")


def run_baseline(
    *,
    method_id: str,
    study_path: Path,
    run_dir: Path,
    max_cycles: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a global-rewrite baseline with or without source-program evidence."""

    if method_id not in BASELINE_METHODS:
        raise ValueError(f"Unknown phase-four method {method_id!r}.")
    study_path = study_path.resolve()
    root = study_path.parents[2]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    source_code = (root / study["source"]["program_evidence"]).read_text(encoding="utf-8")
    stage_program = json.loads(
        (root / study["source"]["stage_representation"]).read_text(encoding="utf-8")
    )
    task = study["task"]
    source = study["source"]
    target = study["target"]
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
    observation = discover_environment(spec, spec.target_robot, spec.target_control_mode)
    _write_json(run_dir / "target_observation.json", observation)

    cycles: list[dict[str, Any]] = []
    incumbent_code = ""
    last_result: dict[str, Any] = {}
    latest_error = ""
    seen: set[str] = set()
    status = "target_budget_exhausted"
    for cycle in range(1, max(1, int(max_cycles)) + 1):
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        prompt = _baseline_prompt(
            method_id=method_id,
            spec=spec,
            observation=observation,
            source_code=source_code if method_id == "whole_rewrite" else "",
            stage_program=stage_program if method_id == "whole_rewrite" else {},
            incumbent_code=incumbent_code,
            previous_result=last_result,
            latest_error=latest_error,
        )
        system = (
            "Write a complete executable Python target adapter for ManiSkill. Return only Python code. "
            "Use measured state and real bounded actions; never mutate simulator state or success."
        )
        (cycle_dir / "system_prompt.txt").write_text(system, encoding="utf-8")
        (cycle_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
        generated = gen_text(system=system, prompt=prompt, fallback_text="")
        record: dict[str, Any] = {
            "cycle": cycle,
            "method_id": method_id,
            "model": generated.model,
            "used_llm": generated.used_llm,
            "usage": generated.usage,
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
            validate_dynamic_adapter(candidate, task_program=FROZEN_PROGRAM, require_program_constant=False)
            trial = run_dynamic_trial(
                spec=spec,
                robot_uid=spec.target_robot,
                control_mode=spec.target_control_mode,
                program=FROZEN_PROGRAM,
                adapter_path=candidate_path,
            )
            trial_path = cycle_dir / "trial_result.json"
            _write_json(trial_path, trial)
            record.update(
                valid=True,
                adapter=str(candidate_path),
                semantic_sha256=semantic_hash,
                trial_result=str(trial_path),
                success=bool(trial.get("success")),
                action_steps=int(trial.get("action_steps") or 0),
                message=str(trial.get("message") or ""),
                failure_reason=(trial.get("failure_diagnosis") or {}).get("reason"),
            )
            incumbent_code = candidate
            last_result = trial
            latest_error = ""
            if trial.get("success"):
                status = "success"
        except Exception as exc:
            latest_error = repr(exc)
            record.update(valid=False, error=latest_error)
        _write_json(cycle_dir / "cycle_record.json", record)
        cycles.append(record)
        if status == "success":
            break

    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    if incumbent_code:
        (artifacts / "target_adapter.py").write_text(incumbent_code.rstrip() + "\n", encoding="utf-8")
    result = {
        "schema": "selective_migration_baseline.v1",
        "method_id": method_id,
        "status": status,
        "success": status == "success",
        "study": str(study_path),
        "spec": asdict(spec),
        "runtime_contract": {"gripper_open": 0.0, "gripper_close": 0.81},
        "generation_settings": generation_settings(),
        "source_program_sha256": stage_program.get("source_sha256") if method_id == "whole_rewrite" else None,
        "source_information": (
            "none beyond task and target interface"
            if method_id == "target_only"
            else "complete source code plus deterministic source-stage representation"
        ),
        "source_metadata_filter": "extracted_source_facts_only",
        "repair_policy": "global complete-module rewrite",
        "cycles": cycles,
        "target_adapter": str(artifacts / "target_adapter.py") if incumbent_code else "",
    }
    _write_json(run_dir / "summary.json", result)
    return result


def validate_reference_result(result: Mapping[str, Any], study: Mapping[str, Any], *, seed: int, max_cycles: int) -> None:
    """Reject old or differently configured selective evidence before paid baseline runs."""
    spec = result.get("spec") or {}
    expected = {
        "env_id": study["task"]["env_id"],
        "source_robot": study["source"]["robot_uid"],
        "target_robot": study["target"]["robot_uid"],
        "source_control_mode": study["source"]["control_mode"],
        "target_control_mode": study["target"]["control_mode"],
        "seed": seed,
        "max_target_cycles": max_cycles,
    }
    mismatches = [key for key, value in expected.items() if spec.get(key) != value]
    if result.get("resume_replay") is not None:
        mismatches.append("resume_replay (comparison requires generation from scratch)")
    if result.get("source_metadata_filter") != "extracted_source_facts_only":
        mismatches.append("source_metadata_filter")
    contract = result.get("runtime_contract") or {}
    for key, value in (("gripper_open", study["target"]["observed_gripper_open"]),
                       ("gripper_close", study["target"]["observed_gripper_closed"])):
        if contract.get(key) != value:
            mismatches.append(key)
    if mismatches:
        raise ValueError("Selective reference is incompatible with the current study: " + ", ".join(mismatches))


def build_comparison_summary(
    *,
    method_results: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    for method_id in ("target_only", "whole_rewrite", "selective"):
        result = method_results[method_id]
        cycles = list(result.get("cycles") or [])
        usages = [row.get("usage") or {} for row in cycles]
        last_runtime_failure = next(
            (
                row.get("failure_reason") or row.get("failed_stage")
                for row in reversed(cycles)
                if not row.get("success") and (row.get("failure_reason") or row.get("failed_stage"))
            ),
            None,
        )
        last_generation_error = next(
            (row.get("error") for row in reversed(cycles) if row.get("error")),
            None,
        )
        last_physical_failure = next(
            (
                row.get("failure_reason") or row.get("failed_stage")
                for row in reversed(cycles)
                if int(row.get("action_steps") or 0) > 0
                and not row.get("success")
                and (row.get("failure_reason") or row.get("failed_stage"))
            ),
            None,
        )
        rows.append({
            "method_id": method_id,
            "success": bool(result.get("success")),
            "status": result.get("status"),
            "generation_cycles": len(cycles),
            "valid_candidates": sum(row.get("valid") is True for row in cycles),
            "executed_trials": sum(bool(row.get("trial_result")) for row in cycles),
            "trials_with_actions": sum(int(row.get("action_steps") or 0) > 0 for row in cycles),
            "target_action_steps": sum(int(row.get("action_steps") or 0) for row in cycles),
            "total_tokens": sum(int(usage.get("total_tokens") or 0) for usage in usages),
            "llm_cost_usd": round(sum(float(usage.get("cost") or 0.0) for usage in usages), 8),
            "last_failure": last_runtime_failure or last_generation_error,
            "last_runtime_failure": last_runtime_failure,
            "last_physical_failure": last_physical_failure,
            "last_generation_error": last_generation_error,
            "completed_stages": list(result.get("locked_stages") or []),
        })
    payload = {
        "schema": "selective_migration_comparison.v1",
        "protocol": dict(protocol),
        "rows": rows,
        "interpretation_limit": (
            "One run per method is a pipeline pilot, not evidence that one method is superior."
        ),
    }
    _write_json(output_path, payload)
    return payload


def summarize_repetitions(comparisons: list[Mapping[str, Any]], *, output_path: Path) -> dict[str, Any]:
    """Retain every completed independent run in the aggregate denominator."""
    rows = []
    for method in ("target_only", "whole_rewrite", "selective"):
        runs = []
        for comparison in comparisons:
            matching = [row for row in comparison["rows"] if row["method_id"] == method]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one {method} result per repetition.")
            runs.append(matching[0])
        successes = sum(bool(row["success"]) for row in runs)
        rows.append({
            "method_id": method, "num_runs": len(runs), "num_success": successes,
            "success_rate": successes / len(runs) if runs else None,
            "infrastructure_failures": sum(row["status"] == "llm_unavailable" for row in runs),
            **{key: sum(int(row.get(key) or 0) for row in runs) for key in
               ("generation_cycles", "valid_candidates", "target_action_steps", "total_tokens")},
            "llm_cost_usd": round(sum(float(row.get("llm_cost_usd") or 0) for row in runs), 8),
        })
    payload = {"schema": "selective_migration_repetitions.v1", "rows": rows,
               "protocols": [comparison["protocol"] for comparison in comparisons],
               "interpretation_limit": "Fixed-scene independent generation pilot; not a held-out-scene generalization estimate."}
    _write_json(output_path, payload)
    return payload


def _baseline_prompt(
    *,
    method_id: str,
    spec: DynamicMigrationSpec,
    observation: Mapping[str, Any],
    source_code: str,
    stage_program: Mapping[str, Any],
    incumbent_code: str,
    previous_result: Mapping[str, Any],
    latest_error: str,
) -> str:
    source_section = (
        "No source robot program or source execution evidence is available. Solve from the task and target interface."
        if method_id == "target_only"
        else f"""Successful Panda source program:
```python
{source_code}
```

Deterministic representation derived from that same successful source execution:
{json.dumps(prompt_stage_program(stage_program), ensure_ascii=False, indent=2)}

Use this evidence, but rewrite the complete target adapter globally each round. No stage is locked."""
    )
    return f"""Generate a target adapter for this controlled baseline.

Method: {method_id}
Experiment: {json.dumps(asdict(spec), ensure_ascii=False, indent=2)}
Target interface: {json.dumps(_compact_observation(observation), ensure_ascii=False, indent=2)}

{source_section}

Required module contract:
```python
from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    def run_source_actions(self) -> bool:
        # Implement the complete target policy here using measured state and bounded actions.
        ...

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
```

Use only Python standard-library modules and numpy. Do not import torch, sapien, mani_skill,
gymnasium, motion planners, or controllers. Do not override _step, _run_stage, _snapshot,
_official_evaluation, _official_success, _entity_catalog, _robot_root_quat, or
	_world_delta_to_action_delta. Runtime values are flat numpy vectors. The inherited API is:
	- self._entity_pos("cubeA") / self._entity_pos("cubeB") -> flat xyz arrays
	- self._tcp_pos() -> flat TCP xyz array
	- self._official_evaluation() -> official task booleans
	- self._is_grasping_entity("cubeA") -> target grasp detector
	- self._move_towards(world_target, gripper=value, steps=count)
	- self._repeat_action(delta_xyz, gripper=value, steps=count)
	Do not call _entity_catalog() or subscript Actor objects. Both motion helpers require gripper and
	steps as keyword-only arguments and neither accepts tol. Check self._early_stop() in explicit loops.
	Measured gripper mapping: 0.0 opens, 0.81 closes. _move_towards returns immediately within
	1 cm of its target, so stationary open/close must use _repeat_action(np.zeros(3), gripper=..., steps=...).
	Never mutate object poses, simulator state, controller internals, or official success.
Returning True is insufficient; the harness independently requires official evaluate()['success'].

Current complete candidate, empty on the first round:
```python
{incumbent_code}
```

Latest real execution evidence:
{json.dumps(_compact_trial(previous_result), ensure_ascii=False, indent=2)}

Latest validation error:
{latest_error}

Return a complete module. For a repair, make a substantive change addressing the evidence; unlike
the selective method, this baseline may rewrite any policy method.
"""


def _semantic_hash(code: str) -> str:
    tree = ast.parse(code)
    value = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=repr) + "\n", encoding="utf-8")
