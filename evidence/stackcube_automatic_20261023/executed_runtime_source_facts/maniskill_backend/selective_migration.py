"""Stage-preserving, failure-localized embodiment migration."""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .code_validation import extract_python_module
from .dynamic_harness import (
    DynamicMigrationSpec,
    discover_environment,
    run_dynamic_trial,
    validate_dynamic_adapter,
)
from .llm import gen_text
from .dynamic_adapter import _scalar_bool


FROZEN_PROGRAM = "ret_val = robot.run_source_actions()"
_STATE_CALLS = {
    "_snapshot",
    "_tcp_pos",
    "_entity_pos",
    "_actor_pos",
    "_entity_quat",
    "_is_grasping_entity",
    "_official_evaluation",
    "_official_success",
}
_ACTION_CALLS = {"_step", "_move_towards", "_repeat_action"}


def generation_settings() -> dict[str, Any]:
    """Record effective public sampling settings without credentials."""
    from llm_client import (
        completion_token_limit, current_provider, default_model, generation_temperature,
        llm_seed, openrouter_reasoning_effort, request_max_retries, request_timeout_seconds,
    )
    provider = current_provider()
    return {
        "provider": provider, "model": default_model(provider),
        "temperature": generation_temperature(), "sampling_seed": llm_seed(),
        "max_tokens": completion_token_limit(),
        "reasoning_effort": openrouter_reasoning_effort() if provider == "openrouter" else None,
        "request_timeout_seconds": request_timeout_seconds(), "request_max_retries": request_max_retries(),
    }


def prompt_stage_program(program: Mapping[str, Any]) -> dict[str, Any]:
    """Expose extracted source facts, excluding post-hoc human validation metadata."""
    keys = ("schema", "task_id", "source_program", "source_sha256", "entrypoint",
            "source_robot", "source_control_mode", "stage_order", "stages", "source_assumptions",
            "official_evaluation_at_trace_end", "trace_summary", "provenance")
    return {key: program[key] for key in keys if key in program}


def method_hashes(code: str) -> dict[str, str]:
    """Hash each function by semantics rather than source formatting."""

    tree = ast.parse(code)
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        result[node.name] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return result


def apply_stage_method(incumbent_code: str, replacement: str, *, stage_id: str) -> str:
    """Splice exactly one generated method while retaining every other source byte."""
    snippet = textwrap.dedent(extract_python_module(replacement)).strip()
    tree = ast.parse(snippet)
    name = f"_stage_{stage_id}"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"Return only def {name}(self), without imports, classes, or other definitions.")
    method = tree.body[0]
    if method.name != name or method.decorator_list:
        raise ValueError(f"The sole replacement must be undecorated def {name}(self).")
    incumbent_tree = ast.parse(incumbent_code)
    matches = [
        node for cls in incumbent_tree.body if isinstance(cls, ast.ClassDef)
        for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError(f"Expected exactly one undecorated incumbent {name} method.")
    original = matches[0]
    lines = incumbent_code.splitlines(keepends=True)
    lines[original.lineno - 1:original.end_lineno] = [
        textwrap.indent(snippet + "\n", " " * original.col_offset)
    ]
    return "".join(lines)


def validate_selective_adapter(
    code: str,
    *,
    stage_order: Sequence[str],
    incumbent_code: str = "",
    allowed_stage: str | None = None,
) -> dict[str, str]:
    """Validate the fixed stage interface and optional single-stage repair."""

    validate_dynamic_adapter(
        code,
        task_program=FROZEN_PROGRAM,
        require_program_constant=False,
    )
    tree = ast.parse(code)
    methods = {
        node.name: node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = [f"_stage_{stage_id}" for stage_id in stage_order]
    missing = [name for name in ["run_source_actions", *required] if name not in methods]
    if missing:
        raise ValueError("Selective adapter is missing methods: " + ", ".join(missing))
    allowed_methods = {"__init__", "run_source_actions", *required}
    extra_methods = sorted(set(methods) - allowed_methods)
    if extra_methods:
        raise ValueError(
            "Selective adapter may not add policy helpers outside stage boundaries: "
            + ", ".join(extra_methods)
        )

    observed_calls: list[tuple[str, str]] = []
    for node in sorted(ast.walk(methods["run_source_actions"]), key=lambda item: getattr(item, "lineno", 0)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_run_stage"
        ):
            continue
        stage_arg = node.args[0] if node.args else None
        callback_arg = node.args[1] if len(node.args) > 1 else None
        if not (
            isinstance(stage_arg, ast.Constant)
            and isinstance(stage_arg.value, str)
            and isinstance(callback_arg, ast.Attribute)
        ):
            raise ValueError("run_source_actions must call _run_stage with literal stage names and methods.")
        observed_calls.append((stage_arg.value, callback_arg.attr))
    expected_calls = [(stage_id, f"_stage_{stage_id}") for stage_id in stage_order]
    if observed_calls != expected_calls:
        raise ValueError(
            "run_source_actions changed the frozen stage order; "
            f"expected {expected_calls!r}, got {observed_calls!r}."
        )

    for method_name in required:
        node = methods[method_name]
        positional = list(node.args.posonlyargs) + list(node.args.args)
        if len(positional) != 1 or positional[0].arg != "self":
            raise ValueError(f"{method_name} must have signature {method_name}(self).")
        calls = {
            item.func.attr
            for item in ast.walk(node)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute)
        }
        if "_entity_catalog" in calls:
            raise ValueError(
                f"{method_name} may not use _entity_catalog(); call _entity_pos(name) instead."
            )
        if not calls.intersection(_STATE_CALLS):
            raise ValueError(f"{method_name} must read measured physical state.")
        if not calls.intersection(_ACTION_CALLS):
            raise ValueError(f"{method_name} must execute at least one bounded action path.")
        if not any(isinstance(item, (ast.If, ast.For, ast.While)) for item in ast.walk(node)):
            raise ValueError(f"{method_name} must branch on measured progress or a bounded loop.")

    hashes = method_hashes(code)
    if not incumbent_code:
        return hashes
    if allowed_stage not in stage_order:
        raise ValueError("A valid failed stage is required before local repair.")
    allowed_method = f"_stage_{allowed_stage}"
    incumbent_hashes = method_hashes(incumbent_code)
    if set(hashes) != set(incumbent_hashes):
        raise ValueError("Local repair changed the module method set.")
    changed = sorted(name for name in hashes if hashes[name] != incumbent_hashes[name])
    if changed != [allowed_method]:
        raise ValueError(
            f"Only {allowed_method} may change in this repair; changed methods: {changed!r}."
        )
    return hashes


def failed_stage(result: Mapping[str, Any]) -> str | None:
    snapshot = result.get("runtime_snapshot")
    if isinstance(snapshot, Mapping):
        runtime = snapshot.get("stage_runtime")
        if isinstance(runtime, Mapping) and runtime.get("failed_stage"):
            return str(runtime["failed_stage"])
    events = result.get("execution_log")
    if isinstance(events, list):
        for event in reversed(events):
            if not isinstance(event, Mapping) or event.get("phase") != "stage_end":
                continue
            values = event.get("values")
            if isinstance(values, Mapping) and values.get("ok") is False:
                return str(values.get("stage_id"))
    return None


def completed_stages(result: Mapping[str, Any]) -> list[str]:
    snapshot = result.get("runtime_snapshot")
    if not isinstance(snapshot, Mapping):
        return []
    runtime = snapshot.get("stage_runtime")
    if not isinstance(runtime, Mapping):
        return []
    return [str(value) for value in runtime.get("completed_stages") or []]


def stage_boundary_states(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read observed stage-end states from action attribution, not generated claims."""
    states = {}
    for row in result.get("source_action_trace") or []:
        function = row.get("source_function") or ""
        snapshot = row.get("state_after")
        if not function.startswith("_stage_") or not isinstance(snapshot, Mapping):
            continue
        states[function[len("_stage_"):]] = {
            "step": snapshot.get("step"),
            "tcp": snapshot.get("tcp"),
            "entities": {
                name: value for name, value in (snapshot.get("entities") or {}).items()
                if not name.startswith("segmentation_id_map")
            },
            "official_evaluation": snapshot.get("official_evaluation") or {},
        }
    return states


def locate_repair_stage(result: Mapping[str, Any], stage_program: Mapping[str, Any]) -> dict[str, Any]:
    """Question a passed stage when it contradicts a source-observed new true predicate."""
    runtime_failed = failed_stage(result)
    location = {"runtime_failed_stage": runtime_failed, "repair_stage": runtime_failed,
                "reason": "runtime_failed_stage"}
    if result.get("success"):
        return {"runtime_failed_stage": None, "repair_stage": None, "reason": "official_success"}
    boundaries = stage_boundary_states(result)
    completed = completed_stages(result)
    previous = None
    for stage in stage_program.get("stages") or []:
        stage_id = stage["stage_id"]
        source_eval = ((stage.get("runtime") or {}).get("final_state") or {}).get("official_evaluation") or {}
        required = [
            key for key, value in source_eval.items()
            if isinstance(value, bool) and value and previous is not None
            and previous.get(key) is False
        ]
        previous = source_eval
        observed = (boundaries.get(stage_id) or {}).get("official_evaluation") or {}
        violated = [key for key in required if key in observed and not _scalar_bool(observed[key])]
        if stage_id in completed and violated:
            return {
                "runtime_failed_stage": runtime_failed, "repair_stage": stage_id,
                "reason": "source_stage_postcondition_not_reached",
                "evidence": {"source_observed_new_true_predicates": required,
                             "violated_predicates": violated,
                             "target_boundary": boundaries[stage_id]},
            }
    return location


def run_selective_migration(
    *,
    study_path: Path,
    run_dir: Path,
    max_cycles: int = 3,
    seed: int = 0,
    dry_run: bool = False,
    resume_adapter: Path | None = None,
) -> dict[str, Any]:
    """Generate once, then repair only the first failed stage."""

    if dry_run and resume_adapter is not None:
        raise ValueError("--resume-adapter replays real actions and cannot be combined with --dry-run.")
    study_path = study_path.resolve()
    root = study_path.parents[2]
    study = json.loads(study_path.read_text(encoding="utf-8"))
    stage_path = root / study["source"]["stage_representation"]
    stage_program = json.loads(stage_path.read_text(encoding="utf-8"))
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

    cycles: list[dict[str, Any]] = []
    incumbent_code = ""
    incumbent_hashes: dict[str, str] = {}
    repair_stage: str | None = None
    locked_stages: set[str] = set()
    last_result: dict[str, Any] = {}
    latest_error = ""
    status = "target_budget_exhausted"
    resume_record = None
    if resume_adapter is not None:
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
        observed_completed = completed_stages(last_result)
        locked_stages.update(observed_completed)
        if repair_stage in stage_order:
            locked_stages.difference_update(stage_order[stage_order.index(repair_stage):])
        resume_record = {
            "adapter": str(resume_adapter), "method_hashes": incumbent_hashes,
            "success": bool(last_result.get("success")),
            "action_steps": int(last_result.get("action_steps") or 0),
            "repair_localization": last_result["repair_localization"],
        }
        if last_result.get("success"):
            status = "success"
        elif repair_stage is None:
            status = "unlocalized_runtime_failure"

    for cycle in range(1, max(1, int(max_cycles)) + 1):
        if status in {"success", "unlocalized_runtime_failure"}:
            break
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        prompt = _selective_prompt(
            spec=spec,
            stage_program=stage_program,
            target_observation=target_observation,
            incumbent_code=incumbent_code,
            previous_result=last_result,
            repair_stage=repair_stage,
            locked_stages=sorted(locked_stages),
            latest_error=latest_error,
        )
        method_only = bool(incumbent_code and repair_stage)
        system = (
            f"Repair one ManiSkill policy method. Return only def _stage_{repair_stage}(self) -> bool, "
            "with no class, imports, explanations, or other methods. Read measured state and execute bounded actions."
            if method_only else
            "Write a complete executable Python target adapter for ManiSkill. Return only Python code. "
            "Preserve the supplied task-stage program exactly and never mutate simulator state or success."
        )
        (cycle_dir / "system_prompt.txt").write_text(system, encoding="utf-8")
        (cycle_dir / "user_prompt.txt").write_text(prompt, encoding="utf-8")
        generated = gen_text(
            system=system,
            prompt=prompt,
            fallback_text="",
            dry_run=dry_run,
        )
        record: dict[str, Any] = {
            "cycle": cycle,
            "model": generated.model,
            "used_llm": generated.used_llm,
            "usage": generated.usage,
            "allowed_stage": repair_stage,
            "generation_unit": "stage_method" if method_only else "complete_module",
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
            if method_only:
                (cycle_dir / "repair_method.py").write_text(extract_python_module(generated.text) + "\n", encoding="utf-8")
                candidate = apply_stage_method(incumbent_code, generated.text, stage_id=repair_stage)
            else:
                candidate = extract_python_module(generated.text)
            candidate_path = cycle_dir / "target_adapter.py"
            candidate_path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
            hashes = validate_selective_adapter(
                candidate,
                stage_order=stage_order,
                incumbent_code=incumbent_code,
                allowed_stage=repair_stage,
            )
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
            observed_failed_stage = failed_stage(trial)
            observed_completed = completed_stages(trial)
            selected_repair_stage = localization["repair_stage"]
            record.update(
                valid=True,
                adapter=str(candidate_path),
                method_hashes=hashes,
                trial_result=str(trial_path),
                success=bool(trial.get("success")),
                action_steps=int(trial.get("action_steps") or 0),
                failed_stage=observed_failed_stage,
                repair_localization=localization,
                completed_stages=observed_completed,
                message=str(trial.get("message") or ""),
            )
            incumbent_code = candidate
            incumbent_hashes = hashes
            last_result = trial
            latest_error = ""
            locked_stages.update(observed_completed)
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
        if status in {"success", "unlocalized_runtime_failure", "passed_stage_replay_instability"}:
            break

    artifacts = run_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    if incumbent_code:
        (artifacts / "target_adapter.py").write_text(incumbent_code.rstrip() + "\n", encoding="utf-8")
    result = {
        "schema": "selective_migration_result.v1",
        "status": status,
        "success": status == "success",
        "study": str(study_path),
        "spec": asdict(spec),
        "runtime_contract": {"gripper_open": 0.0, "gripper_close": 0.81},
        "generation_settings": generation_settings(),
        "repair_output_format": "single_stage_method",
        "source_metadata_filter": "extracted_source_facts_only",
        "source_program_sha256": stage_program.get("source_sha256"),
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


def _selective_prompt(
    *,
    spec: DynamicMigrationSpec,
    stage_program: Mapping[str, Any],
    target_observation: Mapping[str, Any],
    incumbent_code: str,
    previous_result: Mapping[str, Any],
    repair_stage: str | None,
    locked_stages: Sequence[str],
    latest_error: str,
) -> str:
    if incumbent_code and repair_stage:
        return _stage_repair_prompt(
            spec=spec, stage_program=stage_program, target_observation=target_observation,
            incumbent_code=incumbent_code, previous_result=previous_result,
            repair_stage=repair_stage, locked_stages=locked_stages, latest_error=latest_error,
        )
    stage_order = [str(value) for value in stage_program["stage_order"]]
    stage_contract = "\n".join(
        f'- "{stage_id}" -> self._stage_{stage_id}' for stage_id in stage_order
    )
    if incumbent_code and repair_stage:
        repair_rule = (
            f"This is a local repair. Change exactly one method: _stage_{repair_stage}. "
            "Every other method must remain semantically identical; AST hashes are checked."
        )
    else:
        repair_rule = "This is initial generation. Implement all required stage methods."
    return f"""Create the target-side adapter for this fixed selective-migration study.

Experiment:
{json.dumps(asdict(spec), ensure_ascii=False, indent=2)}

Frozen source-derived stage program:
{json.dumps(prompt_stage_program(stage_program), ensure_ascii=False, indent=2)}

Observed target interface and reset state:
{json.dumps(_compact_observation(target_observation), ensure_ascii=False, indent=2)}

Required stage interface and order:
{stage_contract}

The public method must contain these five literal calls in exactly this order:
    if not self._run_stage("approach", self._stage_approach): return False
    if not self._run_stage("grasp", self._stage_grasp): return False
    if not self._run_stage("lift", self._stage_lift): return False
    if not self._run_stage("align", self._stage_align): return False
    if not self._run_stage("release", self._stage_release): return False
    return self._official_success()

Each _stage_* method must read measured state, use a bounded loop or branch, execute real actions,
and return True only when its own stage relation is measured. Use exact entity names cubeA and cubeB.
Use the observed 7D pd_ee_delta_pose layout. The inherited _make_action fills translation and the
last gripper channel while leaving rotation increments zero. Measured finger-pad separation confirms
gripper_open=0.0 and gripper_close=0.81 for this mode. Keep world-frame object relations and let inherited motion helpers convert world
translation into the robot-root action frame. Do not change the environment, controller, object
poses, official evaluation, or FROZEN_PROGRAM. Define build_robot(env, *, control_mode, robot_uid).

The module contract is literal, not optional:
```python
from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedSelectiveRobot(ManiSkillDynamicRobot):
    def __init__(self, env: Any, *, control_mode: str, robot_uid: str) -> None:
        super().__init__(env, control_mode=control_mode, robot_uid=robot_uid,
                         gripper_open=0.0, gripper_close=0.81,
                         move_steps=30, settle_steps=12, max_delta_m=0.04)

    # Implement exactly the five required _stage_* methods here.

    def run_source_actions(self) -> bool:
        # Insert the five literal _run_stage calls shown above, unchanged.
        ...

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedSelectiveRobot:
    return GeneratedSelectiveRobot(env, control_mode=control_mode, robot_uid=robot_uid)
```
Use only Python standard-library modules and numpy. Do not import torch, sapien, mani_skill, gymnasium,
or controller implementations. Do not override _step, _run_stage, _snapshot, _official_evaluation,
_official_success, _entity_catalog, _robot_root_quat, or _world_delta_to_action_delta. Do not define
your own environment wrappers or task-success logic. The generated class may define only __init__,
the five required _stage_* methods, and run_source_actions; do not add helper methods. Inside every
stage call documented inherited APIs directly, including self._early_stop() in bounded loops.
Return the complete module, not a patch.

Runtime measurement values are flat numpy vectors. Read them only through these inherited methods:
- self._entity_pos("cubeA") and self._entity_pos("cubeB") return flat xyz arrays.
- self._tcp_pos() returns a flat TCP xyz array.
- self._official_evaluation() returns the official task booleans.
- self._is_grasping_entity("cubeA") returns the target grasp detector.
- self._move_towards(world_target, gripper=value, steps=count) moves toward a world-frame xyz target.
- self._repeat_action(delta_xyz, gripper=value, steps=count) repeats one normalized translation/gripper command.
Do not call _entity_catalog() and do not subscript Actor objects. The JSON target observation above is
prompt evidence, not the runtime Python object layout. Prefer self._move_towards(world_target,
gripper=..., steps=...) for closed-loop translation; it already normalizes commands, checks episode
termination, and executes self._step(action). Both motion helpers require gripper and steps as
keyword-only arguments and neither accepts tol.
_move_towards returns immediately within 1 cm of its target, so it cannot reliably open or close
at the current TCP position. For stationary gripper commands use
self._repeat_action(np.zeros(3, dtype=np.float32), gripper=..., steps=...).

{repair_rule}
Stages retained after observed postcondition checks: {json.dumps(list(locked_stages))}

Current complete target adapter:
```python
{incumbent_code}
```

Latest real execution evidence:
{json.dumps(_compact_trial(previous_result), ensure_ascii=False, indent=2)}

Latest validation error:
{latest_error}
"""


def _stage_repair_prompt(
    *, spec: DynamicMigrationSpec, stage_program: Mapping[str, Any],
    target_observation: Mapping[str, Any], incumbent_code: str,
    previous_result: Mapping[str, Any], repair_stage: str,
    locked_stages: Sequence[str], latest_error: str,
) -> str:
    name = f"_stage_{repair_stage}"
    source_stage = next(stage for stage in stage_program["stages"] if stage["stage_id"] == repair_stage)
    tree = ast.parse(incumbent_code)
    methods = {node.name: node for cls in tree.body if isinstance(cls, ast.ClassDef)
               for node in cls.body if isinstance(node, ast.FunctionDef)}
    current = ast.get_source_segment(incumbent_code, methods[name])
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
    return f"""Repair one failed target method for {spec.env_id}, {spec.source_robot} -> {spec.target_robot}.
Return exactly one Python function: def {name}(self) -> bool.
The harness will insert it into the saved adapter. Do not output the class, imports, or other methods.
All other methods and the stage order are preserved by the harness. Retained stages: {list(locked_stages)}.
The replacement must change executable logic; copying the current body or editing comments is rejected.

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
Never modify the simulator, object poses, controller, official evaluator, runtime methods, or other stages.

Current initialization (read-only context):
{initialization}

Current method to replace:
{current}

Real execution evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Each candidate restarts from reset and replays retained stages. The final failed snapshot is not the
entry state of this method: use the measured preceding-stage boundary and re-read state at runtime.
Use the source object-relative relation to meet the official predicates; TCP arrival alone does not
establish the object's pose. Return True only after measuring this stage's completion.
Latest validation error: {latest_error}
Return only the replacement function.
"""


def _compact_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "env_id",
        "robot_uid",
        "control_mode",
        "supported_robots",
        "action_space",
        "controller_action_mapping",
        "tcp",
        "pose_entities",
        "entity_aliases",
        "official_evaluation",
        "evaluate_source",
    )
    return {key: value.get(key) for key in keys if key in value}


def _compact_trial(value: Mapping[str, Any]) -> dict[str, Any]:
    if not value:
        return {}
    keys = (
        "success",
        "message",
        "execution_error",
        "official_success",
        "official_evaluation",
        "action_steps",
        "runtime_snapshot",
        "failure_diagnosis",
        "terminated",
        "truncated",
        "repair_localization",
    )
    compact = {key: value.get(key) for key in keys if key in value}
    compact["stage_boundary_states"] = stage_boundary_states(value)
    events = value.get("execution_log")
    if isinstance(events, list):
        compact["stage_events"] = [
            event
            for event in events
            if isinstance(event, Mapping) and event.get("api") == "state"
        ][-20:]
    return compact


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=repr) + "\n", encoding="utf-8")
