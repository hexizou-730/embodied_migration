"""Discovery-first migration for ManiSkill tasks without registered cases."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from lmp.executor import execute_lmp, is_safe

from .code_validation import extract_python_module, validate_generated_adapter_module
from .dynamic_adapter import ManiSkillDynamicRobot, iter_pose_entities
from .dynamic_adapter import ManiSkillSceneAdapter, _scalar_bool, _to_numpy
from .env_adapter import ManiSkillEnvAdapter
from .experiment_environment import capture_experiment_environment
from .llm import gen_text


TASK_PROGRAM_NAME = "TASK_PROGRAM"
_ENV_ID = re.compile(r".+-v\d+$", re.IGNORECASE)
_DANGEROUS_CALLS = {"step", "reset"}
_PROTECTED_METHODS = {
    "_step",
    "_official_evaluation",
    "_official_success",
    "_snapshot",
    "_entity_catalog",
}
_STATE_CALLS = {
    "_tcp_pos",
    "_actor_pos",
    "_entity_pos",
    "_entity_quat",
    "_region_pos",
    "_is_grasping_entity",
    "_official_evaluation",
    "_official_success",
    "_snapshot",
}
_TRUSTED_ACTION_HELPERS = {"_move_towards", "_repeat_action"}


def _success_from_ret_val(ret_val: Any) -> bool:
    if ret_val is True:
        return True
    if ret_val is None:
        return False
    if isinstance(ret_val, str):
        lowered = ret_val.lower()
        return not (lowered.startswith("failure") or lowered.startswith("infeasible"))
    return bool(ret_val)


@dataclass(frozen=True)
class DynamicMigrationSpec:
    env_id: str
    task_label: str
    source_robot: str
    target_robot: str
    source_control_mode: str = "pd_ee_delta_pos"
    target_control_mode: str = "pd_ee_delta_pos"
    seed: int = 0
    max_episode_steps: int = 500
    max_source_cycles: int = 2
    max_target_cycles: int = 3
    obs_mode: str = "state"
    sim_backend: str = "auto"
    render_backend: str = "gpu"


@dataclass(frozen=True)
class SourceActions:
    """An existing source action module, inspected without executing its imports."""

    path: str
    code: str
    env_id: str
    robot_uid: str
    control_mode: str
    program: str
    entrypoint: str


def read_source_actions(path: Path) -> SourceActions:
    code = path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(path))
    constants = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if "run" in functions:
        entrypoint, program = "run", "ret_val = robot.run_source_actions()"
    elif "build_robot" in functions and constants.get(TASK_PROGRAM_NAME):
        entrypoint, program = "build_robot", constants[TASK_PROGRAM_NAME].strip()
    else:
        raise ValueError("Source file needs run(env), or build_robot(...) and TASK_PROGRAM.")
    return SourceActions(
        path=str(path.resolve()), code=code, env_id=constants.get("ENV_ID", ""),
        robot_uid=constants.get("SOURCE_ROBOT", ""),
        control_mode=constants.get("CONTROL_MODE", "pd_ee_delta_pos"),
        program=program, entrypoint=entrypoint,
    )


def looks_like_env_id(value: str) -> bool:
    return bool(_ENV_ID.fullmatch(str(value or "").strip()))


def discover_environment(spec: DynamicMigrationSpec, robot_uid: str, control_mode: str) -> Dict[str, Any]:
    """Inspect one initialized environment without executing task actions."""

    adapter = ManiSkillEnvAdapter(
        spec.env_id,
        robot_uid=robot_uid,
        obs_mode=spec.obs_mode,
        control_mode=control_mode,
        sim_backend=spec.sim_backend,
        render_backend=spec.render_backend,
        max_episode_steps=spec.max_episode_steps,
        reward_mode="sparse",
    )
    try:
        env = adapter.make()
        observation, reset_info = adapter.reset(seed=spec.seed)
        base = getattr(env, "unwrapped", env)
        agent = getattr(base, "agent", None)
        tcp = _read_tcp(agent)
        entities: Dict[str, Any] = {}
        for name, actor in iter_pose_entities(base):
            try:
                entities[name] = {
                    "position": np.round(_to_numpy(actor.pose.p), 6).tolist(),
                    "quaternion": np.round(_to_numpy(actor.pose.q), 6).tolist(),
                }
            except Exception:
                continue
        evaluation = _safe_evaluate(base)
        controller = getattr(agent, "controller", None)
        cls = base.__class__
        return {
            "env_id": spec.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "environment_class": f"{cls.__module__}.{cls.__name__}",
            "environment_doc": _trim(inspect.getdoc(cls) or "", 6000),
            "supported_robots": _jsonable(getattr(base, "SUPPORTED_ROBOTS", getattr(cls, "SUPPORTED_ROBOTS", None))),
            "action_space": _space_summary(getattr(env, "action_space", None)),
            "observation_shape": list(getattr(observation, "shape", ())) if hasattr(observation, "shape") else None,
            "controller_summary": _trim(repr(controller), 10000),
            "controller_action_mapping": _jsonable(getattr(controller, "action_mapping", None)),
            "tcp": tcp,
            "pose_entities": entities,
            "entity_aliases": ManiSkillDynamicRobot._entity_aliases(entities),
            "reset_info": _jsonable(reset_info),
            "official_evaluation": evaluation,
            "environment_source": _source_of(cls, 24000),
            "evaluate_source": _source_of(getattr(cls, "evaluate", None), 10000),
        }
    finally:
        adapter.close()


def extract_task_program(code: str, *, required: bool = True) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == TASK_PROGRAM_NAME for target in targets):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value.strip()
    if required:
        raise ValueError(f"Generated source adapter must define {TASK_PROGRAM_NAME} as a string.")
    return ""


def validate_dynamic_adapter(code: str, *, task_program: str, require_program_constant: bool) -> None:
    validate_generated_adapter_module(code)
    safe, reason = is_safe(task_program)
    if not safe:
        raise ValueError(f"Generated high-level program is unsafe: {reason}")
    program_tree = ast.parse(task_program)
    if not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "ret_val" for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        ))
        for node in program_tree.body
    ):
        raise ValueError("Generated high-level program must assign ret_val.")
    program_methods = _program_robot_methods(program_tree)
    if not program_methods:
        raise ValueError("Generated high-level program must call at least one public robot skill.")
    if any(name.startswith("_") for name in program_methods):
        raise ValueError("High-level program may call only public robot skills.")

    tree = ast.parse(code)
    methods = {
        node.name: node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(program_methods - set(methods))
    if missing:
        raise ValueError("Generated adapter does not implement program skills: " + ", ".join(missing))
    protected = sorted(_PROTECTED_METHODS & set(methods))
    if protected:
        raise ValueError("Generated adapter overrides protected runtime methods: " + ", ".join(protected))

    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    dangerous = sorted(name for name in calls if name in _DANGEROUS_CALLS or name.startswith("set_"))
    if dangerous:
        raise ValueError("Generated adapter calls simulator-state mutation methods: " + ", ".join(dangerous))
    inherited_action_helpers = {
        name for name in _TRUSTED_ACTION_HELPERS if name in calls and name not in methods
    }
    if "_step" not in calls and not inherited_action_helpers:
        raise ValueError(
            "Generated adapter must execute actions through self._step(action) or a trusted "
            "inherited action helper."
        )
    if "_early_stop" not in calls and not inherited_action_helpers:
        raise ValueError(
            "Generated adapter must check self._early_stop() directly or use a trusted "
            "inherited action helper that checks it."
        )
    if not (_STATE_CALLS & calls):
        accepted = ", ".join(f"self.{name}(...)" for name in sorted(_STATE_CALLS))
        raise ValueError(
            "Generated adapter must read physical state before deciding actions. "
            f"Call at least one documented measurement API: {accepted}. "
            "Store the measured value and use it in a runtime branch or action calculation; "
            "direct self.env attribute reads do not satisfy this contract."
        )
    if not any(isinstance(node, ast.If) for node in ast.walk(tree)):
        raise ValueError("Generated adapter must contain a measured runtime branch.")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets: Sequence[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
            if any(isinstance(target, ast.Attribute) and target.attr in {"pose", "success"} for target in targets):
                raise ValueError("Generated adapter may not assign simulator pose or success fields.")
    if require_program_constant and extract_task_program(code, required=True) != task_program.strip():
        raise ValueError("Source module TASK_PROGRAM extraction changed unexpectedly.")
    embedded = extract_task_program(code, required=False)
    if not require_program_constant and embedded and embedded != task_program.strip():
        raise ValueError("Target adapter may not modify the frozen source TASK_PROGRAM.")


class _RemoveDocstrings(ast.NodeTransformer):
    """Normalize formatting-only changes before comparing generated modules."""

    def _visit_body(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _visit_body
    visit_ClassDef = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body


def _semantic_code_sha256(code: str) -> str:
    tree = _RemoveDocstrings().visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return _sha256_text(normalized)


def _concise_execution_error(message: str) -> str:
    lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "unknown execution error")[:1200]


class _RecordedEnv:
    """Record actual simulator steps independently of generated logging code."""

    def __init__(self, env: Any, max_steps: int) -> None:
        self._env = env
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        self.trace: list[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def snapshot(self) -> Dict[str, Any]:
        base = getattr(self._env, "unwrapped", self._env)
        return {
            "step": self.steps,
            "tcp": _read_tcp(getattr(base, "agent", None)),
            "entities": {
                name: _jsonable(actor.pose.p) for name, actor in iter_pose_entities(base)
            },
            "official_evaluation": _safe_evaluate(base),
        }

    def step(self, action: Any) -> Any:
        if self.done or self.steps >= self.max_steps:
            raise RuntimeError("Episode ended; no more actions may be executed.")
        result = self._env.step(action)
        self.steps += 1
        self.terminated = _scalar_bool(result[2])
        self.truncated = _scalar_bool(result[3]) or self.steps >= self.max_steps
        self.done = self.terminated or self.truncated
        if self.steps == 1 or self.steps % 10 == 0 or self.done:
            self.trace.append({**self.snapshot(), "action": _jsonable(action)})
        if self.done:
            evaluation = _safe_evaluate(getattr(self._env, "unwrapped", self._env))
            if _scalar_bool(evaluation.get("success", False)):
                raise _OfficialSuccessReached
        return result


class _OfficialSuccessReached(RuntimeError):
    """Stop a policy cleanly when ManiSkill terminates on official success."""


def run_dynamic_trial(
    *,
    spec: DynamicMigrationSpec,
    robot_uid: str,
    control_mode: str,
    program: str,
    adapter_path: Path,
    source_entrypoint: str = "build_robot",
) -> Dict[str, Any]:
    adapter = ManiSkillEnvAdapter(
        spec.env_id,
        robot_uid=robot_uid,
        obs_mode=spec.obs_mode,
        control_mode=control_mode,
        sim_backend=spec.sim_backend,
        render_backend=spec.render_backend,
        max_episode_steps=spec.max_episode_steps,
        reward_mode="sparse",
    )
    robot = None
    recorded = None
    try:
        env = adapter.make()
        _, reset_info = adapter.reset(seed=spec.seed)
        recorded = _RecordedEnv(env, spec.max_episode_steps)
        recorded.trace.append(recorded.snapshot())
        module = _load_module(adapter_path)
        try:
            if source_entrypoint == "run":
                run = module.run
                parameters = inspect.signature(run).parameters
                ret_val = run(recorded, seed=spec.seed) if "seed" in parameters else run(recorded)
                code_ok, code_message = True, "source run(env) completed"
            else:
                build_robot = getattr(module, "build_robot", None)
                if not callable(build_robot):
                    raise ValueError("Generated module must define callable build_robot(...).")
                robot = build_robot(recorded, control_mode=control_mode, robot_uid=robot_uid)
                scene = ManiSkillSceneAdapter()
                code_ok, code_message, locals_dict = execute_lmp(
                    program,
                    {"scene": scene, "robot": robot},
                    verbose=False,
                )
                ret_val = locals_dict.get("ret_val")
        except _OfficialSuccessReached:
            ret_val = True
            code_ok = True
            code_message = "official task success terminated the episode"
        official = _safe_evaluate(getattr(env, "unwrapped", env))
        official_success = _scalar_bool(official.get("success", False))
        returned_success = bool(code_ok and (source_entrypoint == "run" or _success_from_ret_val(ret_val)))
        success = bool(returned_success and official_success and recorded.steps > 0)
        execution_error = "" if code_ok else code_message
        if success:
            message = (
                "source actions reached official success"
                if source_entrypoint == "run"
                else "adapter returned success and official evaluate() returned success"
            )
        elif not code_ok:
            summary = _concise_execution_error(code_message)
            message = (
                f"Task program failed before the first simulator action: {summary}"
                if recorded.steps == 0
                else summary
            )
        elif recorded.steps == 0:
            message = "No simulator actions executed; source or target was not tested."
        elif source_entrypoint == "run":
            message = "source actions did not reach official environment success"
        elif returned_success and not official_success:
            message = "adapter returned success but official evaluate() success is false"
        else:
            message = _last_failure_message(robot) or "task program returned failure"
        snapshot = _safe_snapshot(robot) if robot is not None else recorded.snapshot()
        result = {
            "success": success,
            "message": message,
            "env_id": spec.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "seed": spec.seed,
            "max_episode_steps": spec.max_episode_steps,
            "code_ok": bool(code_ok),
            "execution_error": execution_error,
            "ret_val": _jsonable(ret_val),
            "official_success": bool(official_success),
            "official_evaluation": official,
            "reset_info": _jsonable(reset_info),
            "execution_log": _jsonable(robot.execution_log()) if robot is not None else [],
            "runtime_snapshot": snapshot,
            "action_steps": recorded.steps,
            "state_trace": [*recorded.trace, recorded.snapshot()],
            "terminated": recorded.terminated,
            "truncated": recorded.truncated,
        }
        if robot is not None:
            cls = type(robot)
            policy_sources = []
            for policy_cls in cls.__mro__:
                if policy_cls in {ManiSkillDynamicRobot, object}:
                    continue
                source = _source_of(policy_cls, 14000)
                if source:
                    policy_sources.append(source)
            result["implementation_context"] = {
                "class_path": f"{cls.__module__}.{cls.__name__}",
                "policy_source": "\n\n".join(policy_sources),
                "inherited_action_helpers": {
                    "_move_towards": "closed-loop TCP delta motion; checks _early_stop and calls _step",
                    "_repeat_action": "bounded repeated action; checks _early_stop and calls _step",
                    "_make_action": "maps xyz plus gripper command into the current action space",
                },
            }
        result["failure_diagnosis"] = _dynamic_diagnosis(result)
        return result
    except Exception as exc:
        result = {
            "success": False,
            "message": repr(exc),
            "execution_error": repr(exc),
            "env_id": spec.env_id,
            "robot_uid": robot_uid,
            "control_mode": control_mode,
            "seed": spec.seed,
            "failure_layer": "runtime_setup",
        }
        if recorded is not None:
            result.update(
                action_steps=recorded.steps,
                state_trace=recorded.trace,
                terminated=recorded.terminated,
                truncated=recorded.truncated,
            )
        result["failure_diagnosis"] = _dynamic_diagnosis(result)
        return result
    finally:
        if robot is not None and hasattr(robot, "close"):
            robot.close()
        adapter.close()


def run_dynamic_agent_migration(
    *,
    spec: DynamicMigrationSpec,
    run_dir: Path,
    dry_run: bool = False,
    source_actions: SourceActions | None = None,
    progress: Callable[[str], None] | None = None,
    enforce_environment: bool = False,
) -> Dict[str, Any]:
    """Validate supplied source actions (or synthesize them), then migrate."""

    report = progress or (lambda message: None)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = run_dir / "dynamic_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    if source_actions is not None:
        (artifacts / "source_adapter.py").write_text(source_actions.code, encoding="utf-8")
        (artifacts / "task_program.py").write_text(source_actions.program + "\n", encoding="utf-8")
    runtime_environment = capture_experiment_environment(asdict(spec))
    runtime_environment_path = run_dir / "runtime_environment.json"
    _write_json(runtime_environment_path, runtime_environment)
    manifest_path = run_dir / "case_manifest.json"
    manifest: Dict[str, Any] = {
        "schema": "dynamic_migration_manifest.v1",
        **asdict(spec),
        "registered_case": False,
        "success_authority": "env.unwrapped.evaluate()['success']",
        "llm_may_modify": ["source task program before source success", "source adapter", "target adapter"],
        "llm_may_not_modify": ["environment", "controller", "simulator state", "official success"],
        "runtime_environment": str(runtime_environment_path),
        "environment_contract_sha256": runtime_environment["contract_sha256"],
    }
    if source_actions is not None:
        manifest.update(
            source_input={"path": source_actions.path, "sha256": _sha256_text(source_actions.code),
                          "entrypoint": source_actions.entrypoint},
            llm_may_modify=["target adapter"],
        )
    if dry_run:
        manifest["planned_steps"] = [
            "discover source and target environments",
            ("validate the supplied source actions without LLM edits" if source_actions is not None
             else "generate and validate source program plus source adapter"),
            "freeze source program after official source success",
            "generate, execute and repair target adapter",
        ]
        _write_json(manifest_path, manifest)
        payload = {
            "schema": "dynamic_agent_migration_result.v1",
            "status": "dry_run_planned",
            "success": None,
            "manifest": str(manifest_path),
            "runtime_environment": str(runtime_environment_path),
            "run_dir": str(run_dir),
            "spec": asdict(spec),
            "source_cycles": [],
            "target_cycles": [],
        }
        _write_dynamic_summary(run_dir, payload)
        return payload

    if enforce_environment and not runtime_environment["validation"]["ok"]:
        mismatches = runtime_environment["validation"]["mismatches"]
        manifest["outcome"] = {
            "status": "environment_contract_mismatch",
            "success": False,
            "mismatches": mismatches,
        }
        _write_json(manifest_path, manifest)
        payload = {
            "schema": "dynamic_agent_migration_result.v1",
            "status": "environment_contract_mismatch",
            "success": False,
            "message": "runtime does not match experiment_config.json",
            "manifest": str(manifest_path),
            "runtime_environment": str(runtime_environment_path),
            "run_dir": str(run_dir),
            "spec": asdict(spec),
            "source_cycles": [],
            "target_cycles": [],
        }
        _write_dynamic_summary(run_dir, payload)
        return payload

    report("Inspecting source and target simulator interfaces...")
    try:
        source_observation = discover_environment(spec, spec.source_robot, spec.source_control_mode)
        target_observation = discover_environment(spec, spec.target_robot, spec.target_control_mode)
    except Exception as exc:
        manifest["discovery_error"] = repr(exc)
        _write_json(manifest_path, manifest)
        payload = {
            "schema": "dynamic_agent_migration_result.v1",
            "status": "environment_discovery_failed",
            "success": False,
            "message": repr(exc),
            "manifest": str(manifest_path),
            "runtime_environment": str(runtime_environment_path),
            "run_dir": str(run_dir),
            "spec": asdict(spec),
            "source_cycles": [],
            "target_cycles": [],
        }
        _write_dynamic_summary(run_dir, payload)
        return payload

    manifest["source_observation"] = source_observation
    manifest["target_observation"] = target_observation
    source_observation_path = run_dir / "source_observation.json"
    target_observation_path = run_dir / "target_observation.json"
    _write_json(source_observation_path, source_observation)
    _write_json(target_observation_path, target_observation)
    manifest["observation_artifacts"] = {
        "source": str(source_observation_path),
        "target": str(target_observation_path),
    }
    _write_json(manifest_path, manifest)

    source_cycles = []
    source_code = ""
    source_result: Dict[str, Any] = {}
    frozen_program = ""
    if source_actions is not None:
        report("Checking the supplied source actions...")
        source_code = source_actions.code
        source_result = run_dynamic_trial(
            spec=spec, robot_uid=spec.source_robot, control_mode=spec.source_control_mode,
            program=source_actions.program, adapter_path=artifacts / "source_adapter.py",
            source_entrypoint=source_actions.entrypoint,
        )
        _write_json(run_dir / "source_trial.json", source_result)
        source_cycles.append({"cycle": 0, "used_llm": False, "origin": "user_supplied",
                              "adapter_sha256": _sha256_text(source_code), "trial": source_result})
        if source_result.get("success"):
            frozen_program = source_actions.program
        report(f"source_success = {bool(source_result.get('success'))}")

    for cycle in range(1, (0 if source_actions is not None else spec.max_source_cycles) + 1):
        cycle_dir = run_dir / f"source_cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        system_prompt = _dynamic_system_prompt(source=True)
        user_prompt = _source_prompt(spec, source_observation, source_code, source_result)
        system_prompt_path = cycle_dir / "system_prompt.txt"
        user_prompt_path = cycle_dir / "user_prompt.txt"
        system_prompt_path.write_text(system_prompt, encoding="utf-8")
        user_prompt_path.write_text(user_prompt, encoding="utf-8")
        generated = gen_text(
            system=system_prompt,
            prompt=user_prompt,
            fallback_text="",
        )
        record: Dict[str, Any] = {
            "cycle": cycle,
            "used_llm": generated.used_llm,
            "model": generated.model,
            "usage": generated.usage,
            "llm_reason": generated.reason,
            "system_prompt": str(system_prompt_path),
            "user_prompt": str(user_prompt_path),
            "prompt_sha256": _sha256_text(system_prompt + "\n" + user_prompt),
        }
        (cycle_dir / "llm_response.txt").write_text(generated.raw_text or generated.text, encoding="utf-8")
        if not generated.used_llm:
            record["error"] = generated.reason or "LLM unavailable"
            _write_json(cycle_dir / "cycle_record.json", record)
            source_cycles.append(record)
            break
        try:
            candidate = extract_python_module(generated.text)
            candidate_path = cycle_dir / "source_adapter.py"
            candidate_path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
            record.update(
                adapter=str(candidate_path),
                adapter_sha256=_sha256_text(candidate.rstrip() + "\n"),
            )
            program = extract_task_program(candidate)
            validate_dynamic_adapter(candidate, task_program=program, require_program_constant=True)
            program_path = cycle_dir / "task_program.py"
            program_path.write_text(program.rstrip() + "\n", encoding="utf-8")
            trial = run_dynamic_trial(
                spec=spec,
                robot_uid=spec.source_robot,
                control_mode=spec.source_control_mode,
                program=program,
                adapter_path=candidate_path,
            )
            trial_path = cycle_dir / "trial_result.json"
            _write_json(trial_path, trial)
            record.update(
                valid=True,
                program=str(program_path),
                program_sha256=_sha256_text(program.rstrip() + "\n"),
                trial_result=str(trial_path),
                trial=trial,
            )
            source_code, source_result = candidate, trial
            if trial.get("success"):
                frozen_program = program
                (artifacts / "source_adapter.py").write_text(candidate.rstrip() + "\n", encoding="utf-8")
                (artifacts / "task_program.py").write_text(program.rstrip() + "\n", encoding="utf-8")
        except Exception as exc:
            record.update(valid=False, error=repr(exc))
            source_result = {"success": False, "message": repr(exc)}
        _write_json(cycle_dir / "cycle_record.json", record)
        source_cycles.append(record)
        if frozen_program:
            break

    if not frozen_program:
        source_status = "source_validation_failed" if source_actions is not None else "source_budget_exhausted"
        manifest["outcome"] = {
            "status": source_status,
            "success": False,
            "source_cycles": len(source_cycles),
        }
        _write_json(manifest_path, manifest)
        payload = {
            "schema": "dynamic_agent_migration_result.v1",
            "status": source_status,
            "success": False,
            "message": "source program and adapter did not reach official success",
            "manifest": str(manifest_path),
            "runtime_environment": str(runtime_environment_path),
            "run_dir": str(run_dir),
            "spec": asdict(spec),
            "source_cycles": source_cycles,
            "target_cycles": [],
        }
        _write_dynamic_summary(run_dir, payload)
        return payload

    target_cycles = []
    target_code = ""
    target_result: Dict[str, Any] = {}
    target_success = False
    target_status = "target_budget_exhausted"
    for cycle in range(1, spec.max_target_cycles + 1):
        report(f"Target round {cycle}/{spec.max_target_cycles}: requesting LLM code...")
        cycle_dir = run_dir / f"target_cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        system_prompt = _dynamic_system_prompt(source=False)
        user_prompt = _target_prompt(
            spec,
            source_observation,
            target_observation,
            frozen_program,
            source_code,
            source_result,
            target_code,
            target_result,
        )
        system_prompt_path = cycle_dir / "system_prompt.txt"
        user_prompt_path = cycle_dir / "user_prompt.txt"
        system_prompt_path.write_text(system_prompt, encoding="utf-8")
        user_prompt_path.write_text(user_prompt, encoding="utf-8")
        try:
            generated = gen_text(system=system_prompt, prompt=user_prompt, fallback_text="")
        except Exception as exc:
            target_status = "llm_request_failed"
            target_result = {"success": False, "message": f"LLM request failed: {type(exc).__name__}"}
            record = {"cycle": cycle, "valid": False, "error": target_result["message"],
                      "system_prompt": str(system_prompt_path), "user_prompt": str(user_prompt_path)}
            _write_json(cycle_dir / "cycle_record.json", record)
            target_cycles.append(record)
            break
        record = {
            "cycle": cycle,
            "used_llm": generated.used_llm,
            "model": generated.model,
            "usage": generated.usage,
            "llm_reason": generated.reason,
            "system_prompt": str(system_prompt_path),
            "user_prompt": str(user_prompt_path),
            "prompt_sha256": _sha256_text(system_prompt + "\n" + user_prompt),
        }
        (cycle_dir / "llm_response.txt").write_text(generated.raw_text or generated.text, encoding="utf-8")
        if not generated.used_llm:
            target_status = "llm_unavailable"
            record["error"] = generated.reason or "LLM unavailable"
            target_result = {"success": False, "message": record["error"]}
            _write_json(cycle_dir / "cycle_record.json", record)
            target_cycles.append(record)
            break
        try:
            candidate = extract_python_module(generated.text)
            candidate_path = cycle_dir / "target_adapter.py"
            candidate_path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
            record.update(
                adapter=str(candidate_path),
                adapter_sha256=_sha256_text(candidate.rstrip() + "\n"),
                semantic_sha256=_semantic_code_sha256(candidate),
            )
            previous_code = target_code
            if previous_code and _semantic_code_sha256(candidate) == _semantic_code_sha256(previous_code):
                raise ValueError(
                    "Generated target adapter is semantically unchanged from the failed adapter; "
                    "formatting, comments, or docstrings do not count as a repair."
                )
            record["changed_from_previous"] = bool(previous_code)
            target_code = candidate
            validate_dynamic_adapter(candidate, task_program=frozen_program, require_program_constant=False)
            report(f"Target round {cycle}: running generated code ({generated.model})...")
            trial = run_dynamic_trial(
                spec=spec,
                robot_uid=spec.target_robot,
                control_mode=spec.target_control_mode,
                program=frozen_program,
                adapter_path=candidate_path,
            )
            trial_path = cycle_dir / "trial_result.json"
            _write_json(trial_path, trial)
            record.update(valid=True, trial_result=str(trial_path), trial=trial)
            target_code, target_result = candidate, trial
            target_result.pop("implementation_context", None)
            (artifacts / "target_adapter.py").write_text(candidate.rstrip() + "\n", encoding="utf-8")
            if trial.get("success"):
                target_success = True
        except Exception as exc:
            record.update(valid=False, error=repr(exc))
            target_result = {"success": False, "message": repr(exc)}
        _write_json(cycle_dir / "cycle_record.json", record)
        target_cycles.append(record)
        report(f"Target round {cycle}: success={target_success}; {target_result.get('message', '')[:180]}")
        if target_success:
            break

    artifact_hashes = {
        "task_program": _sha256_file(artifacts / "task_program.py"),
        "source_adapter": _sha256_file(artifacts / "source_adapter.py"),
        "target_adapter": _sha256_file(artifacts / "target_adapter.py"),
    }
    manifest["outcome"] = {
        "status": "success" if target_success else target_status,
        "success": target_success,
        "source_cycles": len(source_cycles),
        "target_cycles": len(target_cycles),
        "artifact_sha256": artifact_hashes,
    }
    _write_json(manifest_path, manifest)
    payload = {
        "schema": "dynamic_agent_migration_result.v1",
        "status": "success" if target_success else target_status,
        "success": target_success,
        "message": (
            "target adapter reached official environment success"
            if target_success
            else target_result.get("message")
            or "target adapter did not reach official success within the generation budget"
        ),
        "manifest": str(manifest_path),
        "runtime_environment": str(runtime_environment_path),
        "run_dir": str(run_dir),
        "spec": asdict(spec),
        "frozen_program": str(artifacts / "task_program.py"),
        "source_adapter": str(artifacts / "source_adapter.py"),
        "target_adapter": str(artifacts / "target_adapter.py") if (artifacts / "target_adapter.py").is_file() else "",
        "artifact_sha256": artifact_hashes,
        "source_cycles": source_cycles,
        "target_cycles": target_cycles,
        "source_result": source_result,
        "target_result": target_result,
    }
    _write_dynamic_summary(run_dir, payload)
    return payload


def _dynamic_system_prompt(*, source: bool) -> str:
    role = "source task solver" if source else "target embodiment adapter"
    return (
        f"You write a complete Python module for a ManiSkill {role}. "
        "Return only Python code. Use measured state and bounded self._step(action) loops. "
        "Read state through the documented ManiSkillDynamicRobot measurement methods, and "
        "use those measurements to choose or stop actions. "
        "Never mutate simulator state, actor poses, task success, or controller internals."
    )


def _measurement_api_contract() -> str:
    return """Documented physical-state API on ManiSkillDynamicRobot:
- self._snapshot(): complete TCP, entity poses, official evaluation and episode flags.
- self._tcp_pos(): current tool-center-point position as xyz.
- self._entity_pos(name), self._actor_pos(name), self._region_pos(name): current xyz.
- self._entity_quat(name): current entity orientation quaternion.
- self._is_grasping_entity(name): current grasp detector.
- self._official_evaluation() and self._official_success(): official task state.
At least one method above must be called inside the generated skill. Store its return value
and use that measured value in an if/loop condition or to compute an action. Do not read
self.env internals as a substitute. Re-read measurements after actions to close the loop."""


def _source_prompt(
    spec: DynamicMigrationSpec,
    observation: Mapping[str, Any],
    current_code: str,
    latest_result: Mapping[str, Any],
) -> str:
    return f"""Create the source-side implementation for an unregistered ManiSkill task.

Task request:
{_pretty(asdict(spec))}

Observed source environment:
{_pretty(_prompt_observation(observation))}

Required module contract:
- Import ManiSkillDynamicRobot from maniskill_backend.dynamic_adapter.
- Define TASK_PROGRAM as a Python string. It is the high-level program that will be frozen after source success.
- TASK_PROGRAM may use scene.get_object(name), scene.get_region(name), public robot skills, and must assign ret_val.
- Implement every public robot skill called by TASK_PROGRAM in a subclass of ManiSkillDynamicRobot.
- Define build_robot(env, *, control_mode, robot_uid).
- Use the observed action layout. Override _make_action when the fallback layout is not correct.
- Read physical state, check _early_stop(), execute bounded self._step(action) loops, and return truthful booleans.
- The harness independently checks env.unwrapped.evaluate()['success'].
- Do not call set_pose, set_state, reset, or alter success/controller/simulator fields.

{_measurement_api_contract()}

Minimal closed-loop pattern (adapt entity names and motion to the observed task):
```python
for _ in range(bounded_steps):
    if self._early_stop():
        return False
    tcp = self._tcp_pos()
    target = self._entity_pos("observed_entity_name")
    if np.linalg.norm(target - tcp) < tolerance:
        break
    self._step(self._make_action(bounded_delta, gripper=command))
```

Current failed source module, if any:
```python
{_trim(current_code, 18000)}
```

Latest source execution result, if any:
{_pretty(_prompt_trial(latest_result))}
"""


def _target_prompt(
    spec: DynamicMigrationSpec,
    source_observation: Mapping[str, Any],
    target_observation: Mapping[str, Any],
    program: str,
    source_code: str,
    source_result: Mapping[str, Any],
    current_code: str,
    latest_result: Mapping[str, Any],
) -> str:
    return f"""Migrate the frozen source program to the target robot by writing only the target adapter.

Task request:
{_pretty(asdict(spec))}

Frozen high-level program. Do not change it:
```python
{program}
```

Source environment observation:
{_pretty(_prompt_observation(source_observation))}

Target environment observation:
{_pretty(_prompt_observation(target_observation))}

Successful source adapter:
```python
{_trim(source_code, 18000)}
```

Source action-policy reference. It may omit module-level imports; do not paste it verbatim:
{json.dumps(source_result.get('implementation_context') or {}, ensure_ascii=False)}

Source success evidence:
{_pretty(_prompt_trial(source_result))}

Generic runtime API available to your adapter (contains no task policy):
{_source_of(ManiSkillDynamicRobot, 16000)}

Required target module contract:
- Import ManiSkillDynamicRobot from maniskill_backend.dynamic_adapter.
- Return a complete, self-contained module: every annotation and runtime name must be imported or defined.
- Do not copy the source class verbatim and do not include TASK_PROGRAM or source metadata constants.
- Implement every public robot skill used by the frozen program.
- If the source defines run(env), implement run_source_actions() with the same action intent.
- Define build_robot(env, *, control_mode, robot_uid).
- Adapt action layout, reach, gripper/base channels, contact geometry, timing and state branches to the observed target.
- Execute actions through self._step(action), or inherited _move_towards/_repeat_action helpers.
- Read measured state and use bounded loops; inherited action helpers already check _early_stop().
- Do not modify TASK_PROGRAM, simulator state, object poses, controller internals, or official success.
- Returning True is insufficient: the harness independently checks env.unwrapped.evaluate()['success'].

{_measurement_api_contract()}

Current failed target adapter, if any:
```python
{_trim(current_code, 18000)}
```

Latest target execution, including measured state_trace sampled during execution:
{_pretty(_prompt_trial(latest_result))}

Repair requirement for this round:
- Fix the concrete failure shown above before tuning later motion stages.
- Make a semantic code change; formatting, comments, or docstrings alone are rejected.
- If action_steps is 0, remove the reported pre-action exception so the candidate reaches self._step(action).
"""


def _prompt_observation(observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep the robot contract and task state, not renderer bookkeeping."""

    keys = (
        "env_id", "robot_uid", "control_mode", "environment_class", "environment_doc",
        "supported_robots", "action_space", "controller_summary", "controller_action_mapping",
        "tcp", "entity_aliases", "reset_info", "official_evaluation", "evaluate_source",
    )
    compact = {key: observation.get(key) for key in keys if key in observation}
    compact["pose_entities"] = _prompt_entities(observation.get("pose_entities"))
    return compact


def _prompt_trial(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Bound feedback size while retaining physical counterexamples."""

    keys = (
        "success", "message", "code_ok", "ret_val", "official_success",
        "official_evaluation", "action_steps", "terminated", "truncated",
        "execution_error", "failure_layer", "failure_diagnosis",
    )
    compact = {key: result.get(key) for key in keys if key in result}
    snapshot = result.get("runtime_snapshot")
    if isinstance(snapshot, Mapping):
        compact["runtime_snapshot"] = _prompt_snapshot(snapshot)
    execution_log = result.get("execution_log")
    if isinstance(execution_log, list):
        compact["execution_log_tail"] = execution_log[-8:]
    trace = result.get("state_trace")
    if isinstance(trace, list) and trace:
        if len(trace) <= 12:
            selected = trace
        else:
            indices = sorted({round(index * (len(trace) - 1) / 11) for index in range(12)})
            selected = [trace[index] for index in indices]
        compact["state_trace_sample"] = [
            _prompt_snapshot(item) if isinstance(item, Mapping) else item for item in selected
        ]
    return compact


def _prompt_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    compact = {
        key: snapshot.get(key)
        for key in ("step", "tcp", "action", "official_evaluation", "terminated", "truncated")
        if key in snapshot
    }
    if "entities" in snapshot:
        compact["entities"] = _prompt_entities(snapshot.get("entities"))
    if "entity_aliases" in snapshot:
        compact["entity_aliases"] = snapshot.get("entity_aliases")
    return compact


def _prompt_entities(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    relevant = {
        str(name): item
        for name, item in value.items()
        if not str(name).startswith("segmentation_id_map")
    }
    return dict(list(relevant.items())[:40])


def _program_robot_methods(tree: ast.AST) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "robot"
    }


def _dynamic_diagnosis(result: Mapping[str, Any]) -> Dict[str, Any]:
    message = str(result.get("message") or "")
    execution_error = str(result.get("execution_error") or "")
    if result.get("failure_layer") == "runtime_setup":
        return {
            "layer": "runtime_setup",
            "reason": "environment_or_generated_module_error",
            "repair_hint": "Fix environment compatibility, module structure, or the observed action mapping before task motion.",
        }
    if not result.get("code_ok", True):
        if int(result.get("action_steps") or 0) == 0 and "task entity" in execution_error.lower():
            return {
                "layer": "interface_contract",
                "reason": "semantic_entity_name_mismatch",
                "evidence": _concise_execution_error(execution_error),
                "repair_hint": (
                    "Use an exact pose entity from the target observation or an unambiguous "
                    "runtime semantic alias before changing motion parameters."
                ),
            }
        return {
            "layer": "program",
            "reason": "pre_action_execution_error" if int(result.get("action_steps") or 0) == 0 else "high_level_program_execution_error",
            "evidence": _concise_execution_error(execution_error or message),
            "repair_hint": (
                "Fix the reported exception in the generated adapter before changing motion "
                "parameters; the next candidate must reach self._step(action)."
            ),
        }
    if result.get("truncated"):
        return {
            "layer": "skill_adapter",
            "reason": "episode_budget_exhausted",
            "repair_hint": "Use measured progress guards and stop spending steps on an unreachable or stalled waypoint.",
        }
    if result.get("ret_val") and not result.get("official_success"):
        return {
            "layer": "task_outcome",
            "reason": "adapter_claimed_success_without_official_success",
            "repair_hint": "Read the official evaluation fields and continue physical execution until the real task condition is satisfied.",
        }
    return {
        "layer": "skill_adapter",
        "reason": "official_task_outcome_not_reached",
        "repair_hint": "Compare the runtime snapshot and official evaluation with the intended stage, then change the failed action sequence or geometry.",
        "message": message,
    }


def _read_tcp(agent: Any) -> Any:
    try:
        tcp_pose = getattr(agent, "tcp_pose", None)
        if tcp_pose is not None:
            return np.round(_to_numpy(tcp_pose.p), 6).tolist()
        tcp = getattr(agent, "tcp", None)
        if tcp is not None and getattr(tcp, "pose", None) is not None:
            return np.round(_to_numpy(tcp.pose.p), 6).tolist()
    except Exception:
        return None
    return None


def _space_summary(space: Any) -> Dict[str, Any]:
    if space is None:
        return {"repr": "None"}
    summary: Dict[str, Any] = {
        "repr": _trim(repr(space), 4000),
        "shape": list(getattr(space, "shape", ())),
        "dtype": str(getattr(space, "dtype", "")),
    }
    for name in ("low", "high"):
        value = getattr(space, name, None)
        if value is not None:
            try:
                array = np.asarray(value, dtype=np.float64)
                summary[name] = {
                    "min": float(np.nanmin(array)),
                    "max": float(np.nanmax(array)),
                    "values": array.reshape(-1).tolist() if array.size <= 32 else None,
                }
            except Exception:
                summary[name] = _trim(repr(value), 1000)
    return summary


def _safe_evaluate(base: Any) -> Dict[str, Any]:
    try:
        return _jsonable(dict(base.evaluate() or {}))
    except Exception as exc:
        return {"evaluation_error": repr(exc)}


def _safe_snapshot(robot: Any) -> Dict[str, Any]:
    try:
        return _jsonable(robot._snapshot())
    except Exception as exc:
        return {"snapshot_error": repr(exc)}


def _last_failure_message(robot: Any) -> str:
    try:
        for item in reversed(robot.execution_log()):
            if not item.get("ok") and item.get("message"):
                return str(item["message"])
    except Exception:
        pass
    return ""


def _load_module(path: Path) -> Any:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    name = f"dynamic_adapter_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import generated adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_of(value: Any, limit: int) -> str:
    if value is None:
        return ""
    try:
        return _trim(inspect.getsource(value), limit)
    except Exception:
        return ""


def _trim(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <trimmed {len(text) - limit} chars>"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pretty(value: Any, limit: int = 30000) -> str:
    return _trim(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, default=repr), limit)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        array = _to_numpy(value)
        if array.size == 1:
            return array.reshape(-1)[0].item()
        return array.tolist()
    except Exception:
        return repr(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")


def _write_dynamic_summary(run_dir: Path, payload: Mapping[str, Any]) -> None:
    _write_json(run_dir / "dynamic_summary.json", payload)
    lines = [
        "# Dynamic Migration",
        "",
        f"- env: `{(payload.get('spec') or {}).get('env_id')}`",
        f"- source: `{(payload.get('spec') or {}).get('source_robot')}`",
        f"- target: `{(payload.get('spec') or {}).get('target_robot')}`",
        f"- status: `{payload.get('status')}`",
        f"- success: `{payload.get('success')}`",
        f"- manifest: `{payload.get('manifest')}`",
        "",
        "## Source cycles",
        "",
        "| cycle | model | valid | official success | action steps | adapter sha | message |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for item in payload.get("source_cycles") or []:
        trial = item.get("trial") or {}
        lines.append(
            f"| {item.get('cycle')} | {item.get('model')} | {item.get('valid')} | "
            f"{trial.get('official_success')} | {trial.get('action_steps')} | "
            f"{str(item.get('adapter_sha256') or '')[:12]} | "
            f"{str(trial.get('message') or item.get('error') or '').replace('|', '/') } |"
        )
    lines.extend(
        [
            "",
            "## Target cycles",
            "",
            "| cycle | model | valid | official success | action steps | adapter sha | message |",
            "|---:|---|---|---|---:|---|---|",
        ]
    )
    for item in payload.get("target_cycles") or []:
        trial = item.get("trial") or {}
        lines.append(
            f"| {item.get('cycle')} | {item.get('model')} | {item.get('valid')} | "
            f"{trial.get('official_success')} | {trial.get('action_steps')} | "
            f"{str(item.get('adapter_sha256') or '')[:12]} | "
            f"{str(trial.get('message') or item.get('error') or '').replace('|', '/') } |"
        )
    hashes = payload.get("artifact_sha256") or {}
    if hashes:
        lines.extend(
            [
                "",
                "## Accepted artifact hashes",
                "",
                f"- task program: `{hashes.get('task_program') or ''}`",
                f"- source adapter: `{hashes.get('source_adapter') or ''}`",
                f"- target adapter: `{hashes.get('target_adapter') or ''}`",
            ]
        )
    lines.append("")
    (run_dir / "dynamic_summary.md").write_text("\n".join(lines), encoding="utf-8")
