"""Discovery-first migration for ManiSkill tasks without registered cases."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from lmp.executor import execute_lmp, is_safe

from .code_validation import extract_python_module, validate_generated_adapter_module
from .dynamic_adapter import ManiSkillDynamicRobot, iter_pose_entities, physical_state_extras
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
    "_robot_root_quat",
    "_world_delta_to_action_delta",
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
_MAX_GENERATED_LINES_WITHOUT_ACTION = 2_000
_MAX_GENERATED_SECONDS_WITHOUT_ACTION = 5.0
_LOCAL_REPAIR_PROTECTED_METHODS = {
    "_make_action",
    "_validate_action_space",
    "_world_delta_to_action_delta",
}

_DIAGNOSIS_PROGRESS_RANK = {
    "environment_or_generated_module_error": 0,
    "pre_action_execution_error": 5,
    "semantic_entity_name_mismatch": 6,
    "high_level_program_execution_error": 8,
    "no_action_progress_loop": 10,
    "manipulated_object_left_workspace": 20,
    "contact_not_established": 40,
    "episode_budget_exhausted": 45,
    "official_task_outcome_not_reached": 60,
    "adapter_claimed_success_without_official_success": 65,
}


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
class RepairMethodPolicy:
    """Ablation policy for target-side generation and repair."""

    method_id: str
    structured_diagnosis: bool
    local_repair: bool
    preserve_incumbent: bool
    one_shot: bool = False


REPAIR_METHODS: Dict[str, RepairMethodPolicy] = {
    "one_shot": RepairMethodPolicy("one_shot", False, False, False, True),
    "raw_log": RepairMethodPolicy("raw_log", False, False, False),
    "diagnosis": RepairMethodPolicy("diagnosis", True, True, False),
    "ours": RepairMethodPolicy("ours", True, True, True),
}


def repair_method_policy(method_id: str) -> RepairMethodPolicy:
    try:
        return REPAIR_METHODS[method_id]
    except KeyError as exc:
        choices = ", ".join(REPAIR_METHODS)
        raise ValueError(f"Unknown repair method {method_id!r}; choose one of: {choices}.") from exc


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
                pose = getattr(actor, "pose", actor)
                entities[name] = {
                    "position": np.round(_to_numpy(pose.p), 6).tolist(),
                    "quaternion": np.round(_to_numpy(pose.q), 6).tolist(),
                }
            except Exception:
                continue
        evaluation = _safe_evaluate(base)
        controller = getattr(agent, "controller", None)
        cls = base.__class__
        return {
            **physical_state_extras(base),
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
            **physical_state_extras(base),
            "step": self.steps,
            "tcp": _read_tcp(getattr(base, "agent", None)),
            "entities": {
                name: _jsonable(getattr(actor, "pose", actor).p) for name, actor in iter_pose_entities(base)
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


class _OfficialSuccessReached(BaseException):
    """Stop a policy cleanly when ManiSkill terminates on official success."""


class _NoActionProgress(RuntimeError):
    """Stop generated policy loops that never advance the simulator."""


def _run_with_action_progress_watchdog(
    callback: Callable[[], Any],
    *,
    adapter_path: Path,
    action_counter: Callable[[], int],
) -> Any:
    """Interrupt generated-code loops that repeatedly run without an action."""

    watched_file = str(adapter_path.resolve())
    last_steps = int(action_counter())
    lines_without_action = 0
    last_progress = time.monotonic()
    previous_trace = sys.gettrace()

    def trace(frame: Any, event: str, arg: Any) -> Any:
        nonlocal last_steps, lines_without_action, last_progress
        if event != "line" or str(Path(frame.f_code.co_filename).resolve()) != watched_file:
            return trace
        steps = int(action_counter())
        if steps != last_steps:
            last_steps = steps
            lines_without_action = 0
            last_progress = time.monotonic()
            return trace
        lines_without_action += 1
        idle_seconds = time.monotonic() - last_progress
        if (
            lines_without_action > _MAX_GENERATED_LINES_WITHOUT_ACTION
            or idle_seconds > _MAX_GENERATED_SECONDS_WITHOUT_ACTION
        ):
            raise _NoActionProgress(
                "Generated adapter loop made no simulator progress: "
                f"action_steps stayed at {steps} for {lines_without_action} generated-code lines. "
                "Every loop iteration must execute self._step(action) or break/return."
            )
        return trace

    sys.settrace(trace)
    try:
        return callback()
    finally:
        sys.settrace(previous_trace)


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
                code_ok, code_message, locals_dict = _run_with_action_progress_watchdog(
                    lambda: execute_lmp(
                        program,
                        {"scene": scene, "robot": robot},
                        verbose=False,
                    ),
                    adapter_path=adapter_path,
                    action_counter=lambda: recorded.steps,
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
                    "_make_action": (
                        "maps normalized world-frame xyz into the robot-root controller frame "
                        "and fills the gripper channel"
                    ),
                    "_world_delta_to_action_delta": (
                        "rotates a normalized world xyz command into controller root coordinates"
                    ),
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
    source_provenance: Mapping[str, Any] | None = None,
    method_id: str = "ours",
) -> Dict[str, Any]:
    """Validate supplied source actions (or synthesize them), then migrate."""

    policy = repair_method_policy(method_id)
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
        "method_id": policy.method_id,
        "method_policy": asdict(policy),
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
                          "entrypoint": source_actions.entrypoint,
                          "provenance": _jsonable(source_provenance or {})},
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
        source_cycles.append({"cycle": 0, "used_llm": False,
                              "origin": "frozen_source" if source_provenance else "user_supplied",
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
            if not trial.get("success") and not isinstance(
                trial.get("failure_diagnosis"), Mapping
            ):
                trial["failure_diagnosis"] = _dynamic_diagnosis(trial)
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
    incumbent_code = ""
    incumbent_result: Dict[str, Any] = {}
    incumbent_quality: Dict[str, Any] | None = None
    incumbent_cycle: int | None = None
    latest_code = ""
    latest_result: Dict[str, Any] = {}
    seen_target_semantics: set[str] = set()
    target_success = False
    target_status = "target_budget_exhausted"
    target_cycle_budget = 1 if policy.one_shot else spec.max_target_cycles
    for cycle in range(1, target_cycle_budget + 1):
        report(f"Target round {cycle}/{target_cycle_budget}: requesting LLM code...")
        cycle_dir = run_dir / f"target_cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        system_prompt = _dynamic_system_prompt(source=False)
        reference_code = incumbent_code if policy.preserve_incumbent else latest_code
        reference_result = incumbent_result if policy.preserve_incumbent else latest_result
        repair_focus = reference_result
        repair_scope = (
            _repair_scope(repair_focus)
            if policy.structured_diagnosis
            else {"name": "raw_execution_feedback" if repair_focus else "initial_generation"}
        )
        user_prompt = _target_prompt(
            spec,
            source_observation,
            target_observation,
            frozen_program,
            source_code,
            source_result,
            reference_code,
            reference_result,
            latest_code=(
                latest_code
                if policy.preserve_incumbent and latest_code != incumbent_code
                else ""
            ),
            latest_result=(
                latest_result
                if policy.preserve_incumbent and latest_code != incumbent_code
                else {}
            ),
            repair_scope=repair_scope,
            policy=policy,
        )
        system_prompt_path = cycle_dir / "system_prompt.txt"
        user_prompt_path = cycle_dir / "user_prompt.txt"
        system_prompt_path.write_text(system_prompt, encoding="utf-8")
        user_prompt_path.write_text(user_prompt, encoding="utf-8")
        try:
            generated = gen_text(system=system_prompt, prompt=user_prompt, fallback_text="")
        except Exception as exc:
            target_status = "llm_request_failed"
            latest_result = {"success": False, "message": f"LLM request failed: {type(exc).__name__}"}
            record = {"cycle": cycle, "valid": False, "error": latest_result["message"],
                      "repair_scope": repair_scope,
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
            "repair_scope": repair_scope,
            "method_id": policy.method_id,
            "incumbent_cycle_before": incumbent_cycle,
        }
        (cycle_dir / "llm_response.txt").write_text(generated.raw_text or generated.text, encoding="utf-8")
        if not generated.used_llm:
            target_status = "llm_unavailable"
            record["error"] = generated.reason or "LLM unavailable"
            latest_result = {"success": False, "message": record["error"]}
            _write_json(cycle_dir / "cycle_record.json", record)
            target_cycles.append(record)
            break
        candidate = ""
        try:
            candidate = extract_python_module(generated.text)
            candidate_path = cycle_dir / "target_adapter.py"
            candidate_path.write_text(candidate.rstrip() + "\n", encoding="utf-8")
            semantic_sha256 = _semantic_code_sha256(candidate)
            record.update(
                adapter=str(candidate_path),
                adapter_sha256=_sha256_text(candidate.rstrip() + "\n"),
                semantic_sha256=semantic_sha256,
            )
            previous_code = reference_code
            if semantic_sha256 in seen_target_semantics:
                raise ValueError(
                    "Generated target adapter repeats a previously tested candidate and is "
                    "semantically unchanged; formatting, comments, or docstrings do not count "
                    "as a repair."
                )
            seen_target_semantics.add(semantic_sha256)
            record["changed_from_previous"] = bool(previous_code)
            record["changed_from_incumbent"] = bool(reference_code)
            validate_dynamic_adapter(candidate, task_program=frozen_program, require_program_constant=False)
            if policy.local_repair and reference_code:
                _validate_local_repair(
                    reference_code,
                    candidate,
                    repair_scope=repair_scope,
                )
            report(f"Target round {cycle}: running generated code ({generated.model})...")
            trial = run_dynamic_trial(
                spec=spec,
                robot_uid=spec.target_robot,
                control_mode=spec.target_control_mode,
                program=frozen_program,
                adapter_path=candidate_path,
            )
            if not trial.get("success") and not isinstance(
                trial.get("failure_diagnosis"), Mapping
            ):
                trial["failure_diagnosis"] = _dynamic_diagnosis(trial)
            trial_path = cycle_dir / "trial_result.json"
            _write_json(trial_path, trial)
            trial.pop("implementation_context", None)
            quality = _candidate_quality(trial)
            accepted = (
                _is_better_candidate(quality, incumbent_quality)
                if policy.preserve_incumbent
                else True
            )
            record.update(
                valid=True,
                trial_result=str(trial_path),
                trial=trial,
                candidate_quality=quality,
                accepted_as_incumbent=accepted,
            )
            latest_code, latest_result = candidate, trial
            if accepted:
                incumbent_code = candidate
                incumbent_result = trial
                incumbent_quality = quality
                incumbent_cycle = cycle
                (artifacts / "target_adapter.py").write_text(
                    candidate.rstrip() + "\n", encoding="utf-8"
                )
            else:
                record["rejection_reason"] = "candidate did not improve the incumbent quality"
            if trial.get("success"):
                target_success = True
        except Exception as exc:
            record.update(valid=False, error=repr(exc))
            latest_code = candidate
            latest_result = {"success": False, "message": repr(exc)}
        record["incumbent_cycle_after"] = incumbent_cycle
        record["incumbent_quality_after"] = incumbent_quality
        _write_json(cycle_dir / "cycle_record.json", record)
        target_cycles.append(record)
        report(
            f"Target round {cycle}: success={target_success}; "
            f"accepted={record.get('accepted_as_incumbent', False)}; "
            f"{latest_result.get('message', '')[:150]}"
        )
        if target_success:
            break

    target_result = incumbent_result or latest_result

    artifact_hashes = {
        name: _sha256_file(path) if path.is_file() else None
        for name, path in {
            "task_program": artifacts / "task_program.py",
            "source_adapter": artifacts / "source_adapter.py",
            "target_adapter": artifacts / "target_adapter.py",
        }.items()
    }
    metrics = _migration_metrics(source_cycles, target_cycles)
    manifest["outcome"] = {
        "status": "success" if target_success else target_status,
        "success": target_success,
        "source_cycles": len(source_cycles),
        "target_cycles": len(target_cycles),
        "incumbent_cycle": incumbent_cycle,
        "incumbent_quality": incumbent_quality,
        "artifact_sha256": artifact_hashes,
        "metrics": metrics,
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
        "method_id": policy.method_id,
        "method_policy": asdict(policy),
        "frozen_program": str(artifacts / "task_program.py"),
        "source_adapter": str(artifacts / "source_adapter.py"),
        "target_adapter": str(artifacts / "target_adapter.py") if (artifacts / "target_adapter.py").is_file() else "",
        "artifact_sha256": artifact_hashes,
        "source_cycles": source_cycles,
        "target_cycles": target_cycles,
        "source_result": source_result,
        "target_result": target_result,
        "incumbent_cycle": incumbent_cycle,
        "incumbent_quality": incumbent_quality,
        "metrics": metrics,
    }
    _write_dynamic_summary(run_dir, payload)
    return payload


def _migration_metrics(
    source_cycles: Sequence[Mapping[str, Any]],
    target_cycles: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize execution and LLM cost without changing success semantics."""

    calls = [row for row in target_cycles if row.get("used_llm")]
    prompt_tokens = sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in calls)
    completion_tokens = sum(
        int((row.get("usage") or {}).get("completion_tokens") or 0) for row in calls
    )
    cost = sum(float((row.get("usage") or {}).get("cost") or 0.0) for row in calls)
    executed = [row for row in target_cycles if isinstance(row.get("trial"), Mapping)]
    accepted = [row for row in executed if row.get("accepted_as_incumbent") is True]
    return {
        "generation_cycles": len(target_cycles),
        "repair_cycles": max(0, len(target_cycles) - 1),
        "llm_calls": len(calls),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "llm_cost_usd": round(cost, 8),
        "source_action_steps": sum(
            int((row.get("trial") or {}).get("action_steps") or 0) for row in source_cycles
        ),
        "target_trials": len(executed),
        "accepted_candidates": len(accepted),
        "rejected_candidates": len(executed) - len(accepted),
        "target_action_steps": sum(
            int((row.get("trial") or {}).get("action_steps") or 0) for row in executed
        ),
    }


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
- self._snapshot(): TCP, entity poses/velocities, robot joints/root pose, small numeric task_state,
  official evaluation and episode flags. Values in task_state are measurements, not writable objects.
- self._tcp_pos(): current tool-center-point position as xyz.
- self._entity_pos(name), self._actor_pos(name), self._region_pos(name): current xyz.
- self._entity_quat(name): current entity orientation quaternion.
- self._is_grasping_entity(name): current grasp detector.
- self._official_evaluation() and self._official_success(): official task state.
At least one method above must be called inside the generated skill. Store its return value
and use that measured value in an if/loop condition or to compute an action. Do not read
self.env internals as a substitute. Re-read measurements after actions to close the loop."""


def _action_api_contract() -> str:
    return """Documented action-frame contract:
- TCP and entity positions are flat xyz vectors in the world frame.
- For standard 4D pd_ee_delta_pos, do not override self._make_action().
- Compute a normalized world-frame command, then call self._make_action(command, gripper=...).
  The inherited method rotates world xyz into the robot-root controller frame.
- Prefer inherited self._move_towards(world_target, gripper=..., steps=...) for waypoint motion.
- For a nonstandard observed action layout only, map the extra channels explicitly and call
  self._world_delta_to_action_delta(world_command) before filling EE translation channels."""


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

{_action_api_contract()}

Minimal closed-loop pattern (adapt entity names and motion to the observed task):
```python
for _ in range(bounded_steps):
    if self._early_stop():
        return False
    tcp = self._tcp_pos()
    target = self._entity_pos("observed_entity_name")
    if np.linalg.norm(target - tcp) < tolerance:
        break
    world_command = np.clip((target - tcp) / self.max_delta_m, -1.0, 1.0)
    self._step(self._make_action(world_command, gripper=command))
```

Current failed source module, if any:
```python
{_trim(current_code, 18000)}
```

Latest source execution result, if any:
{_pretty(_prompt_trial(latest_result))}
"""


def _candidate_quality(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deterministic, lexicographically comparable candidate score."""

    diagnosis = result.get("failure_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
    reason = str(diagnosis.get("reason") or "")
    action_steps = int(result.get("action_steps") or 0)
    code_ok = bool(result.get("code_ok", True))
    official = bool(result.get("official_success"))
    success = bool(result.get("success"))
    unsafe = reason in {
        "environment_or_generated_module_error",
        "manipulated_object_left_workspace",
    }
    evaluation = result.get("official_evaluation")
    true_signals = 0
    if isinstance(evaluation, Mapping):
        true_signals = sum(
            1
            for key, value in evaluation.items()
            if str(key) != "success" and isinstance(value, bool) and value
        )
    progress_rank = _DIAGNOSIS_PROGRESS_RANK.get(reason, 30 if code_ok else 0)
    rank = [
        int(success),
        int(official),
        int(code_ok),
        int(action_steps > 0),
        int(not unsafe),
        progress_rank,
        true_signals,
        -action_steps,
    ]
    return {
        "rank": rank,
        "success": success,
        "official_success": official,
        "code_ok": code_ok,
        "acted": action_steps > 0,
        "constraint_safe": not unsafe,
        "diagnosis_reason": reason or None,
        "official_true_signals": true_signals,
        "action_steps": action_steps,
    }


def _is_better_candidate(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any] | None,
) -> bool:
    if incumbent is None:
        return True
    return tuple(candidate.get("rank") or ()) > tuple(incumbent.get("rank") or ())


def _repair_scope(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Map a physical failure to the smallest useful code-editing neighborhood."""

    diagnosis = result.get("failure_diagnosis")
    diagnosis = diagnosis if isinstance(diagnosis, Mapping) else {}
    reason = str(diagnosis.get("reason") or "")
    if not result:
        name = "initial_generation"
        may_change = "Create the complete target task skill and target action mapping."
    elif reason in {
        "environment_or_generated_module_error",
        "pre_action_execution_error",
        "semantic_entity_name_mismatch",
        "high_level_program_execution_error",
    }:
        name = "api_and_execution"
        may_change = "Fix imports, entity names, dimensions, API calls, and the failing statement."
    elif reason == "contact_not_established":
        name = "contact_geometry"
        may_change = (
            "Change only contact side, contact height/offset, short contact pulses, and "
            "state checks inside the task skill or its private motion helpers."
        )
    elif reason == "manipulated_object_left_workspace":
        name = "contact_safety"
        may_change = (
            "Reduce and stage contact commands, add overshoot guards, and reposition safely."
        )
    elif reason in {"no_action_progress_loop", "episode_budget_exhausted"}:
        name = "control_flow_and_budget"
        may_change = (
            "Change bounded loop exits, progress guards, waypoint budgets, and stalled-stage recovery."
        )
    elif reason in {
        "adapter_claimed_success_without_official_success",
        "official_task_outcome_not_reached",
    }:
        name = "task_completion"
        may_change = (
            "Change only the unfinished physical task stage and its measured completion guards."
        )
    else:
        name = "skill_policy"
        may_change = "Change the failed task-skill motion while preserving working interfaces."
    return {
        "name": name,
        "diagnosis_reason": reason or None,
        "may_change": may_change,
        "must_preserve": (
            "Keep the frozen high-level program, public skill signatures, controller/action "
            "mapping, simulator, official success signal, and unrelated working stages unchanged."
        ),
    }


def _method_contracts(code: str) -> tuple[Dict[str, str], Dict[str, str]]:
    tree = ast.parse(code)
    public: Dict[str, str] = {}
    protected: Dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = f"{node.name}.{item.name}"
            if not item.name.startswith("_"):
                public[key] = ast.dump(item.args, include_attributes=False)
            if item.name in _LOCAL_REPAIR_PROTECTED_METHODS:
                protected[key] = ast.dump(item, include_attributes=False)
    return public, protected


def _validate_local_repair(
    incumbent_code: str,
    candidate_code: str,
    *,
    repair_scope: Mapping[str, Any],
) -> None:
    """Reject broad rewrites when the diagnosis calls for a local policy repair."""

    if repair_scope.get("name") in {"initial_generation", "api_and_execution"}:
        return
    incumbent_public, incumbent_protected = _method_contracts(incumbent_code)
    candidate_public, candidate_protected = _method_contracts(candidate_code)
    if candidate_public != incumbent_public:
        raise ValueError(
            "Local repair changed public task-skill methods or signatures; preserve the "
            "incumbent API and modify only the diagnosed policy region."
        )
    if candidate_protected != incumbent_protected:
        raise ValueError(
            "Local repair changed controller/action-mapping methods; preserve _make_action, "
            "_validate_action_space, and _world_delta_to_action_delta."
        )


def _target_prompt(
    spec: DynamicMigrationSpec,
    source_observation: Mapping[str, Any],
    target_observation: Mapping[str, Any],
    program: str,
    source_code: str,
    source_result: Mapping[str, Any],
    incumbent_code: str,
    incumbent_result: Mapping[str, Any],
    *,
    latest_code: str,
    latest_result: Mapping[str, Any],
    repair_scope: Mapping[str, Any],
    policy: RepairMethodPolicy,
) -> str:
    diagnosis_label = (
        "Structured execution evidence with failure diagnosis"
        if policy.structured_diagnosis
        else "Raw execution evidence (no diagnosis or repair hint)"
    )
    reference_label = (
        "Current best target adapter (the preserved incumbent)"
        if policy.preserve_incumbent
        else "Most recent target adapter"
    )
    if policy.one_shot:
        repair_instructions = """This is a one-shot baseline:
- Generate one target adapter from source evidence and target interfaces.
- No target execution feedback or repair round will be provided."""
    elif policy.local_repair:
        repair_instructions = """Repair requirement for this round:
- Return a complete module, but make the smallest semantic change inside the allowed scope.
- Fix the concrete diagnosed failure before tuning later motion stages.
- Preserve public method signatures and controller/action-mapping methods.
- Make a semantic code change; formatting, comments, or docstrings alone are rejected."""
    else:
        repair_instructions = """Repair requirement for this round:
- Use only the raw execution evidence below to revise the complete target adapter.
- Preserve the required module contract and make a semantic code change.
- Do not assume a diagnosis or repair hint that is not present in the raw evidence."""
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
{_pretty(_prompt_trial(source_result, include_diagnosis=policy.structured_diagnosis))}

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
- Every loop iteration must execute a simulator action or break/return; a runtime watchdog rejects no-action loops.
- Do not modify TASK_PROGRAM, simulator state, object poses, controller internals, or official success.
- Returning True is insufficient: the harness independently checks env.unwrapped.evaluate()['success'].

{_measurement_api_contract()}

{_action_api_contract()}

Method under evaluation: {policy.method_id}

{reference_label}:
```python
{_trim(incumbent_code, 18000)}
```

{diagnosis_label}:
{_pretty(_prompt_trial(incumbent_result, include_diagnosis=policy.structured_diagnosis))}

Latest rejected or invalid candidate, if different from the incumbent:
```python
{_trim(latest_code, 10000)}
```

Latest rejected candidate evidence:
{_pretty(_prompt_trial(latest_result, include_diagnosis=policy.structured_diagnosis))}

Allowed local repair scope for this round:
{_pretty(repair_scope)}

{repair_instructions}
- Do not repeat the latest rejected change.
- If action_steps is 0, remove the reported pre-action exception so the candidate reaches self._step(action).
"""


def _prompt_observation(observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep the robot contract and task state, not renderer bookkeeping."""

    keys = (
        "env_id", "robot_uid", "control_mode", "environment_class", "environment_doc",
        "supported_robots", "action_space", "controller_summary", "controller_action_mapping",
        "tcp", "entity_aliases", "reset_info", "official_evaluation", "evaluate_source",
        "task_state", "entity_velocities", "robot_state",
    )
    compact = {key: observation.get(key) for key in keys if key in observation}
    compact["pose_entities"] = _prompt_entities(observation.get("pose_entities"))
    return compact


def _prompt_trial(
    result: Mapping[str, Any], *, include_diagnosis: bool = True
) -> Dict[str, Any]:
    """Bound feedback size while retaining physical counterexamples."""

    keys = (
        "seed", "success", "message", "code_ok", "ret_val", "official_success",
        "official_evaluation", "action_steps", "terminated", "truncated",
        "execution_error", "failure_layer", "failure_diagnosis",
    )
    compact = {key: result.get(key) for key in keys if key in result}
    if not include_diagnosis:
        compact.pop("failure_layer", None)
        compact.pop("failure_diagnosis", None)
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
    failures = result.get("multi_seed_failures")
    if isinstance(failures, list) and failures:
        compact["multi_seed_failure_count"] = len(failures)
        compact["multi_seed_failure_sample"] = [
            _prompt_trial(item)
            for item in failures[:3]
            if isinstance(item, Mapping)
        ]
    return compact


def _prompt_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    compact = {
        key: snapshot.get(key)
        for key in ("step", "tcp", "action", "official_evaluation", "terminated", "truncated",
                    "task_state", "entity_velocities", "robot_state")
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
        if "no simulator progress" in execution_error.lower():
            return {
                "layer": "skill_adapter",
                "reason": "no_action_progress_loop",
                "evidence": _concise_execution_error(execution_error),
                "repair_hint": (
                    "A bounded loop can still hang when an action helper returns without stepping. "
                    "Record action_steps at the start of each iteration and break, reposition, or "
                    "send a bounded explicit action when the counter does not advance."
                ),
            }
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
    escaped = _escaped_object_evidence(result)
    if escaped is not None:
        return {
            "layer": "contact_geometry",
            "reason": "manipulated_object_left_workspace",
            "evidence": escaped,
            "repair_hint": (
                "The action frame is working, but contact is too aggressive or poorly staged. "
                "Use short low-magnitude pulses, re-read the object pose after every pulse, "
                "stop before overshoot, and retreat/reposition when displacement is unexpectedly "
                "large. Do not repeat one long open-loop push."
            ),
        }
    stalled_contact = _stalled_contact_evidence(result)
    if stalled_contact is not None:
        return {
            "layer": "contact_geometry",
            "reason": "contact_not_established",
            "evidence": stalled_contact,
            "repair_hint": (
                "The TCP moved substantially but the manipulated object did not. Do not add "
                "more steps or repeat the same waypoint. Change the contact side or height, "
                "verify object motion after a short action pulse, and use a grasp-and-place "
                "strategy when the observed target gripper supports it."
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


def _escaped_object_evidence(result: Mapping[str, Any]) -> Dict[str, Any] | None:
    trace = result.get("state_trace")
    if not isinstance(trace, list) or len(trace) < 2:
        return None
    first_entities = trace[0].get("entities") if isinstance(trace[0], Mapping) else None
    last_entities = trace[-1].get("entities") if isinstance(trace[-1], Mapping) else None
    if not isinstance(first_entities, Mapping) or not isinstance(last_entities, Mapping):
        return None

    evidence = []
    for name in sorted(set(first_entities) & set(last_entities)):
        lowered = str(name).lower()
        if lowered.startswith("segmentation_id_map") or any(
            token in lowered for token in ("goal", "target", "region", "site")
        ):
            continue
        try:
            initial = np.asarray(first_entities[name], dtype=np.float64).reshape(-1)[:3]
            final = np.asarray(last_entities[name], dtype=np.float64).reshape(-1)[:3]
        except Exception:
            continue
        if initial.size < 3 or final.size < 3 or not np.all(np.isfinite(final)):
            continue
        displacement = float(np.linalg.norm(final - initial))
        fell = bool(final[2] < -0.05 and final[2] < initial[2] - 0.10)
        escaped = bool(displacement > 2.0)
        if fell or escaped:
            evidence.append(
                {
                    "entity": str(name),
                    "initial_position": np.round(initial, 5).tolist(),
                    "final_position": np.round(final, 5).tolist(),
                    "displacement_m": round(displacement, 5),
                    "fell_below_workspace": fell,
                }
            )
    return max(evidence, key=lambda item: item["displacement_m"]) if evidence else None


def _stalled_contact_evidence(result: Mapping[str, Any]) -> Dict[str, Any] | None:
    """Detect substantial TCP motion without corresponding manipulated-object motion."""

    if int(result.get("action_steps") or 0) < 5:
        return None
    trace = result.get("state_trace")
    if not isinstance(trace, list) or len(trace) < 2:
        return None
    first = trace[0] if isinstance(trace[0], Mapping) else {}
    last = trace[-1] if isinstance(trace[-1], Mapping) else {}
    try:
        tcp_initial = np.asarray(first.get("tcp"), dtype=np.float64).reshape(-1)[:3]
        tcp_final = np.asarray(last.get("tcp"), dtype=np.float64).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if tcp_initial.size < 3 or tcp_final.size < 3:
        return None
    tcp_displacement = float(np.linalg.norm(tcp_final - tcp_initial))
    if tcp_displacement < 0.03:
        return None

    first_entities = first.get("entities")
    last_entities = last.get("entities")
    if not isinstance(first_entities, Mapping) or not isinstance(last_entities, Mapping):
        return None
    preferred = ("obj", "object", "cube", "ball", "sphere", "peg", "ycb")
    candidates = [
        name
        for name in sorted(set(first_entities) & set(last_entities))
        if any(token in str(name).lower() for token in preferred)
        and not any(token in str(name).lower() for token in ("goal", "target", "region", "site"))
    ]
    evidence = []
    for name in candidates:
        try:
            initial = np.asarray(first_entities[name], dtype=np.float64).reshape(-1)[:3]
            final = np.asarray(last_entities[name], dtype=np.float64).reshape(-1)[:3]
        except (TypeError, ValueError):
            continue
        if initial.size < 3 or final.size < 3:
            continue
        displacement = float(np.linalg.norm(final - initial))
        if displacement > 0.005:
            continue
        evidence.append(
            {
                "entity": str(name),
                "action_steps": int(result.get("action_steps") or 0),
                "tcp_displacement_m": round(tcp_displacement, 5),
                "object_displacement_m": round(displacement, 5),
                "final_tcp_object_xy_m": round(float(np.linalg.norm(tcp_final[:2] - final[:2])), 5),
                "initial_position": np.round(initial, 5).tolist(),
                "final_position": np.round(final, 5).tolist(),
            }
        )
    return min(evidence, key=lambda item: item["object_displacement_m"]) if evidence else None


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
            "| cycle | model | valid | accepted | repair scope | official success | action steps | adapter sha | message |",
            "|---:|---|---|---|---|---|---:|---|---|",
        ]
    )
    for item in payload.get("target_cycles") or []:
        trial = item.get("trial") or {}
        lines.append(
            f"| {item.get('cycle')} | {item.get('model')} | {item.get('valid')} | "
            f"{item.get('accepted_as_incumbent')} | "
            f"{(item.get('repair_scope') or {}).get('name')} | "
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
