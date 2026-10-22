"""Extract task stages and source-robot dependencies from successful robot code."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


_MOTION_CALLS = {"move_to_pose_with_screw", "move_to_pose"}
_GRASP_CALLS = {"close_gripper"}
_RELEASE_CALLS = {"open_gripper"}


def extract_stage_program(
    source_path: Path,
    *,
    task_id: str,
    source_robot: str,
    source_control_mode: str,
    known_entities: Iterable[str] = (),
    source_action_trace: Iterable[Mapping[str, Any]] = (),
    official_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic stage representation with explicit provenance."""

    source_reference = str(source_path)
    source_path = source_path.resolve()
    code = source_path.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(source_path))
    run = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"),
        None,
    )
    if run is None:
        raise ValueError("Source program must define run(...).")

    assignments = _assignments(run)
    entity_names = set(known_entities)
    operations = _action_operations(run)
    stages: list[dict[str, Any]] = []
    previous_entities: list[str] = []
    for operation in operations:
        target = operation.get("target_symbol")
        dependency_nodes = _dependency_nodes(target, assignments, before_line=int(operation["line"])) if target else []
        entities = sorted(_entity_references(dependency_nodes, entity_names))
        if operation["stage_id"] == "release" and not entities:
            entities = list(previous_entities)
        if entities:
            previous_entities = entities
        stage = _stage_record(operation, entities, dependency_nodes)
        if stages and stages[-1]["stage_id"] == stage["stage_id"]:
            _merge_stage(stages[-1], stage)
        else:
            stages.append(stage)

    trace = [dict(row) for row in source_action_trace]
    _attach_action_trace(stages, trace)
    imports = [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)) and source_robot.lower() in ast.unparse(node).lower()
    ]
    first_action_line = min((stage["source_lines"][0] for stage in stages), default=10**9)
    pre_action_calls = sorted({
        ast.unparse(node)
        for node in ast.walk(run)
        if isinstance(node, ast.Call)
        and node.lineno < first_action_line
        and (
            "planner" in ast.unparse(node.func)
            or "build_grasp_pose" in ast.unparse(node.func)
            or "grasp_info" in ast.unparse(node.func)
            or "actor_obb" in ast.unparse(node.func)
        )
    })
    all_stage_lines = {
        line
        for stage in stages
        for line in range(stage["source_lines"][0], stage["source_lines"][1] + 1)
    }
    unmapped = [row for row in trace if row.get("source_line") not in all_stage_lines]
    return {
        "schema": "source_stage_program.v1",
        "task_id": task_id,
        "source_program": source_reference,
        "source_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "entrypoint": "run",
        "source_robot": source_robot,
        "source_control_mode": source_control_mode,
        "stage_order": [stage["stage_id"] for stage in stages],
        "stages": stages,
        "source_assumptions": {
            "robot_specific_imports": imports,
            "controller": source_control_mode,
            "planner_symbols": sorted(
                {ast.unparse(node.func) for node in ast.walk(run) if isinstance(node, ast.Call) and "planner" in ast.unparse(node.func)}
            ),
            "pre_action_robot_geometry_calls": pre_action_calls,
        },
        "official_evaluation_at_trace_end": dict(official_evaluation or {}),
        "trace_summary": {
            "action_count": len(trace),
            "mapped_action_count": len(trace) - len(unmapped),
            "unmapped_action_count": len(unmapped),
            "unmapped_source_lines": dict(sorted(Counter(row.get("source_line") for row in unmapped).items(), key=lambda item: str(item[0]))),
        },
        "provenance": {
            "stage_boundaries": "AST action calls",
            "entity_dependencies": "recursive AST assignment dependencies",
            "action_counts": "successful runtime source-line attribution",
            "completion_checks": "operation templates plus official evaluate fields",
            "llm_used": False,
        },
    }


def validate_stage_program(payload: Mapping[str, Any], expected_stage_ids: Iterable[str]) -> list[str]:
    errors = []
    expected = list(expected_stage_ids)
    observed = list(payload.get("stage_order") or [])
    if observed != expected:
        errors.append(f"stage_order expected {expected!r}, got {observed!r}")
    stages = payload.get("stages") or []
    if len(stages) != len(expected):
        errors.append("stage count does not match the study protocol")
    for stage in stages:
        if not stage.get("source_lines") or not stage.get("operations"):
            errors.append(f"{stage.get('stage_id')}: missing source provenance")
        if int(stage.get("runtime", {}).get("action_count", 0)) <= 0:
            errors.append(f"{stage.get('stage_id')}: no successful runtime actions attributed")
    trace = payload.get("trace_summary") or {}
    if int(trace.get("unmapped_action_count", 0)):
        errors.append("successful runtime contains actions outside extracted stages")
    return errors


def _assignments(run: ast.AST) -> dict[str, list[tuple[int, ast.AST]]]:
    result: dict[str, list[tuple[int, ast.AST]]] = {}
    for node in ast.walk(run):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    result.setdefault(name, []).append((node.lineno, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            for name in _target_names(node.target):
                result.setdefault(name, []).append((node.lineno, node.value))
    for definitions in result.values():
        definitions.sort(key=lambda item: item[0])
    return result


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in _target_names(item)]
    return []


def _action_operations(run: ast.AST) -> list[dict[str, Any]]:
    result = []
    for node in sorted((item for item in ast.walk(run) if isinstance(item, ast.Call)), key=lambda item: (item.lineno, item.col_offset)):
        name = _call_name(node)
        if name in _MOTION_CALLS:
            if any(keyword.arg == "dry_run" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords):
                continue
            target_expression = ast.unparse(node.args[0]) if node.args else None
            target_symbol = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) else None
            result.append({
                "stage_id": _motion_stage(target_symbol or target_expression or ""),
                "operation": "move_tcp_to_pose",
                "call": ast.unparse(node),
                "target_expression": target_expression,
                "target_symbol": target_symbol,
                "line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
            })
        elif name in _GRASP_CALLS:
            result.append(_simple_operation("grasp", "close_gripper", node))
        elif name in _RELEASE_CALLS:
            result.append(_simple_operation("release", "open_gripper", node))
    return result


def _simple_operation(stage_id: str, operation: str, node: ast.Call) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "operation": operation,
        "call": ast.unparse(node),
        "target_expression": None,
        "target_symbol": None,
        "line": node.lineno,
        "end_line": getattr(node, "end_lineno", node.lineno),
    }


def _motion_stage(target: str) -> str:
    value = target.lower()
    if any(token in value for token in ("reach", "approach", "pregrasp")):
        return "approach"
    if "grasp" in value:
        return "grasp"
    if any(token in value for token in ("lift", "raise")):
        return "lift"
    if any(token in value for token in ("align", "place", "stack", "goal")):
        return "align"
    return "move"


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _dependency_nodes(
    symbol: str,
    assignments: Mapping[str, list[tuple[int, ast.AST]]],
    *,
    before_line: int,
) -> list[ast.AST]:
    result: list[ast.AST] = []
    pending = [(symbol, before_line)]
    seen: set[tuple[str, int]] = set()
    while pending:
        name, cutoff = pending.pop()
        definitions = [item for item in assignments.get(name, []) if item[0] < cutoff]
        if not definitions:
            continue
        line, value = definitions[-1]
        key = (name, line)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        pending.extend(
            (child.id, line)
            for child in ast.walk(value)
            if isinstance(child, ast.Name) and child.id in assignments
        )
    return result


def _entity_references(nodes: Iterable[ast.AST], known_entities: set[str]) -> set[str]:
    return {
        child.attr
        for node in nodes
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute) and child.attr in known_entities
    }


def _stage_record(operation: Mapping[str, Any], entities: list[str], dependencies: list[ast.AST]) -> dict[str, Any]:
    stage_id = str(operation["stage_id"])
    dependency_expressions = sorted({ast.unparse(node) for node in dependencies})
    completion = {
        "approach": "tcp reaches the source-derived pregrasp pose",
        "grasp": "official is_cubeA_grasped is true",
        "lift": "cubeA height increases while grasp remains true",
        "align": "cubeA is aligned above cubeB before release",
        "release": "official success: cubeA is on cubeB, static, and not grasped",
    }.get(stage_id, "stage target is reached")
    transferable_relation = {
        "approach": "derive a collision-clear pregrasp pose from cubeA",
        "grasp": "close the target gripper around cubeA and establish a verified grasp",
        "lift": "increase cubeA height while retaining the grasp",
        "align": "place cubeA above cubeB using the source object-relative stack geometry",
        "release": "release cubeA on cubeB and satisfy the unchanged official success predicate",
    }.get(stage_id, "preserve the source object-relative target")
    adaptation_obligations = {
        "approach": ["replace Panda planner motion", "check target pregrasp reachability"],
        "grasp": [
            "adapt grasp orientation and TCP offset to the target gripper",
            "map target close command",
            "verify official grasp state",
        ],
        "lift": ["replace Panda planner motion", "guard against loss of grasp during transport"],
        "align": [
            "replace Panda planner motion",
            "account for the target TCP-to-held-object transform",
            "preserve cubeA-to-cubeB placement geometry",
        ],
        "release": ["map target open command", "wait for official static and released conditions"],
    }.get(stage_id, ["replace source embodiment calls"])
    return {
        "stage_id": stage_id,
        "source_lines": [operation["line"], operation["end_line"]],
        "operations": [operation["operation"]],
        "source_calls": [operation["call"]],
        "target_symbol": operation.get("target_symbol"),
        "target_expression": operation.get("target_expression"),
        "entities": entities,
        "dependency_expressions": dependency_expressions,
        "completion_check": completion,
        "transferable_relation": transferable_relation,
        "preserve": ["stage order", "object-relative task relation", "completion semantics"],
        "adaptation_obligations": adaptation_obligations,
        "runtime": {"action_count": 0, "first_action_step": None, "last_action_step": None},
    }


def _merge_stage(target: dict[str, Any], addition: Mapping[str, Any]) -> None:
    target["source_lines"][0] = min(target["source_lines"][0], addition["source_lines"][0])
    target["source_lines"][1] = max(target["source_lines"][1], addition["source_lines"][1])
    for key in ("operations", "source_calls", "entities", "dependency_expressions", "preserve", "adaptation_obligations"):
        target[key] = list(dict.fromkeys([*target[key], *addition[key]]))
    if target.get("target_symbol") is None:
        target["target_symbol"] = addition.get("target_symbol")
        target["target_expression"] = addition.get("target_expression")


def _attach_action_trace(stages: list[dict[str, Any]], trace: list[dict[str, Any]]) -> None:
    for stage in stages:
        start, end = stage["source_lines"]
        rows = [row for row in trace if isinstance(row.get("source_line"), int) and start <= row["source_line"] <= end]
        if rows:
            stage["runtime"] = {
                "action_count": len(rows),
                "first_action_step": min(int(row["step"]) for row in rows),
                "last_action_step": max(int(row["step"]) for row in rows),
                "source_line_counts": dict(sorted(Counter(int(row["source_line"]) for row in rows).items())),
            }
            state = rows[-1].get("state_after")
            if isinstance(state, Mapping):
                entities = state.get("entities") if isinstance(state.get("entities"), Mapping) else {}
                stage["runtime"]["final_state"] = {
                    "step": state.get("step"),
                    "tcp": state.get("tcp"),
                    "entities": {name: entities.get(name) for name in stage["entities"] if name in entities},
                    "official_evaluation": state.get("official_evaluation"),
                }
