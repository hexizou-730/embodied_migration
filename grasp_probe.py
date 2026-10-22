#!/usr/bin/env python3
"""Diagnose the saved StackCube grasp without calling an LLM."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot, _first_vector
from maniskill_backend.env_adapter import ManiSkillEnvAdapter
from maniskill_backend.selective_migration import method_hashes, validate_selective_adapter


STAGES = ["approach", "grasp", "lift", "align", "release"]


def repair_grasp_close(code: str, *, z_offset: float | None = None) -> str:
    """Change only stationary motion calls inside grasp to real hold actions."""
    tree = ast.parse(code)
    for method in ast.walk(tree):
        if not isinstance(method, ast.FunctionDef) or method.name != "_stage_grasp":
            continue
        for node in ast.walk(method):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_move_towards"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Attribute)
                and node.args[0].func.attr == "_tcp_pos"
            ):
                node.func.attr = "_repeat_action"
                node.args[0] = ast.parse("np.zeros(3, dtype=np.float32)", mode="eval").body
            if z_offset is not None and isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "grasp_target" for t in node.targets):
                    value = node.value
                    if not isinstance(value, ast.Call) or not isinstance(value.args[0], ast.List):
                        raise ValueError("Expected a literal grasp_target xyz list.")
                    value.args[0].elts[2] = ast.parse(f"cubeA[2] + {float(z_offset)!r}", mode="eval").body
    repaired = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    validate_selective_adapter(repaired, stage_order=STAGES, incumbent_code=code, allowed_stage="grasp")
    return repaired


def correct_gripper_endpoints(code: str) -> str:
    """Explicit configuration intervention; kept separate from local policy repair."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            for keyword in call.keywords:
                if keyword.arg == "gripper_open":
                    keyword.value = ast.Constant(0.0)
                elif keyword.arg == "gripper_close":
                    keyword.value = ast.Constant(0.81)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


class ActionRecorder:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.rows: list[dict[str, Any]] = []
        self.phase = "approach"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.rows.append({"phase": self.phase, "action": np.asarray(action).reshape(-1).tolist()})
        return result


def measure(robot: Any) -> dict[str, Any]:
    base = robot._base_env()
    evaluation = robot._official_evaluation()
    left = _first_vector(base.agent.finger1_link.pose.p, 3)
    right = _first_vector(base.agent.finger2_link.pose.p, 3)
    forces = [
        _first_vector(base.scene.get_pairwise_contact_forces(link, base.cubeA), 3)
        for link in (base.agent.finger1_link, base.agent.finger2_link)
    ]
    return {
        "tcp": robot._tcp_pos().tolist(),
        "cubeA": robot._entity_pos("cubeA").tolist(),
        "qpos": _first_vector(base.agent.robot.get_qpos(), 12).tolist(),
        "is_grasping": bool(robot._is_grasping_entity("cubeA")),
        "finger_pad_separation": float(np.linalg.norm(left - right)),
        "finger_contact_forces": [float(np.linalg.norm(force)) for force in forces],
        "official_evaluation": {
            key: np.asarray(getattr(value, "cpu", lambda: value)()).reshape(-1).tolist()
            for key, value in evaluation.items()
        },
    }


def run_case(path: Path, *, seed: int, label: str) -> dict[str, Any]:
    adapter = ManiSkillEnvAdapter(
        "StackCube-v1", robot_uid="xarm6_robotiq", obs_mode="state",
        control_mode="pd_ee_delta_pose", sim_backend="auto", render_backend="gpu",
        max_episode_steps=500, reward_mode="sparse",
    )
    result: dict[str, Any] = {"label": label, "seed": seed, "adapter": str(path)}
    recorder = None
    try:
        env = adapter.make()
        adapter.reset(seed=seed)
        recorder = ActionRecorder(env)
        module_spec = importlib.util.spec_from_file_location("grasp_diagnostic_candidate", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        robot = module.build_robot(recorder, control_mode="pd_ee_delta_pose", robot_uid="xarm6_robotiq")
        initial = measure(robot)
        result["initial"] = initial
        result["approach_returned"] = bool(robot._run_stage("approach", robot._stage_approach))
        result["after_approach"] = measure(robot)
        recorder.phase = "grasp"
        result["grasp_returned"] = bool(robot._run_stage("grasp", robot._stage_grasp)) if result["approach_returned"] else False
        after_close = measure(robot)
        result["after_close"] = after_close
        if after_close["is_grasping"] and not robot._early_stop():
            recorder.phase = "diagnostic_lift"
            target = robot._tcp_pos() + np.array([0.0, 0.0, 0.08], dtype=np.float32)
            robot._move_towards(target, gripper=robot.gripper_close, steps=40)
            robot._repeat_action(np.zeros(3), gripper=robot.gripper_close, steps=10)
        final = measure(robot)
        result["final"] = final
        result["cube_disp_xy"] = float(np.linalg.norm(np.array(after_close["cubeA"][:2]) - initial["cubeA"][:2]))
        result["lift_dz"] = float(final["cubeA"][2] - after_close["cubeA"][2])
        result["grasp_lift_verified"] = bool(after_close["is_grasping"] and final["is_grasping"] and result["lift_dz"] > 0.04)
        result["stage_events"] = robot.execution_log()
        result["method_hashes"] = method_hashes(path.read_text())
        result["adapter_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["close_actions"] = int(sum(
            row["phase"] == "grasp" and np.isclose(row["action"][-1], robot.gripper_close)
            for row in recorder.rows
        ))
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["actions"] = recorder.rows if recorder is not None else []
        result["action_steps"] = len(result["actions"])
        adapter.close()
    return result


def measure_gripper_endpoints(*, seed: int) -> dict[str, Any]:
    adapter = ManiSkillEnvAdapter(
        "StackCube-v1", robot_uid="xarm6_robotiq", obs_mode="state",
        control_mode="pd_ee_delta_pose", sim_backend="auto", render_backend="gpu",
        max_episode_steps=100, reward_mode="sparse",
    )
    rows = []
    try:
        env = adapter.make()
        for command in (0.0, 0.81):
            adapter.reset(seed=seed)
            robot = ManiSkillDynamicRobot(env, control_mode="pd_ee_delta_pose", robot_uid="xarm6_robotiq")
            robot._repeat_action(np.zeros(3), gripper=command, steps=20)
            rows.append({"command": command, **measure(robot)})
    finally:
        adapter.close()
    ordered = sorted(rows, key=lambda row: row["finger_pad_separation"])
    return {
        "rows": rows,
        "measured_close": ordered[0]["command"],
        "measured_open": ordered[-1]["command"],
        "measurement": "Euclidean distance between finger-pad link origins, in free space",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("evidence/stackcube_pilot_20261022/selective_migration_phase3/artifacts/target_adapter.py"))
    parser.add_argument("--output", type=Path, default=Path("results/stackcube_grasp_diagnostic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-offsets", default="", help="Optional fixed XY diagnostic height offsets, comma separated.")
    parser.add_argument("--correct-gripper", action="store_true", help="Also test independently measured open=0, close=0.81.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    endpoints = measure_gripper_endpoints(seed=args.seed)
    (args.output / "gripper_endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n", encoding="utf-8")
    if args.correct_gripper and (endpoints["measured_open"], endpoints["measured_close"]) != (0.0, 0.81):
        raise RuntimeError("Measured gripper directions do not match the diagnostic configuration.")
    original = args.candidate.read_text(encoding="utf-8")
    cases = [("original", original), ("explicit_close", repair_grasp_close(original))]
    mapped = correct_gripper_endpoints(original) if args.correct_gripper else original
    if args.correct_gripper:
        cases.append(("correct_endpoints", repair_grasp_close(mapped)))
    for offset in filter(None, args.z_offsets.split(",")):
        cases.append((f"z_{float(offset):+.3f}", repair_grasp_close(mapped, z_offset=float(offset))))
    rows = []
    for label, code in cases:
        folder = args.output / label
        folder.mkdir(exist_ok=True)
        path = folder / "target_adapter.py"
        path.write_text(code, encoding="utf-8")
        row = run_case(path, seed=args.seed, label=label)
        (folder / "trial.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        compact = {key: row.get(key) for key in ("label", "action_steps", "close_actions", "grasp_returned", "cube_disp_xy", "lift_dz", "grasp_lift_verified", "error")}
        rows.append(compact)
        print(json.dumps(compact), flush=True)
    payload = {
        "schema": "stackcube_grasp_diagnostic.v1", "seed": args.seed,
        "provenance": "manual diagnostic intervention on saved LLM code; not an LLM migration result",
        "llm_calls": 0, "rows": rows,
        "success_definition": "official grasp detector before and after lift; cube rises more than 4 cm",
        "not_established": "official StackCube task success",
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
