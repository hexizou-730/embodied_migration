#!/usr/bin/env python3
"""Bounded manual diagnostic of StackCube object-relative placement and release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from maniskill_backend.dynamic_harness import DynamicMigrationSpec, run_dynamic_trial
from maniskill_backend.selective_migration import FROZEN_PROGRAM, method_hashes, stage_boundary_states, validate_selective_adapter


STAGES = ["approach", "grasp", "lift", "align", "release"]
ALIGN = '''
def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    goal = cubeB.copy()
    goal[2] += 0.04
    hover = goal.copy()
    hover[2] += 0.07
    for target, limit in ((hover, 120), (goal, 100)):
        for _ in range(limit):
            if self._early_stop():
                return False
            cubeA = self._entity_pos("cubeA")
            error = target - cubeA
            if np.linalg.norm(error) < 0.003:
                break
            command = np.clip(error / 0.10, -0.06, 0.06)
            self._step(self._make_action(command, gripper=self.gripper_close))
        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=6)
    cubeA = self._entity_pos("cubeA")
    ev = self._official_evaluation()
    return bool(np.linalg.norm(cubeA[:2] - goal[:2]) < 0.008
                and ev.get("is_cubeA_on_cubeB", False)
                and self._is_grasping_entity("cubeA"))
'''
RELEASE = '''
def _stage_release(self) -> bool:
    cubeA = self._entity_pos("cubeA")
    for command in np.linspace(self.gripper_close, self.gripper_open, 9)[1:]:
        if self._early_stop():
            return self._official_success()
        self._repeat_action(np.zeros(3), gripper=float(command), steps=3)
    retreat = self._tcp_pos() + np.array([0.0, 0.0, 0.08], dtype=np.float32)
    self._move_towards(retreat, gripper=self.gripper_open, steps=30)
    for _ in range(20):
        if self._early_stop():
            return self._official_success()
        self._repeat_action(np.zeros(3), gripper=self.gripper_open, steps=1)
        if self._official_success():
            return True
    return self._official_success()
'''


def replace_stage(code: str, stage: str, replacement: str) -> str:
    tree = ast.parse(code)
    replacement_node = ast.parse(replacement).body[0]
    if replacement_node.name != f"_stage_{stage}":
        raise ValueError("Replacement does not match its declared stage.")
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for index, node in enumerate(cls.body):
            if isinstance(node, ast.FunctionDef) and node.name == replacement_node.name:
                cls.body[index] = replacement_node
    updated = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    validate_selective_adapter(updated, stage_order=STAGES, incumbent_code=code, allowed_stage=stage)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("evidence/stackcube_followup_20261023/results/selective_migration/corrected_gripper/cycle_02/target_adapter.py"))
    parser.add_argument("--output", type=Path, default=Path("results/stackcube_placement_diagnostic"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        parser.error("Use a new output directory to preserve the previous diagnostic.")
    original = args.candidate.read_text(encoding="utf-8")
    align_code = replace_stage(original, "align", ALIGN)
    cases = [("original", original), ("object_relative_align", align_code),
             ("object_relative_slow_release", replace_stage(align_code, "release", RELEASE))]
    spec = DynamicMigrationSpec(
        env_id="StackCube-v1", task_label="t07_stack_cube", source_robot="panda",
        target_robot="xarm6_robotiq", source_control_mode="pd_joint_pos",
        target_control_mode="pd_ee_delta_pose", seed=args.seed, max_episode_steps=500,
    )
    original_hashes = method_hashes(original)
    rows = []
    for label, code in cases:
        folder = args.output / label
        folder.mkdir(parents=True, exist_ok=True)
        candidate = folder / "target_adapter.py"
        candidate.write_text(code, encoding="utf-8")
        result = run_dynamic_trial(spec=spec, robot_uid=spec.target_robot,
                                   control_mode=spec.target_control_mode,
                                   program=FROZEN_PROGRAM, adapter_path=candidate)
        (folder / "trial_result.json").write_text(json.dumps(result, indent=2, default=repr) + "\n", encoding="utf-8")
        hashes = method_hashes(code)
        row = {"label": label, "success": bool(result.get("success")),
               "official_success": result.get("official_success"), "action_steps": result.get("action_steps", 0),
               "message": result.get("message"), "official_evaluation": result.get("official_evaluation"),
               "stage_boundaries": stage_boundary_states(result),
               "changed_methods": [key for key in hashes if original_hashes.get(key) != hashes[key]],
               "adapter_sha256": hashlib.sha256(code.encode()).hexdigest()}
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("label", "success", "official_success", "action_steps", "message")}), flush=True)
    payload = {"schema": "stackcube_placement_diagnostic.v1", "seed": args.seed,
               "provenance": "manual diagnostic replacements; not LLM-generated migration results",
               "llm_calls": 0, "rows": rows}
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
