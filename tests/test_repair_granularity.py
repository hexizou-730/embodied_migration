from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maniskill_backend.dynamic_harness import DynamicMigrationSpec
from maniskill_backend.llm import LLMTextResult
from maniskill_backend.repair_granularity import (
    _module_repair_prompt,
    resolve_failed_candidate,
    run_granularity_ablation,
    run_granularity_arm,
    run_module_rewrite_repair,
    summarize_granularity_repetitions,
)
from maniskill_backend.selective_migration import _stage_repair_prompt


STAGES = ["approach", "grasp", "lift", "align", "release"]
SOURCE_STAGE_PROGRAM = {
    "stage_order": STAGES,
    "stages": [
        {"stage_id": "lift", "runtime": {"final_state": {"official_evaluation": {"on_support": False}}}},
        {"stage_id": "align", "runtime": {"final_state": {"official_evaluation": {"on_support": True}}}},
    ],
}
REPLAY_TRIAL = {
    "success": False,
    "action_steps": 101,
    "runtime_snapshot": {"stage_runtime": {"completed_stages": STAGES[:-1], "failed_stage": "release"}},
    "source_action_trace": [
        {"source_function": "_stage_align", "state_after": {
            "step": 8, "official_evaluation": {"on_support": False},
        }},
    ],
}
SUCCESS_TRIAL = {"success": True, "action_steps": 31}


def _adapter_code(*, steps: dict[str, int] | None = None, extra_helper: bool = False) -> str:
    counts = steps or {}
    run_lines = "\n".join(
        f'        if not self._run_stage("{stage}", self._stage_{stage}): return False'
        for stage in STAGES
    )
    methods = []
    for stage in STAGES:
        count = counts.get(stage, 1)
        methods.append(
            f'''    def _stage_{stage}(self) -> bool:
        for _ in range({count}):
            state = self._snapshot()
            if self._early_stop():
                return False
            self._step(self._make_action(np.zeros(3, dtype=np.float32), gripper=0.0))
            if state and self._official_success():
                return True
        return False
'''
        )
    helper = "    def _helper(self) -> bool:\n        return True\n\n" if extra_helper else ""
    return f'''from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedRobot(ManiSkillDynamicRobot):
{helper}{''.join(methods)}
    def run_source_actions(self) -> bool:
{run_lines}
        return self._official_success()

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''


def _stage_method_source(code: str, stage_id: str) -> str:
    node = next(
        item for item in ast.walk(ast.parse(code))
        if isinstance(item, ast.FunctionDef) and item.name == f"_stage_{stage_id}"
    )
    return ast.get_source_segment(code, node)


def _write_study(root: Path, *, with_ablation_candidate: str | None = None) -> Path:
    study_dir = root / "benchmark_tasks" / "t07_stack_cube"
    study_dir.mkdir(parents=True)
    study = {
        "source": {"stage_representation": "stages.json", "robot_uid": "panda", "control_mode": "pd_joint_pos"},
        "target": {"robot_uid": "xarm6_robotiq", "control_mode": "pd_ee_delta_pose"},
        "task": {"env_id": "StackCube-v1", "task_id": "t07_stack_cube"},
    }
    if with_ablation_candidate is not None:
        study["repair_granularity_ablation"] = {"failed_candidate": with_ablation_candidate}
    study_path = study_dir / "repair_granularity_ablation.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")
    (root / "stages.json").write_text(json.dumps(SOURCE_STAGE_PROGRAM), encoding="utf-8")
    return study_path


def _section(prompt: str, start_marker: str, end_marker: str) -> str:
    start = prompt.index(start_marker) + len(start_marker)
    end = prompt.index(end_marker, start)
    return prompt[start:end]


SPEC = DynamicMigrationSpec(
    env_id="StackCube-v1", task_label="t07_stack_cube", source_robot="panda", target_robot="xarm6_robotiq",
)


class ModuleRewriteArmTests(unittest.TestCase):
    def test_module_prompt_carries_identical_evidence_to_method_prompt(self) -> None:
        incumbent = _adapter_code()
        shared = dict(
            spec=SPEC,
            stage_program=SOURCE_STAGE_PROGRAM,
            target_observation={"action_space": {"low": [-1], "high": [1]}, "evaluate_source": "return is_cubeA_on_cubeB"},
            incumbent_code=incumbent,
            previous_result=REPLAY_TRIAL,
            repair_stage="align",
            locked_stages=["approach", "grasp", "lift"],
            latest_error="unchanged",
        )
        method_prompt = _stage_repair_prompt(**shared)
        module_prompt = _module_repair_prompt(**shared)
        self.assertEqual(
            _section(method_prompt, "Source-derived relation and successful source-stage observation:\n", "\n\nTarget control mode:"),
            _section(module_prompt, "Source-derived relation and successful source-stage observation:\n", "\n\nTarget control mode:"),
        )
        self.assertEqual(
            _section(method_prompt, "Real execution evidence:\n", "\n\nEach candidate restarts"),
            _section(module_prompt, "Real execution evidence:\n", "\n\nEach candidate restarts"),
        )
        self.assertIn("Return only the replacement function.", method_prompt)
        self.assertIn("Return exactly one Python function", method_prompt)
        self.assertIn("Return only the complete module.", module_prompt)
        self.assertIn("any of the five _stage_* methods and __init__ may change", module_prompt)
        self.assertIn("class GeneratedRobot(ManiSkillDynamicRobot)", module_prompt)
        self.assertIn('The diagnosed repair stage from measured evidence is "align"', module_prompt)

    def test_module_arm_replay_localizes_and_accepts_multi_method_changes(self) -> None:
        incumbent = _adapter_code()
        changed = _adapter_code(steps={"align": 2, "lift": 2})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = _write_study(root)
            candidate = root / "failed_candidate.py"
            candidate.write_text(incumbent, encoding="utf-8")
            expected_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            with patch("maniskill_backend.repair_granularity.discover_environment", return_value={}), \
                    patch("maniskill_backend.repair_granularity.run_dynamic_trial", side_effect=[REPLAY_TRIAL, SUCCESS_TRIAL]), \
                    patch("maniskill_backend.repair_granularity.gen_text", return_value=LLMTextResult(text=changed, used_llm=True, model="test")):
                result = run_module_rewrite_repair(
                    study_path=study_path, run_dir=root / "output",
                    max_cycles=1, resume_adapter=candidate,
                )
        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["repair_output_format"], "complete_module")
        self.assertEqual(result["resume_replay"]["action_steps"], 101)
        self.assertEqual(result["resume_replay"]["failed_candidate_sha256"], expected_sha)
        cycle = result["cycles"][0]
        self.assertTrue(cycle["valid"])
        self.assertEqual(cycle["diagnosed_stage"], "align")
        self.assertEqual(cycle["generation_unit"], "complete_module")
        self.assertEqual(cycle["changed_methods"], ["_stage_align", "_stage_lift"])
        self.assertEqual(cycle["locked_stages_before"], ["approach", "grasp", "lift"])
        self.assertEqual(cycle["action_steps"], 31)
        self.assertEqual(
            result["resume_replay"]["repair_localization"]["reason"], "source_stage_postcondition_not_reached",
        )

    def test_module_arm_rejects_semantically_unchanged_module(self) -> None:
        incumbent = _adapter_code()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = _write_study(root)
            candidate = root / "failed_candidate.py"
            candidate.write_text(incumbent, encoding="utf-8")
            with patch("maniskill_backend.repair_granularity.discover_environment", return_value={}), \
                    patch("maniskill_backend.repair_granularity.run_dynamic_trial", side_effect=[REPLAY_TRIAL]), \
                    patch("maniskill_backend.repair_granularity.gen_text", return_value=LLMTextResult(text=incumbent, used_llm=True, model="test")):
                result = run_module_rewrite_repair(
                    study_path=study_path, run_dir=root / "output",
                    max_cycles=1, resume_adapter=candidate,
                )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "target_budget_exhausted")
        cycle = result["cycles"][0]
        self.assertFalse(cycle["valid"])
        self.assertIn("semantically unchanged", cycle["error"])

    def test_module_arm_enforces_stage_interface(self) -> None:
        incumbent = _adapter_code()
        invalid = _adapter_code(steps={"align": 2}, extra_helper=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = _write_study(root)
            candidate = root / "failed_candidate.py"
            candidate.write_text(incumbent, encoding="utf-8")
            with patch("maniskill_backend.repair_granularity.discover_environment", return_value={}), \
                    patch("maniskill_backend.repair_granularity.run_dynamic_trial", side_effect=[REPLAY_TRIAL]), \
                    patch("maniskill_backend.repair_granularity.gen_text", return_value=LLMTextResult(text=invalid, used_llm=True, model="test")):
                result = run_module_rewrite_repair(
                    study_path=study_path, run_dir=root / "output",
                    max_cycles=1, resume_adapter=candidate,
                )
        self.assertFalse(result["success"])
        cycle = result["cycles"][0]
        self.assertFalse(cycle["valid"])
        self.assertIn("policy helpers", cycle["error"])


class AblationOrchestrationTests(unittest.TestCase):
    def test_ablation_runs_paired_arms_with_advancing_sampling_seeds(self) -> None:
        incumbent = _adapter_code()
        changed = _adapter_code(steps={"align": 2, "lift": 2})
        replacement = _stage_method_source(_adapter_code(steps={"align": 2}), "align")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = _write_study(root, with_ablation_candidate="candidates/failed.py")
            candidate = root / "candidates" / "failed.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(incumbent, encoding="utf-8")
            candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            with patch.dict(os.environ, {"EM_LLM_SEED": "5"}), \
                    patch("maniskill_backend.selective_migration.discover_environment", return_value={}), \
                    patch("maniskill_backend.selective_migration.run_dynamic_trial", side_effect=[REPLAY_TRIAL, SUCCESS_TRIAL] * 2), \
                    patch("maniskill_backend.selective_migration.gen_text", return_value=LLMTextResult(text=replacement, used_llm=True, model="test")), \
                    patch("maniskill_backend.repair_granularity.discover_environment", return_value={}), \
                    patch("maniskill_backend.repair_granularity.run_dynamic_trial", side_effect=[REPLAY_TRIAL, SUCCESS_TRIAL] * 2), \
                    patch("maniskill_backend.repair_granularity.gen_text", return_value=LLMTextResult(text=changed, used_llm=True, model="test")):
                summary = run_granularity_ablation(
                    study_path=study_path, run_dir=root / "results",
                    resume_adapter=candidate, max_cycles=1, repetitions=2,
                )
            manifest = json.loads((root / "results" / "repetition_manifest.json").read_text(encoding="utf-8"))
            method_r1 = json.loads((root / "results" / "r01" / "method_patch" / "summary.json").read_text(encoding="utf-8"))
            module_r1 = json.loads((root / "results" / "r01" / "module_rewrite" / "summary.json").read_text(encoding="utf-8"))
            module_r2 = json.loads((root / "results" / "r02" / "module_rewrite" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sampling_seeds"], [5, 6])
        self.assertEqual(manifest["failed_candidate"]["sha256"], candidate_sha)
        self.assertEqual(summary["failed_candidate_sha256"], candidate_sha)
        self.assertEqual(method_r1["generation_settings"]["sampling_seed"], 5)
        self.assertEqual(module_r1["generation_settings"]["sampling_seed"], 5)
        self.assertEqual(module_r2["generation_settings"]["sampling_seed"], 6)
        rows = {row["arm_id"]: row for row in summary["arms"]}
        self.assertEqual(rows["method_patch"]["num_runs"], 2)
        self.assertEqual(rows["method_patch"]["num_success"], 2)
        self.assertEqual(rows["method_patch"]["target_action_steps"], 62)
        self.assertEqual(rows["method_patch"]["replay_action_steps_excluded"], 202)
        self.assertEqual(rows["module_rewrite"]["num_runs"], 2)
        self.assertEqual(rows["module_rewrite"]["num_success"], 2)
        self.assertEqual(rows["module_rewrite"]["target_action_steps"], 62)
        self.assertEqual(len(summary["paired_by_sampling_seed"]), 2)
        for paired in summary["paired_by_sampling_seed"]:
            self.assertTrue(paired["method_patch_success"])
            self.assertTrue(paired["module_rewrite_success"])

    def test_ablation_restores_sampling_seed_env(self) -> None:
        incumbent = _adapter_code()
        replacement = _stage_method_source(_adapter_code(steps={"align": 2}), "align")
        changed = _adapter_code(steps={"align": 2, "lift": 2})
        saved = os.environ.pop("EM_LLM_SEED", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                study_path = _write_study(root, with_ablation_candidate="candidates/failed.py")
                candidate = root / "candidates" / "failed.py"
                candidate.parent.mkdir(parents=True)
                candidate.write_text(incumbent, encoding="utf-8")
                with patch("maniskill_backend.selective_migration.discover_environment", return_value={}), \
                        patch("maniskill_backend.selective_migration.run_dynamic_trial", side_effect=[REPLAY_TRIAL, SUCCESS_TRIAL]), \
                        patch("maniskill_backend.selective_migration.gen_text", return_value=LLMTextResult(text=replacement, used_llm=True, model="test")), \
                        patch("maniskill_backend.repair_granularity.discover_environment", return_value={}), \
                        patch("maniskill_backend.repair_granularity.run_dynamic_trial", side_effect=[REPLAY_TRIAL, SUCCESS_TRIAL]), \
                        patch("maniskill_backend.repair_granularity.gen_text", return_value=LLMTextResult(text=changed, used_llm=True, model="test")):
                    run_granularity_ablation(
                        study_path=study_path, run_dir=root / "results",
                        resume_adapter=candidate, max_cycles=1, repetitions=1,
                    )
                self.assertNotIn("EM_LLM_SEED", os.environ)
        finally:
            if saved is not None:
                os.environ["EM_LLM_SEED"] = saved

    def test_unknown_arm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown repair-granularity arm"):
            run_granularity_arm(arm_id="mixed", study_path=Path("unused.json"), run_dir=Path("unused"),
                                resume_adapter=Path("unused.py"))

    def test_resolve_failed_candidate_reads_pinned_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = _write_study(root, with_ablation_candidate="candidates/failed.py")
            candidate = root / "candidates" / "failed.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(_adapter_code(), encoding="utf-8")
            self.assertEqual(resolve_failed_candidate(study_path), candidate)
            study = json.loads(study_path.read_text(encoding="utf-8"))
            pinned_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
            study["repair_granularity_ablation"]["failed_candidate_sha256"] = pinned_sha
            study_path.write_text(json.dumps(study), encoding="utf-8")
            self.assertEqual(resolve_failed_candidate(study_path), candidate)
            study["repair_granularity_ablation"]["failed_candidate_sha256"] = "0" * 64
            study_path.write_text(json.dumps(study), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                resolve_failed_candidate(study_path)
            (root / "candidates" / "failed.py").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                resolve_failed_candidate(study_path)

    def test_resolve_failed_candidate_requires_ablation_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = _write_study(root)
            with self.assertRaisesRegex(ValueError, "pass --resume-adapter"):
                resolve_failed_candidate(study_path)


class AblationSummaryTests(unittest.TestCase):
    def test_summary_counts_failures_and_excludes_replays(self) -> None:
        method_run = {
            "success": True, "status": "success", "repair_output_format": "single_stage_method",
            "resume_replay": {"action_steps": 101},
            "cycles": [
                {"valid": True, "action_steps": 31, "success": True, "trial_result": "t.json",
                 "usage": {"total_tokens": 1000, "cost": 0.001}},
            ],
        }
        module_run = {
            "success": False, "status": "target_budget_exhausted", "repair_output_format": "complete_module",
            "resume_replay": {"action_steps": 101},
            "cycles": [
                {"valid": True, "action_steps": 40, "success": False, "trial_result": "t.json",
                 "usage": {"total_tokens": 5000, "cost": 0.002}},
                {"valid": False, "action_steps": 0, "usage": {"total_tokens": 6000, "cost": 0.003}},
            ],
        }
        method_infra = {
            "success": False, "status": "llm_unavailable", "repair_output_format": "single_stage_method",
            "resume_replay": {"action_steps": 101},
            "cycles": [{"valid": False, "used_llm": False, "action_steps": 0, "usage": {}}],
        }
        payload = summarize_granularity_repetitions([
            {"sampling_seed": 0, "failed_candidate_sha256": "abc", "arms": {"method_patch": method_run, "module_rewrite": module_run}},
            {"sampling_seed": 1, "failed_candidate_sha256": "abc", "arms": {"method_patch": method_infra, "module_rewrite": module_run}},
        ])
        rows = {row["arm_id"]: row for row in payload["arms"]}
        self.assertEqual(rows["method_patch"]["num_runs"], 2)
        self.assertEqual(rows["method_patch"]["num_success"], 1)
        self.assertEqual(rows["method_patch"]["success_rate"], 0.5)
        self.assertEqual(rows["method_patch"]["infrastructure_failures"], 1)
        self.assertEqual(rows["method_patch"]["target_action_steps"], 31)
        self.assertEqual(rows["method_patch"]["replay_action_steps_excluded"], 202)
        self.assertEqual(rows["method_patch"]["cycles_to_success"], [1])
        self.assertEqual(rows["module_rewrite"]["num_runs"], 2)
        self.assertEqual(rows["module_rewrite"]["num_success"], 0)
        self.assertEqual(rows["module_rewrite"]["target_action_steps"], 80)
        self.assertEqual(rows["module_rewrite"]["generation_cycles"], 4)
        self.assertEqual(rows["module_rewrite"]["valid_candidates"], 2)
        self.assertEqual(rows["module_rewrite"]["total_tokens"], 22000)
        self.assertEqual(payload["failed_candidate_sha256"], "abc")
        self.assertEqual(payload["paired_by_sampling_seed"][0]["method_patch_success"], True)
        self.assertEqual(payload["paired_by_sampling_seed"][0]["module_rewrite_success"], False)
        self.assertIn("not from-scratch generation success", payload["interpretation_limit"])


if __name__ == "__main__":
    unittest.main()
