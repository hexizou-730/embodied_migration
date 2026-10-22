from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from maniskill_backend.dynamic_harness import DynamicMigrationSpec
from maniskill_backend.selective_comparison import _baseline_prompt, build_comparison_summary, summarize_repetitions, validate_reference_result


class SelectiveComparisonTests(unittest.TestCase):
    def test_aggregate_keeps_failures_in_the_denominator(self) -> None:
        methods = ("target_only", "whole_rewrite", "selective")
        first = {"protocol": {"sampling_seed": 0}, "rows": [
            {"method_id": method, "success": method == "selective", "status": "success" if method == "selective" else "target_budget_exhausted"}
            for method in methods
        ]}
        second = {"protocol": {"sampling_seed": 1}, "rows": [
            {"method_id": method, "success": False, "status": "llm_unavailable"}
            for method in methods
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            result = summarize_repetitions([first, second], output_path=Path(tmp) / "summary.json")
            self.assertEqual(result["rows"][2]["num_runs"], 2)
            self.assertEqual(result["rows"][2]["success_rate"], 0.5)
            self.assertEqual(result["rows"][2]["infrastructure_failures"], 1)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                summarize_repetitions([{"protocol": {}, "rows": []}], output_path=Path(tmp) / "bad.json")

    def setUp(self) -> None:
        self.spec = DynamicMigrationSpec(
            env_id="StackCube-v1",
            task_label="t07_stack_cube",
            source_robot="panda",
            target_robot="xarm6_robotiq",
            source_control_mode="pd_joint_pos",
            target_control_mode="pd_ee_delta_pose",
        )
        self.observation = {
            "env_id": "StackCube-v1",
            "robot_uid": "xarm6_robotiq",
            "action_space": {"shape": [7]},
            "official_evaluation": {"success": False},
        }

    def test_target_only_prompt_excludes_source_program(self) -> None:
        prompt = _baseline_prompt(
            method_id="target_only",
            spec=self.spec,
            observation=self.observation,
            source_code="SECRET_SOURCE_MARKER",
            stage_program={"stage_order": ["approach"]},
            incumbent_code="",
            previous_result={},
            latest_error="",
        )
        self.assertNotIn("SECRET_SOURCE_MARKER", prompt)
        self.assertNotIn('"stage_order"', prompt)
        self.assertIn("No source robot program", prompt)
        self.assertIn("_move_towards(world_target, gripper=value, steps=count)", prompt)
        self.assertIn("neither accepts tol", prompt)
        self.assertIn("gripper_open=0.0, gripper_close=0.81", prompt)

    def test_reused_reference_rejects_old_gripper_contract_and_seed(self) -> None:
        study = {
            "task": {"env_id": self.spec.env_id},
            "source": {"robot_uid": "panda", "control_mode": "pd_joint_pos"},
            "target": {"robot_uid": "xarm6_robotiq", "control_mode": "pd_ee_delta_pose",
                       "observed_gripper_open": 0.0, "observed_gripper_closed": 0.81},
        }
        reference = {
            "spec": {**vars(self.spec), "seed": 0, "max_target_cycles": 3},
            "runtime_contract": {"gripper_open": 0.0, "gripper_close": 0.81},
            "source_metadata_filter": "extracted_source_facts_only",
        }
        validate_reference_result(reference, study, seed=0, max_cycles=3)
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_reference_result(reference, study, seed=1, max_cycles=3)
        with self.assertRaisesRegex(ValueError, "gripper_open"):
            validate_reference_result({**reference, "runtime_contract": {}}, study, seed=0, max_cycles=3)
        with self.assertRaisesRegex(ValueError, "resume_replay"):
            validate_reference_result({**reference, "resume_replay": {}}, study, seed=0, max_cycles=3)
        with self.assertRaisesRegex(ValueError, "source_metadata_filter"):
            validate_reference_result({**reference, "source_metadata_filter": None}, study, seed=0, max_cycles=3)

    def test_whole_rewrite_prompt_includes_same_source_evidence(self) -> None:
        prompt = _baseline_prompt(
            method_id="whole_rewrite",
            spec=self.spec,
            observation=self.observation,
            source_code="SECRET_SOURCE_MARKER",
            stage_program={"stage_order": ["approach", "grasp"], "validation": {"manual_evaluation_only": "MANUAL_LABEL_SECRET"}},
            incumbent_code="",
            previous_result={},
            latest_error="",
        )
        self.assertIn("SECRET_SOURCE_MARKER", prompt)
        self.assertIn('"stage_order"', prompt)
        self.assertIn("No stage is locked", prompt)
        self.assertNotIn("MANUAL_LABEL_SECRET", prompt)

    def test_comparison_summary_uses_identical_metrics(self) -> None:
        result = {
            "status": "target_budget_exhausted",
            "success": False,
            "cycles": [
                {
                    "valid": True,
                    "trial_result": "cycle_01/trial_result.json",
                    "action_steps": 6,
                    "failure_reason": "contact_not_established",
                    "usage": {"total_tokens": 100, "cost": 0.01},
                }
            ],
        }
        selective = {
            **result,
            "locked_stages": ["approach"],
            "cycles": [
                {
                    "valid": True,
                    "trial_result": "cycle_01/trial_result.json",
                    "action_steps": 6,
                    "failed_stage": "grasp",
                    "usage": {"total_tokens": 120, "cost": 0.02},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            summary = build_comparison_summary(
                method_results={
                    "target_only": result,
                    "whole_rewrite": result,
                    "selective": selective,
                },
                output_path=path,
                protocol={"seed": 0},
            )
            persisted = json.loads(path.read_text())
        self.assertEqual(summary, persisted)
        self.assertEqual(summary["rows"][0]["target_action_steps"], 6)
        self.assertEqual(summary["rows"][0]["trials_with_actions"], 1)
        self.assertEqual(summary["rows"][0]["last_runtime_failure"], "contact_not_established")
        self.assertEqual(summary["rows"][0]["last_physical_failure"], "contact_not_established")
        self.assertIsNone(summary["rows"][0]["last_generation_error"])
        self.assertEqual(summary["rows"][2]["completed_stages"], ["approach"])
        self.assertEqual(summary["rows"][2]["last_failure"], "grasp")

    def test_official_success_is_not_classified_as_a_failure(self) -> None:
        result = {"success": True, "status": "success", "cycles": [
            {"success": False, "failure_reason": "contact_not_established", "action_steps": 10},
            {"success": True, "failure_reason": "official_task_success", "action_steps": 12},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_comparison_summary(
                method_results={method: result for method in ("target_only", "whole_rewrite", "selective")},
                output_path=Path(tmp) / "summary.json", protocol={},
            )
        for row in summary["rows"]:
            self.assertTrue(row["success"])
            self.assertEqual(row["last_runtime_failure"], "contact_not_established")
            self.assertEqual(row["last_physical_failure"], "contact_not_established")


if __name__ == "__main__":
    unittest.main()
