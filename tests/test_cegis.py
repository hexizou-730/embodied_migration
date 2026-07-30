import json
import tempfile
import unittest
from pathlib import Path

from maniskill_backend.active_probe import select_active_probe_plan
from maniskill_backend.cases import get_full_migration_case
from maniskill_backend.counterexample_loop import CEGISConfig, run_counterexample_loop
from maniskill_backend.counterexamples import (
    counterexample_from_trial,
    select_counterexample,
)
from maniskill_backend.embodiment_contracts import (
    compare_embodiments,
    get_embodiment_contract,
)
from maniskill_backend.evidence import build_evidence_ledger, detect_adapter_provenance
from maniskill_backend.failure_diagnosis import diagnose_failure
from maniskill_backend.module_generation_runner import build_module_generation_prompt
from maniskill_backend.structured_probe import get_probe_spec


class CEGISBackendTest(unittest.TestCase):
    def test_embodiment_contracts_capture_action_layout(self):
        xarm = get_embodiment_contract("xarm6")
        fetch = get_embodiment_contract("fetch")
        self.assertEqual(xarm.action_dim, 4)
        self.assertEqual(fetch.action_dim, 9)
        self.assertEqual(fetch.action_channels[-1].name, "base")
        self.assertTrue(xarm.validate_action_shape((4,))["valid"])
        self.assertFalse(xarm.validate_action_shape((9,))["valid"])
        diff = compare_embodiments("panda", "fetch")
        self.assertIn("action_dim", [item["field"] for item in diff["differences"]])
        self.assertIn("mobile_base", [item["field"] for item in diff["differences"]])

    def test_failure_diagnosis_names_violated_constraints(self):
        diagnosis = diagnose_failure(
            task_id="pull_cube",
            success=False,
            message="Episode ended during descent.",
            failure_type="contact execution failure",
            runtime_diagnostics={
                "stage": "descent",
                "tcp_stage_error_norm": 0.2,
                "tcp_cube_xy": 0.04,
                "cube_goal_xy": 0.2,
            },
        )
        self.assertEqual(diagnosis["reason"], "contact_side_reachability_failure")
        self.assertIn(
            "contact_pose_reachable_before_descent",
            diagnosis["violated_constraints"],
        )

    def test_counterexample_selection_prefers_complete_evidence(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        sparse = {
            "seed": 1,
            "success": False,
            "message": "failed",
            "failure_diagnosis": {
                "layer": "skill_adapter",
                "reason": "pull_skill_execution_failed",
                "confidence": 0.4,
            },
        }
        complete = {
            "seed": 5,
            "success": False,
            "message": "Episode ended during descent.",
            "runtime_diagnostics": {
                "stage": "descent",
                "tcp_stage_error_norm": 0.09,
                "tcp_cube_xy": 0.03,
                "cube_goal_xy": 0.2,
                "cube_pos": [0.0, 0.05, 0.02],
                "tcp_pos": [0.1, 0.05, 0.04],
            },
            "failure_diagnosis": {
                "layer": "contact_geometry",
                "reason": "contact_side_reachability_failure",
                "confidence": 0.9,
                "violated_constraints": [
                    "contact_pose_reachable_before_descent"
                ],
            },
        }
        selected = select_counterexample(case, [sparse, complete])
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.seed, 5)
        self.assertGreater(selected.information_score, 10.0)
        self.assertIn("contact_pose_reachable_before_descent", selected.violated_constraints)

    def test_active_probe_focuses_on_diagnosed_constraint(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        spec = get_probe_spec(case)
        counterexample = {
            "seed": 5,
            "failure_reason": "contact_side_reachability_failure",
            "violated_constraints": ["contact_pose_reachable_before_descent"],
        }
        plan = select_active_probe_plan(case, spec, counterexample, budget=6)
        self.assertEqual(plan["schema"], "active_probe_plan.v1")
        self.assertLessEqual(plan["candidate_count"], 6)
        self.assertIn("contact_x_offset", plan["controlled_parameters"])
        self.assertIn("drag_strength", plan["frozen_parameters"])
        self.assertTrue(plan["candidates"])

    def test_module_prompt_contains_contract_counterexample_and_guard(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        target_result = {
            "seed": 5,
            "success": False,
            "message": "Episode ended during descent.",
            "failure_diagnosis": {
                "layer": "contact_geometry",
                "reason": "contact_side_reachability_failure",
                "repair_hint": "Choose contact pose adaptively.",
                "evidence": {"stage": "descent", "tcp_stage_error_norm": 0.09},
            },
        }
        counterexample = counterexample_from_trial(case, target_result).to_dict()
        prompt = build_module_generation_prompt(
            case=case,
            target_result=target_result,
            attempts=[],
            counterexample=counterexample,
        )
        self.assertIn("Embodiment contract: xarm6_robotiq", prompt)
        self.assertIn("Selected simulation counterexample", prompt)
        self.assertIn("contact_pose_reachable_before_descent", prompt)
        self.assertIn("Generate guarded behavior", prompt)

    def test_cegis_dry_run_builds_reproducible_plan_without_simulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CEGISConfig(
                case_id="case02_pull_cube_panda_to_xarm6",
                development_seeds="0-4",
                held_out_seeds="100-104",
                max_cycles=2,
                dry_run=True,
            )
            payload = run_counterexample_loop(
                config,
                output_root=Path(tmpdir),
                run_name="dry",
            )
            self.assertEqual(payload["status"], "dry_run_planned")
            self.assertFalse(payload["success"])
            self.assertEqual(
                payload["evidence_policy"]["held_out_seeds_used_for_repair"],
                False,
            )
            summary_path = Path(tmpdir) / "dry" / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema"], "counterexample_guided_adapter_synthesis.v1")

    def test_evidence_ledger_does_not_call_oracle_llm_success(self):
        payload = build_evidence_ledger()
        fetch = next(
            item
            for item in payload["cases"]
            if item["case_id"] == "case01_pull_cube_panda_to_fetch"
        )
        self.assertEqual(fetch["adapter_provenance"], "hand_written_oracle")
        self.assertFalse(fetch["reproducible_success_in_tracked_artifacts"])

    def test_provenance_detector_recognizes_neutral_seed(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        target = Path(case.target_adapter_path)
        seed = Path(case.seed_adapter_path)
        self.assertEqual(
            detect_adapter_provenance(target, seed_path=seed),
            "neutral_seed",
        )


if __name__ == "__main__":
    unittest.main()
