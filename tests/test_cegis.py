import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maniskill_backend.active_probe import select_active_probe_plan
from maniskill_backend.cases import get_full_migration_case
from maniskill_backend.counterexample_loop import (
    HELD_OUT_FEEDBACK_POLICY,
    PROBE_BUDGET_SCOPE,
    CEGISConfig,
    _probe_batch_budget,
    _probe_case_count,
    _probe_command,
    run_counterexample_loop,
)
from maniskill_backend.counterexamples import (
    counterexample_from_trial,
    select_counterexample,
    write_counterexample_set,
)
from maniskill_backend.embodiment_contracts import (
    compare_embodiments,
    get_embodiment_contract,
    observe_runtime_contract,
)
from maniskill_backend.evidence import build_evidence_ledger, detect_adapter_provenance
from maniskill_backend.failure_diagnosis import diagnose_failure
from maniskill_backend.module_generation_runner import (
    build_module_generation_prompt,
    guarded_adapter_validation_error,
)
from maniskill_backend.paper_benchmark import PAPER_METHODS, build_paper_plan
from maniskill_backend.structured_probe import (
    fixed_grid_probe_batch,
    get_probe_spec,
    space_filling_probe_cases,
)
from scripts.structured_probe_runner import _summarize_measured_batch


class CEGISBackendTest(unittest.TestCase):
    def test_probe_budget_is_a_total_run_budget_and_command_uses_remaining_cases(self):
        self.assertEqual(PROBE_BUDGET_SCOPE, "per_run_total")
        self.assertEqual(_probe_batch_budget(8, 2), 4)
        self.assertEqual(_probe_batch_budget(8, 4), 2)
        self.assertEqual(_probe_batch_budget(6, 3), 2)
        self.assertEqual(_probe_batch_budget(4, 0), 0)
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        config = CEGISConfig(case_id=case.case_id, probe_budget=8)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            command = _probe_command(
                case,
                config,
                root,
                counterexample_path=root / "counterexample.json",
                previous_probe_path=None,
                seed=1,
                budget=3,
            )
            self.assertEqual(command[command.index("--max-cases") + 1], "3")
            self.assertEqual(command[command.index("--suggestion-budget") + 1], "3")
            self.assertEqual(command[command.index("--probe-selection") + 1], "active")
            result_path = root / "probe.json"
            result_path.write_text(
                json.dumps({"num_cases": 3, "all_probe_cases": [{}, {}, {}]}),
                encoding="utf-8",
            )
            self.assertEqual(_probe_case_count(result_path), 3)
            result_path.write_text(
                json.dumps(
                    {
                        "num_cases": 6,
                        "num_new_cases": 2,
                        "all_probe_cases": [{}, {}, {}, {}, {}, {}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_probe_case_count(result_path), 2)

            fixed = CEGISConfig(
                case_id=case.case_id,
                probe_budget=8,
                probe_strategy="fixed_grid",
            )
            fixed_command = _probe_command(
                case,
                fixed,
                root,
                counterexample_path=root / "counterexample.json",
                previous_probe_path=None,
                seed=2,
                budget=4,
                fixed_grid_offset=4,
            )
            self.assertEqual(
                fixed_command[fixed_command.index("--probe-selection") + 1],
                "fixed_grid",
            )
            self.assertEqual(
                fixed_command[fixed_command.index("--fixed-grid-total-budget") + 1],
                "8",
            )
            self.assertEqual(
                fixed_command[fixed_command.index("--fixed-grid-offset") + 1],
                "4",
            )

    def test_fixed_grid_batches_are_disjoint_slices_of_one_frozen_design(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        spec = get_probe_spec(case)
        complete = space_filling_probe_cases(spec, budget=8)
        first = fixed_grid_probe_batch(
            spec,
            total_budget=8,
            offset=0,
            batch_size=4,
        )
        second = fixed_grid_probe_batch(
            spec,
            total_budget=8,
            offset=4,
            batch_size=4,
        )
        self.assertEqual(first + second, complete)
        self.assertTrue(
            {item["case_index"] for item in first}.isdisjoint(
                item["case_index"] for item in second
            )
        )

    def test_probe_summary_carries_cumulative_history_but_marks_new_batch(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        spec = get_probe_spec(case)
        prior = {
            "all_probe_cases": [
                {
                    "contact_x_offset": 0.08,
                    "contact_z_offset": 0.01,
                    "approach_height": 0.06,
                    "drag_strength": -0.6,
                    "down_bias": -0.02,
                    "stages": 5,
                    "score": 1.0,
                    "probe_seed": 1,
                }
            ]
        }
        current = {
            "results": [
                {
                    "contact_x_offset": 0.12,
                    "contact_z_offset": 0.014,
                    "approach_height": 0.09,
                    "drag_strength": -0.8,
                    "down_bias": -0.02,
                    "stages": 5,
                    "score": 2.0,
                }
            ]
        }
        summary = _summarize_measured_batch(
            spec,
            current,
            adaptive_source=prior,
            diagnosis={},
            seed=5,
            top_k=8,
        )
        self.assertEqual(summary["num_cases"], 2)
        self.assertEqual(summary["num_previous_cases"], 1)
        self.assertEqual(summary["num_new_cases"], 1)
        self.assertEqual(summary["new_probe_cases"][0]["probe_seed"], 5)
        self.assertEqual(
            {item["probe_seed"] for item in summary["all_probe_cases"]},
            {1, 5},
        )

    def test_held_out_failure_stops_without_becoming_repair_feedback(self):
        import maniskill_backend.counterexample_loop as loop_module
        from unittest.mock import patch

        commands = []

        def fake_run_command(command, *, log_path, dry_run):
            commands.append(list(command))
            if "scripts/multiseed_eval.py" in command:
                output_dir = Path(command[command.index("--output-dir") + 1])
                jsonl_name = command[command.index("--jsonl-name") + 1]
                success = jsonl_name == "development.jsonl"
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / jsonl_name).write_text(
                    "\n".join(
                        [
                            json.dumps({"type": "metadata"}),
                            json.dumps({"type": "trial", "seed": 0, "success": success}),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return {"returncode": 0, "command": list(command), "dry_run": dry_run}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CEGISConfig(
                case_id="case02_pull_cube_panda_to_xarm6",
                development_seeds="0",
                held_out_seeds="100",
                max_cycles=3,
                min_trials_for_accept=1,
                from_zero=False,
            )
            with patch.object(loop_module, "_run_command", side_effect=fake_run_command):
                payload = run_counterexample_loop(
                    config,
                    output_root=Path(tmpdir),
                    run_name="held_out_rejection",
                )

        generation_commands = [
            command
            for command in commands
            if "maniskill_backend.module_generation_runner" in command
        ]
        self.assertEqual(payload["status"], "held_out_rejected")
        self.assertEqual(len(payload["cycles"]), 1)
        self.assertEqual(len(generation_commands), 1)
        self.assertFalse(payload["held_out"]["used_for_repair"])
        self.assertFalse(payload["cycles"][0]["repair_feedback"]["held_out_used"])
        self.assertEqual(
            payload["evidence_policy"]["held_out_feedback_policy"],
            HELD_OUT_FEEDBACK_POLICY,
        )

    def test_fixed_grid_cegis_reuses_outer_loop_and_advances_probe_offset(self):
        import maniskill_backend.counterexample_loop as loop_module
        from unittest.mock import patch

        commands = []
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        spec = get_probe_spec(case)

        def fake_run_command(command, *, log_path, dry_run):
            commands.append(list(command))
            if "scripts/multiseed_eval.py" in command:
                output_dir = Path(command[command.index("--output-dir") + 1])
                jsonl_name = command[command.index("--jsonl-name") + 1]
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / jsonl_name).write_text(
                    "\n".join(
                        [
                            json.dumps({"type": "metadata"}),
                            json.dumps(
                                {
                                    "type": "trial",
                                    "seed": 1,
                                    "success": False,
                                    "message": "Episode ended during descent.",
                                    "failure_diagnosis": {
                                        "layer": "contact_geometry",
                                        "reason": "contact_side_reachability_failure",
                                        "confidence": 0.9,
                                        "violated_constraints": [
                                            "contact_pose_reachable_before_descent"
                                        ],
                                    },
                                    "runtime_diagnostics": {
                                        "stage": "descent",
                                        "tcp_stage_error_norm": 0.2,
                                    },
                                }
                            ),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if "scripts/structured_probe_runner.py" in command:
                output_root = Path(command[command.index("--output-dir") + 1])
                batch = int(command[command.index("--max-cases") + 1])
                offset = int(command[command.index("--fixed-grid-offset") + 1])
                result_path = output_root / case.case_id / f"{spec.probe_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(
                        {
                            "schema": "structured_probe_result.v1",
                            "num_cases": offset + batch,
                            "num_new_cases": batch,
                            "all_probe_cases": [{}] * (offset + batch),
                        }
                    ),
                    encoding="utf-8",
                )
            return {"returncode": 0, "command": list(command), "dry_run": dry_run}

        with tempfile.TemporaryDirectory() as tmpdir:
            config = CEGISConfig(
                case_id=case.case_id,
                development_seeds="1",
                held_out_seeds="100",
                max_cycles=3,
                probe_budget=8,
                probe_strategy="fixed_grid",
                prompt_policy="full",
                min_trials_for_accept=1,
                from_zero=False,
            )
            with patch.object(loop_module, "_run_command", side_effect=fake_run_command):
                payload = run_counterexample_loop(
                    config,
                    output_root=Path(tmpdir),
                    run_name="fixed_grid_loop",
                )

        generation_commands = [
            command
            for command in commands
            if "maniskill_backend.module_generation_runner" in command
        ]
        probe_commands = [
            command
            for command in commands
            if "scripts/structured_probe_runner.py" in command
        ]
        self.assertEqual(len(generation_commands), 3)
        self.assertTrue(
            all(command[command.index("--prompt-policy") + 1] == "full" for command in generation_commands)
        )
        self.assertEqual(len(probe_commands), 2)
        self.assertEqual(
            [int(command[command.index("--fixed-grid-offset") + 1]) for command in probe_commands],
            [0, 4],
        )
        self.assertNotIn("--adaptive-from", probe_commands[0])
        self.assertIn("--adaptive-from", probe_commands[1])
        self.assertEqual(payload["probe_budget"]["cases_used"], 8)
        self.assertEqual(payload["config"]["probe_strategy"], "fixed_grid")

    def test_full_method_enforces_guarded_adapter_structure(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        unguarded = """
class Robot:
    pass

def build_robot(env, *, control_mode, robot_uid):
    return Robot()
"""
        self.assertIn("override pull", guarded_adapter_validation_error(unguarded, case))
        guarded = """
class Robot:
    def pull(self, obj, target):
        tcp = self._tcp_pos()
        cube = self._actor_pos("cube")
        if self._early_stop():
            return self._fail("pull", {}, "infeasible: episode ended")
        if tcp[0] > cube[0]:
            ok = self._pull_cube_success()
            return self._log("pull", {}, ok, ok, "")
        return self._fail("pull", {}, "infeasible: contact pose unreachable")

def build_robot(env, *, control_mode, robot_uid):
    return Robot()
"""
        self.assertIsNone(guarded_adapter_validation_error(guarded, case))

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

    def test_runtime_contract_observes_live_action_interface(self):
        class Controller:
            def __init__(self):
                self.controllers = {"arm": object(), "gripper_active": object()}

        space = SimpleNamespace(
            shape=(4,),
            dtype=np.dtype("float32"),
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        env = SimpleNamespace(
            action_space=space,
            unwrapped=SimpleNamespace(agent=SimpleNamespace(controller=Controller())),
        )
        observed = observe_runtime_contract(env, get_embodiment_contract("xarm6"))
        self.assertTrue(observed["valid"])
        self.assertEqual(observed["observed_action_shape"], [4])
        self.assertEqual(observed["controller_channels"], ["arm", "gripper_active"])
        self.assertTrue(observed["bounds_finite"])

        mismatch = observe_runtime_contract(env, get_embodiment_contract("fetch"))
        self.assertFalse(mismatch["valid"])

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

    def test_counterexample_selection_avoids_exact_cycle_repetition(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        trials = [
            {
                "seed": 1,
                "success": False,
                "message": "descent failed",
                "failure_diagnosis": {
                    "layer": "contact_geometry",
                    "reason": "contact_side_reachability_failure",
                    "confidence": 0.9,
                    "evidence": {"stage": "descent", "tcp_cube_xy": 0.2},
                },
                "runtime_diagnostics": {"stage": "descent", "tcp_cube_xy": 0.2},
            },
            {
                "seed": 2,
                "success": False,
                "message": "drag failed",
                "failure_diagnosis": {
                    "layer": "contact_geometry",
                    "reason": "contact_established_but_drag_progress_insufficient",
                    "confidence": 0.9,
                    "evidence": {"stage": "drag", "tcp_cube_xy": 0.03},
                },
                "runtime_diagnostics": {"stage": "drag", "tcp_cube_xy": 0.03},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = write_counterexample_set(
                Path(tmpdir) / "counterexamples.json",
                case,
                trials,
                selection_history=[{
                    "seed": 1,
                    "failure_reason": "contact_side_reachability_failure",
                    "stage": "descent",
                }],
            )
        self.assertEqual(payload["selection_policy"], "information_score_plus_history_novelty")
        self.assertEqual(payload["selected"]["seed"], 2)
        self.assertGreater(payload["selected"]["novelty_bonus"], 0.0)
        self.assertEqual(payload["portfolio_size"], 2)
        self.assertEqual(payload["selected"]["supporting_counterexamples"][0]["seed"], 1)
        self.assertNotEqual(
            payload["selected"]["supporting_counterexamples"][0]["failure_reason"],
            payload["selected"]["failure_reason"],
        )

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

    def test_active_probe_can_focus_from_constraint_without_known_reason(self):
        case = get_full_migration_case("case05_push_cube_panda_to_fetch")
        spec = get_probe_spec(case)
        plan = select_active_probe_plan(
            case,
            spec,
            {
                "seed": 3,
                "failure_reason": "new_unseen_failure_label",
                "violated_constraints": ["object_goal_error_decreases_during_drag"],
            },
            budget=4,
        )
        self.assertEqual(
            plan["controlled_parameters"],
            ["drag_strength", "stages", "down_bias"],
        )
        self.assertIn("contact_x_offset", plan["frozen_parameters"])

    def test_active_probe_uses_measured_kernel_ucb_after_first_batch(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        spec = get_probe_spec(case)
        counterexample = {
            "seed": 2,
            "failure_reason": "good_alignment_no_displacement_no_grasp",
            "violated_constraints": ["gripper_force_closure"],
        }
        previous = {
            "results": [
                {
                    "grasp_z_offset": 0.008,
                    "close_steps": 12,
                    "close_command": -0.6,
                    "settle_steps": 8,
                    "score": -8.0,
                },
                {
                    "grasp_z_offset": 0.016,
                    "close_steps": 24,
                    "close_command": -1.0,
                    "settle_steps": 16,
                    "score": 12.0,
                },
            ]
        }
        first = select_active_probe_plan(
            case,
            spec,
            counterexample,
            previous_probe=previous,
            budget=5,
        )
        second = select_active_probe_plan(
            case,
            spec,
            counterexample,
            previous_probe=previous,
            budget=5,
        )
        self.assertEqual(first["selection_mode"], "measurement_guided_kernel_ucb")
        self.assertEqual(first["optimizer"]["name"], "kernel_ucb")
        self.assertEqual(first["optimizer"]["observations"], 2)
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(len(first["candidates"]), 5)
        self.assertIn("surrogate_predicted_score", first["candidates"][0])
        self.assertIn("surrogate_uncertainty", first["candidates"][0])
        self.assertGreater(first["measured_parameter_sensitivity"]["grasp_z_offset"], 0.0)
        tried = {
            (0.008, 12, -0.6, 8),
            (0.016, 24, -1.0, 16),
        }
        selected = {
            (
                item["grasp_z_offset"],
                item["close_steps"],
                item["close_command"],
                item["settle_steps"],
            )
            for item in first["candidates"]
        }
        self.assertTrue(selected.isdisjoint(tried))

    def test_fetch_probe_covers_base_and_contact_constraints(self):
        case = get_full_migration_case("case01_pull_cube_panda_to_fetch")
        spec = get_probe_spec(case)
        self.assertEqual(spec.probe_id, "pull_cube_fetch_base_contact")
        self.assertIn("base_speed", spec.parameter_grid)
        self.assertIn("contact_x_offset", spec.parameter_grid)
        fixed = space_filling_probe_cases(spec, budget=8)
        self.assertEqual(len(fixed), 8)
        self.assertEqual(
            {float(item["base_speed"]) for item in fixed},
            {-0.3, 0.15, 0.3},
        )
        self.assertTrue(
            all(item["suggestion_reason"] == "deterministic_space_filling_fixed_grid" for item in fixed)
        )

        counterexample = {
            "seed": 2,
            "failure_reason": "contact_side_reachability_failure",
            "violated_constraints": ["contact_pose_reachable_before_descent"],
        }
        active = select_active_probe_plan(case, spec, counterexample, budget=8)
        self.assertIn("base_speed", active["controlled_parameters"])
        self.assertIn("base_steps", active["controlled_parameters"])
        self.assertLessEqual(active["candidate_count"], 8)

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

    def test_prompt_ablation_policies_do_not_leak_full_method_feedback(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        target_result = {
            "success": False,
            "message": "Episode ended during descent.",
            "failure_diagnosis": {
                "layer": "contact_geometry",
                "reason": "contact_side_reachability_failure",
                "repair_hint": "Choose contact pose adaptively.",
            },
        }
        counterexample = {
            "seed": 5,
            "failure_reason": "contact_side_reachability_failure",
            "violated_constraints": ["contact_pose_reachable_before_descent"],
        }
        probe = {"prompt_feedback": "measured contact_x_offset=0.08"}
        one_shot = build_module_generation_prompt(
            case=case,
            target_result=target_result,
            attempts=[],
            counterexample=counterexample,
            probe_feedback=probe,
            prompt_policy="one_shot",
        )
        self.assertIn("prompt_policy: one_shot", one_shot)
        self.assertNotIn("Embodiment contract: xarm6_robotiq", one_shot)
        self.assertNotIn("Diagnosis-guided repair instruction", one_shot)
        self.assertNotIn("measured contact_x_offset=0.08", one_shot)
        self.assertNotIn("Generate guarded behavior", one_shot)

        full = build_module_generation_prompt(
            case=case,
            target_result=target_result,
            attempts=[],
            counterexample=counterexample,
            probe_feedback=probe,
            prompt_policy="full",
        )
        self.assertIn("Embodiment contract: xarm6_robotiq", full)
        self.assertIn("Diagnosis-guided repair instruction", full)
        self.assertIn("measured contact_x_offset=0.08", full)
        self.assertIn("Generate guarded behavior", full)

    def test_paper_plan_freezes_repetitions_and_held_out_split(self):
        plan = build_paper_plan(tier_id="paper", plan_name="unit")
        self.assertEqual(plan["tier"]["development_seeds"], "0-9")
        self.assertEqual(plan["tier"]["held_out_seeds"], "100-129")
        self.assertEqual(plan["tier"]["repetitions"], 3)
        self.assertEqual(plan["num_runs"], len(PAPER_METHODS) * 5 * 3)
        self.assertEqual(plan["num_blocked_runs"], 9)
        ours = [run for run in plan["runs"] if run["method_id"] == "Ours"]
        self.assertEqual(len(ours), 15)
        self.assertEqual(sum(run["ready"] for run in ours), 15)
        hypotheses = {item["hypothesis_id"]: item for item in plan["registered_hypotheses"]}
        self.assertEqual(hypotheses["H1"]["population"], "official_supported")
        self.assertEqual(hypotheses["H2"]["comparator_method"], "B5")
        self.assertTrue(all(item["applicable"] for item in hypotheses.values()))

        pilot_resources = build_paper_plan(
            tier_id="pilot", plan_name="resource_unit"
        )["resource_upper_bounds"]
        self.assertEqual(pilot_resources["total_simulator_episodes"], 715)
        self.assertEqual(pilot_resources["total_llm_calls"], 65)
        self.assertEqual(pilot_resources["total_probe_cases"], 80)
        self.assertEqual(
            pilot_resources["by_method"]["B5"]["probe_cases"],
            pilot_resources["by_method"]["Ours"]["probe_cases"],
        )
        self.assertEqual(
            pilot_resources["by_method"]["B5"]["simulator_episodes"],
            pilot_resources["by_method"]["Ours"]["simulator_episodes"],
        )
        self.assertEqual(PAPER_METHODS["B5"].runner_kind, "cegis")
        self.assertEqual(PAPER_METHODS["B5"].prompt_policy, "full")
        self.assertEqual(PAPER_METHODS["B5"].probe_strategy, "fixed_grid")
        self.assertEqual(PAPER_METHODS["Ours"].probe_strategy, "active")
        pilot_plan = build_paper_plan(tier_id="pilot", plan_name="budget_unit")
        pilot_budget = pilot_plan["budget_contract"]["llm_api_calls_per_run"]
        self.assertEqual(pilot_budget["B2"], 1)
        self.assertEqual(
            {pilot_budget[item] for item in ("B3", "B4", "B5", "Ours")},
            {3},
        )

        paper_resources = plan["resource_upper_bounds"]
        self.assertEqual(paper_resources["total_simulator_episodes"], 6585)
        self.assertEqual(paper_resources["total_environment_steps"], 3292500)
        self.assertEqual(paper_resources["total_llm_calls"], 315)
        self.assertEqual(paper_resources["total_probe_cases"], 240)
        paper_budget = plan["budget_contract"]["llm_api_calls_per_run"]
        self.assertEqual(
            {paper_budget[item] for item in ("B3", "B4", "B5", "Ours")},
            {5},
        )
        for run in plan["runs"]:
            command = run["command"]
            observed = int(command[command.index("--llm-call-budget") + 1])
            self.assertEqual(observed, paper_budget[run["method_id"]])

        smoke = build_paper_plan(
            tier_id="pilot",
            method_ids=["B0", "Ours"],
            case_ids=["case02_pull_cube_panda_to_xarm6"],
            plan_name="smoke",
            dry_run_commands=True,
        )
        self.assertEqual(smoke["num_ready_runs"], 2)
        self.assertTrue(all(run["command"][-1] == "--dry-run" for run in smoke["runs"]))

    def test_push_case_has_probe_and_guarded_contract(self):
        case = get_full_migration_case("case05_push_cube_panda_to_fetch")
        probe = get_probe_spec(case)
        self.assertEqual(probe.probe_id, "push_cube_fetch_base_contact")
        self.assertIn("base_speed", probe.parameter_grid)
        guarded = """
class Robot:
    def push(self, obj, target):
        tcp = self._tcp_pos()
        cube = self._actor_pos("cube")
        if self._early_stop():
            return self._fail("push", {}, "episode ended")
        if tcp[0] < cube[0]:
            ok = self._push_cube_success()
            return self._log("push", {}, ok, ok, "")
        return self._fail("push", {}, "rear side unreachable")

def build_robot(env, *, control_mode, robot_uid):
    return Robot()
"""
        self.assertIsNone(guarded_adapter_validation_error(guarded, case))

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
            generation_command = payload["cycles"][0]["generation"]["command"]
            self.assertIn("--no-implicit-probe-feedback", generation_command)
            preview = payload["nested_command_preview"]
            active_probe = preview["structured_probe"]["command"]
            repair = preview["repair_generation"]["command"]
            self.assertEqual(
                active_probe[active_probe.index("--probe-selection") + 1],
                "active",
            )
            self.assertNotIn("--fixed-grid-total-budget", active_probe)
            self.assertIn("--counterexample-json", repair)
            self.assertIn("--probe-feedback-json", repair)
            self.assertEqual(repair[repair.index("--prompt-policy") + 1], "full")
            self.assertEqual(
                preview["counterexample_selection"]["held_out_used"],
                False,
            )

            fixed = run_counterexample_loop(
                CEGISConfig(
                    case_id="case02_pull_cube_panda_to_xarm6",
                    development_seeds="0-4",
                    held_out_seeds="100-104",
                    max_cycles=2,
                    probe_strategy="fixed_grid",
                    dry_run=True,
                ),
                output_root=Path(tmpdir),
                run_name="fixed",
            )
            fixed_probe = fixed["nested_command_preview"]["structured_probe"]["command"]
            self.assertEqual(
                fixed_probe[fixed_probe.index("--probe-selection") + 1],
                "fixed_grid",
            )
            self.assertEqual(
                fixed_probe[fixed_probe.index("--fixed-grid-total-budget") + 1],
                "8",
            )
            self.assertEqual(
                fixed_probe[fixed_probe.index("--fixed-grid-offset") + 1],
                "0",
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
