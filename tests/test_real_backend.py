import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lmp.executor import execute_lmp
from maniskill_backend.agent_planner import fallback_agent_actions, plan_agent_actions, validate_agent_plan
from maniskill_backend.autonomous_harness import (
    build_harness_plan,
    load_multiseed_jsonl,
    select_failure_seed,
    tool_inventory,
)
from maniskill_backend.cases import (
    PRIMARY_FULL_MIGRATION_CASE,
    PRIMARY_FULL_MIGRATION_CASE_ID,
    get_full_migration_case,
)
from maniskill_backend.evaluation import TrialRecord, classify_failure, classify_failure_layer
from maniskill_backend.failure_diagnosis import diagnose_failure
from maniskill_backend.generalization import build_generalization_report, generalization_report_to_markdown
from maniskill_backend.iterative_runner import _code_diff, build_iterative_prompt
from maniskill_backend.migration import METHODS, MigrationRequest, build_migration_prompt
from maniskill_backend.module_generation_runner import (
    build_module_generation_prompt,
    extract_python_module,
    pick_cube_runtime_diagnostic_error,
    validate_generated_adapter_module,
)
from maniskill_backend.profiles import get_robot_profile, iter_robot_profiles
from maniskill_backend.real_runner import (
    _build_robot_adapter,
    _build_robot_adapter_from_module,
    _default_control_mode,
)
from maniskill_backend.reporting import build_oracle_code, build_real_failure_report, success_from_ret_val
from maniskill_backend.results import append_jsonl, summarize_records
from maniskill_backend.skill_adapter import ManiSkillPickCubeRobot, ManiSkillPullCubeRobot, ManiSkillSceneAdapter
from maniskill_backend.structured_probe import (
    get_probe_spec,
    probe_grid,
    suggest_next_probe_cases,
    summarize_probe_results,
)
from maniskill_backend.tasks import get_task_spec, iter_task_specs
from maniskill_backend.view import records_to_md
from llm_client import api_key_env, current_provider, deepseek_thinking_mode, default_model


class RealBackendTest(unittest.TestCase):
    def test_current_scope_includes_pull_and_pick_cube(self):
        self.assertEqual([profile.name for profile in iter_robot_profiles()], ["panda", "fetch", "xarm6_robotiq"])
        self.assertEqual([task.task_id for task in iter_task_specs()], ["pull_cube", "pick_cube"])
        task = get_task_spec("PullCube-v1")
        self.assertEqual(task.task_id, "pull_cube")
        self.assertEqual(task.maniskill_env_id, "PullCube-v1")
        self.assertEqual(task.source_robot, "panda")
        self.assertEqual(task.target_robots, ("panda", "fetch", "xarm6_robotiq"))
        self.assertIn("robot.pull", task.source_program)
        pick = get_task_spec("PickCube-v1")
        self.assertEqual(pick.task_id, "pick_cube")
        self.assertEqual(pick.maniskill_env_id, "PickCube-v1")
        self.assertEqual(pick.target_robots, ("panda", "xarm6_robotiq"))
        self.assertIn("robot.grasp", pick.source_program)
        self.assertIn("robot.place", pick.source_program)

    def test_primary_case_is_xarm6_pick_cube(self):
        case = get_full_migration_case(PRIMARY_FULL_MIGRATION_CASE_ID)
        self.assertIs(case, PRIMARY_FULL_MIGRATION_CASE)
        self.assertEqual(case.case_id, "case03_pick_cube_panda_to_xarm6")
        self.assertEqual(case.task_id, "pick_cube")
        self.assertEqual(case.source_robot, "panda")
        self.assertEqual(case.target_robot, "xarm6_robotiq")
        self.assertEqual(case.target_control_mode, "pd_ee_delta_pos")
        self.assertEqual(case.target_program_path, "maniskill_backend/case_programs/case03_pick_cube.py")
        self.assertEqual(case.target_adapter_module, "maniskill_backend.generated_adapters.case03_xarm6_pick_cube")
        self.assertEqual(case.target_adapter_path, "maniskill_backend/generated_adapters/case03_xarm6_pick_cube.py")
        self.assertIn("grasp_geometry", case.migration_layers)

    def test_case01_remains_fetch_failure_case(self):
        case = get_full_migration_case("case01_pull_cube_panda_to_fetch")
        self.assertEqual(case.target_robot, "fetch")
        self.assertEqual(case.target_adapter_module, "maniskill_backend.generated_adapters.case01_fetch_pull_cube")

    def test_case02_remains_xarm6_pull_success_case(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        self.assertEqual(case.task_id, "pull_cube")
        self.assertEqual(case.target_robot, "xarm6_robotiq")
        self.assertEqual(case.target_adapter_module, "maniskill_backend.generated_adapters.case02_xarm6_pull_cube")

    def test_profiles_are_promptable(self):
        panda = get_robot_profile("panda").to_prompt_section()
        fetch = get_robot_profile("fetch").to_prompt_section()
        xarm = get_robot_profile("xarm6_robotiq").to_prompt_section()
        self.assertIn("Robot Profile: panda", panda)
        self.assertIn("Robot Profile: fetch", fetch)
        self.assertIn("Robot Profile: xarm6_robotiq", xarm)
        self.assertIn("mobile_base", fetch)
        self.assertIn("fixed-base", xarm)

    def test_method_set_is_real_only(self):
        self.assertEqual(METHODS, ("source-copy", "llm_card_report", "oracle"))

    def test_deepseek_provider_config(self):
        with patch.dict(os.environ, {"EM_LLM_PROVIDER": "deepseek"}, clear=True):
            self.assertEqual(current_provider(), "deepseek")
            self.assertEqual(default_model(), "deepseek-v4-pro")
            self.assertEqual(api_key_env(), "DEEPSEEK_API_KEY")
            self.assertEqual(deepseek_thinking_mode(), "disabled")

    def test_removed_task_specs_are_not_current_scope(self):
        for old_task in ("stack_cube", "peg_insertion", "pull_cube_tool"):
            with self.assertRaises(KeyError):
                get_task_spec(old_task)

    def test_module_generation_extracts_complete_python_module(self):
        text = """Here is the module:
```python
from typing import Any

def build_robot(env: Any, *, control_mode: str, robot_uid: str):
    return object()
```"""
        module = extract_python_module(text)
        self.assertIn("def build_robot", module)
        validate_generated_adapter_module(module)

    def test_module_generation_rejects_unsafe_module(self):
        unsafe = """import subprocess

def build_robot(env, *, control_mode: str, robot_uid: str):
    subprocess.run(["echo", "bad"])
"""
        with self.assertRaises(ValueError):
            validate_generated_adapter_module(unsafe)

    def test_module_generation_allows_position_variable_methods(self):
        safe = """import numpy as np

def build_robot(env, *, control_mode: str, robot_uid: str):
    goal_pos = np.zeros(3)
    return goal_pos.copy()
"""
        validate_generated_adapter_module(safe)

    def test_module_generation_prompt_requests_pull_cube_adapter(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        prompt = build_module_generation_prompt(
            case=case,
            target_result={
                "success": False,
                "failure_layer": "controller_primitive",
                "message": "cube was not pulled to target",
            },
            attempts=[],
        )
        self.assertIn("complete Python module", prompt)
        self.assertIn("ManiSkillPullCubeRobot", prompt)
        self.assertIn(case.target_adapter_path, prompt)
        self.assertIn("Migration design space", prompt)
        self.assertIn("infeasible:", prompt)

    def test_module_generation_retry_prompt_requires_strategy_change(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        failure = {
            "success": False,
            "failure_layer": "skill_adapter",
            "message": "cube_goal_xy=0.2000, tcp_cube_xy=0.3006",
        }
        prompt = build_module_generation_prompt(
            case=case,
            target_result=failure,
            attempts=[
                {
                    "round": 1,
                    "module_valid": True,
                    "module_kept": True,
                    "verification_ok": True,
                    "target_result": failure,
                }
            ],
        )
        self.assertIn("Do not return a module identical", prompt)
        self.assertIn("farther positive-x sweep start", prompt)

    def test_module_generation_prompt_requests_pick_cube_grasp_adapter(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        prompt = build_module_generation_prompt(
            case=case,
            target_result={
                "success": False,
                "failure_layer": "skill_adapter",
                "message": "cube was not grasped",
            },
            attempts=[],
        )
        self.assertIn("ManiSkillPickCubeRobot", prompt)
        self.assertIn("real grasping task", prompt)
        self.assertIn("robot.grasp(cube), then robot.place(cube, goal)", prompt)
        self.assertIn("self._is_grasping('cube')", prompt)
        self.assertIn("no chasing a displaced cube", prompt)
        self.assertIn("preserve the grasp, set held_object", prompt)
        self.assertIn("tcp_grasp_xy=...", prompt)
        self.assertIn("tcp_grasp_z=...", prompt)
        self.assertIn("cube_disp_xy=...", prompt)
        self.assertNotIn("cube_pos.z=-0.8996", prompt)
        self.assertNotIn("farther positive-x sweep start", prompt)

    def test_module_generation_prompt_includes_pick_probe_feedback(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        generic_prompt = (
            Path("results/structured_probes")
            / case.case_id
            / "pick_cube_xarm6_close_envelope_prompt.txt"
        )
        generic_json = generic_prompt.with_name("pick_cube_xarm6_close_envelope.json")
        feedback_path = Path("results/xarm6_pick_grasp_probe_prompt.txt")
        previous = feedback_path.read_text(encoding="utf-8") if feedback_path.exists() else None
        previous_generic_prompt = generic_prompt.read_text(encoding="utf-8") if generic_prompt.exists() else None
        previous_generic_json = generic_json.read_text(encoding="utf-8") if generic_json.exists() else None
        generic_prompt.unlink(missing_ok=True)
        generic_json.unlink(missing_ok=True)
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(
            "best_probe_case:\n"
            "  grasp_z_offset=0.012, close_steps=24, close_command=-0.6, "
            "cube_disp_xy=0.003, is_grasping_after_close=True\n",
            encoding="utf-8",
        )
        try:
            prompt = build_module_generation_prompt(
                case=case,
                target_result={"success": False, "failure_layer": "skill_adapter", "message": "cube was not grasped"},
                attempts=[],
            )
            self.assertIn("Structured probe feedback", prompt)
            self.assertIn("legacy `scripts/xarm6_pick_grasp_probe.py`", prompt)
            self.assertIn("grasp_z_offset=0.012", prompt)
            self.assertIn("is_grasping_after_close=True", prompt)
        finally:
            if previous is None:
                feedback_path.unlink(missing_ok=True)
            else:
                feedback_path.write_text(previous, encoding="utf-8")
            generic_prompt.parent.mkdir(parents=True, exist_ok=True)
            if previous_generic_prompt is not None:
                generic_prompt.write_text(previous_generic_prompt, encoding="utf-8")
            if previous_generic_json is not None:
                generic_json.write_text(previous_generic_json, encoding="utf-8")

    def test_module_generation_prompt_includes_generic_structured_probe_feedback(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        probe_dir = Path("results/structured_probes") / case.case_id
        prompt_path = probe_dir / "pick_cube_xarm6_close_envelope_prompt.txt"
        json_path = probe_dir / "pick_cube_xarm6_close_envelope.json"
        previous_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None
        previous_json = json_path.read_text(encoding="utf-8") if json_path.exists() else None
        probe_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            "Structured probe feedback.\n"
            "probe_id=pick_cube_xarm6_close_envelope\n"
            "successful_probe_cases=0\n"
            "best_probe_case:\n"
            "  grasp_z_offset=0.016\n"
            "  cube_disp_xy=0.00458\n",
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(
                {
                    "schema": "structured_probe_result.v1",
                    "probe_id": "pick_cube_xarm6_close_envelope",
                    "case_id": case.case_id,
                    "dry_run": False,
                    "num_cases": 32,
                    "num_success": 0,
                }
            ),
            encoding="utf-8",
        )
        try:
            prompt = build_module_generation_prompt(
                case=case,
                target_result={"success": False, "failure_layer": "skill_adapter", "message": "cube was not grasped"},
                attempts=[],
            )
            self.assertIn("Structured probe feedback", prompt)
            self.assertIn("scripts/structured_probe_runner.py", prompt)
            self.assertIn("pick_cube_xarm6_close_envelope_prompt.txt", prompt)
            self.assertIn("probe_id=pick_cube_xarm6_close_envelope", prompt)
            self.assertIn("grasp_z_offset=0.016", prompt)
            self.assertNotIn("legacy `scripts/xarm6_pick_grasp_probe.py`", prompt)
        finally:
            if previous_prompt is None:
                prompt_path.unlink(missing_ok=True)
            else:
                prompt_path.write_text(previous_prompt, encoding="utf-8")
            if previous_json is None:
                json_path.unlink(missing_ok=True)
            else:
                json_path.write_text(previous_json, encoding="utf-8")

    def test_module_generation_prompt_includes_pull_structured_probe_feedback(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        probe_dir = Path("results/structured_probes") / case.case_id
        prompt_path = probe_dir / "pull_cube_xarm6_contact_geometry_prompt.txt"
        json_path = probe_dir / "pull_cube_xarm6_contact_geometry.json"
        previous_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else None
        previous_json = json_path.read_text(encoding="utf-8") if json_path.exists() else None
        probe_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            "Structured probe feedback.\n"
            "probe_id=pull_cube_xarm6_contact_geometry\n"
            "successful_probe_cases=1\n"
            "best_probe_case:\n"
            "  contact_x_offset=0.12\n"
            "  drag_strength=-0.8\n"
            "  cube_goal_improvement=0.19\n",
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(
                {
                    "schema": "structured_probe_result.v1",
                    "probe_id": "pull_cube_xarm6_contact_geometry",
                    "case_id": case.case_id,
                    "dry_run": False,
                    "num_cases": 32,
                    "num_success": 1,
                }
            ),
            encoding="utf-8",
        )
        try:
            prompt = build_module_generation_prompt(
                case=case,
                target_result={"success": False, "failure_layer": "skill_adapter", "message": "cube was not pulled"},
                attempts=[],
            )
            self.assertIn("Structured probe feedback", prompt)
            self.assertIn("pull_cube_xarm6_contact_geometry_prompt.txt", prompt)
            self.assertIn("probe_id=pull_cube_xarm6_contact_geometry", prompt)
            self.assertIn("contact_x_offset=0.12", prompt)
            self.assertIn("drag_strength=-0.8", prompt)
        finally:
            if previous_prompt is None:
                prompt_path.unlink(missing_ok=True)
            else:
                prompt_path.write_text(previous_prompt, encoding="utf-8")
            if previous_json is None:
                json_path.unlink(missing_ok=True)
            else:
                json_path.write_text(previous_json, encoding="utf-8")

    def test_xarm6_pick_grasp_probe_script_documents_parameter_sweep(self):
        script = Path("scripts/xarm6_pick_grasp_probe.py").read_text(encoding="utf-8")
        self.assertIn("grasp_z_offsets", script)
        self.assertIn("close_steps", script)
        self.assertIn("close_commands", script)
        self.assertIn("cube_disp_xy", script)
        self.assertIn("is_grasping_after_close", script)
        self.assertIn("xarm6_pick_grasp_probe_prompt.txt", script)

    def test_xarm6_pull_contact_probe_script_documents_parameter_sweep(self):
        script = Path("scripts/xarm6_pull_contact_probe.py").read_text(encoding="utf-8")
        self.assertIn("contact_x_offsets", script)
        self.assertIn("contact_z_offsets", script)
        self.assertIn("approach_heights", script)
        self.assertIn("drag_strengths", script)
        self.assertIn("cube_goal_improvement", script)
        self.assertIn("tcp_contact_xy", script)
        self.assertIn("xarm6_pull_contact_probe_prompt.txt", script)

    def test_structured_probe_spec_expands_pick_close_envelope_grid(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        spec = get_probe_spec(
            case,
            diagnosis={"reason": "good_alignment_no_displacement_no_grasp"},
        )
        self.assertEqual(spec.probe_id, "pick_cube_xarm6_close_envelope")
        self.assertEqual(spec.case_id, case.case_id)
        self.assertEqual(spec.robot_uid, "xarm6_robotiq")
        self.assertIn("grasp_z_offset", spec.parameter_grid)
        self.assertIn("close_steps", spec.parameter_grid)
        self.assertIn("close_command", spec.parameter_grid)
        self.assertIn("settle_steps", spec.parameter_grid)
        grid = probe_grid(spec)
        self.assertEqual(len(grid), 32)
        self.assertEqual(grid[0]["case_index"], 1)
        self.assertIn("cube_disp_xy", spec.primary_metrics)
        self.assertIn("is_grasping_after_close", spec.success_keys)

    def test_structured_probe_spec_expands_pull_contact_grid(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        spec = get_probe_spec(case, diagnosis={"reason": "pull_contact_failure"})
        self.assertEqual(spec.probe_id, "pull_cube_xarm6_contact_geometry")
        self.assertEqual(spec.case_id, case.case_id)
        self.assertEqual(spec.robot_uid, "xarm6_robotiq")
        self.assertIn("contact_x_offset", spec.parameter_grid)
        self.assertIn("contact_z_offset", spec.parameter_grid)
        self.assertIn("approach_height", spec.parameter_grid)
        self.assertIn("drag_strength", spec.parameter_grid)
        self.assertIn("down_bias", spec.parameter_grid)
        self.assertIn("stages", spec.parameter_grid)
        grid = probe_grid(spec)
        self.assertEqual(len(grid), 32)
        self.assertEqual(grid[0]["case_index"], 1)
        self.assertIn("cube_goal_improvement", spec.primary_metrics)
        self.assertIn("tcp_contact_xy", spec.primary_metrics)
        self.assertIn("task_success", spec.success_keys)

    def test_structured_probe_summary_builds_prompt_feedback(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        spec = get_probe_spec(case)
        payload = summarize_probe_results(
            spec,
            [
                {
                    "grasp_z_offset": 0.016,
                    "close_steps": 12,
                    "close_command": -0.6,
                    "settle_steps": 8,
                    "is_grasping_after_close": False,
                    "is_grasping_after_lift": False,
                    "cube_disp_xy": 0.00458,
                    "tcp_grasp_xy": 0.00239,
                    "tcp_grasp_z": 0.00152,
                    "cube_lift_delta_z": -0.00026,
                    "score": -5.492,
                },
                {
                    "grasp_z_offset": 0.02,
                    "close_steps": 24,
                    "close_command": -1.0,
                    "settle_steps": 16,
                    "is_grasping_after_close": True,
                    "is_grasping_after_lift": True,
                    "cube_disp_xy": 0.006,
                    "tcp_grasp_xy": 0.003,
                    "tcp_grasp_z": 0.002,
                    "cube_lift_delta_z": 0.04,
                    "score": 210.0,
                },
            ],
            top_k=1,
        )
        self.assertEqual(payload["schema"], "structured_probe_result.v1")
        self.assertEqual(payload["num_cases"], 2)
        self.assertEqual(payload["num_success"], 1)
        self.assertEqual(payload["best_probe_case"]["grasp_z_offset"], 0.02)
        self.assertEqual(len(payload["top_probe_cases"]), 1)
        self.assertIn("Structured probe feedback", payload["prompt_feedback"])
        self.assertIn("successful_probe_cases=1", payload["prompt_feedback"])
        self.assertIn("Use this as measured physical evidence", payload["prompt_feedback"])
        self.assertIn("next_probe_suggestions", payload)
        self.assertIn("next_probe_suggestions", payload["prompt_feedback"])

    def test_structured_probe_suggests_score_guided_local_refinements(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        spec = get_probe_spec(case)
        results = [
            {
                "grasp_z_offset": 0.016,
                "close_steps": 12,
                "close_command": -0.6,
                "settle_steps": 8,
                "is_grasping_after_close": False,
                "is_grasping_after_lift": False,
                "cube_disp_xy": 0.00458,
                "tcp_grasp_xy": 0.00239,
                "tcp_grasp_z": 0.00152,
                "cube_lift_delta_z": -0.00026,
                "score": -5.492,
            },
            {
                "grasp_z_offset": 0.012,
                "close_steps": 12,
                "close_command": -0.6,
                "settle_steps": 8,
                "is_grasping_after_close": False,
                "is_grasping_after_lift": False,
                "cube_disp_xy": 0.00526,
                "tcp_grasp_xy": 0.00266,
                "tcp_grasp_z": 0.0,
                "cube_lift_delta_z": 0.0,
                "score": -5.792,
            },
        ]
        suggestions = suggest_next_probe_cases(spec, results, budget=6)
        self.assertEqual(len(suggestions), 6)
        tried = {
            (item["grasp_z_offset"], item["close_steps"], item["close_command"], item["settle_steps"])
            for item in results
        }
        suggested = {
            (item["grasp_z_offset"], item["close_steps"], item["close_command"], item["settle_steps"])
            for item in suggestions
        }
        self.assertTrue(suggested.isdisjoint(tried))
        self.assertTrue(any(item["grasp_z_offset"] == 0.018 for item in suggestions))
        self.assertTrue(any(item["close_command"] == -0.4 for item in suggestions))
        self.assertIn("source_anchor_rank", suggestions[0])
        self.assertIn("suggestion_reason", suggestions[0])

    def test_structured_probe_runner_has_dry_run_entrypoint(self):
        script = Path("scripts/structured_probe_runner.py").read_text(encoding="utf-8")
        self.assertIn("--case", script)
        self.assertIn("--dry-run", script)
        self.assertIn("--adaptive-from", script)
        self.assertIn("--suggest-only", script)
        self.assertIn("structured_probe", script)
        self.assertIn("pick_cube_xarm6_close_envelope", script)
        self.assertIn("pull_cube_xarm6_contact_geometry", script)
        self.assertIn("xarm6_pull_contact_probe", script)

    def test_xarm6_pick_probe_accepts_explicit_probe_plan(self):
        script = Path("scripts/xarm6_pick_grasp_probe.py").read_text(encoding="utf-8")
        self.assertIn("probe_plan_json", script)
        self.assertIn("probe_plan_mode", script)
        self.assertIn("load_probe_plan", script)

    def test_module_generation_pick_retry_changes_grasp_strategy(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        failure = {
            "success": False,
            "failure_layer": "skill_adapter",
            "message": "cube was not grasped",
            "failure_diagnosis": {
                "layer": "contact_geometry",
                "reason": "good_alignment_no_displacement_no_grasp",
                "repair_hint": "change close timing",
                "evidence": {"tcp_grasp_xy": 0.002, "tcp_grasp_z": 0.001, "cube_disp_xy": 0.004},
            },
        }
        prompt = build_module_generation_prompt(
            case=case,
            target_result=failure,
            attempts=[
                {
                    "round": 1,
                    "module_valid": True,
                    "module_kept": True,
                    "verification_ok": True,
                    "target_result": failure,
                }
            ],
        )
        self.assertIn("Do not return a module identical", prompt)
        self.assertIn("Required substantive strategy change", prompt)
        self.assertIn("cube-displacement guard", prompt)
        self.assertIn("one or two bounded candidates", prompt)
        self.assertIn("failure_diagnosis", prompt)
        self.assertIn("Diagnosis-guided repair instruction", prompt)

    def test_pick_cube_runtime_requires_close_time_diagnostics(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        failure = {
            "success": False,
            "message": "all grasp candidates failed; is_grasping=False, tcp_cube_xyz=0.0911",
            "execution_log": [{"api": "grasp"}],
        }
        error = pick_cube_runtime_diagnostic_error(case, failure)
        self.assertIsNotNone(error)
        self.assertIn("tcp_grasp_xy", error or "")
        self.assertIn("tcp_grasp_z", error or "")
        self.assertIn("cube_disp_xy", error or "")

        diagnostic_failure = {
            "success": False,
            "message": "grasp failed; tcp_grasp_xy=0.0040, tcp_grasp_z=0.0180, cube_disp_xy=0.0020, is_grasping=False",
            "execution_log": [{"api": "grasp"}],
        }
        self.assertIsNone(pick_cube_runtime_diagnostic_error(case, diagnostic_failure))

        diagnostic_alias_failure = {
            "success": False,
            "message": "cube displaced by 0.0359m during close; tcp_grasp_xy=0.0015, tcp_grasp_z=0.0020",
            "execution_log": [{"api": "grasp"}],
        }
        self.assertIsNone(pick_cube_runtime_diagnostic_error(case, diagnostic_alias_failure))

        grasping_failure = {
            "success": False,
            "message": (
                "all grasp candidates failed; is_grasping=True, tcp_grasp_xy=0.0010, "
                "tcp_grasp_z=0.0010, cube_disp_xy=0.0020"
            ),
            "execution_log": [{"api": "grasp"}],
        }
        grasping_error = pick_cube_runtime_diagnostic_error(case, grasping_failure)
        self.assertIsNotNone(grasping_error)
        self.assertIn("must not report grasp failure while is_grasping=True", grasping_error or "")

        unknown_diagnostic_failure = {
            "success": False,
            "message": "grasp failed; tcp_grasp_xy=0.0096, tcp_grasp_z=0.1328, cube_disp_xy=unknown",
            "execution_log": [{"api": "grasp"}],
        }
        unknown_error = pick_cube_runtime_diagnostic_error(case, unknown_diagnostic_failure)
        self.assertIsNotNone(unknown_error)
        self.assertIn("cube_disp_xy", unknown_error or "")

    def test_migration_prompt_exposes_pull_api(self):
        request = MigrationRequest.from_ids(
            task_id="pull_cube",
            target_robot="fetch",
            method="llm_card_report",
        )
        prompt = build_migration_prompt(request)
        self.assertIn("robot.pull(obj, target)", prompt)
        self.assertIn("robot.pull(cube, goal", prompt)
        self.assertIn("Do not invent grasp/place/tool APIs", prompt)
        self.assertIn("infeasible:", prompt)

    def test_iterative_prompt_exposes_pull_parameters(self):
        task = get_task_spec("pull_cube")
        prompt = build_iterative_prompt(
            task=task,
            source_robot="panda",
            target_robot="fetch",
            previous_attempts=[
                {
                    "attempt": 1,
                    "code": task.source_program,
                    "result": {
                        "success": False,
                        "failure_type": "contact execution failure",
                        "message": "cube was not pulled to target",
                        "execution_log": [
                            {
                                "step": 1,
                                "api": "pull",
                                "args": {"contact_x_offset": 0.07},
                                "result": False,
                                "ok": False,
                            }
                        ],
                        "final_info": {"success": False},
                    },
                }
            ],
        )
        self.assertIn("contact_x_offset", prompt)
        self.assertIn("contact_z_offset", prompt)
        self.assertIn("Previous target attempts", prompt)
        self.assertIn("infeasible:", prompt)

    def test_default_control_mode_and_adapter(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

        self.assertEqual(_default_control_mode("pull_cube", "fetch"), "pd_ee_delta_pos")
        robot = _build_robot_adapter("pull_cube", Env(), "pd_ee_delta_pos", "fetch")
        self.assertIsInstance(robot, ManiSkillPullCubeRobot)
        self.assertEqual(robot.robot_uid, "fetch")
        xarm_robot = _build_robot_adapter("pull_cube", Env(), "pd_ee_delta_pos", "xarm6_robotiq")
        self.assertIsInstance(xarm_robot, ManiSkillPullCubeRobot)
        self.assertEqual(xarm_robot.robot_uid, "xarm6_robotiq")
        pick_robot = _build_robot_adapter("pick_cube", Env(), "pd_ee_delta_pos", "panda")
        self.assertIsInstance(pick_robot, ManiSkillPickCubeRobot)
        self.assertEqual(pick_robot.robot_uid, "panda")

    def test_generated_target_adapter_module_loads(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

        robot = _build_robot_adapter_from_module(
            "maniskill_backend.generated_adapters.case01_fetch_pull_cube",
            Env(),
            "pd_ee_delta_pos",
            "fetch",
        )
        self.assertIsInstance(robot, ManiSkillPullCubeRobot)
        self.assertEqual(robot.robot_uid, "fetch")
        xarm_robot = _build_robot_adapter_from_module(
            "maniskill_backend.generated_adapters.case02_xarm6_pull_cube",
            Env(),
            "pd_ee_delta_pos",
            "xarm6_robotiq",
        )
        self.assertIsInstance(xarm_robot, ManiSkillPullCubeRobot)
        self.assertEqual(xarm_robot.robot_uid, "xarm6_robotiq")
        pick_robot = _build_robot_adapter_from_module(
            "maniskill_backend.generated_adapters.case03_xarm6_pick_cube",
            Env(),
            "pd_ee_delta_pos",
            "xarm6_robotiq",
        )
        self.assertIsInstance(pick_robot, ManiSkillPickCubeRobot)
        self.assertEqual(pick_robot.robot_uid, "xarm6_robotiq")

    def test_pull_cube_robot_action_shape(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

        robot = ManiSkillPullCubeRobot(Env(), robot_uid="panda", control_mode="pd_ee_delta_pos")
        action = robot._make_action(np.array([2.0, -2.0, 0.25]), gripper=-1.0)
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.allclose(action, np.array([1.0, -1.0, 0.25, -1.0])))

    def test_lmp_executor_still_supports_pull_program(self):
        class Robot:
            def pull(self, obj, target):
                return obj.name == "cube" and target.name == "goal"

        ok, message, locals_dict = execute_lmp(
            get_task_spec("pull_cube").source_program,
            {"scene": ManiSkillSceneAdapter(), "robot": Robot()},
        )
        self.assertTrue(ok, message)
        self.assertTrue(locals_dict["ret_val"])

    def test_lmp_executor_supports_pick_program(self):
        class Robot:
            def grasp(self, obj):
                return obj.name == "cube"

            def place(self, obj, target):
                return obj.name == "cube" and target.name == "goal"

        ok, message, locals_dict = execute_lmp(
            get_task_spec("pick_cube").source_program,
            {"scene": ManiSkillSceneAdapter(), "robot": Robot()},
        )
        self.assertTrue(ok, message)
        self.assertTrue(locals_dict["ret_val"])

    def test_oracle_code_is_source_program(self):
        task = get_task_spec("pull_cube")
        self.assertEqual(build_oracle_code(task), task.source_program.strip())

    def test_success_from_ret_val(self):
        self.assertTrue(success_from_ret_val(True))
        self.assertTrue(success_from_ret_val("ok"))
        self.assertFalse(success_from_ret_val(None))
        self.assertFalse(success_from_ret_val("failure: pull"))
        self.assertFalse(success_from_ret_val("infeasible: contact pose unreachable"))

    def test_failure_classifier(self):
        self.assertEqual(
            classify_failure(success=False, message="cube was not pulled to target"),
            "contact execution failure",
        )
        self.assertEqual(
            classify_failure(success=False, message="infeasible: contact pose outside workspace"),
            "impossible-task refusal failure",
        )
        self.assertEqual(
            classify_failure(success=False, message="cube slipped during lift"),
            "gripper/force failure",
        )
        self.assertEqual(
            classify_failure(success=False, message="cube was not moved to goal"),
            "execution failure",
        )

    def test_failure_layer_classifier(self):
        self.assertEqual(classify_failure_layer(success=False, code_ok=False, message="NameError"), "program")
        self.assertEqual(
            classify_failure_layer(
                success=False,
                info={
                    "execution_log": [
                        {"api": "pull", "ok": False, "message": "planner failed during contact move"}
                    ],
                    "final_info": {"planner_status": "failed"},
                },
            ),
            "controller_primitive",
        )

    def test_structured_pull_failure_diagnosis_contact_reachability(self):
        diagnosis = diagnose_failure(
            task_id="pull_cube",
            success=False,
            message="Episode ended during descent.",
            failure_type="contact execution failure",
            failure_layer="skill_adapter",
            execution_log=[{"api": "pull", "ok": False, "message": "Episode ended during descent."}],
            runtime_diagnostics={
                "stage": "descent",
                "tcp_stage_error_norm": 0.14752,
                "tcp_stage_error_xyz": [0.14363, 0.03361, -0.00086],
                "tcp_cube_xy": 0.04109,
                "cube_goal_xy": 0.27472,
            },
        )
        self.assertEqual(diagnosis["layer"], "contact_geometry")
        self.assertEqual(diagnosis["reason"], "contact_side_reachability_failure")
        self.assertIn("reachability precheck", diagnosis["repair_hint"])

    def test_structured_pick_failure_diagnosis_close_patterns(self):
        side_push = diagnose_failure(
            task_id="pick_cube",
            success=False,
            message=(
                "cube displaced laterally during close; tcp_grasp_xy=0.0037, "
                "tcp_grasp_z=0.0040, cube_disp_xy=0.0406, is_grasping=False"
            ),
            failure_type="gripper/force failure",
            failure_layer="skill_adapter",
            execution_log=[{"api": "grasp", "ok": False}],
        )
        self.assertEqual(side_push["layer"], "contact_geometry")
        self.assertEqual(side_push["reason"], "gripper_envelope_side_push")

        no_grasp = diagnose_failure(
            task_id="pick_cube",
            success=False,
            message=(
                "grasp failed after close; tcp_grasp_xy=0.0027, tcp_grasp_z=0.0002, "
                "cube_disp_xy=0.0052, is_grasping=False"
            ),
            failure_type="gripper/force failure",
            failure_layer="skill_adapter",
            execution_log=[{"api": "grasp", "ok": False}],
        )
        self.assertEqual(no_grasp["layer"], "contact_geometry")
        self.assertEqual(no_grasp["reason"], "good_alignment_no_displacement_no_grasp")
        self.assertIn("close timing", no_grasp["repair_hint"])

    def test_generalization_strategy_selects_reachability_repair(self):
        summary = {
            "task_id": "pull_cube",
            "robot_uid": "xarm6_robotiq",
            "adapter_module": "maniskill_backend.generated_adapters.case02_xarm6_pull_cube",
            "adapter_sha256": "abc",
            "rows": [
                {
                    "seed": 0,
                    "success": True,
                    "failure_type": "success",
                    "failure_layer": "success",
                    "final_info": {"elapsed_steps": [460]},
                },
                {
                    "seed": 1,
                    "success": False,
                    "failure_type": "unknown failure",
                    "failure_layer": "skill_adapter",
                    "final_info": {"elapsed_steps": [500]},
                    "runtime_diagnostics": {"stage": "descent"},
                    "failure_diagnosis": {
                        "layer": "contact_geometry",
                        "reason": "contact_side_reachability_failure",
                        "confidence": 0.8,
                    },
                },
                {
                    "seed": 2,
                    "success": False,
                    "failure_type": "unknown failure",
                    "failure_layer": "skill_adapter",
                    "final_info": {"elapsed_steps": [500]},
                    "runtime_diagnostics": {"stage": "approach"},
                    "failure_diagnosis": {
                        "layer": "contact_geometry",
                        "reason": "contact_side_reachability_failure",
                        "confidence": 0.8,
                    },
                },
            ],
        }
        report = build_generalization_report(summary, success_threshold=0.8, min_trials_for_accept=3)
        self.assertEqual(report["schema"], "generalization_strategy.v1")
        self.assertEqual(report["status"], "needs_repair")
        self.assertEqual(report["selected_strategy"], "reachability_aware_contact_selection")
        self.assertEqual(report["failure_reason_counts"]["contact_side_reachability_failure"], 2)
        self.assertIn(1, report["failure_seed_clusters"]["contact_side_reachability_failure:descent"])
        md = generalization_report_to_markdown(report)
        self.assertIn("Generalization Strategy Selection", md)
        self.assertIn("reachability_aware_contact_selection", md)

    def test_generalization_strategy_accepts_high_success_rate(self):
        summary = {
            "task_id": "pull_cube",
            "robot_uid": "xarm6_robotiq",
            "adapter_module": "adapter",
            "adapter_sha256": "abc",
            "rows": [
                {"seed": 0, "success": True, "failure_type": "success", "failure_layer": "success"},
                {"seed": 1, "success": True, "failure_type": "success", "failure_layer": "success"},
                {"seed": 2, "success": True, "failure_type": "success", "failure_layer": "success"},
                {"seed": 3, "success": True, "failure_type": "success", "failure_layer": "success"},
                {
                    "seed": 4,
                    "success": False,
                    "failure_type": "contact execution failure",
                    "failure_layer": "skill_adapter",
                },
            ],
        }
        report = build_generalization_report(summary, success_threshold=0.8, min_trials_for_accept=5)
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["selected_strategy"], "accept_current_adapter")
        self.assertEqual(report["success_rate"], 0.8)

    def test_pullcube_multiseed_summary_includes_generalization_strategy(self):
        from scripts.pullcube_multiseed_eval import build_summary, summary_to_markdown

        metadata = {
            "task_id": "pull_cube",
            "robot_uid": "xarm6_robotiq",
            "adapter_module": "adapter",
            "adapter_sha256": "abc",
            "code_file": "case.py",
            "git_rev": "abc123",
            "control_mode": "pd_ee_delta_pos",
            "max_episode_steps": 500,
            "method": "target-module-generation",
            "adapter_path": "adapter.py",
            "obs_mode": "state",
            "sim_backend": "auto",
            "render_backend": "gpu",
            "seeds": [0, 1],
        }
        results = [
            {"seed": 0, "success": True, "failure_type": "success", "failure_layer": "success"},
            {
                "seed": 1,
                "success": False,
                "failure_type": "unknown failure",
                "failure_layer": "skill_adapter",
                "message": "Episode ended during descent.",
                "runtime_diagnostics": {"stage": "descent"},
                "failure_diagnosis": {"reason": "contact_side_reachability_failure"},
            },
        ]
        summary = build_summary(metadata, results, success_threshold=0.8, min_trials_for_accept=2)
        self.assertIn("generalization_strategy", summary)
        self.assertEqual(
            summary["generalization_strategy"]["selected_strategy"],
            "reachability_aware_contact_selection",
        )
        md = summary_to_markdown(summary)
        self.assertIn("Generalization Strategy Selection", md)
        self.assertIn("Failure Seed Clusters", md)

    def test_offline_generalization_selector_script_documents_input(self):
        script = Path("scripts/select_generalization_strategy.py").read_text(encoding="utf-8")
        self.assertIn("--input", script)
        self.assertIn("pullcube_multiseed_eval.py", script)
        self.assertIn("generalization_strategy", script)

    def test_autonomous_harness_exposes_bounded_tool_inventory(self):
        case = get_full_migration_case("case02_pull_cube_panda_to_xarm6")
        tools = tool_inventory(case, seed=5)
        names = [tool.name for tool in tools]
        self.assertIn("run_single_seed", names)
        self.assertIn("run_multi_seed", names)
        self.assertIn("run_structured_probe", names)
        self.assertIn("run_llm_repair", names)
        probe_tool = next(tool for tool in tools if tool.name == "run_structured_probe")
        self.assertIn("scripts/structured_probe_runner.py", probe_tool.command_template)
        self.assertIn("--seed 5", probe_tool.command_template)
        self.assertIn("failure_diagnosis", probe_tool.inputs)
        self.assertIn("probe_json", probe_tool.outputs)

    def test_autonomous_harness_loads_multiseed_and_selects_probe_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "pullcube.jsonl"
            rows = [
                {
                    "type": "metadata",
                    "task_id": "pull_cube",
                    "robot_uid": "xarm6_robotiq",
                    "adapter_module": "adapter",
                    "adapter_sha256": "abc",
                },
                {
                    "type": "trial",
                    "seed": 0,
                    "success": True,
                    "failure_type": "success",
                    "failure_layer": "success",
                    "final_info": {"elapsed_steps": [460]},
                },
                {
                    "type": "trial",
                    "seed": 5,
                    "success": False,
                    "failure_type": "unknown failure",
                    "failure_layer": "contact_geometry",
                    "message": "Episode ended during descent.",
                    "runtime_diagnostics": {
                        "stage": "descent",
                        "tcp_cube_xy": 0.02912,
                        "tcp_stage_error_norm": 0.09094,
                    },
                    "failure_diagnosis": {
                        "layer": "contact_geometry",
                        "reason": "contact_side_reachability_failure",
                        "confidence": 0.9,
                        "repair_hint": "Choose contact pose adaptively.",
                    },
                    "final_info": {"elapsed_steps": [500]},
                },
                {
                    "type": "trial",
                    "seed": 8,
                    "success": False,
                    "failure_type": "unknown failure",
                    "failure_layer": "contact_geometry",
                    "message": "Episode ended during approach.",
                    "runtime_diagnostics": {
                        "stage": "approach",
                        "tcp_cube_xy": 0.64941,
                        "tcp_stage_error_norm": 0.7718,
                    },
                    "failure_diagnosis": {
                        "layer": "contact_geometry",
                        "reason": "contact_side_reachability_failure",
                        "confidence": 0.9,
                    },
                    "final_info": {"elapsed_steps": [500]},
                },
            ]
            jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            summary = load_multiseed_jsonl(jsonl_path)
            self.assertEqual(summary["num_trials"], 3)
            self.assertEqual(summary["num_success"], 1)
            self.assertEqual(summary["generalization_strategy"]["selected_strategy"], "reachability_aware_contact_selection")
            near = select_failure_seed(summary, policy="near_contact")
            severe = select_failure_seed(summary, policy="severe_reachability")
            self.assertEqual(near["seed"], 5)
            self.assertEqual(severe["seed"], 8)

    def test_autonomous_harness_splits_agent_observation_from_human_suggestion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "pullcube.jsonl"
            rows = [
                {
                    "type": "metadata",
                    "task_id": "pull_cube",
                    "robot_uid": "xarm6_robotiq",
                    "adapter_module": "maniskill_backend.generated_adapters.case02_xarm6_pull_cube",
                    "adapter_sha256": "abc",
                },
                {"type": "trial", "seed": 0, "success": True, "failure_type": "success"},
                {
                    "type": "trial",
                    "seed": 5,
                    "success": False,
                    "failure_type": "unknown failure",
                    "failure_layer": "contact_geometry",
                    "runtime_diagnostics": {
                        "stage": "descent",
                        "tcp_cube_xy": 0.02912,
                        "tcp_stage_error_norm": 0.09094,
                    },
                    "failure_diagnosis": {
                        "layer": "contact_geometry",
                        "reason": "contact_side_reachability_failure",
                        "repair_hint": "Choose contact pose adaptively.",
                    },
                },
            ]
            jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            plan = build_harness_plan(
                case_id="case02_pull_cube_panda_to_xarm6",
                multiseed_jsonl=jsonl_path,
                include_existing_probe=False,
            )
            observation = plan["agent_observation"]
            human_report = plan["human_report"]
            action = human_report["suggested_next_action"]
            self.assertEqual(action["tool"], "run_structured_probe")
            self.assertIn("scripts/structured_probe_runner.py", action["command"])
            self.assertIn("--seed 5", action["command"])
            self.assertIn("contact_side_reachability_failure", action["command"])
            self.assertNotIn("recommended_next_action", observation)
            self.assertNotIn("agent_prompt", observation)
            self.assertEqual(observation["schema"], "agent_observation.v1")
            self.assertIn("low_level_interface", observation)
            self.assertIn("env.step(action)", observation["low_level_interface"]["execution_boundary"])
            self.assertEqual(observation["constraints"]["frozen"], ["controller", "simulator", "success_signal", "high_level_program"])
            failures = observation["latest_results"]["multiseed"]["failure_rows"]
            self.assertEqual(failures[0]["seed"], 5)
            diagnosis = failures[0]["failure_diagnosis"]
            self.assertIn("reason", diagnosis)
            self.assertNotIn("repair_hint", diagnosis)

    def test_agent_planner_uses_observation_tools_without_human_report(self):
        observation = {
            "schema": "agent_observation.v1",
            "latest_results": {"multiseed": {}, "structured_probe": {}},
            "allowed_tools": [
                {"name": "run_multi_seed"},
                {"name": "run_structured_probe"},
                {"name": "run_llm_repair"},
            ],
        }
        fallback = fallback_agent_actions(observation)
        self.assertEqual(fallback["actions"][0]["tool"], "run_multi_seed")
        plan = validate_agent_plan(
            {
                "actions": [
                    {"tool": "run_multi_seed", "args": {}},
                    {"tool": "unsafe_shell", "args": {"cmd": "rm -rf /"}},
                ]
            },
            observation,
            max_actions=2,
        )
        self.assertEqual([item["tool"] for item in plan["actions"]], ["run_multi_seed"])
        dry_plan = plan_agent_actions(observation, dry_run=True)
        self.assertEqual(dry_plan["actions"][0]["tool"], "run_multi_seed")
        self.assertFalse(dry_plan["used_llm"])

    def test_short_auto_pull_entrypoint_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "auto.py",
                    "pull",
                    "--dry-run",
                    "--run-name",
                    "unit",
                    "--output-root",
                    tmpdir,
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("autonomous_loop_result.v1", completed.stdout)
            summary = Path(tmpdir) / "unit" / "summary.json"
            self.assertTrue(summary.exists())
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["cycles"][0]["status"], "dry_run_planned")

    def test_pull_cube_failure_report(self):
        task = get_task_spec("pull_cube")
        profile = get_robot_profile("fetch")
        record = TrialRecord(
            task_id="pull_cube",
            source_robot="panda",
            target_robot="fetch",
            method="source-copy",
            seed=0,
            generated_code=task.source_program,
            success=False,
            failure_type="contact execution failure",
            failure_layer="skill_adapter",
            message="cube was not pulled to target",
            info={
                "execution_log": [
                    {
                        "step": 1,
                        "api": "pull",
                        "args": {"obj": "cube", "target": "goal"},
                        "ok": False,
                        "message": "cube was not pulled to target",
                    }
                ]
            },
        )
        report = build_real_failure_report(task=task, target_profile=profile, failed_record=record)
        text = report.to_prompt_section()
        self.assertIn("robot.pull(cube, goal)", text)
        self.assertIn("Do not add robot.grasp(cube)", text)

    def test_pick_cube_failure_report(self):
        task = get_task_spec("pick_cube")
        profile = get_robot_profile("xarm6_robotiq")
        record = TrialRecord(
            task_id="pick_cube",
            source_robot="panda",
            target_robot="xarm6_robotiq",
            method="source-copy",
            seed=0,
            generated_code=task.source_program,
            success=False,
            failure_type="gripper/force failure",
            failure_layer="skill_adapter",
            message="cube was not grasped",
            info={
                "execution_log": [
                    {
                        "step": 1,
                        "api": "grasp",
                        "args": {"obj": "cube"},
                        "ok": False,
                        "message": "cube was not grasped",
                    }
                ]
            },
        )
        report = build_real_failure_report(task=task, target_profile=profile, failed_record=record)
        text = report.to_prompt_section()
        self.assertIn("robot.grasp(cube)", text)
        self.assertIn("real gripper grasp", text)
        self.assertIn("Do not replace grasping with pushing", text)

    def test_results_summary_and_markdown(self):
        records = [
            TrialRecord(
                task_id="pull_cube",
                source_robot="panda",
                target_robot="fetch",
                method="source-copy",
                seed=0,
                generated_code="ret_val = robot.pull(cube, goal)",
                success=True,
                failure_type="success",
            ),
            TrialRecord(
                task_id="pull_cube",
                source_robot="panda",
                target_robot="fetch",
                method="llm_card_report",
                seed=1,
                generated_code="ret_val = 'failure: pull'",
                success=False,
                failure_type="contact execution failure",
                failure_layer="skill_adapter",
            ),
        ]
        summary = summarize_records(records)
        source_copy = next(row for row in summary if row["method"] == "source-copy")
        self.assertEqual(source_copy["success_rate"], 1.0)
        md = records_to_md([record.to_dict() for record in records])
        self.assertIn("pull_cube", md)
        self.assertIn("```python", md)

    def test_append_jsonl(self):
        record = TrialRecord(
            task_id="pull_cube",
            source_robot="panda",
            target_robot="fetch",
            method="source-copy",
            seed=0,
            generated_code="ret_val = True",
            success=True,
            failure_type="success",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            append_jsonl(path, [record])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["task_id"], "pull_cube")

    def test_code_diff_marks_parameter_changes(self):
        diff = _code_diff(
            "ret_val = robot.pull(cube, goal, contact_x_offset=0.07)",
            "ret_val = robot.pull(cube, goal, contact_x_offset=0.10, stages=6)",
        )
        self.assertIn("-ret_val", diff)
        self.assertIn("+ret_val", diff)
        self.assertIn("contact_x_offset=0.10", diff)


if __name__ == "__main__":
    unittest.main()
