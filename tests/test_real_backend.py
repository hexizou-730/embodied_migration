import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lmp.executor import execute_lmp
from maniskill_backend.cases import (
    PRIMARY_FULL_MIGRATION_CASE,
    PRIMARY_FULL_MIGRATION_CASE_ID,
    get_full_migration_case,
)
from maniskill_backend.evaluation import TrialRecord, classify_failure, classify_failure_layer
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
        self.assertIn("REAL GRASPING", prompt)
        self.assertIn("robot.grasp(cube) followed by robot.place(cube, goal)", prompt)
        self.assertIn("self._is_grasping('cube')", prompt)
        self.assertIn("Do not chase a displaced cube", prompt)
        self.assertIn("cube_pos.z=-0.8996", prompt)
        self.assertIn("preserve the grasp and return success", prompt)
        self.assertIn("tcp_cube_xyz=0.0102", prompt)
        self.assertIn("pushed the cube laterally by 0.1513m", prompt)
        self.assertIn("slow near-vertical descent", prompt)
        self.assertIn("Close the gripper only when the measured xy and z residuals", prompt)
        self.assertIn("final tcp_cube_xyz after retreat is not enough", prompt)
        self.assertIn("does NOT prove the close-time TCP was 0.1573m away", prompt)
        self.assertIn("reduced cube displacement after close to 0.0015m", prompt)
        self.assertIn("cube_half_size=0.02m", prompt)
        self.assertIn("Z-focused offset set", prompt)
        self.assertIn("tcp_grasp_xy and tcp_grasp_z", prompt)
        self.assertIn("tcp_cube_xyz=0.0911", prompt)
        self.assertIn("cube_disp_xy", prompt)
        self.assertIn("tcp_grasp_xy=0.0076", prompt)
        self.assertIn("cube_pos=[-0.0509, -0.3862, 0.02]", prompt)
        self.assertIn("gripper-envelope side push", prompt)
        self.assertIn("tcp_grasp_xy=0.0027", prompt)
        self.assertIn("cube_disp_xy=0.0052", prompt)
        self.assertIn("good-alignment/no-displacement/no-grasp", prompt)
        self.assertIn("close-envelope/force failure", prompt)
        self.assertIn("cube_disp_xy=0.0406", prompt)
        self.assertIn("Do not use grasp_z_offset=0.0 as the first", prompt)
        self.assertIn("side-push regression", prompt)
        self.assertIn("nonzero first Z offset", prompt)
        self.assertIn("displaced the cube by 0.0359m", prompt)
        self.assertIn("z_offset=-0.005", prompt)
        self.assertIn("Always include the exact key `cube_disp_xy=...`", prompt)
        self.assertIn("positive Z close-height sweep", prompt)
        self.assertNotIn("farther positive-x sweep start", prompt)

    def test_module_generation_pick_retry_changes_grasp_strategy(self):
        case = get_full_migration_case("case03_pick_cube_panda_to_xarm6")
        failure = {
            "success": False,
            "failure_layer": "skill_adapter",
            "message": "cube was not grasped",
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
        self.assertIn("bounded grasp-offset search", prompt)
        self.assertIn("cube-displacement guard", prompt)
        self.assertIn("Preserve enough episode budget for transport", prompt)
        self.assertIn("finish xy alignment while safely above the cube", prompt)
        self.assertIn("Add phase-specific diagnostics before close and after close", prompt)

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
