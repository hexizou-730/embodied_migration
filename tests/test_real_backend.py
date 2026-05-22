import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lmp.executor import execute_lmp
from maniskill_backend.cases import (
    PRIMARY_FULL_MIGRATION_CASE,
    PRIMARY_FULL_MIGRATION_CASE_ID,
    get_full_migration_case,
)
from maniskill_backend.env_adapter import can_import_maniskill
from maniskill_backend.evaluation import TrialRecord, classify_failure, classify_failure_layer
from maniskill_backend.full_stack_runner import (
    extract_unified_diff,
    patch_paths,
    validate_patch,
)
from maniskill_backend.iterative_runner import _code_diff, build_iterative_prompt
from maniskill_backend.migration import METHODS, MigrationRequest, build_migration_prompt
from maniskill_backend.profiles import get_robot_profile
from maniskill_backend.real_runner import _build_robot_adapter, _default_control_mode
from maniskill_backend.reporting import (
    build_oracle_code,
    build_real_failure_report,
    success_from_ret_val,
)
from maniskill_backend.results import append_jsonl, summarize_records
from maniskill_backend.sim_check import summarize_value
from maniskill_backend.skill_adapter import (
    ManiSkillPandaPegInsertionPlannerRobot,
    ManiSkillPickCubeRobot,
    ManiSkillPullCubeToolPlannerRobot,
    ManiSkillSceneAdapter,
    ManiSkillStackCubePlannerRobot,
    ManiSkillXArmPickCubePlannerRobot,
)
from maniskill_backend.tasks import get_task_spec
from maniskill_backend.view import records_to_md


class RealBackendTest(unittest.TestCase):
    def test_profiles_and_real_tasks_are_promptable(self):
        profile = get_robot_profile("xarm6_robotiq")
        task = get_task_spec("pick_cube")
        self.assertIn("Official Specs", profile.to_prompt_section())
        self.assertIn("Derived Task Priors", profile.to_prompt_section())
        self.assertIn("payload_kg", profile.to_prompt_section())
        self.assertIn("recommended_alignment_tolerance_m", profile.to_prompt_section())
        self.assertNotIn("\n  alignment_tolerance_m:", profile.to_prompt_section())
        self.assertIn("# Task: pick_cube", task.to_prompt_section())
        self.assertIn("中文任务: 抓取方块", task.to_prompt_section())
        self.assertIn("ManiSkill env: PickCube-v1", task.to_prompt_section())

    def test_method_set_is_real_only(self):
        self.assertEqual(METHODS, ("source-copy", "llm_card_report", "oracle"))

    def test_static_task_specs_are_removed(self):
        with self.assertRaises(KeyError):
            get_task_spec("PlugCharger-v1")

    def test_case01_is_pull_cube_tool_full_migration(self):
        case = get_full_migration_case(PRIMARY_FULL_MIGRATION_CASE_ID)
        self.assertIs(case, PRIMARY_FULL_MIGRATION_CASE)
        self.assertEqual(case.case_number, 1)
        self.assertEqual(case.task_id, "pull_cube_tool")
        self.assertEqual(case.source_robot, "panda")
        self.assertEqual(case.target_robot, "xarm6_robotiq")
        self.assertEqual(case.target_control_mode, "pd_joint_pos")
        self.assertEqual(
            case.target_program_path,
            "maniskill_backend/case_programs/case01_pull_cube_tool.py",
        )
        self.assertIn("skill_adapter", case.migration_layers)
        self.assertIn("controller_primitive", case.migration_layers)

    def test_full_stack_patch_guard_extracts_allowed_diff(self):
        text = """```diff
diff --git a/maniskill_backend/skill_adapter.py b/maniskill_backend/skill_adapter.py
--- a/maniskill_backend/skill_adapter.py
+++ b/maniskill_backend/skill_adapter.py
@@
-old
+new
```"""
        patch = extract_unified_diff(text)
        self.assertIn("diff --git", patch)
        self.assertEqual(
            patch_paths(patch),
            ("maniskill_backend/skill_adapter.py",),
        )
        self.assertEqual(validate_patch(patch), patch_paths(patch))

    def test_full_stack_patch_guard_rejects_out_of_scope_diff(self):
        patch = """diff --git a/tests/test_real_backend.py b/tests/test_real_backend.py
--- a/tests/test_real_backend.py
+++ b/tests/test_real_backend.py
@@
-old
+new
"""
        with self.assertRaises(ValueError):
            validate_patch(patch)

    def test_migration_prompt_card_report_method(self):
        request = MigrationRequest.from_ids(
            task_id="pick_cube",
            target_robot="xarm6_robotiq",
            method="llm_card_report",
        )
        prompt = build_migration_prompt(request)
        self.assertIn("Capability Card", prompt)
        self.assertIn("xarm6_robotiq", prompt)
        self.assertIn("Failure Report", prompt)
        self.assertIn("ret_val", prompt)
        self.assertIn("robot.place", prompt)

    def test_source_copy_prompt_does_not_include_target_card(self):
        request = MigrationRequest.from_ids(
            task_id="pick_cube",
            target_robot="xarm6_robotiq",
            method="source-copy",
        )
        prompt = build_migration_prompt(request)
        self.assertIn("No target Capability Card", prompt)
        self.assertNotIn("# Failure Report", prompt)

    def test_pick_cube_task_is_available(self):
        task = get_task_spec("pick_cube")
        self.assertEqual(task.task_id, "pick_cube")
        self.assertEqual(task.name_cn, "抓取方块")
        self.assertEqual(task.source_robot, "panda")
        self.assertEqual(task.maniskill_env_id, "PickCube-v1")
        self.assertIn("robot.place", task.source_program)

    def test_official_env_name_still_works_as_alias(self):
        task = get_task_spec("PickCube-v1")
        self.assertEqual(task.task_id, "pick_cube")

    def test_oracle_code_is_real_source_program(self):
        task = get_task_spec("peg_insertion")
        self.assertEqual(task.source_robot, "panda_wristcam")
        self.assertIn("xarm6_robotiq", task.target_robots)
        oracle = build_oracle_code(task)
        self.assertIn("robot.align_to_target", oracle)
        self.assertNotIn("robot.recommended_alignment_tolerance", oracle)

    def test_stack_cube_task_is_available(self):
        task = get_task_spec("stack_cube")
        self.assertEqual(task.task_id, "stack_cube")
        self.assertEqual(task.maniskill_env_id, "StackCube-v1")
        self.assertEqual(task.source_robot, "panda")
        self.assertIn("cubeA", task.source_program)
        self.assertIn("cubeB", task.source_program)

    def test_pull_cube_tool_task_is_available(self):
        task = get_task_spec("pull_cube_too")
        self.assertEqual(task.task_id, "pull_cube_tool")
        self.assertEqual(task.maniskill_env_id, "PullCubeTool-v1")
        self.assertEqual(task.name_cn, "用工具拉方块")
        self.assertIn("robot.hook_object", task.source_program)
        self.assertIn("robot.pull_with_tool", task.source_program)

    def test_success_from_ret_val(self):
        self.assertTrue(success_from_ret_val(True))
        self.assertTrue(success_from_ret_val("ok"))
        self.assertFalse(success_from_ret_val(None))
        self.assertFalse(success_from_ret_val("failure: grasp"))

    def test_default_xarm_pick_cube_control_uses_planner_path(self):
        self.assertEqual(_default_control_mode("pick_cube", "xarm6_robotiq"), "pd_joint_pos")
        self.assertEqual(_default_control_mode("pick_cube", "panda"), "pd_ee_delta_pos")
        self.assertEqual(_default_control_mode("peg_insertion", "panda_wristcam"), "pd_joint_pos")
        self.assertEqual(_default_control_mode("stack_cube", "xarm6_robotiq"), "pd_joint_pos")
        self.assertEqual(_default_control_mode("pull_cube_tool", "xarm6_robotiq"), "pd_joint_pos")

    def test_pick_cube_skill_adapter_action_shape(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        scene = ManiSkillSceneAdapter()
        robot = ManiSkillPickCubeRobot(Env())
        cube = scene.get_object("cube")
        goal = scene.get_region("goal")
        action = robot._make_action(np.array([2.0, -2.0, 0.5]), gripper=-1.0)
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.allclose(action, np.array([1.0, -1.0, 0.5, -1.0])))
        self.assertFalse(robot.insert(cube, goal, speed=0.1))
        self.assertEqual(robot.execution_log()[-1]["api"], "insert")

    def test_pick_cube_adapter_allows_robot_specific_gripper_signs(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        robot = ManiSkillPickCubeRobot(Env(), gripper_open=-1.0, gripper_close=1.0)
        open_action = robot._make_action(np.zeros(3), gripper=robot.gripper_open)
        close_action = robot._make_action(np.zeros(3), gripper=robot.gripper_close)
        self.assertEqual(open_action[-1], -1.0)
        self.assertEqual(close_action[-1], 1.0)

    def test_real_runner_uses_xarm_delta_path_when_requested(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        robot = _build_robot_adapter("pick_cube", Env(), "pd_ee_delta_pos", "xarm6_robotiq")
        self.assertEqual(robot.gripper_open, -1.0)
        self.assertEqual(robot.gripper_close, 1.0)
        self.assertGreaterEqual(robot.move_steps, 36)

    def test_real_runner_uses_xarm_planner_for_joint_pos(self):
        class Env:
            pass

        robot = _build_robot_adapter("pick_cube", Env(), "pd_joint_pos", "xarm6_robotiq")
        self.assertIsInstance(robot, ManiSkillXArmPickCubePlannerRobot)

    def test_real_runner_uses_panda_peg_planner_for_joint_pos(self):
        class Env:
            pass

        robot = _build_robot_adapter("peg_insertion", Env(), "pd_joint_pos", "panda_wristcam")
        self.assertIsInstance(robot, ManiSkillPandaPegInsertionPlannerRobot)

    def test_real_runner_uses_stack_cube_planner_for_joint_pos(self):
        class Env:
            pass

        robot = _build_robot_adapter("stack_cube", Env(), "pd_joint_pos", "xarm6_robotiq")
        self.assertIsInstance(robot, ManiSkillStackCubePlannerRobot)
        self.assertEqual(robot.robot_uid, "xarm6_robotiq")

    def test_real_runner_uses_pull_cube_tool_planner_for_joint_pos(self):
        class Env:
            pass

        robot = _build_robot_adapter("pull_cube_tool", Env(), "pd_joint_pos", "panda")
        self.assertIsInstance(robot, ManiSkillPullCubeToolPlannerRobot)
        self.assertEqual(robot.robot_uid, "panda")

    def test_pull_cube_tool_xarm_defaults_to_base_pull_frame(self):
        class Env:
            pass

        panda = ManiSkillPullCubeToolPlannerRobot(Env(), robot_uid="panda", control_mode="pd_joint_pos")
        xarm = ManiSkillPullCubeToolPlannerRobot(Env(), robot_uid="xarm6_robotiq", control_mode="pd_joint_pos")
        self.assertEqual(panda._default_pull_frame(None), "tool")
        self.assertEqual(xarm._default_pull_frame(None), "toward_base")
        self.assertEqual(xarm._default_pull_frame("tool-local"), "tool")
        self.assertEqual(xarm._default_pull_frame("towards_base"), "toward_base")

    def test_iterative_prompt_exposes_xarm_tool_grasp_offset(self):
        task = get_task_spec("pull_cube_tool")
        prompt = build_iterative_prompt(
            task=task,
            source_robot="panda",
            target_robot="xarm6_robotiq",
            previous_attempts=[],
        )
        self.assertIn("tool_grasp_x_offset=0.08", prompt)

    def test_pick_cube_place_accepts_success_while_held(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Pose:
            def __init__(self, p):
                self.p = np.array(p, dtype=np.float32)

        class Entity:
            def __init__(self, p):
                self.pose = Pose(p)

        class Agent:
            tcp_pose = Pose([0.0, 0.0, 0.0])

        class Env:
            action_space = Space()
            cube = Entity([0.0, 0.0, 0.0])
            goal_site = Entity([0.0, 0.0, 0.0])
            agent = Agent()

            @property
            def unwrapped(self):
                return self

            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(np.asarray(action).copy())
                return None, 0.0, False, False, {"success": [True], "is_obj_placed": [True]}

        scene = ManiSkillSceneAdapter()
        env = Env()
        robot = ManiSkillPickCubeRobot(env, gripper_open=-1.0, gripper_close=1.0)
        self.assertTrue(robot.place(scene.get_object("cube"), scene.get_region("goal")))
        self.assertEqual(robot.execution_log()[-1]["message"], "cube moved to goal while held")
        self.assertTrue(all(action[-1] == 1.0 for action in env.actions))

    def test_pick_cube_place_compensates_tcp_object_offset(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Pose:
            def __init__(self, p):
                self.p = np.array(p, dtype=np.float32)

        class Entity:
            def __init__(self, p):
                self.pose = Pose(p)

        class Agent:
            tcp_pose = Pose([0.0, 0.0, 0.0])

        class Env:
            action_space = Space()
            cube = Entity([0.0, 0.0, 0.0])
            goal_site = Entity([0.2, 0.0, 0.05])
            agent = Agent()

            @property
            def unwrapped(self):
                return self

            def step(self, action):
                return None, 0.0, False, False, {}

        robot = ManiSkillPickCubeRobot(Env())
        robot.tcp_to_obj_at_grasp = np.array([0.0, 0.0, 0.04], dtype=np.float32)
        self.assertTrue(np.allclose(robot._held_tcp_offset(), [0.0, 0.0, 0.04]))

    def test_pick_cube_adapter_rejects_bad_control_mode(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        with self.assertRaises(ValueError):
            ManiSkillPickCubeRobot(Env(), control_mode="pd_joint_delta_pos")

    def test_failure_classifier(self):
        self.assertEqual(
            classify_failure(success=False, message="peg is misaligned by 3cm"),
            "alignment failure",
        )
        self.assertEqual(
            classify_failure(success=False, code_ok=False, message="NameError: robot.foo"),
            "api mismatch",
        )
        self.assertEqual(
            classify_failure(success=False, message="cube was not placed at goal"),
            "execution failure",
        )
        self.assertEqual(
            classify_failure(
                success=False,
                message="peg was not inserted; peg_head_pos_at_hole=[-0.1, 0.0, 0.0]",
            ),
            "insertion failure",
        )
        self.assertEqual(
            classify_failure(success=False, message="cubeA was not stably stacked on cubeB"),
            "placement stability failure",
        )
        self.assertEqual(
            classify_failure(
                success=False,
                message="tool pull failed; cube was not pulled into workspace; cube_distance=[0.59]",
            ),
            "tool-use execution failure",
        )

    def test_failure_layer_classifier(self):
        self.assertEqual(
            classify_failure_layer(
                success=False,
                code_ok=False,
                message="NameError: robot.missing",
            ),
            "program",
        )
        self.assertEqual(
            classify_failure_layer(
                success=False,
                info={
                    "execution_log": [
                        {
                            "api": "pull_with_tool",
                            "ok": False,
                            "message": "tool pull failed",
                        }
                    ]
                },
            ),
            "skill_adapter",
        )
        self.assertEqual(
            classify_failure_layer(
                success=False,
                info={
                    "execution_log": [
                        {
                            "api": "place",
                            "ok": False,
                            "message": "planner failed during pose plan",
                        }
                    ],
                    "final_info": {"planner_status": "failed"},
                },
            ),
            "controller_primitive",
        )

    def test_pull_cube_tool_prompt_forbids_direct_tool_grasp(self):
        request = MigrationRequest.from_ids(
            task_id="pull_cube_tool",
            target_robot="xarm6_robotiq",
            method="llm_card_report",
        )
        prompt = build_migration_prompt(request)
        self.assertIn("hook_object(tool, cube) already grasps", prompt)
        self.assertIn("Do not call robot.grasp(tool)", prompt)

    def test_iterative_prompt_exposes_pull_tool_parameters(self):
        task = get_task_spec("pull_cube_tool")
        prompt = build_iterative_prompt(
            task=task,
            source_robot="panda",
            target_robot="xarm6_robotiq",
            previous_attempts=[
                {
                    "attempt": 1,
                    "code": task.source_program,
                    "result": {
                        "success": False,
                        "failure_type": "tool-use execution failure",
                        "message": "tool pull failed",
                        "execution_log": [
                            {
                                "step": 1,
                                "api": "pull_with_tool",
                                "args": {"distance": 0.35},
                                "result": False,
                                "ok": False,
                            }
                        ],
                        "final_info": {"cube_distance": 0.59},
                    },
                }
            ],
        )
        self.assertIn("iterative robot code migration", prompt)
        self.assertIn("distance=0.35", prompt)
        self.assertIn("stages=1", prompt)
        self.assertIn("pull_frame", prompt)
        self.assertIn("Previous target attempts", prompt)
        self.assertIn("cube_distance", prompt)

    def test_code_diff_marks_parameter_changes(self):
        diff = _code_diff(
            "ret_val = robot.pull_with_tool(tool, cube, workspace, distance=0.35)",
            "ret_val = robot.pull_with_tool(tool, cube, workspace, distance=0.55, stages=3)",
        )
        self.assertIn("-ret_val", diff)
        self.assertIn("+ret_val", diff)
        self.assertIn("distance=0.55", diff)

    def test_real_failure_report_mentions_pick_cube_skill_failure(self):
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
            failure_type="execution failure",
            message="cube was not placed at goal",
            info={
                "execution_log": [
                    {"step": 1, "api": "grasp", "args": {"obj": "cube"}, "ok": True},
                    {
                        "step": 2,
                        "api": "place",
                        "args": {"obj": "cube", "target": "goal"},
                        "ok": False,
                        "message": "cube was not placed at goal",
                    },
                ]
            },
        )
        report = build_real_failure_report(task=task, target_profile=profile, failed_record=record)
        text = report.to_prompt_section()
        self.assertIn("pick_cube", text)
        self.assertIn("step 2: place", text)
        self.assertIn("real ManiSkill-backed skill", text)

    def test_pull_cube_tool_report_does_not_suggest_direct_grasp(self):
        task = get_task_spec("pull_cube_tool")
        profile = get_robot_profile("xarm6_robotiq")
        record = TrialRecord(
            task_id="pull_cube_tool",
            source_robot="panda",
            target_robot="xarm6_robotiq",
            method="source-copy",
            seed=0,
            generated_code=task.source_program,
            success=False,
            failure_type="tool-use execution failure",
            message="tool pull failed; cube was not pulled into workspace",
            info={
                "execution_log": [
                    {"step": 1, "api": "hook_object", "args": {"tool": "l_shape_tool", "obj": "cube"}, "ok": True},
                    {
                        "step": 2,
                        "api": "pull_with_tool",
                        "args": {"tool": "l_shape_tool", "obj": "cube", "target": "workspace"},
                        "ok": False,
                        "message": "tool pull failed; cube was not pulled into workspace",
                    },
                ]
            },
        )
        report = build_real_failure_report(task=task, target_profile=profile, failed_record=record)
        text = report.to_prompt_section()
        self.assertIn("source-level tool-use order was already correct", text)
        self.assertIn("Do not add robot.grasp(tool)", text)
        self.assertNotIn("requires grasping the tool before pulling", text)

    def test_results_summary_and_jsonl(self):
        records = [
            TrialRecord(
                task_id="pick_cube",
                source_robot="panda",
                target_robot="xarm6_robotiq",
                method="source-copy",
                seed=0,
                generated_code="ret_val = True",
                success=True,
                failure_type="success",
            )
        ]
        self.assertEqual(summarize_records(records)[0]["success_rate"], 1.0)
        self.assertEqual(summarize_records(records)[0]["dominant_failure_layer"], "success")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.jsonl"
            append_jsonl(path, records)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["task_id"], "pick_cube")

    def test_existing_lmp_executor_still_works(self):
        ok, message, _ = execute_lmp("ret_val = 2", {}, verbose=False)
        self.assertTrue(ok)
        self.assertEqual(message, "ret_val=2")

    def test_maniskill_import_check_returns_tuple(self):
        ok, message = can_import_maniskill()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(message, str)

    def test_sim_check_summarizes_dict(self):
        summary = summarize_value({"b": 1, "a": 2})
        self.assertEqual(summary["type"], "dict")
        self.assertEqual(summary["keys"], ["a", "b"])

    def test_records_to_md_formats_generated_code(self):
        record = TrialRecord(
            task_id="pick_cube",
            source_robot="panda",
            target_robot="xarm6_robotiq",
            method="source-copy",
            seed=0,
            generated_code="ret_val = True",
            success=True,
            failure_type="success",
            info={"execution_log": [{"step": 1, "api": "grasp", "args": {"obj": "cube"}, "ok": True}]},
        )
        text = records_to_md([record.to_dict()])
        self.assertIn("### Generated Code", text)
        self.assertIn("ret_val = True", text)
        self.assertIn("Execution Log", text)
        self.assertIn("failure_layer", text)


if __name__ == "__main__":
    unittest.main()
