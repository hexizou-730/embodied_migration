import unittest

import numpy as np

from lmp.executor import execute_lmp
from maniskill_backend.env_adapter import can_import_maniskill
from maniskill_backend.evaluation import TrialRecord, classify_failure
from maniskill_backend.migration import MigrationRequest, build_migration_prompt
from maniskill_backend.profiles import get_robot_profile
from maniskill_backend.real_runner import _build_robot_adapter
from maniskill_backend.sim_check import run_check, summarize_value
from maniskill_backend.skill_adapter import ManiSkillPickCubeRobot, ManiSkillSceneAdapter
from maniskill_backend.static_benchmark import run_static_benchmark, summarize_records
from maniskill_backend.static_runner import build_static_report, run_static_trial
from maniskill_backend.tasks import get_task_spec
from maniskill_backend.view import records_to_md


class StaticBackendTest(unittest.TestCase):
    def test_profiles_and_tasks_are_promptable(self):
        profile = get_robot_profile("so100")
        task = get_task_spec("PegInsertionSide-v1")
        self.assertIn("Official Specs", profile.to_prompt_section())
        self.assertIn("Derived Task Priors", profile.to_prompt_section())
        self.assertIn("payload_kg", profile.to_prompt_section())
        self.assertIn("recommended_alignment_tolerance_m", profile.to_prompt_section())
        self.assertNotIn("\n  alignment_tolerance_m:", profile.to_prompt_section())
        self.assertIn("workspace_radius_m", profile.to_prompt_section())
        self.assertIn("PegInsertionSide-v1", task.to_prompt_section())

    def test_migration_prompt_card_report_method(self):
        request = MigrationRequest.from_ids(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="llm_card_only",
        )
        prompt = build_migration_prompt(request)
        self.assertIn("Capability Card", prompt)
        self.assertIn("so100", prompt)
        self.assertIn("ret_val", prompt)
        self.assertIn("robot.place", prompt)

    def test_pick_cube_task_is_available(self):
        task = get_task_spec("PickCube-v1")
        self.assertEqual(task.maniskill_env_id, "PickCube-v1")
        self.assertIn("robot.place", task.source_program)

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

    def test_real_runner_uses_longer_xarm_pick_cube_moves(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        robot = _build_robot_adapter("PickCube-v1", Env(), "pd_ee_delta_pos", "xarm6_robotiq")
        self.assertEqual(robot.gripper_open, -1.0)
        self.assertEqual(robot.gripper_close, 1.0)
        self.assertGreaterEqual(robot.move_steps, 36)

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

    def test_pick_cube_adapter_rejects_unsupported_action_shape(self):
        class Space:
            shape = (8,)
            dtype = np.float32
            low = -np.ones(8, dtype=np.float32)
            high = np.ones(8, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        with self.assertRaises(RuntimeError):
            ManiSkillPickCubeRobot(Env())

    def test_pick_cube_adapter_stops_on_terminated(self):
        counter = {"steps": 0}

        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                counter["steps"] += 1
                return None, 0.0, True, False, {}

        robot = ManiSkillPickCubeRobot(Env())
        robot._repeat_action(np.zeros(3), gripper=-1.0, steps=10)
        self.assertEqual(counter["steps"], 1)
        self.assertTrue(robot.terminated)
        self.assertTrue(robot._early_stop())

    def test_pick_cube_adapter_stops_on_truncated(self):
        counter = {"steps": 0}

        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                counter["steps"] += 1
                return None, 0.0, False, True, {}

        robot = ManiSkillPickCubeRobot(Env())
        robot._repeat_action(np.zeros(3), gripper=1.0, steps=10)
        self.assertEqual(counter["steps"], 1)
        self.assertTrue(robot.truncated)

    def test_pick_cube_adapter_clearance_overrides_are_stored(self):
        class Space:
            shape = (4,)
            dtype = np.float32
            low = -np.ones(4, dtype=np.float32)
            high = np.ones(4, dtype=np.float32)

        class Env:
            action_space = Space()

            def step(self, action):
                return None, 0.0, False, False, {}

        robot = ManiSkillPickCubeRobot(
            Env(),
            pregrasp_clearance_m=0.015,
            release_clearance_m=0.04,
            above_clearance_m=0.12,
            control_mode="pd_ee_delta_pose",
        )
        self.assertAlmostEqual(robot.pregrasp_clearance_m, 0.015)
        self.assertAlmostEqual(robot.release_clearance_m, 0.04)
        self.assertAlmostEqual(robot.above_clearance_m, 0.12)
        self.assertEqual(robot.control_mode, "pd_ee_delta_pose")

    def test_migration_prompt_no_card_baseline(self):
        request = MigrationRequest.from_ids(
            task_id="PullCubeTool-v1",
            target_robot="so100",
            method="llm_no_card",
        )
        prompt = build_migration_prompt(request)
        self.assertIn("No target Capability Card", prompt)
        self.assertNotIn("tool_hook_requires_alignment", prompt)
        self.assertNotIn("For tool tasks, call align_to_target", prompt)

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
            classify_failure(
                success=False,
                message="tool-use ordering failure: hook_object before alignment",
            ),
            "tool-use ordering failure",
        )

    def test_failure_classifier_recognises_explicit_refusal_phrases(self):
        self.assertEqual(
            classify_failure(
                success=False,
                message="ret_val='failure: requires sub-centimeter insertion precision'",
            ),
            "impossible-task refusal failure",
        )
        self.assertEqual(
            classify_failure(
                success=False,
                message="failure: target outside fixed-base workspace",
            ),
            "impossible-task refusal failure",
        )
        self.assertEqual(
            classify_failure(
                success=False,
                message="this is not feasible for the so100",
            ),
            "impossible-task refusal failure",
        )

    def test_failure_classifier_recognises_lmp_level_refusal(self):
        self.assertEqual(
            classify_failure(
                success=False,
                info={"ret_val": "'failure: capability mismatch'", "execution_log": []},
            ),
            "impossible-task refusal failure",
        )

    def test_failure_classifier_does_not_reclassify_real_skill_failures(self):
        self.assertEqual(
            classify_failure(
                success=False,
                message="alignment failure: tolerance too tight",
                info={
                    "ret_val": "'failure: alignment'",
                    "execution_log": [{"api": "align_to_target", "ok": False}],
                },
            ),
            "alignment failure",
        )

    def test_existing_lmp_executor_still_works(self):
        ok, message, _ = execute_lmp("ret_val = 2", {}, verbose=False)
        self.assertTrue(ok)
        self.assertEqual(message, "ret_val=2")

    def test_maniskill_import_check_returns_tuple(self):
        ok, message = can_import_maniskill()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(message, str)

    def test_sim_check_import_only_result_shape(self):
        result = run_check(env_id="PegInsertionSide-v1")
        self.assertIn("import_ok", result)
        self.assertIn("env_id", result)
        self.assertIn("robot_uid", result)

    def test_sim_check_summarizes_dict(self):
        summary = summarize_value({"b": 1, "a": 2})
        self.assertEqual(summary["type"], "dict")
        self.assertEqual(summary["keys"], ["a", "b"])

    def test_static_source_copy_fails_for_limited_robot(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="source-copy",
        )
        self.assertFalse(record.success)
        self.assertEqual(record.failure_type, "alignment failure")
        self.assertIn("execution_log", record.info)
        self.assertEqual(record.info["execution_log"][0]["api"], "grasp")
        self.assertEqual(record.info["execution_log"][1]["api"], "align_to_target")
        self.assertFalse(record.info["execution_log"][1]["ok"])

    def test_static_oracle_succeeds_for_limited_robot(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="oracle",
        )
        self.assertTrue(record.success)
        self.assertEqual(record.failure_type, "success")

    def test_static_benchmark_summary(self):
        records = run_static_benchmark(
            tasks=("PegInsertionSide-v1",),
            methods=("source-copy", "oracle"),
            seeds=(0,),
        )
        rows = summarize_records(records)
        self.assertEqual(len(records), 8)
        self.assertEqual(len(rows), 8)
        oracle_rows = [row for row in rows if row["method"] == "oracle"]
        self.assertTrue(all(row["success_rate"] == 1.0 for row in oracle_rows))

    def test_plug_charger_source_copy_and_oracle(self):
        source_copy = run_static_trial(
            task_id="PlugCharger-v1",
            target_robot="so100",
            method="source-copy",
        )
        oracle = run_static_trial(
            task_id="PlugCharger-v1",
            target_robot="so100",
            method="oracle",
        )
        self.assertFalse(source_copy.success)
        self.assertEqual(source_copy.failure_type, "insertion speed failure")
        self.assertTrue(oracle.success)

    def test_plug_charger_report_mentions_contact_speed(self):
        record = run_static_trial(
            task_id="PlugCharger-v1",
            target_robot="so100",
            method="llm_card_report",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertIn("Failure Report", record.failure_report)
        self.assertIn("target Capability Card", record.failure_report)
        self.assertIn("insertion speed failure", record.failure_report)
        self.assertIn("Execution log failed at", record.failure_report)
        self.assertIn("failed_skill_call", record.failure_report)
        self.assertNotIn("speed=0.008", record.failure_report)
        self.assertEqual(record.info["report_source_method"], "source-copy")
        self.assertFalse(record.info["report_source_log"][-1]["ok"])

    def test_pick_cube_report_mentions_real_grasp_failure(self):
        task = get_task_spec("PickCube-v1")
        profile = get_robot_profile("xarm6_robotiq")
        failed = TrialRecord(
            task_id=task.task_id,
            source_robot=task.source_robot,
            target_robot=profile.name,
            method="source-copy",
            seed=0,
            generated_code=task.source_program,
            success=False,
            failure_type="gripper/force failure",
            message="cube was not grasped",
            prompt="",
            info={
                "execution_log": [
                    {
                        "step": 1,
                        "api": "grasp",
                        "args": {"obj": "cube"},
                        "result": False,
                        "ok": False,
                        "message": "cube was not grasped",
                        "failure_type": "execution failure",
                    }
                ]
            },
        )
        report = build_static_report(
            task=task,
            target_profile=profile,
            failed_record=failed,
        ).to_prompt_section()
        self.assertIn("grasp_cube", report)
        self.assertIn("cube was not grasped", report)
        self.assertIn("real high-level skill wrapper", report)
        self.assertIn("control mode", report)
        self.assertNotIn("insert(...", report)

    def test_plug_multi_source_copy_and_oracle(self):
        source_copy = run_static_trial(
            task_id="PlugMulti-v1",
            target_robot="so100",
            method="source-copy",
        )
        oracle = run_static_trial(
            task_id="PlugMulti-v1",
            target_robot="so100",
            method="oracle",
        )
        self.assertFalse(source_copy.success)
        self.assertEqual(source_copy.failure_type, "alignment failure")
        self.assertTrue(oracle.success)

    def test_plug_multi_report_mentions_all_causes(self):
        record = run_static_trial(
            task_id="PlugMulti-v1",
            target_robot="so100",
            method="llm_card_report",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertIn("every align_to_target", record.failure_report)
        self.assertIn("max(target ik_accuracy_m, target recommended_alignment_tolerance_m)", record.failure_report)
        self.assertIn("0.010 m empirical contact margin", record.failure_report)
        self.assertIn("75% of the target robot insertion speed limit", record.failure_report)
        self.assertNotIn("tolerance=0.045", record.failure_report)
        self.assertNotIn("speed=0.006", record.failure_report)
        self.assertIn("multiple coupled failure causes", record.failure_report)

    def test_pull_cube_tool_source_copy_and_oracle(self):
        source_copy = run_static_trial(
            task_id="PullCubeTool-v1",
            target_robot="so100",
            method="source-copy",
        )
        oracle = run_static_trial(
            task_id="PullCubeTool-v1",
            target_robot="so100",
            method="oracle",
        )
        self.assertFalse(source_copy.success)
        self.assertEqual(source_copy.failure_type, "tool-use ordering failure")
        self.assertIn("hook_object", source_copy.message)
        self.assertTrue(oracle.success)

    def test_peg_multi_source_copy_and_oracle(self):
        source_copy = run_static_trial(
            task_id="PegMulti-v1",
            target_robot="so100",
            method="source-copy",
        )
        oracle = run_static_trial(
            task_id="PegMulti-v1",
            target_robot="so100",
            method="oracle",
        )
        self.assertFalse(source_copy.success)
        self.assertEqual(source_copy.failure_type, "alignment failure")
        self.assertTrue(oracle.success)

    def test_peg_multi_report_mentions_all_causes(self):
        record = run_static_trial(
            task_id="PegMulti-v1",
            target_robot="so100",
            method="llm_card_report",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertIn("every align_to_target", record.failure_report)
        self.assertIn("max(target ik_accuracy_m, target recommended_alignment_tolerance_m)", record.failure_report)
        self.assertIn("0.010 m empirical contact margin", record.failure_report)
        self.assertIn("75% of the target robot insertion speed limit", record.failure_report)
        self.assertNotIn("tolerance=0.045", record.failure_report)
        self.assertNotIn("speed=0.006", record.failure_report)
        self.assertIn("multiple coupled failure causes", record.failure_report)

    def test_three_task_benchmark_summary(self):
        records = run_static_benchmark(
            tasks=("PegInsertionSide-v1", "PlugCharger-v1", "PullCubeTool-v1"),
            methods=("source-copy", "oracle"),
            seeds=(0,),
        )
        rows = summarize_records(records)
        self.assertEqual(len(records), 24)
        self.assertEqual(len(rows), 24)
        self.assertEqual(
            {"PegInsertionSide-v1", "PlugCharger-v1", "PullCubeTool-v1"},
            {row["task_id"] for row in rows},
        )

    def test_llm_card_only_dry_run_uses_fallback_without_api(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="llm_card_only",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertFalse(record.info["used_llm"])
        self.assertEqual(record.info["llm_reason"], "dry_run")

    def test_llm_no_card_is_supported(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="llm_no_card",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertFalse(record.info["used_llm"])

    def test_llm_card_report_dry_run_includes_failure_report(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="llm_card_report",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertIn("Failure Report", record.failure_report)
        self.assertIn("target Capability Card", record.failure_report)
        self.assertIn("max(target ik_accuracy_m, target recommended_alignment_tolerance_m)", record.failure_report)
        self.assertNotIn("tolerance=0.035", record.failure_report)

    def test_pull_cube_report_mentions_tool_ordering(self):
        record = run_static_trial(
            task_id="PullCubeTool-v1",
            target_robot="so100",
            method="llm_card_report",
            dry_run=True,
        )
        self.assertTrue(record.success)
        self.assertIn("Failure Report", record.failure_report)
        self.assertIn("hook_object", record.failure_report)
        self.assertIn("align_to_target", record.failure_report)
        self.assertIn("max(target ik_accuracy_m, target recommended_alignment_tolerance_m)", record.failure_report)

    def test_old_failure_method_name_still_works_as_alias(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="llm_card_failure",
            dry_run=True,
        )
        self.assertEqual(record.method, "llm_card_report")

    def test_records_to_md_formats_generated_code(self):
        record = run_static_trial(
            task_id="PegInsertionSide-v1",
            target_robot="so100",
            method="oracle",
        )
        text = records_to_md([record.to_dict()])
        self.assertIn("## Trial 1", text)
        self.assertIn("### Capability Card", text)
        self.assertIn("### Execution Log", text)
        self.assertIn("### Generated Code", text)
        self.assertIn("robot.insert", text)

    def test_view_md_can_include_multiple_methods(self):
        records = [
            run_static_trial(
                task_id="PegInsertionSide-v1",
                target_robot="so100",
                method="source-copy",
            ).to_dict(),
            run_static_trial(
                task_id="PegInsertionSide-v1",
                target_robot="so100",
                method="oracle",
            ).to_dict(),
        ]
        text = records_to_md(records)
        self.assertIn("`source-copy`", text)
        self.assertIn("`oracle`", text)


if __name__ == "__main__":
    unittest.main()
