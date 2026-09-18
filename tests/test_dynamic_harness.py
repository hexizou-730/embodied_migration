from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot
from maniskill_backend.dynamic_harness import (
    DynamicMigrationSpec,
    _dynamic_diagnosis,
    _prompt_trial,
    extract_task_program,
    looks_like_env_id,
    read_source_actions,
    run_dynamic_agent_migration,
    run_dynamic_trial,
    validate_dynamic_adapter,
)
from maniskill_backend.experiment_environment import (
    compare_with_contract,
    load_experiment_contract,
)
from maniskill_backend.llm import LLMTextResult


SOURCE_CODE = '''
from __future__ import annotations
from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

TASK_PROGRAM = "ret_val = robot.solve_task()"

class GeneratedRobot(ManiSkillDynamicRobot):
    def solve_task(self) -> bool:
        for _ in range(2):
            if self._early_stop():
                return self._fail("solve_task", {}, "episode ended")
            state = self._snapshot()
            self._step(self._make_action(np.zeros(3, dtype=np.float32), gripper=0.0))
            if state and self._official_success():
                return self._log("solve_task", {}, True, True, "")
        return self._fail("solve_task", {}, "official task outcome not reached")

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''


TARGET_CODE = SOURCE_CODE.replace('TASK_PROGRAM = "ret_val = robot.solve_task()"\n', "")


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(
            shape=(4,),
            dtype=np.float32,
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.cube = SimpleNamespace(
            pose=SimpleNamespace(
                p=np.array([0.1, 0.2, 0.03], dtype=np.float32),
                q=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        )
        self.agent = SimpleNamespace(
            tcp_pose=SimpleNamespace(p=np.array([0.0, 0.0, 0.2], dtype=np.float32)),
            is_grasping=lambda actor: actor is self.cube,
        )
        self.unwrapped = self

    def evaluate(self):
        return {"success": False, "object_at_goal": False}

    def step(self, action):
        return {}, 0.0, False, False, {"success": False}


class DynamicHarnessTests(unittest.TestCase):
    def test_multi_seed_prompt_keeps_bounded_physical_counterexamples(self) -> None:
        failure = {
            "seed": 0,
            "success": False,
            "message": "ball left workspace",
            "action_steps": 20,
            "state_trace": [
                {"step": 0, "entities": {"ball": [[0.0, 0.2, 0.035]]}},
                {"step": 20, "entities": {"ball": [[3.0, 2.0, -0.8]]}},
            ],
        }
        compact = _prompt_trial(
            {
                "success": False,
                "message": "4/4 validation seeds failed",
                "multi_seed_failures": [failure, failure, failure, failure],
            }
        )
        self.assertEqual(compact["multi_seed_failure_count"], 4)
        self.assertEqual(len(compact["multi_seed_failure_sample"]), 3)
        self.assertEqual(
            compact["multi_seed_failure_sample"][0]["state_trace_sample"][-1]["entities"]["ball"],
            [[3.0, 2.0, -0.8]],
        )

    def test_diagnosis_detects_manipulated_object_leaving_workspace(self) -> None:
        diagnosis = _dynamic_diagnosis(
            {
                "success": False,
                "code_ok": True,
                "state_trace": [
                    {
                        "step": 0,
                        "entities": {
                            "ball": [[-0.1, 0.65, 0.035]],
                            "goal_region": [[-0.3, -0.8, 0.001]],
                        },
                    },
                    {
                        "step": 200,
                        "entities": {
                            "ball": [[10.2, 3.1, -0.88]],
                            "goal_region": [[-0.3, -0.8, 0.001]],
                        },
                    },
                ],
            }
        )
        self.assertEqual(diagnosis["reason"], "manipulated_object_left_workspace")
        self.assertEqual(diagnosis["evidence"]["entity"], "ball")
        self.assertTrue(diagnosis["evidence"]["fell_below_workspace"])

    def test_official_success_termination_is_not_reported_as_runtime_failure(self) -> None:
        env = _FakeEnv()
        succeeded = False

        def evaluate():
            return {"success": succeeded}

        def step(action):
            nonlocal succeeded
            succeeded = True
            return {}, 1.0, True, False, {"success": True}

        env.evaluate = evaluate
        env.step = step

        def run(recorded, seed=None):
            recorded.step(np.zeros(4, dtype=np.float32))
            raise AssertionError("execution must stop at official success")

        spec = DynamicMigrationSpec(
            env_id="PushCube-v1",
            task_label="t01_push_cube",
            source_robot="panda",
            target_robot="panda",
            source_control_mode="pd_joint_pos",
            target_control_mode="pd_joint_pos",
            seed=0,
        )
        with patch("maniskill_backend.dynamic_harness.ManiSkillEnvAdapter") as adapter_cls, patch(
            "maniskill_backend.dynamic_harness._load_module",
            return_value=SimpleNamespace(run=run),
        ):
            adapter = adapter_cls.return_value
            adapter.make.return_value = env
            adapter.reset.return_value = ({}, {})
            result = run_dynamic_trial(
                spec=spec,
                robot_uid="panda",
                control_mode="pd_joint_pos",
                program="ret_val = True",
                adapter_path=Path("unused.py"),
                source_entrypoint="run",
            )

        self.assertTrue(result["success"])
        self.assertTrue(result["official_success"])
        self.assertEqual(result["action_steps"], 1)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])

    def test_experiment_contract_detects_runtime_mismatch(self) -> None:
        contract = load_experiment_contract()
        actual = {
            "python": {"version": "3.12.2"},
            "packages": {
                "mani_skill": {"version": None},
                "sapien": {"version": None},
            },
            "cuda": {"driver_api": None},
            "llm": dict(contract["llm"]),
            "migration": {
                "source_robot": "panda",
                "target_robot": "xarm6_robotiq",
                "seed": 0,
            },
        }
        mismatches = compare_with_contract(contract, actual)
        fields = {item["field"] for item in mismatches}
        self.assertIn("python.version", fields)
        self.assertIn("packages.mani_skill.version", fields)
        self.assertIn("cuda.driver_api", fields)

    def test_env_id_detection(self) -> None:
        self.assertTrue(looks_like_env_id("TurnFaucet-v1"))
        self.assertTrue(looks_like_env_id("StackPyramid-v12"))
        self.assertFalse(looks_like_env_id("turn_faucet"))

    def test_read_source_actions_accepts_plain_run_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.py"
            path.write_text(
                'ENV_ID = "PushCube-v1"\n'
                'SOURCE_ROBOT = "panda"\n'
                'def run(env):\n'
                '    env.step([0.0, 0.0, 0.0, -1.0])\n',
                encoding="utf-8",
            )
            source = read_source_actions(path)
        self.assertEqual(source.env_id, "PushCube-v1")
        self.assertEqual(source.robot_uid, "panda")
        self.assertEqual(source.entrypoint, "run")
        self.assertEqual(source.program, "ret_val = robot.run_source_actions()")

    def test_extract_and_validate_source_module(self) -> None:
        program = extract_task_program(SOURCE_CODE)
        self.assertEqual(program, "ret_val = robot.solve_task()")
        validate_dynamic_adapter(SOURCE_CODE, task_program=program, require_program_constant=True)

    def test_validate_target_against_frozen_program(self) -> None:
        validate_dynamic_adapter(
            TARGET_CODE,
            task_program="ret_val = robot.solve_task()",
            require_program_constant=False,
        )

    def test_validate_accepts_trusted_inherited_action_helpers(self) -> None:
        code = '''
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedRobot(ManiSkillDynamicRobot):
    def solve_task(self):
        state = self._snapshot()
        if state:
            self._move_towards(np.zeros(3), gripper=0.0, steps=2)
        return self._official_success()

def build_robot(env, *, control_mode, robot_uid):
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''
        validate_dynamic_adapter(
            code,
            task_program="ret_val = robot.solve_task()",
            require_program_constant=False,
        )

    def test_validate_accepts_region_position_as_physical_state(self) -> None:
        code = '''
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedRobot(ManiSkillDynamicRobot):
    def solve_task(self):
        goal = self._region_pos("goal")
        if goal.size:
            self._move_towards(goal, gripper=0.0, steps=2)
        return self._official_success()

def build_robot(env, *, control_mode, robot_uid):
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''
        validate_dynamic_adapter(
            code,
            task_program="ret_val = robot.solve_task()",
            require_program_constant=False,
        )

    def test_missing_state_error_lists_documented_measurement_api(self) -> None:
        code = TARGET_CODE.replace("state = self._snapshot()", "state = True").replace(
            "self._official_success()", "False"
        )
        with self.assertRaisesRegex(ValueError, r"self\._snapshot\(\.\.\.\)"):
            validate_dynamic_adapter(
                code,
                task_program="ret_val = robot.solve_task()",
                require_program_constant=False,
            )

    def test_overridden_helper_does_not_bypass_action_validation(self) -> None:
        code = '''
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedRobot(ManiSkillDynamicRobot):
    def _move_towards(self, target, *, gripper, steps):
        return None

    def solve_task(self):
        state = self._snapshot()
        if state:
            self._move_towards(np.zeros(3), gripper=0.0, steps=2)
        return self._official_success()

def build_robot(env, *, control_mode, robot_uid):
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''
        with self.assertRaisesRegex(ValueError, "execute actions"):
            validate_dynamic_adapter(
                code,
                task_program="ret_val = robot.solve_task()",
                require_program_constant=False,
            )

    def test_rejects_direct_pose_mutation(self) -> None:
        dangerous = TARGET_CODE.replace(
            "state = self._snapshot()",
            "state = self._snapshot()\n            self._entity('cube').set_pose(None)",
        )
        with self.assertRaisesRegex(ValueError, "simulator-state mutation"):
            validate_dynamic_adapter(
                dangerous,
                task_program="ret_val = robot.solve_task()",
                require_program_constant=False,
            )

    def test_rejects_direct_environment_step(self) -> None:
        dangerous = TARGET_CODE.replace(
            "state = self._snapshot()",
            "state = self._snapshot()\n            self.env.step(None)",
        )
        with self.assertRaisesRegex(ValueError, "simulator-state mutation"):
            validate_dynamic_adapter(
                dangerous,
                task_program="ret_val = robot.solve_task()",
                require_program_constant=False,
            )

    def test_rejects_unlisted_setter_mutation(self) -> None:
        dangerous = TARGET_CODE.replace(
            "state = self._snapshot()",
            "state = self._snapshot()\n            self._entity('cube').set_linear_velocity(None)",
        )
        with self.assertRaisesRegex(ValueError, "simulator-state mutation"):
            validate_dynamic_adapter(
                dangerous,
                task_program="ret_val = robot.solve_task()",
                require_program_constant=False,
            )

    def test_dynamic_robot_exposes_state_without_task_policy(self) -> None:
        robot = ManiSkillDynamicRobot(
            _FakeEnv(),
            control_mode="pd_ee_delta_pos",
            robot_uid="panda",
        )
        snapshot = robot._snapshot()
        np.testing.assert_allclose(snapshot["entities"]["cube"]["position"], [0.1, 0.2, 0.03])
        self.assertTrue(robot._is_grasping_entity("cube"))
        self.assertFalse(snapshot["official_evaluation"]["success"])

    def test_dynamic_robot_flattens_single_environment_pose_vectors(self) -> None:
        env = _FakeEnv()
        env.agent.tcp_pose.p = np.array([[0.0, 0.0, 0.2]], dtype=np.float32)
        env.cube.pose.p = np.array([[0.1, 0.2, 0.03]], dtype=np.float32)
        robot = ManiSkillDynamicRobot(
            env,
            control_mode="pd_ee_delta_pos",
            robot_uid="panda",
        )
        self.assertEqual(robot._tcp_pos().shape, (3,))
        self.assertEqual(robot._entity_pos("cube").shape, (3,))

    def test_numeric_goals_and_object_velocity_are_available_read_only(self) -> None:
        env = _FakeEnv()
        env.goal_pos = np.array([[0.4, 0.2, 0.1]], dtype=np.float32)
        env.goal_pose = SimpleNamespace(p=env.goal_pos, q=np.array([[1., 0., 0., 0.]]))
        env.cube.linear_velocity = np.array([[0.1, -0.2, 0.0]])
        robot = ManiSkillDynamicRobot(env, control_mode="pd_ee_delta_pos", robot_uid="panda")
        np.testing.assert_allclose(robot._region_pos("goal_pos"), [0.4, 0.2, 0.1])
        np.testing.assert_allclose(robot._region_pos("goal_pose"), [0.4, 0.2, 0.1])
        state = robot._snapshot()
        np.testing.assert_allclose(state["entity_velocities"]["cube"]["linear_velocity"], [0.1, -0.2, 0.])
        state["task_state"]["goal_pos"][0][0] = 99
        self.assertAlmostEqual(float(env.goal_pos[0, 0]), 0.4)
        env.invalid_point = np.array([1., 2.])
        with self.assertRaisesRegex(ValueError, "not an xyz point"):
            robot._region_pos("invalid_point")

    def test_world_delta_is_rotated_into_robot_root_controller_frame(self) -> None:
        env = _FakeEnv()
        half = np.sqrt(0.5)
        env.agent.robot = SimpleNamespace(
            pose=SimpleNamespace(
                q=np.array([[half, 0.0, 0.0, -half]], dtype=np.float32)
            )
        )
        robot = ManiSkillDynamicRobot(
            env,
            control_mode="pd_ee_delta_pos",
            robot_uid="panda",
        )
        action = robot._make_action(
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            gripper=0.25,
        )
        np.testing.assert_allclose(action[:3], [0.0, 1.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(action[3]), 0.25)

    def test_dynamic_robot_resolves_unambiguous_cube_to_obj_alias(self) -> None:
        env = _FakeEnv()
        env.obj = env.cube
        del env.cube
        env.agent.is_grasping = lambda actor: actor is env.obj
        robot = ManiSkillDynamicRobot(
            env,
            control_mode="pd_ee_delta_pos",
            robot_uid="xarm6_robotiq",
        )
        np.testing.assert_allclose(robot._entity_pos("cube"), [0.1, 0.2, 0.03])
        self.assertEqual(robot._snapshot()["entity_aliases"]["cube"], "obj")

    def test_dynamic_dry_run_writes_manifest_without_llm_or_simulation(self) -> None:
        spec = DynamicMigrationSpec(
            env_id="TurnFaucet-v1",
            task_label="TurnFaucet-v1",
            source_robot="panda",
            target_robot="fetch",
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_dynamic_agent_migration(spec=spec, run_dir=run_dir, dry_run=True)
            self.assertIsNone(result["success"])
            self.assertEqual(result["status"], "dry_run_planned")
            manifest = json.loads((run_dir / "case_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["registered_case"])
            self.assertEqual(manifest["env_id"], "TurnFaucet-v1")
            self.assertTrue((run_dir / "runtime_environment.json").is_file())
            runtime = json.loads((run_dir / "runtime_environment.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime["schema"], "embodied_migration_runtime.v1")
            self.assertEqual(runtime["actual"]["migration"]["target_robot"], "fetch")
            self.assertEqual(len(runtime["contract_sha256"]), 64)

    def test_environment_contract_blocks_real_run_before_llm(self) -> None:
        spec = DynamicMigrationSpec(
            env_id="TurnFaucet-v1",
            task_label="TurnFaucet-v1",
            source_robot="panda",
            target_robot="fetch",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "maniskill_backend.dynamic_harness.capture_experiment_environment"
            ) as capture, patch(
                "maniskill_backend.dynamic_harness.gen_text"
            ) as generate:
                capture.return_value = {
                    "contract_sha256": "a" * 64,
                    "validation": {
                        "ok": False,
                        "mismatches": [
                            {"field": "python.version", "expected": "3.10.20", "observed": "3.12.2"}
                        ],
                    },
                }
                result = run_dynamic_agent_migration(
                    spec=spec,
                    run_dir=Path(tmp) / "run",
                    enforce_environment=True,
                )
        self.assertEqual(result["status"], "environment_contract_mismatch")
        self.assertFalse(result["success"])
        generate.assert_not_called()

    @patch("maniskill_backend.dynamic_harness.run_dynamic_trial")
    @patch("maniskill_backend.dynamic_harness.discover_environment")
    @patch("maniskill_backend.dynamic_harness.gen_text")
    def test_dynamic_run_archives_prompts_trials_and_hashes(
        self,
        generate,
        discover,
        run_trial,
    ) -> None:
        generate.side_effect = [
            LLMTextResult(SOURCE_CODE, True, "test-model", raw_text=SOURCE_CODE),
            LLMTextResult(TARGET_CODE, True, "test-model", raw_text=TARGET_CODE),
        ]
        discover.side_effect = [
            {"robot_uid": "panda", "action_space": {"shape": [4]}},
            {"robot_uid": "fetch", "action_space": {"shape": [9]}},
        ]
        run_trial.side_effect = [
            {"success": True, "official_success": True, "message": "source success"},
            {"success": True, "official_success": True, "message": "target success"},
        ]
        spec = DynamicMigrationSpec(
            env_id="TurnFaucet-v1",
            task_label="TurnFaucet-v1",
            source_robot="panda",
            target_robot="fetch",
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_dynamic_agent_migration(spec=spec, run_dir=run_dir)
            self.assertTrue(result["success"])
            self.assertTrue((run_dir / "source_observation.json").is_file())
            self.assertTrue((run_dir / "target_observation.json").is_file())
            for role in ("source", "target"):
                cycle_dir = run_dir / f"{role}_cycle_01"
                self.assertTrue((cycle_dir / "system_prompt.txt").is_file())
                self.assertTrue((cycle_dir / "user_prompt.txt").is_file())
                self.assertTrue((cycle_dir / "trial_result.json").is_file())
                cycle_record = json.loads((cycle_dir / "cycle_record.json").read_text(encoding="utf-8"))
                self.assertEqual(len(cycle_record["prompt_sha256"]), 64)
                self.assertEqual(len(cycle_record["adapter_sha256"]), 64)
            manifest = json.loads((run_dir / "case_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["outcome"]["success"])
            self.assertTrue(all(result["artifact_sha256"].values()))

    @patch("maniskill_backend.dynamic_harness.run_dynamic_trial")
    @patch("maniskill_backend.dynamic_harness.discover_environment")
    @patch("maniskill_backend.dynamic_harness.gen_text")
    def test_supplied_source_actions_skip_source_generation(
        self,
        generate,
        discover,
        run_trial,
    ) -> None:
        generate.return_value = LLMTextResult(TARGET_CODE, True, "test-model", raw_text=TARGET_CODE)
        discover.side_effect = [
            {"robot_uid": "panda", "action_space": {"shape": [4]}},
            {"robot_uid": "fetch", "action_space": {"shape": [9]}},
        ]
        run_trial.side_effect = [
            {"success": True, "message": "supplied source success"},
            {"success": True, "message": "target success"},
        ]
        spec = DynamicMigrationSpec(
            env_id="TurnFaucet-v1", task_label="TurnFaucet-v1",
            source_robot="panda", target_robot="fetch",
        )
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.py"
            source_path.write_text(SOURCE_CODE, encoding="utf-8")
            source = read_source_actions(source_path)
            result = run_dynamic_agent_migration(
                spec=spec, run_dir=Path(tmp) / "run", source_actions=source,
            )
        self.assertTrue(result["success"])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(run_trial.call_count, 2)
        self.assertEqual(result["source_cycles"][0]["origin"], "user_supplied")

    @patch("maniskill_backend.dynamic_harness.run_dynamic_trial")
    @patch("maniskill_backend.dynamic_harness.discover_environment")
    @patch("maniskill_backend.dynamic_harness.gen_text")
    def test_failed_target_trace_is_fed_to_next_llm_round(
        self,
        generate,
        discover,
        run_trial,
    ) -> None:
        repaired = TARGET_CODE.replace("range(2)", "range(3)")
        generate.side_effect = [
            LLMTextResult(TARGET_CODE, True, "test-model", raw_text=TARGET_CODE),
            LLMTextResult(repaired, True, "test-model", raw_text=repaired),
        ]
        discover.side_effect = [
            {"robot_uid": "panda", "action_space": {"shape": [4]}},
            {"robot_uid": "fetch", "action_space": {"shape": [9]}},
        ]
        execution_error = (
            "Traceback (most recent call last):\n"
            "AttributeError: Unknown task entity 'cube'. Available pose entities: obj"
        )
        run_trial.side_effect = [
            {"success": True, "message": "supplied source success"},
            {
                "success": False,
                "message": "Task program failed before the first simulator action",
                "code_ok": False,
                "execution_error": execution_error,
                "action_steps": 0,
                "state_trace": [{"step": 0, "tcp": [0.1, 0.2, 0.3]}],
            },
            {"success": True, "message": "target success", "action_steps": 1},
        ]
        spec = DynamicMigrationSpec(
            env_id="TurnFaucet-v1", task_label="TurnFaucet-v1",
            source_robot="panda", target_robot="fetch", max_target_cycles=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.py"
            source_path.write_text(SOURCE_CODE, encoding="utf-8")
            result = run_dynamic_agent_migration(
                spec=spec,
                run_dir=Path(tmp) / "run",
                source_actions=read_source_actions(source_path),
            )
        second_prompt = generate.call_args_list[1].kwargs["prompt"]
        self.assertTrue(result["success"])
        self.assertIn("Unknown task entity 'cube'", second_prompt)
        self.assertIn('"action_steps": 0', second_prompt)
        self.assertIn('"step": 0', second_prompt)
        self.assertIn("Current failed target adapter", second_prompt)
        self.assertIn("class GeneratedRobot", second_prompt)
        first, second = result["target_cycles"]
        self.assertNotEqual(first["semantic_sha256"], second["semantic_sha256"])
        self.assertTrue(second["changed_from_previous"])
        self.assertEqual(second["trial"]["action_steps"], 1)

    @patch("maniskill_backend.dynamic_harness.run_dynamic_trial")
    @patch("maniskill_backend.dynamic_harness.discover_environment")
    @patch("maniskill_backend.dynamic_harness.gen_text")
    def test_formatting_only_target_repair_is_rejected(
        self,
        generate,
        discover,
        run_trial,
    ) -> None:
        formatting_only = "\n\n# no behavioral change\n" + TARGET_CODE
        generate.side_effect = [
            LLMTextResult(TARGET_CODE, True, "test-model", raw_text=TARGET_CODE),
            LLMTextResult(formatting_only, True, "test-model", raw_text=formatting_only),
        ]
        discover.side_effect = [
            {"robot_uid": "panda", "action_space": {"shape": [4]}},
            {"robot_uid": "fetch", "action_space": {"shape": [9]}},
        ]
        run_trial.side_effect = [
            {"success": True, "message": "supplied source success"},
            {"success": False, "message": "target failed", "action_steps": 1},
        ]
        spec = DynamicMigrationSpec(
            env_id="TurnFaucet-v1", task_label="TurnFaucet-v1",
            source_robot="panda", target_robot="fetch", max_target_cycles=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.py"
            source_path.write_text(SOURCE_CODE, encoding="utf-8")
            result = run_dynamic_agent_migration(
                spec=spec,
                run_dir=Path(tmp) / "run",
                source_actions=read_source_actions(source_path),
            )
        self.assertFalse(result["success"])
        self.assertIn("semantically unchanged", result["target_cycles"][1]["error"])
        self.assertEqual(run_trial.call_count, 2)

    def test_migrate_cli_accepts_source_file_without_registered_case(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "actions.py"
            source_path.write_text(
                'ENV_ID = "TurnFaucet-v1"\nSOURCE_ROBOT = "panda"\n'
                'def run(env):\n    env.step([0.0, 0.0, 0.0, 0.0])\n',
                encoding="utf-8",
            )
            process = subprocess.run(
                [sys.executable, "migrate.py", str(source_path), "fetch",
                 "--dry-run", "--output-root", tmp, "--run-name", "source_file_test"],
                cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            self.assertIn("status = dry_run_planned", process.stdout)
            payload = json.loads((Path(tmp) / "source_file_test" / "dynamic_summary.json").read_text())
            self.assertIsNone(payload["success"])


if __name__ == "__main__":
    unittest.main()
