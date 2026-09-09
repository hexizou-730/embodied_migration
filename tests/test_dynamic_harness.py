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
    extract_task_program,
    looks_like_env_id,
    read_source_actions,
    run_dynamic_agent_migration,
    validate_dynamic_adapter,
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
        run_trial.side_effect = [
            {"success": True, "message": "supplied source success"},
            {
                "success": False,
                "message": "contact missed",
                "state_trace": [{"step": 10, "tcp": [0.1, 0.2, 0.3]}],
            },
            {"success": True, "message": "target success"},
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
        self.assertIn("contact missed", second_prompt)
        self.assertIn('"step": 10', second_prompt)
        self.assertIn("Current failed target adapter", second_prompt)
        self.assertIn("class GeneratedRobot", second_prompt)

    def test_migrate_cli_accepts_unregistered_env_in_dry_run(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(
                [
                    sys.executable,
                    "migrate.py",
                    "--task",
                    "TurnFaucet-v1",
                    "--source",
                    "panda",
                    "--target",
                    "fetch",
                    "--mode",
                    "agent",
                    "--dry-run",
                    "--output-root",
                    tmp,
                    "--run-name",
                    "dynamic_cli_test",
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            payload = json.loads((Path(tmp) / "dynamic_cli_test" / "summary.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["case"]["registered"])
            self.assertIsNone(payload["result"]["success"])

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
