from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from maniskill_backend.llm import LLMTextResult
from maniskill_backend.source_programs import (
    freeze_source_program,
    load_source_program_catalog,
    source_program_status,
    statically_check_candidate,
    synthesize_source_program,
)


class SourceProgramTests(unittest.TestCase):
    def test_catalog_covers_twenty_tasks_with_requested_source_split(self) -> None:
        _, specs = load_source_program_catalog()
        self.assertEqual(len(specs), 20)
        by_id = {spec.task_id: spec for spec in specs}
        self.assertEqual(by_id["t12_open_cabinet_drawer"].source_robot, "fetch")
        self.assertEqual(by_id["t13_open_cabinet_door"].source_robot, "fetch")
        for task_id, spec in by_id.items():
            if task_id not in {"t12_open_cabinet_drawer", "t13_open_cabinet_door"}:
                self.assertTrue(spec.source_robot.startswith("panda"), task_id)

    def test_reference_candidates_are_statically_consistent(self) -> None:
        _, specs = load_source_program_catalog()
        ready = [spec for spec in specs if spec.origin.startswith("maniskill_3.0.1")]
        self.assertEqual(len(ready), 11)
        for spec in ready:
            result = statically_check_candidate(spec)
            self.assertTrue(result["ok"], (spec.task_id, result))

    def test_all_twenty_candidates_exist_but_none_are_predeclared_frozen(self) -> None:
        _, specs = load_source_program_catalog()
        rows = {row["task_id"]: row for row in source_program_status(specs)}
        self.assertEqual(len(rows), 20)
        for row in rows.values():
            self.assertEqual(row["candidate_status"], "candidate_ready")
            self.assertFalse(row["frozen"])

    def test_freeze_rejects_partial_multi_seed_success(self) -> None:
        _, specs = load_source_program_catalog()
        spec = specs[0]
        validation = {
            "success": False,
            "trials": [
                {"seed": 0, "success": True},
                {"seed": 1, "success": False},
            ],
        }
        with self.assertRaisesRegex(ValueError, "cannot be frozen"):
            freeze_source_program(spec, validation, minimum_seed_count=2)

    def test_freeze_writes_code_and_checksum_record_after_full_pass(self) -> None:
        _, specs = load_source_program_catalog()
        original = specs[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.py"
            candidate.write_text("ENV_ID = 'X-v1'\n", encoding="utf-8")
            spec = replace(
                original,
                candidate_path=candidate,
                frozen_path=root / "frozen" / "task.py",
            )
            validation = {
                "success": True,
                "environment_contract_sha256": "a" * 64,
                "trials": [
                    {"seed": 0, "success": True},
                    {"seed": 1, "success": True},
                ],
            }
            record = freeze_source_program(spec, validation, minimum_seed_count=2)
            saved = json.loads(spec.frozen_path.with_suffix(".json").read_text())
        self.assertEqual(record["source_sha256"], saved["source_sha256"])
        self.assertEqual(saved["validated_seeds"], [0, 1])
        self.assertEqual(saved["success_rate"], 1.0)

    def test_synthesis_contract_mismatch_stops_before_simulator_and_llm(self) -> None:
        _, specs = load_source_program_catalog()
        spec = specs[0]
        runtime = {
            "contract_sha256": "a" * 64,
            "validation": {
                "ok": False,
                "mismatches": [
                    {"field": "python.version", "expected": "3.10.20", "observed": "3.12.2"}
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_programs.capture_experiment_environment",
            return_value=runtime,
        ), patch(
            "maniskill_backend.source_programs.discover_environment"
        ) as discover, patch(
            "maniskill_backend.source_programs.gen_text"
        ) as generate:
            result = synthesize_source_program(
                spec,
                seeds=[0, 1],
                output_dir=Path(tmp),
                max_cycles=2,
                obs_mode="state",
                sim_backend="auto",
                render_backend="gpu",
            )
        self.assertEqual(result["status"], "environment_contract_mismatch")
        discover.assert_not_called()
        generate.assert_not_called()

    def test_invalid_generation_is_shown_to_next_repair_cycle(self) -> None:
        _, specs = load_source_program_catalog()
        invalid = '''
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot
ENV_ID = "RollBall-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"
TASK_PROGRAM = "ret_val = robot.solve_task()"
# FIRST_INVALID_MARKER
class GeneratedRobot(ManiSkillDynamicRobot):
    def solve_task(self):
        for _ in range(2):
            if self._early_stop():
                return False
            self._step(self._make_action(np.zeros(3), gripper=0.0))
        return False
def build_robot(env, *, control_mode, robot_uid):
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text(invalid.replace("# FIRST_INVALID_MARKER\n", ""), encoding="utf-8")
            spec = replace(specs[2], candidate_path=bootstrap, frozen_path=root / "frozen.py")
            runtime = {
                "contract_sha256": "a" * 64,
                "validation": {"ok": True, "mismatches": []},
            }
            generated = LLMTextResult(invalid, True, "test-model", raw_text=invalid)
            with patch(
                "maniskill_backend.source_programs.capture_experiment_environment",
                return_value=runtime,
            ), patch(
                "maniskill_backend.source_programs.discover_environment",
                return_value={"robot_uid": "panda", "action_space": {"shape": [4]}},
            ), patch(
                "maniskill_backend.source_programs.gen_text",
                side_effect=[generated, generated],
            ) as generate:
                result = synthesize_source_program(
                    spec,
                    seeds=[0, 1],
                    output_dir=root / "out",
                    max_cycles=2,
                    obs_mode="state",
                    sim_backend="auto",
                    render_backend="gpu",
                )

        second_prompt = generate.call_args_list[1].kwargs["prompt"]
        self.assertFalse(result["success"])
        self.assertIn("FIRST_INVALID_MARKER", second_prompt)
        self.assertIn("self._snapshot()", second_prompt)
        self.assertIn("Generated adapter must read physical state", second_prompt)


if __name__ == "__main__":
    unittest.main()
