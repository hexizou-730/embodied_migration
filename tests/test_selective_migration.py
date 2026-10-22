from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot
from maniskill_backend.selective_migration import (
    completed_stages,
    apply_stage_method,
    _selective_prompt,
    failed_stage,
    method_hashes,
    locate_repair_stage,
    stage_boundary_states,
    run_selective_migration,
    prompt_stage_program,
    validate_selective_adapter,
)
from maniskill_backend.llm import LLMTextResult


STAGES = ["approach", "grasp", "lift", "align", "release"]


def _adapter_code(*, grasp_steps: int = 1, swapped: bool = False) -> str:
    calls = list(STAGES)
    if swapped:
        calls[0], calls[1] = calls[1], calls[0]
    run_lines = "\n".join(
        f'        if not self._run_stage("{stage}", self._stage_{stage}): return False'
        for stage in calls
    )
    methods = []
    for stage in STAGES:
        count = grasp_steps if stage == "grasp" else 1
        methods.append(
            f'''    def _stage_{stage}(self) -> bool:
        for _ in range({count}):
            state = self._snapshot()
            if self._early_stop():
                return False
            self._step(self._make_action(np.zeros(3, dtype=np.float32), gripper=0.0))
            if state and self._official_success():
                return True
        return False
'''
        )
    return f'''from typing import Any
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot

class GeneratedRobot(ManiSkillDynamicRobot):
{''.join(methods)}
    def run_source_actions(self) -> bool:
{run_lines}
        return self._official_success()

def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> GeneratedRobot:
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = SimpleNamespace(
            shape=(4,),
            dtype=np.float32,
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
        )
        self.agent = SimpleNamespace(
            tcp_pose=SimpleNamespace(p=np.array([0.0, 0.0, 0.2], dtype=np.float32)),
        )
        self.unwrapped = self

    def evaluate(self):
        return {"success": False}

    def step(self, action):
        return {}, 0.0, False, False, {}


class SelectiveMigrationTests(unittest.TestCase):
    def test_initial_prompt_does_not_include_human_validation_annotations(self) -> None:
        from maniskill_backend.dynamic_harness import DynamicMigrationSpec
        program = {"stage_order": STAGES, "stages": [{"stage_id": "align", "transferable_relation": "EXTRACTED_SOURCE_FACT"}],
                   "validation": {"manual_evaluation_only": "MANUAL_LABEL_SECRET"},
                   "input_evidence": {"study": "PRIVATE_STUDY_PATH"}}
        prompt = _selective_prompt(
            spec=DynamicMigrationSpec(env_id="StackCube-v1", task_label="stack", source_robot="panda", target_robot="xarm6_robotiq"),
            stage_program=program, target_observation={}, incumbent_code="", previous_result={},
            repair_stage=None, locked_stages=[], latest_error="",
        )
        self.assertIn("EXTRACTED_SOURCE_FACT", prompt)
        self.assertNotIn("MANUAL_LABEL_SECRET", prompt)
        self.assertNotIn("PRIVATE_STUDY_PATH", prompt)
        self.assertNotIn("validation", prompt_stage_program(program))

    def test_dry_run_cannot_replay_real_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "real actions"):
            run_selective_migration(study_path=Path("unused.json"), run_dir=Path("unused"),
                                    dry_run=True, resume_adapter=Path("unused.py"))

    def test_resume_replays_and_repairs_upstream_stage_without_changing_passed_prefix(self) -> None:
        code = _adapter_code()
        repaired = code.replace("def _stage_align(self) -> bool:\n        for _ in range(1):", "def _stage_align(self) -> bool:\n        for _ in range(2):")
        node = next(n for n in ast.walk(ast.parse(repaired)) if isinstance(n, ast.FunctionDef) and n.name == "_stage_align")
        replacement = ast.get_source_segment(repaired, node)
        replay = {
            "success": False, "action_steps": 12,
            "runtime_snapshot": {"stage_runtime": {"completed_stages": STAGES[:-1], "failed_stage": "release"}},
            "source_action_trace": [{"source_function": "_stage_align", "state_after": {
                "step": 8, "official_evaluation": {"on_support": False},
            }}],
        }
        source = {"stage_order": STAGES, "stages": [
            {"stage_id": "lift", "runtime": {"final_state": {"official_evaluation": {"on_support": False}}}},
            {"stage_id": "align", "runtime": {"final_state": {"official_evaluation": {"on_support": True}}}},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_path = root / "benchmark_tasks/task/study.json"
            study_path.parent.mkdir(parents=True)
            study_path.write_text(json.dumps({
                "source": {"stage_representation": "stages.json", "robot_uid": "panda", "control_mode": "pd_joint_pos"},
                "target": {"robot_uid": "xarm6_robotiq", "control_mode": "pd_ee_delta_pose"},
                "task": {"env_id": "StackCube-v1", "task_id": "stack"},
            }))
            (root / "stages.json").write_text(json.dumps(source))
            candidate = root / "candidate.py"
            candidate.write_text(code)
            with patch("maniskill_backend.selective_migration.discover_environment", return_value={}), \
                    patch("maniskill_backend.selective_migration.run_dynamic_trial", side_effect=[replay, {"success": True, "action_steps": 14}]) as trial, \
                    patch("maniskill_backend.selective_migration.gen_text", return_value=LLMTextResult(text=replacement, used_llm=True, model="test")) as generate:
                result = run_selective_migration(study_path=study_path, run_dir=root / "output", max_cycles=1, resume_adapter=candidate)
            self.assertEqual(trial.call_count, 2)
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(result["success"])
            self.assertEqual(result["cycles"][0]["allowed_stage"], "align")
            self.assertEqual(result["cycles"][0]["locked_stages_before"], ["approach", "grasp", "lift"])
            self.assertEqual(result["resume_replay"]["repair_localization"]["runtime_failed_stage"], "release")
            self.assertEqual(result["cycles"][0]["generation_unit"], "stage_method")

    def test_method_patch_preserves_all_other_code_and_rejects_extra_definitions(self) -> None:
        original = _adapter_code()
        new = _adapter_code(grasp_steps=2)
        method = next(n for n in ast.walk(ast.parse(new)) if isinstance(n, ast.FunctionDef) and n.name == "_stage_grasp")
        snippet = ast.get_source_segment(new, method)
        assembled = apply_stage_method(original, snippet, stage_id="grasp")
        validate_selective_adapter(assembled, stage_order=STAGES, incumbent_code=original, allowed_stage="grasp")
        self.assertEqual(assembled.split("    def _stage_grasp")[0], original.split("    def _stage_grasp")[0])
        self.assertEqual(assembled.split("    def _stage_lift")[1], original.split("    def _stage_lift")[1])
        for invalid in ("import os\n" + snippet, "class Replacement:\n    pass", snippet + "\ndef other():\n    pass"):
            with self.assertRaisesRegex(ValueError, "Return only"):
                apply_stage_method(original, invalid, stage_id="grasp")
        with self.assertRaisesRegex(ValueError, "sole replacement"):
            apply_stage_method(original, snippet, stage_id="align")

    def test_method_prompt_is_focused_on_measured_failure(self) -> None:
        from maniskill_backend.dynamic_harness import DynamicMigrationSpec
        prompt = _selective_prompt(
            spec=DynamicMigrationSpec(env_id="StackCube-v1", task_label="stack", source_robot="panda", target_robot="xarm6_robotiq"),
            stage_program={"stage_order": STAGES, "stages": [{"stage_id": "align", "transferable_relation": "SOURCE_RELATION"}]},
            target_observation={}, incumbent_code=_adapter_code(), previous_result={},
            repair_stage="align", locked_stages=["approach", "grasp", "lift"], latest_error="unchanged",
        )
        self.assertIn("Return exactly one Python function", prompt)
        self.assertIn("SOURCE_RELATION", prompt)
        self.assertNotIn("def _stage_grasp", prompt)
        self.assertNotIn("class GeneratedRobot", prompt)

    def test_source_postcondition_can_reopen_a_falsely_passed_stage(self) -> None:
        source = {"stages": [
            {"stage_id": "lift", "runtime": {"final_state": {"official_evaluation": {"on_support": False}}}},
            {"stage_id": "align", "runtime": {"final_state": {"official_evaluation": {"on_support": True}}}},
        ]}
        result = {
            "runtime_snapshot": {"stage_runtime": {"completed_stages": ["lift", "align"], "failed_stage": "release"}},
            "source_action_trace": [
                {"source_function": "_stage_align", "state_after": {
                    "step": 20, "tcp": [[0.0, 0.0, 0.06]],
                    "entities": {"held": [[0.01, 0.0, 0.07]], "segmentation_id_map[2]": [[0, 0, 0]]},
                    "official_evaluation": {"on_support": False},
                }},
            ],
        }
        location = locate_repair_stage(result, source)
        self.assertEqual(location["runtime_failed_stage"], "release")
        self.assertEqual(location["repair_stage"], "align")
        self.assertEqual(location["evidence"]["violated_predicates"], ["on_support"])
        self.assertNotIn("segmentation_id_map[2]", stage_boundary_states(result)["align"]["entities"])
        result["source_action_trace"][0]["state_after"]["official_evaluation"]["on_support"] = True
        self.assertEqual(locate_repair_stage(result, source)["repair_stage"], "release")
        result["source_action_trace"] = []
        self.assertEqual(locate_repair_stage(result, source)["repair_stage"], "release")

    def test_official_success_takes_priority_over_intermediate_stage_mismatch(self) -> None:
        result = {"success": True, "runtime_snapshot": {"stage_runtime": {"failed_stage": "release"}}}
        self.assertIsNone(locate_repair_stage(result, {"stages": []})["repair_stage"])

    def test_stage_interface_accepts_exact_source_order(self) -> None:
        hashes = validate_selective_adapter(_adapter_code(), stage_order=STAGES)
        self.assertEqual(set(name for name in hashes if name.startswith("_stage_")), {
            f"_stage_{stage}" for stage in STAGES
        })

    def test_stage_interface_rejects_reordered_program(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen stage order"):
            validate_selective_adapter(_adapter_code(swapped=True), stage_order=STAGES)

    def test_stage_interface_rejects_policy_helpers_outside_stages(self) -> None:
        code = _adapter_code().replace(
            "    def _stage_approach(self) -> bool:\n",
            "    def _helper(self) -> bool:\n"
            "        return True\n\n"
            "    def _stage_approach(self) -> bool:\n",
        )
        with self.assertRaisesRegex(ValueError, "policy helpers"):
            validate_selective_adapter(code, stage_order=STAGES)

    def test_stage_interface_rejects_raw_entity_catalog(self) -> None:
        code = _adapter_code().replace(
            "            state = self._snapshot()\n",
            "            state = self._entity_catalog()\n",
            1,
        )
        with self.assertRaisesRegex(ValueError, "_entity_pos"):
            validate_selective_adapter(code, stage_order=STAGES)

    def test_local_repair_changes_only_failed_stage(self) -> None:
        incumbent = _adapter_code(grasp_steps=1)
        repaired = _adapter_code(grasp_steps=2)
        hashes = validate_selective_adapter(
            repaired,
            stage_order=STAGES,
            incumbent_code=incumbent,
            allowed_stage="grasp",
        )
        self.assertNotEqual(
            method_hashes(incumbent)["_stage_grasp"],
            hashes["_stage_grasp"],
        )
        with self.assertRaisesRegex(ValueError, "Only _stage_approach"):
            validate_selective_adapter(
                repaired,
                stage_order=STAGES,
                incumbent_code=incumbent,
                allowed_stage="approach",
            )

    def test_runtime_records_completed_and_failed_stages(self) -> None:
        robot = ManiSkillDynamicRobot(
            _FakeEnv(),
            control_mode="pd_ee_delta_pos",
            robot_uid="fake",
        )

        def passed() -> bool:
            robot._step(robot._make_action(np.zeros(3, dtype=np.float32), gripper=0.0))
            return True

        def failed() -> bool:
            robot._step(robot._make_action(np.zeros(3, dtype=np.float32), gripper=0.0))
            return False

        self.assertTrue(robot._run_stage("approach", passed))
        self.assertFalse(robot._run_stage("grasp", failed))
        result = {
            "runtime_snapshot": robot._snapshot(),
            "execution_log": robot.execution_log(),
        }
        self.assertEqual(completed_stages(result), ["approach"])
        self.assertEqual(failed_stage(result), "grasp")

    def test_runtime_rejects_zero_action_stage_success(self) -> None:
        robot = ManiSkillDynamicRobot(
            _FakeEnv(),
            control_mode="pd_ee_delta_pos",
            robot_uid="fake",
        )
        self.assertFalse(robot._run_stage("approach", lambda: True))
        self.assertEqual(robot.failed_stage, "approach")


if __name__ == "__main__":
    unittest.main()
