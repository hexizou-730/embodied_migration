from __future__ import annotations

import json
import unittest
from pathlib import Path

from maniskill_backend.dynamic_harness import DynamicMigrationSpec
from maniskill_backend.selective_migration import (
    _entity_phrase,
    _selective_prompt,
    _stage_count_word,
    apply_stage_semantics,
    prompt_stage_program,
    validate_selective_adapter,
)


STAGES3 = ["approach", "grasp", "align"]
STACKCUBE_ORDER = ["approach", "grasp", "lift", "align", "release"]

T04_PROGRAM = {
    "stage_order": STAGES3,
    "stages": [
        {"stage_id": "approach", "entities": ["cube"], "completion_check": "generic approach text"},
        {"stage_id": "grasp", "entities": ["cube"], "completion_check": "generic grasp text"},
        {"stage_id": "align", "entities": ["cube", "goal_site"], "completion_check": "generic align text"},
    ],
}

SPEC_T04 = DynamicMigrationSpec(
    env_id="PickCube-v1", task_label="t04_pick_cube",
    source_robot="panda", target_robot="xarm6_robotiq",
    source_control_mode="pd_joint_pos", target_control_mode="pd_ee_delta_pose",
)


def _adapter_code(stages: list[str]) -> str:
    run_lines = "\n".join(
        f'        if not self._run_stage("{stage}", self._stage_{stage}): return False'
        for stage in stages
    )
    methods = []
    for stage in stages:
        methods.append(
            f'''    def _stage_{stage}(self) -> bool:
        for _ in range(1):
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


class PromptByteParityTests(unittest.TestCase):
    def test_stackcube_initial_prompt_is_byte_identical_to_frozen_fixture(self) -> None:
        spec = DynamicMigrationSpec(
            env_id="StackCube-v1", task_label="t07_stack_cube",
            source_robot="panda", target_robot="xarm6_robotiq",
            source_control_mode="pd_joint_pos", target_control_mode="pd_ee_delta_pose",
            seed=0, max_episode_steps=1000, max_target_cycles=3,
        )
        stage_program = json.loads(
            Path("evidence/stackcube_pilot_20261022/source_stage_program.json").read_text(encoding="utf-8")
        )
        observation = json.loads(
            Path(
                "evidence/stackcube_automatic_20261023/results/selective_migration/"
                "method_patch_r01/target_observation.json"
            ).read_text(encoding="utf-8")
        )
        prompt = _selective_prompt(
            spec=spec, stage_program=stage_program, target_observation=observation,
            incumbent_code="", previous_result={}, repair_stage=None,
            locked_stages=[], latest_error="",
        )
        fixture = Path("tests/fixtures/stackcube_selective_initial_prompt.txt").read_text(encoding="utf-8")
        self.assertEqual(prompt, fixture)


class StageGeneralizationTests(unittest.TestCase):
    def test_three_stage_prompt_adapts_counts_and_entities(self) -> None:
        prompt = _selective_prompt(
            spec=SPEC_T04, stage_program=T04_PROGRAM, target_observation={},
            incumbent_code="", previous_result={}, repair_stage=None,
            locked_stages=[], latest_error="",
        )
        self.assertIn("The public method must contain these three literal calls in exactly this order:", prompt)
        self.assertIn(
            '    if not self._run_stage("approach", self._stage_approach): return False\n'
            '    if not self._run_stage("grasp", self._stage_grasp): return False\n'
            '    if not self._run_stage("align", self._stage_align): return False\n'
            "    return self._official_success()",
            prompt,
        )
        self.assertIn("Use exact entity names cube and goal_site.", prompt)
        self.assertIn("# Implement exactly the three required _stage_* methods here.", prompt)
        self.assertIn("# Insert the three literal _run_stage calls shown above, unchanged.", prompt)
        self.assertNotIn("cubeA", prompt)
        self.assertNotIn("five", prompt)

    def test_entity_phrase_uses_first_seen_order(self) -> None:
        self.assertEqual(_entity_phrase(T04_PROGRAM), "cube and goal_site")
        self.assertEqual(_entity_phrase({"stages": [{"entities": ["cubeA"]}]}), "cubeA")
        self.assertEqual(_entity_phrase({"stages": []}), "")
        self.assertEqual(_stage_count_word(STAGES3), "three")
        self.assertEqual(_stage_count_word(STACKCUBE_ORDER), "five")
        self.assertEqual(_stage_count_word(["a"] * 7), "7")

    def test_stage_semantics_flow_into_prompt_and_record_provenance(self) -> None:
        study = {"stage_semantics": {
            "align": {"completion_check": "the cube rests on the goal site officially"},
            "grasp": {"completion_check": "official cube grasp is true"},
        }}
        program = apply_stage_semantics(T04_PROGRAM, study)
        self.assertEqual(program["stage_semantics_source"], "study_config_declared")
        self.assertEqual(
            program["stages"][2]["completion_check"], "the cube rests on the goal site officially",
        )
        self.assertEqual(program["stages"][0]["completion_check"], "generic approach text")
        prompt = _selective_prompt(
            spec=SPEC_T04, stage_program=program, target_observation={},
            incumbent_code="", previous_result={}, repair_stage=None,
            locked_stages=[], latest_error="",
        )
        self.assertIn("the cube rests on the goal site officially", prompt)
        self.assertIn("official cube grasp is true", prompt)
        self.assertNotIn("stage_semantics_source", prompt)
        self.assertNotIn("stage_semantics_source", prompt_stage_program(program))

    def test_semantics_are_optional_and_unknown_stages_fail_loudly(self) -> None:
        unchanged = apply_stage_semantics(T04_PROGRAM, {})
        self.assertEqual(unchanged["stages"], T04_PROGRAM["stages"])
        self.assertNotIn("stage_semantics_source", unchanged)
        with self.assertRaisesRegex(ValueError, "unknown stages: lift"):
            apply_stage_semantics(T04_PROGRAM, {"stage_semantics": {"lift": {"completion_check": "x"}}})

    def test_validate_adapter_accepts_variable_stage_orders(self) -> None:
        code3 = _adapter_code(STAGES3)
        hashes = validate_selective_adapter(code3, stage_order=STAGES3)
        self.assertEqual(set(name for name in hashes if name.startswith("_stage_")), {
            "_stage_approach", "_stage_grasp", "_stage_align",
        })
        with self.assertRaises(ValueError):
            validate_selective_adapter(_adapter_code(STACKCUBE_ORDER), stage_order=STAGES3)


if __name__ == "__main__":
    unittest.main()
