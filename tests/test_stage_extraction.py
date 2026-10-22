from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from maniskill_backend.stage_extraction import extract_stage_program, validate_stage_program


SOURCE = '''
def run(env, seed=None):
    obb = get_actor_obb(env.cubeA)
    grasp_pose = build_grasp_pose(obb)
    reach_pose = offset_pose(grasp_pose)
    planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    lift_pose = lift(grasp_pose)
    planner.move_to_pose_with_screw(lift_pose)
    goal_pose = env.cubeB.pose
    align_pose = align(lift_pose, goal_pose, env.cubeA.pose)
    planner.move_to_pose_with_screw(align_pose)
    planner.open_gripper()
'''


class StageExtractionTests(unittest.TestCase):
    def test_extracts_stack_stages_entities_and_runtime_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.py"
            source.write_text(SOURCE)
            lines = {
                text.strip(): index
                for index, text in enumerate(SOURCE.splitlines(), 1)
                if text.strip()
            }
            action_lines = [
                lines["planner.move_to_pose_with_screw(reach_pose)"],
                lines["planner.move_to_pose_with_screw(grasp_pose)"],
                lines["planner.close_gripper()"],
                lines["planner.move_to_pose_with_screw(lift_pose)"],
                lines["planner.move_to_pose_with_screw(align_pose)"],
                lines["planner.open_gripper()"],
            ]
            result = extract_stage_program(
                source,
                task_id="t07_stack_cube",
                source_robot="panda",
                source_control_mode="pd_joint_pos",
                known_entities=["cubeA", "cubeB"],
                source_action_trace=[
                    {"step": step, "source_line": line, "source_function": "run"}
                    for step, line in enumerate(action_lines, 1)
                ],
                official_evaluation={"success": True},
            )

        expected = ["approach", "grasp", "lift", "align", "release"]
        self.assertEqual(result["stage_order"], expected)
        self.assertEqual(result["stages"][0]["entities"], ["cubeA"])
        self.assertEqual(result["stages"][3]["entities"], ["cubeA", "cubeB"])
        self.assertEqual(result["stages"][4]["entities"], ["cubeA", "cubeB"])
        self.assertEqual(result["stages"][1]["runtime"]["action_count"], 2)
        self.assertEqual(result["trace_summary"]["unmapped_action_count"], 0)
        self.assertEqual(validate_stage_program(result, expected), [])

    def test_validation_rejects_missing_runtime_stage(self) -> None:
        payload = {
            "stage_order": ["approach"],
            "stages": [{
                "stage_id": "approach",
                "source_lines": [1, 1],
                "operations": ["move_tcp_to_pose"],
                "runtime": {"action_count": 0},
            }],
            "trace_summary": {"unmapped_action_count": 0},
        }
        self.assertIn("approach: no successful runtime actions attributed", validate_stage_program(payload, ["approach"]))


if __name__ == "__main__":
    unittest.main()
