from pathlib import Path
import unittest

import numpy as np

from grasp_probe import correct_gripper_endpoints, repair_grasp_close
from maniskill_backend.selective_migration import method_hashes


CANDIDATE = Path(__file__).resolve().parents[1] / "evidence/stackcube_pilot_20261022/selective_migration_phase3/artifacts/target_adapter.py"


class GraspDiagnosticTests(unittest.TestCase):
    def test_endpoint_correction_changes_only_configuration(self):
        original = CANDIDATE.read_text()
        mapped = correct_gripper_endpoints(original)
        before, after = method_hashes(original), method_hashes(mapped)
        self.assertEqual([name for name in after if before.get(name) != after[name]], ["__init__"])
        self.assertIn("gripper_open=0.0, gripper_close=0.81", mapped)

    def test_stationary_close_was_a_noop_and_repair_executes_actions(self):
        original = CANDIDATE.read_text()
        repaired = repair_grasp_close(original)
        hashes = method_hashes(original)
        changed = [name for name, value in method_hashes(repaired).items() if hashes.get(name) != value]
        self.assertEqual(changed, ["_stage_grasp"])
        for code, expected in ((original, 0), (repaired, 72)):
            namespace = {}
            exec(compile(code, "candidate", "exec"), namespace)
            cls = namespace["GeneratedSelectiveRobot"]
            robot = cls.__new__(cls)
            robot.move_steps, robot.settle_steps = 30, 12
            robot.gripper_open, robot.gripper_close = 0.81, 0.0
            robot.max_delta_m = 0.04
            robot._early_stop = lambda: False
            robot._entity_pos = lambda name: np.array([0.0, 0.0, 0.02])
            robot._tcp_pos = lambda: np.array([0.0, 0.0, 0.02])
            robot._is_grasping_entity = lambda name: False
            robot._make_action = lambda delta, *, gripper: [*delta, gripper]
            actions = []
            robot._step = actions.append
            self.assertFalse(robot._stage_grasp())
            self.assertEqual(len(actions), expected)
            self.assertTrue(all(action[-1] == 0.0 for action in actions))


if __name__ == "__main__":
    unittest.main()
