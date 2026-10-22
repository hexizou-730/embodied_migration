from pathlib import Path
import unittest

from placement_probe import ALIGN, RELEASE, replace_stage
from maniskill_backend.selective_migration import method_hashes


class PlacementDiagnosticTests(unittest.TestCase):
    def test_interventions_preserve_approach_grasp_lift(self):
        path = Path(__file__).resolve().parents[1] / "evidence/stackcube_followup_20261023/results/selective_migration/corrected_gripper/cycle_02/target_adapter.py"
        original = path.read_text()
        align = replace_stage(original, "align", ALIGN)
        both = replace_stage(align, "release", RELEASE)
        before, aligned, released = map(method_hashes, (original, align, both))
        self.assertEqual([name for name in aligned if aligned[name] != before[name]], ["_stage_align"])
        self.assertEqual([name for name in released if released[name] != before[name]], ["_stage_align", "_stage_release"])
        with self.assertRaisesRegex(ValueError, "declared stage"):
            replace_stage(original, "align", RELEASE)


if __name__ == "__main__":
    unittest.main()
