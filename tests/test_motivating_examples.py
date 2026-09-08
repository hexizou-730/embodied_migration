import json
import tempfile
import unittest
from pathlib import Path

from maniskill_backend.motivating_examples import (
    build_motivating_examples,
    motivating_examples_markdown,
    summarize_pick_probe,
    summarize_pull_multiseed,
)


class MotivatingExamplesTest(unittest.TestCase):
    def test_builds_traceable_examples_from_real_style_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pull = root / "pull.jsonl"
            pull_rows = [
                {"type": "metadata", "seeds": [0, 1, 2, 3, 4]},
                {"type": "trial", "seed": 0, "success": True},
                {
                    "type": "trial",
                    "seed": 1,
                    "success": False,
                    "message": "Episode ended during descent.",
                    "diagnostics": {"stage": "descent", "tcp_stage_error_norm": 0.14},
                },
                {
                    "type": "trial",
                    "seed": 2,
                    "success": False,
                    "message": "Episode ended during approach.",
                    "diagnostics": {"stage": "approach", "tcp_stage_error_norm": 0.31},
                },
                {"type": "trial", "seed": 3, "success": True},
                {
                    "type": "trial",
                    "seed": 4,
                    "success": False,
                    "message": "Episode ended during descent.",
                    "diagnostics": {"stage": "descent", "tcp_stage_error_norm": 0.09},
                },
            ]
            pull.write_text("\n".join(json.dumps(row) for row in pull_rows), encoding="utf-8")

            probe = root / "pick.json"
            probe.write_text(
                json.dumps(
                    {
                        "schema": "structured_probe_result.v1",
                        "probe_id": "pick_cube_xarm6_close_envelope",
                        "dry_run": False,
                        "all_probe_cases": [
                            {
                                "grasp_z_offset": 0.016,
                                "close_steps": 12,
                                "close_command": -0.6,
                                "settle_steps": 8,
                                "tcp_grasp_xy": 0.0024,
                                "tcp_grasp_z": 0.0015,
                                "cube_disp_xy": 0.0046,
                                "is_grasping_after_close": False,
                                "is_grasping_after_lift": False,
                                "score": -5.49,
                            },
                            {
                                "grasp_z_offset": 0.012,
                                "is_grasping_after_close": False,
                                "score": -6.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_motivating_examples(pull_results=pull, pick_probe=probe)
            self.assertTrue(payload["paper_ready"])
            pull_summary = summarize_pull_multiseed(pull)
            self.assertEqual(pull_summary["success_rate"], 0.4)
            self.assertEqual(pull_summary["failure_stage_counts"], {"approach": 1, "descent": 2})
            pick_summary = summarize_pick_probe(probe)
            self.assertEqual(pick_summary["num_probe_cases"], 2)
            self.assertEqual(pick_summary["num_grasping_cases"], 0)
            self.assertNotEqual(pick_summary["source"]["sha256"], "missing")
            markdown = motivating_examples_markdown(payload)
            self.assertIn("Only values recomputed", markdown)
            self.assertNotIn("must not be quoted", markdown)

    def test_missing_files_are_explicitly_not_paper_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = build_motivating_examples(
                pull_results=root / "missing_pull.jsonl",
                pick_probe=root / "missing_pick.json",
            )
            self.assertFalse(payload["paper_ready"])
            self.assertEqual(len(payload["missing_or_incomplete_examples"]), 2)
            self.assertIn("must not be quoted", motivating_examples_markdown(payload))


if __name__ == "__main__":
    unittest.main()
