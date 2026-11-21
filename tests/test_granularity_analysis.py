from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from maniskill_backend.granularity_analysis import (
    analyze_ablation_run,
    bootstrap_mean_ci,
    load_ablation_run,
    sign_test_p,
    wilson_interval,
)


CANDIDATE_SHA = "f" * 64


def _cycle(valid: bool, *, action_steps: int = 0, tokens: int = 0, cost: float = 0.0,
           success: bool = False, error: str = "") -> dict:
    return {
        "cycle": 1, "model": "test", "used_llm": True,
        "usage": {"total_tokens": tokens, "cost": cost},
        "valid": valid, "action_steps": action_steps, "success": success,
        "error": error,
    }


def _summary(arm_id: str, records: list[dict], *, success: bool, sampling_seed: int) -> dict:
    return {
        "schema": "repair_granularity_module_rewrite.v1" if arm_id == "module_rewrite" else "selective_migration_result.v1",
        "arm_id": arm_id,
        "status": "success" if success else "target_budget_exhausted",
        "success": success,
        "repair_output_format": "complete_module" if arm_id == "module_rewrite" else "single_stage_method",
        "generation_settings": {"sampling_seed": sampling_seed},
        "resume_replay": {"action_steps": 101},
        "failed_candidate_sha256": CANDIDATE_SHA if arm_id == "module_rewrite" else None,
        "cycles": records,
    }


def _write_run(root: Path, *, aggregate: dict | None = None) -> None:
    root.mkdir(parents=True)
    (root / "repetition_manifest.json").write_text(json.dumps({
        "sampling_seeds": [0, 1],
        "failed_candidate": {"path": "candidates/failed.py", "sha256": CANDIDATE_SHA},
    }), encoding="utf-8")
    method_r1 = [_cycle(True, action_steps=31, tokens=1000, cost=0.001, success=True)]
    module_r1 = [
        _cycle(True, action_steps=60, tokens=3000, cost=0.002, success=False),
        _cycle(True, action_steps=80, tokens=6000, cost=0.003, success=True),
    ]
    method_r2 = [_cycle(True, action_steps=35, tokens=1200, cost=0.001, success=False)]
    module_r2 = [_cycle(False, tokens=800, error="semantically unchanged")]
    plan = {
        ("r01", "method_patch"): (method_r1, True, 0),
        ("r01", "module_rewrite"): (module_r1, True, 0),
        ("r02", "method_patch"): (method_r2, False, 1),
        ("r02", "module_rewrite"): (module_r2, False, 1),
    }
    for (rep, arm), (records, success, seed) in plan.items():
        arm_dir = root / rep / arm
        arm_dir.mkdir(parents=True)
        for index, record in enumerate(records, start=1):
            cycle_dir = arm_dir / f"cycle_{index:02d}"
            cycle_dir.mkdir()
            (cycle_dir / "cycle_record.json").write_text(json.dumps(record), encoding="utf-8")
        (arm_dir / "summary.json").write_text(
            json.dumps(_summary(arm, records, success=success, sampling_seed=seed)), encoding="utf-8"
        )
    if aggregate is None:
        aggregate = {
            "arms": [
                {"arm_id": "method_patch", "num_runs": 2, "num_success": 1,
                 "target_action_steps": 66, "total_tokens": 2200},
                {"arm_id": "module_rewrite", "num_runs": 2, "num_success": 1,
                 "target_action_steps": 140, "total_tokens": 9800},
            ],
            "paired_by_sampling_seed": [{}, {}],
        }
    (root / "summary.json").write_text(json.dumps(aggregate), encoding="utf-8")


class StatsPrimitiveTests(unittest.TestCase):
    def test_wilson_interval_bounds(self) -> None:
        interval = wilson_interval(2, 3)
        self.assertIsNotNone(interval)
        low, high = interval
        self.assertLessEqual(low, 2 / 3)
        self.assertGreaterEqual(high, 2 / 3)
        self.assertEqual(wilson_interval(0, 0), None)

    def test_sign_test_exact_values(self) -> None:
        self.assertIsNone(sign_test_p(0, 0))
        self.assertEqual(sign_test_p(1, 0), 1.0)
        self.assertAlmostEqual(sign_test_p(8, 0), 2 * (0.5 ** 8))
        self.assertEqual(sign_test_p(2, 1), 1.0)

    def test_bootstrap_is_deterministic_and_bracketing(self) -> None:
        values = [109.0, -35.0]
        first = bootstrap_mean_ci(values)
        second = bootstrap_mean_ci(values)
        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        low, high = first
        self.assertEqual((low, high), (-35.0, 109.0))
        self.assertIsNone(bootstrap_mean_ci([]))


class GranularityAnalysisTests(unittest.TestCase):
    def test_full_analysis_recomputes_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root)
            output = root / "analysis"
            payload = analyze_ablation_run(root, output_dir=output)
            report = (output / "granularity_analysis.md").read_text(encoding="utf-8")
            self.assertTrue((output / "granularity_analysis.json").is_file())
        self.assertEqual(payload["evidence_checks"]["failed"], 0)
        rows = {row["arm_id"]: row for row in payload["arms"]}
        self.assertEqual(rows["method_patch"]["num_runs"], 2)
        self.assertEqual(rows["method_patch"]["num_success"], 1)
        self.assertEqual(rows["method_patch"]["target_action_steps"], 66)
        self.assertEqual(rows["method_patch"]["replay_action_steps_excluded"], 202)
        self.assertEqual(rows["method_patch"]["llm_calls"], 2)
        self.assertEqual(rows["module_rewrite"]["num_success"], 1)
        self.assertEqual(rows["module_rewrite"]["generation_cycles"], 3)
        self.assertEqual(rows["module_rewrite"]["valid_candidates"], 2)
        self.assertEqual(rows["module_rewrite"]["target_action_steps"], 140)
        self.assertEqual(rows["module_rewrite"]["total_tokens"], 9800)
        self.assertIsNotNone(rows["method_patch"]["success_wilson_95"])
        paired = payload["paired"]
        self.assertEqual(paired["num_pairs"], 2)
        self.assertEqual(paired["discordant_method_only"], 0)
        self.assertEqual(paired["discordant_module_only"], 0)
        self.assertIsNone(paired["sign_test_p_two_sided"])
        self.assertEqual(paired["action_difference_bootstrap_95"], [-35.0, 109.0])
        self.assertEqual(paired["mean_action_difference_module_minus_method"], 37.0)
        self.assertEqual(paired["mean_token_difference_module_minus_method"], 3800.0)
        self.assertEqual(paired["token_difference_bootstrap_95"], [-400.0, 8000.0])
        self.assertTrue(any("not separated from zero" in note for note in paired["cost_guidance"]))
        self.assertIn("| method_patch | 2 | 1 |", report)
        self.assertIn("| module_rewrite | 2 | 1 |", report)
        self.assertIn("resume-based", report.lower())

    def test_drift_between_raw_evidence_and_summary_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            _write_run(root)
            tampered = root / "r01" / "method_patch" / "cycle_01" / "cycle_record.json"
            record = json.loads(tampered.read_text(encoding="utf-8"))
            record["action_steps"] = 99
            tampered.write_text(json.dumps(record), encoding="utf-8")
            payload = analyze_ablation_run(root, output_dir=root / "analysis")
            self.assertGreater(payload["evidence_checks"]["failed"], 0)
            names = {failure["name"] for failure in payload["evidence_checks"]["failed_details"]}
            self.assertTrue(any("target_action_steps_drift" in name for name in names))
            report = (root / "analysis" / "granularity_analysis.md").read_text(encoding="utf-8")
            self.assertIn("FAILED:", report)

    def test_missing_manifest_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_ablation_run(Path(tmp))


if __name__ == "__main__":
    unittest.main()
