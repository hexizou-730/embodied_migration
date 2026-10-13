from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maniskill_backend.generation_stability import run_generation_stability


class GenerationStabilityTests(unittest.TestCase):
    def _plan(self) -> dict:
        return {
            "task_id": "t01_push_cube",
            "source_robot": "panda",
            "target_robot": "xarm6_robotiq",
            "target_control_mode": "pd_ee_delta_pos",
            "seed": 0,
            "repetitions": 3,
            "max_cycles": 8,
            "max_episode_steps": 1000,
            "task_timeout": 20,
            "asset_timeout": 5,
            "download_assets": False,
            "obs_mode": "state",
            "sim_backend": "auto",
            "render_backend": "gpu",
            "source_hashes": {"t01_push_cube": "abc"},
        }

    @patch("maniskill_backend.generation_stability.run_worker")
    @patch("maniskill_backend.generation_stability.capture_experiment_environment")
    def test_independent_repetitions_and_metric_aggregation(self, capture, run_worker) -> None:
        capture.return_value = {"validation": {"ok": True, "mismatches": []}}

        def fake_worker(command, *, log, timeout):
            worker_path = Path(command[-1])
            config = json.loads(worker_path.read_text())
            repetition = int(worker_path.parent.name.split("_")[-1])
            success = repetition in {1, 3}
            result = {
                "success": success,
                "status": "success" if success else "target_budget_exhausted",
                "message": "done",
                "metrics": {
                    "generation_cycles": repetition,
                    "repair_cycles": max(0, repetition - 1),
                    "target_action_steps": repetition * 10,
                    "total_tokens": repetition * 100,
                    "llm_cost_usd": repetition / 100,
                },
            }
            Path(config["output_dir"], "worker_result.json").write_text(json.dumps(result))
            return 0, False

        run_worker.side_effect = fake_worker
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "stability"
            result = run_generation_stability(self._plan(), output)
            saved = json.loads((output / "summary.json").read_text())

        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["successful_repetitions"], 2)
        self.assertAlmostEqual(result["generation_success_rate"], 2 / 3)
        self.assertEqual(result["total_generation_cycles"], 6)
        self.assertEqual(result["total_target_action_steps"], 60)
        self.assertEqual(result["total_tokens"], 600)
        self.assertEqual(result["total_llm_cost_usd"], 0.06)
        self.assertEqual(saved["mean_cycles_to_success"], 2.0)
        self.assertEqual(run_worker.call_count, 3)
        worker_dirs = [Path(call.args[0][-1]).parent.name for call in run_worker.call_args_list]
        self.assertEqual(worker_dirs, ["repetition_001", "repetition_002", "repetition_003"])

    @patch("maniskill_backend.generation_stability.run_worker")
    @patch("maniskill_backend.generation_stability.capture_experiment_environment")
    def test_environment_mismatch_blocks_all_generation(self, capture, run_worker) -> None:
        capture.return_value = {
            "validation": {"ok": False, "mismatches": [{"field": "llm.model"}]}
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = run_generation_stability(self._plan(), Path(tmp) / "stability")
        self.assertEqual(result["state"], "environment_contract_mismatch")
        self.assertEqual(result["completed_repetitions"], 0)
        run_worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
