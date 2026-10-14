from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maniskill_backend.method_benchmark import run_method_benchmark


class MethodBenchmarkTests(unittest.TestCase):
    def test_runs_each_method_and_aggregates_results(self) -> None:
        plan = {
            "task_id": "t01_push_cube",
            "source_robot": "panda",
            "target_robot": "xarm6_robotiq",
            "seed": 0,
            "methods": ["one_shot", "ours"],
            "repetitions": 2,
            "max_cycles": 3,
            "task_timeout": 5,
        }

        def worker(command, **kwargs):
            config_path = Path(command[-1])
            config = json.loads(config_path.read_text(encoding="utf-8"))
            method_id = config["plan"]["method_id"]
            success = method_id == "ours"
            output = Path(config["output_dir"])
            result = {
                "success": success,
                "status": "success" if success else "target_budget_exhausted",
                "metrics": {
                    "generation_cycles": 2 if success else 1,
                    "repair_cycles": 1 if success else 0,
                    "target_action_steps": 20 if success else 5,
                    "total_tokens": 100 if success else 40,
                    "llm_cost_usd": 0.02 if success else 0.01,
                },
            }
            (output / "worker_result.json").write_text(json.dumps(result), encoding="utf-8")
            return 0, False

        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.method_benchmark.capture_experiment_environment",
            return_value={"validation": {"ok": True}},
        ), patch(
            "maniskill_backend.method_benchmark.run_worker", side_effect=worker
        ):
            output = Path(tmp) / "benchmark"
            result = run_method_benchmark(plan, output)

        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["completed_runs"], 4)
        self.assertEqual(result["method_summary"]["one_shot"]["successes"], 0)
        self.assertEqual(result["method_summary"]["ours"]["successes"], 2)
        self.assertEqual(result["method_summary"]["ours"]["total_tokens"], 200)

    def test_rejects_unknown_method(self) -> None:
        plan = {
            "task_id": "t01_push_cube",
            "source_robot": "panda",
            "target_robot": "xarm6_robotiq",
            "seed": 0,
            "methods": ["unknown"],
            "repetitions": 1,
            "max_cycles": 1,
            "task_timeout": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown benchmark methods"):
                run_method_benchmark(plan, Path(tmp) / "benchmark")


if __name__ == "__main__":
    unittest.main()
