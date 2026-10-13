from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from maniskill_backend.adapter_evaluation import evaluate_fixed_adapter


class AdapterEvaluationTests(unittest.TestCase):
    def _evidence(self, root: Path) -> Path:
        evidence = root / "evidence"
        evidence.mkdir()
        adapter = evidence / "target_adapter.py"
        adapter.write_text("# immutable generated adapter\n", encoding="utf-8")
        digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        (evidence / "generation_cycle.json").write_text(
            json.dumps({"adapter_sha256": digest}), encoding="utf-8"
        )
        (evidence / "replay_result.json").write_text(
            json.dumps(
                {
                    "schema": "generated_adapter_replay.v1",
                    "source_generation_cycle": 8,
                    "program": "ret_val = robot.run_source_actions()",
                    "spec": {
                        "env_id": "PushCube-v1",
                        "task_label": "t01_push_cube",
                        "source_robot": "panda",
                        "target_robot": "xarm6_robotiq",
                        "source_control_mode": "pd_joint_pos",
                        "target_control_mode": "pd_ee_delta_pos",
                        "seed": 0,
                        "max_episode_steps": 1000,
                        "max_source_cycles": 0,
                        "max_target_cycles": 8,
                        "obs_mode": "state",
                        "sim_backend": "auto",
                        "render_backend": "gpu",
                    },
                }
            ),
            encoding="utf-8",
        )
        return evidence

    @patch("maniskill_backend.adapter_evaluation.run_dynamic_trial")
    @patch("maniskill_backend.adapter_evaluation.capture_experiment_environment")
    def test_fixed_adapter_runs_all_seeds_without_repair(self, capture, run_trial) -> None:
        capture.return_value = {"validation": {"ok": True, "mismatches": []}}
        run_trial.side_effect = [
            {
                "success": True,
                "official_success": True,
                "action_steps": 20,
                "message": "success",
            },
            {
                "success": False,
                "official_success": False,
                "action_steps": 12,
                "message": "no contact",
                "failure_diagnosis": {
                    "layer": "contact_geometry",
                    "reason": "contact_not_established",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = self._evidence(root)
            result = evaluate_fixed_adapter(
                evidence_dir=evidence,
                output_dir=root / "run",
                seeds=(100, 101),
            )
            summary = json.loads((root / "run/summary.json").read_text())

        self.assertTrue(result["complete"])
        self.assertEqual(result["num_success"], 1)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(result["code_repairs"], 0)
        self.assertEqual(summary["failure_reasons"], {"contact_not_established": 1})
        self.assertEqual(summary["results"][0]["failure_layer"], "success")
        self.assertEqual(summary["results"][0]["failure_reason"], "success")
        self.assertEqual([call.kwargs["spec"].seed for call in run_trial.call_args_list], [100, 101])
        self.assertEqual(
            len({str(call.kwargs["adapter_path"]) for call in run_trial.call_args_list}),
            1,
        )

    @patch("maniskill_backend.adapter_evaluation.run_dynamic_trial")
    @patch("maniskill_backend.adapter_evaluation.capture_experiment_environment")
    def test_environment_mismatch_stops_before_simulation(self, capture, run_trial) -> None:
        capture.return_value = {
            "validation": {
                "ok": False,
                "mismatches": [{"field": "python.version"}],
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evaluate_fixed_adapter(
                evidence_dir=self._evidence(root),
                output_dir=root / "run",
                seeds=(100,),
            )
        self.assertEqual(result["state"], "environment_contract_mismatch")
        run_trial.assert_not_called()

    def test_adapter_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = self._evidence(root)
            (evidence / "target_adapter.py").write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                evaluate_fixed_adapter(
                    evidence_dir=evidence,
                    output_dir=root / "run",
                    seeds=(100,),
                )


if __name__ == "__main__":
    unittest.main()
