from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from maniskill_backend.acceptance import verify_acceptance


class AcceptanceTests(unittest.TestCase):
    def test_acceptance_requires_generation_actions_and_official_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            migration = run / "migration"
            artifacts = migration / "dynamic_artifacts"
            artifacts.mkdir(parents=True)
            adapter = artifacts / "target_adapter.py"
            adapter.write_text("# generated adapter\n")
            manifest = migration / "manifest.json"
            manifest.write_text("{}\n")
            details = migration / "dynamic_summary.json"
            details.write_text(json.dumps({
                "status": "success",
                "success": True,
                "target_adapter": str(adapter),
                "manifest": str(manifest),
                "artifact_sha256": {"target_adapter": "abc"},
                "target_result": {"official_success": True, "action_steps": 12},
            }))
            (run / "runtime_environment.json").write_text(
                json.dumps({"validation": {"ok": True}})
            )
            (run / "summary.json").write_text(json.dumps({
                "schema": "target_migration_batch.v1",
                "state": "complete",
                "complete": True,
                "results": [{
                    "task_id": "t01_push_cube",
                    "status": "success",
                    "success": True,
                    "details": str(details),
                    "target_adapter": str(adapter),
                    "metrics": {"generation_cycles": 2, "target_action_steps": 12},
                }],
            }))

            report = verify_acceptance(
                run, task_id="t01_push_cube", target_robot="xarm6_robotiq"
            )
            self.assertTrue(report["passed"])
            self.assertTrue((run / "acceptance_report.md").is_file())

            summary = json.loads((run / "summary.json").read_text())
            summary["results"][0]["metrics"]["target_action_steps"] = 0
            (run / "summary.json").write_text(json.dumps(summary))
            report = verify_acceptance(
                run, task_id="t01_push_cube", target_robot="xarm6_robotiq"
            )
            self.assertFalse(report["passed"])
            self.assertIn("target_actions_executed", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
