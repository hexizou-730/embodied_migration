from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maniskill_backend.source_programs import _sha256_file, load_source_program_catalog
from maniskill_backend.target_preparation import (
    frozen_source_specs,
    migrate_task,
    run_migrations,
)


SOURCE = '''
ENV_ID = "PushCube-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_joint_pos"

def run(env, seed=None):
    env.step([0.0])
'''


class TargetPreparationTests(unittest.TestCase):
    def test_selection_rejects_non_frozen_sources(self):
        _, specs = load_source_program_catalog()
        rows = [
            {"task_id": spec.task_id, "frozen": spec.task_id == specs[0].task_id}
            for spec in specs
        ]
        with patch(
            "maniskill_backend.target_preparation.source_program_status",
            return_value=rows,
        ):
            selected = frozen_source_specs(specs)
            self.assertEqual([spec.task_id for spec in selected], [specs[0].task_id])
            with self.assertRaisesRegex(ValueError, "only valid frozen"):
                frozen_source_specs(specs, specs[1].task_id)

    def test_worker_preflights_interface_then_migrates_immutable_source(self):
        _, originals = load_source_program_catalog()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "source.py"
            frozen.write_text(SOURCE, encoding="utf-8")
            record = frozen.with_suffix(".json")
            record.write_text(
                json.dumps(
                    {
                        "schema": "frozen_source_program.v2",
                        "source_sha256": _sha256_file(frozen),
                        "validated_seeds": list(range(10)),
                        "success_rate": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            spec = replace(
                originals[0],
                frozen_path=frozen,
                source_robot="panda",
                control_mode="pd_joint_pos",
            )
            plan = {
                "source_hashes": {spec.task_id: _sha256_file(frozen)},
                "target_robot": "xarm6_robotiq",
                "target_control_mode": "pd_ee_delta_pos",
                "seed": 0,
                "max_episode_steps": 1000,
                "max_cycles": 5,
                "obs_mode": "state",
                "sim_backend": "auto",
                "render_backend": "gpu",
                "download_assets": True,
                "asset_timeout": 5,
            }
            result_payload = {
                "success": True,
                "status": "success",
                "message": "official success",
                "target_adapter": str(root / "target.py"),
                "metrics": {
                    "generation_cycles": 2,
                    "repair_cycles": 1,
                    "target_action_steps": 42,
                    "total_tokens": 100,
                    "llm_cost_usd": 0.01,
                },
            }
            observation = {
                "action_space": {"shape": [4]},
                "controller_action_mapping": None,
                "tcp": [0.0, 0.0, 0.2],
                "entity_aliases": {"cube": "obj"},
                "supported_robots": ["panda"],
            }
            task_catalog = {
                "tasks": [
                    {
                        "task_id": spec.task_id,
                        "required_capabilities": ["cartesian_translation"],
                    }
                ]
            }
            completed = SimpleNamespace(returncode=0)
            with patch(
                "maniskill_backend.target_preparation.load_source_program_catalog",
                return_value=({}, [spec]),
            ), patch(
                "maniskill_backend.target_preparation.source_program_status",
                return_value=[{"task_id": spec.task_id, "frozen": True}],
            ), patch(
                "maniskill_backend.target_preparation.subprocess.run", return_value=completed
            ) as assets, patch(
                "maniskill_backend.target_preparation.discover_environment",
                return_value=observation,
            ), patch(
                "maniskill_backend.target_preparation.load_task_catalog",
                return_value=task_catalog,
            ), patch(
                "maniskill_backend.target_preparation.run_dynamic_agent_migration",
                return_value=result_payload,
            ) as migrate:
                result = migrate_task(
                    {"task_id": spec.task_id, "plan": plan, "output_dir": str(root / "out")}
                )
            self.assertTrue(result["success"])
            self.assertEqual(result["metrics"]["repair_cycles"], 1)
            self.assertEqual(assets.call_count, 2)
            self.assertEqual(
                migrate.call_args.kwargs["source_provenance"]["kind"],
                "frozen_source_program.v2",
            )
            interface = json.loads((root / "out" / "target_interface.json").read_text())
            self.assertEqual(interface["action_space"]["shape"], [4])

    def test_interface_failure_stops_before_llm_migration(self):
        _, originals = load_source_program_catalog()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "source.py"
            frozen.write_text(SOURCE, encoding="utf-8")
            frozen.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "source_sha256": _sha256_file(frozen),
                        "validated_seeds": list(range(10)),
                        "success_rate": 1.0,
                    }
                )
            )
            spec = replace(originals[0], frozen_path=frozen)
            plan = {
                "source_hashes": {spec.task_id: _sha256_file(frozen)},
                "target_robot": "missing_robot",
                "target_control_mode": "pd_ee_delta_pos",
                "seed": 0,
                "max_episode_steps": 1000,
                "max_cycles": 2,
                "obs_mode": "state",
                "sim_backend": "auto",
                "render_backend": "gpu",
                "download_assets": False,
                "asset_timeout": 5,
            }
            with patch(
                "maniskill_backend.target_preparation.load_source_program_catalog",
                return_value=({}, [spec]),
            ), patch(
                "maniskill_backend.target_preparation.source_program_status",
                return_value=[{"task_id": spec.task_id, "frozen": True}],
            ), patch(
                "maniskill_backend.target_preparation.subprocess.run",
                return_value=SimpleNamespace(returncode=0),
            ), patch(
                "maniskill_backend.target_preparation.discover_environment",
                side_effect=RuntimeError("robot cannot initialize"),
            ), patch(
                "maniskill_backend.target_preparation.run_dynamic_agent_migration"
            ) as migrate:
                result = migrate_task(
                    {"task_id": spec.task_id, "plan": plan, "output_dir": str(root / "out")}
                )
            self.assertEqual(result["status"], "target_interface_failed")
            migrate.assert_not_called()

    def test_batch_checkpoints_success_and_failure(self):
        _, originals = load_source_program_catalog()
        specs = originals[:2]
        plan = {
            "task_ids": [spec.task_id for spec in specs],
            "target_robot": "xarm6_robotiq",
            "seed": 0,
            "task_timeout": 5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            def worker(command, **kwargs):
                config = json.loads(Path(command[-1]).read_text())
                success = config["task_id"] == specs[0].task_id
                result = {
                    "task_id": config["task_id"],
                    "success": success,
                    "status": "success" if success else "target_budget_exhausted",
                    "metrics": {"generation_cycles": 2, "target_action_steps": 10},
                }
                Path(config["output_dir"], "worker_result.json").write_text(json.dumps(result))
                return 0, False

            with patch(
                "maniskill_backend.target_preparation.load_source_program_catalog",
                return_value=({}, specs),
            ), patch(
                "maniskill_backend.target_preparation.capture_experiment_environment",
                return_value={"validation": {"ok": True}},
            ), patch(
                "maniskill_backend.target_preparation.run_worker", side_effect=worker
            ):
                result = run_migrations(plan, output)
                self.assertEqual(result["succeeded"], 1)
                self.assertEqual(result["state"], "complete")
                self.assertTrue((output / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
