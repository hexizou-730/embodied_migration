from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from maniskill_backend.source_preparation import (
    ensure_assets, preparation_options, prepare_task, run_preparation, run_worker, use_contract_llm,
)
from maniskill_backend.source_programs import (
    REPO_ROOT, _sha256_file, _support_code_sha256, freeze_source_program,
    load_source_program_catalog,
)


class SourcePreparationTests(unittest.TestCase):
    def test_queue_isolates_failures_checkpoints_and_resumes(self):
        catalog, originals = load_source_program_catalog()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = []
            for original in (originals[2], originals[0]):
                candidate = root / (original.task_id + ".py")
                candidate.write_text(original.candidate_path.read_text())
                specs.append(replace(original, candidate_path=candidate,
                                     frozen_path=root / "frozen" / candidate.name))
            args = SimpleNamespace(max_cycles=2, task_timeout=5, asset_timeout=1,
                no_download=False, obs_mode="state", sim_backend="auto", render_backend="gpu")
            plan = preparation_options(args, specs, list(range(10)))
            output = root / "run"
            calls = []

            def worker(command, **kwargs):
                config = json.loads(Path(command[-1]).read_text())
                task = config["task_id"]
                calls.append(task)
                progress = json.loads((output / "summary.json").read_text())
                self.assertEqual(progress["current_task"], task)
                if task == originals[2].task_id:
                    return -9, True
                spec = next(spec for spec in specs if spec.task_id == task)
                evidence = {
                    "success": True, "task": {"task_id": task}, "environment_valid": True,
                    "environment_contract_sha256": _sha256_file(REPO_ROOT / "experiment_config.json"),
                    "candidate": {"sha256": _sha256_file(spec.candidate_path)},
                    "support_code_sha256": _support_code_sha256(spec),
                    "trials": [{"seed": i, "success": True, "official_success": True, "action_steps": 20}
                               for i in range(10)],
                }
                freeze_source_program(spec, evidence, minimum_seed_count=2)
                (Path(config["output_dir"]) / "worker_result.json").write_text(json.dumps(
                    {"task_id": task, "success": True, "status": "frozen"}))
                return 0, False

            with patch("maniskill_backend.source_preparation.load_source_program_catalog", return_value=(catalog, specs)), patch(
                "maniskill_backend.source_preparation.capture_experiment_environment",
                return_value={"validation": {"ok": True}},
            ), patch("maniskill_backend.source_preparation.run_worker", side_effect=worker):
                result = run_preparation(plan, output)
                self.assertEqual(calls, [originals[0].task_id, originals[2].task_id])
                self.assertEqual(result["frozen"], 1)
                self.assertEqual(result["state"], "incomplete")
                self.assertEqual(result["results"][1]["status"], "worker_timeout")
                run_preparation(plan, output)
                self.assertEqual(len(calls), 2)
                run_preparation(plan, output, retry_failed=True)
                self.assertEqual(len(calls), 3)

    def test_environment_mismatch_stops_before_any_worker(self):
        _, specs = load_source_program_catalog()
        plan = {"task_ids": [specs[0].task_id], "seeds": list(range(10))}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_preparation.capture_experiment_environment",
            return_value={"validation": {"ok": False, "mismatches": [{"field": "python.version"}]}},
        ), patch("maniskill_backend.source_preparation.run_worker") as worker:
            result = run_preparation(plan, Path(tmp))
            self.assertTrue((Path(tmp) / "summary.json").exists())
        self.assertEqual(result["state"], "environment_contract_mismatch")
        worker.assert_not_called()

    def test_plan_change_requires_new_run_name(self):
        _, specs = load_source_program_catalog()
        plan = {"task_ids": [specs[0].task_id], "seeds": list(range(10))}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "plan.json").write_text(json.dumps(dict(plan, seeds=[0, 1])))
            with self.assertRaisesRegex(ValueError, "new --run-name"):
                run_preparation(plan, Path(tmp))

    def test_assets_failure_does_not_spend_llm_calls(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_preparation.subprocess.run", return_value=SimpleNamespace(returncode=1)
        ), patch("maniskill_backend.source_preparation.validate_source_program") as validate, patch(
            "maniskill_backend.source_preparation.synthesize_source_program"
        ) as synthesize:
            result = prepare_task({"task_id": "t03_roll_ball", "output_dir": tmp,
                                   "plan": {"download_assets": True, "asset_timeout": 1}})
        self.assertEqual(result["status"], "assets_blocked")
        validate.assert_not_called()
        synthesize.assert_not_called()

    def test_assets_timeout_is_not_a_task_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_preparation.subprocess.run",
            side_effect=subprocess.TimeoutExpired("assets", 1),
        ):
            result = prepare_task({"task_id": "t03_roll_ball", "output_dir": tmp,
                                   "plan": {"download_assets": True, "asset_timeout": 1}})
        self.assertEqual(result["status"], "assets_timeout")

    def test_worker_timeout_stops_real_subprocess(self):
        with tempfile.TemporaryFile(mode="w+") as log:
            code, timed_out = run_worker([sys.executable, "-c", "import time; time.sleep(30)"],
                                         log=log, timeout=0.1)
        self.assertTrue(timed_out)
        self.assertLess(code, 0)

    def test_pinned_llm_settings_do_not_overwrite_api_key(self):
        import os
        with patch.dict(os.environ, {"EM_MODEL": "old-model", "OPENROUTER_API_KEY": "test-secret"}):
            use_contract_llm()
            self.assertEqual(os.environ["EM_MODEL"], "openai/gpt-5.6-luna")
            self.assertEqual(os.environ["EM_MAX_TOKENS"], "16384")
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "test-secret")

    def test_assets_download_stages_before_install_and_recovers_empty_directory(self):
        @dataclass
        class Asset:
            output_dir: Path
            target_path: str

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = Asset(root, "assets/toy")
            destination = root / asset.target_path
            destination.mkdir(parents=True)
            assets = SimpleNamespace(DATA_GROUPS={}, DATA_SOURCES={"toy": asset},
                is_data_source_downloaded=lambda uid: destination.exists())
            modules = {name: ModuleType(name) for name in (
                "mani_skill", "mani_skill.envs", "mani_skill.agents", "mani_skill.agents.registration",
                "mani_skill.utils", "mani_skill.utils.registration",
            )}
            modules["mani_skill.utils"].assets = assets
            modules["mani_skill.utils.registration"].REGISTERED_ENVS = {
                "PushCube-v1": SimpleNamespace(asset_download_ids=["toy"])}
            modules["mani_skill.agents.registration"].REGISTERED_AGENTS = {
                "panda": SimpleNamespace(asset_download_ids=[])}
            def download(source, **kwargs):
                self.assertNotEqual(source.output_dir, root)
                self.assertEqual(list(destination.iterdir()), [])
                staged = source.output_dir / source.target_path
                staged.mkdir(parents=True)
                (staged / "mesh.urdf").write_text("test fixture")
            modules["mani_skill.utils"].download_asset = SimpleNamespace(download=download)
            with patch.dict(sys.modules, modules):
                ensure_assets("t01_push_cube", download=True)
            self.assertEqual((destination / "mesh.urdf").read_text(), "test fixture")
            self.assertEqual(list(root.glob(".source-download-*")), [])


if __name__ == "__main__":
    unittest.main()
