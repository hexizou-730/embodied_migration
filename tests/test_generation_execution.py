"""Exercise generation orchestration without an API or simulator."""

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from maniskill_backend import module_generation_runner as runner
from maniskill_backend.cases import get_full_migration_case
from maniskill_backend.counterexample_loop import CEGISConfig, _module_generation_command


CASE_ID = "case02_pull_cube_panda_to_xarm6"
OLD = "def build_robot(env, *, control_mode, robot_uid):\n    return None\n"
NEW = "def build_robot(env, *, control_mode, robot_uid):\n    return env.agent\n"


class GenerationExecutionTest(unittest.TestCase):
    def _run(self, *, policy="full", guard_error=None, verification_ok=True,
             dry_run=False, initial_success=False, force=False):
        case = get_full_migration_case(CASE_ID)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            module = root / case.target_adapter_path
            module.parent.mkdir(parents=True)
            module.write_text(OLD, encoding="utf-8")
            stack.enter_context(patch.object(runner, "REPO_ROOT", root))
            stack.enter_context(patch.object(runner, "_git_diff", return_value=""))
            stack.enter_context(patch.object(runner, "build_module_generation_prompt", return_value="prompt"))
            source = stack.enter_context(patch.object(runner, "_run_source_trial", return_value={"success": True}))
            target = stack.enter_context(patch.object(runner, "_run_target_program_trial", side_effect=[
                {"success": initial_success}, {"success": True, "message": "verified"},
            ]))
            generate = stack.enter_context(patch.object(runner, "gen_text", return_value=SimpleNamespace(
                text=NEW, raw_text=NEW, used_llm=True, model="mock-model",
                reason="", usage={},
            )))
            guard = stack.enter_context(patch.object(runner, "guarded_adapter_validation_error", return_value=guard_error))
            verify = stack.enter_context(patch.object(runner, "_run_command", return_value={"ok": verification_ok}))
            result = runner.run_module_generation_migration(
                case_id=CASE_ID, max_attempts=1, dry_run=dry_run,
                attach_analysis=False, prompt_policy=policy, force_regeneration=force,
            )
            return result, module.read_text(encoding="utf-8"), {
                "source": source.call_count, "target": target.call_count,
                "generate": generate.call_count, "guard": guard.call_count,
                "verify": verify.call_count,
            }

    def test_changed_candidate_reaches_verification_and_target_evaluation(self):
        for policy, guards in (("full", 1), ("one_shot", 0)):
            with self.subTest(policy=policy):
                result, code, calls = self._run(policy=policy)
                self.assertTrue(result["success"], result["attempts"])
                self.assertTrue(result["attempts"][0]["module_kept"])
                self.assertEqual(code, NEW)
                self.assertEqual(calls, {"source": 1, "target": 2, "generate": 1, "guard": guards, "verify": 1})

    def test_rejected_guard_does_not_apply_or_evaluate_candidate(self):
        result, code, calls = self._run(guard_error="missing physical-state guard")
        self.assertFalse(result["success"])
        self.assertIn("missing physical-state guard", result["attempts"][0]["module_error"])
        self.assertFalse(result["attempts"][0]["module_applied"])
        self.assertEqual(code, OLD)
        self.assertEqual(calls["target"], 1)
        self.assertEqual(calls["verify"], 0)

    def test_failed_verification_restores_previous_adapter(self):
        result, code, calls = self._run(verification_ok=False)
        self.assertFalse(result["success"])
        self.assertFalse(result["attempts"][0]["module_kept"])
        self.assertEqual(code, OLD)
        self.assertEqual(calls["target"], 1)

    def test_dry_run_never_calls_llm_simulator_verifier_or_mutates_adapter(self):
        result, code, calls = self._run(dry_run=True)
        self.assertEqual(result["message"], "dry_run_planned")
        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], [])
        self.assertEqual(code, OLD)
        self.assertFalse(any(calls.values()), calls)

    def test_forced_generation_does_not_reuse_initial_success(self):
        result, code, calls = self._run(initial_success=True, force=True)
        self.assertTrue(result["success"])
        self.assertEqual(calls["generate"], 1)
        self.assertEqual(code, NEW)
        case = get_full_migration_case(CASE_ID)
        command = _module_generation_command(
            case, CEGISConfig(case_id=CASE_ID), Path("cycle"), seed=0,
            counterexample_path=None, probe_path=None, cycle=1,
        )
        self.assertIn("--force-regeneration", command)

    def test_rejected_generation_does_not_claim_old_success(self):
        result, code, calls = self._run(initial_success=True, force=True, guard_error="rejected")
        self.assertFalse(result["success"])
        self.assertEqual(code, OLD)
        self.assertEqual(calls["target"], 1)


if __name__ == "__main__":
    unittest.main()
