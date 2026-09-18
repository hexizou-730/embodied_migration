from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from maniskill_backend.llm import LLMTextResult
from maniskill_backend.source_programs import (
    REPO_ROOT,
    _multi_seed_feedback,
    _repair_prompt,
    _sha256_file,
    _support_code_sha256,
    _validate_source_repair,
    freeze_source_program,
    load_source_program_catalog,
    source_program_status,
    statically_check_candidate,
    synthesize_source_program,
    validate_source_program,
)
from maniskill_backend.dynamic_harness import DynamicMigrationSpec, _prompt_trial


class SourceProgramTests(unittest.TestCase):
    def test_catalog_covers_twenty_tasks_with_requested_source_split(self) -> None:
        _, specs = load_source_program_catalog()
        self.assertEqual(len(specs), 20)
        by_id = {spec.task_id: spec for spec in specs}
        self.assertEqual(by_id["t12_open_cabinet_drawer"].source_robot, "fetch")
        self.assertEqual(by_id["t13_open_cabinet_door"].source_robot, "fetch")
        for task_id, spec in by_id.items():
            if task_id not in {"t12_open_cabinet_drawer", "t13_open_cabinet_door"}:
                self.assertTrue(spec.source_robot.startswith("panda"), task_id)

    def test_reference_candidates_are_statically_consistent(self) -> None:
        _, specs = load_source_program_catalog()
        ready = [spec for spec in specs if spec.origin.startswith("maniskill_3.0.1")]
        self.assertEqual(len(ready), 11)
        for spec in ready:
            result = statically_check_candidate(spec)
            self.assertTrue(result["ok"], (spec.task_id, result))

    def test_all_twenty_candidates_exist_but_none_are_predeclared_frozen(self) -> None:
        _, specs = load_source_program_catalog()
        with tempfile.TemporaryDirectory() as tmp:
            unfrozen = [replace(spec, frozen_path=Path(tmp) / f"{spec.task_id}.py") for spec in specs]
            rows = {row["task_id"]: row for row in source_program_status(unfrozen)}
        self.assertEqual(len(rows), 20)
        for row in rows.values():
            self.assertEqual(row["candidate_status"], "candidate_ready")
            self.assertFalse(row["frozen"])

    def test_freeze_rejects_partial_multi_seed_success(self) -> None:
        _, specs = load_source_program_catalog()
        spec = specs[0]
        validation = {
            "success": False,
            "trials": [
                {"seed": 0, "success": True},
                {"seed": 1, "success": False},
            ],
        }
        with self.assertRaisesRegex(ValueError, "cannot be frozen"):
            freeze_source_program(spec, validation, minimum_seed_count=2)

    def test_freeze_writes_code_and_checksum_record_after_full_pass(self) -> None:
        _, specs = load_source_program_catalog()
        original = specs[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.py"
            candidate.write_text(original.candidate_path.read_text(), encoding="utf-8")
            spec = replace(
                original,
                candidate_path=candidate,
                frozen_path=root / "frozen" / "task.py",
            )
            validation = complete_evidence(spec)
            record = freeze_source_program(spec, validation, minimum_seed_count=2)
            saved = json.loads(spec.frozen_path.with_suffix(".json").read_text())
        self.assertEqual(record["source_sha256"], saved["source_sha256"])
        self.assertEqual(saved["validated_seeds"], list(range(10)))
        self.assertEqual(saved["success_rate"], 1.0)

    def test_freeze_rejects_smoke_duplicates_claims_and_changed_code(self):
        _, specs = load_source_program_catalog()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.py"
            candidate.write_text(specs[0].candidate_path.read_text())
            spec = replace(specs[0], candidate_path=candidate, frozen_path=Path(tmp) / "frozen.py")
            for kind in ("smoke", "duplicates", "unofficial", "no_actions", "stale", "contract"):
                evidence = complete_evidence(spec)
                if kind == "smoke":
                    evidence["trials"] = evidence["trials"][:2]
                elif kind == "duplicates":
                    evidence["trials"][9]["seed"] = 0
                elif kind == "unofficial":
                    evidence["trials"][0]["official_success"] = False
                elif kind == "no_actions":
                    evidence["trials"][0]["action_steps"] = 0
                elif kind == "stale":
                    evidence["candidate"]["sha256"] = "stale"
                else:
                    evidence["environment_valid"] = False
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    freeze_source_program(spec, evidence, minimum_seed_count=2)
                self.assertFalse(spec.frozen_path.exists())

    def test_feedback_loads_real_trace_and_preserves_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_003.json"
            path.write_text(json.dumps({"success": False, "action_steps": 25,
                "state_trace": [{"step": 0, "tcp": {"p": [1, 2, 3]}},
                                {"step": 25, "tcp": {"p": [4, 5, 6]}}],
                "execution_error": "contact did not progress"}))
            validation = {"success": False, "status": "validation_failed",
                          "trials": [{"seed": 3, "success": False, "trial_path": str(path)}]}
            feedback = _prompt_trial(_multi_seed_feedback(validation))
        sample = feedback["multi_seed_failure_sample"][0]
        self.assertEqual(sample["seed"], 3)
        self.assertEqual(sample["state_trace_sample"][-1]["tcp"]["p"], [4, 5, 6])
        self.assertEqual(sample["execution_error"], "contact did not progress")
        self.assertFalse(_multi_seed_feedback({"status": "candidate_invalid"})["success"])

    def test_native_joint_source_repair_keeps_planner_contract(self):
        code = '''
from source_programs.vendor.mani_skill_motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
def run(env, seed=None, debug=False, vis=False):
    planner = PandaArmMotionPlanningSolver(env, base_pose=env.unwrapped.agent.robot.pose)
    pose = env.unwrapped.agent.tcp.pose.sp
    return planner.move_to_pose_with_screw(pose)
'''
        _validate_source_repair(code, native_run=True)
        for mutation in ("env.reset()", "env.unwrapped.obj.set_pose(pose)", "env.unwrapped.success = True"):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _validate_source_repair(code + "\n    " + mutation, native_run=True)
        _, specs = load_source_program_catalog()
        base = DynamicMigrationSpec(env_id=specs[0].env_id, task_label=specs[0].task_id,
                                    source_robot="panda", target_robot="panda")
        system, prompt = _repair_prompt(specs[0], base, {}, code, {})
        self.assertIn("pd_joint_pos", system)
        self.assertIn("NOT normalized xyz", prompt)
        self.assertIn("def move_to_pose_with_RRTConnect", prompt)

    def test_validation_saves_trial_links_and_does_not_repair_infrastructure(self):
        _, specs = load_source_program_catalog()
        runtime = {"contract_sha256": "a" * 64, "validation": {"ok": True}}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_programs.capture_experiment_environment", return_value=runtime
        ), patch("maniskill_backend.source_programs.run_dynamic_trial", return_value={
            "success": False, "action_steps": 0, "message": "URLError: download failed"
        }) as run:
            result = validate_source_program(specs[0], seeds=[0, 1], output_dir=Path(tmp),
                obs_mode="state", sim_backend="auto", render_backend="gpu", enforce_environment=True)
            self.assertTrue(Path(result["trials"][0]["trial_path"]).exists())
        self.assertEqual(result["status"], "infrastructure_failure")
        self.assertEqual(run.call_count, 1)

    def test_synthesis_smoke_then_full_validation_preserves_llm_provenance(self):
        _, specs = load_source_program_catalog()
        code = '''
from source_programs.vendor.mani_skill_motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
ENV_ID = "PushCube-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_joint_pos"
def run(env, seed=None, debug=False, vis=False):
    planner = PandaArmMotionPlanningSolver(env, base_pose=env.unwrapped.agent.robot.pose)
    return planner.move_to_pose_with_screw(env.unwrapped.agent.tcp.pose.sp)
'''
        generated = LLMTextResult(code, True, "test-model", raw_text=code)
        malformed = LLMTextResult("def run(: pass", True, "test-model", raw_text="def run(: pass")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "candidate.py"
            path.write_text(specs[0].candidate_path.read_text())
            spec = replace(specs[0], candidate_path=path, frozen_path=root / "frozen.py")
            runtime = {"contract_sha256": _sha256_file(REPO_ROOT / "experiment_config.json"),
                       "validation": {"ok": True}}
            evaluated = []
            def validate(generated_spec, **kwargs):
                evaluated.append(kwargs["seeds"])
                evidence = complete_evidence(generated_spec)
                evidence["trials"] = [row for row in evidence["trials"] if row["seed"] in kwargs["seeds"]]
                evidence["seeds"] = kwargs["seeds"]
                return evidence
            with patch("maniskill_backend.source_programs.capture_experiment_environment", return_value=runtime), patch(
                "maniskill_backend.source_programs.discover_environment", return_value={}
            ), patch("maniskill_backend.source_programs.gen_text", side_effect=[malformed, generated]) as generate, patch(
                "maniskill_backend.source_programs.validate_source_program", side_effect=validate
            ):
                result = synthesize_source_program(spec, seeds=list(range(10)), output_dir=root / "out",
                    max_cycles=2, obs_mode="state", sim_backend="auto", render_backend="gpu")
                resumed = synthesize_source_program(spec, seeds=list(range(10)), output_dir=root / "out",
                    max_cycles=2, obs_mode="state", sim_backend="auto", render_backend="gpu")
            self.assertEqual(evaluated, [[0, 1], list(range(10))])
            self.assertEqual(generate.call_count, 2)
            self.assertTrue(result["success"])
            self.assertTrue(resumed["success"])
            record = freeze_source_program(spec, result["validation"], minimum_seed_count=2)
            self.assertEqual(record["provenance"]["kind"], "llm_repair")
            self.assertEqual(record["provenance"]["model"], "test-model")
            provenance = json.loads(path.with_suffix(".provenance.json").read_text())
            self.assertEqual(provenance["candidate_sha256"], _sha256_file(path))

    def test_explicit_retry_archives_unavailable_api_attempt(self):
        _, specs = load_source_program_catalog()
        runtime = {"contract_sha256": "a" * 64, "validation": {"ok": True}}
        unavailable = LLMTextResult("", False, "test-model", reason="missing_api_key")
        invalid = LLMTextResult("x = 1", True, "test-model", raw_text="x = 1")
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_programs.capture_experiment_environment", return_value=runtime
        ), patch("maniskill_backend.source_programs.discover_environment", return_value={}), patch(
            "maniskill_backend.source_programs.gen_text", side_effect=[unavailable, invalid]
        ) as generate:
            kwargs = dict(seeds=[0, 1], output_dir=Path(tmp), max_cycles=1,
                          obs_mode="state", sim_backend="auto", render_backend="gpu")
            first = synthesize_source_program(specs[2], **kwargs)
            resumed = synthesize_source_program(specs[2], **kwargs)
            self.assertEqual(first["status"], "llm_unavailable")
            self.assertEqual(resumed["status"], "llm_unavailable")
            self.assertEqual(generate.call_count, 1)
            synthesize_source_program(specs[2], retry_unavailable=True, **kwargs)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(len(list((Path(tmp) / specs[2].task_id).glob("synthesis_cycle_01_unavailable_*"))), 1)

    def test_synthesis_contract_mismatch_stops_before_simulator_and_llm(self) -> None:
        _, specs = load_source_program_catalog()
        spec = specs[0]
        runtime = {
            "contract_sha256": "a" * 64,
            "validation": {
                "ok": False,
                "mismatches": [
                    {"field": "python.version", "expected": "3.10.20", "observed": "3.12.2"}
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "maniskill_backend.source_programs.capture_experiment_environment",
            return_value=runtime,
        ), patch(
            "maniskill_backend.source_programs.discover_environment"
        ) as discover, patch(
            "maniskill_backend.source_programs.gen_text"
        ) as generate:
            result = synthesize_source_program(
                spec,
                seeds=[0, 1],
                output_dir=Path(tmp),
                max_cycles=2,
                obs_mode="state",
                sim_backend="auto",
                render_backend="gpu",
            )
        self.assertEqual(result["status"], "environment_contract_mismatch")
        discover.assert_not_called()
        generate.assert_not_called()

    def test_invalid_generation_is_shown_to_next_repair_cycle(self) -> None:
        _, specs = load_source_program_catalog()
        invalid = '''
import numpy as np
from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot
ENV_ID = "RollBall-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"
TASK_PROGRAM = "ret_val = robot.solve_task()"
# FIRST_INVALID_MARKER
class GeneratedRobot(ManiSkillDynamicRobot):
    def solve_task(self):
        for _ in range(2):
            if self._early_stop():
                return False
            self._step(self._make_action(np.zeros(3), gripper=0.0))
        return False
def build_robot(env, *, control_mode, robot_uid):
    return GeneratedRobot(env, control_mode=control_mode, robot_uid=robot_uid)
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text(invalid.replace("# FIRST_INVALID_MARKER\n", ""), encoding="utf-8")
            spec = replace(specs[2], candidate_path=bootstrap, frozen_path=root / "frozen.py")
            runtime = {
                "contract_sha256": "a" * 64,
                "validation": {"ok": True, "mismatches": []},
            }
            generated = LLMTextResult(invalid, True, "test-model", raw_text=invalid)
            with patch(
                "maniskill_backend.source_programs.capture_experiment_environment",
                return_value=runtime,
            ), patch(
                "maniskill_backend.source_programs.discover_environment",
                return_value={"robot_uid": "panda", "action_space": {"shape": [4]}},
            ), patch(
                "maniskill_backend.source_programs.gen_text",
                side_effect=[generated, generated],
            ) as generate:
                result = synthesize_source_program(
                    spec,
                    seeds=[0, 1],
                    output_dir=root / "out",
                    max_cycles=2,
                    obs_mode="state",
                    sim_backend="auto",
                    render_backend="gpu",
                )
                resumed = synthesize_source_program(
                    spec, seeds=[0, 1], output_dir=root / "out", max_cycles=2,
                    obs_mode="state", sim_backend="auto", render_backend="gpu",
                )
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(len(resumed["cycles"]), 2)

        second_prompt = generate.call_args_list[1].kwargs["prompt"]
        self.assertFalse(result["success"])
        self.assertIn("FIRST_INVALID_MARKER", second_prompt)
        self.assertIn("self._snapshot()", second_prompt)
        self.assertIn("Generated adapter must read physical state", second_prompt)


def complete_evidence(spec):
    return {
        "success": True, "success_rate": 1.0, "task": {"task_id": spec.task_id},
        "seeds": list(range(10)), "environment_valid": True,
        "environment_contract_sha256": _sha256_file(REPO_ROOT / "experiment_config.json"),
        "support_code_sha256": _support_code_sha256(spec),
        "candidate": {"sha256": _sha256_file(spec.candidate_path)},
        "trials": [{"seed": seed, "success": True, "official_success": True, "action_steps": 20}
                   for seed in range(10)],
    }


if __name__ == "__main__":
    unittest.main()
