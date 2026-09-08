import csv
import hashlib
import json
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paper import (
    _apply_frozen_llm_environment,
    _build_motivation_evidence_plan,
    _load_completed_summary,
    _paper_status,
    _run_plan,
    _smoke_nested_preview_audit,
    _start_frozen_remote_plan,
    _stream_benchmark_command,
    _summarize,
    _validate_and_rebind_frozen_plan,
    _wilson_interval,
    _write_summary_csv,
    _write_summary_latex,
)
from maniskill_backend.paper_preflight import run_paper_preflight, write_paper_preflight
from maniskill_backend.paper_package import (
    build_evidence_export,
    build_remote_package,
    verify_remote_package_manifest,
)
from maniskill_backend.paper_benchmark import PAPER_METHODS, build_paper_plan
from maniskill_backend.counterexample_loop import CEGISConfig, run_counterexample_loop
from maniskill_backend.paper_verify import (
    _cegis_protocol_errors,
    _llm_budget_errors,
    _llm_config_errors,
    _probe_budget_errors,
    verify_paper_plan,
)
from maniskill_backend.llm_preflight import run_llm_preflight
from maniskill_backend.protocol_audit import run_protocol_audit
from maniskill_backend.source_baseline import run_source_baseline_gate
from maniskill_backend.diagnosis_eval import build_annotation_template, evaluate_annotations
from maniskill_backend.paper_statistics import paired_method_comparisons


class PaperCliTest(unittest.TestCase):
    def test_smoke_nested_preview_audit_checks_controlled_probe_selector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan = build_paper_plan(
                tier_id="pilot",
                method_ids=["B5", "Ours"],
                case_ids=["case02_pull_cube_panda_to_xarm6"],
                output_root=str(root),
                plan_name="nested",
                dry_run_commands=True,
            )
            plan_dir = root / "nested"
            for run in plan["runs"]:
                strategy = PAPER_METHODS[run["method_id"]].probe_strategy
                preview = run_counterexample_loop(
                    CEGISConfig(
                        case_id=run["case_id"],
                        development_seeds="0-4",
                        held_out_seeds="100-104",
                        max_episode_steps=500,
                        probe_strategy=strategy,
                        dry_run=True,
                    ),
                    output_root=root / "previews",
                    run_name=run["method_id"],
                )
                summary = {
                    "details": {
                        "cegis_summary": preview,
                        "cegis_run": {
                            "command": [
                                "python", "cegis.py", "--case", run["case_id"],
                                "--development-seeds", "0-4", "--held-out-seeds", "100-104",
                                "--max-cycles", "3", "--attempts-per-cycle", "1",
                                "--probe-budget", "8", "--probe-strategy", strategy,
                                "--prompt-policy", "full", "--success-threshold", "0.8",
                                "--min-trials-for-accept", "5", "--obs-mode", "state",
                                "--sim-backend", "auto", "--render-backend", "gpu",
                                "--max-episode-steps", "500",
                            ]
                        },
                    },
                }
                summary_path = plan_dir / "runs" / run["run_id"] / "summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary), encoding="utf-8")

            audit = _smoke_nested_preview_audit(plan, plan_dir)
            self.assertTrue(audit["valid"])
            self.assertEqual(audit["num_valid"], 2)
            self.assertEqual(audit["num_controlled_pairs_valid"], 1)

            subset = {**plan, "runs": [r for r in plan["runs"] if r["method_id"] == "Ours"]}
            single_audit = _smoke_nested_preview_audit(subset, plan_dir)
            self.assertTrue(single_audit["valid"])
            self.assertEqual(single_audit["num_controlled_pairs_expected"], 0)

            ours = next(row for row in plan["runs"] if row["method_id"] == "Ours")
            ours_path = plan_dir / "runs" / ours["run_id"] / "summary.json"
            broken = json.loads(ours_path.read_text(encoding="utf-8"))
            command = broken["details"]["cegis_summary"]["nested_command_preview"]["structured_probe"]["command"]
            command[command.index("--probe-selection") + 1] = "fixed_grid"
            ours_path.write_text(json.dumps(broken), encoding="utf-8")
            audit = _smoke_nested_preview_audit(plan, plan_dir)
            self.assertFalse(audit["valid"])

            command[command.index("--probe-selection") + 1] = "active"
            outer = broken["details"]["cegis_run"]["command"]
            outer[outer.index("--probe-budget") + 1] = "9"
            ours_path.write_text(json.dumps(broken), encoding="utf-8")
            self.assertFalse(_smoke_nested_preview_audit(plan, plan_dir)["valid"])

            # Matching omissions must not masquerade as a controlled comparison.
            outer[outer.index("--probe-budget") + 1] = "8"
            ours_path.write_text(json.dumps(broken), encoding="utf-8")
            for run in plan["runs"]:
                path = plan_dir / "runs" / run["run_id"] / "summary.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                outer = payload["details"]["cegis_run"]["command"]
                index = outer.index("--max-cycles")
                del outer[index:index + 2]
                path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(_smoke_nested_preview_audit(plan, plan_dir)["valid"])

    def test_verifier_reads_nested_cegis_feedback_and_probe_budget(self):
        clean = {
            "budget": {"explicit_probe_case_budget": 8},
            "metrics": {"probe_cases": 8},
            "details": {
                "cegis_summary": {
                    "config": {
                        "probe_strategy": "active",
                        "prompt_policy": "full",
                    },
                    "evidence_policy": {
                        "held_out_seeds_used_for_repair": False,
                        "held_out_feedback_policy": "evaluation_only_no_repair",
                    },
                    "probe_budget": {
                        "scope": "per_run_total",
                        "total": 8,
                        "cases_used": 8,
                    },
                    "cycles": [
                        {"cycle": 1, "repair_feedback": {"held_out_used": False}},
                        {
                            "cycle": 2,
                            "repair_feedback": {"held_out_used": False},
                            "held_out_evaluation": {"used_for_repair": False},
                        },
                    ],
                    "held_out": {"used_for_repair": False},
                }
            },
        }
        tier = {"probe_budget": 8}
        self.assertEqual(_cegis_protocol_errors(clean), [])
        self.assertEqual(_probe_budget_errors(clean, "Ours", tier), [])
        fixed = json.loads(json.dumps(clean))
        fixed["details"]["cegis_summary"]["config"]["probe_strategy"] = "fixed_grid"
        self.assertEqual(_probe_budget_errors(fixed, "B5", tier), [])

        leaked = json.loads(json.dumps(clean))
        leaked["details"]["cegis_summary"]["cycles"].append(
            {"cycle": 3, "repair_feedback": {"held_out_used": True}}
        )
        self.assertTrue(_cegis_protocol_errors(leaked))
        over_budget = json.loads(json.dumps(clean))
        over_budget["metrics"]["probe_cases"] = 9
        self.assertTrue(_probe_budget_errors(over_budget, "Ours", tier))

    def test_verifier_rejects_runtime_llm_config_drift(self):
        expected = {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "max_tokens": 8192,
            "temperature": 0.2,
            "deepseek_thinking": "enabled",
        }
        summary = {
            "runtime": {
                "llm_provider": "deepseek",
                "llm_model": "deepseek-v4-pro",
                "llm_max_tokens": 8192,
                "llm_temperature": 0.2,
                "deepseek_thinking": "enabled",
            }
        }
        self.assertEqual(_llm_config_errors(summary, "Ours", expected), [])
        summary["runtime"]["llm_model"] = "another-model"
        self.assertTrue(_llm_config_errors(summary, "Ours", expected))
        self.assertEqual(_llm_config_errors(summary, "B0", expected), [])

    def test_verifier_enforces_frozen_llm_call_budget(self):
        contract = {
            "llm_api_calls_per_run": {
                "B0": 0,
                "B2": 1,
                "B3": 5,
                "B4": 5,
                "B5": 5,
                "Ours": 5,
            }
        }
        summary = {
            "budget": {
                "llm_api_call_budget": 5,
                "posthoc_analysis_llm_enabled": False,
            },
            "metrics": {"llm_api_calls": 4},
        }
        self.assertEqual(_llm_budget_errors(summary, "Ours", contract), [])

        over_budget = json.loads(json.dumps(summary))
        over_budget["metrics"]["llm_api_calls"] = 6
        self.assertTrue(_llm_budget_errors(over_budget, "Ours", contract))

        wrong_declaration = json.loads(json.dumps(summary))
        wrong_declaration["budget"]["llm_api_call_budget"] = 3
        self.assertTrue(_llm_budget_errors(wrong_declaration, "Ours", contract))

    def test_motivation_evidence_plan_has_two_bounded_real_simulation_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pull = root / "pull.jsonl"
            pick = (
                root
                / "structured_probes"
                / "case03_pick_cube_panda_to_xarm6"
                / "pick_cube_xarm6_close_envelope.json"
            )
            plan = _build_motivation_evidence_plan(
                python_executable="python",
                pull_results=pull,
                pick_probe=pick,
                seeds="0-9",
                max_episode_steps=500,
                probe_max_episode_steps=220,
                probe_cases=32,
                sim_backend="auto",
                render_backend="gpu",
            )
            self.assertEqual(len(plan["commands"]), 2)
            pull_command = plan["commands"][0]["command"]
            probe_command = plan["commands"][1]["command"]
            self.assertIn("scripts/pullcube_multiseed_eval.py", pull_command)
            self.assertIn("0-9", pull_command)
            self.assertIn("scripts/structured_probe_runner.py", probe_command)
            self.assertIn("32", probe_command)
            self.assertEqual(plan["commands"][0]["expected_artifact"], str(pull))
            self.assertEqual(plan["commands"][1]["expected_artifact"], str(pick))

    def test_paired_method_comparison_aligns_seed_and_reports_exact_test(self):
        def run(method, repetition, outcomes):
            return {
                "method_id": method,
                "case_id": "case",
                "repetition": repetition,
                "support_status": "official_supported",
                "details": {
                    "held_out": {
                        "rows": [
                            {"seed": seed, "success": success}
                            for seed, success in outcomes.items()
                        ]
                    }
                },
            }

        rows = paired_method_comparisons(
            [
                run("Ours", 1, {100: True, 101: True, 102: False}),
                run("B0", 1, {100: False, 101: True, 102: False, 999: True}),
            ]
        )
        case_row = next(row for row in rows if row["case_id"] == "case")
        self.assertEqual(case_row["num_paired_trials"], 3)
        self.assertEqual(case_row["reference_only_success"], 1)
        self.assertEqual(case_row["comparator_only_success"], 0)
        self.assertEqual(case_row["paired_success_rate_difference"], 0.3333)
        self.assertEqual(case_row["mcnemar_exact_p"], 1.0)
        self.assertLessEqual(case_row["hierarchical_bootstrap_95_low"], 0.3333)
        self.assertGreaterEqual(case_row["hierarchical_bootstrap_95_high"], 0.3333)
        self.assertEqual(case_row["hierarchical_bootstrap_iterations"], 2000)
        self.assertIn("holm_adjusted_p", case_row)
        support_row = next(
            row for row in rows if row["case_id"] == "__support__:official_supported"
        )
        self.assertEqual(support_row["support_status"], "official_supported")
        self.assertEqual(support_row["num_paired_trials"], 3)
        self.assertEqual(rows, paired_method_comparisons([
            run("Ours", 1, {100: True, 101: True, 102: False}),
            run("B0", 1, {100: False, 101: True, 102: False, 999: True}),
        ]))

    def test_diagnosis_annotation_template_is_blind_and_evaluable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trials = root / "trials.jsonl"
            annotations = root / "annotations.jsonl"
            trial = {
                "type": "trial",
                "task_id": "pull_cube",
                "robot_uid": "fetch",
                "seed": 1,
                "success": False,
                "failure_type": "contact execution failure",
                "message": "Episode ended during descent.",
                "runtime_diagnostics": {
                    "stage": "descent",
                    "tcp_stage_error_norm": 0.2,
                    "tcp_cube_xy": 0.04,
                    "cube_goal_xy": 0.2,
                },
            }
            trials.write_text(json.dumps(trial) + "\n", encoding="utf-8")
            made = build_annotation_template([trials], annotations)
            self.assertEqual(made["num_failures"], 1)
            row = json.loads(annotations.read_text(encoding="utf-8"))
            self.assertNotIn("predicted_layer", row)
            self.assertFalse(any("prediction" in key for key in row))
            row["label_layer"] = "contact_geometry"
            row["label_reason"] = "contact_side_reachability_failure"
            annotations.write_text(json.dumps(row) + "\n", encoding="utf-8")
            secondary = root / "annotations_secondary.jsonl"
            secondary_row = dict(row)
            secondary_row["annotator"] = "reviewer_b"
            secondary.write_text(json.dumps(secondary_row) + "\n", encoding="utf-8")
            result = evaluate_annotations(annotations, secondary_path=secondary)
            self.assertTrue(result["ready"])
            self.assertEqual(result["layer_accuracy"], 1.0)
            self.assertEqual(result["reason_accuracy"], 1.0)
            agreement = result["inter_annotator_agreement"]
            self.assertEqual(agreement["layer"]["num_pairs"], 1)
            self.assertEqual(agreement["layer"]["cohen_kappa"], 1.0)

    def test_remote_package_contains_plan_and_excludes_credentials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_json = root / "plan.json"
            plan_md = root / "plan.md"
            plan_json.write_text("{}", encoding="utf-8")
            plan_md.write_text("# Plan", encoding="utf-8")
            payload = build_remote_package(
                output_dir=root / "packages",
                plan={
                    "plan_name": "unit",
                    "tier": {"tier_id": "pilot"},
                    "num_runs": 1,
                    "num_ready_runs": 1,
                    "num_blocked_runs": 0,
                    "llm_config": {"provider": "deepseek"},
                },
                plan_files={"json": str(plan_json), "markdown": str(plan_md)},
                preparation_checks={
                    "protocol_audit_valid": True,
                    "static_preflight_ready": True,
                    "llm_config_frozen": True,
                    "llm_config_explicit": True,
                    "stochastic_repetitions_configured": True,
                },
            )
            self.assertTrue(payload["archive_sha256"])
            with tarfile.open(payload["archive"], "r:gz") as archive:
                names = archive.getnames()
            self.assertIn("embodied_migration/remote_plan/plan.json", names)
            self.assertIn("embodied_migration/REMOTE_RUN.md", names)
            self.assertFalse(any(Path(name).name == ".env" for name in names))
            with tarfile.open(payload["archive"], "r:gz") as archive:
                run_instructions = archive.extractfile("embodied_migration/REMOTE_RUN.md")
                self.assertIsNotNone(run_instructions)
                instructions = run_instructions.read().decode("utf-8")
                self.assertIn("export DEEPSEEK_API_KEY=YOUR_KEY", instructions)
                self.assertIn("python paper.py start", instructions)
                self.assertIn("python paper.py execute", instructions)
                self.assertIn("python paper.py status", instructions)
                manifest_stream = archive.extractfile(
                    "embodied_migration/remote_plan/package_manifest.json"
                )
                self.assertIsNotNone(manifest_stream)
                packaged_manifest = json.loads(manifest_stream.read().decode("utf-8"))
                self.assertTrue(
                    packaged_manifest["remote_execution"]["automatic_evidence_export"]
                )
                self.assertEqual(
                    packaged_manifest["remote_execution"]["status_command"],
                    "python paper.py status",
                )
                self.assertEqual(
                    packaged_manifest["remote_execution"]["command"],
                    "python paper.py start",
                )
                unpacked = root / "unpacked"
                archive.extractall(unpacked, filter="data")
            verified = verify_remote_package_manifest(
                unpacked / "embodied_migration" / "remote_plan" / "package_manifest.json",
                repo_root=unpacked / "embodied_migration",
            )
            self.assertTrue(verified["valid"])
            self.assertGreater(verified["files_checked"], 10)
            readme = unpacked / "embodied_migration" / "README.md"
            readme.write_text(
                readme.read_text(encoding="utf-8") + "\ntampered\n",
                encoding="utf-8",
            )
            tampered = verify_remote_package_manifest(
                unpacked / "embodied_migration" / "remote_plan" / "package_manifest.json",
                repo_root=unpacked / "embodied_migration",
            )
            self.assertFalse(tampered["valid"])
            self.assertTrue(
                any("SHA mismatch: README.md" in error for error in tampered["errors"])
            )

    def test_execute_restores_frozen_non_secret_llm_environment(self):
        frozen = {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "max_tokens": 8192,
            "temperature": 0.2,
            "deepseek_thinking": "enabled",
        }
        with patch.dict(
            "os.environ",
            {
                "EM_LLM_PROVIDER": "openrouter",
                "EM_MODEL": "stale-model",
                "EM_MAX_TOKENS": "1024",
                "EM_TEMPERATURE": "0",
                "DEEPSEEK_API_KEY": "sk-secret-stays-in-shell",
            },
            clear=True,
        ):
            result = _apply_frozen_llm_environment(frozen, required=True)
            self.assertTrue(result["applied"])
            self.assertEqual(
                result["overridden_variables"],
                ["EM_LLM_PROVIDER", "EM_MAX_TOKENS", "EM_MODEL", "EM_TEMPERATURE"],
            )
            preflight = run_llm_preflight(["Ours"])
            self.assertEqual(preflight["provider"], "deepseek")
            self.assertEqual(preflight["model"], "deepseek-v4-pro")
            self.assertEqual(preflight["max_tokens"], 8192)
            self.assertEqual(preflight["temperature"], 0.2)
            self.assertEqual(preflight["deepseek_thinking"], "enabled")
            self.assertTrue(preflight["api_key_present"])

        self.assertFalse(
            _apply_frozen_llm_environment({}, required=False)["applied"]
        )

    def test_evidence_export_contains_plan_and_external_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_dir = root / "results" / "paper" / "pilot_unit"
            run_dir = plan_dir / "runs" / "run_01"
            run_dir.mkdir(parents=True)
            (plan_dir / "plan.json").write_text("{}", encoding="utf-8")
            (plan_dir / "verification.json").write_text(
                json.dumps({"valid": True, "complete": True}),
                encoding="utf-8",
            )
            bundle = root / "evidence" / "paper_runs" / "run_01"
            bundle.mkdir(parents=True)
            (bundle / "final_adapter.py").write_text("value = 1\n", encoding="utf-8")
            (run_dir / "summary.json").write_text(
                json.dumps({"evidence_bundle": {"bundle_dir": str(bundle)}}),
                encoding="utf-8",
            )
            payload = build_evidence_export(
                plan_dir=plan_dir,
                output_dir=root / "exports",
                verification={
                    "valid": True,
                    "complete": True,
                    "plan_name": "pilot_unit",
                    "tier": "pilot",
                },
                repo_root=root,
            )
            self.assertTrue(payload["archive_sha256"])
            with tarfile.open(payload["archive"], "r:gz") as archive:
                names = set(archive.getnames())
            self.assertIn("results/paper/pilot_unit/plan.json", names)
            self.assertIn("results/paper/pilot_unit/verification.json", names)
            self.assertIn("evidence/paper_runs/run_01/final_adapter.py", names)
            self.assertIn("EVIDENCE_EXPORT_MANIFEST.json", names)

    def test_frozen_plan_rejects_command_tampering_and_rebinds_python(self):
        plan = build_paper_plan(
            tier_id="pilot",
            method_ids=["B0", "Ours"],
            case_ids=["case02_pull_cube_panda_to_xarm6"],
            plan_name="remote_unit",
            sim_backend="auto",
            render_backend="gpu",
            llm_config={
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "max_tokens": 8192,
                "temperature": 0.0,
                "deepseek_thinking": "enabled",
            },
        )
        rebound, errors = _validate_and_rebind_frozen_plan(plan)
        self.assertEqual(errors, [])
        self.assertTrue(rebound["runs"][0]["command"][0])
        self.assertIn("--sim-backend", rebound["runs"][0]["command"])
        self.assertEqual(rebound["runtime_config"]["render_backend"], "gpu")

        tampered = json.loads(json.dumps(plan))
        tampered["runs"][0]["command"][tampered["runs"][0]["command"].index("--probe-budget") + 1] = "999"
        _, errors = _validate_and_rebind_frozen_plan(tampered)
        self.assertIn(
            "frozen run matrix or command arguments do not match local benchmark code",
            errors,
        )

    def test_llm_preflight_only_requires_key_for_llm_methods(self):
        with patch.dict(
            "os.environ",
            {
                "EM_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "",
                "EM_MODEL": "test-model",
                "EM_MAX_TOKENS": "4096",
                "EM_TEMPERATURE": "0",
            },
            clear=False,
        ):
            baseline = run_llm_preflight(["B0", "B1"])
            self.assertTrue(baseline["ready"])
            self.assertFalse(baseline["required"])
            generation = run_llm_preflight(["B2", "Ours"])
            self.assertFalse(generation["ready"])
            self.assertIn("missing OPENROUTER_API_KEY", generation["errors"])

        with patch.dict(
            "os.environ",
            {
                "EM_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "sk-valid",
                "EM_MODEL": "deepseek-v4-pro",
                "EM_DEEPSEEK_THINKING": "enabled",
            },
            clear=False,
        ):
            generation = run_llm_preflight(["Ours"])
            self.assertTrue(generation["ready"])

            deterministic_paper = run_llm_preflight(
                ["Ours"], require_stochastic_repetitions=True
            )
            self.assertFalse(deterministic_paper["ready"])
            self.assertIn("EM_TEMPERATURE > 0", deterministic_paper["errors"][0])

        with patch.dict(
            "os.environ",
            {
                "EM_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "sk-valid",
                "EM_MODEL": "deepseek-v4-pro",
                "EM_DEEPSEEK_THINKING": "enabled",
                "EM_TEMPERATURE": "0.2",
            },
            clear=False,
        ):
            stochastic_paper = run_llm_preflight(
                ["Ours"], require_stochastic_repetitions=True
            )
            self.assertTrue(stochastic_paper["ready"])
            self.assertEqual(generation["deepseek_thinking"], "enabled")

    def test_paper_verifier_accepts_complete_consistent_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_id = "pilot__b0__case__r01"
            run = {
                "run_id": run_id,
                "method_id": "B0",
                "case_id": "case",
                "repetition": 1,
                "development_seeds": "0-1",
                "held_out_seeds": "100-101",
                "ready": True,
            }
            plan = {
                "schema": "embodied_migration_paper_plan.v1",
                "plan_name": "verify",
                "tier": {
                    "success_threshold": 0.5,
                    "min_trials_for_accept": 2,
                },
                "source_baseline_gate": {
                    "development_seeds": "0-1",
                    "held_out_seeds": "100-101",
                    "success_threshold": 0.5,
                    "min_trials_for_accept": 2,
                },
                "budget_contract": {
                    "llm_api_calls_per_run": {"B0": 0},
                },
                "runs": [run],
            }
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            source_dir = root / "source_baselines"
            source_dir.mkdir()
            (source_dir / "summary.json").write_text(
                json.dumps({
                    "schema": "source_baseline_gate.v1",
                    "ready": True,
                    "request": {
                        "development_seeds": "0-1",
                        "held_out_seeds": "100-101",
                        "success_threshold": 0.5,
                        "min_trials_for_accept": 2,
                    },
                }),
                encoding="utf-8",
            )

            evidence_dir = root / "evidence"
            evidence_dir.mkdir()
            adapter = evidence_dir / "final_adapter.py"
            adapter.write_text("value = 1\n", encoding="utf-8")
            adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
            artifacts = {}
            for name in ("development_jsonl", "held_out_jsonl", "commands_log"):
                path = evidence_dir / f"{name}.txt"
                path.write_text("evidence\n", encoding="utf-8")
                artifacts[name] = str(path)
            artifacts["final_adapter"] = str(adapter)
            manifest_path = evidence_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({
                    "schema": "paper_run_evidence.v1",
                    "run_id": run_id,
                    "artifacts": artifacts,
                }),
                encoding="utf-8",
            )

            run_dir = root / "runs" / run_id
            run_dir.mkdir(parents=True)
            summary = {
                "schema": "paper_method_run.v1",
                **{key: run[key] for key in (
                    "run_id", "method_id", "case_id", "repetition",
                    "development_seeds", "held_out_seeds",
                )},
                "success": True,
                "held_out_used_for_repair": False,
                "budget": {
                    "llm_api_call_budget": 0,
                    "posthoc_analysis_llm_enabled": False,
                },
                "metrics": {"llm_api_calls": 0},
                "adapter_sha256": adapter_sha,
                "adapter_provenance": "source_copy",
                "generation_integrity": {"not_applicable": True},
                "frozen_interface_integrity": {"valid": True},
                "runtime": {
                    "git_rev": "abc",
                    "packages": {name: "1" for name in ("mani-skill", "gymnasium", "sapien", "numpy")},
                },
                "details": {
                    "development": {
                        "num_trials": 2,
                        "success_rate": 1.0,
                        "rows": [{"seed": 0}, {"seed": 1}],
                    },
                    "held_out": {
                        "num_trials": 2,
                        "success_rate": 1.0,
                        "rows": [{"seed": 100}, {"seed": 101}],
                    },
                },
                "evidence_bundle": {"manifest": str(manifest_path)},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            verified = verify_paper_plan(root)
            self.assertTrue(verified["valid"])
            self.assertTrue(verified["complete"])
            self.assertEqual(verified["num_valid_runs"], 1)

    def test_source_baseline_gate_deduplicates_tasks_and_resumes(self):
        calls = []

        def fake_runner(args):
            calls.append((args.task, args.robot, args.seeds, args.adapter_module))
            return {
                "summary": {
                    "num_trials": 5,
                    "num_success": 5,
                    "success_rate": 1.0,
                    "generalization_strategy": {"status": "accepted"},
                },
                "wrote": {"jsonl": "trial.jsonl", "markdown": "trial.md"},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "source"
            case_ids = [
                "case01_pull_cube_panda_to_fetch",
                "case02_pull_cube_panda_to_xarm6",
                "case03_pick_cube_panda_to_xarm6",
                "case04_pick_cube_panda_to_fetch",
                "case05_push_cube_panda_to_fetch",
            ]
            first = run_source_baseline_gate(
                case_ids,
                tier_id="pilot",
                output_dir=output,
                runner=fake_runner,
            )
            self.assertTrue(first["ready"])
            self.assertFalse(first["reused"])
            self.assertEqual(len(calls), 6)
            self.assertEqual({item[0] for item in calls}, {"pull_cube", "pick_cube", "push_cube"})
            self.assertEqual({item[2] for item in calls}, {"0-4", "100-104"})

            second = run_source_baseline_gate(
                case_ids,
                tier_id="pilot",
                output_dir=output,
                runner=lambda args: self.fail("cached source baseline should not rerun"),
            )
            self.assertTrue(second["ready"])
            self.assertTrue(second["reused"])

    def test_protocol_audit_enforces_leakage_and_budget_contracts(self):
        payload = run_protocol_audit()
        self.assertTrue(payload["valid"])
        names = {item["name"] for item in payload["checks"]}
        self.assertIn("prompt_information_boundaries", names)
        self.assertIn("paper_development_held_out_disjoint", names)
        self.assertIn("paper_equal_total_probe_budget", names)
        self.assertIn("pilot_equal_iterative_llm_budget", names)
        self.assertIn("paper_equal_iterative_llm_budget", names)
        self.assertIn("held_out_is_evaluation_only_no_repair", names)
        self.assertIn("runtime_backends_are_frozen_in_every_command", names)
        self.assertIn("all_generation_starts_are_neutral", names)
        self.assertIn("primary_hypotheses_are_preregistered_on_official_support", names)

    def test_static_preflight_checks_frozen_case_interfaces(self):
        payload = run_paper_preflight(
            ["case04_pick_cube_panda_to_fetch"],
            static_only=True,
        )
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["rows"][0]["support_status"], "official_supported")
        self.assertTrue(payload["rows"][0]["static_ok"])
        self.assertIsNone(payload["rows"][0]["runtime_ok"])
        with tempfile.TemporaryDirectory() as tmpdir:
            wrote = write_paper_preflight(payload, Path(tmpdir))
            self.assertTrue(Path(wrote["json"]).is_file())
            self.assertIn("case04_pick_cube_panda_to_fetch", Path(wrote["markdown"]).read_text())

    def test_wilson_interval_and_pooled_seed_statistics(self):
        self.assertEqual(_wilson_interval(0, 0), (None, None))
        low, high = _wilson_interval(8, 10)
        self.assertLess(low, 0.8)
        self.assertGreater(high, 0.8)
        plan = {
            "plan_name": "stats",
            "tier": {"tier_id": "paper"},
            "num_runs": 2,
            "num_ready_runs": 2,
            "num_blocked_runs": 0,
            "case_definitions": [{"case_id": "case", "support_status": "official_supported"}],
        }
        runs = []
        for repetition, outcomes in enumerate(((True, True, False), (True, False, False)), start=1):
            rate = sum(outcomes) / len(outcomes)
            runs.append({
                "method_id": "Ours",
                "case_id": "case",
                "success": False,
                "development_success_rate": rate,
                "held_out_success_rate": rate,
                "adapter_provenance": "llm_generated_cegis",
                "metrics": {},
                "details": {"held_out": {"rows": [{"success": value} for value in outcomes]}},
            })
        summary = _summarize(plan, runs, [])
        row = summary["rows"][0]
        self.assertEqual(row["held_out_total_trials"], 6)
        self.assertEqual(row["held_out_total_successes"], 3)
        self.assertEqual(row["held_out_pooled_success_rate"], 0.5)
        self.assertGreater(row["held_out_success_rate_std"], 0.0)
        self.assertEqual(row["support_status"], "official_supported")
        subgroup = summary["support_subgroups"][0]
        self.assertEqual(subgroup["support_status"], "official_supported")
        self.assertEqual(subgroup["held_out_total_trials"], 6)
        self.assertEqual(subgroup["held_out_total_successes"], 3)

    def test_run_plan_resumes_matching_completed_run(self):
        run = {
            "run_id": "pilot__b0__case__r01",
            "method_id": "B0",
            "case_id": "case",
            "repetition": 1,
            "development_seeds": "0-4",
            "held_out_seeds": "100-104",
            "ready": True,
            "command": ["python", "must-not-run.py"],
        }
        summary = {
            "schema": "paper_method_run.v1",
            **{key: run[key] for key in (
                "run_id",
                "method_id",
                "case_id",
                "repetition",
                "development_seeds",
                "held_out_seeds",
            )},
            "success": True,
            "development_success_rate": 1.0,
            "held_out_success_rate": 1.0,
            "adapter_provenance": "source_copy",
            "metrics": {},
        }
        plan = {
            "plan_name": "unit",
            "tier": {"tier_id": "pilot"},
            "num_runs": 1,
            "num_ready_runs": 1,
            "num_blocked_runs": 0,
            "runs": [run],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "runs" / run["run_id"] / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(summary), encoding="utf-8")
            with patch("paper.subprocess.run") as mocked:
                result = _run_plan(plan, root, resume=True)
            mocked.assert_not_called()
            self.assertEqual(result["num_completed_runs"], 1)
            self.assertEqual(result["num_skipped_existing"], 1)
            self.assertEqual(result["num_pending_ready_runs"], 0)

    def test_benchmark_command_streams_to_durable_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "commands.log"
            returncode, tail = _stream_benchmark_command(
                [sys.executable, "-c", "print('streamed-line')"],
                command_log=log_path,
                run_id="unit_run",
            )
            self.assertEqual(returncode, 0)
            self.assertIn("streamed-line", tail)
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("## unit_run", log)
            self.assertIn("streamed-line", log)
            self.assertIn("returncode=0", log)

    def test_detached_remote_start_writes_launcher_and_status(self):
        plan = {
            "schema": "embodied_migration_paper_plan.v1",
            "plan_name": "detached_unit",
            "output_root": "results/paper",
            "tier": {"tier_id": "pilot", "repetitions": 1},
            "method_definitions": [{"method_id": "B0"}],
            "case_definitions": [],
            "llm_config": {},
            "num_ready_runs": 1,
            "runs": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            remote_plan = root / "remote_plan"
            remote_plan.mkdir()
            plan_path = remote_plan / "plan.json"
            manifest_path = remote_plan / "package_manifest.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")
            process = unittest.mock.Mock(pid=43210)
            args = SimpleNamespace(
                plan_file="remote_plan/plan.json",
                package_manifest="remote_plan/package_manifest.json",
                plan_name="",
                output_root="results/paper",
                rerun_completed=False,
            )
            with (
                patch("paper.REPO_ROOT", root),
                patch("paper.verify_remote_package_manifest", return_value={"valid": True}),
                patch("paper._validate_and_rebind_frozen_plan", return_value=(plan, [])),
                patch("paper.run_llm_preflight", return_value={"ready": True}),
                patch("paper._process_alive", return_value=False),
                patch("paper.subprocess.Popen", return_value=process) as popen,
            ):
                started = _start_frozen_remote_plan(args)
            self.assertTrue(started["valid"])
            self.assertTrue(started["started"])
            self.assertEqual(started["pid"], 43210)
            self.assertTrue(Path(started["launcher"]).is_file())
            self.assertTrue(Path(started["execute_log"]).is_file())
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

            with (
                patch("paper.REPO_ROOT", root),
                patch("paper._process_alive", return_value=True),
            ):
                status = _paper_status(args)
            self.assertEqual(status["state"], "launching")
            self.assertEqual(status["launcher_pid"], 43210)
            self.assertTrue(status["launcher_alive"])

    def test_run_plan_writes_atomic_progress_and_status(self):
        run = {
            "run_id": "pilot__b0__case__r01",
            "method_id": "B0",
            "case_id": "case",
            "repetition": 1,
            "development_seeds": "0-1",
            "held_out_seeds": "100-101",
            "ready": True,
            "command": [sys.executable, "unused.py"],
        }
        plan = {
            "schema": "embodied_migration_paper_plan.v1",
            "plan_name": "unit_progress",
            "tier": {"tier_id": "pilot"},
            "num_runs": 1,
            "num_ready_runs": 1,
            "num_blocked_runs": 0,
            "runs": [run],
        }
        completed = {
            "schema": "paper_method_run.v1",
            **{
                key: run[key]
                for key in (
                    "run_id",
                    "method_id",
                    "case_id",
                    "repetition",
                    "development_seeds",
                    "held_out_seeds",
                )
            },
            "success": False,
            "development_success_rate": 0.0,
            "held_out_success_rate": 0.0,
            "adapter_provenance": "source_copy",
            "metrics": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "runs" / run["run_id"] / "summary.json"

            def execute(*_args, **_kwargs):
                summary_path.parent.mkdir(parents=True)
                summary_path.write_text(json.dumps(completed), encoding="utf-8")
                return 0, "done"

            with patch("paper._stream_benchmark_command", side_effect=execute):
                summary = _run_plan(plan, root, resume=False)
            progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["state"], "completed")
            self.assertEqual(progress["completed_runs"], 1)
            self.assertEqual(progress["pending_runs"], 0)
            self.assertEqual(summary["progress_file"], str(root / "progress.json"))

            repository = root / "repository"
            plan_dir = repository / "results" / "paper" / "unit_progress"
            plan_dir.mkdir(parents=True)
            (plan_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            with patch("paper.REPO_ROOT", repository):
                not_started = _paper_status(
                    SimpleNamespace(
                        plan_file="remote_plan/plan.json",
                        plan_name="unit_progress",
                        output_root="results/paper",
                    )
                )
            self.assertEqual(not_started["state"], "not_started")
            self.assertEqual(not_started["completed_runs"], 0)
            self.assertEqual(not_started["pending_runs"], 1)

            (plan_dir / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            with patch("paper.REPO_ROOT", repository):
                status = _paper_status(
                    SimpleNamespace(
                        plan_file="remote_plan/plan.json",
                        plan_name="unit_progress",
                        output_root="results/paper",
                    )
                )
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["completed_runs"], 1)
            self.assertEqual(status["next_action"], "run python paper.py execute")

    def test_resume_rejects_backend_or_llm_config_drift(self):
        run = {
            "run_id": "pilot__b2__case__r01",
            "method_id": "B2",
            "case_id": "case",
            "repetition": 1,
            "development_seeds": "0-4",
            "held_out_seeds": "100-104",
        }
        llm_config = {
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "max_tokens": 8192,
            "temperature": 0.0,
            "deepseek_thinking": "enabled",
        }
        summary = {
            "schema": "paper_method_run.v1",
            **run,
            "sim_backend": "auto",
            "render_backend": "gpu",
            "runtime": {
                "llm_provider": "deepseek",
                "llm_model": "deepseek-v4-pro",
                "llm_max_tokens": 8192,
                "llm_temperature": 0.0,
                "deepseek_thinking": "enabled",
            },
        }
        plan = {
            "runtime_config": {"sim_backend": "auto", "render_backend": "gpu"},
            "llm_config": llm_config,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertIsNotNone(_load_completed_summary(path, run, plan))
            summary["runtime"]["llm_model"] = "different-model"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertIsNone(_load_completed_summary(path, run, plan))
            summary["runtime"]["llm_model"] = "deepseek-v4-pro"
            summary["render_backend"] = "cpu"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertIsNone(_load_completed_summary(path, run, plan))

    def test_summary_exports_csv_and_latex(self):
        rows = [{
            "method_id": "Ours",
            "case_id": "case02_pull_cube",
            "num_repetitions": 3,
            "num_accepted": 2,
            "mean_development_success_rate": 0.9,
            "mean_held_out_success_rate": 0.8,
            "mean_simulator_calls": 12,
            "mean_environment_steps": 4000,
            "mean_llm_api_calls": 2,
            "mean_total_tokens": 8000,
            "adapter_provenance": ["llm_generated_cegis"],
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            tex_path = Path(tmpdir) / "summary.tex"
            _write_summary_csv(csv_path, rows)
            _write_summary_latex(tex_path, rows)
            with csv_path.open(encoding="utf-8") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(exported[0]["method_id"], "Ours")
            latex = tex_path.read_text(encoding="utf-8")
            self.assertIn("case02\\_pull\\_cube", latex)
            self.assertIn("0.8", latex)


if __name__ == "__main__":
    unittest.main()
