"""Machine-check the frozen paper protocol for leakage and budget asymmetry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from maniskill_backend.cases import iter_full_migration_cases
from maniskill_backend.counterexample_loop import (
    HELD_OUT_FEEDBACK_POLICY,
    PROBE_BUDGET_SCOPE,
    _probe_batch_budget,
)
from maniskill_backend.evidence import detect_adapter_provenance
from maniskill_backend.module_generation_runner import PROMPT_POLICIES
from maniskill_backend.paper_benchmark import (
    PAPER_HYPOTHESES,
    PAPER_METHODS,
    PAPER_RESEARCH_QUESTIONS,
    PAPER_TIERS,
    build_paper_plan,
)
from maniskill_backend.paper_preflight import run_paper_preflight
from maniskill_backend.structured_probe import get_probe_spec
from maniskill_backend.source_baseline import SOURCE_TASKS


REPO_ROOT = Path(__file__).resolve().parents[1]


def _seed_set(text: str) -> set[int]:
    result: set[int] = set()
    for part in str(text or "").split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(item))
    return result


def _check(checks: list[Dict[str, Any]], name: str, valid: bool, evidence: Any) -> None:
    checks.append({"name": name, "valid": bool(valid), "evidence": evidence})


def run_protocol_audit() -> Dict[str, Any]:
    checks: list[Dict[str, Any]] = []
    runtime_configs: Dict[str, Any] = {}
    cases = [case for case in iter_full_migration_cases() if case.benchmark_enabled]
    static = run_paper_preflight([case.case_id for case in cases], static_only=True)
    _check(checks, "all_case_interfaces_exist", static["ready"], static["rows"])

    expected_prompt_boundaries = {
        "one_shot": {"diagnosis": False, "probe": False, "contract": False, "guarded_adapter": False},
        "raw_failure": {"diagnosis": False, "probe": False, "contract": False, "guarded_adapter": False},
        "diagnosis": {"diagnosis": True, "probe": False, "contract": False, "guarded_adapter": False},
        "fixed_probe": {"diagnosis": True, "probe": True, "contract": False, "guarded_adapter": False},
        "full": {"diagnosis": True, "probe": True, "contract": True, "guarded_adapter": True},
    }
    observed_boundaries = {
        name: {key: bool(PROMPT_POLICIES[name].get(key)) for key in expected}
        for name, expected in expected_prompt_boundaries.items()
    }
    _check(
        checks,
        "prompt_information_boundaries",
        observed_boundaries == expected_prompt_boundaries,
        observed_boundaries,
    )

    for tier_id, tier in PAPER_TIERS.items():
        development = _seed_set(tier.development_seeds)
        held_out = _seed_set(tier.held_out_seeds)
        _check(
            checks,
            f"{tier_id}_development_held_out_disjoint",
            development.isdisjoint(held_out),
            {"development": sorted(development), "held_out": sorted(held_out)},
        )
        plan = build_paper_plan(tier_id=tier_id, plan_name=f"audit_{tier_id}")
        budget_contract = plan.get("budget_contract") or {}
        runtime_config = plan.get("runtime_config") or {}
        runtime_configs[tier_id] = {
            "declared": runtime_config,
            "commands_match": all(
                _command_arg(run.get("command") or [], "--sim-backend")
                == runtime_config.get("sim_backend")
                and _command_arg(run.get("command") or [], "--render-backend")
                == runtime_config.get("render_backend")
                for run in plan.get("runs") or []
            ),
        }
        source_gate = plan.get("source_baseline_gate") or {}
        _check(
            checks,
            f"{tier_id}_source_baseline_gate_matches_seed_protocol",
            source_gate.get("required") is True
            and source_gate.get("development_seeds") == tier.development_seeds
            and source_gate.get("held_out_seeds") == tier.held_out_seeds
            and source_gate.get("success_threshold") == tier.success_threshold
            and source_gate.get("min_trials_for_accept") == tier.min_trials_for_accept,
            source_gate,
        )
        b5 = [run for run in plan["runs"] if run["method_id"] == "B5" and run["ready"]]
        ours = [run for run in plan["runs"] if run["method_id"] == "Ours" and run["ready"]]
        b5_budgets = {run["case_id"]: _command_arg(run["command"], "--probe-budget") for run in b5}
        ours_budgets = {run["case_id"]: _command_arg(run["command"], "--probe-budget") for run in ours}
        declared_probe_budgets = budget_contract.get("explicit_probe_cases_per_run") or {}
        _check(
            checks,
            f"{tier_id}_equal_total_probe_budget",
            b5_budgets == ours_budgets
            and set(b5_budgets) == {case.case_id for case in cases}
            and PROBE_BUDGET_SCOPE == "per_run_total"
            and declared_probe_budgets.get("B5") == tier.probe_budget
            and declared_probe_budgets.get("Ours") == tier.probe_budget
            and budget_contract.get("simulator_calls_policy")
            == "same_B5_Ours_cap_actual_calls_measured",
            {
                "B5": b5_budgets,
                "Ours": ours_budgets,
                "contract": declared_probe_budgets,
                "scope": PROBE_BUDGET_SCOPE,
                "simulator_calls_policy": budget_contract.get("simulator_calls_policy"),
                "first_active_batch": _probe_batch_budget(
                    tier.probe_budget,
                    max(0, tier.max_cycles - 1),
                ),
            },
        )
        llm_budgets = budget_contract.get("llm_api_calls_per_run") or {}
        iterative_methods = ("B3", "B4", "B5", "Ours")
        command_budgets = {
            method_id: {
                _command_arg(run["command"], "--llm-call-budget")
                for run in plan["runs"]
                if run["method_id"] == method_id and run["ready"]
            }
            for method_id in iterative_methods
        }
        _check(
            checks,
            f"{tier_id}_equal_iterative_llm_budget",
            all(llm_budgets.get(method_id) == tier.max_cycles for method_id in iterative_methods)
            and llm_budgets.get("B2") == 1
            and all(values == {str(tier.max_cycles)} for values in command_budgets.values())
            and budget_contract.get("posthoc_analysis_llm_calls_included") is False
            and budget_contract.get("generation_flag") == "--no-analysis"
            and budget_contract.get("simulator_calls_policy")
            == "same_B5_Ours_cap_actual_calls_measured",
            {
                "contract": budget_contract,
                "commands": command_budgets,
            },
        )
        b5_method = PAPER_METHODS["B5"]
        ours_method = PAPER_METHODS["Ours"]
        method_resources = (plan.get("resource_upper_bounds") or {}).get("by_method") or {}
        _check(
            checks,
            f"{tier_id}_B5_Ours_same_outer_loop_and_prompt",
            b5_method.runner_kind == ours_method.runner_kind == "cegis"
            and b5_method.prompt_policy == ours_method.prompt_policy == "full"
            and b5_method.probe_strategy == "fixed_grid"
            and ours_method.probe_strategy == "active"
            and method_resources.get("B5", {}).get("simulator_episodes")
            == method_resources.get("Ours", {}).get("simulator_episodes"),
            {
                "B5": {
                    "runner_kind": b5_method.runner_kind,
                    "prompt_policy": b5_method.prompt_policy,
                    "probe_strategy": b5_method.probe_strategy,
                    "resources": method_resources.get("B5"),
                },
                "Ours": {
                    "runner_kind": ours_method.runner_kind,
                    "prompt_policy": ours_method.prompt_policy,
                    "probe_strategy": ours_method.probe_strategy,
                    "resources": method_resources.get("Ours"),
                },
            },
        )

    _check(
        checks,
        "runtime_backends_are_frozen_in_every_command",
        all(
            item.get("declared") == {"sim_backend": "auto", "render_backend": "gpu"}
            and item.get("commands_match") is True
            for item in runtime_configs.values()
        ),
        runtime_configs,
    )

    _check(
        checks,
        "held_out_is_evaluation_only_no_repair",
        HELD_OUT_FEEDBACK_POLICY == "evaluation_only_no_repair",
        {
            "policy": HELD_OUT_FEEDBACK_POLICY,
            "allowed_feedback": [
                "initial_case_seed",
                "development_counterexample",
                "active_probe_on_development_counterexample",
            ],
            "forbidden_feedback": ["held_out_trial", "held_out_failure", "held_out_probe"],
        },
    )

    seed_provenance = {}
    for case in cases:
        seed = REPO_ROOT / case.seed_adapter_path
        seed_provenance[case.case_id] = detect_adapter_provenance(seed)
    _check(
        checks,
        "all_generation_starts_are_neutral",
        all(value == "neutral_seed" for value in seed_provenance.values()),
        seed_provenance,
    )

    probes = {}
    for case in cases:
        try:
            probes[case.case_id] = get_probe_spec(case).probe_id
        except KeyError:
            probes[case.case_id] = "missing"
    _check(
        checks,
        "all_ours_cases_have_executable_probe_specs",
        all(value != "missing" for value in probes.values()),
        probes,
    )

    oracle_cases = {
        run["case_id"]
        for run in build_paper_plan(tier_id="pilot", plan_name="audit_oracle")["runs"]
        if run["method_id"] == "Oracle" and run["ready"]
    }
    _check(
        checks,
        "oracle_is_separate_and_explicit",
        oracle_cases == {
            "case01_pull_cube_panda_to_fetch",
            "case02_pull_cube_panda_to_xarm6",
        }
        and PAPER_METHODS["Oracle"].uses_llm is False,
        sorted(oracle_cases),
    )

    support = {case.case_id: case.support_status for case in cases}
    _check(
        checks,
        "case_support_status_declared",
        all(value in {"official_supported", "stress_test_override"} for value in support.values()),
        support,
    )
    source_tasks = {case.task_id for case in cases}
    _check(
        checks,
        "all_tasks_have_source_baseline_adapters",
        source_tasks.issubset(SOURCE_TASKS),
        {task: SOURCE_TASKS.get(task, {}).get("adapter_module", "missing") for task in sorted(source_tasks)},
    )
    hypotheses = build_paper_plan(tier_id="paper", plan_name="audit_hypotheses").get(
        "registered_hypotheses"
    ) or []
    _check(
        checks,
        "primary_hypotheses_are_preregistered_on_official_support",
        len(hypotheses) == len(PAPER_HYPOTHESES)
        and all(item.get("applicable") is True for item in hypotheses)
        and all(
            item.get("population") == "official_supported"
            for item in hypotheses
            if item.get("role") == "primary"
        ),
        hypotheses,
    )
    research_questions = build_paper_plan(
        tier_id="paper", plan_name="audit_research_questions"
    ).get("registered_research_questions") or []
    _check(
        checks,
        "research_questions_declare_metrics_and_artifacts",
        len(research_questions) == len(PAPER_RESEARCH_QUESTIONS)
        and all(item.get("question") for item in research_questions)
        and all(item.get("primary_metrics") for item in research_questions)
        and all(item.get("required_artifacts") for item in research_questions),
        research_questions,
    )
    return {
        "schema": "embodied_migration_protocol_audit.v1",
        "valid": all(item["valid"] for item in checks),
        "num_checks": len(checks),
        "num_passed": sum(item["valid"] for item in checks),
        "checks": checks,
    }


def _command_arg(command: Iterable[str], option: str) -> str:
    values = list(command)
    try:
        return values[values.index(option) + 1]
    except (ValueError, IndexError):
        return "missing"


def protocol_audit_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Paper Protocol Audit",
        "",
        f"- valid: `{payload.get('valid')}`",
        f"- checks passed: `{payload.get('num_passed')}/{payload.get('num_checks')}`",
        "",
        "| Check | Valid |",
        "|---|---|",
    ]
    for item in payload.get("checks") or []:
        lines.append(f"| `{item.get('name')}` | {item.get('valid')} |")
    lines.append("")
    return "\n".join(lines)


def write_protocol_audit(payload: Mapping[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "protocol_audit.json"
    md_path = output_dir / "protocol_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")
    md_path.write_text(protocol_audit_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
