"""Canonical paper benchmark specification and deterministic run planning."""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from maniskill_backend.cases import FullMigrationCase, iter_full_migration_cases
from maniskill_backend.structured_probe import get_probe_spec


@dataclass(frozen=True)
class PaperTier:
    tier_id: str
    development_seeds: str
    held_out_seeds: str
    repetitions: int
    max_cycles: int
    probe_budget: int
    success_threshold: float = 0.8
    min_trials_for_accept: int = 5


@dataclass(frozen=True)
class PaperMethod:
    method_id: str
    label: str
    runner_kind: str
    prompt_policy: str = ""
    max_attempts: int = 0
    llm_budget_policy: str = "none"
    probe_strategy: str = "none"
    uses_llm: bool = False
    description: str = ""


@dataclass(frozen=True)
class PaperRunSpec:
    run_id: str
    tier_id: str
    method_id: str
    case_id: str
    repetition: int
    development_seeds: str
    held_out_seeds: str
    ready: bool
    blocked_reason: str
    command: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["command_text"] = " ".join(shlex.quote(part) for part in self.command)
        return payload


PAPER_TIERS: Dict[str, PaperTier] = {
    "pilot": PaperTier(
        tier_id="pilot",
        development_seeds="0-4",
        held_out_seeds="100-104",
        repetitions=1,
        max_cycles=3,
        probe_budget=8,
    ),
    "paper": PaperTier(
        tier_id="paper",
        development_seeds="0-9",
        held_out_seeds="100-129",
        repetitions=3,
        max_cycles=5,
        probe_budget=8,
    ),
}


PAPER_METHODS: Dict[str, PaperMethod] = {
    "B0": PaperMethod(
        "B0",
        "Source adapter copy",
        "evaluate_source_copy",
        description="Unmodified source skill adapter executed on the target embodiment.",
    ),
    "B1": PaperMethod(
        "B1",
        "Shared generic adapter",
        "evaluate_shared_generic",
        description="Task-generic adapter with action-layout compatibility but no target geometry repair.",
    ),
    "B2": PaperMethod(
        "B2",
        "LLM one-shot",
        "direct_generation",
        prompt_policy="one_shot",
        max_attempts=1,
        llm_budget_policy="fixed",
        uses_llm=True,
        description="One LLM candidate from task/profile/current code and the raw initial simulator result.",
    ),
    "B3": PaperMethod(
        "B3",
        "LLM + raw failure",
        "direct_generation",
        prompt_policy="raw_failure",
        llm_budget_policy="tier_max_cycles",
        uses_llm=True,
        description="Iterative repair using raw target failure only.",
    ),
    "B4": PaperMethod(
        "B4",
        "LLM + five-layer diagnosis",
        "direct_generation",
        prompt_policy="diagnosis",
        llm_budget_policy="tier_max_cycles",
        uses_llm=True,
        description="Iterative repair with structured failure diagnosis but no probe or contract.",
    ),
    "B5": PaperMethod(
        "B5",
        "CEGIS + fixed-grid probe",
        "cegis",
        prompt_policy="full",
        llm_budget_policy="tier_max_cycles",
        probe_strategy="fixed_grid",
        uses_llm=True,
        description=(
            "The same counterexample-guided repair loop as Ours, with a "
            "deterministic fixed-grid probe selector."
        ),
    ),
    "Ours": PaperMethod(
        "Ours",
        "Counterexample-guided embodiment adapter synthesis",
        "cegis",
        prompt_policy="full",
        llm_budget_policy="tier_max_cycles",
        probe_strategy="active",
        uses_llm=True,
        description="Contract + development counterexample + active probe + guarded repair.",
    ),
    "Oracle": PaperMethod(
        "Oracle",
        "Hand-written oracle",
        "evaluate_oracle",
        description="Human-written upper bound; never counted as LLM-generated success.",
    ),
}


PAPER_RESEARCH_QUESTIONS: tuple[Dict[str, Any], ...] = (
    {
        "rq_id": "RQ1",
        "question": "Does counterexample-guided adapter synthesis improve held-out task success over one-shot LLM migration?",
        "population": "official_supported",
        "required_methods": ("B2", "Ours"),
        "primary_metrics": (
            "paired_held_out_task_success",
            "paired_hierarchical_bootstrap_95_interval",
            "held_out_pooled_success_rate",
            "held_out_wilson_95_interval",
        ),
        "required_artifacts": (
            "held_out_jsonl",
            "paired_method_comparisons_csv",
        ),
    },
    {
        "rq_id": "RQ2",
        "question": "Does active constraint-guided probing outperform fixed-grid probing under the same outer repair loop, prompt information, explicit probe budget, and iterative LLM repair-call budget?",
        "population": "official_supported",
        "required_methods": ("B5", "Ours"),
        "primary_metrics": (
            "paired_held_out_task_success",
            "paired_hierarchical_bootstrap_95_interval",
            "simulator_calls",
            "probe_cases",
            "cegis_cycles",
        ),
        "required_artifacts": (
            "active_probe_plan_json",
            "structured_probe_result_json",
            "paired_method_comparisons_csv",
        ),
    },
    {
        "rq_id": "RQ3",
        "question": "How accurately does the five-layer diagnosis localize real physical failures?",
        "population": "all_failed_trials",
        "required_methods": (),
        "primary_metrics": (
            "diagnosis_layer_accuracy",
            "diagnosis_reason_accuracy",
            "cohen_kappa",
            "confusion_matrix",
        ),
        "required_artifacts": (
            "blind_diagnosis_annotations_jsonl",
            "diagnosis_evaluation_json",
        ),
    },
    {
        "rq_id": "RQ4",
        "question": "How robust are synthesized adapters across unseen seeds and unsupported-robot stress tests?",
        "population": "official_supported_and_stress_test_reported_separately",
        "required_methods": ("Ours",),
        "primary_metrics": (
            "held_out_pooled_success_rate",
            "held_out_success_rate_std",
            "support_subgroup_success_rate",
        ),
        "required_artifacts": (
            "held_out_jsonl",
            "summary_support_subgroups_csv",
        ),
    },
    {
        "rq_id": "RQ5",
        "question": "What simulator, environment, LLM, time, and code-complexity cost is required for repair?",
        "population": "all_executable_runs",
        "required_methods": (),
        "primary_metrics": (
            "simulator_calls",
            "environment_steps",
            "llm_api_calls",
            "total_tokens",
            "wall_time_seconds",
            "adapter_code_lines",
            "adapter_branch_count",
        ),
        "required_artifacts": (
            "paper_method_run_summary_json",
            "paper_summary_csv",
        ),
    },
)


PAPER_HYPOTHESES: tuple[Dict[str, Any], ...] = (
    {
        "hypothesis_id": "H1",
        "role": "primary",
        "population": "official_supported",
        "reference_method": "Ours",
        "comparator_method": "B2",
        "metric": "paired_held_out_task_success",
        "direction": "reference_greater_than_comparator",
        "test": "two_sided_exact_mcnemar_with_case_level_holm_correction",
    },
    {
        "hypothesis_id": "H2",
        "role": "primary",
        "population": "official_supported",
        "reference_method": "Ours",
        "comparator_method": "B5",
        "metric": "paired_held_out_task_success",
        "direction": "reference_greater_than_comparator",
        "test": "two_sided_exact_mcnemar_with_case_level_holm_correction",
        "budget_condition": "same_outer_loop_prompt_and_equal_probe_llm_budgets",
    },
    {
        "hypothesis_id": "H3",
        "role": "secondary_descriptive",
        "population": "stress_test_override",
        "reference_method": "Ours",
        "comparator_method": "B5",
        "metric": "paired_held_out_task_success",
        "direction": "report_without_pooling_into_primary_population",
        "test": "descriptive_exact_mcnemar",
        "budget_condition": "same_outer_loop_prompt_and_equal_probe_llm_budgets",
    },
)


def normalize_method_id(value: str) -> str:
    text = str(value or "").strip().lower()
    for method_id in PAPER_METHODS:
        if text == method_id.lower():
            return method_id
    available = ", ".join(PAPER_METHODS)
    raise KeyError(f"Unknown paper method {value!r}. Available: {available}")


def parse_selection(value: str, available: Sequence[str]) -> List[str]:
    text = str(value or "all").strip()
    if text.lower() in {"all", "ready"}:
        return list(available)
    requested = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise KeyError(f"Unknown selection {unknown!r}. Available: {', '.join(available)}")
    return list(dict.fromkeys(requested))


def _case_has_probe(case: FullMigrationCase) -> bool:
    try:
        get_probe_spec(case)
    except KeyError:
        return False
    return True


def _oracle_available(case: FullMigrationCase) -> bool:
    return case.case_id in {
        "case01_pull_cube_panda_to_fetch",
        "case02_pull_cube_panda_to_xarm6",
    }


def method_availability(method: PaperMethod, case: FullMigrationCase) -> tuple[bool, str]:
    if not case.benchmark_enabled:
        return False, "Exploratory case: validate source baseline and register matched method/probe backends before paper evaluation."
    if method.method_id == "B5" and not _case_has_probe(case):
        return False, "No executable fixed-grid probe backend is registered for this case."
    if method.method_id == "Ours" and not _case_has_probe(case):
        return False, "The full method requires an executable active-probe backend for this case."
    if method.method_id == "Oracle" and not _oracle_available(case):
        return False, "No hand-written oracle is registered for this case."
    return True, ""


def _seed_count(spec: str) -> int:
    values = set()
    for raw in str(spec or "").split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            left, right = (int(part) for part in item.split("-", 1))
            step = 1 if right >= left else -1
            values.update(range(left, right + step, step))
        else:
            values.add(int(item))
    return len(values)


def method_llm_call_budget(method: PaperMethod, tier: PaperTier) -> int:
    """Return the frozen maximum number of adapter-generation calls per run."""

    if not method.uses_llm:
        return 0
    if method.llm_budget_policy == "fixed":
        return int(method.max_attempts)
    if method.llm_budget_policy == "tier_max_cycles":
        return int(tier.max_cycles)
    raise ValueError(
        f"LLM method {method.method_id!r} has invalid budget policy "
        f"{method.llm_budget_policy!r}."
    )


def _budget_contract(tier: PaperTier, method_ids: Sequence[str]) -> Dict[str, Any]:
    llm_budgets = {
        method_id: method_llm_call_budget(PAPER_METHODS[method_id], tier)
        for method_id in method_ids
    }
    probe_budgets = {
        method_id: tier.probe_budget if method_id in {"B5", "Ours"} else 0
        for method_id in method_ids
    }
    return {
        "scope": "per_run_maximum",
        "llm_api_calls_per_run": llm_budgets,
        "explicit_probe_cases_per_run": probe_budgets,
        "equal_iterative_llm_budget_methods": [
            method_id
            for method_id in ("B3", "B4", "B5", "Ours")
            if method_id in method_ids
        ],
        "equal_probe_budget_methods": [
            method_id for method_id in ("B5", "Ours") if method_id in method_ids
        ],
        "equal_outer_loop_methods": [
            method_id for method_id in ("B5", "Ours") if method_id in method_ids
        ],
        "simulator_calls_policy": "same_B5_Ours_cap_actual_calls_measured",
        "posthoc_analysis_llm_calls_included": False,
        "generation_flag": "--no-analysis",
    }


def _resource_upper_bounds(
    tier: PaperTier,
    runs: Sequence[PaperRunSpec],
    cases_by_id: Mapping[str, FullMigrationCase],
) -> Dict[str, Any]:
    """Conservative episode/API budget for the frozen plan."""

    development_trials = _seed_count(tier.development_seeds)
    held_out_trials = _seed_count(tier.held_out_seeds)
    source_tasks: Dict[str, int] = {}
    by_method: Dict[str, Dict[str, int]] = {}
    for run in runs:
        if not run.ready:
            continue
        case = cases_by_id[run.case_id]
        source_tasks[case.task_id] = max(source_tasks.get(case.task_id, 0), case.max_episode_steps)
        method = PAPER_METHODS[run.method_id]
        llm_calls = method_llm_call_budget(method, tier)
        probe_cases = 0
        if method.runner_kind == "direct_generation":
            # Source check, initial target, then one target trial per candidate.
            episodes = 2 + llm_calls + development_trials + held_out_trials
        elif method.runner_kind == "cegis":
            probe_cases = tier.probe_budget
            episodes = (
                1
                + 2 * llm_calls
                + llm_calls * development_trials
                + held_out_trials
                + probe_cases
            )
        else:
            episodes = development_trials + held_out_trials
        row = by_method.setdefault(
            run.method_id,
            {
                "runs": 0,
                "llm_calls": 0,
                "probe_cases": 0,
                "simulator_episodes": 0,
                "environment_steps": 0,
            },
        )
        row["runs"] += 1
        row["llm_calls"] += llm_calls
        row["probe_cases"] += probe_cases
        row["simulator_episodes"] += episodes
        row["environment_steps"] += episodes * case.max_episode_steps

    source_gate_episodes = len(source_tasks) * (development_trials + held_out_trials)
    source_gate_steps = sum(
        (development_trials + held_out_trials) * max_steps
        for max_steps in source_tasks.values()
    )
    target_episodes = sum(item["simulator_episodes"] for item in by_method.values())
    target_steps = sum(item["environment_steps"] for item in by_method.values())
    return {
        "scope": "conservative_maximum_not_expected_usage",
        "source_gate_tasks": len(source_tasks),
        "source_gate_episodes": source_gate_episodes,
        "target_method_episodes": target_episodes,
        "total_simulator_episodes": source_gate_episodes + target_episodes,
        "total_environment_steps": source_gate_steps + target_steps,
        "total_llm_calls": sum(item["llm_calls"] for item in by_method.values()),
        "total_probe_cases": sum(item["probe_cases"] for item in by_method.values()),
        "by_method": by_method,
        "assumptions": [
            "Every repair cycle and direct-generation attempt reaches its maximum.",
            "Count source checks and initial-target trials as well as generated-candidate trials.",
            "Every ready run evaluates the full development and held-out split allowed by its runner.",
            "Probe cases consume one simulator episode each.",
            "Environment-step bound multiplies every episode by the case maximum.",
        ],
    }


def build_paper_plan(
    *,
    tier_id: str,
    method_ids: Iterable[str] | None = None,
    case_ids: Iterable[str] | None = None,
    output_root: str = "results/paper",
    plan_name: str = "paper_benchmark",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    llm_config: Mapping[str, Any] | None = None,
    dry_run_commands: bool = False,
) -> Dict[str, Any]:
    try:
        tier = PAPER_TIERS[tier_id]
    except KeyError as exc:
        raise KeyError(f"Unknown tier {tier_id!r}. Available: {', '.join(PAPER_TIERS)}") from exc

    selected_methods = [normalize_method_id(item) for item in (method_ids or PAPER_METHODS)]
    budget_contract = _budget_contract(tier, selected_methods)
    cases_by_id = {case.case_id: case for case in iter_full_migration_cases()}
    selected_cases = list(case_ids or [key for key, case in cases_by_id.items() if case.benchmark_enabled])
    unknown_cases = [case_id for case_id in selected_cases if case_id not in cases_by_id]
    if unknown_cases:
        raise KeyError(f"Unknown paper cases: {unknown_cases!r}")

    runs: List[PaperRunSpec] = []
    for method_id in selected_methods:
        method = PAPER_METHODS[method_id]
        for case_id in selected_cases:
            case = cases_by_id[case_id]
            ready, blocked_reason = method_availability(method, case)
            for repetition in range(1, tier.repetitions + 1):
                run_id = f"{tier.tier_id}__{method_id.lower()}__{case.case_id}__r{repetition:02d}"
                command = [
                    sys.executable,
                    "scripts/paper_method_runner.py",
                    "--method",
                    method_id,
                    "--case",
                    case.case_id,
                    "--tier",
                    tier.tier_id,
                    "--repetition",
                    str(repetition),
                    "--development-seeds",
                    tier.development_seeds,
                    "--held-out-seeds",
                    tier.held_out_seeds,
                    "--max-cycles",
                    str(tier.max_cycles),
                    "--llm-call-budget",
                    str(budget_contract["llm_api_calls_per_run"][method_id]),
                    "--probe-budget",
                    str(tier.probe_budget),
                    "--success-threshold",
                    str(tier.success_threshold),
                    "--min-trials-for-accept",
                    str(tier.min_trials_for_accept),
                    "--output-root",
                    output_root,
                    "--plan-name",
                    plan_name,
                    "--run-id",
                    run_id,
                    "--sim-backend",
                    sim_backend,
                    "--render-backend",
                    render_backend,
                ]
                if dry_run_commands:
                    command.append("--dry-run")
                runs.append(
                    PaperRunSpec(
                        run_id=run_id,
                        tier_id=tier.tier_id,
                        method_id=method_id,
                        case_id=case.case_id,
                        repetition=repetition,
                        development_seeds=tier.development_seeds,
                        held_out_seeds=tier.held_out_seeds,
                        ready=ready,
                        blocked_reason=blocked_reason,
                        command=tuple(command),
                    )
                )

    resource_upper_bounds = _resource_upper_bounds(tier, runs, cases_by_id)
    return {
        "schema": "embodied_migration_paper_plan.v1",
        "plan_name": plan_name,
        "output_root": output_root,
        "runtime_config": {
            "sim_backend": sim_backend,
            "render_backend": render_backend,
        },
        "llm_config": dict(llm_config or {}),
        "tier": asdict(tier),
        "budget_contract": budget_contract,
        "frozen_interfaces": [
            "high_level_program",
            "low_level_controller",
            "simulator",
            "task_success_signal",
        ],
        "source_baseline_gate": {
            "required": True,
            "development_seeds": tier.development_seeds,
            "held_out_seeds": tier.held_out_seeds,
            "success_threshold": tier.success_threshold,
            "min_trials_for_accept": tier.min_trials_for_accept,
        },
        "method_definitions": [asdict(PAPER_METHODS[item]) for item in selected_methods],
        "registered_research_questions": [
            {
                **item,
                "applicable": set(item["required_methods"]).issubset(selected_methods),
            }
            for item in PAPER_RESEARCH_QUESTIONS
        ],
        "registered_hypotheses": [
            {
                **item,
                "applicable": item["reference_method"] in selected_methods
                and item["comparator_method"] in selected_methods,
            }
            for item in PAPER_HYPOTHESES
        ],
        "case_definitions": [
            {
                "case_id": cases_by_id[item].case_id,
                "title": cases_by_id[item].title,
                "task_id": cases_by_id[item].task_id,
                "source_robot": cases_by_id[item].source_robot,
                "target_robot": cases_by_id[item].target_robot,
                "support_status": cases_by_id[item].support_status,
                "support_source_url": cases_by_id[item].support_source_url,
            }
            for item in selected_cases
        ],
        "case_ids": selected_cases,
        "num_runs": len(runs),
        "num_ready_runs": sum(1 for run in runs if run.ready),
        "num_blocked_runs": sum(1 for run in runs if not run.ready),
        "resource_upper_bounds": resource_upper_bounds,
        "runs": [run.to_dict() for run in runs],
    }


def paper_plan_markdown(plan: Mapping[str, Any]) -> str:
    tier = plan.get("tier") or {}
    resources = plan.get("resource_upper_bounds") or {}
    runtime = plan.get("runtime_config") or {}
    llm = plan.get("llm_config") or {}
    budget = plan.get("budget_contract") or {}
    llm_budgets = budget.get("llm_api_calls_per_run") or {}
    probe_budgets = budget.get("explicit_probe_cases_per_run") or {}
    lines = [
        "# Paper Benchmark Plan",
        "",
        f"- tier: `{tier.get('tier_id')}`",
        f"- development seeds: `{tier.get('development_seeds')}`",
        f"- held-out seeds: `{tier.get('held_out_seeds')}`",
        f"- independent repetitions: `{tier.get('repetitions')}`",
        f"- planned runs: `{plan.get('num_runs')}`",
        f"- executable runs: `{plan.get('num_ready_runs')}`",
        f"- blocked runs: `{plan.get('num_blocked_runs')}`",
        f"- simulator backend: `{runtime.get('sim_backend')}`",
        f"- render backend: `{runtime.get('render_backend')}`",
        f"- LLM: `{llm.get('provider')} / {llm.get('model')}`",
        f"- LLM max tokens / temperature: `{llm.get('max_tokens')} / {llm.get('temperature')}`",
        f"- max simulator episodes: `{resources.get('total_simulator_episodes')}`",
        f"- max LLM calls: `{resources.get('total_llm_calls')}`",
        f"- max probe cases: `{resources.get('total_probe_cases')}`",
        "- source baseline gate: `required`",
        "",
        "## Methods",
        "",
        "| ID | Method | Prompt policy | Probe selector | LLM | Max LLM calls/run | Probe cases/run |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for method in plan.get("method_definitions") or []:
        lines.append(
            f"| {method.get('method_id')} | {method.get('label')} | "
            f"{method.get('prompt_policy') or '-'} | "
            f"{method.get('probe_strategy') or '-'} | {method.get('uses_llm')} | "
            f"{llm_budgets.get(method.get('method_id'), 0)} | "
            f"{probe_budgets.get(method.get('method_id'), 0)} |"
        )
    lines.extend(
        [
            "",
            "B5 and Ours use the same outer loop and simulator-call cap; actual calls are measured because either method may stop early.",
            "Post-hoc analysis LLM calls are disabled for all budgeted runs.",
        ]
    )
    lines.extend(
        [
            "",
            "## Registered Research Questions",
            "",
            "| ID | Question | Population | Required methods | Metrics | Applicable |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in plan.get("registered_research_questions") or []:
        lines.append(
            f"| {item.get('rq_id')} | {item.get('question')} | "
            f"`{item.get('population')}` | "
            f"{', '.join(item.get('required_methods') or []) or '-'} | "
            f"{', '.join(item.get('primary_metrics') or [])} | "
            f"{item.get('applicable')} |"
        )
    lines.extend(
        [
            "",
            "## Registered Hypotheses",
            "",
            "| ID | Role | Population | Comparison | Metric | Applicable |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in plan.get("registered_hypotheses") or []:
        lines.append(
            f"| {item.get('hypothesis_id')} | {item.get('role')} | "
            f"`{item.get('population')}` | {item.get('reference_method')} > "
            f"{item.get('comparator_method')} | `{item.get('metric')}` | "
            f"{item.get('applicable')} |"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Transfer | Support status |",
            "|---|---|---|",
        ]
    )
    for case in plan.get("case_definitions") or []:
        lines.append(
            f"| `{case.get('case_id')}` | {case.get('source_robot')} -> "
            f"{case.get('target_robot')} ({case.get('task_id')}) | "
            f"`{case.get('support_status')}` |"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Method | Case | Rep | Ready | Reason |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for run in plan.get("runs") or []:
        lines.append(
            f"| `{run.get('run_id')}` | {run.get('method_id')} | `{run.get('case_id')}` | "
            f"{run.get('repetition')} | {run.get('ready')} | {run.get('blocked_reason') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_paper_plan(plan: Mapping[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "plan.json"
    md_path = output_dir / "plan.md"
    commands_path = output_dir / "commands.sh"
    json_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(paper_plan_markdown(plan), encoding="utf-8")
    commands = ["#!/usr/bin/env bash", "set -u", ""]
    for run in plan.get("runs") or []:
        if not run.get("ready"):
            commands.append(f"# BLOCKED {run.get('run_id')}: {run.get('blocked_reason')}")
            continue
        commands.append(str(run.get("command_text") or ""))
    commands_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "commands": str(commands_path),
    }
