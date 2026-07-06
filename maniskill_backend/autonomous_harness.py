"""Autonomous harness state for simulation-in-the-loop adapter repair.

The key split is:

- agent observation: compact JSON facts, tool commands, and simulator outputs.
- human report: optional explanation for the researcher.

The LLM agent should consume only the observation, not the human report.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

from maniskill_backend.cases import FullMigrationCase, get_full_migration_case
from maniskill_backend.generalization import build_generalization_report
from maniskill_backend.structured_probe import get_probe_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HarnessTool:
    """One safe simulator-facing tool exposed to the LLM agent."""

    name: str
    description: str
    inputs: Sequence[str]
    outputs: Sequence[str]
    returns: Sequence[str]
    command_template: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "returns": list(self.returns),
            "command_template": self.command_template,
        }


def tool_inventory(
    case: FullMigrationCase,
    *,
    seeds: str = "0-9",
    seed: int = 0,
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 500,
) -> List[HarnessTool]:
    """Return the bounded tool surface for one migration case."""

    single_seed_cmd = _single_seed_command(
        case,
        seed=seed,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
    )
    probe_cmd = _probe_command(
        case,
        seed=seed,
        diagnosis={},
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
    )
    repair_cmd = _repair_command(case, sim_backend=sim_backend, render_backend=render_backend)
    tools = [
        HarnessTool(
            name="run_single_seed",
            description="Run the current target adapter once in the real ManiSkill environment.",
            inputs=("seed", "case_id", "adapter_module", "max_episode_steps"),
            outputs=("trial_json",),
            returns=("success", "failure_layer", "failure_diagnosis", "runtime_diagnostics"),
            command_template=single_seed_cmd,
        ),
        HarnessTool(
            name="run_structured_probe",
            description="Run a bounded parameter sweep for the current physical bottleneck.",
            inputs=("seed", "case_id", "failure_diagnosis"),
            outputs=("probe_json", "probe_markdown", "probe_prompt_feedback"),
            returns=("probe_table", "best_probe_case", "top_probe_cases", "prompt_feedback"),
            command_template=probe_cmd,
        ),
        HarnessTool(
            name="run_llm_repair",
            description="Ask the LLM to rewrite only the target-side adapter using available tool outputs.",
            inputs=("case_id", "current_adapter", "latest_results", "probe_outputs"),
            outputs=("generated_adapter", "target_result", "migration_analysis"),
            returns=("generated_adapter", "verification_result", "target_result", "migration_analysis"),
            command_template=repair_cmd,
        ),
        HarnessTool(
            name="inspect_results",
            description="Read compact JSON/Markdown artifacts instead of raw simulator internals.",
            inputs=("artifact_path",),
            outputs=("artifact_text",),
            returns=("latest_summary", "failure_clusters", "runtime_diagnostics"),
            command_template="tail -n 120 results/pullcube_xarm6_multiseed.md",
        ),
    ]
    if case.task_id == "pull_cube":
        tools.insert(
            1,
            HarnessTool(
                name="run_multi_seed",
                description="Evaluate current adapter across a seed set without regenerating code.",
                inputs=("seed_range", "case_id", "adapter_module", "max_episode_steps"),
                outputs=("multiseed_jsonl", "multiseed_markdown"),
                returns=("success_rate", "failure_seed_clusters", "generalization_strategy"),
                command_template=_multi_seed_command(
                    case,
                    seeds=seeds,
                    sim_backend=sim_backend,
                    render_backend=render_backend,
                    max_episode_steps=max_episode_steps,
                ),
            ),
        )
    return tools


def load_multiseed_jsonl(path: Path | str) -> Dict[str, Any]:
    """Load a multi-seed JSONL artifact into the same summary shape used by the harness."""

    jsonl_path = Path(path)
    metadata: Dict[str, Any] = {}
    trials: List[Dict[str, Any]] = []
    if not jsonl_path.exists():
        raise FileNotFoundError(f"multi-seed jsonl does not exist: {jsonl_path}")
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        kind = item.pop("type", "")
        if kind == "metadata":
            metadata = item
        elif kind == "trial":
            trials.append(item)

    rows = [_trial_digest(item) for item in trials]
    successes = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    elapsed = [_elapsed_steps(row) for row in rows]
    elapsed = [step for step in elapsed if step is not None]
    summary = {
        **metadata,
        "num_trials": len(rows),
        "num_success": len(successes),
        "num_failure": len(failures),
        "success_rate": round(len(successes) / len(rows), 4) if rows else 0.0,
        "mean_elapsed_steps": round(mean(elapsed), 2) if elapsed else None,
        "rows": rows,
    }
    summary["generalization_strategy"] = build_generalization_report(summary)
    return summary


def select_failure_seed(summary: Mapping[str, Any], *, policy: str = "auto") -> Dict[str, Any]:
    """Select the next seed to probe from failed multi-seed rows."""

    failures = [dict(row) for row in summary.get("rows") or [] if not bool(row.get("success"))]
    if not failures:
        return {
            "seed": None,
            "policy": policy,
            "reason": "No failed seeds are available; current adapter may be accepted or needs a broader seed set.",
            "row": None,
        }

    selected_policy = policy
    if policy == "auto":
        strategy = (summary.get("generalization_strategy") or {}).get("selected_strategy")
        if strategy == "reachability_aware_contact_selection":
            selected_policy = "near_contact"
        else:
            selected_policy = "first"

    if selected_policy == "near_contact":
        ranked = sorted(
            failures,
            key=lambda row: (
                _float_from_runtime(row, "tcp_cube_xy", default=999.0),
                _float_from_runtime(row, "tcp_stage_error_norm", default=999.0),
            ),
        )
        reason = "Probe the closest failed contact case first; it is the cheapest way to test whether geometry repair helps."
    elif selected_policy == "severe_reachability":
        ranked = sorted(
            failures,
            key=lambda row: (
                _float_from_runtime(row, "tcp_stage_error_norm", default=-1.0),
                _float_from_runtime(row, "tcp_cube_xy", default=-1.0),
            ),
            reverse=True,
        )
        reason = "Probe the most severe reachability failure to expose the hard boundary."
    else:
        ranked = failures
        reason = "Probe the first failed seed."

    row = ranked[0]
    return {
        "seed": row.get("seed"),
        "policy": selected_policy,
        "reason": reason,
        "row": row,
    }


def load_probe_payload(path: Path | str) -> Dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"structured probe json does not exist: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def build_harness_plan(
    *,
    case_id: str,
    multiseed_jsonl: Path | str | None = None,
    probe_json: Path | str | None = None,
    include_existing_probe: bool = True,
    seed_policy: str = "auto",
    seeds: str = "0-9",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 500,
) -> Dict[str, Any]:
    """Build a two-channel harness bundle.

    ``agent_observation`` is the only object intended for an LLM agent. It
    contains facts and tool commands, but no natural-language repair report and
    no recommended action. ``human_report`` may include a suggested next action
    for the researcher.
    """

    case = get_full_migration_case(case_id)
    summary = load_multiseed_jsonl(multiseed_jsonl) if multiseed_jsonl else None
    seed_selection = select_failure_seed(summary, policy=seed_policy) if summary else {
        "seed": 0,
        "policy": seed_policy,
        "reason": "No multi-seed result was provided; start from seed 0.",
        "row": None,
    }
    selected_seed = int(seed_selection["seed"] or 0)
    selected_diagnosis = _diagnosis_from_selection(seed_selection)
    probe_payload = _load_probe_for_case(case, probe_json=probe_json, include_existing_probe=include_existing_probe)

    tools = [tool.to_dict() for tool in tool_inventory(
        case,
        seeds=seeds,
        seed=selected_seed,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
    )]
    suggested_action = _next_action(
        case,
        summary=summary,
        probe_payload=probe_payload,
        selected_seed=selected_seed,
        selected_diagnosis=selected_diagnosis,
        sim_backend=sim_backend,
        render_backend=render_backend,
        max_episode_steps=max_episode_steps,
    )
    agent_observation = {
        "schema": "agent_observation.v1",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "source_robot": case.source_robot,
        "target_robot": case.target_robot,
        "current_adapter": {
            "path": case.target_adapter_path,
            "module": case.target_adapter_module,
        },
        "high_level_program": {
            "path": case.target_program_path,
            "fixed": True,
            "editable": False,
        },
        "constraints": _machine_constraints(case),
        "low_level_interface": _low_level_interface(case),
        "latest_results": {
            "multiseed": _compact_multiseed_summary(summary),
            "structured_probe": _compact_probe_payload(probe_payload),
        },
        "allowed_tools": tools,
    }
    human_report = {
        "schema": "human_harness_report.v1",
        "case_id": case.case_id,
        "summary": _compact_multiseed_summary(summary),
        "selected_failure_seed": seed_selection,
        "suggested_next_action": suggested_action,
        "note": (
            "This section is for the researcher. Do not feed it to the LLM agent "
            "when evaluating autonomous behavior."
        ),
    }
    return {
        "schema": "autonomous_harness_bundle.v1",
        "agent_observation": agent_observation,
        "human_report": human_report,
    }


def write_harness_plan(output_dir: Path | str, bundle: Mapping[str, Any]) -> Dict[str, str]:
    observation = bundle.get("agent_observation") or {}
    human_report = bundle.get("human_report") or {}
    out = Path(output_dir) / str(observation.get("case_id") or "unknown_case")
    out.mkdir(parents=True, exist_ok=True)
    observation_path = out / "agent_observation.json"
    human_json_path = out / "human_report.json"
    human_md_path = out / "human_report.md"
    bundle_path = out / "harness_bundle.json"
    observation_path.write_text(json.dumps(observation, indent=2, ensure_ascii=False), encoding="utf-8")
    human_json_path.write_text(json.dumps(human_report, indent=2, ensure_ascii=False), encoding="utf-8")
    human_md_path.write_text(_harness_markdown(bundle), encoding="utf-8")
    bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "agent_observation": str(observation_path),
        "human_report_json": str(human_json_path),
        "human_report_markdown": str(human_md_path),
        "bundle": str(bundle_path),
    }


def _next_action(
    case: FullMigrationCase,
    *,
    summary: Mapping[str, Any] | None,
    probe_payload: Mapping[str, Any] | None,
    selected_seed: int,
    selected_diagnosis: Mapping[str, Any],
    sim_backend: str,
    render_backend: str,
    max_episode_steps: int,
) -> Dict[str, Any]:
    if not summary:
        return {
            "tool": "run_single_seed",
            "reason": "No latest simulator result was provided.",
            "command": _single_seed_command(
                case,
                seed=selected_seed,
                sim_backend=sim_backend,
                render_backend=render_backend,
                max_episode_steps=max_episode_steps,
            ),
        }
    strategy = summary.get("generalization_strategy") or {}
    if strategy.get("status") == "accepted":
        return {
            "tool": "stop_report",
            "reason": "The current adapter meets the configured multi-seed success threshold.",
            "command": "",
        }
    if not probe_payload:
        return {
            "tool": "run_structured_probe",
            "reason": "Generalization failed; run measured structured probing before asking the LLM to guess another adapter.",
            "command": _probe_command(
                case,
                seed=selected_seed,
                diagnosis=selected_diagnosis,
                sim_backend=sim_backend,
                render_backend=render_backend,
                max_episode_steps=max_episode_steps,
            ),
        }
    return {
        "tool": "run_llm_repair",
        "reason": "Structured probe evidence is available and will be injected into the module-generation prompt.",
        "command": _repair_command(case, sim_backend=sim_backend, render_backend=render_backend),
    }


def _single_seed_command(
    case: FullMigrationCase,
    *,
    seed: int,
    sim_backend: str,
    render_backend: str,
    max_episode_steps: int,
) -> str:
    return " ".join(
        [
            "python -m maniskill_backend.real_runner",
            "--task",
            shlex.quote(case.task_id),
            "--robot",
            shlex.quote(case.target_robot),
            "--method target-module-generation",
            "--seed",
            str(seed),
            "--control-mode",
            shlex.quote(case.target_control_mode),
            "--sim-backend",
            shlex.quote(sim_backend),
            "--render-backend",
            shlex.quote(render_backend),
            "--max-episode-steps",
            str(max_episode_steps),
            "--code-file",
            shlex.quote(case.target_program_path),
            "--adapter-module",
            shlex.quote(case.target_adapter_module),
        ]
    )


def _multi_seed_command(
    case: FullMigrationCase,
    *,
    seeds: str,
    sim_backend: str,
    render_backend: str,
    max_episode_steps: int,
) -> str:
    if case.task_id != "pull_cube":
        return "# multi-seed helper currently implemented for PullCube; use run_single_seed for this case"
    return " ".join(
        [
            "python scripts/pullcube_multiseed_eval.py",
            "--seeds",
            shlex.quote(seeds),
            "--robot",
            shlex.quote(case.target_robot),
            "--adapter-module",
            shlex.quote(case.target_adapter_module),
            "--code-file",
            shlex.quote(case.target_program_path),
            "--control-mode",
            shlex.quote(case.target_control_mode),
            "--sim-backend",
            shlex.quote(sim_backend),
            "--render-backend",
            shlex.quote(render_backend),
            "--max-episode-steps",
            str(max_episode_steps),
        ]
    )


def _probe_command(
    case: FullMigrationCase,
    *,
    seed: int,
    diagnosis: Mapping[str, Any],
    sim_backend: str,
    render_backend: str,
    max_episode_steps: int,
) -> str:
    diagnosis_text = json.dumps(diagnosis or {}, ensure_ascii=False)
    return " ".join(
        [
            "python scripts/structured_probe_runner.py",
            "--case",
            shlex.quote(case.case_id),
            "--seed",
            str(seed),
            "--failure-diagnosis-json",
            shlex.quote(diagnosis_text),
            "--sim-backend",
            shlex.quote(sim_backend),
            "--render-backend",
            shlex.quote(render_backend),
            "--max-episode-steps",
            str(max_episode_steps),
        ]
    )


def _repair_command(case: FullMigrationCase, *, sim_backend: str, render_backend: str) -> str:
    return " ".join(
        [
            "python -m maniskill_backend.module_generation_runner",
            "--case",
            shlex.quote(case.case_id),
            "--max-attempts 3",
            "--sim-backend",
            shlex.quote(sim_backend),
            "--render-backend",
            shlex.quote(render_backend),
        ]
    )


def _load_probe_for_case(
    case: FullMigrationCase,
    *,
    probe_json: Path | str | None,
    include_existing_probe: bool,
) -> Dict[str, Any] | None:
    if probe_json:
        return load_probe_payload(probe_json)
    if not include_existing_probe:
        return None
    try:
        spec = get_probe_spec(case)
    except KeyError:
        return None
    default_path = REPO_ROOT / "results" / "structured_probes" / case.case_id / f"{spec.probe_id}.json"
    if not default_path.exists():
        return None
    payload = load_probe_payload(default_path)
    if payload.get("dry_run"):
        return None
    return payload


def _trial_digest(trial: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "seed": trial.get("seed"),
        "success": bool(trial.get("success")),
        "failure_type": trial.get("failure_type"),
        "failure_layer": trial.get("failure_layer"),
        "message": trial.get("message"),
        "final_info": trial.get("final_info") or {},
        "runtime_diagnostics": trial.get("runtime_diagnostics") or {},
        "failure_diagnosis": trial.get("failure_diagnosis") or {},
    }


def _elapsed_steps(row: Mapping[str, Any]) -> int | None:
    raw = (row.get("final_info") or {}).get("elapsed_steps")
    if isinstance(raw, list) and raw:
        return int(raw[0])
    if isinstance(raw, (int, float)):
        return int(raw)
    return None


def _float_from_runtime(row: Mapping[str, Any], key: str, *, default: float) -> float:
    diagnostics = row.get("runtime_diagnostics") or {}
    try:
        return float(diagnostics.get(key))
    except (TypeError, ValueError):
        return default


def _diagnosis_from_selection(selection: Mapping[str, Any]) -> Dict[str, Any]:
    row = selection.get("row") or {}
    diagnosis = dict(row.get("failure_diagnosis") or {})
    if diagnosis:
        return diagnosis
    return {"reason": row.get("failure_type") or "unknown", "evidence": row.get("runtime_diagnostics") or {}}


def _compact_multiseed_summary(summary: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not summary:
        return {}
    strategy = summary.get("generalization_strategy") or {}
    rows = list(summary.get("rows") or [])
    failure_rows = [_machine_result_row(row) for row in rows if not bool(row.get("success"))]
    success_rows = [_machine_result_row(row) for row in rows if bool(row.get("success"))]
    return {
        "num_trials": summary.get("num_trials"),
        "num_success": summary.get("num_success"),
        "num_failure": summary.get("num_failure"),
        "success_rate": summary.get("success_rate"),
        "mean_elapsed_steps": summary.get("mean_elapsed_steps"),
        "success_rows": success_rows,
        "failure_rows": failure_rows,
        "dominant_failure_reason": strategy.get("dominant_failure_reason"),
        "dominant_failure_stage": strategy.get("dominant_failure_stage"),
        "selected_strategy": strategy.get("selected_strategy"),
        "failure_seed_clusters": strategy.get("failure_seed_clusters"),
    }


def _compact_probe_payload(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not payload:
        return {}
    best = payload.get("best_probe_case") or {}
    return {
        "probe_id": payload.get("probe_id"),
        "num_cases": payload.get("num_cases"),
        "num_success": payload.get("num_success"),
        "best_probe_case": best,
        "top_probe_cases": list(payload.get("top_probe_cases") or [])[:5],
        "prompt_feedback_path": ((payload.get("wrote") or {}).get("prompt_feedback")),
    }


def _machine_constraints(case: FullMigrationCase) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {
        "editable": ["target_adapter"],
        "frozen": ["controller", "simulator", "success_signal", "high_level_program"],
        "must_use_real_env_step": True,
        "direct_state_edit_allowed": False,
        "target_adapter_path": case.target_adapter_path,
    }
    if case.task_id == "pull_cube":
        constraints["fixed_program_call"] = "robot.pull(cube, goal)"
    if case.task_id == "pick_cube":
        constraints["fixed_program_call"] = "robot.grasp(cube); robot.place(cube, goal)"
    return constraints


def _low_level_interface(case: FullMigrationCase) -> Dict[str, Any]:
    interface: Dict[str, Any] = {
        "simulator": "ManiSkill",
        "execution_boundary": "real_runner creates env, adapter calls env.step(action)",
        "available_runtime_calls": [
            "env.reset(seed)",
            "env.step(action)",
            "env.action_space",
            "robot._tcp_pos()",
            "robot._actor_pos(name)",
            "robot._region_pos(name)",
            "robot._pull_cube_success() / robot._is_grasping(name) when provided by adapter base",
        ],
        "observable_diagnostics": [
            "success",
            "failure_type",
            "failure_layer",
            "message",
            "elapsed_steps",
            "tcp_pos",
            "cube_pos",
            "goal_pos",
            "stage_target_pos",
            "tcp_stage_error_norm",
            "tcp_cube_xy",
            "cube_goal_xy",
        ],
        "writable_artifacts": [
            "target_adapter module with build_robot(env, *, control_mode, robot_uid)",
            "experiment action plan JSON choosing allowed_tools",
        ],
    }
    if case.target_robot == "xarm6_robotiq":
        interface["known_action_layout"] = {
            "control_mode": case.target_control_mode,
            "action_space_shape": "(4,) for observed xarm6 pd_ee_delta_pos",
            "action_semantics": "action[0:3]=normalized TCP xyz delta, action[3]=active gripper command",
        }
    return interface


def _machine_result_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    diagnosis = dict(row.get("failure_diagnosis") or {})
    diagnosis.pop("repair_hint", None)
    return {
        "seed": row.get("seed"),
        "success": bool(row.get("success")),
        "failure_type": row.get("failure_type"),
        "failure_layer": row.get("failure_layer"),
        "message": row.get("message"),
        "final_info": row.get("final_info") or {},
        "runtime_diagnostics": row.get("runtime_diagnostics") or {},
        "failure_diagnosis": diagnosis,
    }


def _machine_seed_selection(selection: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(selection.get("row") or {})
    diagnosis = dict(row.get("failure_diagnosis") or {})
    diagnosis.pop("repair_hint", None)
    if diagnosis:
        row["failure_diagnosis"] = diagnosis
    return {
        "seed": selection.get("seed"),
        "policy": selection.get("policy"),
        "row": row,
    }


def _fixed_constraints(case: FullMigrationCase) -> List[str]:
    constraints = [
        "高层程序保持不变，只能通过 target-side adapter 迁移。",
        "不能修改 ManiSkill controller、simulator、任务 success signal。",
        "adapter 必须通过真实 env.step(action) 执行，不能直接改物体状态。",
        f"目标 adapter 文件限定为 {case.target_adapter_path}。",
    ]
    if case.task_id == "pull_cube":
        constraints.append("PullCube 的高层调用保持 robot.pull(cube, goal)。")
    if case.task_id == "pick_cube":
        constraints.append("PickCube 的高层调用保持 robot.grasp(cube); robot.place(cube, goal)。")
    return constraints


def _guardrails() -> List[str]:
    return [
        "Agent 只能运行 allowed_tools 里的命令或读取其输出文件。",
        "失败后先看 failure_diagnosis 和 structured probe，再要求 LLM 重写 adapter。",
        "如果多 seed 失败集中在同一物理约束，优先 probe 该约束，不继续无限 prompt。",
        "每轮都保留 adapter diff、result JSONL、probe table，方便复现实验。",
    ]


def _harness_markdown(bundle: Mapping[str, Any]) -> str:
    observation = bundle.get("agent_observation") or {}
    human_report = bundle.get("human_report") or {}
    next_action = human_report.get("suggested_next_action") or {}
    summary = human_report.get("summary") or {}
    lines = [
        "# Human Harness Report",
        "",
        f"- case_id: `{observation.get('case_id')}`",
        f"- task_id: `{observation.get('task_id')}`",
        f"- target robot: `{observation.get('target_robot')}`",
        f"- adapter: `{(observation.get('current_adapter') or {}).get('path')}`",
        "",
        "## Human Summary",
        "",
        f"- success rate: `{summary.get('success_rate')}`",
        f"- selected strategy: `{summary.get('selected_strategy')}`",
        f"- dominant failure reason: `{summary.get('dominant_failure_reason')}`",
        "",
        "## Suggested Next Action For Researcher",
        "",
        f"- tool: `{next_action.get('tool')}`",
        f"- reason: {next_action.get('reason')}",
        "",
        "```bash",
        str(next_action.get("command") or ""),
        "```",
        "",
        "## Agent Input Boundary",
        "",
        "The LLM agent should receive only `agent_observation.json`. This human report is for experiment tracking and should not be used as the agent prompt.",
        "",
    ]
    return "\n".join(lines)
