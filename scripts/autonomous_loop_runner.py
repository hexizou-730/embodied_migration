"""One-command autonomous simulation-in-the-loop repair loop.

Default use:

python auto.py pull

The loop gives the LLM planner only the bottom interface:
agent_observation.json, allowed tools, and raw simulator outputs. The planner
decides which tool to run next. The harness only validates and executes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.agent_planner import plan_agent_actions
from maniskill_backend.autonomous_harness import build_harness_plan, load_multiseed_jsonl, write_harness_plan
from maniskill_backend.cases import get_full_migration_case
from maniskill_backend.structured_probe import get_probe_spec


DEFAULT_CASE = "case02_pull_cube_panda_to_xarm6"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _run_command(command: List[str], *, cwd: Path, log_path: Path, dry_run: bool) -> int:
    printable = " ".join(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        if dry_run:
            log.write("[dry-run] command not executed\n")
            return 0
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return int(process.wait())


def _multiseed_command(args: argparse.Namespace, cycle_dir: Path) -> List[str]:
    return [
        sys.executable,
        "scripts/pullcube_multiseed_eval.py",
        "--seeds",
        args.seeds,
        "--robot",
        args.robot,
        "--adapter-module",
        args.adapter_module,
        "--code-file",
        args.code_file,
        "--control-mode",
        args.control_mode,
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--success-threshold",
        str(args.success_threshold),
        "--min-trials-for-accept",
        str(args.min_trials_for_accept),
        "--output-dir",
        str(cycle_dir),
        "--jsonl-name",
        "multiseed.jsonl",
        "--md-name",
        "multiseed.md",
    ]


def _probe_command(args: argparse.Namespace, seed: int, diagnosis: Mapping[str, Any]) -> List[str]:
    return [
        sys.executable,
        "scripts/structured_probe_runner.py",
        "--case",
        args.case,
        "--seed",
        str(seed),
        "--failure-diagnosis-json",
        json.dumps(diagnosis, ensure_ascii=False),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--top-k",
        str(args.top_k),
        "--max-cases",
        str(args.max_probe_cases),
    ]


def _module_generation_command(args: argparse.Namespace, cycle_dir: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "maniskill_backend.module_generation_runner",
        "--case",
        args.case,
        "--max-attempts",
        str(args.max_attempts),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--jsonl",
        str(cycle_dir / "module_generation.jsonl"),
        "--md",
        str(cycle_dir / "module_generation.md"),
    ]


def _copy_probe_outputs(case_id: str, cycle_dir: Path) -> Dict[str, str]:
    try:
        case = get_full_migration_case(case_id)
        spec = get_probe_spec(case)
    except Exception:
        return {}
    src_dir = REPO_ROOT / "results" / "structured_probes" / case_id
    if not src_dir.exists():
        return {}
    dst_dir = cycle_dir / "structured_probe"
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for suffix in (".json", ".md", "_prompt.txt"):
        name = f"{spec.probe_id}{suffix}"
        src = src_dir / name
        if src.exists():
            dst = dst_dir / name
            shutil.copy2(src, dst)
            copied[suffix] = str(dst.relative_to(REPO_ROOT))
    return copied


def _read_module_generation_success(path: Path) -> bool:
    if not path.exists():
        return False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    try:
        return bool(json.loads(lines[-1]).get("success"))
    except Exception:
        return False


def _write_summary(run_dir: Path, payload: Mapping[str, Any]) -> None:
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_lines = [
        "# Auto Run Summary",
        "",
        f"- case: `{payload.get('case')}`",
        f"- status: `{payload.get('status')}`",
        f"- cycles: `{len(payload.get('cycles') or [])}`",
        f"- run_dir: `{run_dir}`",
        "",
        "## Cycles",
        "",
        "| cycle | success_rate | status | next |",
        "|---:|---:|---|---|",
    ]
    for item in payload.get("cycles") or []:
        md_lines.append(
            f"| {item.get('cycle')} | {item.get('success_rate')} | {item.get('status')} | {item.get('next_step')} |"
        )
    md_lines.append("")
    (run_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")


def _probe_json_path_for_case(case_id: str, cycle_dir: Path) -> Path | None:
    try:
        case = get_full_migration_case(case_id)
        spec = get_probe_spec(case)
    except Exception:
        return None
    candidate = cycle_dir / "structured_probe" / f"{spec.probe_id}.json"
    return candidate if candidate.exists() else None


def _action_seed(action: Mapping[str, Any], harness_bundle: Mapping[str, Any], default_seed: int) -> int:
    args = action.get("args") or {}
    if isinstance(args, Mapping) and args.get("seed") is not None:
        try:
            return int(args.get("seed"))
        except (TypeError, ValueError):
            pass
    selected = (harness_bundle.get("human_report") or {}).get("selected_failure_seed") or {}
    if selected.get("seed") is not None:
        try:
            return int(selected.get("seed"))
        except (TypeError, ValueError):
            pass
    return int(default_seed)


def run_loop(args: argparse.Namespace) -> Dict[str, Any]:
    case = get_full_migration_case(args.case)
    if case.task_id != "pull_cube":
        raise ValueError("The one-command loop currently supports PullCube only. Use the lower-level runners for PickCube.")

    run_dir = REPO_ROOT / args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.output_root / "latest.txt").write_text(str(run_dir), encoding="utf-8")
    command_log = run_dir / "commands.log"
    cycles: List[Dict[str, Any]] = []
    latest_multiseed_jsonl: Path | None = None
    latest_probe_json: Path | None = None

    for cycle in range(1, args.max_cycles + 1):
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        harness_bundle = build_harness_plan(
            case_id=args.case,
            multiseed_jsonl=latest_multiseed_jsonl,
            probe_json=latest_probe_json,
            include_existing_probe=False,
            seed_policy=args.seed_policy,
            seeds=args.seeds,
            sim_backend=args.sim_backend,
            render_backend=args.render_backend,
            max_episode_steps=args.max_episode_steps,
        )
        harness_wrote = write_harness_plan(cycle_dir / "harness", harness_bundle)
        observation = harness_bundle.get("agent_observation") or {}
        plan = plan_agent_actions(
            observation,
            max_actions=1,
            dry_run=args.dry_run or args.planner == "fallback",
        )
        (cycle_dir / "agent_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        action = (plan.get("actions") or [{"tool": "stop", "args": {"reason": "empty_plan"}}])[0]
        tool = str(action.get("tool") or "stop")

        summary: Dict[str, Any] = {}
        success_rate = None
        if latest_multiseed_jsonl and latest_multiseed_jsonl.exists():
            summary = load_multiseed_jsonl(latest_multiseed_jsonl)
            success_rate = summary.get("success_rate")

        print(f"\n[auto] cycle {cycle}/{args.max_cycles}: planner selected {tool}")
        if tool == "stop":
            cycles.append(
                {
                    "cycle": cycle,
                    "status": "stopped_by_agent",
                    "success_rate": success_rate,
                    "next_step": "stop",
                    "action": action,
                    "plan": _display_path(cycle_dir / "agent_plan.json"),
                    "harness": harness_wrote,
                }
            )
            break

        if tool == "run_multi_seed":
            rc = _run_command(_multiseed_command(args, cycle_dir), cwd=REPO_ROOT, log_path=command_log, dry_run=args.dry_run)
            latest_multiseed_jsonl = cycle_dir / "multiseed.jsonl"
            if args.dry_run:
                cycles.append(
                    {
                        "cycle": cycle,
                        "status": "dry_run_planned",
                        "success_rate": None,
                        "next_step": "would_run_multi_seed",
                        "action": action,
                        "plan": _display_path(cycle_dir / "agent_plan.json"),
                        "harness": harness_wrote,
                    }
                )
                break
            if rc != 0:
                cycles.append({"cycle": cycle, "status": "multiseed_failed", "success_rate": None, "next_step": "stop", "action": action})
                break
            summary = load_multiseed_jsonl(latest_multiseed_jsonl)
            success_rate = summary.get("success_rate")
            status = "accepted" if (summary.get("generalization_strategy") or {}).get("status") == "accepted" else "tool_done"
            cycles.append(
                {
                    "cycle": cycle,
                    "status": status,
                    "success_rate": success_rate,
                    "next_step": "stop" if status == "accepted" else "planner_decides",
                    "action": action,
                    "plan": _display_path(cycle_dir / "agent_plan.json"),
                    "harness": harness_wrote,
                }
            )
            if status == "accepted":
                break

        elif tool == "run_structured_probe":
            seed = _action_seed(action, harness_bundle, case.seed)
            selected = (harness_bundle.get("human_report") or {}).get("selected_failure_seed") or {}
            diagnosis = ((selected.get("row") or {}).get("failure_diagnosis") or {})
            rc = _run_command(_probe_command(args, seed, diagnosis), cwd=REPO_ROOT, log_path=command_log, dry_run=args.dry_run)
            copied_probe = _copy_probe_outputs(args.case, cycle_dir)
            latest_probe_json = _probe_json_path_for_case(args.case, cycle_dir)
            cycles.append(
                {
                    "cycle": cycle,
                    "status": "tool_done" if rc == 0 else "probe_failed",
                    "success_rate": success_rate,
                    "next_step": "planner_decides" if rc == 0 else "stop",
                    "action": action,
                    "plan": _display_path(cycle_dir / "agent_plan.json"),
                    "harness": harness_wrote,
                    "probe": copied_probe,
                }
            )
            if rc != 0:
                break

        elif tool == "run_llm_repair":
            rc = _run_command(_module_generation_command(args, cycle_dir), cwd=REPO_ROOT, log_path=command_log, dry_run=args.dry_run)
            module_success = _read_module_generation_success(cycle_dir / "module_generation.jsonl")
            latest_multiseed_jsonl = None
            latest_probe_json = None
            cycles.append(
                {
                    "cycle": cycle,
                    "status": "tool_done" if rc == 0 else "module_generation_failed",
                    "success_rate": success_rate,
                    "next_step": "planner_decides" if rc == 0 else "stop",
                    "action": action,
                    "plan": _display_path(cycle_dir / "agent_plan.json"),
                    "module_generation_success": module_success,
                    "harness": harness_wrote,
                }
            )
            if rc != 0:
                break

        else:
            cycles.append(
                {
                    "cycle": cycle,
                    "status": "unsupported_tool",
                    "success_rate": success_rate,
                    "next_step": "stop",
                    "action": action,
                    "plan": _display_path(cycle_dir / "agent_plan.json"),
                    "harness": harness_wrote,
                }
            )
            break

    status = "accepted" if any(item.get("status") == "accepted" for item in cycles) else "cycle_budget_exhausted"
    payload = {
        "schema": "autonomous_loop_result.v1",
        "case": args.case,
        "run_dir": str(run_dir),
        "status": status,
        "cycles": cycles,
        "dry_run": bool(args.dry_run),
    }
    _write_summary(run_dir, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--min-trials-for-accept", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-probe-cases", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed-policy", choices=("auto", "near_contact", "severe_reachability", "first"), default="auto")
    parser.add_argument("--planner", choices=("llm", "fallback"), default="llm")
    parser.add_argument("--robot", default="xarm6_robotiq")
    parser.add_argument("--adapter-module", default="maniskill_backend.generated_adapters.case02_xarm6_pull_cube")
    parser.add_argument("--code-file", default="maniskill_backend/case_programs/case01_pull_cube.py")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--output-root", default="results/auto_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.run_name:
        args.run_name = f"pull_{_timestamp()}"
    run_loop(args)


if __name__ == "__main__":
    main()
