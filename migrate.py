"""User-facing migration entrypoint.

Minimal success case:

    python migrate.py --task pull_cube --source panda --target xarm6_robotiq

Dry-run without ManiSkill:

    python migrate.py --task PullCube-v1 --source panda --target xarm6 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from maniskill_backend.cases import find_full_migration_case, iter_full_migration_cases
from maniskill_backend.real_runner import run_real_code_trial


REPO_ROOT = Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text).strip("_")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")


def _evaluate_current_adapter(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case = find_full_migration_case(args.task, args.source, args.target)
    code = (REPO_ROOT / case.target_program_path).read_text(encoding="utf-8")
    if args.dry_run:
        return {
            "dry_run": True,
            "would_run": _real_runner_command(case, seed=args.seed),
            "success": None,
            "message": "dry run: current migrated adapter would be evaluated once",
        }
    result = run_real_code_trial(
        task_id=case.task_id,
        robot_uid=case.target_robot,
        method="target-module-generation",
        code=code,
        prompt=f"migrate.py current adapter evaluation for {case.case_id}",
        seed=args.seed,
        control_mode=case.target_control_mode,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps or case.max_episode_steps,
        adapter_module=case.target_adapter_module,
    )
    _write_json(run_dir / "trial_result.json", result)
    return {
        "dry_run": False,
        "success": bool(result.get("success")),
        "message": result.get("message"),
        "trial_result": "trial_result.json",
    }


def _run_generate(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case = find_full_migration_case(args.task, args.source, args.target)
    command = [
        sys.executable,
        "-m",
        "maniskill_backend.module_generation_runner",
        "--case",
        case.case_id,
        "--max-attempts",
        str(args.max_attempts),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--jsonl",
        str(run_dir / "module_generation.jsonl"),
        "--md",
        str(run_dir / "module_generation.md"),
    ]
    if args.dry_run:
        return {
            "dry_run": True,
            "would_run": " ".join(shlex.quote(part) for part in command),
            "success": None,
            "message": "dry run: LLM adapter generation would be launched",
        }
    stdout_path = run_dir / "module_generation_stdout.txt"
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    return {
        "dry_run": False,
        "success": process.returncode == 0,
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "message": "LLM adapter generation finished" if process.returncode == 0 else "LLM adapter generation failed",
    }


def _run_auto(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case = find_full_migration_case(args.task, args.source, args.target)
    if case.task_id != "pull_cube":
        raise ValueError("auto mode currently supports the PullCube harness only.")
    command = [
        sys.executable,
        "auto.py",
        "pull",
        "--case",
        case.case_id,
        "--seeds",
        args.seeds,
        "--max-cycles",
        str(args.max_cycles),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--run-name",
        run_dir.name + "_auto",
    ]
    if args.dry_run:
        command.append("--dry-run")
    stdout_path = run_dir / "auto_stdout.txt"
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    return {
        "dry_run": bool(args.dry_run),
        "success": process.returncode == 0,
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "message": "auto harness loop finished" if process.returncode == 0 else "auto harness loop failed",
    }


def _run_agent(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case = find_full_migration_case(args.task, args.source, args.target)
    command = [
        sys.executable,
        "scripts/agent_migration_runner.py",
        "--case",
        case.case_id,
        "--max-cycles",
        str(args.max_cycles),
        "--attempts-per-cycle",
        str(args.attempts_per_cycle),
        "--seed",
        str(args.seed),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--output-root",
        str(run_dir),
        "--run-name",
        "agent_loop",
    ]
    if args.keep_current_adapter:
        command.append("--keep-current-adapter")
    if args.no_source_check:
        command.append("--no-source-check")
    if args.dry_run:
        command.append("--dry-run")
    stdout_path = run_dir / "agent_stdout.txt"
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    summary_path = run_dir / "agent_loop" / "summary.json"
    agent_summary: dict[str, Any] = {}
    if summary_path.exists():
        agent_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "dry_run": bool(args.dry_run),
        "success": bool(agent_summary.get("success")) if agent_summary else process.returncode == 0,
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "agent_summary": str(summary_path.relative_to(run_dir)) if summary_path.exists() else "",
        "message": (
            "agent migration loop reached success"
            if agent_summary.get("success")
            else "agent migration loop finished without success"
            if process.returncode == 0
            else "agent migration loop failed"
        ),
    }


def _run_online(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case = find_full_migration_case(args.task, args.source, args.target)
    command = [
        sys.executable,
        "scripts/online_harness_runner.py",
        "--case",
        case.case_id,
        "--seed",
        str(args.seed),
        "--planner",
        args.online_planner,
        "--segment-steps",
        str(args.segment_steps),
        "--max-online-steps",
        str(args.max_online_steps),
        "--obs-mode",
        args.obs_mode,
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--output-root",
        str(run_dir),
        "--run-name",
        "online_loop",
    ]
    if args.dry_run:
        command.append("--dry-run")
    stdout_path = run_dir / "online_stdout.txt"
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    summary_path = run_dir / "online_loop" / "summary.json"
    online_summary: dict[str, Any] = {}
    if summary_path.exists():
        online_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.dry_run:
        success: bool | None = None
        message = "dry run: online harness would observe and act in short closed-loop segments"
    else:
        success = bool(online_summary.get("success")) if online_summary else process.returncode == 0
        message = (
            "online harness reached success"
            if online_summary.get("success")
            else "online harness finished without success"
            if process.returncode == 0
            else "online harness failed"
        )
    return {
        "dry_run": bool(args.dry_run),
        "success": success,
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "online_summary": str(summary_path.relative_to(run_dir)) if summary_path.exists() else "",
        "online_trace": "online_loop/online_trace.jsonl" if summary_path.exists() else "",
        "message": message,
    }


def _real_runner_command(case: Any, *, seed: int) -> str:
    parts = [
        "python",
        "-m",
        "maniskill_backend.real_runner",
        "--task",
        case.task_id,
        "--robot",
        case.target_robot,
        "--method",
        "target-module-generation",
        "--seed",
        str(seed),
        "--control-mode",
        case.target_control_mode,
        "--sim-backend",
        "auto",
        "--render-backend",
        "gpu",
        "--max-episode-steps",
        str(case.max_episode_steps),
        "--code-file",
        case.target_program_path,
        "--adapter-module",
        case.target_adapter_module,
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _write_readme(run_dir: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Migration Run",
        "",
        f"- task: `{payload.get('task')}`",
        f"- source: `{payload.get('source')}`",
        f"- target: `{payload.get('target')}`",
        f"- mode: `{payload.get('mode')}`",
        f"- case: `{(payload.get('case') or {}).get('case_id')}`",
        f"- success: `{(payload.get('result') or {}).get('success')}`",
        f"- message: `{(payload.get('result') or {}).get('message')}`",
        "",
        "## Meaning",
        "",
        "This run starts from a user-level migration request:",
        "",
        "```text",
        "task + source robot + target robot",
        "```",
        "",
        "The harness resolves the registered migration case and executes the selected mode.",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _list_cases() -> None:
    for case in iter_full_migration_cases():
        print(f"{case.task_id}: {case.source_robot} -> {case.target_robot}  ({case.case_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a user-facing robot code migration request.")
    parser.add_argument("--task", default="pull_cube", help="Task id, e.g. pull_cube or PullCube-v1.")
    parser.add_argument("--source", default="panda", help="Source robot, e.g. panda.")
    parser.add_argument("--target", default="xarm6_robotiq", help="Target robot, e.g. xarm6_robotiq or xarm6.")
    parser.add_argument("--mode", choices=("evaluate", "generate", "auto", "agent", "online"), default="evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--attempts-per-cycle", type=int, default=1)
    parser.add_argument("--online-planner", choices=("fallback", "llm"), default="fallback")
    parser.add_argument("--segment-steps", type=int, default=8)
    parser.add_argument("--max-online-steps", type=int, default=240)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--output-root", default="results/migrations")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--keep-current-adapter", action="store_true")
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()

    if args.list_cases:
        _list_cases()
        return

    case = find_full_migration_case(args.task, args.source, args.target)
    run_name = args.run_name or f"{_safe_name(case.task_id)}_{_safe_name(case.source_robot)}_to_{_safe_name(case.target_robot)}_{_timestamp()}"
    run_dir = REPO_ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = REPO_ROOT / args.output_root / "latest.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")

    if args.mode == "agent":
        result = _run_agent(args, run_dir)
    elif args.mode == "online":
        result = _run_online(args, run_dir)
    elif args.mode == "generate":
        result = _run_generate(args, run_dir)
    elif args.mode == "auto":
        result = _run_auto(args, run_dir)
    else:
        result = _evaluate_current_adapter(args, run_dir)

    payload = {
        "schema": "migration_request_result.v1",
        "task": args.task,
        "source": args.source,
        "target": args.target,
        "mode": args.mode,
        "run_dir": str(run_dir),
        "case": {
            "case_id": case.case_id,
            "task_id": case.task_id,
            "source_robot": case.source_robot,
            "target_robot": case.target_robot,
            "target_adapter_module": case.target_adapter_module,
            "target_program_path": case.target_program_path,
        },
        "result": result,
    }
    _write_json(run_dir / "summary.json", payload)
    _write_readme(run_dir, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
    if not args.dry_run and result.get("success") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
