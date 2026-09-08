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
from maniskill_backend.dynamic_harness import (
    DynamicMigrationSpec,
    looks_like_env_id,
    run_dynamic_agent_migration,
)
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
        "--seed",
        str(args.seed),
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--obs-mode",
        args.obs_mode,
        "--force-regeneration",
        "--no-analysis",
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
    generation_path = run_dir / "module_generation.jsonl"
    generation = {}
    if generation_path.exists():
        lines = [line for line in generation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            generation = json.loads(lines[-1])
    return {
        "dry_run": False,
        "success": process.returncode == 0 and bool(generation.get("success")),
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "message": generation.get("message", "LLM adapter generation failed or produced no result"),
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
        "success": None if args.dry_run else bool(agent_summary.get("success")),
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "agent_summary": str(summary_path.relative_to(run_dir)) if summary_path.exists() else "",
        "message": (
            "dry run: source baseline and LLM migration planned; no simulation or API call"
            if args.dry_run
            else "agent migration loop reached success"
            if agent_summary.get("success")
            else "agent migration loop finished without success"
            if process.returncode == 0
            else "agent migration loop failed"
        ),
    }


def _run_dynamic_agent(args: argparse.Namespace, run_dir: Path, env_id: str) -> dict[str, Any]:
    spec = DynamicMigrationSpec(
        env_id=env_id,
        task_label=env_id,
        source_robot=_normalize_dynamic_robot(args.source),
        target_robot=_normalize_dynamic_robot(args.target),
        source_control_mode=args.source_control_mode,
        target_control_mode=args.target_control_mode,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps or 500,
        max_source_cycles=args.source_max_cycles,
        max_target_cycles=args.max_cycles,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    return run_dynamic_agent_migration(spec=spec, run_dir=run_dir, dry_run=args.dry_run)


def _normalize_dynamic_robot(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return {
        "franka": "panda",
        "franka_panda": "panda",
        "xarm6": "xarm6_robotiq",
    }.get(text, text)


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
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with stdout_path.open("w", encoding="utf-8") as stream:
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            if '"online_event"' in line:
                print(line.rstrip(), flush=True)
    returncode = int(process.wait())
    summary_path = run_dir / "online_loop" / "summary.json"
    online_summary: dict[str, Any] = {}
    if summary_path.exists():
        online_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.dry_run:
        success: bool | None = None
        message = "dry run: online harness would observe and act in short closed-loop segments"
    else:
        success = bool(online_summary.get("success")) if online_summary else returncode == 0
        message = (
            "online harness reached success"
            if online_summary.get("success")
            else "online harness finished without success"
            if returncode == 0
            else "online harness failed"
        )
    return {
        "dry_run": bool(args.dry_run),
        "success": success,
        "returncode": returncode,
        "stdout": str(stdout_path.relative_to(run_dir)),
        "online_summary": str(summary_path.relative_to(run_dir)) if summary_path.exists() else "",
        "online_trace": "online_loop/online_trace.jsonl" if summary_path.exists() else "",
        "message": message,
    }


def _run_cegis(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    case = find_full_migration_case(args.task, args.source, args.target)
    command = [
        sys.executable,
        "scripts/cegis_migration_runner.py",
        "--case",
        case.case_id,
        "--development-seeds",
        args.development_seeds,
        "--held-out-seeds",
        args.held_out_seeds,
        "--max-cycles",
        str(args.max_cycles),
        "--attempts-per-cycle",
        str(args.attempts_per_cycle),
        "--probe-budget",
        str(args.probe_budget),
        "--success-threshold",
        str(args.success_threshold),
        "--min-trials-for-accept",
        str(args.min_trials_for_accept),
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
        "cegis_loop",
    ]
    if args.keep_current_adapter:
        command.append("--keep-current-adapter")
    if args.no_source_check:
        command.append("--no-source-check")
    if args.dry_run:
        command.append("--dry-run")
    stdout_path = run_dir / "cegis_stdout.txt"
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    summary_path = run_dir / "cegis_loop" / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    success = None if args.dry_run else bool(summary.get("success"))
    return {
        "dry_run": bool(args.dry_run),
        "success": success,
        "returncode": int(process.returncode),
        "stdout": str(stdout_path.relative_to(run_dir)),
        "cegis_summary": str(summary_path.relative_to(run_dir)) if summary_path.exists() else "",
        "status": summary.get("status"),
        "message": (
            "dry run: CEGIS migration loop planned"
            if args.dry_run
            else "CEGIS migration accepted on held-out seeds"
            if summary.get("success")
            else "CEGIS migration finished without held-out acceptance"
        ),
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
        "Registered tasks use a frozen benchmark case. Unregistered ManiSkill envs use runtime discovery and a generated case manifest.",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _list_cases() -> None:
    for case in iter_full_migration_cases():
        print(f"{case.task_id}: {case.source_robot} -> {case.target_robot}  ({case.case_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a user-facing robot code migration request.")
    parser.add_argument("--task", default="pull_cube", help="Task id, e.g. pull_cube or PullCube-v1.")
    parser.add_argument(
        "--env-id",
        default="",
        help="ManiSkill environment id for discovery-first migration. Supplying it bypasses the registered case table.",
    )
    parser.add_argument("--source", default="panda", help="Source robot, e.g. panda.")
    parser.add_argument("--target", default="xarm6_robotiq", help="Target robot, e.g. xarm6_robotiq or xarm6.")
    parser.add_argument("--source-control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--target-control-mode", default="pd_ee_delta_pos")
    parser.add_argument(
        "--mode",
        choices=("evaluate", "generate", "auto", "agent", "online", "cegis"),
        default="evaluate",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--source-max-cycles", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--attempts-per-cycle", type=int, default=1)
    parser.add_argument("--online-planner", choices=("fallback", "llm"), default="llm")
    parser.add_argument("--segment-steps", type=int, default=8)
    parser.add_argument("--max-online-steps", type=int, default=360)
    parser.add_argument("--development-seeds", default="0-4")
    parser.add_argument("--held-out-seeds", default="100-109")
    parser.add_argument("--probe-budget", type=int, default=8)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--min-trials-for-accept", type=int, default=5)
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

    dynamic_env_id = str(args.env_id or "").strip()
    case = None
    if not dynamic_env_id:
        try:
            case = find_full_migration_case(args.task, args.source, args.target)
        except KeyError as exc:
            if args.mode == "agent" and looks_like_env_id(args.task):
                dynamic_env_id = args.task
            else:
                parser.error(
                    f"{exc} For an unregistered task, use --env-id <ManiSkillEnv-v1> with --mode agent."
                )

    if dynamic_env_id and args.mode != "agent":
        parser.error("Discovery-first unregistered tasks currently use --mode agent.")
    if case is not None and case.task_id == "stack_pyramid" and args.mode not in {"evaluate", "generate", "agent"}:
        parser.error("StackPyramid currently supports evaluate/generate/agent. Online tools and structured probes are not implemented for this task yet.")

    run_task_name = dynamic_env_id or (case.task_id if case is not None else args.task)
    run_source = _normalize_dynamic_robot(args.source) if dynamic_env_id else case.source_robot
    run_target = _normalize_dynamic_robot(args.target) if dynamic_env_id else case.target_robot
    run_name = args.run_name or f"{_safe_name(run_task_name)}_{_safe_name(run_source)}_to_{_safe_name(run_target)}_{_timestamp()}"
    run_dir = REPO_ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = REPO_ROOT / args.output_root / "latest.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")

    if dynamic_env_id:
        result = _run_dynamic_agent(args, run_dir, dynamic_env_id)
    elif args.mode == "cegis":
        result = _run_cegis(args, run_dir)
    elif args.mode == "agent":
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
        "case": (
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "source_robot": case.source_robot,
                "target_robot": case.target_robot,
                "target_adapter_module": case.target_adapter_module,
                "target_program_path": case.target_program_path,
                "registered": True,
            }
            if case is not None
            else {
                "case_id": "dynamic",
                "task_id": args.task,
                "env_id": dynamic_env_id,
                "source_robot": run_source,
                "target_robot": run_target,
                "manifest": result.get("manifest"),
                "registered": False,
            }
        ),
        "result": result,
    }
    _write_json(run_dir / "summary.json", payload)
    _write_readme(run_dir, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
    if not args.dry_run and result.get("success") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
