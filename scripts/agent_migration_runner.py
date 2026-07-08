"""Generic autonomous migration loop.

This is the user-facing "agent" layer: start from task/source/target, restore a
neutral seed adapter, ask the LLM to generate a target adapter, run real
simulation, optionally run structured probes, and repeat until success or budget
exhaustion.
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

from maniskill_backend.cases import FullMigrationCase, find_full_migration_case, get_full_migration_case
from maniskill_backend.structured_probe import get_probe_spec


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(text)).strip("_")


def _read_latest_jsonl(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=repr), encoding="utf-8")


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


def _restore_seed_adapter(case: FullMigrationCase, run_dir: Path, *, dry_run: bool) -> Dict[str, Any]:
    seed_path = REPO_ROOT / case.seed_adapter_path if case.seed_adapter_path else None
    target_path = REPO_ROOT / case.target_adapter_path
    backup_path = run_dir / "initial_adapter_backup.py"
    details: Dict[str, Any] = {
        "target_adapter_path": case.target_adapter_path,
        "seed_adapter_path": case.seed_adapter_path,
        "restored": False,
    }
    if not seed_path:
        details["message"] = "case has no seed adapter path; current adapter kept"
        return details
    if not seed_path.exists():
        raise FileNotFoundError(f"seed adapter does not exist: {seed_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not backup_path.exists():
        shutil.copy2(target_path, backup_path)
    details["backup_path"] = str(backup_path)
    if dry_run:
        details["message"] = "dry run: seed adapter would be restored"
        return details
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_path, target_path)
    details["restored"] = True
    details["message"] = "seed adapter restored for from-zero migration"
    return details


def _module_generation_command(args: argparse.Namespace, case: FullMigrationCase, cycle_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "maniskill_backend.module_generation_runner",
        "--case",
        case.case_id,
        "--max-attempts",
        str(args.attempts_per_cycle),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--jsonl",
        str(cycle_dir / "module_generation.jsonl"),
        "--md",
        str(cycle_dir / "module_generation.md"),
    ]
    if args.no_source_check:
        command.append("--no-source-check")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _probe_command(args: argparse.Namespace, case: FullMigrationCase, result: Mapping[str, Any]) -> List[str]:
    final_target = result.get("final_target_result") or {}
    diagnosis = final_target.get("failure_diagnosis") or {}
    return [
        sys.executable,
        "scripts/structured_probe_runner.py",
        "--case",
        case.case_id,
        "--seed",
        str(args.seed),
        "--failure-diagnosis-json",
        json.dumps(diagnosis, ensure_ascii=False),
        "--sim-backend",
        args.sim_backend,
        "--render-backend",
        args.render_backend,
        "--max-episode-steps",
        str(args.max_episode_steps or case.max_episode_steps),
        "--max-cases",
        str(args.max_probe_cases),
        "--top-k",
        str(args.top_k),
        "--output-dir",
        "results/structured_probes",
    ]


def _has_structured_probe(case: FullMigrationCase) -> bool:
    try:
        get_probe_spec(case)
        return True
    except Exception:
        return False


def resolve_case(args: argparse.Namespace) -> FullMigrationCase:
    if args.case:
        return get_full_migration_case(args.case)
    return find_full_migration_case(args.task, args.source, args.target)


def run_agent_migration(args: argparse.Namespace) -> Dict[str, Any]:
    case = resolve_case(args)
    run_name = args.run_name or (
        f"{_safe_name(case.task_id)}_{_safe_name(case.source_robot)}_to_"
        f"{_safe_name(case.target_robot)}_agent_{_timestamp()}"
    )
    run_dir = REPO_ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.output_root / "latest.txt").write_text(str(run_dir), encoding="utf-8")
    command_log = run_dir / "commands.log"

    seed_restore: Dict[str, Any] = {"restored": False, "message": "current adapter kept"}
    if args.from_zero:
        seed_restore = _restore_seed_adapter(case, run_dir, dry_run=args.dry_run)

    cycles: List[Dict[str, Any]] = []
    status = "cycle_budget_exhausted"
    latest_result: Dict[str, Any] = {}
    for cycle in range(1, args.max_cycles + 1):
        cycle_dir = run_dir / f"cycle_{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        generation_jsonl = cycle_dir / "module_generation.jsonl"
        generation_rc = _run_command(
            _module_generation_command(args, case, cycle_dir),
            cwd=REPO_ROOT,
            log_path=command_log,
            dry_run=args.dry_run,
        )
        latest_result = _read_latest_jsonl(generation_jsonl)
        cycle_record: Dict[str, Any] = {
            "cycle": cycle,
            "module_generation_returncode": generation_rc,
            "module_generation_jsonl": str(generation_jsonl.relative_to(REPO_ROOT)),
            "module_generation_md": str((cycle_dir / "module_generation.md").relative_to(REPO_ROOT)),
            "target_success": latest_result.get("success"),
            "message": latest_result.get("message"),
        }
        if args.dry_run:
            cycle_record["status"] = "dry_run_planned"
            cycles.append(cycle_record)
            status = "dry_run_planned"
            break
        if generation_rc != 0:
            cycle_record["status"] = "module_generation_failed"
            cycles.append(cycle_record)
            status = "command_failed"
            break
        if bool(latest_result.get("success")):
            cycle_record["status"] = "success"
            cycles.append(cycle_record)
            status = "success"
            break

        if cycle < args.max_cycles and _has_structured_probe(case):
            probe_rc = _run_command(
                _probe_command(args, case, latest_result),
                cwd=REPO_ROOT,
                log_path=command_log,
                dry_run=args.dry_run,
            )
            cycle_record["structured_probe_returncode"] = probe_rc
            cycle_record["structured_probe_used"] = True
            if probe_rc != 0:
                cycle_record["status"] = "probe_failed"
                cycles.append(cycle_record)
                status = "command_failed"
                break
        else:
            cycle_record["structured_probe_used"] = False
        cycle_record["status"] = "retry_with_updated_observation"
        cycles.append(cycle_record)

    payload = {
        "schema": "agent_migration_result.v1",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "source_robot": case.source_robot,
        "target_robot": case.target_robot,
        "run_dir": str(run_dir),
        "from_zero": bool(args.from_zero),
        "seed_restore": seed_restore,
        "status": status,
        "success": status == "success",
        "cycles": cycles,
        "latest_result": latest_result,
    }
    _write_json(run_dir / "summary.json", payload)
    (run_dir / "summary.md").write_text(_summary_md(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=repr))
    return payload


def _summary_md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Agent Migration Run",
        "",
        f"- case: `{payload.get('case_id')}`",
        f"- task: `{payload.get('task_id')}`",
        f"- source -> target: `{payload.get('source_robot')} -> {payload.get('target_robot')}`",
        f"- status: `{payload.get('status')}`",
        f"- success: `{payload.get('success')}`",
        f"- from_zero: `{payload.get('from_zero')}`",
        "",
        "## Cycles",
        "",
        "| cycle | status | target_success | message | probe |",
        "|---:|---|---|---|---|",
    ]
    for item in payload.get("cycles") or []:
        message = str(item.get("message") or "").replace("|", "\\|")
        lines.append(
            f"| {item.get('cycle')} | {item.get('status')} | {item.get('target_success')} | "
            f"{message} | {item.get('structured_probe_used')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="", help="Optional exact migration case id.")
    parser.add_argument("--task", default="pull_cube")
    parser.add_argument("--source", default="panda")
    parser.add_argument("--target", default="xarm6_robotiq")
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--attempts-per-cycle", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--max-probe-cases", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-root", default="results/agent_migrations")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--from-zero", action="store_true", default=True)
    parser.add_argument("--keep-current-adapter", action="store_false", dest="from_zero")
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_agent_migration(args)
    if not args.dry_run and not bool(result.get("success")):
        sys.exit(1)


if __name__ == "__main__":
    main()
