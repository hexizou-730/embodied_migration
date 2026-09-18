"""Batch target migration from immutable, multi-seed-validated source programs."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dynamic_harness import (
    DynamicMigrationSpec,
    discover_environment,
    read_source_actions,
    run_dynamic_agent_migration,
)
from .experiment_environment import capture_experiment_environment
from .source_preparation import ensure_assets, run_worker
from .source_programs import (
    REPO_ROOT,
    SourceProgramSpec,
    _read_json,
    _sha256_file,
    _write_json,
    load_source_program_catalog,
    source_program_status,
)
from .task_catalog import load_task_catalog


def frozen_source_specs(
    specs: Iterable[SourceProgramSpec], selection: str = "frozen"
) -> list[SourceProgramSpec]:
    """Select only sources whose code and ten-seed evidence still match."""

    specs = list(specs)
    status = {row["task_id"]: row for row in source_program_status(specs)}
    if selection.strip().lower() == "frozen":
        selected = [spec for spec in specs if status[spec.task_id]["frozen"]]
    else:
        requested = [item.strip() for item in selection.split(",") if item.strip()]
        by_id = {spec.task_id: spec for spec in specs}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError("Unknown task IDs: " + ", ".join(unknown))
        selected = [by_id[task_id] for task_id in requested]
    invalid = [spec.task_id for spec in selected if not status[spec.task_id]["frozen"]]
    if invalid:
        raise ValueError("Target migration accepts only valid frozen sources: " + ", ".join(invalid))
    if not selected:
        raise ValueError("No valid frozen source programs are available.")
    return selected


def migration_plan(args: Any, specs: list[SourceProgramSpec], target_robot: str) -> dict[str, Any]:
    return {
        "schema": "target_migration_plan.v1",
        "task_ids": [spec.task_id for spec in specs],
        "target_robot": target_robot,
        "target_control_mode": args.target_control_mode,
        "seed": args.seed,
        "max_cycles": args.max_cycles,
        "max_episode_steps": args.max_episode_steps,
        "task_timeout": args.task_timeout,
        "asset_timeout": args.asset_timeout,
        "download_assets": not args.no_download,
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "contract_sha256": _sha256_file(REPO_ROOT / "experiment_config.json"),
        "source_hashes": {spec.task_id: _sha256_file(spec.frozen_path) for spec in specs},
        "runner_hashes": {
            name: _sha256_file(REPO_ROOT / name)
            for name in (
                "target.py",
                "maniskill_backend/target_preparation.py",
                "maniskill_backend/dynamic_harness.py",
                "maniskill_backend/dynamic_adapter.py",
                "maniskill_backend/code_validation.py",
            )
        },
    }


def launch_background(argv: list[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(REPO_ROOT / "target.py"), *argv],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json(output_dir / "launcher.json", {"pid": process.pid, "log": str(output_dir / "run.log")})
    return process.pid


def run_migrations(plan: dict[str, Any], output_dir: Path, *, retry_failed: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_ROOT / "results" / "target_migration" / ".migrate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another target migration batch is running in this repository.") from exc
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Target migration received signal {signum}")

        try:
            for sig in handlers:
                signal.signal(sig, interrupt)
            return _run_locked(plan, output_dir, retry_failed=retry_failed)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _run_locked(plan: dict[str, Any], output_dir: Path, *, retry_failed: bool) -> dict[str, Any]:
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and _read_json(plan_path) != plan:
        raise ValueError("Migration settings/code changed. Use a new --run-name to keep evidence separate.")
    _write_json(plan_path, plan)
    _, all_specs = load_source_program_catalog()
    specs = [spec for spec in all_specs if spec.task_id in plan["task_ids"]]
    summary_path = output_dir / "summary.json"
    previous = _read_json(summary_path).get("results", []) if summary_path.exists() else []
    rows = {row["task_id"]: row for row in previous}
    summary: dict[str, Any] = {
        "schema": "target_migration_batch.v1",
        "target_robot": plan["target_robot"],
        "tasks": len(specs),
        "pid": os.getpid(),
        "state": "running",
        "current_task": None,
    }

    def checkpoint() -> None:
        ordered = [rows[spec.task_id] for spec in specs if spec.task_id in rows]
        metrics = _aggregate_metrics(ordered)
        summary.update(
            results=ordered,
            completed=sum(row.get("status") != "running" for row in ordered),
            succeeded=sum(bool(row.get("success")) for row in ordered),
            metrics=metrics,
            complete=len(ordered) == len(specs) and all(row.get("status") != "running" for row in ordered),
            updated_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(summary_path, summary)
        _write_markdown(output_dir / "summary.md", summary)

    runtime = capture_experiment_environment(
        {
            "phase": "target_migration",
            "source_robot": specs[0].source_robot,
            "target_robot": plan["target_robot"],
            "seed": plan["seed"],
        }
    )
    _write_json(output_dir / "runtime_environment.json", runtime)
    if not runtime["validation"]["ok"]:
        summary.update(state="environment_contract_mismatch", mismatches=runtime["validation"]["mismatches"])
        checkpoint()
        return summary

    checkpoint()
    try:
        for spec in specs:
            old = rows.get(spec.task_id, {})
            if old.get("success"):
                continue
            if old.get("status") not in {None, "running", "interrupted"} and not retry_failed:
                continue
            task_dir = output_dir / "tasks" / spec.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            attempts = sorted(task_dir.glob("attempt_*"))
            attempt_dir = task_dir / f"attempt_{len(attempts) + 1:03d}"
            config = {
                "task_id": spec.task_id,
                "plan": plan,
                "output_dir": str(attempt_dir.resolve()),
            }
            attempt_dir.mkdir(parents=True)
            _write_json(attempt_dir / "worker.json", config)
            rows[spec.task_id] = {"task_id": spec.task_id, "success": False, "status": "running"}
            summary["current_task"] = spec.task_id
            checkpoint()
            print(f"[{spec.task_id}] frozen source -> {plan['target_robot']}", flush=True)
            result_path = attempt_dir / "worker_result.json"
            with (attempt_dir / "worker.log").open("a", encoding="utf-8") as log:
                returncode, timed_out = run_worker(
                    [sys.executable, "-u", "-m", "maniskill_backend.target_preparation", str(attempt_dir / "worker.json")],
                    log=log,
                    timeout=plan["task_timeout"],
                )
            if result_path.exists() and returncode == 0:
                row = _read_json(result_path)
            else:
                row = {
                    "task_id": spec.task_id,
                    "success": False,
                    "status": "worker_timeout" if timed_out else "worker_crashed",
                    "returncode": returncode,
                }
            row["attempt"] = attempt_dir.name
            row["log"] = str(attempt_dir / "worker.log")
            rows[spec.task_id] = row
            summary["current_task"] = None
            checkpoint()
            print(f"  {row['status']}; success={row['success']}", flush=True)
    except BaseException:
        task_id = summary.get("current_task")
        if task_id:
            rows[task_id]["status"] = "interrupted"
        summary["state"] = "interrupted"
        checkpoint()
        raise
    summary["state"] = "complete"
    checkpoint()
    return summary


def migrate_task(config: dict[str, Any]) -> dict[str, Any]:
    _, specs = load_source_program_catalog()
    spec = next(spec for spec in specs if spec.task_id == config["task_id"])
    plan = config["plan"]
    output = Path(config["output_dir"])
    row: dict[str, Any] = {"task_id": spec.task_id, "success": False}
    try:
        status = source_program_status([spec])[0]
        if not status["frozen"]:
            return dict(row, status="source_not_frozen")
        source_record = _read_json(spec.frozen_path.with_suffix(".json"))
        if source_record["source_sha256"] != plan["source_hashes"][spec.task_id]:
            return dict(row, status="source_changed_after_plan")

        for robot_uid in (spec.source_robot, plan["target_robot"]):
            assets = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "maniskill_backend.target_preparation",
                    "assets",
                    spec.task_id,
                    "download" if plan["download_assets"] else "check",
                    robot_uid,
                ],
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                timeout=plan["asset_timeout"],
                check=False,
            )
            if assets.returncode:
                return dict(
                    row,
                    status="assets_blocked",
                    message=f"Asset check failed for {robot_uid}; see worker.log.",
                )
        migration_spec = DynamicMigrationSpec(
            env_id=spec.env_id,
            task_label=spec.task_id,
            source_robot=spec.source_robot,
            target_robot=plan["target_robot"],
            source_control_mode=spec.control_mode,
            target_control_mode=plan["target_control_mode"],
            seed=plan["seed"],
            max_episode_steps=plan["max_episode_steps"],
            max_source_cycles=0,
            max_target_cycles=plan["max_cycles"],
            obs_mode=plan["obs_mode"],
            sim_backend=plan["sim_backend"],
            render_backend=plan["render_backend"],
        )
        try:
            observation = discover_environment(
                migration_spec, plan["target_robot"], plan["target_control_mode"]
            )
        except Exception as exc:
            interface = {
                "schema": "target_interface.v1",
                "ok": False,
                "target_robot": plan["target_robot"],
                "control_mode": plan["target_control_mode"],
                "error": repr(exc),
            }
            _write_json(output / "target_interface.json", interface)
            return dict(row, status="target_interface_failed", message=repr(exc), interface=str(output / "target_interface.json"))

        task_catalog = load_task_catalog()
        task = next(item for item in task_catalog["tasks"] if item["task_id"] == spec.task_id)
        interface = {
            "schema": "target_interface.v1",
            "ok": True,
            "env_id": spec.env_id,
            "target_robot": plan["target_robot"],
            "control_mode": plan["target_control_mode"],
            "action_space": observation.get("action_space"),
            "controller_action_mapping": observation.get("controller_action_mapping"),
            "tcp": observation.get("tcp"),
            "entity_aliases": observation.get("entity_aliases"),
            "supported_robots": observation.get("supported_robots"),
            "required_capabilities": task["required_capabilities"],
        }
        _write_json(output / "target_interface.json", interface)

        source = read_source_actions(spec.frozen_path)
        migration_dir = output / "migration"
        result = run_dynamic_agent_migration(
            spec=migration_spec,
            run_dir=migration_dir,
            source_actions=source,
            source_provenance={
                "kind": "frozen_source_program.v2",
                "record": str(spec.frozen_path.with_suffix(".json")),
                "source_sha256": source_record["source_sha256"],
                "validated_seeds": source_record["validated_seeds"],
                "success_rate": source_record["success_rate"],
            },
            enforce_environment=True,
            progress=lambda message: print(message, flush=True),
        )
        return {
            **row,
            "success": bool(result.get("success")),
            "status": result.get("status", "unknown"),
            "message": result.get("message", ""),
            "interface": str(output / "target_interface.json"),
            "details": str(migration_dir / "dynamic_summary.json"),
            "target_adapter": result.get("target_adapter", ""),
            "metrics": result.get("metrics", {}),
        }
    except subprocess.TimeoutExpired:
        return dict(row, status="assets_timeout")
    except Exception as exc:
        return dict(row, status="worker_error", message=repr(exc))


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generation_cycles": sum(int((row.get("metrics") or {}).get("generation_cycles") or 0) for row in rows),
        "repair_cycles": sum(int((row.get("metrics") or {}).get("repair_cycles") or 0) for row in rows),
        "target_action_steps": sum(int((row.get("metrics") or {}).get("target_action_steps") or 0) for row in rows),
        "total_tokens": sum(int((row.get("metrics") or {}).get("total_tokens") or 0) for row in rows),
        "llm_cost_usd": round(sum(float((row.get("metrics") or {}).get("llm_cost_usd") or 0.0) for row in rows), 8),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Target Migration Status",
        "",
        f"Target: `{summary['target_robot']}`",
        "",
        "| task | status | success | cycles | repairs | target steps | tokens | cost USD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("results", []):
        metrics = row.get("metrics") or {}
        lines.append(
            f"| {row['task_id']} | {row.get('status')} | {bool(row.get('success'))} | "
            f"{metrics.get('generation_cycles', 0)} | {metrics.get('repair_cycles', 0)} | "
            f"{metrics.get('target_action_steps', 0)} | {metrics.get('total_tokens', 0)} | "
            f"{metrics.get('llm_cost_usd', 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "assets":
        ensure_assets(
            sys.argv[2],
            download=sys.argv[3] == "download",
            robot_uid=sys.argv[4],
        )
    else:
        configuration = _read_json(Path(sys.argv[1]))
        result = migrate_task(configuration)
        _write_json(Path(configuration["output_dir"]) / "worker_result.json", result)
