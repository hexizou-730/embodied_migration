"""Bounded, resumable source preparation. One isolated simulator worker per task."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .experiment_environment import capture_experiment_environment, load_experiment_contract
from .source_programs import (
    REPO_ROOT, _read_json, _sha256_file, _support_code_sha256, _write_json,
    freeze_source_program, load_source_program_catalog, source_program_status,
    synthesize_source_program, validate_source_program,
)


def use_contract_llm() -> None:
    """Pin public request settings for this process; never write or log API keys."""
    fields = {
        "provider": "EM_LLM_PROVIDER", "model": "EM_MODEL",
        "temperature": "EM_TEMPERATURE", "max_tokens": "EM_MAX_TOKENS",
        "reasoning_effort": "EM_REASONING_EFFORT",
        "exclude_reasoning_from_response": "EM_EXCLUDE_REASONING",
        "seed": "EM_LLM_SEED", "upstream_provider": "EM_OPENROUTER_PROVIDER",
        "allow_provider_fallbacks": "EM_OPENROUTER_ALLOW_FALLBACKS",
    }
    for field, variable in fields.items():
        os.environ[variable] = str(load_experiment_contract()["llm"][field])


def preparation_options(args, specs, seeds) -> dict:
    return {
        "schema": "source_preparation_plan.v1",
        "task_ids": [spec.task_id for spec in specs], "seeds": seeds,
        "max_cycles": args.max_cycles, "task_timeout": args.task_timeout,
        "asset_timeout": args.asset_timeout, "download_assets": not args.no_download,
        "obs_mode": args.obs_mode, "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "contract_sha256": _sha256_file(REPO_ROOT / "experiment_config.json"),
        "support_hashes": {spec.task_id: _support_code_sha256(spec) for spec in specs},
        "runner_hashes": {name: _sha256_file(REPO_ROOT / name) for name in (
            "source.py", "maniskill_backend/source_programs.py",
            "maniskill_backend/source_preparation.py", "maniskill_backend/code_validation.py",
        )},
    }


def launch_background(argv: list[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(REPO_ROOT / "source.py"), *argv],
            cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_json(output_dir / "launcher.json", {"pid": process.pid, "log": str(output_dir / "run.log")})
    return process.pid


def run_preparation(plan: dict, output_dir: Path, *, retry_failed: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = REPO_ROOT / "results" / "source_preparation" / ".prepare.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another source preparation is running in this repository.") from exc
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Source preparation received signal {signum}")
        try:
            for sig in handlers:
                signal.signal(sig, interrupt)
            return _run_locked(plan, output_dir, retry_failed=retry_failed)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _run_locked(plan: dict, output_dir: Path, *, retry_failed: bool) -> dict:
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and _read_json(plan_path) != plan:
        raise ValueError("Preparation settings/code changed. Use a new --run-name to keep evidence separate.")
    _write_json(plan_path, plan)
    _, all_specs = load_source_program_catalog()
    specs = [spec for spec in all_specs if spec.task_id in plan["task_ids"]]
    specs.sort(key=lambda spec: (not spec.origin.startswith("maniskill_"), spec.task_id))
    summary_path = output_dir / "summary.json"
    previous = _read_json(summary_path).get("results", []) if summary_path.exists() else []
    rows = {row["task_id"]: row for row in previous}
    summary = {"schema": "source_program_preparation.v2", "tasks": len(specs),
               "pid": os.getpid(), "state": "running", "current_task": None}

    def checkpoint():
        summary.update(
            results=list(rows.values()), frozen=sum(bool(row.get("success")) for row in rows.values()),
            complete=len(rows) == len(specs) and all(row.get("success") for row in rows.values()),
            updated_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(summary_path, summary)

    runtime = capture_experiment_environment({
        "phase": "source_preparation", "source_robot": specs[0].source_robot,
        "target_robot": specs[0].source_robot, "seed": plan["seeds"][0],
    })
    _write_json(output_dir / "runtime_environment.json", runtime)
    if not runtime["validation"]["ok"]:
        summary.update(state="environment_contract_mismatch", mismatches=runtime["validation"]["mismatches"])
        checkpoint()
        print("Environment mismatch; see runtime_environment.json. No simulation or LLM calls.", flush=True)
        return summary

    frozen = {row["task_id"]: row for row in source_program_status(specs)}
    checkpoint()
    try:
        for spec in specs:
            if frozen[spec.task_id]["frozen"] and _sha256_file(spec.candidate_path) == _sha256_file(spec.frozen_path):
                rows[spec.task_id] = {"task_id": spec.task_id, "success": True, "status": "already_frozen"}
                checkpoint()
                continue
            previous_row = rows.get(spec.task_id, {})
            if previous_row.get("status") not in {None, "running", "interrupted"} and not retry_failed:
                if not previous_row.get("success"):
                    continue
            task_dir = output_dir / "tasks" / spec.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            config = {"task_id": spec.task_id, "plan": plan, "output_dir": str(task_dir.resolve()),
                      "retry_unavailable": retry_failed}
            _write_json(task_dir / "worker.json", config)
            rows[spec.task_id] = {"task_id": spec.task_id, "success": False, "status": "running"}
            summary["current_task"] = spec.task_id
            checkpoint()
            print(f"[{spec.task_id}] validating / repairing / full-seed check", flush=True)
            result_path = task_dir / "worker_result.json"
            if result_path.exists():
                result_path.rename(task_dir / f"previous_result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json")
            with (task_dir / "worker.log").open("a", encoding="utf-8") as log:
                returncode, timed_out = run_worker(
                    [sys.executable, "-u", "-m", "maniskill_backend.source_preparation", str(task_dir / "worker.json")],
                    log=log, timeout=plan["task_timeout"],
                )
            if result_path.exists() and returncode == 0:
                row = _read_json(result_path)
            else:
                row = {"task_id": spec.task_id, "success": False,
                       "status": "worker_timeout" if timed_out else "worker_crashed",
                       "returncode": returncode}
            row["log"] = str(task_dir / "worker.log")
            rows[spec.task_id] = row
            summary["current_task"] = None
            checkpoint()
            print(f"  {row['status']}; frozen={row['success']}", flush=True)
    except BaseException:
        task_id = summary.get("current_task")
        if task_id:
            rows[task_id]["status"] = "interrupted"
        summary["state"] = "interrupted"
        checkpoint()
        raise
    summary["state"] = "complete" if summary["complete"] else "incomplete"
    checkpoint()
    return summary


def run_worker(command: list[str], *, log, timeout: int) -> tuple[int, bool]:
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        _stop_worker(process)
        return process.returncode, True
    except BaseException:
        _stop_worker(process)
        raise


def _stop_worker(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def prepare_task(config: dict) -> dict:
    catalog, specs = load_source_program_catalog()
    spec = next(spec for spec in specs if spec.task_id == config["task_id"])
    plan = config["plan"]
    output = Path(config["output_dir"])
    row = {"task_id": spec.task_id, "success": False}
    try:
        assets = subprocess.run(
            [sys.executable, "-u", "-m", "maniskill_backend.source_preparation", "assets",
             spec.task_id, "download" if plan["download_assets"] else "check"],
            cwd=REPO_ROOT, stdin=subprocess.DEVNULL, timeout=plan["asset_timeout"], check=False,
        )
        if assets.returncode:
            return dict(row, status="assets_blocked", message="See worker.log; no LLM repair was attempted.")
        kwargs = {key: plan[key] for key in ("seeds", "obs_mode", "sim_backend", "render_backend")}
        previous_path = output / "repair" / spec.task_id / "synthesis_summary.json"
        if previous_path.exists():
            previous = _read_json(previous_path)
            if previous.get("success"):
                record = freeze_source_program(spec, previous["validation"], minimum_seed_count=catalog["freeze_policy"]["minimum_seed_count"])
                return dict(row, success=True, status="frozen", path="resumed_success",
                            seeds=record["validated_seeds"], source_sha256=record["source_sha256"])
        attempt = len(list(output.glob("initial_*"))) + 1
        initial_dir = output / f"initial_{attempt:03d}"
        smoke_kwargs = dict(kwargs, seeds=plan["seeds"][:2])
        validation = validate_source_program(spec, output_dir=initial_dir / "smoke", enforce_environment=True, **smoke_kwargs)
        if validation.get("success"):
            validation = validate_source_program(spec, output_dir=initial_dir / "full", enforce_environment=True, **kwargs)
        if validation.get("status") in {"infrastructure_failure", "environment_contract_mismatch"}:
            return dict(row, status=validation["status"], message=validation.get("message"))
        origin = "validated_existing_candidate"
        if not validation.get("success"):
            synthesis = synthesize_source_program(
                spec, output_dir=output / "repair", max_cycles=plan["max_cycles"],
                initial_validation=validation, retry_unavailable=config.get("retry_unavailable", False), **kwargs,
            )
            if not synthesis.get("success"):
                return dict(row, status=synthesis.get("status", "repair_budget_exhausted"),
                            cycles=len(synthesis.get("cycles", [])),
                            candidates=[{key: cycle.get(key) for key in ("cycle", "valid", "num_success", "evaluated_seeds", "error")}
                                        for cycle in synthesis.get("cycles", [])],
                            details=str(output / "repair" / spec.task_id / "synthesis_summary.json"))
            validation = synthesis["validation"]
            origin = "llm_repair"
        record = freeze_source_program(spec, validation, minimum_seed_count=catalog["freeze_policy"]["minimum_seed_count"])
        return dict(row, success=True, status="frozen", path=origin, seeds=record["validated_seeds"],
                    success_rate=validation["success_rate"], source_sha256=record["source_sha256"])
    except subprocess.TimeoutExpired:
        return dict(row, status="assets_timeout", message="Asset preparation exceeded its time budget.")
    except Exception as exc:
        return dict(row, status="worker_error", message=repr(exc))


def ensure_assets(task_id: str, *, download: bool, robot_uid: str | None = None):
    # Use the installed, pinned ManiSkill registry rather than task-specific URLs.
    import mani_skill.envs  # noqa: F401
    from mani_skill.agents.registration import REGISTERED_AGENTS
    from mani_skill.utils import assets, download_asset
    from mani_skill.utils.registration import REGISTERED_ENVS

    _, specs = load_source_program_catalog()
    spec = next(spec for spec in specs if spec.task_id == task_id)
    required = list(REGISTERED_ENVS[spec.env_id].asset_download_ids or [])
    asset_robot = robot_uid or spec.source_robot
    if asset_robot not in REGISTERED_AGENTS:
        raise RuntimeError(f"Unknown ManiSkill robot UID: {asset_robot}")
    required += list(REGISTERED_AGENTS[asset_robot].asset_download_ids or [])
    ids = set()
    for asset_id in required:
        if asset_id in assets.DATA_GROUPS:
            ids.update(assets.expand_data_group_into_individual_data_source_ids(asset_id))
        else:
            ids.add(asset_id)
    for asset_id in sorted(ids):
        source = assets.DATA_SOURCES[asset_id]
        destination = Path(source.output_dir) / source.target_path
        if assets.is_data_source_downloaded(asset_id):
            if not destination.is_dir() or any(destination.iterdir()):
                continue
            print(f"Retrying an empty asset directory left by a failed download: {destination}", flush=True)
        if not download:
            raise RuntimeError(f"Missing asset {asset_id}; rerun prepare without --no-download.")
        print(f"Downloading missing asset: {asset_id}", flush=True)
        Path(source.output_dir).mkdir(parents=True, exist_ok=True)
        # Failed downloads must not leave an empty 'installed' directory in ManiSkill.
        with tempfile.TemporaryDirectory(prefix=".source-download-", dir=source.output_dir) as temp:
            staged_source = replace(source, output_dir=Path(temp))
            download_asset.download(staged_source, non_interactive=True)
            staged = Path(temp) / source.target_path
            if not staged.exists():
                raise RuntimeError(f"Download did not create {asset_id}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_dir() and not any(destination.iterdir()):
                destination.rmdir()
            if destination.exists():
                raise RuntimeError(f"Asset appeared during download; refusing to replace {destination}")
            staged.rename(destination)


if __name__ == "__main__":
    if sys.argv[1] == "assets":
        ensure_assets(sys.argv[2], download=sys.argv[3] == "download")
    else:
        configuration = _read_json(Path(sys.argv[1]))
        result = prepare_task(configuration)
        _write_json(Path(configuration["output_dir"]) / "worker_result.json", result)
