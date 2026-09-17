"""Capture and validate the immutable experiment environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from llm_client import (
    completion_token_limit,
    current_provider,
    default_model,
    generation_temperature,
    llm_seed,
    openrouter_allow_fallbacks,
    openrouter_exclude_reasoning,
    openrouter_reasoning_effort,
    openrouter_upstream_provider,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = REPO_ROOT / "experiment_config.json"


def load_experiment_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != "embodied_migration_experiment.v1":
        raise ValueError(f"Unsupported experiment contract: {contract.get('schema')!r}")
    return contract


def capture_experiment_environment(
    run_spec: Mapping[str, Any],
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Record exact runtime facts and compare them with the tracked contract."""

    contract_bytes = contract_path.read_bytes()
    contract = load_experiment_contract(contract_path)
    actual = {
        "repository": _repository_state(),
        "python": {
            "version": platform.python_version(),
            "major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {
            "mani_skill": _package_record("mani_skill", module_name="mani_skill"),
            "sapien": _package_record("sapien", module_name="sapien"),
            "numpy": _package_record("numpy"),
            "torch": _package_record("torch"),
            "openai": _package_record("openai"),
        },
        "cuda": _cuda_record(),
        "llm": {
            "provider": current_provider(),
            "model": default_model(),
            "temperature": generation_temperature(),
            "max_tokens": completion_token_limit(),
            "reasoning_effort": openrouter_reasoning_effort(),
            "exclude_reasoning_from_response": openrouter_exclude_reasoning(),
            "seed": llm_seed(),
            "upstream_provider": openrouter_upstream_provider(),
            "allow_provider_fallbacks": openrouter_allow_fallbacks(),
        },
        "migration": dict(run_spec),
    }
    mismatches = compare_with_contract(contract, actual)
    return {
        "schema": "embodied_migration_runtime.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path),
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "contract": contract,
        "actual": actual,
        "validation": {"ok": not mismatches, "mismatches": mismatches},
    }


def compare_with_contract(
    contract: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return structured differences without hiding unavailable dependencies."""

    mismatches: list[dict[str, Any]] = []
    runtime = contract["runtime"]
    _compare(
        mismatches,
        "python.version",
        runtime["python_version"],
        actual["python"]["version"],
    )
    for package, expected in runtime["packages"].items():
        _compare(
            mismatches,
            f"packages.{package}.version",
            expected,
            actual["packages"].get(package, {}).get("version"),
        )
    _compare(
        mismatches,
        "cuda.driver_api",
        runtime["cuda_driver_api"],
        actual["cuda"].get("driver_api"),
    )

    expected_llm = contract["llm"]
    for key in (
        "provider",
        "model",
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "exclude_reasoning_from_response",
        "seed",
        "upstream_provider",
        "allow_provider_fallbacks",
    ):
        _compare(mismatches, f"llm.{key}", expected_llm[key], actual["llm"].get(key))

    migration = actual["migration"]
    if not migration.get("source_robot"):
        mismatches.append(_mismatch("migration.source_robot", "non-empty UID", None))
    if not migration.get("target_robot"):
        mismatches.append(_mismatch("migration.target_robot", "non-empty UID", None))
    if contract["seed_policy"]["same_seed_for_source_and_target"] and "seed" not in migration:
        mismatches.append(_mismatch("migration.seed", "explicit integer", None))
    return mismatches


def _compare(
    mismatches: list[dict[str, Any]], field: str, expected: Any, observed: Any
) -> None:
    if observed != expected:
        mismatches.append(_mismatch(field, expected, observed))


def _mismatch(field: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {"field": field, "expected": expected, "observed": observed}


def _package_record(distribution: str, *, module_name: str | None = None) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    record: dict[str, Any] = {"version": version}
    if module_name:
        spec = importlib.util.find_spec(module_name)
        module_path = Path(spec.origin).resolve() if spec and spec.origin else None
        record["module_path"] = str(module_path) if module_path else None
        record["git_commit"] = _containing_git_commit(module_path)
    return record


def _containing_git_commit(path: Path | None) -> str | None:
    if path is None:
        return None
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return _run(["git", "-C", str(parent), "rev-parse", "HEAD"])
    return None


def _repository_state() -> dict[str, Any]:
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status), "status": status.splitlines() if status else []}


def _cuda_record() -> dict[str, Any]:
    output = _run(["nvidia-smi"])
    driver_api = None
    driver = None
    if output:
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
        driver_match = re.search(r"Driver Version:\s*([0-9.]+)", output)
        driver_api = cuda_match.group(1) if cuda_match else None
        driver = driver_match.group(1) if driver_match else None
    gpu_query = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    return {
        "driver_api": driver_api,
        "driver_version": driver,
        "gpus": gpu_query.splitlines() if gpu_query else [],
        "nvcc": _run(["nvcc", "--version"]),
    }


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output or None
