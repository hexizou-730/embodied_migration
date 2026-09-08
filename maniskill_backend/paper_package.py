"""Build a secret-free, checksummed package for remote paper experiments."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "paper.py",
    "migrate.py",
    "auto.py",
    "llm_client.py",
    "README.md",
    "requirements.txt",
    "requirements-maniskill.txt",
)
SOURCE_DIRS = (
    "maniskill_backend",
    "scripts",
    "lmp",
    "tests",
    "capabilities",
    "demos/simple_harness",
)
EXCLUDED_NAMES = {"__pycache__", ".DS_Store", ".env", "results", "evidence"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def _source_files(repo_root: Path) -> Iterable[Path]:
    for name in ROOT_FILES:
        path = repo_root / name
        if path.is_file():
            yield path
    for name in SOURCE_DIRS:
        root = repo_root / name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root)
            if any(part in EXCLUDED_NAMES for part in relative.parts):
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def _add_bytes(archive: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def build_remote_package(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    plan_files: Mapping[str, str],
    preparation_checks: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Archive runtime code and one frozen plan without credentials or prior results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    package_name = f"embodied_migration_{plan.get('plan_name') or 'paper'}"
    archive_path = output_dir / f"{package_name}.tar.gz"
    source_files = list(dict.fromkeys(_source_files(repo_root)))
    file_rows = [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in source_files
    ]
    plan_labels = {"json": "plan", "markdown": "plan", "commands": "commands"}
    plan_rows = []
    plan_sources: Dict[str, Path] = {}
    for label, raw_path in plan_files.items():
        path = Path(raw_path)
        if not path.is_file():
            continue
        archive_relative = Path("remote_plan") / f"{plan_labels.get(label, label)}{path.suffix}"
        plan_sources[label] = path
        plan_rows.append(
            {
                "label": label,
                "path": str(archive_relative),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest: Dict[str, Any] = {
        "schema": "embodied_migration_remote_package.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "plan_name": plan.get("plan_name"),
        "tier": (plan.get("tier") or {}).get("tier_id"),
        "num_runs": plan.get("num_runs"),
        "num_ready_runs": plan.get("num_ready_runs"),
        "num_blocked_runs": plan.get("num_blocked_runs"),
        "contains_credentials": False,
        "preparation_checks": dict(preparation_checks or {}),
        "files": file_rows,
        "plan_files": plan_rows,
    }
    llm_provider = str((plan.get("llm_config") or {}).get("provider") or "")
    api_key_name = {
        "deepseek": "DEEPSEEK_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(llm_provider, "<PROVIDER_API_KEY>")
    manifest["remote_execution"] = {
        "command": "python paper.py start",
        "foreground_command": "python paper.py execute",
        "status_command": "python paper.py status",
        "api_key_env": api_key_name,
        "non_secret_llm_config_source": "remote_plan/plan.json",
        "progress_file": str(
            Path(str(plan.get("output_root") or "results/paper"))
            / str(plan.get("plan_name") or "paper")
            / "progress.json"
        ),
        "automatic_evidence_export": True,
    }
    instructions = f"""# Remote execution

This package contains code and a frozen benchmark plan. It contains no API key or `.env` file.

```bash
tar -xzf PACKAGE.tar.gz
cd embodied_migration
conda activate em-ms
export {api_key_name}=YOUR_KEY
python paper.py start
```

`start` checks the package, LLM configuration, and API key, then launches
`execute` in a detached process that survives the SSH session. `execute`
verifies source and plan hashes, runs audit/preflight/source gates,
restores provider/model/token/temperature/thinking settings from the checksummed
plan, executes that exact plan, verifies the evidence, and creates the return
archive. Only the API key must be supplied by the remote shell.

While it is running, another shell can read durable progress without interrupting it:

```bash
python paper.py status
```

For an intentional foreground run, use `python paper.py execute` instead.
"""
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in source_files:
            relative = path.relative_to(repo_root)
            archive.add(path, arcname=str(Path("embodied_migration") / relative), recursive=False)
        for item in plan_rows:
            archive.add(
                plan_sources[str(item["label"])],
                arcname=str(Path("embodied_migration") / item["path"]),
                recursive=False,
            )
        _add_bytes(
            archive,
            "embodied_migration/remote_plan/package_manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        _add_bytes(archive, "embodied_migration/REMOTE_RUN.md", instructions.encode("utf-8"))

    payload = {
        **manifest,
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }
    manifest_path = output_dir / f"{package_name}.manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["manifest"] = str(manifest_path)
    return payload


def verify_remote_package_manifest(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Verify unpacked source and frozen-plan files before remote execution."""

    errors = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema": "embodied_migration_remote_package_verification.v1",
            "valid": False,
            "manifest": str(manifest_path),
            "errors": [f"cannot read package manifest: {exc!r}"],
            "files_checked": 0,
        }

    rows = list(manifest.get("files") or []) + list(manifest.get("plan_files") or [])
    checks = manifest.get("preparation_checks") or {}
    if checks.get("protocol_audit_valid") is not True:
        errors.append("package was not produced after a valid protocol audit")
    if checks.get("static_preflight_ready") is not True:
        errors.append("package was not produced after a valid static preflight")
    if checks.get("llm_config_frozen") is not True:
        errors.append("package does not freeze its LLM configuration")
    if checks.get("llm_config_explicit") is not True:
        errors.append("package LLM configuration was not explicitly declared")
    if checks.get("stochastic_repetitions_configured") is not True:
        errors.append("package has invalid sampling settings for repeated generations")
    for item in rows:
        path = repo_root / str(item.get("path") or "")
        if not path.is_file():
            errors.append(f"missing packaged file: {item.get('path')}")
            continue
        if _sha256(path) != item.get("sha256"):
            errors.append(f"SHA mismatch: {item.get('path')}")
    if not any(item.get("path") == "remote_plan/plan.json" for item in rows):
        errors.append("package manifest does not bind remote_plan/plan.json")
    return {
        "schema": "embodied_migration_remote_package_verification.v1",
        "valid": not errors,
        "manifest": str(manifest_path),
        "files_checked": len(rows),
        "errors": errors,
    }


def build_evidence_export(
    *,
    plan_dir: Path,
    output_dir: Path,
    verification: Mapping[str, Any],
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Archive one verified plan and its external evidence bundles for return."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [path for path in plan_dir.rglob("*") if path.is_file()]
    for summary_path in sorted((plan_dir / "runs").glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bundle_dir = Path(str((summary.get("evidence_bundle") or {}).get("bundle_dir") or ""))
        if not bundle_dir.is_absolute():
            bundle_dir = repo_root / bundle_dir
        if bundle_dir.is_dir():
            paths.extend(path for path in bundle_dir.rglob("*") if path.is_file())
    unique_paths = []
    seen = set()
    for path in paths:
        try:
            relative = path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            continue
        if relative in seen:
            continue
        seen.add(relative)
        unique_paths.append((path, relative))
    rows = [
        {"path": str(relative), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path, relative in unique_paths
    ]
    plan_name = str(verification.get("plan_name") or plan_dir.name)
    archive_path = output_dir / f"embodied_migration_evidence_{plan_name}.tar.gz"
    manifest = {
        "schema": "embodied_migration_evidence_export.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_name": plan_name,
        "verification_valid": verification.get("valid") is True,
        "verification_complete": verification.get("complete") is True,
        "num_files": len(rows),
        "files": rows,
    }
    instructions = f"""# Evidence return

Extract this archive at the repository root, then run:

```bash
python paper.py verify --tier {str(verification.get('tier') or 'pilot')} --plan-name {plan_name}
```
"""
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, relative in unique_paths:
            archive.add(path, arcname=str(relative), recursive=False)
        _add_bytes(
            archive,
            "EVIDENCE_EXPORT_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        )
        _add_bytes(archive, "EVIDENCE_IMPORT.md", instructions.encode("utf-8"))
    payload = {
        **manifest,
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }
    manifest_path = output_dir / f"embodied_migration_evidence_{plan_name}.manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["manifest"] = str(manifest_path)
    return payload
