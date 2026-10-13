"""Verify one target migration as reproducible end-to-end evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def verify_acceptance(
    run_dir: Path,
    *,
    task_id: str,
    target_robot: str,
) -> dict[str, Any]:
    """Check that a migration generated code, acted, and reached official success."""

    summary_path = run_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    rows = summary.get("results") if isinstance(summary.get("results"), list) else []
    row = next(
        (item for item in rows if isinstance(item, Mapping) and item.get("task_id") == task_id),
        {},
    )
    details_path = _path_from(row.get("details"))
    details = _read_json(details_path) if details_path and details_path.is_file() else {}
    target_result = details.get("target_result") if isinstance(details.get("target_result"), Mapping) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    adapter_path = _path_from(row.get("target_adapter") or details.get("target_adapter"))
    manifest_path = _path_from(details.get("manifest"))
    runtime_path = run_dir / "runtime_environment.json"
    runtime = _read_json(runtime_path) if runtime_path.is_file() else {}
    hashes = details.get("artifact_sha256") if isinstance(details.get("artifact_sha256"), Mapping) else {}

    checks = {
        "batch_summary_present": summary_path.is_file(),
        "batch_schema_valid": summary.get("schema") == "target_migration_batch.v1",
        "batch_complete": summary.get("state") == "complete" and summary.get("complete") is True,
        "task_result_present": bool(row),
        "task_success": row.get("success") is True and row.get("status") == "success",
        "generation_executed": int(metrics.get("generation_cycles") or 0) > 0,
        "target_actions_executed": int(metrics.get("target_action_steps") or 0) > 0,
        "dynamic_result_present": bool(details),
        "dynamic_success": details.get("success") is True and details.get("status") == "success",
        "official_success": target_result.get("official_success") is True,
        "successful_trial_acted": int(target_result.get("action_steps") or 0) > 0,
        "target_adapter_present": bool(adapter_path and adapter_path.is_file()),
        "target_adapter_hashed": bool(hashes.get("target_adapter")),
        "manifest_present": bool(manifest_path and manifest_path.is_file()),
        "runtime_contract_valid": runtime.get("validation", {}).get("ok") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "schema": "end_to_end_acceptance.v1",
        "passed": not failed,
        "task_id": task_id,
        "target_robot": target_robot,
        "run_dir": str(run_dir.resolve()),
        "checks": checks,
        "failed_checks": failed,
        "evidence": {
            "batch_summary": str(summary_path.resolve()),
            "dynamic_summary": str(details_path) if details_path else None,
            "target_adapter": str(adapter_path) if adapter_path else None,
            "manifest": str(manifest_path) if manifest_path else None,
            "runtime_environment": str(runtime_path.resolve()),
            "target_action_steps": int(metrics.get("target_action_steps") or 0),
            "generation_cycles": int(metrics.get("generation_cycles") or 0),
            "repair_cycles": int(metrics.get("repair_cycles") or 0),
            "total_tokens": int(metrics.get("total_tokens") or 0),
            "llm_cost_usd": float(metrics.get("llm_cost_usd") or 0.0),
        },
    }
    _write_json(run_dir / "acceptance_report.json", report)
    _write_markdown(run_dir / "acceptance_report.md", report)
    return report


def _path_from(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# End-to-End Acceptance",
        "",
        f"- result: **{status}**",
        f"- task: `{report['task_id']}`",
        f"- target: `{report['target_robot']}`",
        "",
        "| check | passed |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {passed} |" for name, passed in report["checks"].items())
    lines.extend(["", "## Evidence", "", "```json", json.dumps(report["evidence"], indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
