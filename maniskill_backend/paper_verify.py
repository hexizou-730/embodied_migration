"""Verify that a completed paper plan has internally consistent evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_METHODS = {"B2", "B3", "B4", "B5", "Ours"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else REPO_ROOT / path


def _seed_set(text: Any) -> set[int]:
    result: set[int] = set()
    for part in str(text or "").split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            step = 1 if int(right) >= int(left) else -1
            result.update(range(int(left), int(right) + step, step))
        else:
            result.add(int(item))
    return result


def _observed_seeds(summary: Mapping[str, Any], split: str) -> set[int]:
    rows = (((summary.get("details") or {}).get(split) or {}).get("rows") or [])
    return {int(row["seed"]) for row in rows if row.get("seed") is not None}


def _cegis_protocol_errors(summary: Mapping[str, Any]) -> list[str]:
    """Check the nested feedback ledger, not only the top-level claim."""

    cegis = ((summary.get("details") or {}).get("cegis_summary") or {})
    if not cegis:
        return ["CEGIS run is missing nested CEGIS summary"]
    errors = []
    policy = cegis.get("evidence_policy") or {}
    if policy.get("held_out_seeds_used_for_repair") is not False:
        errors.append("CEGIS evidence policy does not forbid held-out repair feedback")
    if policy.get("held_out_feedback_policy") != "evaluation_only_no_repair":
        errors.append("CEGIS held-out feedback policy is missing or invalid")
    cycles = list(cegis.get("cycles") or [])
    for item in cycles:
        feedback = item.get("repair_feedback") or {}
        if feedback.get("held_out_used") is not False:
            errors.append(f"cycle {item.get('cycle')} repair feedback is not held-out clean")
        held_out = item.get("held_out_evaluation")
        if held_out and held_out.get("used_for_repair") is not False:
            errors.append(f"cycle {item.get('cycle')} held-out result was marked for repair")
    held_out_cycles = [index for index, item in enumerate(cycles) if item.get("held_out_evaluation")]
    if held_out_cycles and held_out_cycles[-1] != len(cycles) - 1:
        errors.append("CEGIS continued repair after held-out evaluation")
    if (cegis.get("held_out") or {}).get("used_for_repair") not in (None, False):
        errors.append("CEGIS final held-out result was marked for repair")
    return errors


def _probe_budget_errors(
    summary: Mapping[str, Any],
    method_id: str,
    tier: Mapping[str, Any],
) -> list[str]:
    if method_id not in {"B5", "Ours"}:
        return []
    budget = int(tier.get("probe_budget") or 0)
    used = int((summary.get("metrics") or {}).get("probe_cases") or 0)
    declared = int(
        (summary.get("budget") or {}).get("explicit_probe_case_budget", -1)
    )
    errors = []
    if declared != budget:
        errors.append(
            f"run probe budget does not match frozen tier: {declared}!={budget}"
        )
    if used > budget:
        errors.append(f"probe cases exceed frozen per-run budget: {used}>{budget}")
    if method_id in {"B5", "Ours"}:
        nested = (((summary.get("details") or {}).get("cegis_summary") or {}).get("probe_budget") or {})
        if nested.get("scope") != "per_run_total":
            errors.append("CEGIS probe budget scope is not per_run_total")
        if int(nested.get("total") or -1) != budget:
            errors.append("CEGIS nested probe budget does not match frozen tier")
        if int(nested.get("cases_used") or 0) > budget:
            errors.append("CEGIS nested probe usage exceeds frozen tier")
        expected_strategy = "active" if method_id == "Ours" else "fixed_grid"
        nested_config = ((summary.get("details") or {}).get("cegis_summary") or {}).get("config") or {}
        if nested_config.get("probe_strategy") != expected_strategy:
            errors.append(
                f"CEGIS probe strategy mismatch: expected {expected_strategy}"
            )
        if nested_config.get("prompt_policy") != "full":
            errors.append("B5/Ours must use the same full prompt information boundary")
    return errors


def _llm_config_errors(
    summary: Mapping[str, Any],
    method_id: str,
    expected: Mapping[str, Any],
) -> list[str]:
    if method_id not in LLM_METHODS:
        return []
    runtime = summary.get("runtime") or {}
    observed = {
        "provider": runtime.get("llm_provider"),
        "model": runtime.get("llm_model"),
        "max_tokens": runtime.get("llm_max_tokens"),
        "temperature": runtime.get("llm_temperature"),
        "deepseek_thinking": runtime.get("deepseek_thinking"),
    }
    return [] if observed == dict(expected) else [
        f"runtime LLM configuration does not match frozen plan: {observed!r}"
    ]


def _llm_budget_errors(
    summary: Mapping[str, Any],
    method_id: str,
    contract: Mapping[str, Any],
) -> list[str]:
    expected_by_method = contract.get("llm_api_calls_per_run") or {}
    if method_id not in expected_by_method:
        return [f"frozen budget contract is missing method {method_id}"]
    expected = int(expected_by_method[method_id])
    declared = int((summary.get("budget") or {}).get("llm_api_call_budget", -1))
    used = int((summary.get("metrics") or {}).get("llm_api_calls") or 0)
    errors = []
    if declared != expected:
        errors.append(
            f"run LLM-call budget does not match frozen contract: {declared}!={expected}"
        )
    if used > expected:
        errors.append(f"LLM API calls exceed frozen per-run budget: {used}>{expected}")
    if method_id not in LLM_METHODS and used != 0:
        errors.append(f"non-LLM method recorded {used} LLM API calls")
    if method_id in LLM_METHODS:
        analysis_enabled = (summary.get("budget") or {}).get(
            "posthoc_analysis_llm_enabled"
        )
        if analysis_enabled is not False:
            errors.append("post-hoc analysis LLM must be disabled for budgeted paper runs")
    return errors


def _verify_source_gate(plan: Mapping[str, Any], plan_dir: Path) -> Dict[str, Any]:
    source_path = plan_dir / "source_baselines" / "summary.json"
    source = _load_json(source_path)
    expected = plan.get("source_baseline_gate") or {}
    errors = []
    if source is None:
        errors.append("missing or invalid source_baselines/summary.json")
    else:
        request = source.get("request") or {}
        for key in (
            "development_seeds",
            "held_out_seeds",
            "success_threshold",
            "min_trials_for_accept",
        ):
            if request.get(key) != expected.get(key):
                errors.append(f"source gate {key} does not match frozen plan")
        if source.get("ready") is not True:
            errors.append("source baseline gate is not ready")
    return {
        "valid": not errors,
        "path": str(source_path),
        "errors": errors,
        "ready": source.get("ready") if source else None,
    }


def _verify_run(
    run: Mapping[str, Any],
    plan_dir: Path,
    tier: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    llm_config: Mapping[str, Any],
    budget_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    run_id = str(run.get("run_id") or "")
    summary_path = plan_dir / "runs" / run_id / "summary.json"
    summary = _load_json(summary_path)
    errors: list[str] = []
    if summary is None:
        return {
            "run_id": run_id,
            "method_id": run.get("method_id"),
            "case_id": run.get("case_id"),
            "status": "missing",
            "valid": False,
            "errors": ["missing or invalid run summary.json"],
        }

    expected_fields = (
        "run_id",
        "method_id",
        "case_id",
        "repetition",
        "development_seeds",
        "held_out_seeds",
    )
    if summary.get("schema") != "paper_method_run.v1":
        errors.append("wrong run summary schema")
    for key in expected_fields:
        if summary.get(key) != run.get(key):
            errors.append(f"{key} does not match plan")
    for key in ("sim_backend", "render_backend"):
        if runtime_config.get(key) is not None and summary.get(key) != runtime_config.get(key):
            errors.append(f"{key} does not match frozen runtime configuration")

    development = _seed_set(run.get("development_seeds"))
    held_out = _seed_set(run.get("held_out_seeds"))
    if not development.isdisjoint(held_out):
        errors.append("development and held-out seeds overlap")
    observed_development = _observed_seeds(summary, "development")
    observed_held_out = _observed_seeds(summary, "held_out")
    if observed_development != development:
        errors.append("development trial seeds are incomplete or unexpected")
    if observed_held_out != held_out:
        errors.append("held-out trial seeds are incomplete or unexpected")
    if summary.get("held_out_used_for_repair") is not False:
        errors.append("held-out_used_for_repair must be false")
    if (summary.get("frozen_interface_integrity") or {}).get("valid") is not True:
        errors.append("frozen interface integrity failed")

    runtime = summary.get("runtime") or {}
    packages = runtime.get("packages") or {}
    if not runtime.get("git_rev") or not all(name in packages for name in ("mani-skill", "gymnasium", "sapien", "numpy")):
        errors.append("runtime version metadata is incomplete")

    method_id = str(run.get("method_id") or "")
    integrity = summary.get("generation_integrity") or {}
    if method_id in {"B5", "Ours"}:
        errors.extend(_cegis_protocol_errors(summary))
    errors.extend(_probe_budget_errors(summary, method_id, tier))
    errors.extend(_llm_config_errors(summary, method_id, llm_config))
    errors.extend(_llm_budget_errors(summary, method_id, budget_contract))
    if summary.get("success") is True and method_id in LLM_METHODS:
        if integrity.get("valid_generated_candidate") is not True:
            errors.append("successful LLM run has no valid generated candidate")
        if not str(summary.get("adapter_provenance") or "").startswith("llm_generated"):
            errors.append("successful LLM run has invalid adapter provenance")

    bundle = summary.get("evidence_bundle") or {}
    manifest_path = _path(bundle.get("manifest"))
    manifest = _load_json(manifest_path)
    if manifest is None:
        errors.append("missing or invalid evidence manifest")
    else:
        if manifest.get("schema") != "paper_run_evidence.v1":
            errors.append("wrong evidence manifest schema")
        if manifest.get("run_id") != run_id:
            errors.append("evidence manifest run_id mismatch")
        artifacts = manifest.get("artifacts") or {}
        required = ["final_adapter", "development_jsonl", "held_out_jsonl", "commands_log"]
        if method_id in LLM_METHODS:
            required.append("module_generation_jsonl")
        missing = [key for key in required if not artifacts.get(key) or not _path(artifacts.get(key)).is_file()]
        if missing:
            errors.append(f"missing evidence artifacts: {', '.join(missing)}")
        adapter = _path(artifacts.get("final_adapter"))
        if adapter.is_file() and _sha256(adapter) != summary.get("adapter_sha256"):
            errors.append("final adapter SHA does not match evaluated adapter")

    min_trials = int(tier.get("min_trials_for_accept") or 0)
    threshold = float(tier.get("success_threshold") or 0.0)
    held_summary = ((summary.get("details") or {}).get("held_out") or {})
    accepted = (
        int(held_summary.get("num_trials") or 0) >= min_trials
        and float(held_summary.get("success_rate") or 0.0) >= threshold
    )
    expected_success = accepted and (summary.get("frozen_interface_integrity") or {}).get("valid") is True
    if method_id in LLM_METHODS:
        expected_success = expected_success and integrity.get("valid_generated_candidate") is True
    if bool(summary.get("success")) != bool(expected_success):
        errors.append("recorded success does not match frozen acceptance rule")

    return {
        "run_id": run_id,
        "method_id": method_id,
        "case_id": run.get("case_id"),
        "status": "valid" if not errors else "invalid",
        "valid": not errors,
        "success": summary.get("success"),
        "errors": errors,
    }


def verify_paper_plan(plan_dir: Path) -> Dict[str, Any]:
    """Verify source gate and every ready run in one frozen paper plan."""

    plan_path = plan_dir / "plan.json"
    plan = _load_json(plan_path)
    if plan is None:
        return {
            "schema": "paper_evidence_verification.v1",
            "valid": False,
            "complete": False,
            "plan_dir": str(plan_dir),
            "errors": ["missing or invalid plan.json"],
            "source_baseline_gate": {},
            "runs": [],
        }
    source = _verify_source_gate(plan, plan_dir)
    rows = [
        _verify_run(
            run,
            plan_dir,
            plan.get("tier") or {},
            plan.get("runtime_config") or {},
            plan.get("llm_config") or {},
            plan.get("budget_contract") or {},
        )
        for run in plan.get("runs") or []
        if run.get("ready")
    ]
    complete = bool(rows) and all(row.get("status") != "missing" for row in rows)
    valid = source["valid"] and complete and all(row["valid"] for row in rows)
    return {
        "schema": "paper_evidence_verification.v1",
        "valid": valid,
        "complete": complete,
        "plan_name": plan.get("plan_name"),
        "tier": (plan.get("tier") or {}).get("tier_id"),
        "plan_dir": str(plan_dir),
        "num_ready_runs": len(rows),
        "num_valid_runs": sum(row["valid"] for row in rows),
        "num_missing_runs": sum(row.get("status") == "missing" for row in rows),
        "errors": [],
        "source_baseline_gate": source,
        "runs": rows,
    }


def verification_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Paper Evidence Verification",
        "",
        f"- valid: `{payload.get('valid')}`",
        f"- complete: `{payload.get('complete')}`",
        f"- source baseline valid: `{(payload.get('source_baseline_gate') or {}).get('valid')}`",
        f"- valid runs: `{payload.get('num_valid_runs')}/{payload.get('num_ready_runs')}`",
        f"- missing runs: `{payload.get('num_missing_runs')}`",
        "",
        "| Run | Method | Case | Status | Success | Errors |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload.get("runs") or []:
        errors = "; ".join(row.get("errors") or []).replace("|", "\\|")
        lines.append(
            f"| `{row.get('run_id')}` | {row.get('method_id')} | `{row.get('case_id')}` | "
            f"{row.get('status')} | {row.get('success')} | {errors} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_verification(payload: Mapping[str, Any], plan_dir: Path) -> Dict[str, str]:
    json_path = plan_dir / "verification.json"
    md_path = plan_dir / "verification.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(verification_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
