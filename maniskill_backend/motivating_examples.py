"""Build paper motivating examples from traceable simulator artifacts.

The report deliberately refuses to reconstruct results from prose. A number is
paper-ready only when it can be recomputed from an existing JSON or JSONL file
whose SHA256 is recorded in the output.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"


def _read_payloads(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        payloads = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        return payloads


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _nested_first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    containers = (
        row,
        _mapping(row.get("diagnostics")),
        _mapping(row.get("runtime_diagnostics")),
        _mapping(row.get("failure_diagnosis")),
        _mapping(row.get("automatic_diagnosis")),
    )
    for container in containers:
        value = _first(container, keys)
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trial_rows(payloads: Iterable[Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for payload in payloads:
        if isinstance(payload, list):
            rows.extend(dict(item) for item in payload if isinstance(item, Mapping))
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("type") == "trial":
            rows.append(dict(payload))
            continue
        summary = _mapping(payload.get("summary"))
        candidates = payload.get("rows") or summary.get("rows")
        if isinstance(candidates, list):
            rows.extend(dict(item) for item in candidates if isinstance(item, Mapping))
    return rows


def _failure_stage(row: Mapping[str, Any]) -> str:
    explicit = _nested_first(row, ("stage", "failure_stage"))
    if explicit:
        return str(explicit)
    message = str(row.get("message") or "").lower()
    for stage in ("approach", "descent", "contact", "drag", "settle", "grasp", "place"):
        if stage in message:
            return stage
    return "unknown"


def summarize_pull_multiseed(path: Path) -> Dict[str, Any]:
    rows = _trial_rows(_read_payloads(path))
    successes = sum(bool(row.get("success")) for row in rows)
    failures = [row for row in rows if not row.get("success")]
    stage_counts = Counter(_failure_stage(row) for row in failures)
    stage_errors = [
        value
        for row in failures
        if (value := _float(_nested_first(row, ("tcp_stage_error_norm",)))) is not None
    ]
    paper_ready = path.is_file() and len(rows) >= 5 and bool(failures)
    return {
        "example_id": "pullcube_cross_seed_reachability",
        "evidence_available": path.is_file() and bool(rows),
        "paper_ready": paper_ready,
        "source": {"path": str(path), "sha256": _sha256(path)},
        "num_trials": len(rows),
        "num_success": successes,
        "num_failure": len(rows) - successes,
        "success_rate": round(successes / len(rows), 4) if rows else None,
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "tcp_stage_error_norm_min": round(min(stage_errors), 5) if stage_errors else None,
        "tcp_stage_error_norm_max": round(max(stage_errors), 5) if stage_errors else None,
        "claim": (
            f"The same adapter succeeded on {successes}/{len(rows)} seeds; failures concentrated in "
            f"{', '.join(f'{key}={value}' for key, value in sorted(stage_counts.items()))}."
            if paper_ready
            else "No paper-ready cross-seed PullCube artifact is available."
        ),
        "limitations": (
            []
            if paper_ready
            else ["Need at least five real trials and at least one failed seed in a tracked JSON/JSONL artifact."]
        ),
    }


def _probe_rows(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    candidates = (
        payload.get("all_probe_cases")
        or payload.get("probe_cases")
        or payload.get("results")
        or payload.get("rows")
        or []
    )
    return [dict(item) for item in candidates if isinstance(item, Mapping)]


def _grasping(row: Mapping[str, Any]) -> bool:
    return any(
        bool(row.get(key))
        for key in (
            "task_success",
            "is_grasping_after_lift",
            "is_grasping_after_close",
            "is_grasping",
        )
    )


def _best_probe(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    explicit = payload.get("best_probe_case")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    if not rows:
        return {}
    return dict(max(rows, key=lambda row: _float(row.get("score")) or float("-inf")))


def summarize_pick_probe(path: Path) -> Dict[str, Any]:
    payloads = _read_payloads(path)
    payload = next((item for item in reversed(payloads) if isinstance(item, Mapping)), {})
    rows = _probe_rows(payload)
    grasping = sum(_grasping(row) for row in rows)
    best = _best_probe(payload, rows)
    dry_run = bool(payload.get("dry_run"))
    paper_ready = path.is_file() and bool(rows) and not dry_run
    best_metrics = {
        key: best.get(key)
        for key in (
            "grasp_z_offset",
            "close_steps",
            "close_command",
            "settle_steps",
            "tcp_grasp_xy",
            "tcp_grasp_z",
            "cube_disp_xy",
            "cube_lift_delta_z",
            "is_grasping_after_close",
            "is_grasping_after_lift",
            "score",
        )
        if best.get(key) is not None
    }
    return {
        "example_id": "pickcube_alignment_without_force_closure",
        "evidence_available": path.is_file() and bool(rows),
        "paper_ready": paper_ready,
        "source": {"path": str(path), "sha256": _sha256(path)},
        "probe_id": payload.get("probe_id"),
        "num_probe_cases": len(rows),
        "num_grasping_cases": grasping,
        "best_probe_case": best_metrics,
        "claim": (
            f"Structured probing tested {len(rows)} close envelopes and found {grasping} force-closure cases."
            if paper_ready
            else "No paper-ready real PickCube probe artifact is available."
        ),
        "limitations": (
            []
            if paper_ready
            else ["Need a non-dry-run structured probe JSON containing at least one measured case."]
        ),
    }


def build_motivating_examples(*, pull_results: Path, pick_probe: Path) -> Dict[str, Any]:
    examples = [
        summarize_pull_multiseed(pull_results),
        summarize_pick_probe(pick_probe),
    ]
    missing = [item["example_id"] for item in examples if not item["paper_ready"]]
    return {
        "schema": "embodied_migration_motivating_examples.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_ready": not missing,
        "missing_or_incomplete_examples": missing,
        "examples": examples,
    }


def motivating_examples_markdown(payload: Mapping[str, Any]) -> str:
    examples = {item.get("example_id"): item for item in payload.get("examples") or []}
    pull = examples.get("pullcube_cross_seed_reachability") or {}
    pick = examples.get("pickcube_alignment_without_force_closure") or {}
    lines = [
        "# Motivating Examples",
        "",
        f"- paper ready: `{payload.get('paper_ready')}`",
        f"- missing/incomplete: `{', '.join(payload.get('missing_or_incomplete_examples') or []) or 'none'}`",
        "",
        "Only values recomputed from the files and SHA256 hashes below may be cited.",
        "",
        "## PullCube: cross-seed reachability",
        "",
        "| Trials | Success | Failure | Rate | Failure stages | TCP stage error range |",
        "|---:|---:|---:|---:|---|---|",
        f"| {pull.get('num_trials')} | {pull.get('num_success')} | {pull.get('num_failure')} | "
        f"{pull.get('success_rate')} | {pull.get('failure_stage_counts')} | "
        f"[{pull.get('tcp_stage_error_norm_min')}, {pull.get('tcp_stage_error_norm_max')}] |",
        "",
        str(pull.get("claim") or ""),
        "",
        "## PickCube: alignment versus force closure",
        "",
        "| Probe cases | Grasping cases | Best measured case |",
        "|---:|---:|---|",
        f"| {pick.get('num_probe_cases')} | {pick.get('num_grasping_cases')} | "
        f"`{json.dumps(pick.get('best_probe_case') or {}, ensure_ascii=False)}` |",
        "",
        str(pick.get("claim") or ""),
        "",
        "## Provenance",
        "",
        "| Example | Path | SHA256 | Ready |",
        "|---|---|---|---|",
    ]
    for item in payload.get("examples") or []:
        source = item.get("source") or {}
        lines.append(
            f"| `{item.get('example_id')}` | `{source.get('path')}` | `{source.get('sha256')}` | "
            f"{item.get('paper_ready')} |"
        )
    lines.append("")
    if not payload.get("paper_ready"):
        lines.extend(
            [
                "> Incomplete rows are placeholders for the remote GPU artifacts. They must not be quoted as current repository results.",
                "",
            ]
        )
    return "\n".join(lines)


def write_motivating_examples(payload: Mapping[str, Any], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "motivating_examples.json"
    md_path = output_dir / "motivating_examples.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(motivating_examples_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}
