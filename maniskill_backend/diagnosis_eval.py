"""Build and evaluate independent annotations for the five-layer diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from maniskill_backend.failure_diagnosis import diagnose_failure


def _trial_id(row: Mapping[str, Any]) -> str:
    stable = {
        "task_id": row.get("task_id"),
        "robot_uid": row.get("robot_uid"),
        "seed": row.get("seed"),
        "message": row.get("message"),
        "adapter_module": row.get("adapter_module"),
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=False, default=repr).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _iter_trials(paths: Iterable[Path]) -> Iterable[tuple[Path, Dict[str, Any]]]:
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "metadata":
                continue
            if row.get("type") == "trial":
                row = {key: value for key, value in row.items() if key != "type"}
            if "success" in row:
                yield path, row


def build_annotation_template(paths: Sequence[Path], output_path: Path) -> Dict[str, Any]:
    """Create deduplicated failure rows without exposing repair hints to annotators."""

    rows = []
    seen = set()
    for source, trial in _iter_trials(paths):
        if bool(trial.get("success")):
            continue
        identifier = _trial_id(trial)
        if identifier in seen:
            continue
        seen.add(identifier)
        rows.append(
            {
                "trial_id": identifier,
                "source_path": str(source),
                "task_id": trial.get("task_id"),
                "robot_uid": trial.get("robot_uid"),
                "seed": trial.get("seed"),
                "success": False,
                "message": trial.get("message"),
                "failure_type": trial.get("failure_type"),
                "code_ok": trial.get("code_ok", True),
                "execution_log": trial.get("execution_log") or [],
                "final_info": trial.get("final_info") or {},
                "runtime_diagnostics": trial.get("runtime_diagnostics") or {},
                "initial_runtime_diagnostics": trial.get("initial_runtime_diagnostics") or {},
                "label_layer": "",
                "label_reason": "",
                "annotator": "",
                "notes": "",
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
    return {"num_failures": len(rows), "output": str(output_path)}


def _confusion(rows: Sequence[Mapping[str, Any]], label_key: str, prediction_key: str) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for row in rows:
        expected = str(row.get(label_key) or "")
        predicted = str(row.get(prediction_key) or "")
        matrix.setdefault(expected, {})[predicted] = matrix.setdefault(expected, {}).get(predicted, 0) + 1
    return matrix


def _labeled_by_trial(path: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("trial_id")): row
        for _, row in _iter_trials([path])
        if str(row.get("trial_id") or "") and str(row.get("label_layer") or "").strip()
    }


def _cohen_kappa(pairs: Sequence[tuple[str, str]]) -> Dict[str, Any]:
    if not pairs:
        return {"num_pairs": 0, "observed_agreement": None, "cohen_kappa": None}
    total = len(pairs)
    left_counts: Dict[str, int] = {}
    right_counts: Dict[str, int] = {}
    for left, right in pairs:
        left_counts[left] = left_counts.get(left, 0) + 1
        right_counts[right] = right_counts.get(right, 0) + 1
    observed = sum(left == right for left, right in pairs) / total
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts.get(label, 0) / total) * (right_counts.get(label, 0) / total)
        for label in categories
    )
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "num_pairs": total,
        "observed_agreement": round(observed, 4),
        "cohen_kappa": round(kappa, 4),
    }


def _inter_annotator_agreement(primary: Path, secondary: Path) -> Dict[str, Any]:
    left = _labeled_by_trial(primary)
    right = _labeled_by_trial(secondary)
    shared = sorted(set(left) & set(right))
    layer_pairs = [
        (str(left[key].get("label_layer") or ""), str(right[key].get("label_layer") or ""))
        for key in shared
    ]
    reason_pairs = [
        (str(left[key].get("label_reason") or ""), str(right[key].get("label_reason") or ""))
        for key in shared
        if str(left[key].get("label_reason") or "").strip()
        and str(right[key].get("label_reason") or "").strip()
    ]
    return {
        "primary_path": str(primary),
        "secondary_path": str(secondary),
        "layer": _cohen_kappa(layer_pairs),
        "reason": _cohen_kappa(reason_pairs),
    }


def evaluate_annotations(path: Path, *, secondary_path: Path | None = None) -> Dict[str, Any]:
    rows = []
    total = 0
    for _, row in _iter_trials([path]):
        total += 1
        if not str(row.get("label_layer") or "").strip():
            continue
        predicted = diagnose_failure(
            task_id=str(row.get("task_id") or ""),
            success=False,
            code_ok=bool(row.get("code_ok", True)),
            message=str(row.get("message") or ""),
            failure_type=str(row.get("failure_type") or ""),
            execution_log=row.get("execution_log"),
            final_info=row.get("final_info") or {},
            runtime_diagnostics=row.get("runtime_diagnostics") or {},
            initial_runtime_diagnostics=row.get("initial_runtime_diagnostics") or {},
        )
        rows.append(
            {
                **row,
                "predicted_layer": predicted.get("layer"),
                "predicted_reason": predicted.get("reason"),
            }
        )
    layer_correct = sum(row["label_layer"] == row["predicted_layer"] for row in rows)
    reason_labeled = [row for row in rows if str(row.get("label_reason") or "").strip()]
    reason_correct = sum(row["label_reason"] == row["predicted_reason"] for row in reason_labeled)
    payload = {
        "schema": "five_layer_diagnosis_evaluation.v1",
        "annotation_path": str(path),
        "num_template_rows": total,
        "num_labeled_rows": len(rows),
        "coverage": round(len(rows) / total, 4) if total else 0.0,
        "layer_accuracy": round(layer_correct / len(rows), 4) if rows else None,
        "reason_accuracy": round(reason_correct / len(reason_labeled), 4) if reason_labeled else None,
        "layer_confusion": _confusion(rows, "label_layer", "predicted_layer"),
        "reason_confusion": _confusion(reason_labeled, "label_reason", "predicted_reason"),
        "ready": bool(rows),
    }
    if secondary_path is not None:
        payload["inter_annotator_agreement"] = _inter_annotator_agreement(path, secondary_path)
    return payload


def diagnosis_evaluation_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Five-Layer Diagnosis Evaluation",
        "",
        f"- labeled/total: `{payload.get('num_labeled_rows')}/{payload.get('num_template_rows')}`",
        f"- annotation coverage: `{payload.get('coverage')}`",
        f"- layer accuracy: `{payload.get('layer_accuracy')}`",
        f"- reason accuracy: `{payload.get('reason_accuracy')}`",
        "",
        "## Layer Confusion",
        "",
        "| Human label | System prediction | Count |",
        "|---|---|---:|",
    ]
    for expected, predicted_rows in (payload.get("layer_confusion") or {}).items():
        for predicted, count in predicted_rows.items():
            lines.append(f"| `{expected}` | `{predicted}` | {count} |")
    agreement = payload.get("inter_annotator_agreement") or {}
    if agreement:
        layer = agreement.get("layer") or {}
        reason = agreement.get("reason") or {}
        lines.extend(
            [
                "",
                "## Inter-Annotator Agreement",
                "",
                f"- layer pairs: `{layer.get('num_pairs')}`; kappa: `{layer.get('cohen_kappa')}`",
                f"- reason pairs: `{reason.get('num_pairs')}`; kappa: `{reason.get('cohen_kappa')}`",
            ]
        )
    return "\n".join(lines) + "\n"
