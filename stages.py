"""Extract and validate the stage representation used by selective migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from maniskill_backend.stage_extraction import extract_stage_program, validate_stage_program


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    study = json.loads(args.study.read_text(encoding="utf-8"))
    trial = json.loads(args.trial.read_text(encoding="utf-8"))
    source_path = Path(study["source"]["program_evidence"])
    first_snapshot = (trial.get("state_trace") or [{}])[0]
    known_entities = [
        name for name in (first_snapshot.get("entities") or {}) if not name.startswith("segmentation_id_map")
    ]
    payload = extract_stage_program(
        source_path,
        task_id=study["task"]["task_id"],
        source_robot=study["source"]["robot_uid"],
        source_control_mode=study["source"]["control_mode"],
        known_entities=known_entities,
        source_action_trace=trial.get("source_action_trace") or [],
        official_evaluation=trial.get("official_evaluation") or {},
    )
    evaluation_only = study.get("manual_evaluation_only") or {}
    expected = evaluation_only.get("expected_stage_order") or []
    errors = validate_stage_program(payload, expected)
    expected_entities = evaluation_only.get("expected_entities") or {}
    for stage in payload["stages"]:
        stage_id = stage["stage_id"]
        if stage_id in expected_entities and stage["entities"] != expected_entities[stage_id]:
            errors.append(
                f"{stage_id}: entities expected {expected_entities[stage_id]!r}, got {stage['entities']!r}"
            )
    payload["validation"] = {
        "ok": not errors,
        "errors": errors,
        "manual_evaluation_only": evaluation_only,
    }
    payload["input_evidence"] = {"study": str(args.study), "successful_trial": str(args.trial)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"stages = {', '.join(payload['stage_order'])}")
    print(f"trace = {payload['trace_summary']['mapped_action_count']}/{payload['trace_summary']['action_count']} actions mapped")
    print(f"valid = {not errors}")
    print(f"output = {args.output}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
