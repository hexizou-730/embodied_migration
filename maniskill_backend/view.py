"""Pretty-print JSONL trial logs for humans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_records(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        lines = lines[-limit:]
    return [json.loads(line) for line in lines]


def record_to_md(record: Dict[str, Any], index: int) -> str:
    info = record.get("info") or {}
    fields = [
        ("task", record.get("task_id", "")),
        ("task_cn", info.get("task_name_cn", "")),
        ("target", record.get("target_robot", "")),
        ("method", record.get("method", "")),
        ("success", record.get("success", "")),
        ("failure_type", record.get("failure_type", "")),
        ("used_llm", info.get("used_llm", "")),
        ("message", record.get("message", "")),
    ]
    lines = [f"## Trial {index}", ""]
    for key, value in fields:
        lines.append(f"- **{key}**: `{value}`")

    card = info.get("capability_card")
    if card:
        lines.extend(["", "### Capability Card", "", "```text", str(card).strip(), "```", ""])

    failure_report = record.get("failure_report")
    if failure_report:
        lines.extend(
            ["### Failure Report", "", "```text", str(failure_report).strip(), "```", ""]
        )

    report_source_log = info.get("report_source_log")
    if report_source_log:
        source_method = info.get("report_source_method", "previous attempt")
        lines.extend([f"### Report Source Log ({source_method})", "", "```text"])
        lines.extend(_format_execution_log(report_source_log))
        lines.extend(["```", ""])

    execution_log = info.get("execution_log")
    if execution_log:
        lines.extend(["### Execution Log", "", "```text"])
        lines.extend(_format_execution_log(execution_log))
        lines.extend(["```", ""])

    lines.extend(["### Generated Code", "", "```python"])
    lines.append((record.get("generated_code") or "").strip())
    lines.extend(["```", ""])

    raw_text = info.get("llm_raw_text")
    if raw_text:
        lines.extend(["### Raw LLM Text", "", "```text", str(raw_text).strip(), "```", ""])

    return "\n".join(lines)


def _format_execution_log(execution_log: Any) -> List[str]:
    lines = []
    if not isinstance(execution_log, list):
        return lines
    for event in execution_log:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(args.items()))
        line = (
            f"{event.get('step', '?')}. {event.get('api', 'unknown')}({arg_text}) "
            f"-> {event.get('result')} ok={event.get('ok')}"
        )
        if event.get("failure_type"):
            line += f" [{event.get('failure_type')}]"
        if event.get("message"):
            line += f" {event.get('message')}"
        lines.append(line)
    return lines


def records_to_md(records: Iterable[Dict[str, Any]]) -> str:
    return "\n---\n\n".join(
        record_to_md(record, index) for index, record in enumerate(records, start=1)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format JSONL trial logs.")
    parser.add_argument("path", nargs="?", default="results/real_trials.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", default="results/real_trials.md")
    parser.add_argument("--json", action="store_true", help="Pretty-print JSON instead of Markdown.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = load_records(Path(args.path), limit=args.limit)
    if args.json:
        text = json.dumps(records, ensure_ascii=False, indent=2)
    else:
        text = records_to_md(records)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote: {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
