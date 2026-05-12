"""
Build a qualitative casebook from benchmark trial JSON files.

Usage:
    python -m benchmark.build_casebook results/runs/stage7_mobile_dual_seeded

Outputs:
    results/runs/<run_id>/casebook/
      qualitative_cases.csv
      qualitative_casebook.md
      qualitative_casebook.tex

The casebook is meant for the paper's failure-case analysis and generated-code
adaptation analysis. It selects a small, deterministic set of representative
cases and includes first/final code snippets, code diffs, and Failure Report
excerpts when available.
"""
import argparse
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from benchmark.experiment_logging import choose_failure_subtype, choose_failure_type


FEATURES_OF_INTEREST = [
    "used_mobile_navigate_to",
    "used_mobile_is_reachable",
    "used_dual_arm_api",
    "used_dual_choose_arm",
    "used_dual_hold",
    "used_low_release_height",
    "used_numpy",
    "used_loop",
    "used_conditional",
    "used_refusal_ret_val",
]


CASE_CATEGORIES = [
    "successful_feedback_recovery",
    "added_mobile_navigation",
    "added_dual_arm_api",
    "added_low_release_height",
    "added_numpy_geometry",
    "refusal_success",
    "persistent_failure",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Benchmark run directory containing trials/*.json.")
    ap.add_argument("--max-per-category", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    records = load_records(run_dir)
    if not records:
        raise SystemExit(f"No trial JSON files found under {run_dir / 'trials'}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "casebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = select_cases(records, max_per_category=args.max_per_category)
    write_cases_csv(out_dir / "qualitative_cases.csv", cases)
    write_casebook_md(out_dir / "qualitative_casebook.md", cases)
    write_casebook_tex(out_dir / "qualitative_casebook.tex", cases)

    print(f"Wrote qualitative casebook to: {out_dir}")
    print(f"Selected cases: {len(cases)}")


def load_records(run_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted((run_dir / "trials").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_trial_path"] = str(path)
        normalize_record(record)
        records.append(record)
    return records


def normalize_record(record: Dict[str, Any]) -> None:
    record.setdefault("canonical_mode", record.get("mode", ""))
    record.setdefault("task_family", infer_task_family(str(record.get("task", ""))))
    record.setdefault("scene_variant", "fixed")
    record.setdefault("scene_seed", "")
    if record.get("success"):
        record["failure_type"] = ""
        record["failure_subtype"] = ""
    else:
        record["failure_type"] = record.get("failure_type") or choose_failure_type(record)
        record["failure_subtype"] = record.get("failure_subtype") or choose_failure_subtype(record)


def infer_task_family(task_name: str) -> str:
    if task_name.startswith("arrange_") or task_name in {"mirror_layout", "sort_left_to_right"}:
        return "geometric"
    if task_name in {"wide_blue_to_tray", "collect_red_and_blue_to_tray"}:
        return "mobility"
    if task_name in {"hold_red_while_place_green", "lift_red_and_green_together"}:
        return "bimanual"
    if task_name.startswith("refuse_"):
        return "refusal"
    return "basic"


def select_cases(records: Sequence[Dict[str, Any]], max_per_category: int) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        categories = categories_for_record(record)
        for category in categories:
            buckets[category].append(record)

    cases = []
    seen_trial_ids = set()
    for category in CASE_CATEGORIES:
        ranked = sorted(
            buckets.get(category, []),
            key=lambda record: case_rank(record, category),
        )
        picked = 0
        for record in ranked:
            trial_id = record.get("trial_id", "")
            # Allow the same trial to appear in at most one category, so the
            # casebook remains compact and diverse.
            if trial_id in seen_trial_ids:
                continue
            cases.append(case_from_record(record, category, len(cases) + 1))
            seen_trial_ids.add(trial_id)
            picked += 1
            if picked >= max_per_category:
                break
    return cases


def categories_for_record(record: Dict[str, Any]) -> List[str]:
    attempts = attempts_for(record)
    categories = []
    if record.get("success") and len(attempts) > 1:
        categories.append("successful_feedback_recovery")
    if feature_added(attempts, "used_mobile_navigate_to"):
        categories.append("added_mobile_navigation")
    if feature_added(attempts, "used_dual_arm_api"):
        categories.append("added_dual_arm_api")
    if feature_added(attempts, "used_low_release_height"):
        categories.append("added_low_release_height")
    if feature_added(attempts, "used_numpy"):
        categories.append("added_numpy_geometry")
    if record.get("task_family") == "refusal" and record.get("success"):
        categories.append("refusal_success")
    if not record.get("success"):
        categories.append("persistent_failure")
    return categories


def case_rank(record: Dict[str, Any], category: str) -> tuple:
    attempts = attempts_for(record)
    # Prefer card_failure/failure because those best expose adaptation, then
    # mobile cases, then shorter examples for readability.
    mode_rank = {
        "card_failure": 0,
        "failure": 1,
        "card": 2,
        "fewshot": 3,
        "api": 4,
    }.get(record.get("canonical_mode", ""), 5)
    robot_rank = 0 if record.get("robot") == "mobile" else 1
    return (
        mode_rank,
        robot_rank,
        -len(attempts),
        record.get("task_family", ""),
        record.get("task", ""),
        record.get("trial_id", ""),
    )


def case_from_record(record: Dict[str, Any], category: str, case_index: int) -> Dict[str, Any]:
    attempts = attempts_for(record)
    first = attempts[0] if attempts else {}
    final = attempts[-1] if attempts else {}
    first_code = first.get("code", "")
    final_code = final.get("code", "")
    case_id = f"case_{case_index:02d}_{slug(category)}"
    return {
        "case_id": case_id,
        "category": category,
        "trial_id": record.get("trial_id", ""),
        "run_id": record.get("run_id", ""),
        "canonical_mode": record.get("canonical_mode", ""),
        "robot": record.get("robot", ""),
        "task_family": record.get("task_family", ""),
        "task": record.get("task", ""),
        "scene_variant": record.get("scene_variant", ""),
        "scene_seed": record.get("scene_seed", ""),
        "success": bool(record.get("success")),
        "attempts": len(attempts),
        "failure_type": record.get("failure_type", ""),
        "failure_subtype": record.get("failure_subtype", ""),
        "final_reason": record.get("final_reason", ""),
        "info": record.get("info", ""),
        "adaptation_summary": adaptation_summary(first, final, record),
        "failure_report_excerpt": failure_report_excerpt(final.get("prompt", "")),
        "first_code": first_code,
        "final_code": final_code,
        "code_diff": code_diff(first_code, final_code),
        "first_code_excerpt": excerpt(first_code, 360),
        "final_code_excerpt": excerpt(final_code, 360),
        "trial_path": record.get("_trial_path", ""),
    }


def adaptation_summary(first: Dict[str, Any], final: Dict[str, Any], record: Dict[str, Any]) -> str:
    before = first.get("code_features", {}) or {}
    after = final.get("code_features", {}) or {}
    added = [feature for feature in FEATURES_OF_INTEREST if after.get(feature) and not before.get(feature)]
    removed = [feature for feature in FEATURES_OF_INTEREST if before.get(feature) and not after.get(feature)]
    parts = []
    if len(attempts_for(record)) > 1:
        parts.append(f"retry_count={len(attempts_for(record)) - 1}")
    if added:
        parts.append("added=" + ",".join(added))
    if removed:
        parts.append("removed=" + ",".join(removed))
    if record.get("failure_subtype"):
        parts.append(f"final_failure_subtype={record.get('failure_subtype')}")
    if not parts:
        parts.append("single_attempt_or_no_feature_change")
    return "; ".join(parts)


def attempts_for(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(record.get("attempts") or [])


def feature_added(attempts: Sequence[Dict[str, Any]], feature: str) -> bool:
    if len(attempts) < 2:
        return False
    first = attempts[0].get("code_features", {}) or {}
    final = attempts[-1].get("code_features", {}) or {}
    return bool(final.get(feature)) and not bool(first.get(feature))


def failure_report_excerpt(prompt: str, max_chars: int = 1200) -> str:
    marker = "# Failure Report"
    idx = prompt.find(marker)
    if idx < 0:
        return ""
    tail = prompt[idx:]
    next_instruction = tail.find("\n## Instruction")
    if next_instruction >= 0:
        tail = tail[:next_instruction]
    return excerpt(tail, max_chars)


def code_diff(first_code: str, final_code: str, max_lines: int = 80) -> str:
    if first_code == final_code:
        return ""
    lines = list(
        difflib.unified_diff(
            first_code.splitlines(),
            final_code.splitlines(),
            fromfile="first_attempt.py",
            tofile="final_attempt.py",
            lineterm="",
        )
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["... diff truncated ..."]
    return "\n".join(lines)


def write_cases_csv(path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "category",
        "trial_id",
        "canonical_mode",
        "robot",
        "task_family",
        "task",
        "scene_variant",
        "scene_seed",
        "success",
        "attempts",
        "failure_type",
        "failure_subtype",
        "adaptation_summary",
        "first_code_excerpt",
        "final_code_excerpt",
        "trial_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: case.get(field, "") for field in fields})


def write_casebook_md(path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Qualitative Casebook",
        "",
        "This file is auto-generated from trial JSON artifacts. Use these cases",
        "for the paper's failure-case analysis and generated-code adaptation analysis.",
        "",
        "## Selected Cases",
        "",
    ]
    if not cases:
        lines.append("_No representative cases were found._")
    for case in cases:
        lines.extend(render_case_md(case))
    path.write_text("\n".join(lines), encoding="utf-8")


def render_case_md(case: Dict[str, Any]) -> List[str]:
    lines = [
        f"### {case['case_id']}: {case['category']}",
        "",
        f"- Trial: `{case['trial_id']}`",
        f"- Method / robot / task: `{case['canonical_mode']}` / `{case['robot']}` / `{case['task']}`",
        f"- Task family / seed: `{case['task_family']}` / `{case['scene_seed']}`",
        f"- Success / attempts: `{case['success']}` / `{case['attempts']}`",
        f"- Failure type: `{case['failure_type']}` / `{case['failure_subtype']}`",
        f"- Adaptation summary: `{case['adaptation_summary']}`",
        "",
    ]
    if case["failure_report_excerpt"]:
        lines.extend([
            "Failure Report excerpt:",
            "",
            "```text",
            case["failure_report_excerpt"],
            "```",
            "",
        ])
    lines.extend([
        "First attempt code:",
        "",
        "```python",
        case["first_code"].strip(),
        "```",
        "",
        "Final attempt code:",
        "",
        "```python",
        case["final_code"].strip(),
        "```",
        "",
    ])
    if case["code_diff"]:
        lines.extend([
            "Code diff:",
            "",
            "```diff",
            case["code_diff"],
            "```",
            "",
        ])
    return lines


def write_casebook_tex(path: Path, cases: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated qualitative case snippets.",
        "% Keep only the strongest cases in the final paper.",
        "",
    ]
    if not cases:
        lines.append("% No cases selected.")
    for case in cases:
        lines.extend(render_case_tex(case))
    path.write_text("\n".join(lines), encoding="utf-8")


def render_case_tex(case: Dict[str, Any]) -> List[str]:
    lines = [
        f"\\paragraph{{{latex_escape(case['case_id'])}: {latex_escape(case['category'])}.}}",
        (
            f"Method: \\texttt{{{latex_escape(case['canonical_mode'])}}}; "
            f"robot: \\texttt{{{latex_escape(case['robot'])}}}; "
            f"task: \\texttt{{{latex_escape(case['task'])}}}; "
            f"attempts: {case['attempts']}. "
            f"Adaptation: {latex_escape(case['adaptation_summary'])}."
        ),
        "",
        "\\begin{verbatim}",
        snippet_for_verbatim(case["code_diff"] or case["final_code"], 1800),
        "\\end{verbatim}",
        "",
    ]
    return lines


def excerpt(value: Any, max_chars: int = 500) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def snippet_for_verbatim(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) > max_chars:
        return text[: max_chars - 24] + "\n... snippet truncated ..."
    return text


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "case"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    main()
