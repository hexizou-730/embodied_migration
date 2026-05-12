"""
Analyze benchmark run artifacts and export paper-ready tables.

Usage:
    python -m benchmark.analyze_results results/runs/stage4_full
    python -m benchmark.analyze_results              # analyze newest run

Outputs, for a single run:
    results/runs/<run_id>/tables/
      method_summary.csv/.tex
      robot_method_summary.csv/.tex
      task_family_method_summary.csv/.tex
      task_method_summary.csv
      scene_variant_method_summary.csv
      seed_method_summary.csv
      migration_score.csv
      paired_method_deltas.csv
      failure_breakdown.csv/.tex
      failure_cases.csv
      generated_code_features.csv
      code_changes_after_feedback.csv
      code_change_summary.csv
      analysis_report.md
"""
import argparse
import csv
import difflib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from benchmark.experiment_logging import (
    choose_failure_subtype,
    choose_failure_type,
    extract_error_excerpt,
    flatten_code_features,
    jsonable,
)


FEATURE_FIELDS = [
    "used_mobile_navigate_to",
    "used_mobile_is_reachable",
    "used_dual_arm_api",
    "used_dual_left_arm",
    "used_dual_right_arm",
    "used_dual_choose_arm",
    "used_dual_hold",
    "used_dual_coordinated_lift",
    "used_dual_coordinated_place",
    "used_pick_and_place",
    "used_pick",
    "used_place",
    "used_move_ee_to",
    "used_low_release_height",
    "checked_return_value",
    "used_numpy",
    "used_scene_get_names",
    "used_scene_get_position",
    "used_loop",
    "used_conditional",
    "used_refusal_ret_val",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run_dirs",
        nargs="*",
        help="Run directories containing trials/*.json. Defaults to newest results/runs/*.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <run_dir>/tables for one run.",
    )
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs] or [_latest_run_dir()]
    records = load_trial_records(run_dirs)
    if not records:
        raise SystemExit("No trial JSON files found.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif len(run_dirs) == 1:
        out_dir = run_dirs[0] / "tables"
    else:
        out_dir = Path("results") / "analysis" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = build_tables(records)
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)

    write_latex_table(
        out_dir / "method_summary.tex",
        tables["method_summary"],
        columns=[
            "canonical_mode",
            "n",
            "success_rate",
            "success_ci95",
            "mean_attempts",
            "recovered_after_feedback_rate",
            "exec_error_rate",
            "action_failure_rate",
            "check_failure_rate",
            "ret_val_failure_rate",
        ],
        caption="Overall ablation results by method.",
        label="tab:method-summary",
    )
    write_latex_table(
        out_dir / "robot_method_summary.tex",
        tables["robot_method_summary"],
        columns=["canonical_mode", "robot", "n", "success_rate", "mean_attempts"],
        caption="Ablation results by method and robot embodiment.",
        label="tab:robot-method-summary",
    )
    write_latex_table(
        out_dir / "task_family_method_summary.tex",
        tables["task_family_method_summary"],
        columns=["canonical_mode", "task_family", "n", "success_rate", "mean_attempts"],
        caption="Ablation results by task family.",
        label="tab:task-family-summary",
    )
    write_latex_table(
        out_dir / "failure_breakdown.tex",
        tables["failure_breakdown"],
        columns=["canonical_mode", "failure_type", "failure_subtype", "count", "share_of_failures"],
        caption="Failure type breakdown.",
        label="tab:failure-breakdown",
    )
    write_latex_table(
        out_dir / "migration_score.tex",
        tables["migration_score"],
        columns=["canonical_mode", "task_family", "migration_score", "migration_rate"],
        caption="Cross-embodiment migration score by method.",
        label="tab:migration-score",
    )
    write_latex_table(
        out_dir / "paired_method_deltas.tex",
        tables["paired_method_deltas"],
        columns=[
            "method",
            "n_matched_trials",
            "baseline_success_rate",
            "method_success_rate",
            "absolute_delta_pct_points",
            "net_improvements",
        ],
        caption="Paired method deltas relative to the few-shot baseline.",
        label="tab:paired-method-deltas",
    )
    write_report(out_dir / "analysis_report.md", run_dirs, tables)

    print(f"Wrote analysis tables to: {out_dir}")
    print(f"Main report: {out_dir / 'analysis_report.md'}")


def _latest_run_dir() -> Path:
    root = Path("results") / "runs"
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "trials").is_dir()
    ] if root.exists() else []
    if not candidates:
        raise SystemExit("No run directories found under results/runs.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_trial_records(run_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    records = []
    for run_dir in run_dirs:
        for path in sorted((run_dir / "trials").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["_trial_path"] = str(path)
            normalize_record(record, fallback_run_id=run_dir.name)
            records.append(record)
    return records


def normalize_record(record: Dict[str, Any], fallback_run_id: str) -> None:
    record.setdefault("run_id", fallback_run_id)
    record.setdefault("canonical_mode", record.get("mode", ""))
    record.setdefault("task_family", infer_task_family(str(record.get("task", ""))))
    record.setdefault("scene_variant", "fixed")
    record.setdefault("scene_seed", "")
    record.setdefault("attempt_count", len(record.get("attempts") or []))
    if record.get("success"):
        record["failure_type"] = ""
        record["failure_subtype"] = ""
        record["error_excerpt"] = record.get("error_excerpt", "")
    else:
        record["failure_type"] = record.get("failure_type") or choose_failure_type(record)
        record["failure_subtype"] = (
            record.get("failure_subtype") or choose_failure_subtype(record)
        )
        record["error_excerpt"] = record.get("error_excerpt") or extract_error_excerpt(record)


def infer_task_family(task_name: str) -> str:
    if task_name.startswith("arrange_") or task_name in {"mirror_layout", "sort_left_to_right"}:
        return "geometric"
    if task_name in {"wide_blue_to_tray", "collect_red_and_blue_to_tray"}:
        return "mobility"
    if task_name in {
        "hold_red_while_place_green",
        "lift_red_and_green_together",
        "lift_red_green_together_to_tray",
    }:
        return "bimanual"
    if task_name.startswith("refuse_"):
        return "refusal"
    return "basic"


def build_tables(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "method_summary": aggregate(records, ["canonical_mode"]),
        "robot_method_summary": aggregate(records, ["canonical_mode", "robot"]),
        "task_family_method_summary": aggregate(records, ["canonical_mode", "task_family"]),
        "task_method_summary": aggregate(records, ["canonical_mode", "task_family", "task"]),
        "scene_variant_method_summary": aggregate(records, ["canonical_mode", "scene_variant"]),
        "seed_method_summary": aggregate(records, ["canonical_mode", "scene_seed"]),
        "migration_score": migration_score(records),
        "paired_method_deltas": paired_method_deltas(records),
        "failure_breakdown": failure_breakdown(records),
        "failure_cases": failure_cases(records),
        "generated_code_features": generated_code_features(records),
        "code_changes_after_feedback": code_changes_after_feedback(records),
        "code_change_summary": code_change_summary(records),
    }


def aggregate(records: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(group_value(record.get(k, "")) for k in keys)].append(record)

    rows = []
    for key_values, items in sorted(groups.items()):
        n = len(items)
        successes = sum(1 for item in items if bool(item.get("success")))
        attempts = [int(item.get("attempt_count") or len(item.get("attempts") or [])) for item in items]
        failures = [item for item in items if not item.get("success")]
        recovered = sum(
            1 for item in items
            if item.get("success") and int(item.get("attempt_count") or 0) > 1
        )
        row = dict(zip(keys, key_values))
        row.update({
            "n": n,
            "successes": successes,
            "failures": n - successes,
            "success_rate": pct(successes, n),
            "success_ci95": ci95(successes, n),
            "mean_attempts": f"{mean(attempts):.2f}",
            "recovered_after_feedback": recovered,
            "recovered_after_feedback_rate": pct(recovered, n),
            "exec_error_rate": pct(sum(1 for x in failures if x.get("exec_error")), n),
            "action_failure_rate": pct(sum(1 for x in failures if x.get("action_failure")), n),
            "check_failure_rate": pct(sum(1 for x in failures if x.get("check_failure")), n),
            "ret_val_failure_rate": pct(sum(1 for x in failures if x.get("ret_val_failed")), n),
        })
        rows.append(row)
    return rows


def migration_score(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tasks that transfer across all evaluated embodiments for each method."""
    robots = sorted({record.get("robot", "") for record in records})
    methods = sorted({record.get("canonical_mode", "") for record in records})
    families = sorted({record.get("task_family", "") for record in records})
    rows = []
    for mode in methods:
        for family in ["all"] + families:
            subset = [
                r for r in records
                if r.get("canonical_mode") == mode
                and (family == "all" or r.get("task_family") == family)
            ]
            task_names = sorted({r.get("task", "") for r in subset})
            transferred = 0
            for task in task_names:
                per_robot_ok = []
                for robot in robots:
                    items = [
                        r for r in subset
                        if r.get("robot") == robot and r.get("task") == task
                    ]
                    if not items:
                        per_robot_ok.append(False)
                    else:
                        per_robot_ok.append(
                            mean([1.0 if x.get("success") else 0.0 for x in items]) > 0.5
                        )
                if all(per_robot_ok):
                    transferred += 1
            rows.append({
                "canonical_mode": mode,
                "task_family": family,
                "robots": ",".join(robots),
                "transferred_tasks": transferred,
                "total_tasks": len(task_names),
                "migration_score": f"{transferred}/{len(task_names)}",
                "migration_rate": pct(transferred, len(task_names)),
            })
    return rows


def paired_method_deltas(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare each method against fewshot on matched robot/task/seed cells."""
    baseline = "fewshot"
    methods = sorted({record.get("canonical_mode", "") for record in records if record.get("canonical_mode") != baseline})
    group_keys = ["robot", "task_family", "task", "scene_variant", "scene_seed", "trial_index"]
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = tuple(record.get(k, "") for k in group_keys)
        by_key[key][record.get("canonical_mode", "")] = record

    rows = []
    for method in methods:
        paired = [
            (items[baseline], items[method])
            for items in by_key.values()
            if baseline in items and method in items
        ]
        if not paired:
            continue
        base_success = sum(1 for base, _ in paired if base.get("success"))
        method_success = sum(1 for _, item in paired if item.get("success"))
        improved = sum(1 for base, item in paired if (not base.get("success")) and item.get("success"))
        regressed = sum(1 for base, item in paired if base.get("success") and (not item.get("success")))
        rows.append({
            "baseline": baseline,
            "method": method,
            "n_matched_trials": len(paired),
            "baseline_success_rate": pct(base_success, len(paired)),
            "method_success_rate": pct(method_success, len(paired)),
            "absolute_delta_pct_points": f"{100.0 * (method_success - base_success) / len(paired):+.1f}",
            "improved_cases": improved,
            "regressed_cases": regressed,
            "net_improvements": improved - regressed,
        })
    return rows


def failure_breakdown(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = [record for record in records if not record.get("success")]
    mode_counts = Counter(record.get("canonical_mode", "") for record in failures)
    counter = Counter(
        (
            record.get("canonical_mode", ""),
            record.get("failure_type", ""),
            record.get("failure_subtype", ""),
        )
        for record in failures
    )
    rows = []
    for (mode, failure_type, subtype), count in sorted(counter.items()):
        rows.append({
            "canonical_mode": mode,
            "failure_type": failure_type,
            "failure_subtype": subtype,
            "count": count,
            "share_of_failures": pct(count, mode_counts[mode]),
        })
    return rows


def group_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def failure_cases(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        if record.get("success"):
            continue
        attempts = list(record.get("attempts") or [])
        last = attempts[-1] if attempts else {}
        rows.append({
            "run_id": record.get("run_id", ""),
            "trial_id": record.get("trial_id", ""),
            "canonical_mode": record.get("canonical_mode", ""),
            "robot": record.get("robot", ""),
            "task_family": record.get("task_family", ""),
            "task": record.get("task", ""),
            "scene_variant": record.get("scene_variant", ""),
            "scene_seed": record.get("scene_seed", ""),
            "attempts": record.get("attempt_count", len(attempts)),
            "failure_type": record.get("failure_type", ""),
            "failure_subtype": record.get("failure_subtype", ""),
            "final_reason": record.get("final_reason", ""),
            "info": record.get("info", ""),
            "error_excerpt": record.get("error_excerpt", ""),
            "expected": compact_json(record.get("expected", {})),
            "actual": compact_json(record.get("actual", {})),
            "last_code_excerpt": excerpt(last.get("code", "")),
            "trial_path": record.get("_trial_path", ""),
        })
    return rows


def generated_code_features(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record.get("canonical_mode", ""), record.get("robot", ""))].append(record)

    rows = []
    for (mode, robot), items in sorted(groups.items()):
        feature_rows = [flatten_code_features(item.get("attempts") or []) for item in items]
        row = {
            "canonical_mode": mode,
            "robot": robot,
            "n": len(items),
            "success_rate": pct(sum(1 for item in items if item.get("success")), len(items)),
            "mean_lines_of_code": f"{mean([int(f.get('lines_of_code') or 0) for f in feature_rows]):.2f}",
        }
        for feature in FEATURE_FIELDS:
            row[f"{feature}_rate"] = pct(sum(1 for f in feature_rows if f.get(feature)), len(feature_rows))
        rows.append(row)
    return rows


def code_changes_after_feedback(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        attempts = list(record.get("attempts") or [])
        if len(attempts) < 2:
            continue
        first, last = attempts[0], attempts[-1]
        first_code = first.get("code", "")
        last_code = last.get("code", "")
        first_features = first.get("code_features", {})
        last_features = last.get("code_features", {})
        row = {
            "run_id": record.get("run_id", ""),
            "trial_id": record.get("trial_id", ""),
            "canonical_mode": record.get("canonical_mode", ""),
            "robot": record.get("robot", ""),
            "task_family": record.get("task_family", ""),
            "task": record.get("task", ""),
            "scene_variant": record.get("scene_variant", ""),
            "scene_seed": record.get("scene_seed", ""),
            "success": bool(record.get("success")),
            "attempts": len(attempts),
            "line_delta": int(last_features.get("lines_of_code") or 0) - int(first_features.get("lines_of_code") or 0),
            "sequence_similarity": f"{difflib.SequenceMatcher(None, first_code, last_code).ratio():.3f}",
            "first_code_excerpt": excerpt(first_code),
            "last_code_excerpt": excerpt(last_code),
        }
        for feature in FEATURE_FIELDS:
            before = bool(first_features.get(feature))
            after = bool(last_features.get(feature))
            row[f"added_{feature}"] = bool(after and not before)
            row[f"removed_{feature}"] = bool(before and not after)
        rows.append(row)
    return rows


def code_change_summary(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = code_changes_after_feedback(records)
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("canonical_mode", ""), row.get("robot", ""))].append(row)

    summary = []
    for (mode, robot), items in sorted(groups.items()):
        out = {
            "canonical_mode": mode,
            "robot": robot,
            "n_retry_trials": len(items),
            "retry_success_rate": pct(sum(1 for item in items if item.get("success")), len(items)),
            "mean_line_delta": f"{mean([int(item.get('line_delta') or 0) for item in items]):.2f}",
            "mean_sequence_similarity": f"{mean([float(item.get('sequence_similarity') or 0) for item in items]):.3f}",
        }
        for feature in FEATURE_FIELDS:
            out[f"added_{feature}_rate"] = pct(
                sum(1 for item in items if item.get(f"added_{feature}")),
                len(items),
            )
        summary.append(out)
    return summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k, "")) for k in fieldnames})


def ordered_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    seen = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen or ["empty"]


def write_latex_table(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    caption: str,
    label: str,
) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\toprule",
        " & ".join(latex_escape(col.replace("_", " ")) for col in columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(latex_escape(str(row.get(col, ""))) for col in columns) + " \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, run_dirs: Sequence[Path], tables: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = [
        "# Benchmark Analysis Report",
        "",
        "Run directories:",
    ]
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.extend([
        "",
        "## Method Summary",
        "",
        markdown_table(
            tables["method_summary"],
            ["canonical_mode", "n", "success_rate", "success_ci95", "mean_attempts", "recovered_after_feedback_rate"],
        ),
        "",
        "## Task Family Summary",
        "",
        markdown_table(
            tables["task_family_method_summary"],
            ["canonical_mode", "task_family", "n", "success_rate", "mean_attempts"],
        ),
        "",
        "## Failure Breakdown",
        "",
        markdown_table(
            tables["failure_breakdown"],
            ["canonical_mode", "failure_type", "failure_subtype", "count", "share_of_failures"],
        ),
        "",
        "## Generated Code Differences After Feedback",
        "",
        markdown_table(
            tables["code_change_summary"],
            ["canonical_mode", "robot", "n_retry_trials", "retry_success_rate", "mean_line_delta", "mean_sequence_similarity"],
        ),
        "",
        "## Migration Score",
        "",
        markdown_table(
            tables["migration_score"],
            ["canonical_mode", "task_family", "migration_score", "migration_rate"],
        ),
        "",
        "## Paired Method Delta vs Few-Shot",
        "",
        markdown_table(
            tables["paired_method_deltas"],
            ["method", "n_matched_trials", "baseline_success_rate", "method_success_rate", "absolute_delta_pct_points", "net_improvements"],
        ),
        "",
        "See CSV files in this directory for full failure cases and per-feature code analysis.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], limit: int = 20) -> str:
    if not rows:
        return "_No rows._"
    shown = list(rows[:limit])
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in shown
    ]
    if len(rows) > limit:
        body.append("| " + " | ".join(["..."] * len(columns)) + " |")
    return "\n".join([header, sep] + body)


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{100.0 * numerator / denominator:.1f}%"


def ci95(successes: int, n: int) -> str:
    if n <= 0:
        return "[0.0%, 0.0%]"
    # Wilson score interval, z=1.96.
    z = 1.96
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return f"[{100 * lo:.1f}%, {100 * hi:.1f}%]"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def compact_json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)


def excerpt(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def latex_escape(text: str) -> str:
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
