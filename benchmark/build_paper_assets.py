"""
Build paper-ready assets from benchmark analysis tables.

Usage:
    python -m benchmark.build_paper_assets results/runs/stage5_mobile_dual_seeded
    python -m benchmark.build_paper_assets results/runs/stage5_mobile_dual_seeded --analyze-first

Outputs:
    results/runs/<run_id>/paper_assets/
      experiment_manifest.json
      paper_results_section.md
      figure_index.md
      fig_method_success.svg
      fig_robot_method_success.svg
      fig_task_family_success.svg
      fig_migration_score.svg
      table_includes.tex

The script uses only the Python standard library and writes SVG directly, so it
does not add plotting dependencies to the project.
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


METHOD_ORDER = ["api", "fewshot", "card", "failure", "card_failure", "baseline", "b", "ba"]
METHOD_LABELS = {
    "api": "API",
    "fewshot": "Few-shot",
    "card": "Card",
    "failure": "Failure",
    "card_failure": "Card+Failure",
    "baseline": "Baseline",
    "b": "Failure",
    "ba": "Card+Failure",
}
METHOD_COLORS = {
    "api": "#7a869a",
    "fewshot": "#4c78a8",
    "card": "#59a14f",
    "failure": "#f28e2b",
    "card_failure": "#e15759",
    "baseline": "#4c78a8",
    "b": "#f28e2b",
    "ba": "#e15759",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory containing tables/*.csv.")
    ap.add_argument(
        "--analyze-first",
        action="store_true",
        help="Run benchmark.analyze_results before generating paper assets.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <run_dir>/paper_assets.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if args.analyze_first:
        subprocess.run(
            ["python", "-m", "benchmark.analyze_results", str(run_dir)],
            check=True,
        )

    tables_dir = run_dir / "tables"
    if not tables_dir.exists():
        raise SystemExit(
            f"Missing {tables_dir}. Run python -m benchmark.analyze_results {run_dir} first."
        )

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "paper_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(tables_dir)
    manifest = build_manifest(run_dir, tables)
    write_json(out_dir / "experiment_manifest.json", manifest)
    build_figures(out_dir, tables)
    write_table_includes(out_dir / "table_includes.tex", run_dir)
    write_results_section(out_dir / "paper_results_section.md", manifest, tables)
    write_figure_index(out_dir / "figure_index.md")

    print(f"Wrote paper assets to: {out_dir}")
    print(f"Results-section draft: {out_dir / 'paper_results_section.md'}")


def load_tables(tables_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    tables = {}
    for path in sorted(tables_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            tables[path.stem] = list(csv.DictReader(f))
    return tables


def build_manifest(run_dir: Path, tables: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    metadata = read_json_if_exists(run_dir / "metadata.json")
    method_rows = tables.get("method_summary", [])
    robots = metadata.get("robots") or sorted({r.get("robot", "") for r in tables.get("robot_method_summary", [])})
    tasks = metadata.get("task_names") or sorted({r.get("task", "") for r in tables.get("task_method_summary", [])})
    modes = metadata.get("modes") or [r.get("canonical_mode", "") for r in method_rows]
    scene_variants = sorted({
        row.get("scene_variant", "")
        for row in tables.get("scene_variant_method_summary", [])
        if row.get("scene_variant", "")
    })
    scene_seeds = sorted({
        row.get("scene_seed", "")
        for row in tables.get("seed_method_summary", [])
        if row.get("scene_seed", "")
    }, key=natural_sort_value)
    trial_count = sum(int(row.get("n") or 0) for row in method_rows)
    return {
        "run_dir": str(run_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "robots": robots,
        "modes": modes,
        "tasks": tasks,
        "n_tasks": len(tasks),
        "n_robots": len(robots),
        "n_trials": metadata.get("n_trials", ""),
        "trial_count": trial_count,
        "scene_variant": metadata.get("scene_variant", "") or ",".join(scene_variants),
        "seed_base": metadata.get("seed_base", ""),
        "scene_seeds": metadata.get("scene_seeds", []) or scene_seeds,
        "llm_model": metadata.get("llm_model", ""),
        "llm_temperature": metadata.get("llm_temperature", ""),
        "llm_cache_enabled": metadata.get("llm_cache_enabled", ""),
        "llm_cache_dir": metadata.get("llm_cache_dir", ""),
        "best_method": best_method(method_rows),
        "method_summary": method_rows,
    }


def best_method(method_rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    if not method_rows:
        return {}
    return max(
        method_rows,
        key=lambda row: (parse_percent(row.get("success_rate")), -float(row.get("mean_attempts") or 0)),
    )


def build_figures(out_dir: Path, tables: Dict[str, List[Dict[str, str]]]) -> None:
    method_rows = sort_method_rows(tables.get("method_summary", []))
    write_bar_chart(
        out_dir / "fig_method_success.svg",
        title="Overall Task Success by Method",
        rows=[
            {
                "label": method_label(row.get("canonical_mode", "")),
                "value": parse_percent(row.get("success_rate")),
                "color": method_color(row.get("canonical_mode", "")),
            }
            for row in method_rows
        ],
        y_label="Success Rate (%)",
    )

    robot_rows = tables.get("robot_method_summary", [])
    write_grouped_bar_chart(
        out_dir / "fig_robot_method_success.svg",
        title="Success by Robot Embodiment",
        rows=[
            {
                "group": row.get("robot", ""),
                "series": method_label(row.get("canonical_mode", "")),
                "series_key": row.get("canonical_mode", ""),
                "value": parse_percent(row.get("success_rate")),
            }
            for row in robot_rows
        ],
        y_label="Success Rate (%)",
    )

    family_rows = tables.get("task_family_method_summary", [])
    write_grouped_bar_chart(
        out_dir / "fig_task_family_success.svg",
        title="Success by Task Family",
        rows=[
            {
                "group": row.get("task_family", ""),
                "series": method_label(row.get("canonical_mode", "")),
                "series_key": row.get("canonical_mode", ""),
                "value": parse_percent(row.get("success_rate")),
            }
            for row in family_rows
        ],
        y_label="Success Rate (%)",
    )

    migration_rows = [
        row for row in tables.get("migration_score", [])
        if row.get("task_family") == "all"
    ]
    migration_rows = sort_method_rows(migration_rows)
    write_bar_chart(
        out_dir / "fig_migration_score.svg",
        title="Cross-Embodiment Migration Score",
        rows=[
            {
                "label": method_label(row.get("canonical_mode", "")),
                "value": parse_percent(row.get("migration_rate")),
                "color": method_color(row.get("canonical_mode", "")),
                "note": row.get("migration_score", ""),
            }
            for row in migration_rows
        ],
        y_label="Transferred Tasks (%)",
    )


def write_bar_chart(path: Path, title: str, rows: Sequence[Dict[str, Any]], y_label: str) -> None:
    width = 900
    height = 520
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 110
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    rows = list(rows)
    n = max(1, len(rows))
    slot = chart_w / n
    bar_w = min(88, slot * 0.58)
    parts = svg_header(width, height)
    parts.append(text(width / 2, 32, title, size=24, weight="700", anchor="middle"))
    parts.extend(axis_parts(margin_left, margin_top, chart_w, chart_h, y_label))
    for i, row in enumerate(rows):
        value = clamp(float(row.get("value", 0)), 0, 100)
        x = margin_left + i * slot + (slot - bar_w) / 2
        bar_h = chart_h * value / 100
        y = margin_top + chart_h - bar_h
        parts.append(rect(x, y, bar_w, bar_h, row.get("color", "#4c78a8")))
        parts.append(text(x + bar_w / 2, y - 8, f"{value:.1f}%", size=13, anchor="middle"))
        if row.get("note"):
            parts.append(text(x + bar_w / 2, y - 25, str(row["note"]), size=12, anchor="middle", fill="#555"))
        parts.append(rotated_text(x + bar_w / 2, height - 42, row.get("label", ""), size=14))
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_bar_chart(
    path: Path,
    title: str,
    rows: Sequence[Dict[str, Any]],
    y_label: str,
) -> None:
    width = 1000
    height = 560
    margin_left = 80
    margin_right = 40
    margin_top = 90
    margin_bottom = 120
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    groups = sorted({row.get("group", "") for row in rows if row.get("group", "")})
    series_keys = sort_methods({row.get("series_key", "") for row in rows if row.get("series_key", "")})
    values = {
        (row.get("group", ""), row.get("series_key", "")): clamp(float(row.get("value", 0)), 0, 100)
        for row in rows
    }
    group_slot = chart_w / max(1, len(groups))
    inner_gap = 4
    bar_w = min(28, (group_slot * 0.72) / max(1, len(series_keys)) - inner_gap)
    parts = svg_header(width, height)
    parts.append(text(width / 2, 32, title, size=24, weight="700", anchor="middle"))
    parts.extend(axis_parts(margin_left, margin_top, chart_w, chart_h, y_label))
    parts.extend(legend_parts(width - 360, 58, series_keys))

    for gi, group in enumerate(groups):
        group_x = margin_left + gi * group_slot
        bars_w = len(series_keys) * (bar_w + inner_gap) - inner_gap
        start_x = group_x + (group_slot - bars_w) / 2
        for si, key in enumerate(series_keys):
            value = values.get((group, key), 0)
            x = start_x + si * (bar_w + inner_gap)
            bar_h = chart_h * value / 100
            y = margin_top + chart_h - bar_h
            parts.append(rect(x, y, bar_w, bar_h, method_color(key)))
        parts.append(text(group_x + group_slot / 2, height - 56, group, size=15, anchor="middle"))
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def axis_parts(x: float, y: float, w: float, h: float, y_label: str) -> List[str]:
    parts = []
    for tick in range(0, 101, 20):
        ty = y + h - h * tick / 100
        parts.append(line(x, ty, x + w, ty, "#e6e8ee", width=1))
        parts.append(text(x - 12, ty + 4, str(tick), size=12, anchor="end", fill="#4a4f5c"))
    parts.append(line(x, y, x, y + h, "#222", width=1.5))
    parts.append(line(x, y + h, x + w, y + h, "#222", width=1.5))
    parts.append(rotated_text(22, y + h / 2, y_label, size=14, angle=-90))
    return parts


def legend_parts(x: float, y: float, series_keys: Sequence[str]) -> List[str]:
    parts = []
    for i, key in enumerate(series_keys):
        lx = x + (i % 3) * 115
        ly = y + (i // 3) * 24
        parts.append(rect(lx, ly - 11, 14, 14, method_color(key)))
        parts.append(text(lx + 20, ly, method_label(key), size=13, fill="#333"))
    return parts


def write_table_includes(path: Path, run_dir: Path) -> None:
    rel = f"{run_dir}/tables"
    lines = [
        "% Auto-generated table includes for the paper.",
        f"\\input{{{rel}/method_summary.tex}}",
        f"\\input{{{rel}/robot_method_summary.tex}}",
        f"\\input{{{rel}/task_family_method_summary.tex}}",
        f"\\input{{{rel}/migration_score.tex}}",
        f"\\input{{{rel}/paired_method_deltas.tex}}",
        f"\\input{{{rel}/failure_breakdown.tex}}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_section(
    path: Path,
    manifest: Dict[str, Any],
    tables: Dict[str, List[Dict[str, str]]],
) -> None:
    best = manifest.get("best_method") or {}
    method_rows = sort_method_rows(tables.get("method_summary", []))
    migration_rows = [
        row for row in tables.get("migration_score", [])
        if row.get("task_family") == "all"
    ]
    paired_rows = tables.get("paired_method_deltas", [])
    lines = [
        "# Auto-Drafted Results Section",
        "",
        "## Main Results",
        "",
        (
            f"We evaluated {count_phrase(manifest.get('n_robots'), 'robot embodiment')} on "
            f"{count_phrase(manifest.get('n_tasks'), 'task')} across "
            f"{count_phrase(manifest.get('trial_count'), 'logged trial')}. "
            f"The scene variant was `{manifest.get('scene_variant')}` with seeds "
            f"`{manifest.get('scene_seeds')}`. The LLM model was "
            f"`{manifest.get('llm_model')}` at temperature "
            f"`{manifest.get('llm_temperature')}`."
        ),
        "",
        (
            f"The strongest method in the aggregate table is "
            f"`{best.get('canonical_mode', '<missing>')}`, with success rate "
            f"{best.get('success_rate', '<missing>')} and mean attempts "
            f"{best.get('mean_attempts', '<missing>')}. "
            "Insert Table~\\ref{tab:method-summary} here."
        ),
        "",
        "Method summary:",
        "",
        markdown_table(
            method_rows,
            ["canonical_mode", "n", "success_rate", "success_ci95", "mean_attempts", "recovered_after_feedback_rate"],
        ),
        "",
        "## Cross-Embodiment Migration",
        "",
        (
            "The migration score counts a task as transferred only when all evaluated "
            "embodiments solve it above the success threshold. Insert "
            "Table~\\ref{tab:migration-score} and Figure `fig_migration_score.svg` here."
        ),
        "",
        markdown_table(
            sort_method_rows(migration_rows),
            ["canonical_mode", "task_family", "migration_score", "migration_rate"],
        ),
        "",
        "## Paired Ablation Deltas",
        "",
        (
            "Matched trials compare each method against `fewshot` on the same "
            "robot, task, scene variant, seed, and trial index. This makes the "
            "ablation less sensitive to random initial layouts."
        ),
        "",
        markdown_table(
            paired_rows,
            ["method", "n_matched_trials", "baseline_success_rate", "method_success_rate", "absolute_delta_pct_points", "net_improvements"],
        ),
        "",
        "## Failure Analysis",
        "",
        (
            "Use `failure_breakdown.csv` for quantitative taxonomy results and "
            "`failure_cases.csv` for qualitative examples. The qualitative section "
            "should quote generated code only sparingly and should focus on the "
            "difference between the first and corrected attempts."
        ),
        "",
        "## Generated-Code Adaptation",
        "",
        (
            "Use `code_changes_after_feedback.csv` to identify representative cases "
            "where Failure Reports add navigation, explicit dual-arm APIs, "
            "low release height, NumPy geometry, or explicit refusal. "
            "Use `generated_code_features.csv` for aggregate "
            "feature-rate claims."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure_index(path: Path) -> None:
    lines = [
        "# Figure Index",
        "",
        "- `fig_method_success.svg`: overall success rate by method.",
        "- `fig_robot_method_success.svg`: success rate split by robot embodiment.",
        "- `fig_task_family_success.svg`: success rate split by task family.",
        "- `fig_migration_score.svg`: cross-embodiment migration score.",
        "",
        "Suggested paper placement:",
        "",
        "- Main experiment figure: `fig_method_success.svg`.",
        "- Embodiment analysis: `fig_robot_method_success.svg`.",
        "- Task generalization analysis: `fig_task_family_success.svg`.",
        "- Migration claim figure: `fig_migration_score.svg`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], limit: int = 12) -> str:
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


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }</style>',
    ]


def text(
    x: float,
    y: float,
    value: str,
    size: int = 12,
    fill: str = "#111",
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape_xml(value)}</text>'
    )


def rotated_text(
    x: float,
    y: float,
    value: str,
    size: int = 12,
    angle: int = -35,
    fill: str = "#111",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="middle" transform="rotate({angle} {x:.1f} {y:.1f})">'
        f'{escape_xml(value)}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" rx="2"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def sort_method_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        list(rows),
        key=lambda row: method_sort_key(row.get("canonical_mode", "")),
    )


def sort_methods(methods: Iterable[str]) -> List[str]:
    return sorted(set(methods), key=method_sort_key)


def method_sort_key(method: str) -> tuple:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (len(METHOD_ORDER), method)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#6b7280")


def parse_percent(value: Any) -> float:
    text_value = str(value or "0").strip().replace("%", "")
    try:
        return float(text_value)
    except ValueError:
        return 0.0


def natural_sort_value(value: Any) -> tuple:
    text_value = str(value)
    try:
        return (0, int(text_value))
    except ValueError:
        return (1, text_value)


def count_phrase(count: Any, noun: str) -> str:
    try:
        n = int(count)
    except (TypeError, ValueError):
        return f"{count} {noun}s"
    suffix = "" if n == 1 else "s"
    return f"{n} {noun}{suffix}"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def escape_xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
