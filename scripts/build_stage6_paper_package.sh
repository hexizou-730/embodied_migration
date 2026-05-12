#!/usr/bin/env bash
set -euo pipefail

# Build tables + paper assets for a completed benchmark run.
#
# Usage:
#   bash scripts/build_stage6_paper_package.sh stage5_mobile_dual_seeded
#   bash scripts/build_stage6_paper_package.sh results/runs/stage5_mobile_dual_seeded
#   bash scripts/build_stage6_paper_package.sh /absolute/path/to/run_dir

RUN_ARG="${1:-stage5_mobile_dual_seeded}"

if [[ "${RUN_ARG}" == /* ]]; then
  RUN_DIR="${RUN_ARG}"
elif [[ "${RUN_ARG}" == results/runs/* ]]; then
  RUN_DIR="${RUN_ARG}"
else
  RUN_DIR="results/runs/${RUN_ARG}"
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

echo "Stage-6 paper package input: ${RUN_DIR}"

python -m benchmark.analyze_results "${RUN_DIR}"
python -m benchmark.build_paper_assets "${RUN_DIR}"

REQUIRED_TABLES=(
  "${RUN_DIR}/tables/method_summary.csv"
  "${RUN_DIR}/tables/method_summary.tex"
  "${RUN_DIR}/tables/robot_method_summary.csv"
  "${RUN_DIR}/tables/robot_method_summary.tex"
  "${RUN_DIR}/tables/task_family_method_summary.csv"
  "${RUN_DIR}/tables/task_family_method_summary.tex"
  "${RUN_DIR}/tables/migration_score.csv"
  "${RUN_DIR}/tables/migration_score.tex"
  "${RUN_DIR}/tables/paired_method_deltas.csv"
  "${RUN_DIR}/tables/paired_method_deltas.tex"
  "${RUN_DIR}/tables/failure_breakdown.csv"
  "${RUN_DIR}/tables/failure_breakdown.tex"
  "${RUN_DIR}/tables/generated_code_features.csv"
  "${RUN_DIR}/tables/code_changes_after_feedback.csv"
  "${RUN_DIR}/tables/analysis_report.md"
)

REQUIRED_ASSETS=(
  "${RUN_DIR}/paper_assets/experiment_manifest.json"
  "${RUN_DIR}/paper_assets/paper_results_section.md"
  "${RUN_DIR}/paper_assets/figure_index.md"
  "${RUN_DIR}/paper_assets/table_includes.tex"
  "${RUN_DIR}/paper_assets/fig_method_success.svg"
  "${RUN_DIR}/paper_assets/fig_robot_method_success.svg"
  "${RUN_DIR}/paper_assets/fig_task_family_success.svg"
  "${RUN_DIR}/paper_assets/fig_migration_score.svg"
)

for path in "${REQUIRED_TABLES[@]}" "${REQUIRED_ASSETS[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing expected Stage-6 output: ${path}" >&2
    exit 1
  fi
done

echo "Stage-6 paper package:"
echo "  ${RUN_DIR}/tables"
echo "  ${RUN_DIR}/paper_assets"
echo "Stage-6 verification: OK"
