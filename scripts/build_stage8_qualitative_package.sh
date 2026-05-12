#!/usr/bin/env bash
set -euo pipefail

# Build Stage-8 qualitative package for a completed run.
#
# Usage:
#   bash scripts/build_stage8_qualitative_package.sh stage7_mobile_dual_seeded
#   bash scripts/build_stage8_qualitative_package.sh results/runs/stage7_mobile_dual_seeded

RUN_ARG="${1:-stage7_mobile_dual_seeded}"

if [[ "${RUN_ARG}" == results/runs/* ]]; then
  RUN_DIR="${RUN_ARG}"
else
  RUN_DIR="results/runs/${RUN_ARG}"
fi

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

python -m benchmark.analyze_results "${RUN_DIR}"
python -m benchmark.audit_run "${RUN_DIR}" --fail-on-missing
python -m benchmark.build_paper_assets "${RUN_DIR}"
python -m benchmark.build_casebook "${RUN_DIR}"

echo "Stage-8 qualitative package:"
echo "  ${RUN_DIR}/casebook/qualitative_casebook.md"
echo "  ${RUN_DIR}/casebook/qualitative_cases.csv"
echo "  ${RUN_DIR}/casebook/qualitative_casebook.tex"
