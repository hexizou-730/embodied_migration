#!/usr/bin/env bash
set -euo pipefail

# Stage-4 strict ablation runner for the Fixed Dual-arm + Mobile Dual-arm line.
#
# Usage:
#   bash scripts/run_stage4_mobile_dual_ablation.sh
#   bash scripts/run_stage4_mobile_dual_ablation.sh my_run_id
#
# Useful overrides:
#   TRIALS=3 TASKS=mobility bash scripts/run_stage4_mobile_dual_ablation.sh
#   MODES="fewshot card card_failure" bash scripts/run_stage4_mobile_dual_ablation.sh

RUN_ID="${1:-stage4_mobile_dual_ablation}"
TRIALS="${TRIALS:-1}"
SEED_BASE="${SEED_BASE:-0}"
TASKS="${TASKS:-migration}"
SCENE_VARIANT="${SCENE_VARIANT:-fixed}"
ROBOTS="${ROBOTS:-dual_arm mobile_dual_arm dual_franka}"
MODES="${MODES:-api fewshot card failure card_failure}"
MODEL="${MODEL:-${EM_MODEL:-anthropic/claude-sonnet-4.5}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CACHE_DIR="${CACHE_DIR:-results/llm_cache}"
NO_CACHE="${NO_CACHE:-0}"
OFFLINE_CACHE_ONLY="${OFFLINE_CACHE_ONLY:-0}"

EXTRA_ARGS=()
if [[ "${NO_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--no-cache)
fi
if [[ "${OFFLINE_CACHE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--offline-cache-only)
fi

echo "Stage-4 Fixed-dual/Mobile-dual strict ablation"
echo "Run id: ${RUN_ID}"
echo "Robots: ${ROBOTS}"
echo "Modes: ${MODES}"
echo "Tasks: ${TASKS}"
echo "Trials: ${TRIALS}"
echo "Scene variant: ${SCENE_VARIANT}"
echo "Model: ${MODEL}"
echo "Temperature: ${TEMPERATURE}"
echo "Cache dir: ${CACHE_DIR}"

python -m benchmark.run_benchmark \
  --robots ${ROBOTS} \
  --modes ${MODES} \
  --tasks "${TASKS}" \
  --trials "${TRIALS}" \
  --scene-variant "${SCENE_VARIANT}" \
  --seed-base "${SEED_BASE}" \
  --model "${MODEL}" \
  --temperature "${TEMPERATURE}" \
  --cache-dir "${CACHE_DIR}" \
  --run-id "${RUN_ID}" \
  "${EXTRA_ARGS[@]}"

RUN_DIR="results/runs/${RUN_ID}"
python -m benchmark.analyze_results "${RUN_DIR}"
python -m benchmark.audit_run "${RUN_DIR}" --fail-on-missing
python -m benchmark.build_paper_assets "${RUN_DIR}"

echo "Stage-4 complete:"
echo "  ${RUN_DIR}/summary.csv"
echo "  ${RUN_DIR}/tables"
echo "  ${RUN_DIR}/audit/audit_report.md"
echo "  ${RUN_DIR}/paper_assets"
