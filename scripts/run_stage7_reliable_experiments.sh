#!/usr/bin/env bash
set -euo pipefail

# Reliable Fixed Dual-arm + Mobile Dual-arm experiment pipeline with cache, resume, audit, and paper assets.
#
# Usage:
#   bash scripts/run_stage7_reliable_experiments.sh
#   bash scripts/run_stage7_reliable_experiments.sh stage7_mobile_dual_seeded
#
# Useful overrides:
#   TRIALS=5 MODEL=anthropic/claude-sonnet-4.5 bash scripts/run_stage7_reliable_experiments.sh
#   OFFLINE_CACHE_ONLY=1 bash scripts/run_stage7_reliable_experiments.sh stage7_mobile_dual_seeded

RUN_ID="${1:-stage7_mobile_dual_seeded}"
TRIALS="${TRIALS:-5}"
SEED_BASE="${SEED_BASE:-0}"
TASKS="${TASKS:-migration}"
SCENE_VARIANT="${SCENE_VARIANT:-seeded}"
ROBOTS="${ROBOTS:-dual_arm mobile_dual_arm dual_franka}"
MODES="${MODES:-api fewshot card failure card_failure}"
MODEL="${MODEL:-${EM_MODEL:-anthropic/claude-sonnet-4.5}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CACHE_DIR="${CACHE_DIR:-results/llm_cache}"
OFFLINE_CACHE_ONLY="${OFFLINE_CACHE_ONLY:-0}"

EXTRA_ARGS=()
if [[ "${OFFLINE_CACHE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--offline-cache-only)
fi

echo "Stage-7 reliable run id: ${RUN_ID}"
echo "Model: ${MODEL}"
echo "Temperature: ${TEMPERATURE}"
echo "Cache dir: ${CACHE_DIR}"
echo "Resume: enabled"

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
  --resume \
  --run-id "${RUN_ID}" \
  "${EXTRA_ARGS[@]}"

RUN_DIR="results/runs/${RUN_ID}"
python -m benchmark.analyze_results "${RUN_DIR}"
python -m benchmark.audit_run "${RUN_DIR}" --fail-on-missing
python -m benchmark.build_paper_assets "${RUN_DIR}"

echo "Stage-7 complete:"
echo "  ${RUN_DIR}/summary.csv"
echo "  ${RUN_DIR}/audit/audit_report.md"
echo "  ${RUN_DIR}/tables"
echo "  ${RUN_DIR}/paper_assets"
