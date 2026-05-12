#!/usr/bin/env bash
set -euo pipefail

# One-command Stage-5 seeded experiment runner for the Fixed Dual-arm + Mobile Dual-arm line.
#
# Usage:
#   bash scripts/run_stage5_experiments.sh
#   bash scripts/run_stage5_experiments.sh my_run_id
#
# Optional environment overrides:
#   TRIALS=5 SEED_BASE=100 TASKS=migration bash scripts/run_stage5_experiments.sh stage5_mobile_dual_seeded

RUN_ID="${1:-stage5_mobile_dual_seeded}"
TRIALS="${TRIALS:-5}"
SEED_BASE="${SEED_BASE:-0}"
TASKS="${TASKS:-migration}"
SCENE_VARIANT="${SCENE_VARIANT:-seeded}"
ROBOTS="${ROBOTS:-dual_arm mobile_dual_arm dual_franka}"
MODES="${MODES:-api fewshot card failure card_failure}"
MODEL="${MODEL:-${EM_MODEL:-anthropic/claude-sonnet-4.5}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CACHE_DIR="${CACHE_DIR:-results/llm_cache}"
NO_CACHE="${NO_CACHE:-0}"
OFFLINE_CACHE_ONLY="${OFFLINE_CACHE_ONLY:-0}"
VALIDATE_SEEDS="${VALIDATE_SEEDS:-1}"

EXTRA_ARGS=()
if [[ "${NO_CACHE}" == "1" ]]; then
  EXTRA_ARGS+=(--no-cache)
fi
if [[ "${OFFLINE_CACHE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--offline-cache-only)
fi

echo "Stage-5 run id: ${RUN_ID}"
echo "Robots: ${ROBOTS}"
echo "Modes: ${MODES}"
echo "Tasks: ${TASKS}"
echo "Trials/seeds: ${TRIALS}"
echo "Scene variant: ${SCENE_VARIANT}"
echo "Seed base: ${SEED_BASE}"
echo "Model: ${MODEL}"
echo "Temperature: ${TEMPERATURE}"
echo "Cache dir: ${CACHE_DIR}"

if [[ "${VALIDATE_SEEDS}" == "1" ]]; then
  case "${TASKS}" in
    migration|mobility|bimanual|all)
      python -m benchmark.validate_seeded_scenes \
        --robots ${ROBOTS} \
        --tasks "${TASKS}" \
        --trials "${TRIALS}" \
        --scene-variant "${SCENE_VARIANT}" \
        --seed-base "${SEED_BASE}"
      ;;
    *)
      echo "Skipping seeded-scene oracle validation for TASKS=${TASKS}"
      ;;
  esac
fi

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

echo "Stage-5 seeded experiment is ready:"
echo "  ${RUN_DIR}/summary.csv"
echo "  ${RUN_DIR}/tables"
echo "  ${RUN_DIR}/audit/audit_report.md"
