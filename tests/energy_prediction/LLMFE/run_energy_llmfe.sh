#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LLM-FE energy forecasting debug launcher
# =============================================================================
# Run from the LLMFE repository root:
#   bash run_energy_llmfe_debug.sh
#
# This follows the original repo style: set API/env variables in bash, then call
# the Python entrypoint. The original LLM-FE repo expects API configuration in
# its launcher script / environment before running the main program. :contentReference[oaicite:0]{index=0}
# =============================================================================

# ---- API configuration -------------------------------------------------------
# Preferred: export this before running the script:
#   export OPENAI_API_KEY="..."
#
# Or uncomment this line locally. Do not commit real keys.
# TODO: never commit this (just as safeguard)
export OPENAI_API_KEY=""
export API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"


if [[ -z "${OPENAI_API_KEY:-}" && -z "${API_KEY:-}" ]]; then
  echo "[Error] Neither OPENAI_API_KEY nor API_KEY is set."
  echo "Run: export OPENAI_API_KEY='your_key_here'"
  exit 1
fi

# ---- Paths ------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# "/mnt/e/repos/LLM_traffic_query/tests/energy_prediction/LLMFE/data_energy/ashrae_ts_sample_ratio_0p02_seed_155"
# "/mnt/e/repos/LLM_traffic_query/tests/energy_prediction/dataset/ashrae-energy-prediction/processed_features/ashrae_train_cleaned_plus_manual_features"
DATA_PARQUET="/mnt/e/repos/LLM_traffic_query/tests/energy_prediction/dataset/ashrae-energy-prediction/processed_features/ashrae_train_cleaned_plus_manual_features"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_ROOT}/results_energy_llmfe/debug_${STAMP}"

mkdir -p "${OUT_DIR}"

# ---- Debug-scale LLM-FE controls -------------------------------------------
# Use a small run first. Increase these later for the real 12h ablation.
WALL_CLOCK_HOURS=2.0
MAX_CANDIDATES=10
EVALUATE_TIMEOUT_SECONDS=1200

# ---- Model / run controls ---------------------------------------------------
API_MODEL=gpt-3.5-turbo
SEED=155
NUM_THREADS=0

echo "[Run] LLM-FE energy forecasting debug"
echo "[Run] repo root                 : ${REPO_ROOT}"
echo "[Run] data parquet              : ${DATA_PARQUET}"
echo "[Run] out dir                   : ${OUT_DIR}"
echo "[Run] wall clock hours          : ${WALL_CLOCK_HOURS}"
echo "[Run] max candidates            : ${MAX_CANDIDATES}"
echo "[Run] eval timeout seconds      : ${EVALUATE_TIMEOUT_SECONDS}"
echo "[Run] API model                 : ${API_MODEL}"
echo "[Run] seed                      : ${SEED}"
echo

export PYTHONUNBUFFERED=1 # unbuffer python to se live logs
python -u main_energy_forecasting.py \
  --data_parquet "${DATA_PARQUET}" \
  --out_dir "${OUT_DIR}" \
  --wall_clock_hours "${WALL_CLOCK_HOURS}" \
  --max_candidates "${MAX_CANDIDATES}" \
  --evaluate_timeout_seconds "${EVALUATE_TIMEOUT_SECONDS}" \
  --use_api \
  --api_model "${API_MODEL}" \
  --seed "${SEED}" \
  --num_threads "${NUM_THREADS}" \
  --learning_rate 0.05 \
  --num_leaves 63 \
  --min_data_in_leaf 500 \
  --feature_fraction 0.8 \
  --bagging_fraction 0.8 \
  --bagging_freq 1 \
  --lambda_l2 1.0 \
  --max_bin 255 \
  --n_estimators 12000 \
  --early_stopping_rounds 200 \
  --early_stopping_min_delta 10e-4 \
  --train_end_day 280 \
  --inner_eval_end_day 320 \
  --final_test_end_day 366 \
  --early_stopping_tail_days 21 \
  2>&1 | tee "${OUT_DIR}/run.log"

echo
echo "[Done] Output written to: ${OUT_DIR}"
echo "[Check] Candidate audit:"
echo "        ${OUT_DIR}/candidate_audit.jsonl"
echo "[Check] Best candidate:"
echo "        ${OUT_DIR}/best_candidate.py"
echo "        ${OUT_DIR}/best_candidate.json"