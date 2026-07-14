#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${METIS15_TPU_CANARY_LOG_ROOT:-$ROOT_DIR/tmp/metis15_tpu_v6e_quality_canary}"
MAX_STEPS="${METIS15_TPU_CANARY_MAX_STEPS:-32}"
LOG_INTERVAL="${METIS15_TPU_CANARY_LOG_INTERVAL:-1}"
MIN_LOSS_DROP_FRAC="${METIS15_TPU_CANARY_MIN_LOSS_DROP_FRAC:-0.03}"
MAX_FINAL_LOSS="${METIS15_TPU_CANARY_MAX_FINAL_LOSS:-}"

mkdir -p "$LOG_ROOT"
CANARY_OUT="$LOG_ROOT/checkpoints"
CANARY_LOG="$LOG_ROOT/train.log"

echo "Metis-1.5 TPU v6e fixed-batch quality canary"
echo "  log: $CANARY_LOG"
echo "  max steps: $MAX_STEPS"
echo "  min loss drop frac: $MIN_LOSS_DROP_FRAC"
if [[ -n "$MAX_FINAL_LOSS" ]]; then
  echo "  max final loss: $MAX_FINAL_LOSS"
fi

(
  export METIS15_TPU_MAX_STEPS="$MAX_STEPS"
  export METIS15_TPU_LOG_INTERVAL="$LOG_INTERVAL"
  export METIS15_TPU_SKIP_CHECKPOINT="${METIS15_TPU_CANARY_SKIP_CHECKPOINT:-1}"
  export METIS15_TPU_AUTO_RESUME=0
  export METIS15_TPU_PROFILE_COMPONENTS=1
  export METIS15_TPU_LOG_EXPERT_HISTOGRAMS=1
  export METIS15_TPU_FIXED_BATCH=1
  export METIS15_TPU_LOCAL_BATCH_SIZE="${METIS15_TPU_CANARY_LOCAL_BATCH_SIZE:-1}"
  export METIS15_TPU_GRAD_ACCUM_STEPS="${METIS15_TPU_CANARY_GRAD_ACCUM_STEPS:-1}"
  export METIS15_TPU_SYNTHETIC="${METIS15_TPU_CANARY_SYNTHETIC:-${METIS15_TPU_SYNTHETIC:-0}}"
  export METIS15_OUT_DIR="$CANARY_OUT"
  export METIS15_TPU_TRAIN_LOG="$CANARY_OUT/train.log"
  "$ROOT_DIR/scripts/metis15_tpu_v6e_pretrain.sh"
) 2>&1 | tee "$CANARY_LOG"

audit_cmd=(
  "$ROOT_DIR/scripts/analyze_metis15_tpu_logs.py"
  "$CANARY_LOG"
  --min-logged-steps 4
  --require-profile
  --require-qk-clip
  --require-expert-hist
  --require-loss-decrease
  --min-loss-drop-frac "$MIN_LOSS_DROP_FRAC"
  --min-valid-assign-frac 0.99
  --max-expert-drop-frac "${METIS15_TPU_CANARY_MAX_EXPERT_DROP_FRAC:-0.01}"
  --max-qk-logit "${METIS15_TPU_CANARY_MAX_QK_LOGIT:-1000}"
  --perf-warmup-steps "${METIS15_TPU_PERF_WARMUP_STEPS:-3}"
)

if [[ -n "$MAX_FINAL_LOSS" ]]; then
  audit_cmd+=(--max-final-loss "$MAX_FINAL_LOSS")
fi

"${audit_cmd[@]}"
echo "Metis-1.5 TPU quality canary passed."
