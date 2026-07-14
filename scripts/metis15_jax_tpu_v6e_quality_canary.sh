#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
LOG_ROOT="${METIS15_JAX_CANARY_LOG_ROOT:-$ROOT_DIR/tmp/metis15_jax_tpu_v6e_quality_canary}"
MAX_STEPS="${METIS15_JAX_CANARY_MAX_STEPS:-16}"
MIN_LOSS_DROP_FRAC="${METIS15_JAX_CANARY_MIN_LOSS_DROP_FRAC:-0.02}"
LOCAL_BATCH_SIZE="${METIS15_JAX_CANARY_LOCAL_BATCH_SIZE:-${METIS15_JAX_LOCAL_BATCH_SIZE:-}}"
GRAD_ACCUM_STEPS="${METIS15_JAX_CANARY_GRAD_ACCUM_STEPS:-${METIS15_JAX_GRAD_ACCUM_STEPS:-}}"
BLOCK_SIZE="${METIS15_JAX_CANARY_BLOCK_SIZE:-${METIS15_JAX_BLOCK_SIZE:-}}"
CAPACITY_FACTOR="${METIS15_JAX_CANARY_CAPACITY_FACTOR:-${METIS15_JAX_EXPERT_CAPACITY_FACTOR:-}}"
OPTIMIZER="${METIS15_JAX_CANARY_OPTIMIZER:-${METIS15_JAX_OPTIMIZER:-adamuon}}"
ADAMUON_MATRIX_POLICY="${METIS15_JAX_CANARY_ADAMUON_MATRIX_POLICY:-${METIS15_JAX_ADAMUON_MATRIX_POLICY:-all}}"
REMAT_MODE="${METIS15_JAX_CANARY_REMAT_MODE:-${METIS15_JAX_REMAT_MODE:-manifest}}"
DTYPE="${METIS15_JAX_CANARY_DTYPE:-${METIS15_JAX_DTYPE:-}}"
WEIGHT_DTYPE="${METIS15_JAX_CANARY_WEIGHT_DTYPE:-${METIS15_JAX_WEIGHT_DTYPE:-}}"
CE_LOGITS_DTYPE="${METIS15_JAX_CANARY_CE_LOGITS_DTYPE:-${METIS15_JAX_CE_LOGITS_DTYPE:-}}"
EXPERT_EXECUTION="${METIS15_JAX_CANARY_EXPERT_EXECUTION:-${METIS15_JAX_EXPERT_EXECUTION:-data_parallel}}"
TINY_CONFIG="${METIS15_JAX_CANARY_TINY_CONFIG:-0}"
SYNTHETIC="${METIS15_JAX_CANARY_SYNTHETIC:-1}"
REQUIRE_TPU="${METIS15_JAX_CANARY_REQUIRE_TPU:-}"
DATA_DIR="${METIS15_DATA_DIR:-}"
MAX_FINAL_LOSS="${METIS15_JAX_CANARY_MAX_FINAL_LOSS:-}"

if [[ -z "$REQUIRE_TPU" ]]; then
  if [[ "$TINY_CONFIG" == "1" ]]; then
    REQUIRE_TPU=0
  else
    REQUIRE_TPU=1
  fi
fi

mkdir -p "$LOG_ROOT"
OUT_DIR="$LOG_ROOT/checkpoints"
TRAIN_LOG="$LOG_ROOT/train.log"

echo "Metis-1.5 JAX TPU v6e quality canary"
echo "  log: $TRAIN_LOG"
echo "  max steps: $MAX_STEPS"
echo "  block size: ${BLOCK_SIZE:-manifest}"
echo "  capacity factor: ${CAPACITY_FACTOR:-manifest}"
echo "  optimizer: $OPTIMIZER"
echo "  AdaMuon matrix policy: $ADAMUON_MATRIX_POLICY"
echo "  remat mode: $REMAT_MODE"
echo "  dtype: ${DTYPE:-manifest}"
echo "  weight dtype: ${WEIGHT_DTYPE:-manifest}"
echo "  CE logits dtype: ${CE_LOGITS_DTYPE:-manifest}"
echo "  expert execution: $EXPERT_EXECUTION"
echo "  tiny config: $TINY_CONFIG"
echo "  synthetic: $SYNTHETIC"
echo "  require TPU: $REQUIRE_TPU"
echo "  min loss drop frac: $MIN_LOSS_DROP_FRAC"

preflight=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_preflight.py"
  --manifest "$MANIFEST"
  --stage pretrain
)
if [[ -n "$LOCAL_BATCH_SIZE" ]]; then
  preflight+=(--local-batch-size "$LOCAL_BATCH_SIZE")
fi
if [[ "$REQUIRE_TPU" != "1" ]]; then
  preflight+=(--skip-device-check)
fi
"${preflight[@]}"

cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/train_metis15_jax_tpu.py"
  --manifest "$MANIFEST"
  --stage pretrain
  --max-steps "$MAX_STEPS"
  --out-dir "$OUT_DIR"
  --checkpoint-interval 0
  --skip-checkpoint
  --expert-execution "$EXPERT_EXECUTION"
  --optimizer "$OPTIMIZER"
  --adamuon-matrix-policy "$ADAMUON_MATRIX_POLICY"
)
if [[ "$TINY_CONFIG" == "1" ]]; then
  cmd+=(--tiny-config)
fi
if [[ -n "$LOCAL_BATCH_SIZE" ]]; then
  cmd+=(--local-batch-size "$LOCAL_BATCH_SIZE")
fi
if [[ -n "$GRAD_ACCUM_STEPS" ]]; then
  cmd+=(--grad-accum-steps "$GRAD_ACCUM_STEPS")
fi
if [[ -n "$BLOCK_SIZE" ]]; then
  cmd+=(--block-size "$BLOCK_SIZE")
fi
if [[ -n "$CAPACITY_FACTOR" ]]; then
  cmd+=(--expert-capacity-factor "$CAPACITY_FACTOR")
fi
if [[ "$REMAT_MODE" == "on" ]]; then
  cmd+=(--remat-layers)
elif [[ "$REMAT_MODE" == "off" ]]; then
  cmd+=(--no-remat-layers)
elif [[ "$REMAT_MODE" != "manifest" ]]; then
  echo "Unsupported remat mode: $REMAT_MODE (use manifest, on, off)" >&2
  exit 1
fi
if [[ -n "$DTYPE" ]]; then
  cmd+=(--dtype "$DTYPE")
fi
if [[ -n "$WEIGHT_DTYPE" ]]; then
  cmd+=(--weight-dtype "$WEIGHT_DTYPE")
fi
if [[ -n "$CE_LOGITS_DTYPE" ]]; then
  cmd+=(--ce-logits-dtype "$CE_LOGITS_DTYPE")
fi
if [[ "$SYNTHETIC" == "1" ]]; then
  cmd+=(--synthetic-data)
elif [[ -n "$DATA_DIR" ]]; then
  cmd+=(--data-dir "$DATA_DIR")
else
  echo "Set METIS15_DATA_DIR or METIS15_JAX_CANARY_SYNTHETIC=1." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"

audit=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/analyze_metis15_jax_tpu_logs.py"
  "$TRAIN_LOG"
  --min-logged-steps 4
  --require-loss-decrease
  --min-loss-drop-frac "$MIN_LOSS_DROP_FRAC"
  --min-valid-assign-frac "${METIS15_JAX_CANARY_MIN_VALID_ASSIGN_FRAC:-0.95}"
  --max-expert-drop-frac "${METIS15_JAX_CANARY_MAX_EXPERT_DROP_FRAC:-0.10}"
  --max-qk-logit "${METIS15_JAX_CANARY_MAX_QK_LOGIT:-1000}"
  --require-mor-disabled
  --perf-warmup-steps "${METIS15_JAX_PERF_WARMUP_STEPS:-3}"
)
if [[ "$REQUIRE_TPU" == "1" ]]; then
  audit+=(--require-tpu)
fi
if [[ -n "$MAX_FINAL_LOSS" ]]; then
  audit+=(--max-final-loss "$MAX_FINAL_LOSS")
fi
"${audit[@]}"
echo "Metis-1.5 JAX TPU quality canary passed."
