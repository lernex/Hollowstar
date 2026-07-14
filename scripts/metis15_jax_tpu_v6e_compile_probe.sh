#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
LOG_ROOT="${METIS15_JAX_COMPILE_LOG_ROOT:-$ROOT_DIR/tmp/metis15_jax_tpu_v6e_compile_probe}"
MAX_STEPS="${METIS15_JAX_COMPILE_MAX_STEPS:-3}"
LOCAL_BATCH_SIZE="${METIS15_JAX_COMPILE_LOCAL_BATCH_SIZE:-${METIS15_JAX_LOCAL_BATCH_SIZE:-}}"
GRAD_ACCUM_STEPS="${METIS15_JAX_COMPILE_GRAD_ACCUM_STEPS:-${METIS15_JAX_GRAD_ACCUM_STEPS:-}}"
GRAD_ACCUM_IMPL="${METIS15_JAX_COMPILE_GRAD_ACCUM_IMPL:-${METIS15_JAX_GRAD_ACCUM_IMPL:-}}"
BLOCK_SIZE="${METIS15_JAX_COMPILE_BLOCK_SIZE:-${METIS15_JAX_BLOCK_SIZE:-}}"
CAPACITY_FACTOR="${METIS15_JAX_COMPILE_CAPACITY_FACTOR:-${METIS15_JAX_EXPERT_CAPACITY_FACTOR:-}}"
OPTIMIZER="${METIS15_JAX_COMPILE_OPTIMIZER:-${METIS15_JAX_OPTIMIZER:-adamuon}}"
ADAMUON_MATRIX_POLICY="${METIS15_JAX_COMPILE_ADAMUON_MATRIX_POLICY:-${METIS15_JAX_ADAMUON_MATRIX_POLICY:-all}}"
MUON_NS_STEPS="${METIS15_JAX_COMPILE_MUON_NS_STEPS:-${METIS15_JAX_MUON_NS_STEPS:-}}"
QK_CLIP_INTERVAL="${METIS15_JAX_COMPILE_QK_CLIP_INTERVAL:-${METIS15_JAX_QK_CLIP_INTERVAL:-}}"
QK_CLIP_WARMUP_STEPS="${METIS15_JAX_COMPILE_QK_CLIP_WARMUP_STEPS:-${METIS15_JAX_QK_CLIP_WARMUP_STEPS:-}}"
REMAT_MODE="${METIS15_JAX_COMPILE_REMAT_MODE:-${METIS15_JAX_REMAT_MODE:-manifest}}"
REMAT_ATTENTION="${METIS15_JAX_COMPILE_REMAT_ATTENTION:-${METIS15_JAX_REMAT_ATTENTION:-manifest}}"
ATTENTION_BACKEND="${METIS15_JAX_COMPILE_ATTENTION_BACKEND:-${METIS15_JAX_ATTENTION_BACKEND:-manifest}}"
MOE_BACKEND="${METIS15_JAX_COMPILE_MOE_BACKEND:-${METIS15_JAX_MOE_BACKEND:-manifest}}"
DTYPE="${METIS15_JAX_COMPILE_DTYPE:-${METIS15_JAX_DTYPE:-}}"
WEIGHT_DTYPE="${METIS15_JAX_COMPILE_WEIGHT_DTYPE:-${METIS15_JAX_WEIGHT_DTYPE:-}}"
CE_LOGITS_DTYPE="${METIS15_JAX_COMPILE_CE_LOGITS_DTYPE:-${METIS15_JAX_CE_LOGITS_DTYPE:-}}"
CE_LOSS_IMPL="${METIS15_JAX_COMPILE_CE_LOSS_IMPL:-${METIS15_JAX_CE_LOSS_IMPL:-}}"
EXPERT_EXECUTION="${METIS15_JAX_COMPILE_EXPERT_EXECUTION:-${METIS15_JAX_EXPERT_EXECUTION:-data_parallel}}"
BATCH_SHARDING="${METIS15_JAX_COMPILE_BATCH_SHARDING:-${METIS15_JAX_BATCH_SHARDING:-replicated}}"
MESH_SHAPE="${METIS15_JAX_COMPILE_MESH_SHAPE:-${METIS15_JAX_MESH_SHAPE:-1x8}}"
TINY_CONFIG="${METIS15_JAX_COMPILE_TINY_CONFIG:-0}"
REQUIRE_TPU="${METIS15_JAX_COMPILE_REQUIRE_TPU:-}"
REQUIRE_COMPILE_LOG="${METIS15_JAX_COMPILE_REQUIRE_LOG:-1}"

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

export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"

echo "Metis-1.5 JAX TPU v6e compile probe"
echo "  log: $TRAIN_LOG"
echo "  max steps: $MAX_STEPS"
echo "  block size: ${BLOCK_SIZE:-manifest}"
echo "  capacity factor: ${CAPACITY_FACTOR:-manifest}"
echo "  optimizer: $OPTIMIZER"
echo "  AdaMuon matrix policy: $ADAMUON_MATRIX_POLICY"
echo "  grad accum impl: ${GRAD_ACCUM_IMPL:-manifest}"
echo "  remat mode: $REMAT_MODE"
echo "  attention remat: $REMAT_ATTENTION"
echo "  attention backend: $ATTENTION_BACKEND"
echo "  MoE backend: $MOE_BACKEND"
echo "  muon ns steps: ${MUON_NS_STEPS:-manifest}"
echo "  qk clip interval: ${QK_CLIP_INTERVAL:-manifest}"
echo "  qk clip warmup steps: ${QK_CLIP_WARMUP_STEPS:-manifest}"
echo "  dtype: ${DTYPE:-manifest}"
echo "  weight dtype: ${WEIGHT_DTYPE:-manifest}"
echo "  CE logits dtype: ${CE_LOGITS_DTYPE:-manifest}"
echo "  CE loss impl: ${CE_LOSS_IMPL:-manifest}"
echo "  expert execution: $EXPERT_EXECUTION"
echo "  batch sharding: $BATCH_SHARDING"
echo "  mesh shape: $MESH_SHAPE"
echo "  tiny config: $TINY_CONFIG"
echo "  require TPU: $REQUIRE_TPU"
echo "  JAX_LOG_COMPILES: $JAX_LOG_COMPILES"

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
  --synthetic-data
  --expert-execution "$EXPERT_EXECUTION"
  --batch-sharding "$BATCH_SHARDING"
  --mesh-shape "$MESH_SHAPE"
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
if [[ -n "$GRAD_ACCUM_IMPL" ]]; then
  cmd+=(--grad-accum-impl "$GRAD_ACCUM_IMPL")
fi
if [[ -n "$BLOCK_SIZE" ]]; then
  cmd+=(--block-size "$BLOCK_SIZE")
fi
if [[ -n "$CAPACITY_FACTOR" ]]; then
  cmd+=(--expert-capacity-factor "$CAPACITY_FACTOR")
fi
if [[ -n "$MUON_NS_STEPS" ]]; then
  cmd+=(--muon-ns-steps "$MUON_NS_STEPS")
fi
if [[ -n "$QK_CLIP_INTERVAL" ]]; then
  cmd+=(--qk-clip-interval "$QK_CLIP_INTERVAL")
fi
if [[ -n "$QK_CLIP_WARMUP_STEPS" ]]; then
  cmd+=(--qk-clip-warmup-steps "$QK_CLIP_WARMUP_STEPS")
fi
if [[ "$REMAT_MODE" == "on" ]]; then
  cmd+=(--remat-layers)
elif [[ "$REMAT_MODE" == "off" ]]; then
  cmd+=(--no-remat-layers)
elif [[ "$REMAT_MODE" != "manifest" ]]; then
  echo "Unsupported remat mode: $REMAT_MODE (use manifest, on, off)" >&2
  exit 1
fi
if [[ "$REMAT_ATTENTION" == "on" || "$REMAT_ATTENTION" == "1" || "$REMAT_ATTENTION" == "true" ]]; then
  cmd+=(--remat-attention)
elif [[ "$REMAT_ATTENTION" == "off" || "$REMAT_ATTENTION" == "0" || "$REMAT_ATTENTION" == "false" ]]; then
  cmd+=(--no-remat-attention)
elif [[ "$REMAT_ATTENTION" != "manifest" ]]; then
  echo "Unsupported attention remat mode: $REMAT_ATTENTION (use manifest, on/off, 1/0, true/false)" >&2
  exit 1
fi
if [[ "$ATTENTION_BACKEND" != "manifest" ]]; then
  cmd+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "$MOE_BACKEND" != "manifest" ]]; then
  cmd+=(--moe-backend "$MOE_BACKEND")
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
if [[ -n "$CE_LOSS_IMPL" ]]; then
  cmd+=(--ce-loss-impl "$CE_LOSS_IMPL")
fi

mkdir -p "$OUT_DIR"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"

compile_markers="$(
  grep -E "Compiling|Finished XLA compilation|Finished tracing|Finished jaxpr to MLIR" "$TRAIN_LOG" | wc -l | tr -d ' '
)"
echo "compile_log_markers=$compile_markers"
if [[ "$REQUIRE_COMPILE_LOG" == "1" && "$compile_markers" == "0" ]]; then
  echo "Expected JAX compile markers in $TRAIN_LOG but found none." >&2
  exit 1
fi

audit=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/analyze_metis15_jax_tpu_logs.py"
  "$TRAIN_LOG"
  --min-logged-steps 2
  --min-valid-assign-frac "${METIS15_JAX_COMPILE_MIN_VALID_ASSIGN_FRAC:-0.95}"
  --max-expert-drop-frac "${METIS15_JAX_COMPILE_MAX_EXPERT_DROP_FRAC:-0.10}"
  --max-qk-logit "${METIS15_JAX_COMPILE_MAX_QK_LOGIT:-1000}"
  --require-mor-disabled
  --perf-warmup-steps "${METIS15_JAX_PERF_WARMUP_STEPS:-1}"
)
if [[ "$REQUIRE_TPU" == "1" ]]; then
  audit+=(--require-tpu)
fi
"${audit[@]}"

echo "Metis-1.5 JAX TPU compile probe passed."
