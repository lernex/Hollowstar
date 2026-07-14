#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${METIS15_REPO_DIR:-/opt/dlami/nvme/metis}"
IMAGE="${METIS15_H100_IMAGE:-lernex/metis-gpu:metis15-h100-single-latent-v1}"
DATA_DIR="${METIS15_DATA_DIR:-data/metis15_base}"
OUT_DIR="${METIS15_OUT_DIR:-checkpoints/h100_benchmark}"
BATCH_SIZE="${METIS15_BATCH_SIZE:-4}"
GRAD_ACCUM="${METIS15_GRAD_ACCUM:-1}"
MAX_STEPS="${METIS15_MAX_STEPS:-20}"
LR="${METIS15_LR:-1e-5}"
WARMUP_STEPS="${METIS15_WARMUP_STEPS:-1}"
PREFETCH_BATCHES="${METIS15_PREFETCH_BATCHES:-1}"
OPTIMIZER="${METIS15_OPTIMIZER:-adamw}"
LM_LOSS_IMPL="${METIS15_LM_LOSS_IMPL:-standard}"
LOG_INTERVAL="${METIS15_LOG_INTERVAL:-5}"
MOE_BACKEND="${METIS15_MOE_BACKEND:-torch_grouped_safe}"
MOE_DISPATCH_MODE="${METIS15_MOE_DISPATCH_MODE:-bucketed}"
MOE_TORCH_GROUPED_MIN_M="${METIS15_MOE_TORCH_GROUPED_MIN_M:-8}"
FUSED_ADAMW="${METIS15_FUSED_ADAMW:-1}"
DEBUG_NONFINITE="${METIS15_DEBUG_NONFINITE:-0}"
DEBUG_FORWARD_NONFINITE_HOOKS="${METIS15_DEBUG_FORWARD_NONFINITE_HOOKS:-0}"
DEBUG_ABSMAX_TOP_K="${METIS15_DEBUG_ABSMAX_TOP_K:-0}"
RETAIN_STANDARD_CE_LOGITS="${METIS15_RETAIN_STANDARD_CE_LOGITS:-1}"
DISABLE_MOE_AUX="${METIS15_DISABLE_MOE_AUX:-0}"
DISABLE_MOE_BALANCE_UPDATE="${METIS15_DISABLE_MOE_BALANCE_UPDATE:-0}"
DISABLE_MOE_FUSED_COMBINE="${METIS15_DISABLE_MOE_FUSED_COMBINE:-0}"
MOE_STATIC_CAPACITY="${METIS15_MOE_STATIC_CAPACITY:-}"
MOE_CAPACITY_FACTOR="${METIS15_MOE_CAPACITY_FACTOR:-}"
MOE_CAPACITY_ALIGNMENT="${METIS15_MOE_CAPACITY_ALIGNMENT:-}"
MOE_OVERFLOW_MODE="${METIS15_MOE_OVERFLOW_MODE:-}"
MOE_ROUTER_OVERRIDE="${METIS15_MOE_ROUTER_OVERRIDE:-}"

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

EXTRA_ARGS=()
if truthy "$FUSED_ADAMW"; then
  EXTRA_ARGS+=(--fused-adamw)
fi
if truthy "$DEBUG_NONFINITE"; then
  EXTRA_ARGS+=(--debug-nonfinite)
fi
if truthy "$DEBUG_FORWARD_NONFINITE_HOOKS"; then
  EXTRA_ARGS+=(--debug-forward-nonfinite-hooks)
fi
if [[ "$DEBUG_ABSMAX_TOP_K" != "0" ]]; then
  EXTRA_ARGS+=(--debug-absmax-top-k "$DEBUG_ABSMAX_TOP_K")
fi
if truthy "$RETAIN_STANDARD_CE_LOGITS"; then
  EXTRA_ARGS+=(--retain-standard-ce-logits)
fi
if truthy "$DISABLE_MOE_AUX"; then
  EXTRA_ARGS+=(--disable-moe-aux)
fi
if truthy "$DISABLE_MOE_BALANCE_UPDATE"; then
  EXTRA_ARGS+=(--disable-moe-balance-update)
fi
if truthy "$DISABLE_MOE_FUSED_COMBINE"; then
  EXTRA_ARGS+=(--disable-moe-fused-combine)
fi
if [[ -n "$MOE_STATIC_CAPACITY" ]]; then
  EXTRA_ARGS+=(--moe-static-capacity "$MOE_STATIC_CAPACITY")
fi
if [[ -n "$MOE_CAPACITY_FACTOR" ]]; then
  EXTRA_ARGS+=(--moe-capacity-factor "$MOE_CAPACITY_FACTOR")
fi
if [[ -n "$MOE_CAPACITY_ALIGNMENT" ]]; then
  EXTRA_ARGS+=(--moe-capacity-alignment "$MOE_CAPACITY_ALIGNMENT")
fi
if [[ -n "$MOE_OVERFLOW_MODE" ]]; then
  EXTRA_ARGS+=(--moe-overflow-mode "$MOE_OVERFLOW_MODE")
fi
if [[ -n "$MOE_ROUTER_OVERRIDE" ]]; then
  EXTRA_ARGS+=(--moe-router-override "$MOE_ROUTER_OVERRIDE")
fi

docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO_DIR:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e METIS_SWIGLU_IMPL="${METIS_SWIGLU_IMPL:-torch}" \
  -e METIS_DISABLE_TRITON_SWIGLU="${METIS_DISABLE_TRITON_SWIGLU:-1}" \
  -e METIS_TRITON_SWIGLU_SURFACES="${METIS_TRITON_SWIGLU_SURFACES:-grouped_experts}" \
  -e METIS_TRITON_SWIGLU_BACKWARD="${METIS_TRITON_SWIGLU_BACKWARD:-torch}" \
  -e METIS_SWIGLU_COMPILE_MODE="${METIS_SWIGLU_COMPILE_MODE:-reduce-overhead}" \
  -e METIS_DEBUG_SWIGLU_FINITE="${METIS_DEBUG_SWIGLU_FINITE:-0}" \
  -e METIS_DISABLE_PERF_COUNTERS="${METIS_DISABLE_PERF_COUNTERS:-0}" \
  -e METIS_ROTARY_CACHE_CLONE="${METIS_ROTARY_CACHE_CLONE:-0}" \
  -e METIS_DISABLE_ROTARY_CACHE="${METIS_DISABLE_ROTARY_CACHE:-0}" \
  -e METIS_DYNAMO_DISABLE_GROUPED_EXPERTS="${METIS_DYNAMO_DISABLE_GROUPED_EXPERTS:-0}" \
  -e METIS_TORCH_GROUPED_SAFE_SYNC="${METIS_TORCH_GROUPED_SAFE_SYNC:-1}" \
  -e METIS_PREINIT_ADAMW_STATE="${METIS_PREINIT_ADAMW_STATE:-0}" \
  -e METIS_ASYNC_METRICS="${METIS_ASYNC_METRICS:-1}" \
  "$IMAGE" \
  python scripts/train_mamba_lm.py \
    --data-dir "$DATA_DIR" \
    --manifest configs/metis15_manifest.json \
    --train-stage pretrain \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --max-steps "$MAX_STEPS" \
    --lr "$LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --weight-decay 0.1 \
    --eval-interval 1000000 \
    --eval-iters 2 \
    --log-interval "$LOG_INTERVAL" \
    --checkpoint-interval 1000000 \
    --skip-final-eval \
    --skip-final-checkpoint \
    --dtype bf16 \
    --fp8-expert-precision bf16 \
    --moe-backend "$MOE_BACKEND" \
    --moe-dispatch-mode "$MOE_DISPATCH_MODE" \
    --moe-torch-grouped-min-m "$MOE_TORCH_GROUPED_MIN_M" \
    --lm-loss-impl "$LM_LOSS_IMPL" \
    --optimizer "$OPTIMIZER" \
    --prefetch-batches "$PREFETCH_BATCHES" \
    --matmul-precision high \
    --tf32 \
    "${EXTRA_ARGS[@]}"
