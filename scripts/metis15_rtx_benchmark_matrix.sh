#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${METIS15_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_base}"
OUT_ROOT="${METIS15_BENCH_OUT_ROOT:-$ROOT_DIR/checkpoints/bench_metis15_rtx}"
STEPS="${METIS15_BENCH_STEPS:-300}"
WARMUP_STEPS="${METIS15_BENCH_WARMUP_STEPS:-20}"
BATCH_SIZE="${METIS15_BENCH_BATCH_SIZE:-18}"
GRAD_ACCUM="${METIS15_BENCH_GRAD_ACCUM:-11}"
LR="${METIS15_BENCH_LR:-1.2e-4}"
WEIGHT_DECAY="${METIS15_BENCH_WEIGHT_DECAY:-0.1}"
PREFETCH_BATCHES="${METIS15_PREFETCH_BATCHES:-4}"
RUN_BASELINES="${METIS15_BENCH_BASELINES:-1}"
LOW_PRECISION="${METIS15_BENCH_LOW_PRECISION:-fp8}"
SMOKE_RECIPES="${METIS15_BENCH_SMOKE_RECIPES:-}"
FP8_EXPERT_PRECISION="${METIS15_BENCH_FP8_EXPERT_PRECISION:-bf16}"
FP8_PAD_MULTIPLE="${METIS15_BENCH_FP8_PAD_MULTIPLE:-}"
HYBRID_ADAMW_IMPL="${METIS15_BENCH_HYBRID_ADAMW_IMPL:-loop}"
FUSED_ADAMW="${METIS15_BENCH_FUSED_ADAMW:-0}"
STATIC_CAPACITY_16="${METIS15_BENCH_STATIC_CAPACITY_16:-8704}"
STATIC_CAPACITY_18="${METIS15_BENCH_STATIC_CAPACITY_18:-9728}"
FORCE_BALANCED_STATIC_CAPACITY="${METIS15_BENCH_FORCE_BALANCED_STATIC_CAPACITY:-$("$PYTHON_BIN" - "$MANIFEST" "$BATCH_SIZE" <<'PY'
import json
import math
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
batch_size = int(sys.argv[2])
model = manifest["model"]
block_size = int(model["block_size"])
top_k = int(model["moe_top_k"])
num_experts = int(model["moe_num_experts"])
print(math.ceil(batch_size * block_size * top_k / num_experts))
PY
)}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOW_PRECISION_ARGS=()
case "$LOW_PRECISION" in
  none|bf16)
    ;;
  nvfp4)
    LOW_PRECISION_ARGS+=(
      --nvfp4
      --nvfp4-disable-rht
      --nvfp4-disable-2d-quantization
      --nvfp4-disable-stochastic-rounding
    )
    ;;
  fp8)
    LOW_PRECISION_ARGS+=(--fp8)
    if [[ -n "$FP8_EXPERT_PRECISION" ]]; then
      LOW_PRECISION_ARGS+=(--fp8-expert-precision "$FP8_EXPERT_PRECISION")
    fi
    if [[ -n "$FP8_PAD_MULTIPLE" ]]; then
      LOW_PRECISION_ARGS+=(--fp8-pad-multiple "$FP8_PAD_MULTIPLE")
    fi
    ;;
  *)
    echo "METIS15_BENCH_LOW_PRECISION must be one of: none, bf16, nvfp4, fp8" >&2
    exit 1
    ;;
esac
if [[ -z "$SMOKE_RECIPES" ]]; then
  case "$LOW_PRECISION" in
    fp8)
      SMOKE_RECIPES="fp8,bf16"
      ;;
    none|bf16)
      SMOKE_RECIPES="bf16"
      ;;
    nvfp4)
      SMOKE_RECIPES="fp8,fp8_block,nvfp4,bf16"
      ;;
  esac
fi

if [[ "${METIS15_SKIP_KERNEL_SMOKE:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_metis15_blackwell_kernels.py" \
    --recipes "$SMOKE_RECIPES" \
    --nvfp4-disable-rht \
    --nvfp4-disable-2d-quantization \
    --nvfp4-disable-stochastic-rounding
fi

mkdir -p "$OUT_ROOT"

run_case() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  rm -rf "$out_dir"
  echo
  echo "=== benchmark: $name ==="
  torchrun --standalone --nproc_per_node=1 "$ROOT_DIR/scripts/train_mamba_lm.py" \
    --manifest "$MANIFEST" \
    --data-dir "$DATA_DIR" \
    --out-dir "$out_dir" \
    --train-stage pretrain \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --max-steps "$STEPS" \
    --warmup-steps "$WARMUP_STEPS" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --beta1 0.9 \
    --beta2 0.95 \
    --log-interval "${METIS15_BENCH_LOG_INTERVAL:-10}" \
    --eval-interval 0 \
    --checkpoint-interval 0 \
    --skip-final-eval \
    --skip-final-checkpoint \
    --dtype bf16 \
    --matmul-precision highest \
    --optimizer muon_adamw \
    --hybrid-adamw-impl "$HYBRID_ADAMW_IMPL" \
    --lm-loss-impl liger_fused_linear_ce \
    --prefetch-batches "$PREFETCH_BATCHES" \
    "${LOW_PRECISION_ARGS[@]}" \
    "$@"
}

if [[ "$FUSED_ADAMW" == "1" ]]; then
  LOW_PRECISION_ARGS+=(--fused-adamw)
fi

if [[ "$RUN_BASELINES" == "1" ]]; then
  run_case loop_sdpa_standard_ce_no_prefetch \
    --moe-dispatch-mode loop \
    --lm-loss-impl standard \
    --prefetch-batches 0

  run_case grouped_sdpa_standard_ce \
    --moe-dispatch-mode grouped \
    --lm-loss-impl standard
fi

run_case grouped_sdpa \
  --moe-dispatch-mode grouped

run_case bucketed_sdpa \
  --moe-dispatch-mode bucketed

run_case bucketed_static_fallback \
  --moe-dispatch-mode bucketed \
  --moe-static-capacity "$STATIC_CAPACITY_18" \
  --moe-capacity-alignment 128 \
  --moe-overflow-mode fallback

run_case bucketed_static_graphable \
  --moe-dispatch-mode bucketed \
  --moe-static-capacity "$STATIC_CAPACITY_18" \
  --moe-capacity-alignment 128 \
  --moe-graphable

run_case fused_moe_force_balanced_bmm_cf1 \
  --moe-backend torch_bmm \
  --moe-dispatch-mode bucketed \
  --moe-router-override force_balanced \
  --moe-static-capacity "$FORCE_BALANCED_STATIC_CAPACITY" \
  --moe-capacity-alignment 1 \
  --moe-overflow-mode error \
  --moe-memory-efficient-permutation

run_case fused_moe_force_balanced_grouped_cf1 \
  --moe-backend torch_grouped_safe \
  --moe-dispatch-mode bucketed \
  --moe-router-override force_balanced \
  --moe-static-capacity "$FORCE_BALANCED_STATIC_CAPACITY" \
  --moe-capacity-alignment 1 \
  --moe-overflow-mode error \
  --moe-memory-efficient-permutation

if [[ "$BATCH_SIZE" == "16" ]]; then
  run_case bucketed_static_8704_graphable \
    --moe-dispatch-mode bucketed \
    --moe-static-capacity "$STATIC_CAPACITY_16" \
    --moe-capacity-alignment 128 \
    --moe-graphable
fi

if [[ "$LOW_PRECISION" == "fp8" && -z "$FP8_EXPERT_PRECISION" && -z "$FP8_PAD_MULTIPLE" ]]; then
  run_case grouped_sdpa_no_split_padding \
    --moe-dispatch-mode grouped \
    --fp8-pad-multiple 1
fi

if [[ "$LOW_PRECISION" == "fp8" && -z "$FP8_EXPERT_PRECISION" ]]; then
  run_case grouped_sdpa_bf16_experts \
    --moe-dispatch-mode grouped \
    --fp8-expert-precision bf16

  run_case bucketed_sdpa_bf16_experts \
    --moe-dispatch-mode bucketed \
    --fp8-expert-precision bf16
fi

if [[ "$LOW_PRECISION" == "nvfp4" ]]; then
  run_case grouped_sdpa_final_all_nvfp4 \
    --moe-dispatch-mode grouped \
    --nvfp4-final-expert-layers 0
fi

run_case grouped_capacity_110 \
  --moe-dispatch-mode grouped \
  --moe-capacity-factor 1.10 \
  --moe-capacity-alignment 128

if [[ "${METIS15_BENCH_ATTENTION_SWEEP:-1}" == "1" ]]; then
  run_case grouped_te_attention \
    --moe-dispatch-mode grouped \
    --te-dot-product-attention

  run_case grouped_sdpa_no_native_gqa \
    --moe-dispatch-mode grouped \
    --disable-native-gqa-attention
fi

echo
echo "Benchmark matrix complete: $OUT_ROOT"
