#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_base}"
OUT_DIR="${METIS15_OUT_DIR:-$ROOT_DIR/checkpoints/metis15_base}"
TRAIN_STAGE="${METIS15_TRAIN_STAGE:-pretrain}"
INIT_CHECKPOINT="${METIS15_INIT_CHECKPOINT:-}"
INIT_CHECKPOINT_S3_URI="${METIS15_INIT_CHECKPOINT_S3_URI:-}"
RESUME_TRAINING="${METIS15_RESUME:-1}"
ENABLE_FP8="${METIS15_FP8:-0}"
NVFP4_EXPLICIT=0
if [[ -n "${METIS15_NVFP4+x}" ]]; then
  NVFP4_EXPLICIT=1
fi
ENABLE_NVFP4="${METIS15_NVFP4:-0}"
TF32_FLAG="${METIS15_TF32:-0}"
MATMUL_PRECISION="${METIS15_MATMUL_PRECISION:-highest}"
TRAINING_MODE="${METIS15_TRAINING_MODE:-}"
FP8_DPA_FLAG="${METIS15_FP8_DPA:-0}"
FP8_MHA_FLAG="${METIS15_FP8_MHA:-0}"
FP8_EXPERT_PRECISION="${METIS15_FP8_EXPERT_PRECISION:-}"
TE_DOT_PRODUCT_ATTENTION="${METIS15_TE_DOT_PRODUCT_ATTENTION:-0}"
DISABLE_NATIVE_GQA_ATTENTION="${METIS15_DISABLE_NATIVE_GQA_ATTENTION:-0}"
OPTIMIZER_NAME="${METIS15_OPTIMIZER:-}"
FUSED_ADAMW="${METIS15_FUSED_ADAMW:-0}"
HYBRID_ADAMW_IMPL="${METIS15_HYBRID_ADAMW_IMPL:-loop}"
MUON_INCLUDE_ROUTED_EXPERTS="${METIS15_MUON_INCLUDE_ROUTED_EXPERTS:-0}"
MUON_BETA="${METIS15_MUON_BETA:-}"
MUON_NS_STEPS="${METIS15_MUON_NS_STEPS:-}"
MUON_LR_SCALE="${METIS15_MUON_LR_SCALE:-}"
LM_LOSS_IMPL="${METIS15_LM_LOSS_IMPL:-standard}"
RETAIN_STANDARD_CE_LOGITS="${METIS15_RETAIN_STANDARD_CE_LOGITS:-1}"
PREFETCH_BATCHES="${METIS15_PREFETCH_BATCHES:-4}"
MOE_DISPATCH_MODE="${METIS15_MOE_DISPATCH_MODE:-}"
MOE_BACKEND="${METIS15_MOE_BACKEND:-}"
MOE_STATIC_CAPACITY="${METIS15_MOE_STATIC_CAPACITY:-}"
MOE_CAPACITY_FACTOR="${METIS15_MOE_CAPACITY_FACTOR:-}"
MOE_CAPACITY_ALIGNMENT="${METIS15_MOE_CAPACITY_ALIGNMENT:-}"
MOE_OVERFLOW_MODE="${METIS15_MOE_OVERFLOW_MODE:-}"
MOE_GRAPHABLE="${METIS15_MOE_GRAPHABLE:-0}"
MOE_FUSED_COMBINE="${METIS15_MOE_FUSED_COMBINE:-1}"
MOE_MEMORY_EFFICIENT_PERMUTATION="${METIS15_MOE_MEMORY_EFFICIENT_PERMUTATION:-}"
MOE_PERMUTE_FUSION="${METIS15_MOE_PERMUTE_FUSION:-}"
FP8_PAD_MULTIPLE="${METIS15_FP8_PAD_MULTIPLE:-}"
NVFP4_FINAL_EXPERT_LAYERS="${METIS15_NVFP4_FINAL_EXPERT_LAYERS:-}"
NVFP4_FINAL_EXPERT_PRECISION="${METIS15_NVFP4_FINAL_EXPERT_PRECISION:-}"
COMPILE_FLAG="${METIS15_COMPILE:-0}"
COMPILE_MODE="${METIS15_COMPILE_MODE:-default}"
COMPILE_LOW_PRECISION="${METIS15_COMPILE_LOW_PRECISION:-0}"
S3_ROOT="${METIS15_S3_ROOT:-}"

if [[ "$TRAIN_STAGE" != "pretrain" && "$TRAIN_STAGE" != "continued_pretrain" ]]; then
  echo "METIS15_TRAIN_STAGE must be pretrain or continued_pretrain, got: $TRAIN_STAGE" >&2
  exit 1
fi

if [[ "$TRAIN_STAGE" == "continued_pretrain" ]]; then
  DEFAULT_DATA_S3_URI="${S3_ROOT:+$S3_ROOT/pretrain-shards/continued}"
  DEFAULT_CHECKPOINT_S3_URI="${S3_ROOT:+$S3_ROOT/checkpoints/continued}"
  DEFAULT_INIT_CHECKPOINT_S3_URI="${S3_ROOT:+$S3_ROOT/checkpoints/base}"
else
  DEFAULT_DATA_S3_URI="${S3_ROOT:+$S3_ROOT/pretrain-shards/base}"
  DEFAULT_CHECKPOINT_S3_URI="${S3_ROOT:+$S3_ROOT/checkpoints/base}"
  DEFAULT_INIT_CHECKPOINT_S3_URI=""
fi
DATA_S3_URI="${METIS15_S3_PRETRAIN_URI:-$DEFAULT_DATA_S3_URI}"
CHECKPOINT_S3_URI="${METIS15_S3_CHECKPOINTS_URI:-$DEFAULT_CHECKPOINT_S3_URI}"
INIT_CHECKPOINT_S3_URI="${INIT_CHECKPOINT_S3_URI:-$DEFAULT_INIT_CHECKPOINT_S3_URI}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Metis manifest not found: $MANIFEST" >&2
  exit 1
fi

MANIFEST_ATTENTION_BACKEND="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(manifest.get("model", {}).get("attention_backend", "auto"))
PY
)"
MANIFEST_FP8_ENABLED="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(bool(manifest.get("hardware", {}).get("fp8", {}).get("enabled", False))))
PY
)"
MANIFEST_NVFP4_ENABLED="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(manifest.get("model", {}).get("low_precision_mode") == "nvfp4" or bool(manifest.get("hardware", {}).get("nvfp4", {}).get("enabled", False))))
PY
)"
if [[ -z "${METIS15_NVFP4+x}" ]]; then
  ENABLE_NVFP4="$MANIFEST_NVFP4_ENABLED"
fi
if [[ "$ENABLE_FP8" != "0" && "$NVFP4_EXPLICIT" == "0" ]]; then
  # The manifest may default to NVFP4 for Blackwell, but an explicit FP8 run
  # should behave like the H100-style delayed-scaling path and not pass both
  # low-precision recipe flags to train_mamba_lm.py.
  ENABLE_NVFP4=0
fi
if [[ "$ENABLE_FP8" != "0" ]]; then
  FP8_EXPERT_PRECISION="${FP8_EXPERT_PRECISION:-bf16}"
  MOE_DISPATCH_MODE="${MOE_DISPATCH_MODE:-bucketed}"
fi
if [[ -n "$MOE_STATIC_CAPACITY" || "$MOE_GRAPHABLE" == "1" ]]; then
  MOE_DISPATCH_MODE="${MOE_DISPATCH_MODE:-bucketed}"
fi

if [[ "$ENABLE_NVFP4" != "0" && "$MANIFEST_NVFP4_ENABLED" == "1" ]]; then
  NVFP4_RUNTIME_SUPPORTED="$("$PYTHON_BIN" - "$ROOT_DIR" "$MANIFEST" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
sys.path.insert(0, str(root / "src"))
try:
    from metis_mamba.fp8 import transformer_engine_runtime_supports_nvfp4

    manifest = json.loads(manifest_path.read_text())
    model = manifest.get("model", {})
    hardware_nvfp4 = manifest.get("hardware", {}).get("nvfp4", {})
    print(int(transformer_engine_runtime_supports_nvfp4(
        disable_rht=bool(model.get("nvfp4_disable_rht", hardware_nvfp4.get("disable_rht", False))),
        disable_2d_quantization=bool(
            model.get("nvfp4_disable_2d_quantization", hardware_nvfp4.get("disable_2d_quantization", False))
        ),
        disable_stochastic_rounding=bool(
            model.get("nvfp4_disable_stochastic_rounding", hardware_nvfp4.get("disable_stochastic_rounding", False))
        ),
    )))
except Exception:
    print("unknown")
PY
)"
  if [[ "$NVFP4_RUNTIME_SUPPORTED" == "0" ]]; then
    if [[ "$NVFP4_EXPLICIT" == "1" ]]; then
      cat >&2 <<'EOF'
METIS15_NVFP4=1 was requested, but this GPU/runtime does not support the
Metis NVFP4 path safely. On RTX PRO 6000 / SM120, the default Transformer
Engine NVFP4 recipe exposes NVFP4 but fails exact Metis GEMM smoke tests.

Use the SM120-safe reduced recipe in the manifest or set METIS15_NVFP4=0
to run the BF16 fallback.
EOF
      exit 1
    fi
    echo "NVFP4 is enabled in the manifest, but this GPU/runtime failed the SM120 support gate; using BF16 fallback."
    ENABLE_NVFP4=0
  fi
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is required for the Metis AWS launcher." >&2
  exit 1
fi

if [[ "$MANIFEST_ATTENTION_BACKEND" == "flash_attention_3" ]] && ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    from hopper import flash_attn_interface  # noqa: F401
PY
then
  cat >&2 <<'EOF'
FlashAttention-3 is not installed.

Install a FlashAttention build that explicitly supports the active GPU before
launching, or leave the manifest on the default SDPA/Transformer Engine sweep path.
EOF
  exit 1
fi

if [[ ( "$ENABLE_FP8" != "0" && "$MANIFEST_FP8_ENABLED" == "1" ) || ( "$ENABLE_NVFP4" != "0" && "$MANIFEST_NVFP4_ENABLED" == "1" ) ]]; then
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import transformer_engine.pytorch  # noqa: F401
PY
  then
    cat >&2 <<'EOF'
Transformer Engine is not installed.

Install the official PyTorch package before launching FP8 training:
  pip install --no-build-isolation transformer_engine[pytorch]
EOF
    exit 1
  fi
fi

if [[ -n "$S3_ROOT" ]]; then
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import boto3  # noqa: F401
PY
  then
    cat >&2 <<'EOF'
The Metis S3-backed launcher requires boto3 in the active environment.

Install the GPU requirements bundle before launching:
  pip install -r requirements-gpu-train.txt
EOF
    exit 1
  fi
fi

eval "$("$PYTHON_BIN" - "$MANIFEST" "$TRAIN_STAGE" <<'PY'
import json
import math
import os
import shlex
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
stage_name = sys.argv[2]
hardware = manifest["hardware"]
stage = manifest[stage_name]
model = manifest["model"]
launcher = hardware.get("launcher", {})
fp8 = hardware.get("fp8", {})

world_size = int(os.environ.get("METIS15_WORLD_SIZE", hardware["world_size"]))
local_batch_size = int(os.environ.get("METIS15_LOCAL_BATCH_SIZE", stage["local_batch_size"]))
grad_accum_steps = int(os.environ.get("METIS15_GRAD_ACCUM_STEPS", stage["grad_accum_steps"]))
target_train_tokens = int(stage["target_train_tokens"])
seq_len = int(model["block_size"])
tokens_per_step = world_size * local_batch_size * grad_accum_steps * seq_len
max_steps = max(1, math.ceil(target_train_tokens / tokens_per_step))
warmup_steps = max(1, round(max_steps * float(stage["warmup_ratio"])))
gates = stage.get("gates", [])
gate_summary = ",".join(
    f"{gate.get('label', 'gate_' + str(gate['tokens']))}@{max(1, math.ceil(int(gate['tokens']) / tokens_per_step))}"
    for gate in gates
)

values = {
    "TARGET_CLUSTER": hardware.get("target_cluster", "aws_p5"),
    "TRAIN_STAGE": stage_name,
    "STAGE_TRAINING_MODE": stage.get("training_mode", ""),
    "WORLD_SIZE": world_size,
    "LOCAL_BATCH_SIZE": local_batch_size,
    "GRAD_ACCUM_STEPS": grad_accum_steps,
    "MAX_STEPS": max_steps,
    "WARMUP_STEPS": warmup_steps,
    "BASE_LR": stage["base_lr"],
    "WEIGHT_DECAY": stage["weight_decay"],
    "BETA1": stage["optimizer_beta1"],
    "BETA2": stage["optimizer_beta2"],
    "LOG_INTERVAL": stage.get("log_interval", 20),
    "EVAL_INTERVAL": stage.get("eval_interval", 1000),
    "CHECKPOINT_INTERVAL": stage["checkpoint_interval"],
    "TOKENS_PER_STEP": tokens_per_step,
    "PREFERRED_PRECISION": hardware.get("preferred_precision", hardware.get("precision", model.get("torch_dtype", "bfloat16"))),
    "FALLBACK_PRECISION": hardware.get("fallback_precision", hardware.get("precision", model.get("torch_dtype", "bfloat16"))),
    "ATTENTION_BACKEND": model.get("attention_backend", "auto"),
    "OMP_THREADS": launcher.get("omp_num_threads", 8),
    "NCCL_DEBUG_LEVEL": launcher.get("nccl_debug", "WARN"),
    "GATE_SUMMARY": gate_summary,
    "FP8_ENABLED": int(bool(fp8.get("enabled", False))),
    "FP8_FORMAT": fp8.get("format", "HYBRID"),
    "FP8_MARGIN": fp8.get("margin", 0),
    "FP8_AMAX_HISTORY_LEN": fp8.get("amax_history_len", 16),
    "FP8_AMAX_COMPUTE_ALGO": fp8.get("amax_compute_algo", "max"),
    "MODEL_NAME": manifest.get("name", "Metis"),
    "MOE_EXPERT_PARALLEL_SIZE": os.environ.get(
        "METIS15_MOE_EXPERT_PARALLEL_SIZE",
        model.get("moe_expert_parallel_size", 1),
    ),
    "MANIFEST_MOE_MEMORY_EFFICIENT_PERMUTATION": int(bool(model.get("moe_memory_efficient_permutation", False))),
    "MANIFEST_MOE_PERMUTE_FUSION": int(bool(model.get("moe_permute_fusion", True))),
    "NVFP4_ENABLED": int(model.get("low_precision_mode") == "nvfp4" or bool(hardware.get("nvfp4", {}).get("enabled", False))),
    "NVFP4_DISABLE_RHT": int(bool(model.get("nvfp4_disable_rht", hardware.get("nvfp4", {}).get("disable_rht", False)))),
    "NVFP4_DISABLE_2D_QUANTIZATION": int(bool(model.get("nvfp4_disable_2d_quantization", hardware.get("nvfp4", {}).get("disable_2d_quantization", False)))),
    "NVFP4_DISABLE_STOCHASTIC_ROUNDING": int(bool(model.get("nvfp4_disable_stochastic_rounding", hardware.get("nvfp4", {}).get("disable_stochastic_rounding", False)))),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_THREADS}"
export NCCL_DEBUG="${NCCL_DEBUG:-$NCCL_DEBUG_LEVEL}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVTE_DEBUG="${NVTE_DEBUG:-0}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export METIS_TORCH_GROUPED_SAFE_SYNC="${METIS_TORCH_GROUPED_SAFE_SYNC:-0}"
export METIS_ASYNC_METRICS="${METIS_ASYNC_METRICS:-1}"
MOE_MEMORY_EFFICIENT_PERMUTATION="${MOE_MEMORY_EFFICIENT_PERMUTATION:-$MANIFEST_MOE_MEMORY_EFFICIENT_PERMUTATION}"
MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-$MANIFEST_MOE_PERMUTE_FUSION}"

if [[ -n "$DATA_S3_URI" && ! -f "$DATA_DIR/meta.json" ]]; then
  echo "Hydrating $TRAIN_STAGE data from S3: $DATA_S3_URI"
  rm -rf "$DATA_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$DATA_S3_URI" --local-dir "$DATA_DIR" --optional >/dev/null || true
fi

if [[ "$RESUME_TRAINING" != "0" && -n "$CHECKPOINT_S3_URI" && ! -f "$OUT_DIR/latest.pt" ]]; then
  echo "Hydrating checkpoints from S3: $CHECKPOINT_S3_URI"
  rm -rf "$OUT_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$CHECKPOINT_S3_URI" --local-dir "$OUT_DIR" --optional >/dev/null || true
fi

if [[ -n "$INIT_CHECKPOINT" && ! -f "$INIT_CHECKPOINT" && -n "$INIT_CHECKPOINT_S3_URI" ]]; then
  echo "Hydrating init checkpoint directory from S3: $INIT_CHECKPOINT_S3_URI"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$INIT_CHECKPOINT_S3_URI" --local-dir "$(dirname "$INIT_CHECKPOINT")" --optional >/dev/null || true
fi

if [[ -n "$INIT_CHECKPOINT" && ! -f "$INIT_CHECKPOINT" ]]; then
  echo "Init checkpoint is missing at $INIT_CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$DATA_DIR/meta.json" ]]; then
  echo "$TRAIN_STAGE data is missing at $DATA_DIR/meta.json" >&2
  exit 1
fi

echo "Launching $MODEL_NAME $TRAIN_STAGE on $TARGET_CLUSTER"
echo "  manifest: $MANIFEST"
echo "  train stage: $TRAIN_STAGE"
if [[ -n "$STAGE_TRAINING_MODE" ]]; then
  echo "  stage training mode: $STAGE_TRAINING_MODE"
fi
echo "  data dir: $DATA_DIR"
echo "  out dir: $OUT_DIR"
if [[ -n "$INIT_CHECKPOINT" ]]; then
  echo "  init checkpoint: $INIT_CHECKPOINT"
fi
echo "  world size: $WORLD_SIZE"
echo "  local batch size: $LOCAL_BATCH_SIZE"
echo "  grad accum steps: $GRAD_ACCUM_STEPS"
echo "  tokens per optimizer step: $TOKENS_PER_STEP"
echo "  max steps: $MAX_STEPS"
echo "  warmup steps: $WARMUP_STEPS"
echo "  preferred precision: $PREFERRED_PRECISION"
echo "  launcher precision: $FALLBACK_PRECISION"
echo "  attention backend: $ATTENTION_BACKEND"
if [[ "$TE_DOT_PRODUCT_ATTENTION" == "1" ]]; then
  echo "  TE DotProductAttention: enabled"
fi
if [[ "$DISABLE_NATIVE_GQA_ATTENTION" == "1" ]]; then
  echo "  native GQA attention: disabled"
fi
echo "  LM loss impl: $LM_LOSS_IMPL"
if [[ "$RETAIN_STANDARD_CE_LOGITS" == "1" ]]; then
  echo "  retain standard CE logits: enabled"
fi
echo "  prefetch batches: $PREFETCH_BATCHES"
echo "  hybrid AdamW impl: $HYBRID_ADAMW_IMPL"
if [[ "$FUSED_ADAMW" == "1" ]]; then
  echo "  fused/foreach AdamW: enabled"
fi
if [[ -n "$MOE_DISPATCH_MODE" ]]; then
  echo "  MoE dispatch override: $MOE_DISPATCH_MODE"
fi
if [[ -n "$MOE_BACKEND" ]]; then
  echo "  MoE backend override: $MOE_BACKEND"
fi
if [[ -n "$MOE_STATIC_CAPACITY" ]]; then
  echo "  MoE static capacity override: $MOE_STATIC_CAPACITY"
fi
if [[ -n "$MOE_CAPACITY_FACTOR" ]]; then
  echo "  MoE capacity factor override: $MOE_CAPACITY_FACTOR"
fi
if [[ -n "$MOE_OVERFLOW_MODE" ]]; then
  echo "  MoE overflow mode override: $MOE_OVERFLOW_MODE"
fi
if [[ -n "${MOE_ROUTER_OVERRIDE:-}" ]]; then
  echo "  MoE router override: $MOE_ROUTER_OVERRIDE"
fi
if [[ "$MOE_GRAPHABLE" == "1" ]]; then
  echo "  MoE graphable static path: enabled"
fi
if [[ "$MOE_EXPERT_PARALLEL_SIZE" != "1" ]]; then
  echo "  MoE expert parallel size: $MOE_EXPERT_PARALLEL_SIZE"
fi
if [[ "$MOE_MEMORY_EFFICIENT_PERMUTATION" == "1" ]]; then
  echo "  MoE memory-efficient permutation: enabled"
fi
if [[ "$MOE_PERMUTE_FUSION" == "0" ]]; then
  echo "  MoE permute fusion: disabled"
fi
if [[ "$ENABLE_NVFP4" != "0" && "$NVFP4_ENABLED" == "1" ]]; then
  echo "  low precision: NVFP4 mixed training"
fi
if [[ -n "$TRAINING_MODE" ]]; then
  echo "  training mode: $TRAINING_MODE"
fi
echo "  log interval: $LOG_INTERVAL"
echo "  eval interval: $EVAL_INTERVAL"
if [[ -n "$GATE_SUMMARY" ]]; then
  echo "  gates: $GATE_SUMMARY"
fi

cmd=(
  torchrun
  --standalone
  --nproc_per_node="$WORLD_SIZE"
  "$ROOT_DIR/scripts/train_mamba_lm.py"
  --manifest "$MANIFEST"
  --data-dir "$DATA_DIR"
  --out-dir "$OUT_DIR"
  --train-stage "$TRAIN_STAGE"
  --batch-size "$LOCAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --max-steps "$MAX_STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --lr "$BASE_LR"
  --weight-decay "$WEIGHT_DECAY"
  --beta1 "$BETA1"
  --beta2 "$BETA2"
  --lm-loss-impl "$LM_LOSS_IMPL"
  --prefetch-batches "$PREFETCH_BATCHES"
  --log-interval "$LOG_INTERVAL"
  --eval-interval "$EVAL_INTERVAL"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --dtype "${METIS15_DTYPE:-$FALLBACK_PRECISION}"
  --hybrid-adamw-impl "$HYBRID_ADAMW_IMPL"
  --matmul-precision "$MATMUL_PRECISION"
)

if [[ "$FUSED_ADAMW" == "1" ]]; then
  cmd+=(--fused-adamw)
fi
if [[ "$RETAIN_STANDARD_CE_LOGITS" == "1" ]]; then
  cmd+=(--retain-standard-ce-logits)
fi

if [[ -n "$TRAINING_MODE" ]]; then
  cmd+=(--training-mode "$TRAINING_MODE")
fi

if [[ -n "$OPTIMIZER_NAME" ]]; then
  cmd+=(--optimizer "$OPTIMIZER_NAME")
fi
if [[ "$MUON_INCLUDE_ROUTED_EXPERTS" == "1" ]]; then
  cmd+=(--muon-include-routed-experts)
fi
if [[ -n "$MUON_BETA" ]]; then
  cmd+=(--muon-beta "$MUON_BETA")
fi
if [[ -n "$MUON_NS_STEPS" ]]; then
  cmd+=(--muon-ns-steps "$MUON_NS_STEPS")
fi
if [[ -n "$MUON_LR_SCALE" ]]; then
  cmd+=(--muon-lr-scale "$MUON_LR_SCALE")
fi
if [[ -n "$MOE_DISPATCH_MODE" ]]; then
  cmd+=(--moe-dispatch-mode "$MOE_DISPATCH_MODE")
fi
if [[ -n "$MOE_BACKEND" ]]; then
  cmd+=(--moe-backend "$MOE_BACKEND")
fi
if [[ -n "$MOE_STATIC_CAPACITY" ]]; then
  cmd+=(--moe-static-capacity "$MOE_STATIC_CAPACITY")
fi
if [[ -n "$MOE_CAPACITY_FACTOR" ]]; then
  cmd+=(--moe-capacity-factor "$MOE_CAPACITY_FACTOR")
fi
if [[ -n "$MOE_CAPACITY_ALIGNMENT" ]]; then
  cmd+=(--moe-capacity-alignment "$MOE_CAPACITY_ALIGNMENT")
fi
if [[ -n "$MOE_OVERFLOW_MODE" ]]; then
  cmd+=(--moe-overflow-mode "$MOE_OVERFLOW_MODE")
fi
if [[ -n "${MOE_ROUTER_OVERRIDE:-}" ]]; then
  cmd+=(--moe-router-override "$MOE_ROUTER_OVERRIDE")
fi
if [[ "$MOE_FUSED_COMBINE" == "0" ]]; then
  cmd+=(--disable-moe-fused-combine)
fi
if [[ "$MOE_GRAPHABLE" == "1" ]]; then
  cmd+=(--moe-graphable)
fi
if [[ "$MOE_MEMORY_EFFICIENT_PERMUTATION" == "1" ]]; then
  cmd+=(--moe-memory-efficient-permutation)
fi
if [[ "$MOE_PERMUTE_FUSION" == "0" ]]; then
  cmd+=(--disable-moe-permute-fusion)
fi
if [[ "$MOE_EXPERT_PARALLEL_SIZE" != "1" ]]; then
  cmd+=(--moe-expert-parallel-size "$MOE_EXPERT_PARALLEL_SIZE")
fi
if [[ -n "$FP8_PAD_MULTIPLE" ]]; then
  cmd+=(--fp8-pad-multiple "$FP8_PAD_MULTIPLE")
fi
if [[ -n "$NVFP4_FINAL_EXPERT_LAYERS" ]]; then
  cmd+=(--nvfp4-final-expert-layers "$NVFP4_FINAL_EXPERT_LAYERS")
fi
if [[ -n "$NVFP4_FINAL_EXPERT_PRECISION" ]]; then
  cmd+=(--nvfp4-final-expert-precision "$NVFP4_FINAL_EXPERT_PRECISION")
fi
if [[ "$TE_DOT_PRODUCT_ATTENTION" == "1" ]]; then
  cmd+=(--te-dot-product-attention)
fi
if [[ "$DISABLE_NATIVE_GQA_ATTENTION" == "1" ]]; then
  cmd+=(--disable-native-gqa-attention)
fi

if [[ -n "$INIT_CHECKPOINT" ]]; then
  cmd+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

if [[ "$TF32_FLAG" == "1" ]]; then
  cmd+=(--tf32)
fi
if [[ "$COMPILE_FLAG" == "1" ]]; then
  cmd+=(--compile --compile-mode "$COMPILE_MODE")
  if [[ "$COMPILE_LOW_PRECISION" == "1" ]]; then
    cmd+=(--allow-low-precision-compile)
  fi
fi

if [[ "$ENABLE_FP8" != "0" && "$FP8_ENABLED" == "1" ]]; then
  cmd+=(
    --fp8
    --fp8-format "$FP8_FORMAT"
    --fp8-margin "$FP8_MARGIN"
    --fp8-amax-history-len "$FP8_AMAX_HISTORY_LEN"
    --fp8-amax-compute-algo "$FP8_AMAX_COMPUTE_ALGO"
  )
  if [[ -n "$FP8_EXPERT_PRECISION" ]]; then
    cmd+=(--fp8-expert-precision "$FP8_EXPERT_PRECISION")
  fi
  if [[ "$FP8_DPA_FLAG" == "1" ]]; then
    cmd+=(--fp8-dpa)
  fi
  if [[ "$FP8_MHA_FLAG" == "1" ]]; then
    cmd+=(--fp8-mha)
  fi
fi

if [[ "$ENABLE_NVFP4" != "0" && "$NVFP4_ENABLED" == "1" ]]; then
  cmd+=(--nvfp4)
  if [[ "$NVFP4_DISABLE_RHT" == "1" ]]; then
    cmd+=(--nvfp4-disable-rht)
  fi
  if [[ "$NVFP4_DISABLE_2D_QUANTIZATION" == "1" ]]; then
    cmd+=(--nvfp4-disable-2d-quantization)
  fi
  if [[ "$NVFP4_DISABLE_STOCHASTIC_ROUNDING" == "1" ]]; then
    cmd+=(--nvfp4-disable-stochastic-rounding)
  fi
fi

if [[ "$RESUME_TRAINING" != "0" ]]; then
  cmd+=(--resume)
fi

"${cmd[@]}"

if [[ -n "$CHECKPOINT_S3_URI" && -d "$OUT_DIR" ]]; then
  echo "Uploading $TRAIN_STAGE checkpoints to S3: $CHECKPOINT_S3_URI"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" upload-dir --local-dir "$OUT_DIR" --s3-uri "$CHECKPOINT_S3_URI" >/dev/null
fi
