#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
TRAIN_STAGE="${METIS15_TRAIN_STAGE:-pretrain}"
if [[ "$TRAIN_STAGE" == "continued_pretrain" ]]; then
  DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_continued_pretrain}"
  OUT_DIR="${METIS15_OUT_DIR:-$ROOT_DIR/checkpoints/metis15_continued_pretrain_neuron}"
else
  DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_base}"
  OUT_DIR="${METIS15_OUT_DIR:-$ROOT_DIR/checkpoints/metis15_base_neuron}"
fi

S3_ROOT="${METIS15_S3_ROOT:-}"
if [[ "$TRAIN_STAGE" == "continued_pretrain" ]]; then
  DEFAULT_DATA_S3_URI="${S3_ROOT:+$S3_ROOT/pretrain-shards/continued}"
  DEFAULT_CHECKPOINT_S3_URI="${S3_ROOT:+$S3_ROOT/checkpoints/continued-neuron}"
else
  DEFAULT_DATA_S3_URI="${S3_ROOT:+$S3_ROOT/pretrain-shards/base}"
  DEFAULT_CHECKPOINT_S3_URI="${S3_ROOT:+$S3_ROOT/checkpoints/base-neuron}"
fi
DATA_S3_URI="${METIS15_S3_PRETRAIN_URI:-$DEFAULT_DATA_S3_URI}"
CHECKPOINT_S3_URI="${METIS15_S3_CHECKPOINTS_URI:-$DEFAULT_CHECKPOINT_S3_URI}"

NEURON_WORLD_SIZE="${METIS15_NEURON_WORLD_SIZE:-32}"
LOCAL_BATCH_SIZE="${METIS15_NEURON_LOCAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${METIS15_NEURON_GRAD_ACCUM_STEPS:-16}"
SYNTHETIC_DATA="${METIS15_NEURON_SYNTHETIC:-0}"
SKIP_CHECKPOINT="${METIS15_NEURON_SKIP_CHECKPOINT:-0}"
AUTO_RESUME="${METIS15_NEURON_AUTO_RESUME:-1}"
RESUME_FROM="${METIS15_NEURON_RESUME_FROM:-}"
OVERRIDE_LR_ON_RESUME="${METIS15_NEURON_OVERRIDE_LR_ON_RESUME:-0}"
EXPERT_CAPACITY_FACTOR="${METIS15_NEURON_EXPERT_CAPACITY_FACTOR:-4.0}"
EXPERT_CAPACITY="${METIS15_NEURON_EXPERT_CAPACITY:-}"
DISPATCH_PACK_IMPL="${METIS15_NEURON_DISPATCH_PACK_IMPL:-index_add}"
BALANCED_STATIC_LAYOUT="${METIS15_NEURON_BALANCED_STATIC_LAYOUT:-indexed}"
BALANCED_STATIC_ROUTER_WEIGHTS="${METIS15_NEURON_BALANCED_STATIC_ROUTER_WEIGHTS:-uniform}"
BALANCED_STATIC_ROUTER_INPUT="${METIS15_NEURON_BALANCED_STATIC_ROUTER_INPUT:-hidden}"
ROUTER_OVERRIDE="${METIS15_NEURON_ROUTER_OVERRIDE:-learned}"
LOSS_MODE="${METIS15_NEURON_LOSS_MODE:-real_ce}"
CE_IMPL="${METIS15_NEURON_CE_IMPL:-cross_entropy}"
CE_LOGITS_DTYPE="${METIS15_NEURON_CE_LOGITS_DTYPE:-float32}"
ATTENTION_MODE="${METIS15_NEURON_ATTENTION_MODE:-real}"
ATTENTION_KERNEL="${METIS15_NEURON_ATTENTION_KERNEL:-eager}"
NKI_FLASH_LSE_DTYPE="${METIS15_NEURON_NKI_FLASH_LSE_DTYPE:-auto}"
MOE_MODE="${METIS15_NEURON_MOE_MODE:-real}"
OPTIMIZER_NAME="${METIS15_NEURON_OPTIMIZER:-adamw}"
HYBRID_ADAMW_IMPL="${METIS15_NEURON_HYBRID_ADAMW_IMPL:-loop}"
OPTIMIZER_MASTER_WEIGHTS="${METIS15_NEURON_OPTIMIZER_MASTER_WEIGHTS:-0}"
MUON_BETA="${METIS15_NEURON_MUON_BETA:-}"
MUON_NS_STEPS="${METIS15_NEURON_MUON_NS_STEPS:-}"
MUON_LR_SCALE="${METIS15_NEURON_MUON_LR_SCALE:-}"
MUON_SCALE_MODE="${METIS15_NEURON_MUON_SCALE_MODE:-}"
MUON_INCLUDE_ROUTED_EXPERTS="${METIS15_NEURON_MUON_INCLUDE_ROUTED_EXPERTS:-0}"
TIE_EMBEDDINGS="${METIS15_NEURON_TIE_EMBEDDINGS:-}"
GRAD_SYNC_MODE="${METIS15_NEURON_GRAD_SYNC_MODE:-all_reduce}"
GRAD_SYNC_BUCKET_MB="${METIS15_NEURON_GRAD_SYNC_BUCKET_MB:-0}"
MARK_STEP_EACH_MICROBATCH="${METIS15_NEURON_MARK_STEP_EACH_MICROBATCH:-0}"
LOCAL_LOG_METRICS="${METIS15_NEURON_LOCAL_LOG_METRICS:-0}"
ACTIVATION_CHECKPOINTING="${METIS15_NEURON_ACTIVATION_CHECKPOINTING:-none}"
ACTIVATION_CHECKPOINT_LAYER_INTERVAL="${METIS15_NEURON_ACTIVATION_CHECKPOINT_LAYER_INTERVAL:-1}"
PERF_WARMUP_STEPS="${METIS15_NEURON_PERF_WARMUP_STEPS:-0}"
PROFILE_COMPONENTS="${METIS15_NEURON_PROFILE_COMPONENTS:-0}"
LOG_EXPERT_HISTOGRAMS="${METIS15_NEURON_LOG_EXPERT_HISTOGRAMS:-0}"
DEBUG_GRAD_NORM_INTERVAL="${METIS15_NEURON_DEBUG_GRAD_NORM_INTERVAL:-0}"
DEBUG_PARAM_DELTA_INTERVAL="${METIS15_NEURON_DEBUG_PARAM_DELTA_INTERVAL:-0}"
DISABLE_MOE_BALANCE_UPDATE="${METIS15_NEURON_DISABLE_MOE_BALANCE_UPDATE:-0}"
MOE_BALANCE_BIAS_UPDATE_RATE="${METIS15_NEURON_MOE_BALANCE_BIAS_UPDATE_RATE:-}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Metis manifest not found: $MANIFEST" >&2
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is required for the Metis Neuron launcher." >&2
  exit 1
fi

eval "$("$PYTHON_BIN" - "$MANIFEST" "$TRAIN_STAGE" "$NEURON_WORLD_SIZE" "$LOCAL_BATCH_SIZE" "$GRAD_ACCUM_STEPS" <<'PY'
import json
import math
import os
import shlex
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
stage = manifest[sys.argv[2]]
model = manifest["model"]
world_size = int(sys.argv[3])
local_batch_size = int(sys.argv[4])
grad_accum_steps = int(sys.argv[5])
seq_len = int(os.environ.get("METIS15_NEURON_BLOCK_SIZE", model["block_size"]))
target_train_tokens = int(stage["target_train_tokens"])
tokens_per_step = world_size * local_batch_size * grad_accum_steps * seq_len
max_steps = max(1, math.ceil(target_train_tokens / tokens_per_step))
warmup_steps = max(1, round(max_steps * float(stage.get("warmup_ratio", 0.03))))
values = {
    "MODEL_NAME": manifest.get("name", "Metis"),
    "DEFAULT_MAX_STEPS": max_steps,
    "DEFAULT_WARMUP_STEPS": warmup_steps,
    "BASE_LR": stage["base_lr"],
    "WEIGHT_DECAY": stage["weight_decay"],
    "BETA1": stage["optimizer_beta1"],
    "BETA2": stage["optimizer_beta2"],
    "LOG_INTERVAL": stage.get("log_interval", 20),
    "CHECKPOINT_INTERVAL": stage["checkpoint_interval"],
    "TOKENS_PER_STEP": tokens_per_step,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

MAX_STEPS="${METIS15_NEURON_MAX_STEPS:-$DEFAULT_MAX_STEPS}"
WARMUP_STEPS="${METIS15_NEURON_WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
LOG_INTERVAL="${METIS15_NEURON_LOG_INTERVAL:-$LOG_INTERVAL}"
CHECKPOINT_INTERVAL="${METIS15_NEURON_CHECKPOINT_INTERVAL:-$CHECKPOINT_INTERVAL}"
BASE_LR="${METIS15_NEURON_LR:-$BASE_LR}"
WEIGHT_DECAY="${METIS15_NEURON_WEIGHT_DECAY:-$WEIGHT_DECAY}"
BETA1="${METIS15_NEURON_BETA1:-$BETA1}"
BETA2="${METIS15_NEURON_BETA2:-$BETA2}"
GRAD_CLIP="${METIS15_NEURON_GRAD_CLIP:-1.0}"
PREINIT_OPTIMIZER_STATE="${METIS15_NEURON_PREINIT_OPTIMIZER_STATE:-1}"
EXPERT_ACTIVATION_SAFETY="${METIS15_NEURON_EXPERT_ACTIVATION_SAFETY:-clamp}"
CONSTANT_LR="${METIS15_NEURON_CONSTANT_LR:-1}"

export PJRT_DEVICE="${PJRT_DEVICE:-NEURON}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-$NEURON_WORLD_SIZE}"
export NEURON_CC_CACHE="${NEURON_CC_CACHE:-$ROOT_DIR/.neuron_cc_cache}"
mkdir -p "$NEURON_CC_CACHE" "$OUT_DIR"
if [[ -z "${NEURON_CC_FLAGS:-}" ]]; then
  export NEURON_CC_FLAGS="--cache_dir=$NEURON_CC_CACHE --auto-cast=none ${NEURON_CC_FLAGS_EXTRA:-}"
fi

download_s3_dir() {
  local s3_uri="$1"
  local local_dir="$2"
  local optional="${3:-0}"
  mkdir -p "$local_dir"
  if [[ "$optional" == "1" ]]; then
    if "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$s3_uri" --local-dir "$local_dir" --optional >/dev/null; then
      return 0
    fi
  else
    if "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$s3_uri" --local-dir "$local_dir" >/dev/null; then
      return 0
    fi
  fi
  if command -v aws >/dev/null 2>&1; then
    echo "Python S3 helper failed for $s3_uri; falling back to aws s3 sync." >&2
    aws s3 sync "${s3_uri%/}/" "$local_dir/" --only-show-errors
    return 0
  fi
  return 1
}

if [[ "$SYNTHETIC_DATA" != "1" && -n "$DATA_S3_URI" && ( ! -f "$DATA_DIR/meta.json" || ! -f "$DATA_DIR/train.bin" ) ]]; then
  echo "Hydrating $TRAIN_STAGE data from S3: $DATA_S3_URI"
  rm -rf "$DATA_DIR"
  download_s3_dir "$DATA_S3_URI" "$DATA_DIR" 0
fi

if [[ "$SYNTHETIC_DATA" != "1" && "$AUTO_RESUME" != "0" && -z "$RESUME_FROM" && -n "$CHECKPOINT_S3_URI" && ! -f "$OUT_DIR/latest.pt" ]]; then
  echo "Hydrating $TRAIN_STAGE Neuron checkpoints from S3: $CHECKPOINT_S3_URI"
  rm -rf "$OUT_DIR"
  download_s3_dir "$CHECKPOINT_S3_URI" "$OUT_DIR" 1 || true
fi

if [[ "$SYNTHETIC_DATA" != "1" && ( ! -f "$DATA_DIR/meta.json" || ! -f "$DATA_DIR/train.bin" ) ]]; then
  echo "$TRAIN_STAGE data is missing at $DATA_DIR/meta.json or $DATA_DIR/train.bin" >&2
  echo "Set METIS15_NEURON_SYNTHETIC=1 for compile/performance smoke tests." >&2
  exit 1
fi

if [[ "$AUTO_RESUME" != "0" && -z "$RESUME_FROM" && -f "$OUT_DIR/latest.pt" ]]; then
  RESUME_FROM="$OUT_DIR/latest.pt"
fi

echo "Launching $MODEL_NAME $TRAIN_STAGE on AWS Neuron"
echo "  manifest: $MANIFEST"
echo "  data dir: $DATA_DIR"
echo "  out dir: $OUT_DIR"
echo "  world size / NeuronCores: $NEURON_WORLD_SIZE"
echo "  local batch size: $LOCAL_BATCH_SIZE"
echo "  grad accum steps: $GRAD_ACCUM_STEPS"
echo "  tokens per optimizer step: $TOKENS_PER_STEP"
echo "  max steps: $MAX_STEPS"
echo "  warmup steps: $WARMUP_STEPS"
echo "  expert capacity factor: $EXPERT_CAPACITY_FACTOR"
echo "  dispatch pack impl: $DISPATCH_PACK_IMPL"
echo "  single latent router input: $BALANCED_STATIC_ROUTER_INPUT"
if [[ "$DISPATCH_PACK_IMPL" == "balanced_static" ]]; then
  echo "  balanced static layout: $BALANCED_STATIC_LAYOUT"
  echo "  balanced static router weights: $BALANCED_STATIC_ROUTER_WEIGHTS"
  echo "  balanced static router input: $BALANCED_STATIC_ROUTER_INPUT"
fi
echo "  ablations: router=$ROUTER_OVERRIDE loss=$LOSS_MODE ce_impl=$CE_IMPL ce_logits_dtype=$CE_LOGITS_DTYPE attention=$ATTENTION_MODE attention_kernel=$ATTENTION_KERNEL nki_lse=$NKI_FLASH_LSE_DTYPE moe=$MOE_MODE"
echo "  optimizer: $OPTIMIZER_NAME"
echo "  optimizer master weights: $OPTIMIZER_MASTER_WEIGHTS"
echo "  constant lr: $CONSTANT_LR"
if [[ -n "$TIE_EMBEDDINGS" ]]; then
  echo "  tie embeddings override: $TIE_EMBEDDINGS"
fi
echo "  grad sync mode: $GRAD_SYNC_MODE"
echo "  grad sync bucket MB: $GRAD_SYNC_BUCKET_MB"
echo "  mark step each microbatch: $MARK_STEP_EACH_MICROBATCH"
echo "  local log metrics: $LOCAL_LOG_METRICS"
echo "  grad clip: $GRAD_CLIP"
echo "  preinit optimizer state: $PREINIT_OPTIMIZER_STATE"
echo "  expert activation safety: $EXPERT_ACTIVATION_SAFETY"
echo "  activation checkpointing: $ACTIVATION_CHECKPOINTING (layer_interval=$ACTIVATION_CHECKPOINT_LAYER_INTERVAL)"
echo "  perf warmup steps: $PERF_WARMUP_STEPS"
echo "  debug intervals: grad_norm=$DEBUG_GRAD_NORM_INTERVAL param_delta=$DEBUG_PARAM_DELTA_INTERVAL"
echo "  Neuron compiler cache: $NEURON_CC_CACHE"
if [[ "$SYNTHETIC_DATA" == "1" ]]; then
  echo "  synthetic data: enabled"
fi
if [[ -n "$RESUME_FROM" ]]; then
  echo "  resume from: $RESUME_FROM"
  echo "  override LR on resume: $OVERRIDE_LR_ON_RESUME"
fi

cmd=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="$NEURON_WORLD_SIZE"
  "$ROOT_DIR/scripts/train_metis15_neuron.py"
  --manifest "$MANIFEST"
  --out-dir "$OUT_DIR"
  --device xla
  --batch-size "$LOCAL_BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --max-steps "$MAX_STEPS"
  --warmup-steps "$WARMUP_STEPS"
  --lr "$BASE_LR"
  --weight-decay "$WEIGHT_DECAY"
  --beta1 "$BETA1"
  --beta2 "$BETA2"
  --grad-clip "$GRAD_CLIP"
  --log-interval "$LOG_INTERVAL"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --expert-capacity-factor "$EXPERT_CAPACITY_FACTOR"
  --dispatch-pack-impl "$DISPATCH_PACK_IMPL"
  --balanced-static-layout "$BALANCED_STATIC_LAYOUT"
  --balanced-static-router-weights "$BALANCED_STATIC_ROUTER_WEIGHTS"
  --balanced-static-router-input "$BALANCED_STATIC_ROUTER_INPUT"
  --router-override "$ROUTER_OVERRIDE"
  --loss-mode "$LOSS_MODE"
  --ce-impl "$CE_IMPL"
  --ce-logits-dtype "$CE_LOGITS_DTYPE"
  --attention-mode "$ATTENTION_MODE"
  --attention-kernel "$ATTENTION_KERNEL"
  --nki-flash-lse-dtype "$NKI_FLASH_LSE_DTYPE"
  --moe-mode "$MOE_MODE"
  --grad-sync-mode "$GRAD_SYNC_MODE"
  --grad-sync-bucket-mb "$GRAD_SYNC_BUCKET_MB"
  --activation-checkpointing "$ACTIVATION_CHECKPOINTING"
  --activation-checkpoint-layer-interval "$ACTIVATION_CHECKPOINT_LAYER_INTERVAL"
  --perf-warmup-steps "$PERF_WARMUP_STEPS"
  --expert-activation-safety "$EXPERT_ACTIVATION_SAFETY"
  --optimizer "$OPTIMIZER_NAME"
  --hybrid-adamw-impl "$HYBRID_ADAMW_IMPL"
)

if [[ "$SYNTHETIC_DATA" == "1" ]]; then
  cmd+=(--synthetic-data)
else
  cmd+=(--data-dir "$DATA_DIR")
fi
if [[ "$SKIP_CHECKPOINT" == "1" ]]; then
  cmd+=(--skip-checkpoint)
fi
if [[ "$PREINIT_OPTIMIZER_STATE" == "1" ]]; then
  cmd+=(--preinit-optimizer-state)
fi
if [[ "$CONSTANT_LR" == "1" ]]; then
  cmd+=(--constant-lr)
elif [[ "$CONSTANT_LR" != "0" ]]; then
  echo "METIS15_NEURON_CONSTANT_LR must be 0 or 1; got '$CONSTANT_LR'" >&2
  exit 1
fi
if [[ "$OPTIMIZER_MASTER_WEIGHTS" == "1" ]]; then
  cmd+=(--optimizer-master-weights)
fi
if [[ "$MARK_STEP_EACH_MICROBATCH" == "1" ]]; then
  cmd+=(--mark-step-each-microbatch)
fi
if [[ "$LOCAL_LOG_METRICS" == "1" ]]; then
  cmd+=(--local-log-metrics)
fi
if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
  if [[ "$OVERRIDE_LR_ON_RESUME" == "1" ]]; then
    cmd+=(--override-optimizer-lr-on-resume)
  elif [[ "$OVERRIDE_LR_ON_RESUME" != "0" ]]; then
    echo "METIS15_NEURON_OVERRIDE_LR_ON_RESUME must be 0 or 1; got '$OVERRIDE_LR_ON_RESUME'" >&2
    exit 1
  fi
fi
if [[ -n "$EXPERT_CAPACITY" ]]; then
  cmd+=(--expert-capacity "$EXPERT_CAPACITY")
fi
if [[ "$PROFILE_COMPONENTS" == "1" ]]; then
  cmd+=(--profile-components)
fi
if [[ "$LOG_EXPERT_HISTOGRAMS" == "1" ]]; then
  cmd+=(--log-expert-histograms)
fi
if [[ "$DEBUG_GRAD_NORM_INTERVAL" != "0" ]]; then
  cmd+=(--debug-grad-norm-interval "$DEBUG_GRAD_NORM_INTERVAL")
fi
if [[ "$DEBUG_PARAM_DELTA_INTERVAL" != "0" ]]; then
  cmd+=(--debug-param-delta-interval "$DEBUG_PARAM_DELTA_INTERVAL")
fi
if [[ "$DISABLE_MOE_BALANCE_UPDATE" == "1" ]]; then
  cmd+=(--disable-moe-balance-update)
fi
if [[ -n "$MOE_BALANCE_BIAS_UPDATE_RATE" ]]; then
  cmd+=(--moe-balance-bias-update-rate "$MOE_BALANCE_BIAS_UPDATE_RATE")
fi

append_optional_arg() {
  local value="$1"
  local flag="$2"
  if [[ -n "$value" ]]; then
    cmd+=("$flag" "$value")
  fi
}

append_optional_arg "$MUON_BETA" "--muon-beta"
append_optional_arg "$MUON_NS_STEPS" "--muon-ns-steps"
append_optional_arg "$MUON_LR_SCALE" "--muon-lr-scale"
append_optional_arg "$MUON_SCALE_MODE" "--muon-scale-mode"
if [[ "$MUON_INCLUDE_ROUTED_EXPERTS" == "1" ]]; then
  cmd+=(--muon-include-routed-experts)
fi
if [[ "$TIE_EMBEDDINGS" == "1" ]]; then
  cmd+=(--tie-embeddings)
elif [[ "$TIE_EMBEDDINGS" == "0" ]]; then
  cmd+=(--untie-embeddings)
elif [[ -n "$TIE_EMBEDDINGS" ]]; then
  echo "METIS15_NEURON_TIE_EMBEDDINGS must be 0, 1, or unset; got '$TIE_EMBEDDINGS'" >&2
  exit 1
fi
append_optional_arg "${METIS15_NEURON_BLOCK_SIZE:-}" "--block-size"
append_optional_arg "${METIS15_NEURON_VOCAB_SIZE:-}" "--vocab-size"
append_optional_arg "${METIS15_NEURON_D_MODEL:-}" "--d-model"
append_optional_arg "${METIS15_NEURON_N_LAYER:-}" "--n-layer"
append_optional_arg "${METIS15_NEURON_N_HEADS:-}" "--n-heads"
append_optional_arg "${METIS15_NEURON_N_KV_HEADS:-}" "--n-kv-heads"
append_optional_arg "${METIS15_NEURON_HEAD_DIM:-}" "--head-dim"
append_optional_arg "${METIS15_NEURON_MOE_NUM_EXPERTS:-}" "--moe-num-experts"
append_optional_arg "${METIS15_NEURON_MOE_TOP_K:-}" "--moe-top-k"
append_optional_arg "${METIS15_NEURON_MOE_EXPERT_INTERMEDIATE_SIZE:-}" "--moe-expert-intermediate-size"
append_optional_arg "${METIS15_NEURON_MOE_ROUTED_LATENT_SIZE:-}" "--moe-routed-latent-size"
append_optional_arg "${METIS15_NEURON_MOE_ROUTER_LATENT_SIZE:-}" "--moe-router-latent-size"

"${cmd[@]}"

if [[ -n "$CHECKPOINT_S3_URI" && -d "$OUT_DIR" && "$SKIP_CHECKPOINT" != "1" ]]; then
  echo "Uploading $TRAIN_STAGE Neuron checkpoints to S3: $CHECKPOINT_S3_URI"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" upload-dir --local-dir "$OUT_DIR" --s3-uri "$CHECKPOINT_S3_URI" >/dev/null
fi
