#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
TRAIN_STAGE="${METIS15_TRAIN_STAGE:-pretrain}"
if [[ "$TRAIN_STAGE" == "continued_pretrain" ]]; then
  DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_continued_pretrain}"
  OUT_DIR="${METIS15_OUT_DIR:-$ROOT_DIR/checkpoints/metis15_continued_pretrain_tpu_v6e}"
else
  DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_base}"
  OUT_DIR="${METIS15_OUT_DIR:-$ROOT_DIR/checkpoints/metis15_base_tpu_v6e}"
fi
TRAIN_LOG="${METIS15_TPU_TRAIN_LOG:-$OUT_DIR/train.log}"

GCS_ROOT="${METIS15_GCS_ROOT:-}"
if [[ "$TRAIN_STAGE" == "continued_pretrain" ]]; then
  DEFAULT_DATA_GCS_URI="${GCS_ROOT:+$GCS_ROOT/pretrain-shards/continued}"
  DEFAULT_CHECKPOINT_GCS_URI="${GCS_ROOT:+$GCS_ROOT/checkpoints/continued-tpu-v6e}"
else
  DEFAULT_DATA_GCS_URI="${GCS_ROOT:+$GCS_ROOT/pretrain-shards/base}"
  DEFAULT_CHECKPOINT_GCS_URI="${GCS_ROOT:+$GCS_ROOT/checkpoints/base-tpu-v6e}"
fi
DATA_GCS_URI="${METIS15_GCS_PRETRAIN_URI:-$DEFAULT_DATA_GCS_URI}"
CHECKPOINT_GCS_URI="${METIS15_GCS_CHECKPOINTS_URI:-$DEFAULT_CHECKPOINT_GCS_URI}"

TPU_WORLD_SIZE="${METIS15_TPU_WORLD_SIZE:-8}"
LOCAL_BATCH_SIZE="${METIS15_TPU_LOCAL_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${METIS15_TPU_GRAD_ACCUM_STEPS:-16}"
SYNTHETIC_DATA="${METIS15_TPU_SYNTHETIC:-0}"
SKIP_CHECKPOINT="${METIS15_TPU_SKIP_CHECKPOINT:-0}"
AUTO_RESUME="${METIS15_TPU_AUTO_RESUME:-1}"
RESUME_FROM="${METIS15_TPU_RESUME_FROM:-}"
OVERRIDE_LR_ON_RESUME="${METIS15_TPU_OVERRIDE_LR_ON_RESUME:-1}"
EXPERT_CAPACITY_FACTOR="${METIS15_TPU_EXPERT_CAPACITY_FACTOR:-4.0}"
EXPERT_CAPACITY="${METIS15_TPU_EXPERT_CAPACITY:-}"
DISPATCH_PACK_IMPL="${METIS15_TPU_DISPATCH_PACK_IMPL:-index_add}"
BALANCED_STATIC_LAYOUT="${METIS15_TPU_BALANCED_STATIC_LAYOUT:-indexed}"
BALANCED_STATIC_ROUTER_WEIGHTS="${METIS15_TPU_BALANCED_STATIC_ROUTER_WEIGHTS:-uniform}"
BALANCED_STATIC_ROUTER_INPUT="${METIS15_TPU_BALANCED_STATIC_ROUTER_INPUT:-hidden}"
ROUTER_OVERRIDE="${METIS15_TPU_ROUTER_OVERRIDE:-learned}"
LOSS_MODE="${METIS15_TPU_LOSS_MODE:-real_ce}"
CE_IMPL="${METIS15_TPU_CE_IMPL:-cross_entropy}"
CE_LOGITS_DTYPE="${METIS15_TPU_CE_LOGITS_DTYPE:-float32}"
ATTENTION_MODE="${METIS15_TPU_ATTENTION_MODE:-real}"
ATTENTION_KERNEL="${METIS15_TPU_ATTENTION_KERNEL:-sdpa}"
MOE_MODE="${METIS15_TPU_MOE_MODE:-real}"
OPTIMIZER_NAME="${METIS15_TPU_OPTIMIZER:-muon_adamw}"
HYBRID_ADAMW_IMPL="${METIS15_TPU_HYBRID_ADAMW_IMPL:-loop}"
OPTIMIZER_MASTER_WEIGHTS="${METIS15_TPU_OPTIMIZER_MASTER_WEIGHTS:-1}"
MUON_BETA="${METIS15_TPU_MUON_BETA:-}"
MUON_NS_STEPS="${METIS15_TPU_MUON_NS_STEPS:-}"
MUON_LR_SCALE="${METIS15_TPU_MUON_LR_SCALE:-}"
MUON_SCALE_MODE="${METIS15_TPU_MUON_SCALE_MODE:-match_rms_adamw}"
MUON_INCLUDE_ROUTED_EXPERTS="${METIS15_TPU_MUON_INCLUDE_ROUTED_EXPERTS:-0}"
TIE_EMBEDDINGS="${METIS15_TPU_TIE_EMBEDDINGS:-}"
GRAD_SYNC_MODE="${METIS15_TPU_GRAD_SYNC_MODE:-all_reduce_staged}"
GRAD_SYNC_BUCKET_MB="${METIS15_TPU_GRAD_SYNC_BUCKET_MB:-16}"
MARK_STEP_EACH_MICROBATCH="${METIS15_TPU_MARK_STEP_EACH_MICROBATCH:-0}"
LOCAL_LOG_METRICS="${METIS15_TPU_LOCAL_LOG_METRICS:-0}"
ACTIVATION_CHECKPOINTING="${METIS15_TPU_ACTIVATION_CHECKPOINTING:-none}"
ACTIVATION_CHECKPOINT_LAYER_INTERVAL="${METIS15_TPU_ACTIVATION_CHECKPOINT_LAYER_INTERVAL:-1}"
PERF_WARMUP_STEPS="${METIS15_TPU_PERF_WARMUP_STEPS:-3}"
PROFILE_COMPONENTS="${METIS15_TPU_PROFILE_COMPONENTS:-1}"
LOG_EXPERT_HISTOGRAMS="${METIS15_TPU_LOG_EXPERT_HISTOGRAMS:-0}"
DEBUG_GRAD_NORM_INTERVAL="${METIS15_TPU_DEBUG_GRAD_NORM_INTERVAL:-0}"
DEBUG_PARAM_DELTA_INTERVAL="${METIS15_TPU_DEBUG_PARAM_DELTA_INTERVAL:-0}"
DISABLE_MOE_BALANCE_UPDATE="${METIS15_TPU_DISABLE_MOE_BALANCE_UPDATE:-0}"
MOE_BALANCE_BIAS_UPDATE_RATE="${METIS15_TPU_MOE_BALANCE_BIAS_UPDATE_RATE:-}"
QK_CLIP_THRESHOLD="${METIS15_TPU_QK_CLIP_THRESHOLD:-100}"
QK_CLIP_ALPHA="${METIS15_TPU_QK_CLIP_ALPHA:-0.5}"
QK_CLIP_INTERVAL="${METIS15_TPU_QK_CLIP_INTERVAL:-1}"
QK_CLIP_WARMUP_STEPS="${METIS15_TPU_QK_CLIP_WARMUP_STEPS:-0}"
FIXED_BATCH="${METIS15_TPU_FIXED_BATCH:-0}"
RUN_PREFLIGHT="${METIS15_TPU_PREFLIGHT:-1}"
SKIP_PREFLIGHT_DEVICE_CHECK="${METIS15_TPU_PREFLIGHT_SKIP_DEVICE_CHECK:-0}"
ALLOW_WORLD_SIZE_MISMATCH="${METIS15_TPU_ALLOW_WORLD_SIZE_MISMATCH:-0}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Metis manifest not found: $MANIFEST" >&2
  exit 1
fi

if command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
  TORCHRUN_CMD=("$TORCHRUN_BIN")
elif "$PYTHON_BIN" -c "import torch.distributed.run" >/dev/null 2>&1; then
  TORCHRUN_CMD=("$PYTHON_BIN" -m torch.distributed.run)
else
  echo "torchrun is required for the Metis TPU launcher." >&2
  echo "Set TORCHRUN_BIN=/path/to/torchrun or use a Python with torch.distributed.run." >&2
  exit 1
fi

eval "$("$PYTHON_BIN" - "$MANIFEST" "$TRAIN_STAGE" "$TPU_WORLD_SIZE" "$LOCAL_BATCH_SIZE" "$GRAD_ACCUM_STEPS" <<'PY'
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
seq_len = int(os.environ.get("METIS15_TPU_BLOCK_SIZE", model["block_size"]))
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

MAX_STEPS="${METIS15_TPU_MAX_STEPS:-$DEFAULT_MAX_STEPS}"
WARMUP_STEPS="${METIS15_TPU_WARMUP_STEPS:-$DEFAULT_WARMUP_STEPS}"
LOG_INTERVAL="${METIS15_TPU_LOG_INTERVAL:-$LOG_INTERVAL}"
CHECKPOINT_INTERVAL="${METIS15_TPU_CHECKPOINT_INTERVAL:-$CHECKPOINT_INTERVAL}"
BASE_LR="${METIS15_TPU_LR:-$BASE_LR}"
WEIGHT_DECAY="${METIS15_TPU_WEIGHT_DECAY:-$WEIGHT_DECAY}"
BETA1="${METIS15_TPU_BETA1:-$BETA1}"
BETA2="${METIS15_TPU_BETA2:-$BETA2}"
GRAD_CLIP="${METIS15_TPU_GRAD_CLIP:-1.0}"
PREINIT_OPTIMIZER_STATE="${METIS15_TPU_PREINIT_OPTIMIZER_STATE:-1}"
EXPERT_ACTIVATION_SAFETY="${METIS15_TPU_EXPERT_ACTIVATION_SAFETY:-clamp}"
CONSTANT_LR="${METIS15_TPU_CONSTANT_LR:-0}"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
if [[ -n "${METIS15_TPU_PREMAPPED_BUFFER_SIZE:-}" ]]; then
  export TPU_PREMAPPED_BUFFER_SIZE="$METIS15_TPU_PREMAPPED_BUFFER_SIZE"
fi
mkdir -p "$DATA_DIR" "$OUT_DIR"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  preflight_cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/scripts/metis15_tpu_v6e_preflight.py"
    --manifest "$MANIFEST"
    --train-stage "$TRAIN_STAGE"
    --data-dir "$DATA_DIR"
    --out-dir "$OUT_DIR"
    --world-size "$TPU_WORLD_SIZE"
    --local-batch-size "$LOCAL_BATCH_SIZE"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --dispatch-pack-impl "$DISPATCH_PACK_IMPL"
    --router-override "$ROUTER_OVERRIDE"
    --attention-kernel "$ATTENTION_KERNEL"
    --optimizer "$OPTIMIZER_NAME"
    --hybrid-adamw-impl "$HYBRID_ADAMW_IMPL"
    --muon-scale-mode "$MUON_SCALE_MODE"
    --ce-logits-dtype "$CE_LOGITS_DTYPE"
    --qk-clip-threshold "$QK_CLIP_THRESHOLD"
    --expert-capacity-factor "$EXPERT_CAPACITY_FACTOR"
    --data-gcs-uri "$DATA_GCS_URI"
    --checkpoint-gcs-uri "$CHECKPOINT_GCS_URI"
  )
  if [[ "$SYNTHETIC_DATA" == "1" ]]; then
    preflight_cmd+=(--synthetic-data)
  fi
  if [[ "$SKIP_PREFLIGHT_DEVICE_CHECK" == "1" ]]; then
    preflight_cmd+=(--skip-device-check)
  fi
  if [[ "$ALLOW_WORLD_SIZE_MISMATCH" == "1" ]]; then
    preflight_cmd+=(--allow-world-size-mismatch)
  fi
  append_preflight_block_size="${METIS15_TPU_BLOCK_SIZE:-}"
  if [[ -n "$append_preflight_block_size" ]]; then
    preflight_cmd+=(--block-size "$append_preflight_block_size")
  fi
  "${preflight_cmd[@]}"
elif [[ "$RUN_PREFLIGHT" != "0" ]]; then
  echo "METIS15_TPU_PREFLIGHT must be 0 or 1; got '$RUN_PREFLIGHT'" >&2
  exit 1
fi

gcs_rsync() {
  local gcs_uri="$1"
  local local_dir="$2"
  local optional="${3:-0}"
  mkdir -p "$local_dir"
  if [[ -z "$gcs_uri" ]]; then
    [[ "$optional" == "1" ]] && return 0
    echo "Missing GCS URI for required sync into $local_dir" >&2
    return 1
  fi
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync --recursive "${gcs_uri%/}" "$local_dir"
    return 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${gcs_uri%/}" "$local_dir"
    return 0
  fi
  [[ "$optional" == "1" ]] && return 0
  echo "Install gcloud or gsutil to hydrate $gcs_uri." >&2
  return 1
}

if [[ "$SYNTHETIC_DATA" != "1" && -n "$DATA_GCS_URI" && ( ! -f "$DATA_DIR/meta.json" || ! -f "$DATA_DIR/train.bin" ) ]]; then
  echo "Hydrating $TRAIN_STAGE data from GCS: $DATA_GCS_URI"
  rm -rf "$DATA_DIR"
  gcs_rsync "$DATA_GCS_URI" "$DATA_DIR" 0
fi

if [[ "$SYNTHETIC_DATA" != "1" && "$AUTO_RESUME" != "0" && -z "$RESUME_FROM" && -n "$CHECKPOINT_GCS_URI" && ! -f "$OUT_DIR/latest.pt" ]]; then
  echo "Hydrating $TRAIN_STAGE TPU checkpoints from GCS: $CHECKPOINT_GCS_URI"
  rm -rf "$OUT_DIR"
  gcs_rsync "$CHECKPOINT_GCS_URI" "$OUT_DIR" 1 || true
fi

if [[ "$SYNTHETIC_DATA" != "1" && ( ! -f "$DATA_DIR/meta.json" || ! -f "$DATA_DIR/train.bin" ) ]]; then
  echo "$TRAIN_STAGE data is missing at $DATA_DIR/meta.json or $DATA_DIR/train.bin" >&2
  echo "Set METIS15_TPU_SYNTHETIC=1 for compile/performance smoke tests." >&2
  exit 1
fi

if [[ "$AUTO_RESUME" != "0" && -z "$RESUME_FROM" && -f "$OUT_DIR/latest.pt" ]]; then
  RESUME_FROM="$OUT_DIR/latest.pt"
fi

echo "Launching $MODEL_NAME $TRAIN_STAGE on Google Cloud TPU v6e"
echo "  manifest: $MANIFEST"
echo "  data dir: $DATA_DIR"
echo "  out dir: $OUT_DIR"
echo "  train log: $TRAIN_LOG"
echo "  world size / TPU chips: $TPU_WORLD_SIZE"
echo "  local batch size: $LOCAL_BATCH_SIZE"
echo "  grad accum steps: $GRAD_ACCUM_STEPS"
echo "  tokens per optimizer step: $TOKENS_PER_STEP"
echo "  max steps: $MAX_STEPS"
echo "  warmup steps: $WARMUP_STEPS"
echo "  expert capacity factor: $EXPERT_CAPACITY_FACTOR"
echo "  dispatch pack impl: $DISPATCH_PACK_IMPL"
echo "  single latent router input: $BALANCED_STATIC_ROUTER_INPUT"
echo "  ablations: router=$ROUTER_OVERRIDE loss=$LOSS_MODE ce_impl=$CE_IMPL ce_logits_dtype=$CE_LOGITS_DTYPE attention=$ATTENTION_MODE attention_kernel=$ATTENTION_KERNEL moe=$MOE_MODE"
echo "  optimizer: $OPTIMIZER_NAME"
echo "  optimizer master weights: $OPTIMIZER_MASTER_WEIGHTS"
echo "  qk clip: threshold=$QK_CLIP_THRESHOLD alpha=$QK_CLIP_ALPHA interval=$QK_CLIP_INTERVAL warmup_steps=$QK_CLIP_WARMUP_STEPS"
echo "  constant lr: $CONSTANT_LR"
echo "  fixed batch: $FIXED_BATCH"
echo "  grad sync mode: $GRAD_SYNC_MODE"
echo "  grad sync bucket MB: $GRAD_SYNC_BUCKET_MB"
echo "  mark step each microbatch: $MARK_STEP_EACH_MICROBATCH"
echo "  profile components: $PROFILE_COMPONENTS"
echo "  perf warmup steps: $PERF_WARMUP_STEPS"
echo "  grad clip: $GRAD_CLIP"
echo "  preinit optimizer state: $PREINIT_OPTIMIZER_STATE"
echo "  expert activation safety: $EXPERT_ACTIVATION_SAFETY"
if [[ "$SYNTHETIC_DATA" == "1" ]]; then
  echo "  synthetic data: enabled"
fi
if [[ -n "$RESUME_FROM" ]]; then
  echo "  resume from: $RESUME_FROM"
  echo "  override LR on resume: $OVERRIDE_LR_ON_RESUME"
fi

cmd=(
  "${TORCHRUN_CMD[@]}"
  --standalone
  --nnodes=1
  --nproc_per_node="$TPU_WORLD_SIZE"
  "$ROOT_DIR/scripts/train_metis15_tpu.py"
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
  --moe-mode "$MOE_MODE"
  --grad-sync-mode "$GRAD_SYNC_MODE"
  --grad-sync-bucket-mb "$GRAD_SYNC_BUCKET_MB"
  --activation-checkpointing "$ACTIVATION_CHECKPOINTING"
  --activation-checkpoint-layer-interval "$ACTIVATION_CHECKPOINT_LAYER_INTERVAL"
  --perf-warmup-steps "$PERF_WARMUP_STEPS"
  --expert-activation-safety "$EXPERT_ACTIVATION_SAFETY"
  --optimizer "$OPTIMIZER_NAME"
  --hybrid-adamw-impl "$HYBRID_ADAMW_IMPL"
  --qk-clip-threshold "$QK_CLIP_THRESHOLD"
  --qk-clip-alpha "$QK_CLIP_ALPHA"
  --qk-clip-interval "$QK_CLIP_INTERVAL"
  --qk-clip-warmup-steps "$QK_CLIP_WARMUP_STEPS"
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
  echo "METIS15_TPU_CONSTANT_LR must be 0 or 1; got '$CONSTANT_LR'" >&2
  exit 1
fi
if [[ "$FIXED_BATCH" == "1" ]]; then
  cmd+=(--fixed-batch)
elif [[ "$FIXED_BATCH" != "0" ]]; then
  echo "METIS15_TPU_FIXED_BATCH must be 0 or 1; got '$FIXED_BATCH'" >&2
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
    echo "METIS15_TPU_OVERRIDE_LR_ON_RESUME must be 0 or 1; got '$OVERRIDE_LR_ON_RESUME'" >&2
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
  echo "METIS15_TPU_TIE_EMBEDDINGS must be 0, 1, or unset; got '$TIE_EMBEDDINGS'" >&2
  exit 1
fi
append_optional_arg "${METIS15_TPU_BLOCK_SIZE:-}" "--block-size"
append_optional_arg "${METIS15_TPU_VOCAB_SIZE:-}" "--vocab-size"
append_optional_arg "${METIS15_TPU_D_MODEL:-}" "--d-model"
append_optional_arg "${METIS15_TPU_N_LAYER:-}" "--n-layer"
append_optional_arg "${METIS15_TPU_N_HEADS:-}" "--n-heads"
append_optional_arg "${METIS15_TPU_N_KV_HEADS:-}" "--n-kv-heads"
append_optional_arg "${METIS15_TPU_HEAD_DIM:-}" "--head-dim"
append_optional_arg "${METIS15_TPU_MOE_NUM_EXPERTS:-}" "--moe-num-experts"
append_optional_arg "${METIS15_TPU_MOE_TOP_K:-}" "--moe-top-k"
append_optional_arg "${METIS15_TPU_MOE_EXPERT_INTERMEDIATE_SIZE:-}" "--moe-expert-intermediate-size"
append_optional_arg "${METIS15_TPU_MOE_ROUTED_LATENT_SIZE:-}" "--moe-routed-latent-size"
append_optional_arg "${METIS15_TPU_MOE_ROUTER_LATENT_SIZE:-}" "--moe-router-latent-size"

mkdir -p "$(dirname "$TRAIN_LOG")"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"

if [[ -n "$CHECKPOINT_GCS_URI" && -d "$OUT_DIR" && "$SKIP_CHECKPOINT" != "1" ]]; then
  echo "Uploading $TRAIN_STAGE TPU checkpoints to GCS: $CHECKPOINT_GCS_URI"
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync --recursive "$OUT_DIR" "${CHECKPOINT_GCS_URI%/}"
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$OUT_DIR" "${CHECKPOINT_GCS_URI%/}"
  fi
fi
