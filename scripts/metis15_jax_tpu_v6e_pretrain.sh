#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
STAGE="${METIS15_TRAIN_STAGE:-pretrain}"
OUT_DIR="${METIS15_OUT_DIR:-$ROOT_DIR/checkpoints/metis15_jax_tpu_v6e_$STAGE}"
MAX_STEPS="${METIS15_JAX_MAX_STEPS:-}"
LOCAL_BATCH_SIZE="${METIS15_JAX_LOCAL_BATCH_SIZE:-}"
GRAD_ACCUM_STEPS="${METIS15_JAX_GRAD_ACCUM_STEPS:-}"
BLOCK_SIZE="${METIS15_JAX_BLOCK_SIZE:-}"
EXPERT_CAPACITY_FACTOR="${METIS15_JAX_EXPERT_CAPACITY_FACTOR:-}"
REMAT_MODE="${METIS15_JAX_REMAT_MODE:-}"
DTYPE="${METIS15_JAX_DTYPE:-}"
SYNTHETIC="${METIS15_JAX_SYNTHETIC:-0}"
DATA_DIR="${METIS15_DATA_DIR:-}"
DATA_GCS_URI="${METIS15_JAX_DATA_GCS_URI:-${METIS15_GCS_PRETRAIN_URI:-}}"
CHECKPOINT_INTERVAL="${METIS15_JAX_CHECKPOINT_INTERVAL:-}"
CHECKPOINT_BACKEND="${METIS15_JAX_CHECKPOINT_BACKEND:-orbax}"
GCS_CHECKPOINT_DIR="${METIS15_JAX_GCS_CHECKPOINT_DIR:-}"
RESUME="${METIS15_JAX_RESUME:-1}"
EXPERT_EXECUTION="${METIS15_JAX_EXPERT_EXECUTION:-shard_map}"
SKIP_DEVICE_CHECK="${METIS15_JAX_SKIP_DEVICE_CHECK:-0}"

require_gcloud() {
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud is required for $1." >&2
    exit 1
  fi
}

rsync_from_gcs() {
  local source_uri="$1"
  local local_dir="$2"
  local label="$3"
  local optional="${4:-0}"
  if [[ -z "$source_uri" ]]; then
    return
  fi
  require_gcloud "$label GCS hydration"
  mkdir -p "$local_dir"
  echo "Hydrating $label from GCS: $source_uri -> $local_dir"
  if ! gcloud storage rsync -r "$source_uri" "$local_dir"; then
    if [[ "$optional" == "1" ]]; then
      echo "Optional $label hydration did not complete; continuing." >&2
      return
    fi
    exit 1
  fi
}

preflight=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_preflight.py"
  --manifest "$MANIFEST"
  --stage "$STAGE"
)
if [[ -n "$LOCAL_BATCH_SIZE" ]]; then
  preflight+=(--local-batch-size "$LOCAL_BATCH_SIZE")
fi
if [[ "$SKIP_DEVICE_CHECK" == "1" ]]; then
  preflight+=(--skip-device-check)
fi
"${preflight[@]}"

cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/train_metis15_jax_tpu.py"
  --manifest "$MANIFEST"
  --stage "$STAGE"
  --out-dir "$OUT_DIR"
  --checkpoint-backend "$CHECKPOINT_BACKEND"
  --expert-execution "$EXPERT_EXECUTION"
)
if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$CHECKPOINT_INTERVAL" ]]; then
  cmd+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
fi
if [[ -n "$GCS_CHECKPOINT_DIR" ]]; then
  cmd+=(--gcs-checkpoint-dir "$GCS_CHECKPOINT_DIR")
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
if [[ -n "$EXPERT_CAPACITY_FACTOR" ]]; then
  cmd+=(--expert-capacity-factor "$EXPERT_CAPACITY_FACTOR")
fi
if [[ "$REMAT_MODE" == "on" ]]; then
  cmd+=(--remat-layers)
elif [[ "$REMAT_MODE" == "off" ]]; then
  cmd+=(--no-remat-layers)
elif [[ -n "$REMAT_MODE" && "$REMAT_MODE" != "manifest" ]]; then
  echo "Unsupported METIS15_JAX_REMAT_MODE=$REMAT_MODE (use manifest, on, off)" >&2
  exit 1
fi
if [[ -n "$DTYPE" && "$DTYPE" != "manifest" ]]; then
  cmd+=(--dtype "$DTYPE")
fi
if [[ "$SYNTHETIC" == "1" ]]; then
  cmd+=(--synthetic-data)
elif [[ -n "$DATA_DIR" ]]; then
  if [[ (! -f "$DATA_DIR/meta.json" || ! -f "$DATA_DIR/train.bin") && -n "$DATA_GCS_URI" ]]; then
    rsync_from_gcs "$DATA_GCS_URI" "$DATA_DIR" "$STAGE data" 0
  fi
  if [[ ! -f "$DATA_DIR/meta.json" || ! -f "$DATA_DIR/train.bin" ]]; then
    echo "METIS15_DATA_DIR must contain meta.json and train.bin. Set METIS15_JAX_DATA_GCS_URI or METIS15_GCS_PRETRAIN_URI to hydrate it from GCS." >&2
    exit 1
  fi
  cmd+=(--data-dir "$DATA_DIR")
else
  echo "Set METIS15_DATA_DIR for real training, or METIS15_JAX_SYNTHETIC=1 for synthetic proof." >&2
  exit 1
fi
if [[ "$RESUME" == "1" ]]; then
  if [[ -n "$GCS_CHECKPOINT_DIR" && ! -d "$OUT_DIR/latest" ]]; then
    rsync_from_gcs "$GCS_CHECKPOINT_DIR" "$OUT_DIR" "$STAGE checkpoint resume" 1
  fi
  cmd+=(--resume)
fi

mkdir -p "$OUT_DIR"
"${cmd[@]}" 2>&1 | tee "$OUT_DIR/train.log"
