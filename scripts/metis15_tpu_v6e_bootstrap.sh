#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
TPU_VENV="${METIS15_TPU_VENV:-$ROOT_DIR/.venv-tpu}"
REQUIREMENTS="${METIS15_TPU_REQUIREMENTS:-$ROOT_DIR/requirements-tpu-train.txt}"
TRAIN_STAGE="${METIS15_TRAIN_STAGE:-pretrain}"
GCS_ROOT="${METIS15_GCS_ROOT:-}"
if [[ "$TRAIN_STAGE" == "continued_pretrain" ]]; then
  DEFAULT_DATA_DIR="$ROOT_DIR/data/metis15_continued_pretrain"
  DEFAULT_OUT_DIR="$ROOT_DIR/checkpoints/metis15_continued_pretrain_tpu_v6e"
  DEFAULT_DATA_GCS_URI="${GCS_ROOT:+$GCS_ROOT/pretrain-shards/continued}"
  DEFAULT_CHECKPOINT_GCS_URI="${GCS_ROOT:+$GCS_ROOT/checkpoints/continued-tpu-v6e}"
else
  DEFAULT_DATA_DIR="$ROOT_DIR/data/metis15_base"
  DEFAULT_OUT_DIR="$ROOT_DIR/checkpoints/metis15_base_tpu_v6e"
  DEFAULT_DATA_GCS_URI="${GCS_ROOT:+$GCS_ROOT/pretrain-shards/base}"
  DEFAULT_CHECKPOINT_GCS_URI="${GCS_ROOT:+$GCS_ROOT/checkpoints/base-tpu-v6e}"
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "Missing TPU requirements file: $REQUIREMENTS" >&2
  exit 1
fi

if [[ "${METIS15_TPU_SKIP_VENV:-0}" != "1" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$TPU_VENV"
  # shellcheck disable=SC1091
  source "$TPU_VENV/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS"

export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/scripts:${PYTHONPATH:-}"

if [[ "${METIS15_TPU_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  preflight_cmd=(
    "$ROOT_DIR/scripts/metis15_tpu_v6e_preflight.py"
    --manifest "${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
    --train-stage "$TRAIN_STAGE"
    --data-dir "${METIS15_DATA_DIR:-$DEFAULT_DATA_DIR}"
    --out-dir "${METIS15_OUT_DIR:-$DEFAULT_OUT_DIR}"
    --world-size "${METIS15_TPU_WORLD_SIZE:-8}"
    --local-batch-size "${METIS15_TPU_LOCAL_BATCH_SIZE:-1}"
    --grad-accum-steps "${METIS15_TPU_GRAD_ACCUM_STEPS:-16}"
    --data-gcs-uri "${METIS15_GCS_PRETRAIN_URI:-$DEFAULT_DATA_GCS_URI}"
    --checkpoint-gcs-uri "${METIS15_GCS_CHECKPOINTS_URI:-$DEFAULT_CHECKPOINT_GCS_URI}"
  )
  if [[ "${METIS15_TPU_SYNTHETIC:-0}" == "1" ]]; then
    preflight_cmd+=(--synthetic-data)
  fi
  if [[ "${METIS15_TPU_PREFLIGHT_SKIP_DEVICE_CHECK:-0}" == "1" ]]; then
    preflight_cmd+=(--skip-device-check)
  fi
  if [[ "${METIS15_TPU_ALLOW_WORLD_SIZE_MISMATCH:-0}" == "1" ]]; then
    preflight_cmd+=(--allow-world-size-mismatch)
  fi
  "${preflight_cmd[@]}"
fi

echo
echo "TPU bootstrap complete."
echo "Run: scripts/metis15_tpu_v6e_pretrain.sh"
