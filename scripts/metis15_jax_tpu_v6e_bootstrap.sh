#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
TPU_VENV="${METIS15_JAX_TPU_VENV:-$ROOT_DIR/.venv-jax-tpu}"
REQUIREMENTS="${METIS15_JAX_TPU_REQUIREMENTS:-$ROOT_DIR/requirements-jax-tpu-train.txt}"
TRAIN_STAGE="${METIS15_TRAIN_STAGE:-pretrain}"

if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "Missing JAX TPU requirements file: $REQUIREMENTS" >&2
  exit 1
fi

if [[ "${METIS15_JAX_SKIP_VENV:-0}" != "1" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$TPU_VENV"
  # shellcheck disable=SC1091
  source "$TPU_VENV/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQUIREMENTS"

export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/scripts:${PYTHONPATH:-}"

if [[ "${METIS15_JAX_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  preflight_cmd=(
    "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_preflight.py"
    --manifest "${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
    --stage "$TRAIN_STAGE"
  )
  if [[ -n "${METIS15_JAX_LOCAL_BATCH_SIZE:-}" ]]; then
    preflight_cmd+=(--local-batch-size "$METIS15_JAX_LOCAL_BATCH_SIZE")
  fi
  if [[ "${METIS15_JAX_SKIP_DEVICE_CHECK:-0}" == "1" ]]; then
    preflight_cmd+=(--skip-device-check)
  fi
  "${preflight_cmd[@]}"
fi

echo
echo "JAX TPU bootstrap complete."
echo "Run: scripts/metis15_jax_tpu_v6e_pretrain.sh"
