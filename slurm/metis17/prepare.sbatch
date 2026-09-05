#!/usr/bin/env bash
set -euo pipefail

: "${METIS17_ROOT:?Missing immutable release root}"
: "${METIS17_CODE:?Missing commit-pinned code checkout}"
: "${METIS17_PYTHON:?Missing pinned data interpreter}"
: "${METIS17_WORKERS:?Missing CPU-worker count}"

export PYTHONPATH="${METIS17_CODE}/src"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec "${METIS17_PYTHON}" -B -m metis_data17.cli prep \
  --root "${METIS17_ROOT}" \
  --workers "${METIS17_WORKERS}"
