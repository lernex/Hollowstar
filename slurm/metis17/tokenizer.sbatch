#!/usr/bin/env bash
set -euo pipefail

: "${METIS17_ROOT:?Missing immutable release root}"
: "${METIS17_CODE:?Missing commit-pinned code checkout}"
: "${METIS17_PYTHON:?Missing pinned data interpreter}"
: "${METIS17_WORKERS:?Missing tokenizer worker count}"

export PYTHONPATH="${METIS17_CODE}/src"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${METIS17_WORKERS}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${SLURM_JOB_ID:?Tokenizer must run inside its allocated compute job}"
scratch="${TMPDIR:-/tmp}/metis17-tokenizer-${UID}-${SLURM_JOB_ID}"
umask 077

exec "${METIS17_PYTHON}" -B -m metis_data17.cli tokenizer \
  --root "${METIS17_ROOT}" \
  --scratch-dir "${scratch}"
