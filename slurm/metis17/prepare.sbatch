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

maintenance_args=()
case "${METIS17_DEFER_COMPACTION:-0}" in
  0) ;;
  1) maintenance_args+=(--defer-compaction) ;;
  *) printf 'Invalid METIS17_DEFER_COMPACTION\n' >&2; exit 2 ;;
esac

case "${METIS17_SCREENING_ONLY:-0}" in
  0) ;;
  1) maintenance_args+=(--screening-only) ;;
  *) printf 'Invalid METIS17_SCREENING_ONLY\n' >&2; exit 2 ;;
esac

case "${METIS17_RAW_READERS:-0}" in
  0) ;;
  *[!0-9]*|'') printf 'Invalid METIS17_RAW_READERS\n' >&2; exit 2 ;;
  *) maintenance_args+=(--raw-readers "${METIS17_RAW_READERS}") ;;
esac

exec "${METIS17_PYTHON}" -B -m metis_data17.cli prep \
  --root "${METIS17_ROOT}" \
  --workers "${METIS17_WORKERS}" \
  "${maintenance_args[@]}"
