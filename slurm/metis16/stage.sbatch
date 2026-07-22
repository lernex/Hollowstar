#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${METIS_RUNTIME_DIR:-$ROOT/.metis-runtime}"
PYTHON="${METIS_PYTHON:-$RUNTIME/bin/python}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
exec "$PYTHON" -m metis_data.stage_runner \
  --profile "${METIS_PROFILE:?METIS_PROFILE is required}" \
  --stage "${METIS_STAGE:?METIS_STAGE is required}" \
  --task-index "$TASK_INDEX"

