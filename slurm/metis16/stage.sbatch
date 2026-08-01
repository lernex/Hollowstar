#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME="${METIS_RUNTIME_DIR:-$ROOT/.metis-runtime}"
PYTHON="${METIS_PYTHON:-$RUNTIME/bin/python}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

TASK_OFFSET="${METIS_TASK_OFFSET:-0}"
TASKS_PER_JOB="${METIS_TASKS_PER_JOB:-1}"
TASK_LIMIT="${METIS_TASK_LIMIT:-0}"
for value in "$TASK_OFFSET" "$TASKS_PER_JOB" "$TASK_LIMIT"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "METIS_TASK_OFFSET, METIS_TASKS_PER_JOB, and METIS_TASK_LIMIT must be non-negative integers" >&2
    exit 2
  fi
done
if (( TASKS_PER_JOB < 1 )); then
  echo "METIS_TASKS_PER_JOB must be at least 1" >&2
  exit 2
fi

# One array entry owns TASKS_PER_JOB consecutive global indices. METIS_TASK_LIMIT
# is the exclusive upper bound of this submission's chunk, so a trailing partial
# group can never reach into the range owned by another submission.
GROUP_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
TASK_INDEX="$(( TASK_OFFSET + GROUP_INDEX * TASKS_PER_JOB ))"

if (( TASKS_PER_JOB > 1 )); then
  # Every in-node task is its own single-threaded process. Without these, each
  # worker would size a thread pool from the whole node and oversubscribe the
  # allocation by two orders of magnitude.
  export OMP_NUM_THREADS=1
  export RAYON_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export TOKENIZERS_PARALLELISM=false
else
  # A single-task stage owns the whole node, so its libraries should use it.
  # This matters most for tokenizer_train: the BPE merge loop is inherently
  # sequential, but the word-counting pass that precedes it is Rayon-parallel
  # and is gated on TOKENIZERS_PARALLELISM. Leaving that false would have run
  # the largest job in the build single-threaded on 96 cores.
  export TOKENIZERS_PARALLELISM=true
fi

exec "$PYTHON" -m metis_data.stage_runner \
  --profile "${METIS_PROFILE:?METIS_PROFILE is required}" \
  --stage "${METIS_STAGE:?METIS_STAGE is required}" \
  --task-index "$TASK_INDEX" \
  --task-count "$TASKS_PER_JOB" \
  --task-limit "$TASK_LIMIT"
