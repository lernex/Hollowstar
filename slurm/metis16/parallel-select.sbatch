#!/usr/bin/env bash
#
# Selection across the allocation instead of down one core.
#
# The four phases are separate srun steps because only one of them is ordered.
# `plan` replays the identical routing loop over text-free rows on a single
# core, because a document is selected on the basis of what every document
# before it consumed. Everything either side of it -- decompressing the
# token-count shards, re-serialising the selected rows, compressing them and
# concatenating the fragments -- is per-row work with no ordering constraint,
# so it runs on every node in the allocation.
#
# Submit with metisctl-style exports:
#   sbatch --nodes=16 --exclusive --partition=parry --time=04:00:00 \
#     --export=ALL,METIS_ROOT=...,METIS_PYTHON=...,METIS_PROFILE=... \
#     slurm/metis16/parallel-select.sbatch

set -euo pipefail

ROOT="${METIS_ROOT:?METIS_ROOT is required}"
if [[ ! -d "$ROOT/src/metis_data" ]]; then
  echo "METIS_ROOT does not look like a Metis checkout: $ROOT" >&2
  exit 2
fi
RUNTIME="${METIS_RUNTIME_DIR:-$ROOT/.metis-runtime}"
PYTHON="${METIS_PYTHON:-$RUNTIME/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Metis runtime interpreter is not executable on $(hostname): $PYTHON" >&2
  exit 2
fi
PROFILE="${METIS_PROFILE:?METIS_PROFILE is required}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

NODES="${SLURM_JOB_NUM_NODES:-1}"
EXTRACT_WORKERS="${METIS_SELECT_EXTRACT_WORKERS:-96}"
MATERIALISE_WORKERS="${METIS_SELECT_MATERIALISE_WORKERS:-64}"
CONCAT_WORKERS="${METIS_SELECT_CONCAT_WORKERS:-16}"
GROUPS="${METIS_SELECT_GROUPS:-512}"
BUFFER_GIB="${METIS_SELECT_BUFFER_GIB:-3}"

PLAN_ROOT="$("$PYTHON" - "$PROFILE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[0]).resolve()))
from metis_data.config import load_profile
profile = load_profile(Path(sys.argv[1]))
storage = profile["storage"]
print(Path(storage["lustre_root"]) / storage["directories"]["selected"] / "plan")
PY
)"
echo "plan root: $PLAN_ROOT"

run_phase() {
  local phase="$1"
  shift
  echo "=== $phase on $NODES node(s) at $(date -Is) ==="
  srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 --exclusive \
    "$PYTHON" -m metis_data.select_parallel \
    --profile "$PROFILE" --phase "$phase" --tasks "$NODES" "$@"
}

run_local() {
  local phase="$1"
  shift
  echo "=== $phase on the batch host at $(date -Is) ==="
  "$PYTHON" -m metis_data.select_parallel \
    --profile "$PROFILE" --phase "$phase" --tasks 1 --task 0 "$@"
}

# Key extraction is keyed on the shard index and skips work it already did, so
# a resubmission after a failure does not pay for it twice. The plan and the
# fragments are not resumable: a half-written plan would route a document twice
# and a fragment appended to twice would duplicate its rows, which is exactly
# the kind of failure that verifies cleanly.
run_phase extract --workers "$EXTRACT_WORKERS"

rm -rf "$PLAN_ROOT/place" "$PLAN_ROOT/place-extra" "$PLAN_ROOT/PLAN.json"
run_local plan

rm -rf "$PLAN_ROOT/frag"
run_phase materialise \
  --workers "$MATERIALISE_WORKERS" --groups "$GROUPS" --buffer-gib "$BUFFER_GIB"

rm -rf "$PLAN_ROOT/measure"
run_phase concat --workers "$CONCAT_WORKERS"

run_local seal
echo "=== selection sealed at $(date -Is) ==="
