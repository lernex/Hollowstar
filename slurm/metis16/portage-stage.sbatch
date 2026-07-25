#!/usr/bin/env bash
set -euo pipefail
umask 027

CONFIG="${1:?config path is required}"
CAMPAIGN_ROOT="${2:?campaign root is required}"
STAGE="${3:?stage is required}"
RUNTIME_SETUP="${4:?runtime setup is required}"
MAX_ATTEMPTS="${5:?maximum attempts is required}"

source "$RUNTIME_SETUP"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export MASTER_ADDR
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + (SLURM_JOB_ID % 30000)))"
export PYTHONUNBUFFERED=1
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

case "$STAGE" in
  compute_inventory|single_apu)
    STEP_ARGS=(--nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-task=1)
    ;;
  node_collectives|multinode_collectives)
    STEP_ARGS=(
      "--nodes=$SLURM_NNODES"
      "--ntasks=$SLURM_NTASKS"
      --ntasks-per-node=4
      --gpus-per-task=1
    )
    ;;
  release_verification)
    STEP_ARGS=(
      "--nodes=$SLURM_NNODES"
      "--ntasks=$SLURM_NTASKS"
      --ntasks-per-node=4
    )
    ;;
  *)
    echo "FAIL: unknown Portage stage $STAGE" >&2
    exit 2
    ;;
esac

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
  echo "Metis Portage stage=$STAGE attempt=$attempt/$MAX_ATTEMPTS job=$SLURM_JOB_ID"
  set +e
  srun \
    "${STEP_ARGS[@]}" \
    --exact \
    --kill-on-bad-exit=1 \
    --distribution=block:block \
    --cpu-bind=cores \
    python3 -m metis_portage.worker \
      --config "$CONFIG" \
      --campaign-root "$CAMPAIGN_ROOT" \
      --stage "$STAGE"
  status=$?
  set -e
  if (( status == 0 )); then
    exit 0
  fi
  if (( attempt == MAX_ATTEMPTS )); then
    echo "FAIL: $STAGE exhausted $MAX_ATTEMPTS measured attempts (last status=$status)" >&2
    exit "$status"
  fi
  # A bounded retry handles transient step-creation/RCCL initialization faults.
  # Deterministic correctness or performance gate failures simply fail again
  # and leave their reports intact; they are never converted to success.
  sleep "$((attempt * 15))"
  attempt=$((attempt + 1))
done
