#!/bin/bash
# Short, full-geometry isolation controls; the original queued rows stay intact.
#SBATCH --job-name=more-repair-control
#SBATCH --partition=parry
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --exclude=parrypeak[020,026,063-064]
#SBATCH --time=02:00:00

set -euo pipefail
: "${METIS_REPO:?set an immutable baseline worktree}"
: "${METIS_EXPECTED_REVISION:?pin the training source revision}"
: "${METIS_RELEASE_ROOT:?set the sealed release root}"
: "${METIS_CONTROL_OUTPUT:?set a fresh isolation-study output root}"
: "${METIS_CONTROL_ROW:?set loop-fixed or fixed-depth-adaptive-k}"
: "${METIS_ABLATION_RUNTIME:?set the pinned runtime activation script}"
case "$METIS_CONTROL_ROW" in
  loop-fixed|fixed-depth-adaptive-k) ;;
  *) echo "Unsupported isolation control: $METIS_CONTROL_ROW" >&2; exit 2 ;;
esac
if [ "$(git -C "$METIS_REPO" rev-parse HEAD)" != "$METIS_EXPECTED_REVISION" ]; then
  echo "Baseline source moved; refusing an unversioned comparison" >&2
  exit 2
fi
export WORLD_SIZE=40
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29610
export OMP_NUM_THREADS=6
export PYTHONPATH="$METIS_REPO/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun --kill-on-bad-exit=1 --network=disable_rdzv_get bash -c '
set -euo pipefail
export RANK="$SLURM_PROCID" LOCAL_RANK="$SLURM_LOCALID"
source "$METIS_ABLATION_RUNTIME"
exec python -B -m metis_ablation.train \
  --row "$METIS_CONTROL_ROW" --output "$METIS_CONTROL_OUTPUT" \
  --release-root "$METIS_RELEASE_ROOT" \
  --budget-tokens 50000000000 --schedule-total-steps 25429 \
  --max-steps "${METIS_CONTROL_STEPS:-1000}" \
  --learning-rate 0.00026 --seed 16062026 \
  --checkpoint-every 250 --analysis-every 1000 --telemetry-every 10 \
  --no-resume
'
