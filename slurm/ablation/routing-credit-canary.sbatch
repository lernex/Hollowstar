#!/bin/bash
# A full-geometry numerical pilot, not an 80-APU throughput comparison.
#SBATCH --job-name=more-credit-canary
#SBATCH --partition=parry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --time=00:20:00

set -euo pipefail
: "${METIS_REPO:?use an isolated committed worktree}"
: "${METIS_RELEASE_ROOT:?set the sealed release root}"
: "${METIS_CANARY_OUTPUT:?set a fresh diagnostic output root}"
: "${METIS_ABLATION_RUNTIME:?set the existing runtime activation script}"
export WORLD_SIZE=4
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29567
export OMP_NUM_THREADS=6
export PYTHONPATH="$METIS_REPO/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun --kill-on-bad-exit=1 bash -c '
set -euo pipefail
export RANK="$SLURM_PROCID" LOCAL_RANK="$SLURM_LOCALID"
source "$METIS_ABLATION_RUNTIME"
exec python -B -m metis_ablation.train \
  --row more-core --output "$METIS_CANARY_OUTPUT" \
  --release-root "$METIS_RELEASE_ROOT" \
  --diagnostic-apus 4 --max-steps "${METIS_CANARY_STEPS:-3}" \
  --budget-tokens 50000000000 --schedule-total-steps 25429 \
  --learning-rate 0.00026 --seed 16062026 \
  --compute-allocation-mode joint \
  --joint-max-passes "${METIS_JOINT_MAX_PASSES:-5}" \
  --checkpoint-every 0 --analysis-every 0 --telemetry-every 1 \
  --no-final-checkpoint --no-resume
'
