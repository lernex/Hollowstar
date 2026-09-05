#!/bin/bash
# Three permanent token windows; assessment never writes into the training run.
#SBATCH --job-name=more-pilot-quality
#SBATCH --partition=parry
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --exclude=parrypeak[007,012,020,026,056,063-064]
#SBATCH --time=00:30:00

set -euo pipefail
: "${METIS_REPO:?set the committed evaluator worktree}"
: "${METIS_EXPECTED_REVISION:?pin the evaluator revision}"
: "${METIS_RELEASE_ROOT:?set the sealed release root}"
: "${METIS_QUALITY_CHECKPOINT:?set the checkpoint state.pt file}"
: "${METIS_QUALITY_MANIFEST:?set the recorded run.json file}"
: "${METIS_QUALITY_OUTPUT:?set a fresh assessment output root}"
: "${METIS_ABLATION_RUNTIME:?set the pinned runtime activation script}"
if [ "${SLURM_NTASKS:-0}" != 3 ] || [ "${SLURM_JOB_NUM_NODES:-0}" != 1 ]; then
  echo "All three permanent windows require exactly three ranks on one node" >&2
  exit 2
fi
if [ "$(git -C "$METIS_REPO" rev-parse HEAD)" != "$METIS_EXPECTED_REVISION" ] ||
   [ -n "$(git -C "$METIS_REPO" status --porcelain --untracked-files=no)" ]; then
  echo "Evaluator source is not the clean pinned revision" >&2
  exit 2
fi
test -f "$METIS_QUALITY_CHECKPOINT"
test -f "$METIS_QUALITY_MANIFEST"
export PYTHONPATH="$METIS_REPO/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6
srun --kill-on-bad-exit=1 bash -c '
set -euo pipefail
export LOCAL_RANK="$SLURM_LOCALID"
export ROCR_VISIBLE_DEVICES="$LOCAL_RANK" HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
source "$METIS_ABLATION_RUNTIME"
case "$LOCAL_RANK" in
  0) step=2000 ;;
  1) step=19000 ;;
  2) step=25000 ;;
  *) echo "The assessment allocation must have exactly three ranks" >&2; exit 2 ;;
esac
exec python -B -m metis_ablation.routing_credit_probe \
  --checkpoint "$METIS_QUALITY_CHECKPOINT" \
  --run-manifest "$METIS_QUALITY_MANIFEST" \
  --release-root "$METIS_RELEASE_ROOT" \
  --output "$METIS_QUALITY_OUTPUT/gap8-step-$step" \
  --quality-only --evaluation-gap-blocks 8 --step "$step" \
  --sequences 6 --sequence-length 4096 --repeat-forwards 5 \
  --precision bf16 --device cuda:0 --seed 16062026
'
