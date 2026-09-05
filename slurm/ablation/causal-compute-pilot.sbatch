#!/bin/bash
# Isolated full-model experiments; never point this at an original campaign root.
#SBATCH --job-name=more-causal-pilot
#SBATCH --partition=parry
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --exclude=parrypeak[007,012,020,026,056,063-064]
#SBATCH --time=04:00:00

set -euo pipefail
: "${METIS_REPO:?set a committed immutable worktree}"
: "${METIS_EXPECTED_REVISION:?pin the source revision}"
: "${METIS_RELEASE_ROOT:?set the sealed release root}"
: "${METIS_PILOT_OUTPUT:?set a new qualification-study root}"
: "${METIS_PILOT_EXPERIMENT:?choose an explicit causal or fixed-reference experiment}"
: "${METIS_PILOT_STOP_AFTER_STEPS:?set an absolute resumable boundary}"
: "${METIS_ABLATION_RUNTIME:?set the pinned runtime activation script}"
export METIS_TERMINAL_CRITIC=0
export METIS_CAUSAL_MEMORY_METADATA=disabled
case "$METIS_PILOT_EXPERIMENT" in
  causal-core) export METIS_PILOT_ROW=more-core METIS_PILOT_MODE=causal ;;
  causal-rm) export METIS_PILOT_ROW=more-rm METIS_PILOT_MODE=causal METIS_CAUSAL_MEMORY_METADATA=legacy_confidence ;;
  fixed-rerouted) export METIS_PILOT_ROW=loop-fixed METIS_PILOT_MODE=causal-fixed ;;
  fixed-frozen) export METIS_PILOT_ROW=loop-pathway-frozen METIS_PILOT_MODE=causal-fixed ;;
  terminal-core) export METIS_PILOT_ROW=more-core METIS_PILOT_MODE=causal METIS_TERMINAL_CRITIC=1 ;;
  terminal-rm) export METIS_PILOT_ROW=more-rm METIS_PILOT_MODE=causal METIS_TERMINAL_CRITIC=1 METIS_CAUSAL_MEMORY_METADATA=legacy_confidence ;;
  terminal-fixed-rerouted) export METIS_PILOT_ROW=loop-fixed METIS_PILOT_MODE=causal-fixed METIS_TERMINAL_CRITIC=1 ;;
  terminal-fixed-frozen) export METIS_PILOT_ROW=loop-pathway-frozen METIS_PILOT_MODE=causal-fixed METIS_TERMINAL_CRITIC=1 ;;
  *) echo "Unsupported qualification experiment: $METIS_PILOT_EXPERIMENT" >&2; exit 2 ;;
esac
if [ "$(git -C "$METIS_REPO" rev-parse HEAD)" != "$METIS_EXPECTED_REVISION" ] ||
   [ -n "$(git -C "$METIS_REPO" status --porcelain --untracked-files=no)" ]; then
  echo "Qualification source is not the clean pinned revision" >&2
  exit 2
fi
export WORLD_SIZE="${SLURM_NTASKS:?missing task count}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${METIS_PILOT_MASTER_PORT:-29630}"
export OMP_NUM_THREADS=6
export PYTHONPATH="$METIS_REPO/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun --kill-on-bad-exit=1 --network=disable_rdzv_get bash -c '
set -euo pipefail
export RANK="$SLURM_PROCID" LOCAL_RANK="$SLURM_LOCALID"
source "$METIS_ABLATION_RUNTIME"
if [ "${METIS_PILOT_FRESH_TRITON_CACHE:-0}" = 1 ]; then
  export TRITON_CACHE_DIR="$METIS_NODE_SCRATCH/pilot-triton"
  mkdir -p "$TRITON_CACHE_DIR"
  echo "metis pilot: isolated Triton cache $TRITON_CACHE_DIR" >&2
fi
resume_args=()
if [ "${METIS_PILOT_RESUME:-0}" != 1 ]; then resume_args+=(--no-resume); fi
if [ "${METIS_TERMINAL_CRITIC:-0}" = 1 ]; then
  resume_args+=(--terminal-action-critic --terminal-critic-exploration "${METIS_TERMINAL_EXPLORATION:-0.05}")
  resume_args+=(--terminal-reference-bootstrap-steps "${METIS_TERMINAL_BOOTSTRAP_STEPS:-0}")
fi
if [ -n "${METIS_JOINT_MAX_PASSES:-}" ]; then
  resume_args+=(--joint-max-passes "$METIS_JOINT_MAX_PASSES")
fi
if [ -n "${METIS_JOINT_ROUTER_HIDDEN_DIM:-}" ]; then
  resume_args+=(--joint-router-hidden-dim "$METIS_JOINT_ROUTER_HIDDEN_DIM")
fi
exec python -B -m metis_ablation.train \
  --row "$METIS_PILOT_ROW" --output "$METIS_PILOT_OUTPUT" \
  --release-root "$METIS_RELEASE_ROOT" \
  --compute-allocation-mode "$METIS_PILOT_MODE" \
  --causal-compute-price "${METIS_CAUSAL_COMPUTE_PRICE:-0}" \
  --causal-min-passes "${METIS_CAUSAL_MIN_PASSES:-1}" \
  --causal-memory-metadata "$METIS_CAUSAL_MEMORY_METADATA" \
  --joint-router-exploration "${METIS_JOINT_EXPLORATION:-0.05}" \
  --joint-utility-coefficient "${METIS_JOINT_UTILITY_COEFFICIENT:-1}" \
  --diagnostic-apus "$WORLD_SIZE" \
  --diagnostic-micro-batch "${METIS_PILOT_MICRO_BATCH:-6}" \
  --activation-recompute-policy "${METIS_PILOT_RECOMPUTE:-none}" \
  --budget-tokens 50000000000 --schedule-total-steps 25429 \
  --learning-rate 0.00026 --seed 16062026 \
  --checkpoint-every "${METIS_PILOT_CHECKPOINT_EVERY:-500}" \
  --keep-all-checkpoints --require-routed-expert-gradients \
  --analysis-every 1000 --telemetry-every 10 \
  --stop-after-steps "$METIS_PILOT_STOP_AFTER_STEPS" \
  "${resume_args[@]}"
'
