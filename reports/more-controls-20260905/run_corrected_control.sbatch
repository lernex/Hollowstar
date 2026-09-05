#!/bin/bash
# One allocation preserves the full experiment identity across the expert gate.
#SBATCH --partition=parry
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --time=36:00:00

set -euo pipefail
ulimit -c 0

: "${METIS_REPO:?set an immutable cache-fixed training checkout}"
: "${METIS_CONTROL_REVISION:?set the full training commit SHA}"
: "${METIS_CONTROL_ROW:?set the corrected control row}"
: "${METIS_CONTROL_OUTPUT:?set the separate corrected campaign output root}"
: "${METIS_CONTROL_GATE:?set the archived checkpoint gate script}"
: "${METIS_RELEASE_ROOT:?set the original sealed release root}"
: "${METIS_ABLATION_RUNTIME:?set the pinned runtime activation script}"

case "$METIS_CONTROL_ROW" in
  moe-k4|moe-k8) export WORLD_SIZE=20 ;;
  random-k|random-depth|mor-fixed-k) export WORLD_SIZE=80 ;;
  *) echo "Refusing a row outside the approved corrected controls" >&2; exit 2 ;;
esac
if [ "$((SLURM_JOB_NUM_NODES * 4))" -ne "$WORLD_SIZE" ]; then
  echo "Allocated geometry does not match the sealed control row" >&2
  exit 2
fi
if [ "$(git -C "$METIS_REPO" rev-parse HEAD)" != "$METIS_CONTROL_REVISION" ]; then
  echo "Training checkout is not at the approved revision" >&2
  exit 2
fi
git -C "$METIS_REPO" diff --quiet
git -C "$METIS_REPO" diff --cached --quiet

export PYTHONPATH="$METIS_REPO/src"
export OMP_NUM_THREADS=6
export NCCL_ALGO=Ring
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=29605
run_root="$METIS_CONTROL_OUTPUT/$METIS_CONTROL_ROW"
telemetry="$run_root/telemetry/rank-00000.jsonl"
mkdir -p "$run_root/operational"

watch_progress() {
  local pid="$1" last_size=-1 current_size now
  local last_progress
  last_progress="$(date +%s)"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    current_size=0
    if [ -f "$telemetry" ]; then
      current_size="$(wc -c < "$telemetry")"
    fi
    now="$(date +%s)"
    if [ "$current_size" -ne "$last_size" ]; then
      last_size="$current_size"
      last_progress="$now"
    elif [ "$((now - last_progress))" -ge 1200 ]; then
      echo "No rank-zero progress for 1200s; failing the corrected control" >&2
      if ! timeout --kill-after=2s 15s python3 \
        "$METIS_REPO/src/metis_ablation/runtime_diagnostics.py" \
        --run-root "$run_root" --job-id "$SLURM_JOB_ID" --scheduler-timeout 3 >&2; then
        echo "Failure snapshot unavailable; preserving the stalled-step failure" >&2
      fi
      kill -TERM "$pid" 2>/dev/null || true
      sleep 60
      kill -KILL "$pid" 2>/dev/null || true
      return 124
    fi
  done
}

run_training() {
  local step_pid watchdog_pid status
  srun --kill-on-bad-exit=1 --network=disable_rdzv_get bash -c '
    set -euo pipefail
    export RANK="$SLURM_PROCID"
    export LOCAL_RANK="$SLURM_LOCALID"
    source "$METIS_ABLATION_RUNTIME"
    exec python -m metis_ablation.train \
      --row "$METIS_CONTROL_ROW" \
      --output "$METIS_CONTROL_OUTPUT" \
      --release-root "$METIS_RELEASE_ROOT" \
      --budget-tokens 50000000000 \
      --schedule-total-steps 25429 \
      --seed 16062026 \
      --learning-rate 0.00026 \
      --compute-allocation-mode legacy \
      --activation-recompute-policy none \
      --require-routed-expert-gradients \
      --analysis-every 1000 \
      --telemetry-every 10 \
      "$@"
  ' corrected-training "$@" &
  step_pid=$!
  watch_progress "$step_pid" &
  watchdog_pid=$!
  if wait "$step_pid"; then
    status=0
  else
    status=$?
  fi
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true
  return "$status"
}

run_gate() {
  timeout --kill-after=30s 900s srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --cpus-per-task=4 --kill-on-bad-exit=1 --network=disable_rdzv_get bash -c '
      set -euo pipefail
      export RANK="$SLURM_PROCID"
      export LOCAL_RANK="$SLURM_LOCALID"
      source "$METIS_ABLATION_RUNTIME"
      export OMP_NUM_THREADS=2
      exec python "$METIS_CONTROL_GATE" \
        --run-root "$METIS_CONTROL_OUTPUT/$METIS_CONTROL_ROW" \
        --row "$METIS_CONTROL_ROW" \
        --source-revision "$METIS_CONTROL_REVISION" \
        "$@"
    ' corrected-expert-gate "$@"
}

echo "Corrected control: row=$METIS_CONTROL_ROW source=$METIS_CONTROL_REVISION job=$SLURM_JOB_ID"
if [ -f "$run_root/operational/expert-training-gate.json" ]; then
  run_gate --verify-existing
else
  if [ ! -f "$run_root/checkpoints/step-0000100/state.pt" ] ||
     [ ! -f "$run_root/summary.json" ]; then
    run_training --stop-after-steps 100 --checkpoint-every 100 --no-resume
  fi
  run_gate
fi
echo "Expert gate passed; resuming the identical full 50B-token experiment"
run_training --checkpoint-every 500
