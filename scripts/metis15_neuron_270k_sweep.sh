#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${METIS15_NEURON_VENV:-/mnt/trn1/venvs/aws_neuron_venv_pytorch}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
TORCHRUN="${TORCHRUN:-$VENV/bin/torchrun}"
OUT_ROOT="${METIS15_SWEEP_OUT_ROOT:-/mnt/trn1/runs/metis15_270k_sweep_$(date +%Y%m%d_%H%M%S)}"

export PATH="$VENV/bin:/opt/aws/neuron/bin:${PATH}"
export LD_LIBRARY_PATH="$VENV/lib64/python3.12/site-packages/libneuronxla:$VENV/lib/python3.12/site-packages/libneuronxla:${LD_LIBRARY_PATH:-}"
export PJRT_DEVICE="${PJRT_DEVICE:-NEURON}"
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-32}"
export NEURON_CC_CACHE="${NEURON_CC_CACHE:-/mnt/trn1/neuron_cc_cache}"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS:---cache_dir=$NEURON_CC_CACHE --auto-cast=none}"

WORLD_SIZE="${METIS15_NEURON_WORLD_SIZE:-32}"
BATCH_SIZE="${METIS15_SWEEP_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${METIS15_SWEEP_GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${METIS15_SWEEP_MAX_STEPS:-112}"
PERF_WARMUP="${METIS15_SWEEP_PERF_WARMUP:-8}"
LOG_INTERVAL="${METIS15_SWEEP_LOG_INTERVAL:-32}"
BUCKETS="${METIS15_SWEEP_BUCKETS:-192 256 384 128}"
ORACLE_MODES="${METIS15_SWEEP_ORACLE_MODES:-expert_only none}"
EXTRA_ARGS="${METIS15_SWEEP_EXTRA_ARGS:-}"
BALANCED_STATIC_LAYOUT="${METIS15_SWEEP_BALANCED_STATIC_LAYOUT:-indexed}"
CE_LOGITS_DTYPE="${METIS15_SWEEP_CE_LOGITS_DTYPE:-float32}"
CE_IMPL="${METIS15_SWEEP_CE_IMPL:-cross_entropy}"

common_args=(
  "$ROOT_DIR/scripts/train_metis15_neuron.py"
  --manifest "$ROOT_DIR/configs/metis15_manifest.json"
  --synthetic-data
  --device xla
  --batch-size "$BATCH_SIZE"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --block-size 1024
  --n-layer 19
  --dispatch-pack-impl balanced_static
  --balanced-static-layout "$BALANCED_STATIC_LAYOUT"
  --router-override force_balanced
  --expert-capacity-factor 1.0
  --max-steps "$MAX_STEPS"
  --warmup-steps 1
  --perf-warmup-steps "$PERF_WARMUP"
  --constant-lr
  --skip-checkpoint
  --loss-mode real_ce
  --ce-impl "$CE_IMPL"
  --ce-logits-dtype "$CE_LOGITS_DTYPE"
  --attention-mode real
  --attention-kernel nki_flash_1k
  --nki-flash-lse-dtype bfloat16
  --moe-mode real
  --grad-clip 0
  --expert-activation-safety none
  --preinit-optimizer-state
  --profile-components
  --log-interval "$LOG_INTERVAL"
)

run_case() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  mkdir -p "$out_dir"
  echo "=== metis15 270k sweep: $name ==="
  echo "out_dir=$out_dir"
  "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$WORLD_SIZE" \
    "${common_args[@]}" \
    --out-dir "$out_dir" \
    ${EXTRA_ARGS:+$EXTRA_ARGS} \
    "$@" 2>&1 | tee "$out_dir/train.log"
}

mkdir -p "$OUT_ROOT"
"$PYTHON_BIN" -m py_compile "$ROOT_DIR/scripts/train_metis15_neuron.py"

for bucket in $BUCKETS; do
  run_case "all_reduce_bucket${bucket}" --grad-sync-mode all_reduce --grad-sync-bucket-mb "$bucket"
done

for mode in $ORACLE_MODES; do
  run_case "$mode" --grad-sync-mode "$mode" --grad-sync-bucket-mb 0
done

echo "Sweep complete: $OUT_ROOT"
