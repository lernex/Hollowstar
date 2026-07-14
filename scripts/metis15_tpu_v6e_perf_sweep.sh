#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${METIS15_TPU_SWEEP_LOG_ROOT:-$ROOT_DIR/tmp/metis15_tpu_v6e_perf_sweep}"
MAX_STEPS="${METIS15_TPU_SWEEP_MAX_STEPS:-12}"
LOG_INTERVAL="${METIS15_TPU_SWEEP_LOG_INTERVAL:-1}"
LOCAL_BATCHES="${METIS15_TPU_SWEEP_LOCAL_BATCHES:-1}"
GRAD_ACCUMS="${METIS15_TPU_SWEEP_GRAD_ACCUMS:-16}"
ATTENTION_KERNELS="${METIS15_TPU_SWEEP_ATTENTION_KERNELS:-sdpa eager_gqa}"
GRAD_BUCKETS="${METIS15_TPU_SWEEP_GRAD_BUCKET_MB:-16 32}"
KEEP_GOING="${METIS15_TPU_SWEEP_KEEP_GOING:-1}"
REQUIRE_LOSS_DECREASE="${METIS15_TPU_SWEEP_REQUIRE_LOSS_DECREASE:-0}"
MIN_LOSS_DROP_FRAC="${METIS15_TPU_SWEEP_MIN_LOSS_DROP_FRAC:-0.0}"

mkdir -p "$LOG_ROOT"

echo "Metis-1.5 TPU v6e performance sweep"
echo "  log root: $LOG_ROOT"
echo "  max steps: $MAX_STEPS"
echo "  local batches: $LOCAL_BATCHES"
echo "  grad accum steps: $GRAD_ACCUMS"
echo "  attention kernels: $ATTENTION_KERNELS"
echo "  grad bucket MB: $GRAD_BUCKETS"
echo "  keep going after failed candidate: $KEEP_GOING"
echo "  stability guardrails stay enabled: qk_clip, FP32 CE logits, AdamW-routed experts, capacity_factor=4"

for local_batch in $LOCAL_BATCHES; do
  for grad_accum in $GRAD_ACCUMS; do
    for attention_kernel in $ATTENTION_KERNELS; do
      for grad_bucket_mb in $GRAD_BUCKETS; do
        run_id="bs${local_batch}_ga${grad_accum}_${attention_kernel}_bucket${grad_bucket_mb}"
        out_dir="$LOG_ROOT/$run_id/checkpoints"
        log_file="$LOG_ROOT/$run_id/train.log"
        mkdir -p "$(dirname "$log_file")" "$out_dir"
        echo
        echo "=== sweep $run_id ==="
        run_failed=0
        if ! (
          export METIS15_TPU_SYNTHETIC=1
          export METIS15_TPU_MAX_STEPS="$MAX_STEPS"
          export METIS15_TPU_LOG_INTERVAL="$LOG_INTERVAL"
          export METIS15_TPU_SKIP_CHECKPOINT=1
          export METIS15_TPU_AUTO_RESUME=0
          export METIS15_TPU_PROFILE_COMPONENTS=1
          export METIS15_TPU_PERF_WARMUP_STEPS=3
          export METIS15_TPU_LOCAL_BATCH_SIZE="$local_batch"
          export METIS15_TPU_GRAD_ACCUM_STEPS="$grad_accum"
          export METIS15_TPU_ATTENTION_KERNEL="$attention_kernel"
          export METIS15_TPU_GRAD_SYNC_BUCKET_MB="$grad_bucket_mb"
          export METIS15_OUT_DIR="$out_dir"
          export METIS15_TPU_TRAIN_LOG="$out_dir/train.log"
          "$ROOT_DIR/scripts/metis15_tpu_v6e_pretrain.sh"
        ) 2>&1 | tee "$log_file"; then
          run_failed=1
        fi
        audit_cmd=(
          "$ROOT_DIR/scripts/analyze_metis15_tpu_logs.py"
          "$log_file"
          --min-logged-steps 2
          --require-profile
          --require-qk-clip
          --min-valid-assign-frac 0.99
          --max-qk-logit 1000
          --perf-warmup-steps 3
        )
        if [[ "$REQUIRE_LOSS_DECREASE" == "1" ]]; then
          audit_cmd+=(--require-loss-decrease --min-loss-drop-frac "$MIN_LOSS_DROP_FRAC")
        fi
        if [[ "$run_failed" == "0" ]] && ! "${audit_cmd[@]}"; then
          run_failed=1
        fi
        if [[ "$run_failed" != "0" ]]; then
          echo "Sweep candidate failed: $run_id" >&2
          if [[ "$KEEP_GOING" == "1" ]]; then
            continue
          fi
          exit 1
        fi
      done
    done
  done
done

echo
summary_cmd=(
  "$ROOT_DIR/scripts/summarize_metis15_tpu_sweep.py"
  "$LOG_ROOT"
  --min-logged-steps 2
  --perf-warmup-steps 3
  --min-valid-assign-frac 0.99
  --max-qk-logit 1000
  --write-best-env "$LOG_ROOT/best.env"
)
if [[ "$REQUIRE_LOSS_DECREASE" == "1" ]]; then
  summary_cmd+=(--require-loss-decrease --min-loss-drop-frac "$MIN_LOSS_DROP_FRAC")
fi
"${summary_cmd[@]}"
echo
echo "Sweep logs written under $LOG_ROOT"
echo "Promote a setting only if tok/s improves without worse p95 step time, valid_assign, qk_clip, or fixed-batch loss behavior."
echo "Best safe candidate env, if any: $LOG_ROOT/best.env"
