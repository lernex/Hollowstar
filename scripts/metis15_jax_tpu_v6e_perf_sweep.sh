#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
LOG_ROOT="${METIS15_JAX_SWEEP_LOG_ROOT:-$ROOT_DIR/tmp/metis15_jax_tpu_v6e_perf_sweep}"
MAX_STEPS="${METIS15_JAX_SWEEP_MAX_STEPS:-12}"
LOCAL_BATCHES="${METIS15_JAX_SWEEP_LOCAL_BATCHES:-8}"
GRAD_ACCUMS="${METIS15_JAX_SWEEP_GRAD_ACCUMS:-8}"
GRAD_ACCUM_IMPLS="${METIS15_JAX_SWEEP_GRAD_ACCUM_IMPLS:-loop}"
BLOCK_SIZES="${METIS15_JAX_SWEEP_BLOCK_SIZES:-manifest}"
CAPACITY_FACTORS="${METIS15_JAX_SWEEP_CAPACITY_FACTORS:-manifest}"
OPTIMIZERS="${METIS15_JAX_SWEEP_OPTIMIZERS:-adamuon}"
ADAMUON_MATRIX_POLICIES="${METIS15_JAX_SWEEP_ADAMUON_MATRIX_POLICIES:-all}"
MUON_NS_STEPS_VALUES="${METIS15_JAX_SWEEP_MUON_NS_STEPS:-manifest}"
REMAT_MODES="${METIS15_JAX_SWEEP_REMAT_MODES:-manifest}"
DTYPES="${METIS15_JAX_SWEEP_DTYPES:-manifest}"
WEIGHT_DTYPES="${METIS15_JAX_SWEEP_WEIGHT_DTYPES:-manifest}"
CE_LOGITS_DTYPES="${METIS15_JAX_SWEEP_CE_LOGITS_DTYPES:-manifest}"
CE_LOSS_IMPLS="${METIS15_JAX_SWEEP_CE_LOSS_IMPLS:-standard}"
EXPERT_EXECUTIONS="${METIS15_JAX_SWEEP_EXPERT_EXECUTIONS:-data_parallel}"
BATCH_SHARDINGS="${METIS15_JAX_SWEEP_BATCH_SHARDINGS:-replicated}"
KEEP_GOING="${METIS15_JAX_SWEEP_KEEP_GOING:-1}"
REQUIRE_TPU="${METIS15_JAX_SWEEP_REQUIRE_TPU:-1}"
REQUIRE_LOSS_DECREASE="${METIS15_JAX_SWEEP_REQUIRE_LOSS_DECREASE:-0}"
MIN_LOSS_DROP_FRAC="${METIS15_JAX_SWEEP_MIN_LOSS_DROP_FRAC:-0.0}"
TINY_CONFIG="${METIS15_JAX_SWEEP_TINY_CONFIG:-0}"

mkdir -p "$LOG_ROOT"

echo "Metis-1.5 JAX TPU v6e performance sweep"
echo "  log root: $LOG_ROOT"
echo "  max steps: $MAX_STEPS"
echo "  local batches: $LOCAL_BATCHES"
echo "  grad accum steps: $GRAD_ACCUMS"
echo "  grad accum impls: $GRAD_ACCUM_IMPLS"
echo "  block sizes: $BLOCK_SIZES"
echo "  capacity factors: $CAPACITY_FACTORS"
echo "  optimizers: $OPTIMIZERS"
echo "  AdaMuon matrix policies: $ADAMUON_MATRIX_POLICIES"
echo "  muon ns steps: $MUON_NS_STEPS_VALUES"
echo "  remat modes: $REMAT_MODES"
echo "  dtypes: $DTYPES"
echo "  weight dtypes: $WEIGHT_DTYPES"
echo "  CE logits dtypes: $CE_LOGITS_DTYPES"
echo "  CE loss impls: $CE_LOSS_IMPLS"
echo "  expert executions: $EXPERT_EXECUTIONS"
echo "  batch shardings: $BATCH_SHARDINGS"
echo "  require TPU: $REQUIRE_TPU"
echo "  tiny config: $TINY_CONFIG"
echo "  compile-time warning: first-step tok/s is not steady-state throughput"

for local_batch in $LOCAL_BATCHES; do
  for grad_accum in $GRAD_ACCUMS; do
    for grad_accum_impl in $GRAD_ACCUM_IMPLS; do
      for block_size in $BLOCK_SIZES; do
        for capacity_factor in $CAPACITY_FACTORS; do
          for optimizer in $OPTIMIZERS; do
            for adamuon_matrix_policy in $ADAMUON_MATRIX_POLICIES; do
              for muon_ns_steps in $MUON_NS_STEPS_VALUES; do
              for remat_mode in $REMAT_MODES; do
                for dtype in $DTYPES; do
                  for weight_dtype in $WEIGHT_DTYPES; do
                    for ce_logits_dtype in $CE_LOGITS_DTYPES; do
                      for ce_loss_impl in $CE_LOSS_IMPLS; do
                        for expert_execution in $EXPERT_EXECUTIONS; do
                          for batch_sharding in $BATCH_SHARDINGS; do
                            run_id="bs${local_batch}_ga${grad_accum}_gai${grad_accum_impl}_blk${block_size}_cf${capacity_factor}_opt${optimizer}_ap${adamuon_matrix_policy}_ns${muon_ns_steps}_remat${remat_mode}_dt${dtype}_wdt${weight_dtype}_cedt${ce_logits_dtype}_ce${ce_loss_impl}_${expert_execution}_bsh${batch_sharding}"
                  out_dir="$LOG_ROOT/$run_id/checkpoints"
                  log_file="$LOG_ROOT/$run_id/train.log"
                  mkdir -p "$(dirname "$log_file")" "$out_dir"
                  echo
                  echo "=== JAX sweep $run_id ==="
                  run_failed=0
                  cmd=(
                    "$PYTHON_BIN"
                    "$ROOT_DIR/scripts/train_metis15_jax_tpu.py"
                    --manifest "$MANIFEST"
                    --stage pretrain
                    --max-steps "$MAX_STEPS"
                    --local-batch-size "$local_batch"
                    --grad-accum-steps "$grad_accum"
                    --synthetic-data
                    --checkpoint-interval 0
                    --skip-checkpoint
                    --expert-execution "$expert_execution"
                    --batch-sharding "$batch_sharding"
                    --optimizer "$optimizer"
                    --adamuon-matrix-policy "$adamuon_matrix_policy"
                    --out-dir "$out_dir"
                  )
                  if [[ "$grad_accum_impl" != "manifest" ]]; then
                    cmd+=(--grad-accum-impl "$grad_accum_impl")
                  fi
                  if [[ "$block_size" != "manifest" ]]; then
                    cmd+=(--block-size "$block_size")
                  fi
                  if [[ "$capacity_factor" != "manifest" ]]; then
                    cmd+=(--expert-capacity-factor "$capacity_factor")
                  fi
                  if [[ "$muon_ns_steps" != "manifest" ]]; then
                    cmd+=(--muon-ns-steps "$muon_ns_steps")
                  fi
                  if [[ "$remat_mode" == "on" ]]; then
                    cmd+=(--remat-layers)
                  elif [[ "$remat_mode" == "off" ]]; then
                    cmd+=(--no-remat-layers)
                  elif [[ "$remat_mode" != "manifest" ]]; then
                    echo "Unsupported remat mode: $remat_mode (use manifest, on, off)" >&2
                    exit 1
                  fi
                  if [[ "$dtype" != "manifest" ]]; then
                    cmd+=(--dtype "$dtype")
                  fi
                  if [[ "$weight_dtype" != "manifest" ]]; then
                    cmd+=(--weight-dtype "$weight_dtype")
                  fi
                  if [[ "$ce_logits_dtype" != "manifest" ]]; then
                    cmd+=(--ce-logits-dtype "$ce_logits_dtype")
                  fi
                  if [[ "$ce_loss_impl" != "manifest" ]]; then
                    cmd+=(--ce-loss-impl "$ce_loss_impl")
                  fi
                  if [[ "$TINY_CONFIG" == "1" ]]; then
                    cmd+=(--tiny-config)
                  fi
                  if ! "${cmd[@]}" 2>&1 | tee "$log_file"; then
                    run_failed=1
                  fi
                  audit=(
                    "$PYTHON_BIN"
                    "$ROOT_DIR/scripts/analyze_metis15_jax_tpu_logs.py"
                    "$log_file"
                    --min-logged-steps 2
                    --min-valid-assign-frac "${METIS15_JAX_SWEEP_MIN_VALID_ASSIGN_FRAC:-0.95}"
                    --max-expert-drop-frac "${METIS15_JAX_SWEEP_MAX_EXPERT_DROP_FRAC:-0.10}"
                    --max-qk-logit "${METIS15_JAX_SWEEP_MAX_QK_LOGIT:-1000}"
                    --perf-warmup-steps "${METIS15_JAX_PERF_WARMUP_STEPS:-3}"
                  )
                  if [[ "$REQUIRE_TPU" == "1" ]]; then
                    audit+=(--require-tpu)
                  fi
                  if [[ "$REQUIRE_LOSS_DECREASE" == "1" ]]; then
                    audit+=(--require-loss-decrease --min-loss-drop-frac "$MIN_LOSS_DROP_FRAC")
                  fi
                  if [[ "$run_failed" == "0" ]] && ! "${audit[@]}"; then
                    run_failed=1
                  fi
                  if [[ "$run_failed" != "0" ]]; then
                    echo "JAX sweep candidate failed: $run_id" >&2
                    if [[ "$KEEP_GOING" == "1" ]]; then
                      continue
                    fi
                        exit 1
                      fi
                    done
                  done
                done
              done
            done
          done
          done
          done
        done
      done
    done
  done
done
done
done

summary=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/summarize_metis15_jax_tpu_sweep.py"
  "$LOG_ROOT"
  --min-logged-steps 2
  --perf-warmup-steps "${METIS15_JAX_PERF_WARMUP_STEPS:-3}"
  --min-valid-assign-frac "${METIS15_JAX_SWEEP_MIN_VALID_ASSIGN_FRAC:-0.95}"
  --max-expert-drop-frac "${METIS15_JAX_SWEEP_MAX_EXPERT_DROP_FRAC:-0.10}"
  --max-qk-logit "${METIS15_JAX_SWEEP_MAX_QK_LOGIT:-1000}"
  --write-best-env "$LOG_ROOT/best.env"
)
if [[ "$REQUIRE_TPU" == "1" ]]; then
  summary+=(--require-tpu)
fi
if [[ "$REQUIRE_LOSS_DECREASE" == "1" ]]; then
  summary+=(--require-loss-decrease --min-loss-drop-frac "$MIN_LOSS_DROP_FRAC")
fi
"${summary[@]}"

echo
echo "JAX sweep logs written under $LOG_ROOT"
echo "Promote a setting only after post-warmup tok/s improves without worse p95 step time, assignment drop, qk clip, or fixed-batch loss behavior."
echo "Best safe candidate env, if any: $LOG_ROOT/best.env"
