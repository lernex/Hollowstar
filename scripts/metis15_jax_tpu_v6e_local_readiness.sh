#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv-jax/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-jax/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
LOG_ROOT="${METIS15_JAX_LOCAL_READINESS_LOG_ROOT:-$ROOT_DIR/tmp/metis15_jax_tpu_v6e_local_readiness}"
MESH_XLA_FLAGS="${METIS15_JAX_LOCAL_READINESS_XLA_FLAGS:---xla_force_host_platform_device_count=8}"

mkdir -p "$LOG_ROOT"

run() {
  local label="$1"
  shift
  echo
  echo "=== $label ==="
  "$@"
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  echo
  echo "=== $label ==="
  mkdir -p "$(dirname "$log_file")"
  "$@" 2>&1 | tee "$log_file"
}

echo "Metis-1.5 JAX TPU v6e local readiness"
echo "  root: $ROOT_DIR"
echo "  python: $PYTHON_BIN"
echo "  log root: $LOG_ROOT"
echo "  mesh XLA_FLAGS: $MESH_XLA_FLAGS"

run "py_compile" python3 -m py_compile \
  "$ROOT_DIR/src/metis_mamba/jax_metis.py" \
  "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  "$ROOT_DIR/scripts/smoke_metis15_jax_contracts.py" \
  "$ROOT_DIR/scripts/smoke_metis15_jax_full_shape_contracts.py" \
  "$ROOT_DIR/scripts/smoke_metis15_jax_mesh.py" \
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_sharding_report.py" \
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_preflight.py" \
  "$ROOT_DIR/scripts/analyze_metis15_jax_tpu_logs.py" \
  "$ROOT_DIR/scripts/summarize_metis15_jax_tpu_sweep.py"

run "preflight" "$PYTHON_BIN" "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_preflight.py" --skip-device-check
run "full-shape abstract contracts" "$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_metis15_jax_full_shape_contracts.py"
run "abstract sharding report" "$PYTHON_BIN" "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_sharding_report.py" \
  --json-out "$LOG_ROOT/abstract_sharding_report.json"
run "base contract smoke" "$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_metis15_jax_contracts.py" --steps 4
run "MoR contract smoke" "$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_metis15_jax_contracts.py" --steps 4 --mor

run "8-device shard_map mesh smoke" env XLA_FLAGS="$MESH_XLA_FLAGS" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/smoke_metis15_jax_mesh.py" --steps 2
run "8-device runtime sharding report" env XLA_FLAGS="$MESH_XLA_FLAGS" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_sharding_report.py" \
  --require-runtime \
  --json-out "$LOG_ROOT/runtime_sharding_report.json"

COMPILE_PROBE_LOG="$LOG_ROOT/compile_probe"
QUALITY_CANARY_LOG="$LOG_ROOT/quality_canary"
PERF_SWEEP_LOG="$LOG_ROOT/perf_sweep"
CPT_MOR_TRAINER_LOG="$LOG_ROOT/cpt_mor_trainer"
rm -rf "$COMPILE_PROBE_LOG" "$QUALITY_CANARY_LOG" "$PERF_SWEEP_LOG" "$CPT_MOR_TRAINER_LOG"

run "tiny compile probe" env \
  PYTHON_BIN="$PYTHON_BIN" \
  METIS15_JAX_COMPILE_LOG_ROOT="$COMPILE_PROBE_LOG" \
  METIS15_JAX_COMPILE_TINY_CONFIG=1 \
  METIS15_JAX_COMPILE_REQUIRE_TPU=0 \
  METIS15_JAX_COMPILE_EXPERT_EXECUTION=reference \
  METIS15_JAX_COMPILE_MAX_STEPS=2 \
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_compile_probe.sh"

run "tiny quality canary" env \
  PYTHON_BIN="$PYTHON_BIN" \
  METIS15_JAX_CANARY_LOG_ROOT="$QUALITY_CANARY_LOG" \
  METIS15_JAX_CANARY_TINY_CONFIG=1 \
  METIS15_JAX_CANARY_REQUIRE_TPU=0 \
  METIS15_JAX_CANARY_EXPERT_EXECUTION=reference \
  METIS15_JAX_CANARY_MAX_STEPS=4 \
  METIS15_JAX_CANARY_MIN_LOSS_DROP_FRAC=0.01 \
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_quality_canary.sh"

run_logged "trainer CPT MoR synthetic canary" "$CPT_MOR_TRAINER_LOG/train.log" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage continued_pretrain \
  --synthetic-data \
  --max-steps 4 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --checkpoint-interval 0 \
  --skip-checkpoint \
  --expert-execution reference \
  --out-dir "$CPT_MOR_TRAINER_LOG/out"
run "audit CPT MoR trainer canary" "$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_metis15_jax_tpu_logs.py" \
  "$CPT_MOR_TRAINER_LOG/train.log" \
  --min-logged-steps 4 \
  --require-loss-decrease \
  --min-loss-drop-frac 0.01 \
  --require-mor-active \
  --require-mor-target-increase \
  --require-mor-coef-increase \
  --min-valid-assign-frac 0.90 \
  --max-expert-drop-frac 0.10 \
  --max-qk-logit 1000 \
  --perf-warmup-steps 3

CPT_MOR_RESUME="$LOG_ROOT/cpt_mor_resume"
rm -rf "$CPT_MOR_RESUME"
run_logged "trainer CPT MoR resume" "$CPT_MOR_RESUME/first.log" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage continued_pretrain \
  --synthetic-data \
  --max-steps 2 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --checkpoint-interval 1 \
  --expert-execution reference \
  --out-dir "$CPT_MOR_RESUME/out"
run_logged "trainer CPT MoR resume continuation" "$CPT_MOR_RESUME/continued.log" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage continued_pretrain \
  --synthetic-data \
  --max-steps 4 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --checkpoint-interval 1 \
  --expert-execution reference \
  --resume \
  --out-dir "$CPT_MOR_RESUME/out"
run "audit CPT MoR resume continuation" "$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_metis15_jax_tpu_logs.py" \
  "$CPT_MOR_RESUME/continued.log" \
  --min-logged-steps 2 \
  --require-loss-decrease \
  --min-loss-drop-frac 0.01 \
  --require-mor-active \
  --require-mor-target-increase \
  --require-mor-coef-increase \
  --min-valid-assign-frac 0.90 \
  --max-expert-drop-frac 0.10 \
  --max-qk-logit 1000 \
  --perf-warmup-steps 3
"$PYTHON_BIN" - "$CPT_MOR_RESUME/continued.log" "$CPT_MOR_RESUME/out/jax_train_summary.json" "$ROOT_DIR" <<'PY'
import json
import sys
from pathlib import Path

script_dir = Path(sys.argv[3]) / "scripts"
sys.path.insert(0, str(script_dir))
from analyze_metis15_jax_tpu_logs import parse_log

summary = parse_log(Path(sys.argv[1]))
if len(summary.steps) != 2:
    raise SystemExit(f"Expected two resumed CPT MoR steps, got {len(summary.steps)}")
first, last = summary.steps[0], summary.steps[-1]
if first.step != 3 or first.mor_target < 1.35 or first.mor_coef < 0.015:
    raise SystemExit(
        f"CPT MoR schedule restarted or resumed too early: step={first.step} "
        f"target={first.mor_target:.3f} coef={first.mor_coef:.6f}"
    )
train_summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if train_summary.get("mor_target_depth", 0.0) < 1.50 or train_summary.get("mor_aux_coef", 0.0) < 0.0175:
    raise SystemExit(f"CPT MoR summary did not retain final schedule state: {train_summary}")
print("metis15_jax_cpt_mor_resume_schedule_ok first_target=1.350 final_target=1.500")
PY

run "tiny perf sweep" env \
  PYTHON_BIN="$PYTHON_BIN" \
  METIS15_JAX_SWEEP_LOG_ROOT="$PERF_SWEEP_LOG" \
  METIS15_JAX_SWEEP_TINY_CONFIG=1 \
  METIS15_JAX_SWEEP_REQUIRE_TPU=0 \
  METIS15_JAX_SWEEP_LOCAL_BATCHES=2 \
  METIS15_JAX_SWEEP_GRAD_ACCUMS=2 \
  METIS15_JAX_SWEEP_EXPERT_EXECUTIONS=reference \
  METIS15_JAX_SWEEP_MAX_STEPS=3 \
  METIS15_JAX_SWEEP_REQUIRE_LOSS_DECREASE=1 \
  METIS15_JAX_SWEEP_MIN_LOSS_DROP_FRAC=0.01 \
  "$ROOT_DIR/scripts/metis15_jax_tpu_v6e_perf_sweep.sh"

DATA_SMOKE="$LOG_ROOT/data_resume"
rm -rf "$DATA_SMOKE"
mkdir -p "$DATA_SMOKE/data"
"$PYTHON_BIN" - "$DATA_SMOKE/data" <<'PY'
from pathlib import Path
import json
import numpy as np
import sys

root = Path(sys.argv[1])
tokens = (np.arange(1024, dtype=np.uint16) % 64).astype(np.uint16)
tokens.tofile(root / "train.bin")
(root / "meta.json").write_text(
    json.dumps({"dtype": "uint16", "vocab_size": 64, "train_tokens": int(tokens.size)}, indent=2) + "\n",
    encoding="utf-8",
)
PY

run "trainer real-data resume" "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage pretrain \
  --data-dir "$DATA_SMOKE/data" \
  --max-steps 2 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --checkpoint-interval 1 \
  --out-dir "$DATA_SMOKE/out"
run "trainer real-data resume continuation" "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage pretrain \
  --data-dir "$DATA_SMOKE/data" \
  --max-steps 3 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --checkpoint-interval 1 \
  --resume \
  --out-dir "$DATA_SMOKE/out"
"$PYTHON_BIN" - "$DATA_SMOKE/out/jax_train_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("tokens_seen") != 192:
    raise SystemExit(f"Expected real-data resume tokens_seen=192, got {summary.get('tokens_seen')}")
print("metis15_jax_real_data_resume_ok tokens_seen=192")
PY

MESH_RESUME="$LOG_ROOT/mesh_resume"
rm -rf "$MESH_RESUME"
run "trainer shard_map resume" env XLA_FLAGS="$MESH_XLA_FLAGS" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage pretrain \
  --synthetic-data \
  --max-steps 2 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --expert-execution shard_map \
  --checkpoint-interval 1 \
  --out-dir "$MESH_RESUME"
run "trainer shard_map resume continuation" env XLA_FLAGS="$MESH_XLA_FLAGS" \
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_metis15_jax_tpu.py" \
  --tiny-config \
  --stage pretrain \
  --synthetic-data \
  --max-steps 3 \
  --local-batch-size 2 \
  --grad-accum-steps 2 \
  --expert-execution shard_map \
  --checkpoint-interval 1 \
  --resume \
  --out-dir "$MESH_RESUME"
"$PYTHON_BIN" - "$MESH_RESUME/jax_train_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("tokens_seen") != 192 or summary.get("device_count") != 8:
    raise SystemExit(
        f"Expected shard_map resume tokens_seen=192 and device_count=8, got {summary}"
    )
print("metis15_jax_shard_map_resume_ok tokens_seen=192 devices=8")
PY

echo
echo "metis15_jax_tpu_v6e_local_readiness_ok"
echo "Logs retained under: $LOG_ROOT"
