#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/runpod_metis13_gpu.sh

Environment variables:
  METIS13_SHARED_ROOT               Shared network-volume root. Default: <repo>/.runpod/metis13
  METIS13_PYTHON_BIN                Explicit Python binary to use for training.
  METIS13_SKIP_ENV_SETUP            Set to 1 to skip pip/framework setup and use a prebuilt image env.
  METIS13_FORCE_UNLOCK              Set to 1 to clear a stale GPU training lock on the shared volume.
  METIS13_UPLOAD_RELEASES           If set to 1 and HF_TOKEN exists, upload base/chat/think releases.
  METIS13_BASE_LOCAL_BATCH_SIZE     Override base local batch size.
  METIS13_BASE_GRAD_ACCUM_STEPS     Override base grad accumulation.
  METIS13_CHAT_LOCAL_BATCH_SIZE     Override chat local batch size.
  METIS13_CHAT_GRAD_ACCUM_STEPS     Override chat grad accumulation.
  METIS13_THINK_LOCAL_BATCH_SIZE    Override think local batch size.
  METIS13_THINK_GRAD_ACCUM_STEPS    Override think grad accumulation.
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

choose_bootstrap_python() {
  local candidate
  for candidate in "${BOOTSTRAP_PYTHON:-}" python3.12 python3.11 python3.10 python3 python; do
    if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "No suitable Python interpreter found for Metis-1.3 GPU training." >&2
  return 1
}

SHARED_ROOT="${METIS13_SHARED_ROOT:-$ROOT_DIR/.runpod/metis13}"
PLAN_PATH="$SHARED_ROOT/plans/metis13_plan.json"
CACHE_ROOT="$SHARED_ROOT/cache"
TMP_ROOT="$CACHE_ROOT/tmp"
HF_ASSETS_DIR="$SHARED_ROOT/hf_assets/metis13"
PRETRAIN_DIR="$SHARED_ROOT/data/metis13_base"
CHAT_DIR="$SHARED_ROOT/data/metis13_chat_sft"
REASONING_DIR="$SHARED_ROOT/data/metis13_reasoning_sft"
RUNS_ROOT="$SHARED_ROOT/runs/metis13"
RELEASES_ROOT="$SHARED_ROOT/releases/metis13"
STATE_ROOT="$SHARED_ROOT/state"
LOCK_DIR="$STATE_ROOT/metis13_gpu_train.lock"
HOST_NAME="$(hostname -s 2>/dev/null || hostname || echo unknown)"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

acquire_lock() {
  mkdir -p "$STATE_ROOT"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=runpod_metis13_gpu.sh
EOF
    return 0
  fi

  if [[ -f "$LOCK_DIR/owner.env" ]]; then
    # shellcheck disable=SC1090
    source "$LOCK_DIR/owner.env"
    if [[ "${LOCK_HOST:-}" == "$HOST_NAME" && -n "${LOCK_PID:-}" ]] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
      log "Removing stale local GPU lock from pid ${LOCK_PID}."
      rm -rf "$LOCK_DIR"
    elif [[ "${METIS13_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of existing GPU lock."
      rm -rf "$LOCK_DIR"
    else
      cat >&2 <<EOF
Existing Metis-1.3 GPU lock detected at $LOCK_DIR
Lock host: ${LOCK_HOST:-unknown}
Lock pid:  ${LOCK_PID:-unknown}
Lock time: ${LOCK_CREATED_AT:-unknown}

If that run is no longer real, rerun with METIS13_FORCE_UNLOCK=1.
EOF
      exit 1
    fi
  fi

  mkdir "$LOCK_DIR"
  cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=runpod_metis13_gpu.sh
EOF
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"

if [[ -n "${METIS13_PYTHON_BIN:-}" ]]; then
  PY="$METIS13_PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  log "Creating fresh virtualenv with $(choose_bootstrap_python)."
  "$(choose_bootstrap_python)" -m venv .venv
  PY="$ROOT_DIR/.venv/bin/python"
fi

export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/hf/transformers}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$RUNS_ROOT" "$RELEASES_ROOT" "$TMPDIR"
acquire_lock

if [[ "${METIS13_SKIP_ENV_SETUP:-0}" != "1" ]]; then
  log "Installing GPU training dependencies."
  "$PY" -m pip install --upgrade pip 'setuptools<82' wheel
  "$PY" -m pip install -r requirements-gpu-train.txt
  "$PY" -m pip install --no-build-isolation 'causal-conv1d>=1.4.0' 'mamba-ssm>=2.2.4'
  "$PY" -m pip install -e .
else
  log "Skipping GPU environment setup and using prebuilt Python at $PY."
fi

mkdir -p "$ROOT_DIR/artifacts" "$ROOT_DIR/data"
ln -sfn "$HF_ASSETS_DIR" "$ROOT_DIR/artifacts/metis13_hf_assets"
ln -sfn "$PRETRAIN_DIR" "$ROOT_DIR/data/metis13_base"
ln -sfn "$CHAT_DIR" "$ROOT_DIR/data/metis13_chat_sft"
ln -sfn "$REASONING_DIR" "$ROOT_DIR/data/metis13_reasoning_sft"

json_get() {
  local path="$1"
  shift
  "$PY" - <<'PY' "$path" "$@"
import json
import sys

payload = json.load(open(sys.argv[1]))
value = payload
for key in sys.argv[2:]:
    value = value[key]
print(value)
PY
}

BASE_STEPS="$(json_get "$PLAN_PATH" pretrain steps)"
BASE_WARMUP="$(json_get "$PLAN_PATH" pretrain warmup_steps)"
CHAT_WARMUP="$(json_get "$PLAN_PATH" chat_sft warmup_steps)"
THINK_WARMUP="$(json_get "$PLAN_PATH" reasoning_sft warmup_steps)"

BASE_LR="$(json_get "$PLAN_PATH" pretrain lr)"
CHAT_LR="$(json_get "$PLAN_PATH" chat_sft lr)"
THINK_LR="$(json_get "$PLAN_PATH" reasoning_sft lr)"

CHAT_EPOCHS="$(json_get configs/metis13_manifest.json chat_sft epochs)"
CHAT_MAX_LENGTH="$(json_get configs/metis13_manifest.json chat_sft max_length)"
THINK_EPOCHS="$(json_get configs/metis13_manifest.json reasoning_sft epochs)"
THINK_MAX_LENGTH="$(json_get configs/metis13_manifest.json reasoning_sft max_length)"

BASE_LOCAL_BATCH="${METIS13_BASE_LOCAL_BATCH_SIZE:-$(json_get configs/metis13_manifest.json pretrain local_batch_size)}"
BASE_GRAD_ACCUM="${METIS13_BASE_GRAD_ACCUM_STEPS:-$(json_get configs/metis13_manifest.json pretrain grad_accum_steps)}"
CHAT_LOCAL_BATCH="${METIS13_CHAT_LOCAL_BATCH_SIZE:-$(json_get configs/metis13_manifest.json chat_sft local_batch_size)}"
CHAT_GRAD_ACCUM="${METIS13_CHAT_GRAD_ACCUM_STEPS:-$(json_get configs/metis13_manifest.json chat_sft grad_accum_steps)}"
THINK_LOCAL_BATCH="${METIS13_THINK_LOCAL_BATCH_SIZE:-$(json_get configs/metis13_manifest.json reasoning_sft local_batch_size)}"
THINK_GRAD_ACCUM="${METIS13_THINK_GRAD_ACCUM_STEPS:-$(json_get configs/metis13_manifest.json reasoning_sft grad_accum_steps)}"

BASE_CHECKPOINT_INTERVAL="${METIS13_BASE_CHECKPOINT_INTERVAL:-1000}"
CHAT_CHECKPOINT_INTERVAL="${METIS13_CHAT_CHECKPOINT_INTERVAL:-250}"
THINK_CHECKPOINT_INTERVAL="${METIS13_THINK_CHECKPOINT_INTERVAL:-125}"

BASE_RUN="$RUNS_ROOT/base"
CHAT_RUN="$RUNS_ROOT/chat"
THINK_RUN="$RUNS_ROOT/think"
BASE_RELEASE="$RELEASES_ROOT/base"
CHAT_RELEASE="$RELEASES_ROOT/chat"
THINK_RELEASE="$RELEASES_ROOT/think"
EVAL_REPORT="$RELEASES_ROOT/eval_comparison.json"

log "Training Metis-1.3 base model."
"$PY" scripts/train_mamba_lm.py \
  --resume \
  --manifest configs/metis13_manifest.json \
  --data-dir "$PRETRAIN_DIR" \
  --out-dir "$BASE_RUN" \
  --batch-size "$BASE_LOCAL_BATCH" \
  --grad-accum-steps "$BASE_GRAD_ACCUM" \
  --max-steps "$BASE_STEPS" \
  --lr "$BASE_LR" \
  --warmup-steps "$BASE_WARMUP" \
  --eval-interval 250 \
  --eval-iters 20 \
  --checkpoint-interval "$BASE_CHECKPOINT_INTERVAL" \
  --dtype bf16 \
  --fused-adamw

log "Exporting base release."
"$PY" scripts/export_mamba_checkpoint.py \
  --manifest configs/metis13_manifest.json \
  --checkpoint "$BASE_RUN/best.pt" \
  --assets-dir "$HF_ASSETS_DIR" \
  --out-dir "$BASE_RELEASE" \
  --stage-name base \
  --repo-id "Lernex/Metis-1.3-base"

log "Training Metis-1.3 chat SFT."
"$PY" scripts/train_mamba_sft.py \
  --resume \
  --base-checkpoint "$BASE_RUN/best.pt" \
  --train-jsonl "$CHAT_DIR/train.jsonl" \
  --val-jsonl "$CHAT_DIR/val.jsonl" \
  --tokenizer-path "$HF_ASSETS_DIR/tokenizer.json" \
  --out-dir "$CHAT_RUN" \
  --max-length "$CHAT_MAX_LENGTH" \
  --batch-size "$CHAT_LOCAL_BATCH" \
  --grad-accum-steps "$CHAT_GRAD_ACCUM" \
  --epochs "$CHAT_EPOCHS" \
  --lr "$CHAT_LR" \
  --warmup-steps "$CHAT_WARMUP" \
  --eval-interval 150 \
  --checkpoint-interval "$CHAT_CHECKPOINT_INTERVAL" \
  --dtype bf16 \
  --fused-adamw

log "Exporting chat release."
"$PY" scripts/export_mamba_checkpoint.py \
  --manifest configs/metis13_manifest.json \
  --checkpoint "$CHAT_RUN/best.pt" \
  --assets-dir "$HF_ASSETS_DIR" \
  --out-dir "$CHAT_RELEASE" \
  --stage-name chat \
  --repo-id "Lernex/Metis-1.3-chat"

log "Training Metis-1.3 reasoning SFT."
"$PY" scripts/train_mamba_sft.py \
  --resume \
  --base-checkpoint "$CHAT_RUN/best.pt" \
  --train-jsonl "$REASONING_DIR/train.jsonl" \
  --val-jsonl "$REASONING_DIR/val.jsonl" \
  --tokenizer-path "$HF_ASSETS_DIR/tokenizer.json" \
  --out-dir "$THINK_RUN" \
  --max-length "$THINK_MAX_LENGTH" \
  --batch-size "$THINK_LOCAL_BATCH" \
  --grad-accum-steps "$THINK_GRAD_ACCUM" \
  --epochs "$THINK_EPOCHS" \
  --lr "$THINK_LR" \
  --warmup-steps "$THINK_WARMUP" \
  --eval-interval 100 \
  --checkpoint-interval "$THINK_CHECKPOINT_INTERVAL" \
  --dtype bf16 \
  --fused-adamw

log "Exporting think release."
"$PY" scripts/export_mamba_checkpoint.py \
  --manifest configs/metis13_manifest.json \
  --checkpoint "$THINK_RUN/best.pt" \
  --assets-dir "$HF_ASSETS_DIR" \
  --out-dir "$THINK_RELEASE" \
  --stage-name think \
  --repo-id "Lernex/Metis-1.3-think"

log "Running evaluation suite."
"$PY" scripts/eval_model_suite.py \
  --suite configs/metis13_eval_prompts.json \
  --model base="$BASE_RELEASE" \
  --model chat="$CHAT_RELEASE" \
  --model think="$THINK_RELEASE" \
  --output-path "$EVAL_REPORT"

if [[ "${METIS13_UPLOAD_RELEASES:-0}" == "1" && -n "${HF_TOKEN:-}" ]]; then
  log "Uploading release folders to Hugging Face."
  "$PY" scripts/upload_hf_model.py --create-repo --private --repo-id "Lernex/Metis-1.3-base" --artifact-dir "$BASE_RELEASE" --message "Upload Metis 1.3 base release"
  "$PY" scripts/upload_hf_model.py --create-repo --private --repo-id "Lernex/Metis-1.3-chat" --artifact-dir "$CHAT_RELEASE" --message "Upload Metis 1.3 chat release"
  "$PY" scripts/upload_hf_model.py --create-repo --private --repo-id "Lernex/Metis-1.3-think" --artifact-dir "$THINK_RELEASE" --message "Upload Metis 1.3 think release"
fi

log "Metis-1.3 GPU training flow complete."
echo "Metis-1.3 GPU training complete."
echo "Shared root: $SHARED_ROOT"
echo "Eval report:  $EVAL_REPORT"
