#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/runpod_metis12_gpu.sh

Environment variables:
  METIS12_SHARED_ROOT             Shared network-volume root. Default: <repo>/.runpod/metis12
  METIS12_TORCHTITAN_DIR          TorchTitan checkout path. Default: <shared>/vendor/torchtitan
  METIS12_PYTHON_BIN              Explicit Python binary to use for training.
  METIS12_SKIP_ENV_SETUP          Set to 1 to skip pip/framework setup and use a prebuilt image env.
  METIS12_SPOT_MODE               Set to 1 for preemptible spot pods. Uses shorter checkpoint cadence.
  METIS12_FORCE_UNLOCK            Set to 1 to clear a stale GPU training lock on the shared volume.
  METIS12_NPROC                   torchrun processes. Default: 1
  METIS12_UPLOAD_RELEASES         If set to 1 and HF_TOKEN exists, upload base/chat/think releases.
  METIS12_BASE_CHECKPOINT_INTERVAL  Override base checkpoint interval.
  METIS12_BASE_LOCAL_BATCH_SIZE   Override base local batch size.
  METIS12_BASE_GLOBAL_BATCH_SIZE  Override base global batch size.
  METIS12_CHAT_CHECKPOINT_INTERVAL  Override chat checkpoint interval.
  METIS12_CHAT_LOCAL_BATCH_SIZE   Override chat local batch size.
  METIS12_CHAT_GLOBAL_BATCH_SIZE  Override chat global batch size.
  METIS12_THINK_CHECKPOINT_INTERVAL Override think checkpoint interval.
  METIS12_THINK_LOCAL_BATCH_SIZE  Override think local batch size.
  METIS12_THINK_GLOBAL_BATCH_SIZE Override think global batch size.
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

  echo "No suitable Python interpreter found for Metis-1.2 GPU training." >&2
  return 1
}

SHARED_ROOT="${METIS12_SHARED_ROOT:-$ROOT_DIR/.runpod/metis12}"
TORCHTITAN_DIR="${METIS12_TORCHTITAN_DIR:-$SHARED_ROOT/vendor/torchtitan}"
PLAN_PATH="$SHARED_ROOT/plans/metis12_plan.json"
CACHE_ROOT="$SHARED_ROOT/cache"
TMP_ROOT="$CACHE_ROOT/tmp"
HF_ASSETS_DIR="$SHARED_ROOT/hf_assets/metis12"
PRETRAIN_DIR="$SHARED_ROOT/data/metis12_base"
CHAT_DIR="$SHARED_ROOT/data/metis12_chat_sft"
REASONING_DIR="$SHARED_ROOT/data/metis12_reasoning_sft"
RUNS_ROOT="$SHARED_ROOT/runs/metis12"
RELEASES_ROOT="$SHARED_ROOT/releases/metis12"
STATE_ROOT="$SHARED_ROOT/state"
LOCK_DIR="$STATE_ROOT/metis12_gpu_train.lock"
STAGE_PATH="$STATE_ROOT/metis12_gpu_stage.txt"
STATUS_PATH="$STATE_ROOT/metis12_gpu_status.env"
HOST_NAME="$(hostname -s 2>/dev/null || hostname || echo unknown)"
NPROC="${METIS12_NPROC:-1}"
SPOT_MODE="${METIS12_SPOT_MODE:-0}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

write_status() {
  local stage="$1"
  mkdir -p "$STATE_ROOT"
  printf '%s\n' "$stage" > "$STAGE_PATH"
  cat > "$STATUS_PATH" <<EOF
METIS12_STAGE=$stage
METIS12_UPDATED_AT=$(timestamp)
METIS12_HOST=$HOST_NAME
METIS12_PID=$$
METIS12_SCRIPT=runpod_metis12_gpu.sh
METIS12_SHARED_ROOT=$SHARED_ROOT
EOF
}

acquire_lock() {
  mkdir -p "$STATE_ROOT"

  if mkdir "$LOCK_DIR" 2>/dev/null; then
    cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=runpod_metis12_gpu.sh
EOF
    return 0
  fi

  if [[ -f "$LOCK_DIR/owner.env" ]]; then
    # shellcheck disable=SC1090
    source "$LOCK_DIR/owner.env"
    if [[ "${LOCK_HOST:-}" == "$HOST_NAME" && -n "${LOCK_PID:-}" ]] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
      log "Removing stale local GPU lock from pid ${LOCK_PID}."
      rm -rf "$LOCK_DIR"
    elif [[ "${METIS12_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of existing GPU lock."
      rm -rf "$LOCK_DIR"
    else
      cat >&2 <<EOF
Existing Metis-1.2 GPU lock detected at $LOCK_DIR
Lock host: ${LOCK_HOST:-unknown}
Lock pid:  ${LOCK_PID:-unknown}
Lock time: ${LOCK_CREATED_AT:-unknown}

If that run is no longer real, rerun with METIS12_FORCE_UNLOCK=1.
EOF
      exit 1
    fi
  else
    if [[ "${METIS12_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of malformed GPU lock."
      rm -rf "$LOCK_DIR"
    else
      echo "Malformed GPU lock at $LOCK_DIR. Rerun with METIS12_FORCE_UNLOCK=1 to clear it." >&2
      exit 1
    fi
  fi

  mkdir "$LOCK_DIR"
  cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=runpod_metis12_gpu.sh
EOF
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

trap cleanup EXIT INT TERM

mkdir -p "$RUNS_ROOT" "$RELEASES_ROOT" "$SHARED_ROOT/vendor" "$TMP_ROOT"

acquire_lock
write_status setup

cd "$ROOT_DIR"

if [[ -n "${METIS12_PYTHON_BIN:-}" ]]; then
  PY="$METIS12_PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  log "Creating fresh virtualenv with $(choose_bootstrap_python)."
  "$(choose_bootstrap_python)" -m venv .venv
  PY="$ROOT_DIR/.venv/bin/python"
fi

export PYTHONPATH="$ROOT_DIR/src:$TORCHTITAN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CACHE_ROOT/hf/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/hf/transformers}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

if [[ "${METIS12_SKIP_ENV_SETUP:-0}" != "1" ]]; then
  log "Running GPU setup with TMPDIR=$TMPDIR."
  "$PY" -m pip install --upgrade pip 'setuptools<82' wheel
  "$PY" -m pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

  if [[ ! -d "$TORCHTITAN_DIR/.git" ]]; then
    log "Cloning TorchTitan into $TORCHTITAN_DIR."
    git clone --depth 1 https://github.com/pytorch/torchtitan "$TORCHTITAN_DIR"
  else
    log "Updating TorchTitan checkout."
    git -C "$TORCHTITAN_DIR" pull --ff-only
  fi

  "$PY" -m pip install -r "$TORCHTITAN_DIR/requirements.txt"
  "$PY" -m pip install -e "$TORCHTITAN_DIR"
  USE_CPP=0 "$PY" -m pip install --no-build-isolation git+https://github.com/pytorch/ao.git
  "$PY" -m pip install -r requirements.txt
  "$PY" -m pip install -e .
else
  log "Skipping GPU environment setup and using prebuilt Python at $PY."
fi

mkdir -p "$ROOT_DIR/artifacts" "$ROOT_DIR/data"
ln -sfn "$HF_ASSETS_DIR" "$ROOT_DIR/artifacts/metis12_hf_assets"
ln -sfn "$PRETRAIN_DIR" "$ROOT_DIR/data/metis12_base"
ln -sfn "$CHAT_DIR" "$ROOT_DIR/data/metis12_chat_sft"
ln -sfn "$REASONING_DIR" "$ROOT_DIR/data/metis12_reasoning_sft"

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
CHAT_STEPS="$(json_get "$PLAN_PATH" chat_sft steps)"
CHAT_WARMUP="$(json_get "$PLAN_PATH" chat_sft warmup_steps)"
THINK_STEPS="$(json_get "$PLAN_PATH" reasoning_sft steps)"
THINK_WARMUP="$(json_get "$PLAN_PATH" reasoning_sft warmup_steps)"

BASE_GLOBAL_BATCH="${METIS12_BASE_GLOBAL_BATCH_SIZE:-$(json_get "$PLAN_PATH" pretrain global_batch_size)}"
CHAT_GLOBAL_BATCH="${METIS12_CHAT_GLOBAL_BATCH_SIZE:-$(json_get "$PLAN_PATH" chat_sft global_batch_size)}"
THINK_GLOBAL_BATCH="${METIS12_THINK_GLOBAL_BATCH_SIZE:-$(json_get "$PLAN_PATH" reasoning_sft global_batch_size)}"

BASE_LOCAL_BATCH="${METIS12_BASE_LOCAL_BATCH_SIZE:-8}"
CHAT_LOCAL_BATCH="${METIS12_CHAT_LOCAL_BATCH_SIZE:-16}"
THINK_LOCAL_BATCH="${METIS12_THINK_LOCAL_BATCH_SIZE:-12}"

if [[ "$SPOT_MODE" == "1" ]]; then
  DEFAULT_BASE_CHECKPOINT_INTERVAL=250
  DEFAULT_CHAT_CHECKPOINT_INTERVAL=100
  DEFAULT_THINK_CHECKPOINT_INTERVAL=50
else
  DEFAULT_BASE_CHECKPOINT_INTERVAL=1000
  DEFAULT_CHAT_CHECKPOINT_INTERVAL=250
  DEFAULT_THINK_CHECKPOINT_INTERVAL=100
fi

BASE_CHECKPOINT_INTERVAL="${METIS12_BASE_CHECKPOINT_INTERVAL:-$DEFAULT_BASE_CHECKPOINT_INTERVAL}"
CHAT_CHECKPOINT_INTERVAL="${METIS12_CHAT_CHECKPOINT_INTERVAL:-$DEFAULT_CHAT_CHECKPOINT_INTERVAL}"
THINK_CHECKPOINT_INTERVAL="${METIS12_THINK_CHECKPOINT_INTERVAL:-$DEFAULT_THINK_CHECKPOINT_INTERVAL}"

BASE_RUN="$RUNS_ROOT/base"
CHAT_RUN="$RUNS_ROOT/chat"
THINK_RUN="$RUNS_ROOT/think"
BASE_RELEASE="$RELEASES_ROOT/base"
CHAT_RELEASE="$RELEASES_ROOT/chat"
THINK_RELEASE="$RELEASES_ROOT/think"
EVAL_REPORT="$RELEASES_ROOT/eval_comparison.json"

run_titan() {
  "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m torchtitan.train "$@"
}

release_ready() {
  local release_dir="$1"
  [[ -f "$release_dir/config.json" ]] && [[ -f "$release_dir/model.safetensors" || -f "$release_dir/model.safetensors.index.json" ]]
}

latest_checkpoint_dir() {
  local checkpoints_root="$1"
  find "$checkpoints_root" -maxdepth 1 -type d -name 'step-*' | sort -V | tail -n 1
}

latest_checkpoint_step() {
  local checkpoints_root="$1"
  local latest=""
  latest="$(latest_checkpoint_dir "$checkpoints_root")"
  if [[ -z "$latest" ]]; then
    printf '0\n'
    return 0
  fi
  latest="$(basename "$latest")"
  latest="${latest#step-}"
  printf '%s\n' "$latest"
}

resolve_export_dir() {
  local checkpoints_root="$1"
  local expected_step="$2"
  local exact="$checkpoints_root/step-$expected_step"
  local latest=""

  if [[ -d "$exact" ]]; then
    printf '%s\n' "$exact"
    return 0
  fi

  latest="$(find "$checkpoints_root" -maxdepth 1 -type d -name 'step-*' | sort -V | tail -n 1)"
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi

  echo "No checkpoint export directory found under $checkpoints_root" >&2
  return 1
}

run_base_stage() {
  local latest_step="0"
  local export_dir=""

  if release_ready "$BASE_RELEASE"; then
    log "Base release already exists at $BASE_RELEASE. Skipping base training/export."
    return 0
  fi

  latest_step="$(latest_checkpoint_step "$BASE_RUN/checkpoints")"
  if (( latest_step < BASE_STEPS )); then
    write_status base_train
    if (( latest_step > 0 )); then
      log "Resuming base training from checkpoint step-$latest_step."
    else
      log "Starting base training stage."
    fi
    run_titan \
      --module metis_titan \
      --config metis12_base \
      --hf_assets_path "$HF_ASSETS_DIR" \
      --parallelism.data_parallel_replicate_degree "$NPROC" \
      --training.steps "$BASE_STEPS" \
      --training.local_batch_size "$BASE_LOCAL_BATCH" \
      --training.global_batch_size "$BASE_GLOBAL_BATCH" \
      --lr_scheduler.warmup_steps "$BASE_WARMUP" \
      --dataloader.dataset_path "$PRETRAIN_DIR" \
      --validator.dataloader.dataset_path "$PRETRAIN_DIR" \
      --checkpoint.interval "$BASE_CHECKPOINT_INTERVAL" \
      --checkpoint.folder "$BASE_RUN/checkpoints" \
      --dump_folder "$BASE_RUN"
  else
    log "Base checkpoints already reached step-$latest_step. Skipping directly to base export."
  fi

  write_status base_export
  export_dir="$(resolve_export_dir "$BASE_RUN/checkpoints" "$BASE_STEPS")"
  log "Assembling base release from $export_dir."
  "$PY" scripts/assemble_hf_release.py \
    --manifest configs/metis12_manifest.json \
    --stage-name base \
    --assets-dir "$HF_ASSETS_DIR" \
    --export-dir "$export_dir" \
    --out-dir "$BASE_RELEASE" \
    --repo-id "Lernex/Metis-1.2-base"
}

run_chat_stage() {
  local latest_step="0"
  local export_dir=""

  if release_ready "$CHAT_RELEASE"; then
    log "Chat release already exists at $CHAT_RELEASE. Skipping chat training/export."
    return 0
  fi

  if ! release_ready "$BASE_RELEASE"; then
    echo "Base release is missing at $BASE_RELEASE; cannot start chat SFT." >&2
    exit 1
  fi

  latest_step="$(latest_checkpoint_step "$CHAT_RUN/checkpoints")"
  if (( latest_step < CHAT_STEPS )); then
    write_status chat_train
    if (( latest_step > 0 )); then
      log "Resuming chat SFT from checkpoint step-$latest_step."
    else
      log "Starting chat SFT stage."
    fi
    run_titan \
      --module metis_titan \
      --config metis12_chat \
      --hf_assets_path "$HF_ASSETS_DIR" \
      --parallelism.data_parallel_replicate_degree "$NPROC" \
      --training.steps "$CHAT_STEPS" \
      --training.local_batch_size "$CHAT_LOCAL_BATCH" \
      --training.global_batch_size "$CHAT_GLOBAL_BATCH" \
      --lr_scheduler.warmup_steps "$CHAT_WARMUP" \
      --checkpoint.initial_load_path "$BASE_RELEASE" \
      --checkpoint.initial_load_in_hf \
      --checkpoint.interval "$CHAT_CHECKPOINT_INTERVAL" \
      --checkpoint.folder "$CHAT_RUN/checkpoints" \
      --dump_folder "$CHAT_RUN"
  else
    log "Chat checkpoints already reached step-$latest_step. Skipping directly to chat export."
  fi

  write_status chat_export
  export_dir="$(resolve_export_dir "$CHAT_RUN/checkpoints" "$CHAT_STEPS")"
  log "Assembling chat release from $export_dir."
  "$PY" scripts/assemble_hf_release.py \
    --manifest configs/metis12_manifest.json \
    --stage-name chat \
    --assets-dir "$HF_ASSETS_DIR" \
    --export-dir "$export_dir" \
    --out-dir "$CHAT_RELEASE" \
    --repo-id "Lernex/Metis-1.2-chat"
}

run_think_stage() {
  local latest_step="0"
  local export_dir=""

  if release_ready "$THINK_RELEASE"; then
    log "Think release already exists at $THINK_RELEASE. Skipping reasoning training/export."
    return 0
  fi

  if ! release_ready "$CHAT_RELEASE"; then
    echo "Chat release is missing at $CHAT_RELEASE; cannot start reasoning SFT." >&2
    exit 1
  fi

  latest_step="$(latest_checkpoint_step "$THINK_RUN/checkpoints")"
  if (( latest_step < THINK_STEPS )); then
    write_status think_train
    if (( latest_step > 0 )); then
      log "Resuming reasoning SFT from checkpoint step-$latest_step."
    else
      log "Starting reasoning SFT stage."
    fi
    run_titan \
      --module metis_titan \
      --config metis12_think \
      --hf_assets_path "$HF_ASSETS_DIR" \
      --parallelism.data_parallel_replicate_degree "$NPROC" \
      --training.steps "$THINK_STEPS" \
      --training.local_batch_size "$THINK_LOCAL_BATCH" \
      --training.global_batch_size "$THINK_GLOBAL_BATCH" \
      --lr_scheduler.warmup_steps "$THINK_WARMUP" \
      --checkpoint.initial_load_path "$CHAT_RELEASE" \
      --checkpoint.initial_load_in_hf \
      --checkpoint.interval "$THINK_CHECKPOINT_INTERVAL" \
      --checkpoint.folder "$THINK_RUN/checkpoints" \
      --dump_folder "$THINK_RUN"
  else
    log "Reasoning checkpoints already reached step-$latest_step. Skipping directly to think export."
  fi

  write_status think_export
  export_dir="$(resolve_export_dir "$THINK_RUN/checkpoints" "$THINK_STEPS")"
  log "Assembling think release from $export_dir."
  "$PY" scripts/assemble_hf_release.py \
    --manifest configs/metis12_manifest.json \
    --stage-name think \
    --assets-dir "$HF_ASSETS_DIR" \
    --export-dir "$export_dir" \
    --out-dir "$THINK_RELEASE" \
    --repo-id "Lernex/Metis-1.2-think"
}

run_base_stage
run_chat_stage
run_think_stage

if [[ -f "$EVAL_REPORT" ]]; then
  log "Eval report already exists at $EVAL_REPORT. Skipping eval suite."
else
  write_status eval
  log "Running eval comparison suite."
  "$PY" scripts/eval_model_suite.py \
    --suite configs/metis12_eval_prompts.json \
    --model "base=$BASE_RELEASE" \
    --model "chat=$CHAT_RELEASE" \
    --model "think=$THINK_RELEASE" \
    --output-path "$EVAL_REPORT"
fi

if [[ "${METIS12_UPLOAD_RELEASES:-0}" == "1" && -n "${HF_TOKEN:-}" ]]; then
  write_status upload
  log "Uploading HF releases."
  "$PY" scripts/upload_hf_model.py --create-repo --private --repo-id "Lernex/Metis-1.2-base" --artifact-dir "$BASE_RELEASE" --message "Upload Metis-1.2 base release"
  "$PY" scripts/upload_hf_model.py --create-repo --private --repo-id "Lernex/Metis-1.2-chat" --artifact-dir "$CHAT_RELEASE" --message "Upload Metis-1.2 chat release"
  "$PY" scripts/upload_hf_model.py --create-repo --private --repo-id "Lernex/Metis-1.2-think" --artifact-dir "$THINK_RELEASE" --message "Upload Metis-1.2 think release"
fi

write_status complete
log "Metis-1.2 GPU training flow complete."
echo "Metis-1.2 GPU training flow complete."
echo "Base release:  $BASE_RELEASE"
echo "Chat release:  $CHAT_RELEASE"
echo "Think release: $THINK_RELEASE"
echo "Eval report:   $EVAL_REPORT"
