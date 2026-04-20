#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/runpod_metis13_cpu.sh

Environment variables:
  METIS13_CPU_ROLE               One of: memory, compute, full. Default: full
  METIS13_SHARED_ROOT             Shared network-volume root. Default: <repo>/.runpod/metis13
  METIS13_LOCAL_ROOT              Fast local-disk working root. Default depends on role
  METIS13_FORCE_UNLOCK            Set to 1 to clear a stale CPU prep lock on the shared volume.
  METIS13_START_STAGE             Optional stage to start from. Default: setup
  METIS13_TOKENIZER_MODE          Tokenizer prep mode: stream or sample. Default: sample
  METIS13_KEEP_TOKENIZER_SAMPLE   Set to 1 to keep the large tokenizer sample after tokenizer training.
  METIS13_TOKENIZER_SAMPLES       Tokenizer documents to sample. Default: 8000000
  METIS13_PRETRAIN_MAX_DOCS       Upper bound on streamed docs during pretrain prep. Default: 25000000
  METIS13_PRETRAIN_TARGET_TOKENS  Train-token target for base pretraining. Default: 12000000000
  METIS13_PRETRAIN_VAL_RATIO      Validation holdout ratio for base prep. Default: 0.01
  METIS13_PRETRAIN_ENCODE_BATCH   Docs to tokenize per encode_batch call. Default: 128
  METIS13_CHAT_EXAMPLES           Target chat SFT examples. Default: 500000
  METIS13_REASONING_EXAMPLES      Target reasoning SFT examples. Default: 180000
  METIS13_SFT_SCRATCH_ROOT        Local scratch root for SFT dataset cache/temp files. Default: /tmp/metis13_sft

Role behavior:
  memory  -> tokenizer sample (optional), tokenizer fit, HF asset render
  compute -> pretraining memmaps, chat SFT, reasoning SFT, planning
  full    -> run both halves sequentially on one pod
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
  echo "No suitable Python interpreter found for Metis-1.3 CPU prep." >&2
  return 1
}

CPU_ROLE="${METIS13_CPU_ROLE:-full}"
case "$CPU_ROLE" in
  memory|compute|full) ;;
  *)
    echo "Unsupported METIS13_CPU_ROLE: $CPU_ROLE" >&2
    exit 1
    ;;
esac

TOKENIZER_MODE="${METIS13_TOKENIZER_MODE:-sample}"
case "$TOKENIZER_MODE" in
  stream|sample) ;;
  *)
    echo "Unsupported METIS13_TOKENIZER_MODE: $TOKENIZER_MODE" >&2
    exit 1
    ;;
esac

SHARED_ROOT="${METIS13_SHARED_ROOT:-$ROOT_DIR/.runpod/metis13}"
case "$CPU_ROLE" in
  memory)
    DEFAULT_LOCAL_ROOT="/tmp/metis13_cpu_memory"
    ;;
  compute)
    DEFAULT_LOCAL_ROOT="/tmp/metis13_cpu_compute"
    ;;
  full)
    DEFAULT_LOCAL_ROOT="/tmp/metis13_cpu_prep"
    ;;
esac
LOCAL_ROOT="${METIS13_LOCAL_ROOT:-$DEFAULT_LOCAL_ROOT}"
CACHE_ROOT="$LOCAL_ROOT/cache"
TMP_ROOT="$CACHE_ROOT/tmp"
HF_ASSETS_DIR="$SHARED_ROOT/hf_assets/metis13"
PRETRAIN_DIR="$SHARED_ROOT/data/metis13_base"
CHAT_DIR="$SHARED_ROOT/data/metis13_chat_sft"
REASONING_DIR="$SHARED_ROOT/data/metis13_reasoning_sft"
PLAN_PATH="$SHARED_ROOT/plans/metis13_plan.json"
LOCAL_HF_ASSETS_DIR="$LOCAL_ROOT/hf_assets/metis13"
LOCAL_PRETRAIN_DIR="$LOCAL_ROOT/data/metis13_base"
LOCAL_CHAT_DIR="$LOCAL_ROOT/data/metis13_chat_sft"
LOCAL_REASONING_DIR="$LOCAL_ROOT/data/metis13_reasoning_sft"
LOCAL_PLAN_PATH="$LOCAL_ROOT/plans/metis13_plan.json"
LOCAL_TOKENIZER_SAMPLE_PATH="$LOCAL_ROOT/samples/metis13_tokenizer_sample.jsonl"
STATE_ROOT="$SHARED_ROOT/state"
LOCK_DIR="$STATE_ROOT/metis13_cpu_${CPU_ROLE}.lock"
STAGE_PATH="$STATE_ROOT/metis13_cpu_${CPU_ROLE}_stage.txt"
STATUS_PATH="$STATE_ROOT/metis13_cpu_${CPU_ROLE}_status.env"
LAST_COMPLETED_STAGE_PATH="$STATE_ROOT/metis13_cpu_${CPU_ROLE}_last_completed_stage.txt"
HOST_NAME="$(hostname -s 2>/dev/null || hostname || echo unknown)"

TOKENIZER_SAMPLES="${METIS13_TOKENIZER_SAMPLES:-8000000}"
PRETRAIN_MAX_DOCS="${METIS13_PRETRAIN_MAX_DOCS:-25000000}"
PRETRAIN_TARGET_TOKENS="${METIS13_PRETRAIN_TARGET_TOKENS:-12000000000}"
PRETRAIN_VAL_RATIO="${METIS13_PRETRAIN_VAL_RATIO:-0.01}"
PRETRAIN_ENCODE_BATCH="${METIS13_PRETRAIN_ENCODE_BATCH:-128}"
CHAT_EXAMPLES="${METIS13_CHAT_EXAMPLES:-500000}"
REASONING_EXAMPLES="${METIS13_REASONING_EXAMPLES:-180000}"
SFT_SCRATCH_ROOT="${METIS13_SFT_SCRATCH_ROOT:-/tmp/metis13_sft}"
START_STAGE="${METIS13_START_STAGE:-setup}"
PRETRAIN_REQUIREMENTS_FILE="$ROOT_DIR/requirements-cpu-pretrain.txt"
SFT_REQUIREMENTS_FILE="$ROOT_DIR/requirements-cpu-sft.txt"
ENV_STATE_DIR="$LOCAL_ROOT/env_state"
ENV_FLAVOR_PATH="$ENV_STATE_DIR/metis13_cpu_env_flavor.txt"

PY=""

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

ensure_python_runtime() {
  if [[ -n "$PY" && -x "$PY" ]]; then
    return 0
  fi

  if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
    log "Creating fresh virtualenv with $(choose_bootstrap_python)."
    "$(choose_bootstrap_python)" -m venv .venv
  fi
  PY="$ROOT_DIR/.venv/bin/python"
}

ensure_stage_env() {
  local flavor="$1"
  local requirements_file="$2"

  ensure_python_runtime
  mkdir -p "$ENV_STATE_DIR"

  if [[ -f "$ENV_FLAVOR_PATH" && "$(cat "$ENV_FLAVOR_PATH")" == "$flavor" ]]; then
    return 0
  fi

  log "Installing $flavor CPU prep dependencies with TMPDIR=$TMPDIR."
  "$PY" -m pip install --upgrade pip 'setuptools<82' wheel
  "$PY" -m pip install -r "$requirements_file"
  printf '%s\n' "$flavor" > "$ENV_FLAVOR_PATH"
}

role_runs_stage() {
  local target="$1"
  case "$CPU_ROLE" in
    memory)
      case "$target" in
        setup|tokenizer_sample|tokenizer|hf_assets) return 0 ;;
      esac
      ;;
    compute)
      case "$target" in
        setup|pretrain_data|chat_sft_data|reasoning_sft_data|planning) return 0 ;;
      esac
      ;;
    full)
      return 0
      ;;
  esac
  return 1
}

sync_file_to_shared() {
  local local_path="$1"
  local shared_path="$2"
  mkdir -p "$(dirname "$shared_path")"
  cp -f "$local_path" "$shared_path"
}

sync_dir_to_shared() {
  local local_dir="$1"
  local shared_dir="$2"
  mkdir -p "$(dirname "$shared_dir")"
  rm -rf "$shared_dir"
  mkdir -p "$shared_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$local_dir"/ "$shared_dir"/
  else
    cp -a "$local_dir"/. "$shared_dir"/
  fi
}

sync_dir_from_shared() {
  local shared_dir="$1"
  local local_dir="$2"
  mkdir -p "$(dirname "$local_dir")"
  rm -rf "$local_dir"
  mkdir -p "$local_dir"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$shared_dir"/ "$local_dir"/
  else
    cp -a "$shared_dir"/. "$local_dir"/
  fi
}

stage_index() {
  case "$1" in
    setup) echo 0 ;;
    tokenizer_sample) echo 1 ;;
    tokenizer) echo 2 ;;
    hf_assets) echo 3 ;;
    pretrain_data) echo 4 ;;
    chat_sft_data) echo 5 ;;
    reasoning_sft_data) echo 6 ;;
    planning) echo 7 ;;
    complete) echo 8 ;;
    *)
      echo "Unknown METIS13_START_STAGE: $1" >&2
      exit 1
      ;;
  esac
}

should_run_stage() {
  local target="$1"
  role_runs_stage "$target" && [[ "$(stage_index "$target")" -ge "$(stage_index "$START_STAGE")" ]]
}

write_status() {
  local stage="$1"
  local last_completed=""
  if [[ -f "$LAST_COMPLETED_STAGE_PATH" ]]; then
    last_completed="$(cat "$LAST_COMPLETED_STAGE_PATH")"
  fi
  mkdir -p "$STATE_ROOT"
  printf '%s\n' "$stage" > "$STAGE_PATH"
  cat > "$STATUS_PATH" <<EOF
METIS13_STAGE=$stage
METIS13_UPDATED_AT=$(timestamp)
METIS13_HOST=$HOST_NAME
METIS13_PID=$$
METIS13_SCRIPT=runpod_metis13_cpu.sh
METIS13_CPU_ROLE=$CPU_ROLE
METIS13_SHARED_ROOT=$SHARED_ROOT
METIS13_LOCAL_ROOT=$LOCAL_ROOT
METIS13_LAST_COMPLETED_STAGE=$last_completed
EOF
}

mark_stage_complete() {
  local stage="$1"
  mkdir -p "$STATE_ROOT"
  printf '%s\n' "$stage" > "$LAST_COMPLETED_STAGE_PATH"
  write_status "$stage"
}

acquire_lock() {
  mkdir -p "$STATE_ROOT"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=runpod_metis13_cpu.sh
EOF
    return 0
  fi

  if [[ -f "$LOCK_DIR/owner.env" ]]; then
    # shellcheck disable=SC1090
    source "$LOCK_DIR/owner.env"
    if [[ "${LOCK_HOST:-}" == "$HOST_NAME" && -n "${LOCK_PID:-}" ]] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
      log "Removing stale local CPU prep lock from pid ${LOCK_PID}."
      rm -rf "$LOCK_DIR"
    elif [[ "${METIS13_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of existing CPU prep lock."
      rm -rf "$LOCK_DIR"
    else
      cat >&2 <<EOF
Existing Metis-1.3 CPU prep lock detected at $LOCK_DIR
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
LOCK_SCRIPT=runpod_metis13_cpu.sh
EOF
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

cleanup_tokenizer_sample() {
  if [[ "${METIS13_KEEP_TOKENIZER_SAMPLE:-0}" == "1" ]]; then
    log "Keeping tokenizer sample because METIS13_KEEP_TOKENIZER_SAMPLE=1."
    return 0
  fi
  rm -f "$LOCAL_TOKENIZER_SAMPLE_PATH" "${LOCAL_TOKENIZER_SAMPLE_PATH}.meta.json"
}

cleanup_transient_hf_cache() {
  local cache_path
  for cache_path in \
    "$HF_HOME/hub/datasets--HuggingFaceTB--smoltalk2" \
    "$HF_HOME/hub/datasets--HuggingFaceTB--smoltalk2_everyday_convs_think" \
    "$HF_HOME/hub/datasets--open-thoughts--OpenThoughts-114k"
  do
    if [[ -e "$cache_path" ]]; then
      rm -rf "$cache_path"
    fi
  done
}

cleanup_local_dir() {
  local target="$1"
  if [[ -e "$target" ]]; then
    rm -rf "$target"
  fi
}

tokenizer_sample_is_reusable() {
  [[ -f "$LOCAL_TOKENIZER_SAMPLE_PATH" && -f "${LOCAL_TOKENIZER_SAMPLE_PATH}.meta.json" ]] || return 1
  "$PY" - "$LOCAL_TOKENIZER_SAMPLE_PATH" "${LOCAL_TOKENIZER_SAMPLE_PATH}.meta.json" "$TOKENIZER_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

sample_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
expected_samples = int(sys.argv[3])

try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

if meta.get("max_samples") != expected_samples:
    raise SystemExit(1)
if Path(meta.get("output_path", "")) != sample_path:
    raise SystemExit(1)
if sample_path.stat().st_size <= 0:
    raise SystemExit(1)
PY
}

run_tokenizer_sample_stage() {
  local stage_root="$LOCAL_ROOT/stages/tokenizer_sample"
  local stage_hf_home="$stage_root/hf"
  local stage_tmp="$stage_root/tmp"

  if tokenizer_sample_is_reusable; then
    log "Reusing existing tokenizer sample at $LOCAL_TOKENIZER_SAMPLE_PATH."
    return 0
  fi

  rm -rf "$stage_root"
  mkdir -p "$stage_hf_home/datasets" "$stage_hf_home/transformers" "$stage_tmp"

  HF_HOME="$stage_hf_home" \
  HF_DATASETS_CACHE="$stage_hf_home/datasets" \
  TRANSFORMERS_CACHE="$stage_hf_home/transformers" \
  TMPDIR="$stage_tmp" \
  TMP="$stage_tmp" \
  TEMP="$stage_tmp" \
  "$PY" scripts/build_tokenizer_sample.py \
    --mixture-config configs/metis13_tokenizer_mix.json \
    --max-samples "$TOKENIZER_SAMPLES" \
    --output-path "$LOCAL_TOKENIZER_SAMPLE_PATH"

  cleanup_local_dir "$stage_root"
}

run_tokenizer_stage() {
  local stage_root="$LOCAL_ROOT/stages/tokenizer_train"
  local stage_hf_home="$stage_root/hf"
  local stage_tmp="$stage_root/tmp"

  rm -rf "$stage_root"
  mkdir -p "$stage_hf_home/datasets" "$stage_hf_home/transformers" "$stage_tmp"

  if [[ "$TOKENIZER_MODE" == "sample" ]]; then
    if [[ ! -f "$LOCAL_TOKENIZER_SAMPLE_PATH" ]]; then
      echo "Missing local tokenizer sample for sample-mode tokenizer training: $LOCAL_TOKENIZER_SAMPLE_PATH" >&2
      echo "Either rerun tokenizer_sample first or point METIS13_START_STAGE back to tokenizer_sample." >&2
      exit 1
    fi
    HF_HOME="$stage_hf_home" \
    HF_DATASETS_CACHE="$stage_hf_home/datasets" \
    TRANSFORMERS_CACHE="$stage_hf_home/transformers" \
    TMPDIR="$stage_tmp" \
    TMP="$stage_tmp" \
    TEMP="$stage_tmp" \
    "$PY" scripts/train_tokenizer.py \
      --jsonl-path "$LOCAL_TOKENIZER_SAMPLE_PATH" \
      --max-samples "$TOKENIZER_SAMPLES" \
      --vocab-size 8192 \
      --output-dir "$LOCAL_HF_ASSETS_DIR"
  else
    HF_HOME="$stage_hf_home" \
    HF_DATASETS_CACHE="$stage_hf_home/datasets" \
    TRANSFORMERS_CACHE="$stage_hf_home/transformers" \
    TMPDIR="$stage_tmp" \
    TMP="$stage_tmp" \
    TEMP="$stage_tmp" \
    "$PY" scripts/train_tokenizer.py \
      --mixture-config configs/metis13_tokenizer_mix.json \
      --max-samples "$TOKENIZER_SAMPLES" \
      --vocab-size 8192 \
      --output-dir "$LOCAL_HF_ASSETS_DIR"
  fi

  cleanup_local_dir "$stage_root"
}

run_pretrain_stage() {
  local stage_root="$LOCAL_ROOT/stages/pretrain"
  local stage_hf_home="$stage_root/hf"
  local stage_tmp="$stage_root/tmp"

  rm -rf "$stage_root"
  mkdir -p "$stage_hf_home/datasets" "$stage_hf_home/transformers" "$stage_tmp"

  HF_HOME="$stage_hf_home" \
  HF_DATASETS_CACHE="$stage_hf_home/datasets" \
  TRANSFORMERS_CACHE="$stage_hf_home/transformers" \
  TMPDIR="$stage_tmp" \
  TMP="$stage_tmp" \
  TEMP="$stage_tmp" \
  "$PY" scripts/prepare_streaming_data.py \
    --mixture-config configs/metis13_pretrain_mix.json \
    --tokenizer-path "$LOCAL_HF_ASSETS_DIR/tokenizer.json" \
    --output-dir "$LOCAL_PRETRAIN_DIR" \
    --max-docs "$PRETRAIN_MAX_DOCS" \
    --target-train-tokens "$PRETRAIN_TARGET_TOKENS" \
    --target-val-tokens "$PRETRAIN_VAL_TOKENS" \
    --val-ratio "$PRETRAIN_VAL_RATIO" \
    --encode-batch-size "$PRETRAIN_ENCODE_BATCH"

  cleanup_local_dir "$stage_root"
}

run_sft_stage() {
  local stage_name="$1"
  local mixture_config="$2"
  local output_dir="$3"
  local total_examples="$4"
  shift 4
  local stage_root="$SFT_SCRATCH_ROOT/$stage_name"
  local stage_hf_home="$stage_root/hf"
  local stage_tmp="$stage_root/tmp"

  rm -rf "$stage_root"
  mkdir -p "$stage_hf_home/datasets" "$stage_hf_home/transformers" "$stage_tmp"

  HF_HOME="$stage_hf_home" \
  HF_DATASETS_CACHE="$stage_hf_home/datasets" \
  TRANSFORMERS_CACHE="$stage_hf_home/transformers" \
  TMPDIR="$stage_tmp" \
  TMP="$stage_tmp" \
  TEMP="$stage_tmp" \
  "$PY" scripts/prepare_metis13_sft_data.py \
    --mixture-config "$mixture_config" \
    --output-dir "$output_dir" \
    --total-examples "$total_examples" \
    "$@"

  cleanup_local_dir "$stage_root"
}

require_shared_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $description on shared volume: $path" >&2
    exit 1
  fi
}

require_shared_dir_file() {
  local dir="$1"
  local rel="$2"
  local description="$3"
  require_shared_file "$dir/$rel" "$description"
}

ensure_local_hf_assets() {
  if [[ -f "$LOCAL_HF_ASSETS_DIR/tokenizer.json" && -f "$LOCAL_HF_ASSETS_DIR/config.json" ]]; then
    return 0
  fi
  require_shared_dir_file "$HF_ASSETS_DIR" "tokenizer.json" "Metis-1.3 HF tokenizer assets"
  require_shared_dir_file "$HF_ASSETS_DIR" "config.json" "Metis-1.3 HF config assets"
  log "Hydrating HF assets from shared volume to local disk."
  sync_dir_from_shared "$HF_ASSETS_DIR" "$LOCAL_HF_ASSETS_DIR"
}

ensure_local_pretrain_data() {
  if [[ -f "$LOCAL_PRETRAIN_DIR/meta.json" ]]; then
    return 0
  fi
  require_shared_dir_file "$PRETRAIN_DIR" "meta.json" "Metis-1.3 pretraining data"
  log "Hydrating pretraining data from shared volume to local disk."
  sync_dir_from_shared "$PRETRAIN_DIR" "$LOCAL_PRETRAIN_DIR"
}

ensure_local_chat_data() {
  if [[ -f "$LOCAL_CHAT_DIR/meta.json" ]]; then
    return 0
  fi
  require_shared_dir_file "$CHAT_DIR" "meta.json" "Metis-1.3 chat SFT data"
  log "Hydrating chat SFT data from shared volume to local disk."
  sync_dir_from_shared "$CHAT_DIR" "$LOCAL_CHAT_DIR"
}

ensure_local_reasoning_data() {
  if [[ -f "$LOCAL_REASONING_DIR/meta.json" ]]; then
    return 0
  fi
  require_shared_dir_file "$REASONING_DIR" "meta.json" "Metis-1.3 reasoning SFT data"
  log "Hydrating reasoning SFT data from shared volume to local disk."
  sync_dir_from_shared "$REASONING_DIR" "$LOCAL_REASONING_DIR"
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

acquire_lock
write_status bootstrapping
log "CPU prep role: $CPU_ROLE"
log "Shared root: $SHARED_ROOT"
log "Local root:  $LOCAL_ROOT"

ensure_python_runtime

PRETRAIN_VAL_TOKENS="$("$PY" - <<'PY' "$PRETRAIN_TARGET_TOKENS" "$PRETRAIN_VAL_RATIO"
import math
import sys

train_tokens = int(sys.argv[1])
val_ratio = float(sys.argv[2])
if val_ratio == 0:
    print(0)
else:
    print(max(1, math.ceil(train_tokens * val_ratio / (1.0 - val_ratio))))
PY
)"

mkdir -p "$CACHE_ROOT" "$TMP_ROOT" "$HF_ASSETS_DIR" "$PRETRAIN_DIR" "$CHAT_DIR" "$REASONING_DIR" "$(dirname "$PLAN_PATH")"
mkdir -p "$LOCAL_HF_ASSETS_DIR" "$LOCAL_PRETRAIN_DIR" "$LOCAL_CHAT_DIR" "$LOCAL_REASONING_DIR" "$(dirname "$LOCAL_PLAN_PATH")" "$(dirname "$LOCAL_TOKENIZER_SAMPLE_PATH")"

export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"
export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if should_run_stage setup; then
  write_status setup
  ensure_stage_env pretrain "$PRETRAIN_REQUIREMENTS_FILE"
  mark_stage_complete setup
fi

if should_run_stage tokenizer_sample; then
  write_status tokenizer_sample
  ensure_stage_env pretrain "$PRETRAIN_REQUIREMENTS_FILE"
  if [[ "$TOKENIZER_MODE" == "sample" ]]; then
    log "Prebuilding tokenizer sample on local disk at $LOCAL_TOKENIZER_SAMPLE_PATH."
    run_tokenizer_sample_stage
  else
    log "Skipping tokenizer sample stage because METIS13_TOKENIZER_MODE=stream."
  fi
  mark_stage_complete tokenizer_sample
fi

if should_run_stage tokenizer; then
  write_status tokenizer
  ensure_stage_env pretrain "$PRETRAIN_REQUIREMENTS_FILE"
  log "Training tokenizer on local disk into $LOCAL_HF_ASSETS_DIR using mode=$TOKENIZER_MODE."
  run_tokenizer_stage
  sync_dir_to_shared "$LOCAL_HF_ASSETS_DIR" "$HF_ASSETS_DIR"
  mark_stage_complete tokenizer
fi

if should_run_stage hf_assets; then
  write_status hf_assets
  ensure_stage_env pretrain "$PRETRAIN_REQUIREMENTS_FILE"
  log "Rendering Metis-1.3 config assets."
  "$PY" scripts/render_metis13_hf_assets.py \
    --manifest configs/metis13_manifest.json \
    --tokenizer-dir "$LOCAL_HF_ASSETS_DIR" \
    --output-dir "$LOCAL_HF_ASSETS_DIR"
  sync_dir_to_shared "$LOCAL_HF_ASSETS_DIR" "$HF_ASSETS_DIR"
  cleanup_tokenizer_sample
  mark_stage_complete hf_assets
fi

if should_run_stage pretrain_data; then
  write_status pretrain_data
  ensure_stage_env pretrain "$PRETRAIN_REQUIREMENTS_FILE"
  ensure_local_hf_assets
  log "Preparing pretraining memmaps on local disk."
  run_pretrain_stage
  sync_dir_to_shared "$LOCAL_PRETRAIN_DIR" "$PRETRAIN_DIR"
  mark_stage_complete pretrain_data
fi

if should_run_stage chat_sft_data; then
  write_status chat_sft_data
  ensure_stage_env sft "$SFT_REQUIREMENTS_FILE"
  cleanup_transient_hf_cache
  log "Preparing cleaned chat SFT data on local disk."
  run_sft_stage chat configs/metis13_chat_mix.json "$LOCAL_CHAT_DIR" "$CHAT_EXAMPLES" \
    --max-user-chars 1600 \
    --max-assistant-chars 1100 \
    --max-think-chars 420 \
    --max-answer-chars 380 \
    --min-user-chars 8 \
    --min-assistant-chars 18 \
    --min-user-alpha-ratio 0.46 \
    --min-assistant-alpha-ratio 0.48 \
    --max-urls 1 \
    --max-code-fences 0
  sync_dir_to_shared "$LOCAL_CHAT_DIR" "$CHAT_DIR"
  mark_stage_complete chat_sft_data
fi

if should_run_stage reasoning_sft_data; then
  write_status reasoning_sft_data
  ensure_stage_env sft "$SFT_REQUIREMENTS_FILE"
  cleanup_transient_hf_cache
  log "Preparing cleaned reasoning SFT data on local disk."
  run_sft_stage reasoning configs/metis13_reasoning_mix.json "$LOCAL_REASONING_DIR" "$REASONING_EXAMPLES" \
    --max-user-chars 1700 \
    --max-assistant-chars 1400 \
    --max-think-chars 520 \
    --max-answer-chars 420 \
    --min-user-chars 8 \
    --min-assistant-chars 20 \
    --min-user-alpha-ratio 0.42 \
    --min-assistant-alpha-ratio 0.4 \
    --max-urls 1 \
    --max-code-fences 0
  cleanup_transient_hf_cache
  sync_dir_to_shared "$LOCAL_REASONING_DIR" "$REASONING_DIR"
  mark_stage_complete reasoning_sft_data
fi

if should_run_stage planning; then
  write_status planning
  ensure_local_pretrain_data
  ensure_local_chat_data
  ensure_local_reasoning_data
  log "Writing derived plan from local metadata, then syncing to shared volume."
  "$PY" scripts/plan_metis13.py \
    --manifest configs/metis13_manifest.json \
    --pretrain-meta "$LOCAL_PRETRAIN_DIR/meta.json" \
    --chat-meta "$LOCAL_CHAT_DIR/meta.json" \
    --reasoning-meta "$LOCAL_REASONING_DIR/meta.json" \
    --output-path "$LOCAL_PLAN_PATH"
  sync_file_to_shared "$LOCAL_PLAN_PATH" "$PLAN_PATH"
  mark_stage_complete planning
fi

mkdir -p "$ROOT_DIR/artifacts" "$ROOT_DIR/data"
ln -sfn "$HF_ASSETS_DIR" "$ROOT_DIR/artifacts/metis13_hf_assets"
ln -sfn "$PRETRAIN_DIR" "$ROOT_DIR/data/metis13_base"
ln -sfn "$CHAT_DIR" "$ROOT_DIR/data/metis13_chat_sft"
ln -sfn "$REASONING_DIR" "$ROOT_DIR/data/metis13_reasoning_sft"

mark_stage_complete complete
log "Metis-1.3 CPU prep complete for role $CPU_ROLE."
echo "Metis-1.3 CPU prep complete for role $CPU_ROLE."
echo "Shared root: $SHARED_ROOT"
if [[ "$CPU_ROLE" == "memory" ]]; then
  echo "HF assets:   $HF_ASSETS_DIR"
else
  echo "Plan file:   $PLAN_PATH"
fi
