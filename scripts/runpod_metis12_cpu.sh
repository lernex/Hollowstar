#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/runpod_metis12_cpu.sh

Environment variables:
  METIS12_SHARED_ROOT            Shared network-volume root. Default: <repo>/.runpod/metis12
  METIS12_FORCE_UNLOCK           Set to 1 to clear a stale CPU prep lock on the shared volume.
  METIS12_START_STAGE            Optional stage to start from. Default: setup
  METIS12_KEEP_TOKENIZER_SAMPLE  Set to 1 to keep the large tokenizer sample after tokenizer training.
  METIS12_TOKENIZER_SAMPLES      Tokenizer documents to sample. Default: 5000000
  METIS12_PRETRAIN_MAX_DOCS      Upper bound on streamed docs during pretrain prep. Default: 8000000
  METIS12_PRETRAIN_TARGET_TOKENS Train-token target for base pretraining. Default: 4000000000
  METIS12_PRETRAIN_VAL_RATIO     Validation holdout ratio for base prep. Default: 0.01
  METIS12_PRETRAIN_ENCODE_BATCH  Docs to tokenize per encode_batch call. Default: 128
  METIS12_CHAT_EXAMPLES          Target chat SFT examples. Default: 400000
  METIS12_REASONING_EXAMPLES     Target reasoning SFT examples. Default: 120000
  METIS12_SFT_SCRATCH_ROOT       Local scratch root for SFT dataset cache/temp files. Default: /tmp/metis12_sft
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

  echo "No suitable Python interpreter found for Metis-1.2 CPU prep." >&2
  return 1
}

SHARED_ROOT="${METIS12_SHARED_ROOT:-$ROOT_DIR/.runpod/metis12}"
CACHE_ROOT="$SHARED_ROOT/cache"
TMP_ROOT="$CACHE_ROOT/tmp"
HF_ASSETS_DIR="$SHARED_ROOT/hf_assets/metis12"
PRETRAIN_DIR="$SHARED_ROOT/data/metis12_base"
CHAT_DIR="$SHARED_ROOT/data/metis12_chat_sft"
REASONING_DIR="$SHARED_ROOT/data/metis12_reasoning_sft"
PLAN_PATH="$SHARED_ROOT/plans/metis12_plan.json"
TOKENIZER_SAMPLE_PATH="$SHARED_ROOT/samples/metis12_tokenizer_sample.jsonl"
STATE_ROOT="$SHARED_ROOT/state"
LOCK_DIR="$STATE_ROOT/metis12_cpu_prep.lock"
STAGE_PATH="$STATE_ROOT/metis12_cpu_prep_stage.txt"
STATUS_PATH="$STATE_ROOT/metis12_cpu_prep_status.env"
LAST_COMPLETED_STAGE_PATH="$STATE_ROOT/metis12_cpu_prep_last_completed_stage.txt"
HOST_NAME="$(hostname -s 2>/dev/null || hostname || echo unknown)"

TOKENIZER_SAMPLES="${METIS12_TOKENIZER_SAMPLES:-5000000}"
PRETRAIN_MAX_DOCS="${METIS12_PRETRAIN_MAX_DOCS:-8000000}"
PRETRAIN_TARGET_TOKENS="${METIS12_PRETRAIN_TARGET_TOKENS:-4000000000}"
PRETRAIN_VAL_RATIO="${METIS12_PRETRAIN_VAL_RATIO:-0.01}"
PRETRAIN_ENCODE_BATCH="${METIS12_PRETRAIN_ENCODE_BATCH:-128}"
CHAT_EXAMPLES="${METIS12_CHAT_EXAMPLES:-400000}"
REASONING_EXAMPLES="${METIS12_REASONING_EXAMPLES:-120000}"
SFT_SCRATCH_ROOT="${METIS12_SFT_SCRATCH_ROOT:-/tmp/metis12_sft}"
START_STAGE="${METIS12_START_STAGE:-setup}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
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
      echo "Unknown METIS12_START_STAGE: $1" >&2
      exit 1
      ;;
  esac
}

should_run_stage() {
  local target="$1"
  [[ "$(stage_index "$target")" -ge "$(stage_index "$START_STAGE")" ]]
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
METIS12_STAGE=$stage
METIS12_UPDATED_AT=$(timestamp)
METIS12_HOST=$HOST_NAME
METIS12_PID=$$
METIS12_SCRIPT=runpod_metis12_cpu.sh
METIS12_SHARED_ROOT=$SHARED_ROOT
METIS12_LAST_COMPLETED_STAGE=$last_completed
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
LOCK_SCRIPT=runpod_metis12_cpu.sh
EOF
    return 0
  fi

  if [[ -f "$LOCK_DIR/owner.env" ]]; then
    # shellcheck disable=SC1090
    source "$LOCK_DIR/owner.env"
    if [[ "${LOCK_HOST:-}" == "$HOST_NAME" && -n "${LOCK_PID:-}" ]] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
      log "Removing stale local CPU prep lock from pid ${LOCK_PID}."
      rm -rf "$LOCK_DIR"
    elif [[ "${METIS12_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of existing CPU prep lock."
      rm -rf "$LOCK_DIR"
    else
      cat >&2 <<EOF
Existing Metis-1.2 CPU prep lock detected at $LOCK_DIR
Lock host: ${LOCK_HOST:-unknown}
Lock pid:  ${LOCK_PID:-unknown}
Lock time: ${LOCK_CREATED_AT:-unknown}

If that run is no longer real, rerun with METIS12_FORCE_UNLOCK=1.
EOF
      exit 1
    fi
  else
    if [[ "${METIS12_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of malformed CPU prep lock."
      rm -rf "$LOCK_DIR"
    else
      echo "Malformed CPU prep lock at $LOCK_DIR. Rerun with METIS12_FORCE_UNLOCK=1 to clear it." >&2
      exit 1
    fi
  fi

  mkdir "$LOCK_DIR"
  cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=runpod_metis12_cpu.sh
EOF
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

cleanup_tokenizer_sample() {
  if [[ "${METIS12_KEEP_TOKENIZER_SAMPLE:-0}" == "1" ]]; then
    log "Keeping tokenizer sample because METIS12_KEEP_TOKENIZER_SAMPLE=1."
    return 0
  fi

  local removed=0
  if [[ -f "$TOKENIZER_SAMPLE_PATH" ]]; then
    rm -f "$TOKENIZER_SAMPLE_PATH"
    removed=1
  fi
  if [[ -f "${TOKENIZER_SAMPLE_PATH}.meta.json" ]]; then
    rm -f "${TOKENIZER_SAMPLE_PATH}.meta.json"
    removed=1
  fi
  if [[ "$removed" == "1" ]]; then
    log "Removed tokenizer sample artifacts to recover shared-volume space."
  fi
}

cleanup_transient_hf_cache() {
  local removed=0
  local cache_path

  for cache_path in \
    "$HF_HOME/hub/datasets--HuggingFaceTB--smoltalk2" \
    "$HF_HOME/hub/datasets--HuggingFaceTB--smoltalk2_everyday_convs_think" \
    "$HF_HOME/hub/datasets--open-thoughts--OpenThoughts-114k"
  do
    if [[ -e "$cache_path" ]]; then
      rm -rf "$cache_path"
      removed=1
    fi
  done

  if [[ "$removed" == "1" ]]; then
    log "Removed transient HF dataset caches to recover shared-volume space."
  fi
}

run_sft_stage() {
  local stage_name="$1"
  local mixture_config="$2"
  local output_dir="$3"
  local total_examples="$4"
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
  "$PY" scripts/prepare_metis12_sft_data.py \
    --mixture-config "$mixture_config" \
    --output-dir "$output_dir" \
    --total-examples "$total_examples"
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

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  log "Creating fresh virtualenv with $(choose_bootstrap_python)."
  "$(choose_bootstrap_python)" -m venv .venv
fi

PY="$ROOT_DIR/.venv/bin/python"

PRETRAIN_VAL_TOKENS="$("$PY" - <<'PY' "$PRETRAIN_TARGET_TOKENS" "$PRETRAIN_VAL_RATIO"
import math
import sys

train_tokens = int(sys.argv[1])
val_ratio = float(sys.argv[2])
if not 0 <= val_ratio < 1:
    raise SystemExit("METIS12_PRETRAIN_VAL_RATIO must be in [0, 1).")

if val_ratio == 0:
    print(0)
else:
    print(max(1, math.ceil(train_tokens * val_ratio / (1.0 - val_ratio))))
PY
)"

mkdir -p "$CACHE_ROOT" "$TMP_ROOT" "$HF_ASSETS_DIR" "$PRETRAIN_DIR" "$CHAT_DIR" "$REASONING_DIR" "$(dirname "$PLAN_PATH")"

export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="${TMPDIR:-$TMP_ROOT}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if should_run_stage setup; then
  write_status setup
  log "Installing CPU prep dependencies with TMPDIR=$TMPDIR."
  "$PY" -m pip install --upgrade pip 'setuptools<82' wheel
  "$PY" -m pip install -r requirements-cpu-prep.txt
  mark_stage_complete setup
fi

if should_run_stage tokenizer_sample; then
  write_status tokenizer_sample
  log "Prebuilding tokenizer sample at $TOKENIZER_SAMPLE_PATH."
  "$PY" scripts/build_tokenizer_sample.py \
    --mixture-config configs/metis12_tokenizer_mix.json \
    --max-samples "$TOKENIZER_SAMPLES" \
    --output-path "$TOKENIZER_SAMPLE_PATH"
  mark_stage_complete tokenizer_sample
fi

if should_run_stage tokenizer; then
  write_status tokenizer
  log "Training tokenizer from prebuilt sample into $HF_ASSETS_DIR."
  "$PY" scripts/train_tokenizer.py \
    --jsonl-path "$TOKENIZER_SAMPLE_PATH" \
    --max-samples "$TOKENIZER_SAMPLES" \
    --vocab-size 8192 \
    --output-dir "$HF_ASSETS_DIR"
  mark_stage_complete tokenizer
fi

if should_run_stage hf_assets; then
  write_status hf_assets
  log "Rendering HF assets."
  "$PY" scripts/render_metis12_hf_assets.py \
    --manifest configs/metis12_manifest.json \
    --tokenizer-dir "$HF_ASSETS_DIR" \
    --output-dir "$HF_ASSETS_DIR"

  cleanup_tokenizer_sample
  mark_stage_complete hf_assets
fi

if should_run_stage pretrain_data; then
  write_status pretrain_data
  log "Preparing pretraining memmaps."
  "$PY" scripts/prepare_streaming_data.py \
    --mixture-config configs/metis12_pretrain_mix.json \
    --tokenizer-path "$HF_ASSETS_DIR/tokenizer.json" \
    --output-dir "$PRETRAIN_DIR" \
    --max-docs "$PRETRAIN_MAX_DOCS" \
    --target-train-tokens "$PRETRAIN_TARGET_TOKENS" \
    --target-val-tokens "$PRETRAIN_VAL_TOKENS" \
    --val-ratio "$PRETRAIN_VAL_RATIO" \
    --encode-batch-size "$PRETRAIN_ENCODE_BATCH"
  mark_stage_complete pretrain_data
fi

if should_run_stage chat_sft_data; then
  write_status chat_sft_data
  cleanup_transient_hf_cache
  log "Preparing chat SFT data."
  run_sft_stage chat configs/metis12_chat_mix.json "$CHAT_DIR" "$CHAT_EXAMPLES"
  mark_stage_complete chat_sft_data
fi

if should_run_stage reasoning_sft_data; then
  write_status reasoning_sft_data
  cleanup_transient_hf_cache
  log "Preparing reasoning SFT data."
  run_sft_stage reasoning configs/metis12_reasoning_mix.json "$REASONING_DIR" "$REASONING_EXAMPLES"
  cleanup_transient_hf_cache
  mark_stage_complete reasoning_sft_data
fi

if should_run_stage planning; then
  write_status planning
  log "Writing derived plan to $PLAN_PATH."
  "$PY" scripts/plan_metis12.py \
    --manifest configs/metis12_manifest.json \
    --pretrain-meta "$PRETRAIN_DIR/meta.json" \
    --chat-meta "$CHAT_DIR/meta.json" \
    --reasoning-meta "$REASONING_DIR/meta.json" \
    --output-path "$PLAN_PATH"
  mark_stage_complete planning
fi

mkdir -p "$ROOT_DIR/artifacts" "$ROOT_DIR/data"
ln -sfn "$HF_ASSETS_DIR" "$ROOT_DIR/artifacts/metis12_hf_assets"
ln -sfn "$PRETRAIN_DIR" "$ROOT_DIR/data/metis12_base"
ln -sfn "$CHAT_DIR" "$ROOT_DIR/data/metis12_chat_sft"
ln -sfn "$REASONING_DIR" "$ROOT_DIR/data/metis12_reasoning_sft"

mark_stage_complete complete
log "Metis-1.2 CPU prep complete."
echo "Metis-1.2 CPU prep complete."
echo "Shared root: $SHARED_ROOT"
echo "Plan file:   $PLAN_PATH"
