#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/metis15_cpu_prep.sh

Environment variables:
  METIS15_PREP_MODE                One of: runpod, aws. Default: runpod
  METIS15_MANIFEST                 Metis-1.5 manifest. Default: configs/metis15_manifest.json
  METIS15_S3_ROOT                  Required durable root, e.g. s3://your-bucket/metis15
  METIS15_S3_CREATE_BUCKET         Set to 1 to create the bucket if missing
  METIS15_S3_REGION                Optional bucket region override
  METIS15_LOCAL_ROOT               Local NVMe scratch root. Default depends on mode
  METIS15_FORCE_UNLOCK             Set to 1 to clear a stale local CPU prep lock
  METIS15_START_STAGE              Optional stage to start from. Default: setup
  METIS15_STOP_AFTER_STAGE         Optional stage to stop after, useful for tokenizer-first prep
  METIS15_KEEP_TOKENIZER_SAMPLE    Set to 1 to keep the tokenizer sample JSONL locally
  METIS15_LOCAL_DATASETS           Set to 1 to force local HF-cached datasets instead of streaming. Default: 1
  METIS15_INCREMENTAL_HF_SHARDS    Set to 1 to process large HF parquet datasets one shard at a time. Default: 1
  METIS15_NORMALIZE_WORKERS        Parallel source workers for normalized stages. Default depends on mode
  METIS15_SOURCE_PARTITIONS        Max parallel partitions per large normalized source. Default: 1
  METIS15_PARTITION_TARGET_DOCS    Target docs per normalized source partition. Default: 250000
  METIS15_PARTITION_MIN_DOCS       Smallest source target docs eligible for partitioning. Default: 300000
  METIS15_NORMALIZED_ZSTD_LEVEL    zstd level for normalized JSONL shards. Default: 6
  METIS15_PURGE_HF_CACHE           Set to 1 to delete local HF dataset cache between normalized sources. Default: 0
  METIS15_PARQUET_CACHE_GB         Per-worker parquet prefetch cache limit in GB. Default depends on mode
  METIS15_PARQUET_PREFETCH_COUNT   Max queued parquet shards per worker. Default depends on mode
  METIS15_TOKENIZER_SAMPLES        Tokenizer sample rows from normalized shards. Default: manifest tokenizer.sample_docs
  METIS15_TOKENIZER_VOCAB_SIZE     Tokenizer vocab size. Default: manifest tokenizer.vocab_size
  METIS15_TOKENIZER_MIN_FREQUENCY  Tokenizer min_frequency. Default: manifest tokenizer.min_frequency
  METIS15_ENCODE_BATCH_SIZE        Tokenizer encode batch size for memmap packing. Default: 1024
  METIS15_PRETRAIN_MAX_DOCS        Max normalized docs for the base pretrain mix. Default: 25000000
  METIS15_CONTINUED_MAX_DOCS       Max normalized docs for the continued-pretrain mix. Default: 3000000
  METIS15_NORMALIZED_SHARD_DOCS    Docs per normalized .jsonl.zst shard. Default: 50000
  METIS15_CHAT_EXAMPLES            Override chat SFT examples
  METIS15_REASONING_EXAMPLES       Override reasoning SFT examples
  METIS15_REWARD_EXAMPLES          Override reward-model preference pairs
  METIS15_DPO_PAIRS                Override DPO preference pairs

Stages:
  setup
  normalized_pretrain
  tokenizer_sample
  tokenizer_assets
  pretrain_data
  normalized_continued
  continued_pretrain_data
  chat_sft_data
  reasoning_sft_data
  reward_prefs
  dpo_prefs
  planning
  complete
EOF
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
PREP_MODE="${METIS15_PREP_MODE:-runpod}"
S3_ROOT="${METIS15_S3_ROOT:-}"

case "$PREP_MODE" in
  runpod)
    DEFAULT_LOCAL_ROOT="/tmp/metis15_cpu_prep"
    ;;
  aws)
    DEFAULT_LOCAL_ROOT="/mnt/metis15_cpu_prep"
    ;;
  *)
    echo "Unsupported METIS15_PREP_MODE: $PREP_MODE" >&2
    exit 1
    ;;
esac

if [[ -z "$S3_ROOT" ]]; then
  echo "METIS15_S3_ROOT is required, for example s3://your-bucket/metis15" >&2
  exit 1
fi

LOCAL_ROOT="${METIS15_LOCAL_ROOT:-$DEFAULT_LOCAL_ROOT}"
HF_CACHE_ROOT="$LOCAL_ROOT/cache/hf"
PARQUET_CACHE_ROOT="$LOCAL_ROOT/cache/parquet"
STATE_ROOT="$LOCAL_ROOT/state"
LOCK_DIR="$STATE_ROOT/metis15_cpu_prep.lock"
STAGE_PATH="$STATE_ROOT/metis15_cpu_prep_stage.txt"
STATUS_PATH="$STATE_ROOT/metis15_cpu_prep_status.env"
LAST_COMPLETED_STAGE_PATH="$STATE_ROOT/metis15_cpu_prep_last_completed_stage.txt"
RUN_LOG_PATH="$LOCAL_ROOT/run.log"
HOST_NAME="$(hostname -s 2>/dev/null || hostname || echo unknown)"
PY="$ROOT_DIR/.venv/bin/python"

PRETRAIN_MAX_DOCS="${METIS15_PRETRAIN_MAX_DOCS:-25000000}"
CONTINUED_MAX_DOCS="${METIS15_CONTINUED_MAX_DOCS:-3000000}"
NORMALIZED_SHARD_DOCS="${METIS15_NORMALIZED_SHARD_DOCS:-50000}"
START_STAGE="${METIS15_START_STAGE:-setup}"
STOP_AFTER_STAGE="${METIS15_STOP_AFTER_STAGE:-}"
LOCAL_DATASETS="${METIS15_LOCAL_DATASETS:-1}"
INCREMENTAL_HF_SHARDS="${METIS15_INCREMENTAL_HF_SHARDS:-1}"
case "$PREP_MODE" in
  runpod)
    DEFAULT_NORMALIZE_WORKERS=6
    DEFAULT_PURGE_HF_CACHE=0
    DEFAULT_PARQUET_CACHE_GB=50
    DEFAULT_PARQUET_PREFETCH_COUNT=14
    ;;
  aws)
    DEFAULT_NORMALIZE_WORKERS=4
    DEFAULT_PURGE_HF_CACHE=0
    DEFAULT_PARQUET_CACHE_GB=220
    DEFAULT_PARQUET_PREFETCH_COUNT=16
    ;;
esac
NORMALIZE_WORKERS="${METIS15_NORMALIZE_WORKERS:-$DEFAULT_NORMALIZE_WORKERS}"
SOURCE_PARTITIONS="${METIS15_SOURCE_PARTITIONS:-1}"
PARTITION_TARGET_DOCS="${METIS15_PARTITION_TARGET_DOCS:-250000}"
PARTITION_MIN_DOCS="${METIS15_PARTITION_MIN_DOCS:-300000}"
NORMALIZED_ZSTD_LEVEL="${METIS15_NORMALIZED_ZSTD_LEVEL:-6}"
PURGE_HF_CACHE="${METIS15_PURGE_HF_CACHE:-$DEFAULT_PURGE_HF_CACHE}"
PARQUET_CACHE_GB="${METIS15_PARQUET_CACHE_GB:-$DEFAULT_PARQUET_CACHE_GB}"
PARQUET_PREFETCH_COUNT="${METIS15_PARQUET_PREFETCH_COUNT:-$DEFAULT_PARQUET_PREFETCH_COUNT}"

NORMALIZED_PRETRAIN_DIR="$LOCAL_ROOT/normalized/pretrain"
NORMALIZED_CONTINUED_DIR="$LOCAL_ROOT/normalized/continued"
TOKENIZER_SAMPLE_PATH="$LOCAL_ROOT/samples/metis15_tokenizer_sample.jsonl"
HF_ASSETS_DIR="$LOCAL_ROOT/artifacts/metis15_hf_assets"
PRETRAIN_DIR="$LOCAL_ROOT/data/metis15_base"
CONTINUED_DIR="$LOCAL_ROOT/data/metis15_continued_pretrain"
CHAT_DIR="$LOCAL_ROOT/data/metis15_chat_sft"
REASONING_DIR="$LOCAL_ROOT/data/metis15_reasoning_sft"
REWARD_PREF_DIR="$LOCAL_ROOT/data/metis15_reward_prefs"
DPO_PREF_DIR="$LOCAL_ROOT/data/metis15_dpo_prefs"
PLAN_PATH="$LOCAL_ROOT/plans/metis15_plan.json"
STORAGE_MANIFEST_PATH="$LOCAL_ROOT/plans/metis15_storage_manifest.json"
IDENTITY_SFT_LOCAL="${METIS15_IDENTITY_SFT_LOCAL:-$ROOT_DIR/examples/metis15_identity_sft.jsonl}"

NORMALIZED_PRETRAIN_S3="$S3_ROOT/normalized-shards/pretrain"
NORMALIZED_CONTINUED_S3="$S3_ROOT/normalized-shards/continued"
TOKENIZER_S3="$S3_ROOT/tokenizer"
PRETRAIN_S3="$S3_ROOT/pretrain-shards/base"
CONTINUED_S3="$S3_ROOT/pretrain-shards/continued"
CHAT_S3="$S3_ROOT/chat-sft"
REASONING_S3="$S3_ROOT/reasoning-sft"
REWARD_PREF_S3="$S3_ROOT/reward-prefs"
DPO_PREF_S3="$S3_ROOT/dpo-prefs"
MANIFESTS_S3="$S3_ROOT/manifests"
IDENTITY_SFT_S3="${METIS15_IDENTITY_SFT_S3:-$S3_ROOT/manual-sft/metis15_identity_sft.jsonl}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

log_shell_error() {
  local exit_code=$?
  printf '[%s] ERROR exit=%s line=%s command=%s\n' \
    "$(timestamp)" \
    "$exit_code" \
    "${BASH_LINENO[0]:-unknown}" \
    "${BASH_COMMAND:-unknown}" >&2
  exit "$exit_code"
}

choose_bootstrap_python() {
  local candidate
  for candidate in "${BOOTSTRAP_PYTHON:-}" python3.12 python3.11 python3.10 python3 python; do
    if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "No suitable Python interpreter found for Metis-1.5 CPU prep." >&2
  return 1
}

ensure_python_runtime() {
  if [[ ! -x "$PY" ]]; then
    rm -rf "$ROOT_DIR/.venv"
    log "Creating fresh virtualenv with $(choose_bootstrap_python)."
    "$(choose_bootstrap_python)" -m venv "$ROOT_DIR/.venv"
  fi
}

install_cpu_dependencies() {
  ensure_python_runtime
  "$PY" -m pip install --upgrade pip 'setuptools<82' wheel
  "$PY" -m pip install -r "$ROOT_DIR/requirements-cpu-pretrain.txt"
  "$PY" -m pip install -r "$ROOT_DIR/requirements-cpu-sft.txt"
}

ensure_resume_dependencies() {
  if [[ "$(stage_index "$START_STAGE")" -le "$(stage_index setup)" ]]; then
    return 0
  fi
  if "$PY" - <<'PY' >/dev/null 2>&1
import boto3
import tokenizers
import datasets
PY
  then
    return 0
  fi
  log "Installing CPU prep dependencies for resumed stage $START_STAGE."
  install_cpu_dependencies
}

stage_index() {
  case "$1" in
    setup) echo 0 ;;
    normalized_pretrain) echo 1 ;;
    tokenizer_sample) echo 2 ;;
    tokenizer_assets) echo 3 ;;
    pretrain_data) echo 4 ;;
    normalized_continued) echo 5 ;;
    continued_pretrain_data) echo 6 ;;
    chat_sft_data) echo 7 ;;
    reasoning_sft_data) echo 8 ;;
    reward_prefs) echo 9 ;;
    dpo_prefs) echo 10 ;;
    planning) echo 11 ;;
    complete) echo 12 ;;
    *)
      echo "Unknown Metis-1.5 CPU prep stage: $1" >&2
      exit 1
      ;;
  esac
}

should_run_stage() {
  [[ "$(stage_index "$1")" -ge "$(stage_index "$START_STAGE")" ]]
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
METIS15_STAGE=$stage
METIS15_UPDATED_AT=$(timestamp)
METIS15_HOST=$HOST_NAME
METIS15_PID=$$
METIS15_SCRIPT=metis15_cpu_prep.sh
METIS15_PREP_MODE=$PREP_MODE
METIS15_LOCAL_ROOT=$LOCAL_ROOT
METIS15_S3_ROOT=$S3_ROOT
METIS15_LAST_COMPLETED_STAGE=$last_completed
EOF
}

mark_stage_complete() {
  local stage="$1"
  mkdir -p "$STATE_ROOT"
  printf '%s\n' "$stage" > "$LAST_COMPLETED_STAGE_PATH"
  write_status "$stage"
  if [[ -n "$STOP_AFTER_STAGE" && "$stage" == "$STOP_AFTER_STAGE" && "$stage" != "complete" ]]; then
    log "Reached METIS15_STOP_AFTER_STAGE=$STOP_AFTER_STAGE; stopping cleanly after stage completion."
    exit 0
  fi
}

acquire_lock() {
  mkdir -p "$STATE_ROOT"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=metis15_cpu_prep.sh
EOF
    return 0
  fi

  if [[ -f "$LOCK_DIR/owner.env" ]]; then
    # shellcheck disable=SC1090
    source "$LOCK_DIR/owner.env"
    if [[ "${LOCK_HOST:-}" == "$HOST_NAME" && -n "${LOCK_PID:-}" ]] && ! kill -0 "$LOCK_PID" 2>/dev/null; then
      log "Removing stale local CPU prep lock from pid ${LOCK_PID}."
      rm -rf "$LOCK_DIR"
    elif [[ "${METIS15_FORCE_UNLOCK:-0}" == "1" ]]; then
      log "Forcing unlock of existing CPU prep lock."
      rm -rf "$LOCK_DIR"
    else
      cat >&2 <<EOF
Existing Metis-1.5 CPU prep lock detected at $LOCK_DIR
Lock host: ${LOCK_HOST:-unknown}
Lock pid:  ${LOCK_PID:-unknown}
Lock time: ${LOCK_CREATED_AT:-unknown}

If that run is no longer real, rerun with METIS15_FORCE_UNLOCK=1.
EOF
      exit 1
    fi
  fi

  mkdir "$LOCK_DIR"
  cat > "$LOCK_DIR/owner.env" <<EOF
LOCK_HOST=$HOST_NAME
LOCK_PID=$$
LOCK_CREATED_AT=$(timestamp)
LOCK_SCRIPT=metis15_cpu_prep.sh
EOF
}

cleanup() {
  rm -rf "$LOCK_DIR"
}

run_with_stage_cache() {
  local stage_name="$1"
  shift
  local stage_root="$LOCAL_ROOT/stages/$stage_name"
  local stage_tmp="$stage_root/tmp"
  rm -rf "$stage_root"
  mkdir -p "$HF_CACHE_ROOT/datasets" "$HF_CACHE_ROOT/transformers" "$PARQUET_CACHE_ROOT" "$stage_tmp"
  HF_HOME="$HF_CACHE_ROOT" \
  HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets" \
  TRANSFORMERS_CACHE="$HF_CACHE_ROOT/transformers" \
  METIS_PARQUET_CACHE_ROOT="$PARQUET_CACHE_ROOT" \
  METIS_PARQUET_CACHE_LIMIT_GB="$PARQUET_CACHE_GB" \
  METIS_PARQUET_PREFETCH_COUNT="$PARQUET_PREFETCH_COUNT" \
  TMPDIR="$stage_tmp" \
  TMP="$stage_tmp" \
  TEMP="$stage_tmp" \
  METIS_LOCAL_DATASETS="$LOCAL_DATASETS" \
  METIS_INCREMENTAL_HF_SHARDS="$INCREMENTAL_HF_SHARDS" \
  TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}" \
  "$@"
  rm -rf "$stage_root"
}

upload_dir_to_s3() {
  local local_dir="$1"
  local s3_uri="$2"
  "$PY" "$ROOT_DIR/scripts/s3_artifacts.py" upload-dir --local-dir "$local_dir" --s3-uri "$s3_uri" >/dev/null
}

download_dir_from_s3() {
  local local_dir="$1"
  local s3_uri="$2"
  "$PY" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$s3_uri" --local-dir "$local_dir" >/dev/null
}

upload_file_to_s3() {
  local local_path="$1"
  local s3_uri="$2"
  "$PY" "$ROOT_DIR/scripts/s3_artifacts.py" upload-file --local-path "$local_path" --s3-uri "$s3_uri" >/dev/null
}

ensure_dir_from_s3_if_missing() {
  local local_dir="$1"
  local required_rel="$2"
  local s3_uri="$3"
  local label="$4"
  if [[ -f "$local_dir/$required_rel" ]]; then
    return 0
  fi
  log "Hydrating $label from $s3_uri."
  download_dir_from_s3 "$local_dir" "$s3_uri"
  if [[ ! -f "$local_dir/$required_rel" ]]; then
    echo "Hydrated $label but did not find expected file: $local_dir/$required_rel" >&2
    exit 1
  fi
}

cleanup_tokenizer_sample() {
  if [[ "${METIS15_KEEP_TOKENIZER_SAMPLE:-0}" == "1" ]]; then
    log "Keeping tokenizer sample because METIS15_KEEP_TOKENIZER_SAMPLE=1."
    return 0
  fi
  rm -f "$TOKENIZER_SAMPLE_PATH" "${TOKENIZER_SAMPLE_PATH}.meta.json"
}

mkdir -p "$LOCAL_ROOT"
touch "$RUN_LOG_PATH"
exec > >(tee -a "$RUN_LOG_PATH") 2>&1
trap log_shell_error ERR
trap cleanup EXIT INT TERM

log "==== Metis-1.5 CPU prep start | mode=$PREP_MODE | start_stage=$START_STAGE | s3_root=$S3_ROOT ===="

cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Metis manifest not found: $MANIFEST" >&2
  exit 1
fi

stage_index "$START_STAGE" >/dev/null
if [[ -n "$STOP_AFTER_STAGE" ]]; then
  stage_index "$STOP_AFTER_STAGE" >/dev/null
  if [[ "$(stage_index "$STOP_AFTER_STAGE")" -lt "$(stage_index "$START_STAGE")" ]]; then
    echo "METIS15_STOP_AFTER_STAGE must be the same as or later than METIS15_START_STAGE." >&2
    exit 1
  fi
fi

ensure_python_runtime
ensure_resume_dependencies

eval "$("$PY" - "$MANIFEST" <<'PY'
import json
import math
import os
import shlex
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
data_manifests = manifest["data_manifests"]
strategy = manifest["selected_data_strategy"]

pretrain_mix = data_manifests["pretrain_best_research"]
if strategy.get("pretrain") == "release_clean":
    pretrain_mix = data_manifests["pretrain_release_clean"]

pretrain = manifest["pretrain"]
continued = manifest["continued_pretrain"]
chat = manifest["chat_sft"]
reasoning = manifest["reasoning_sft"]
preference = manifest["preference_optimization"]
data_prep = manifest.get("data_prep", {})

def val_tokens(train_tokens: int, val_ratio: float) -> int:
    if val_ratio <= 0:
        return 0
    return max(1, math.ceil(train_tokens * val_ratio / (1.0 - val_ratio)))

values = {
    "PRETRAIN_MIX": pretrain_mix,
    "CONTINUED_MIX": data_manifests["continued_pretrain"],
    "CHAT_MIX": data_manifests["chat_sft"],
    "REASONING_MIX": data_manifests["reasoning_sft"],
    "PREFERENCE_MIX": data_manifests["preference"],
    "TOKENIZER_VOCAB_SIZE_MANIFEST": int(manifest["tokenizer"]["vocab_size"]),
    "TOKENIZER_SAMPLE_DOCS_MANIFEST": int(manifest["tokenizer"]["sample_docs"]),
    "TOKENIZER_MIN_FREQUENCY_MANIFEST": int(manifest["tokenizer"]["min_frequency"]),
    "PRETRAIN_MAX_DOCS_MANIFEST": int(data_prep.get("pretrain_max_docs", 25000000)),
    "CONTINUED_MAX_DOCS_MANIFEST": int(data_prep.get("continued_max_docs", 3000000)),
    "PRETRAIN_TARGET_TOKENS_MANIFEST": int(pretrain["target_train_tokens"]),
    "PRETRAIN_VAL_TOKENS_MANIFEST": val_tokens(int(pretrain["target_train_tokens"]), float(pretrain["val_ratio"])),
    "CONTINUED_TARGET_TOKENS_MANIFEST": int(continued["target_train_tokens"]),
    "CONTINUED_VAL_TOKENS_MANIFEST": val_tokens(int(continued["target_train_tokens"]), float(continued["val_ratio"])),
    "CHAT_EXAMPLES_MANIFEST": int(chat["target_examples"]),
    "REASONING_EXAMPLES_MANIFEST": int(reasoning["target_examples"]),
    "REASONING_MAX_THINK_CHARS_MANIFEST": int(reasoning.get("recommended_max_think_chars", 320)),
    "REASONING_MAX_ANSWER_CHARS_MANIFEST": int(reasoning.get("recommended_max_answer_chars", 320)),
    "REWARD_EXAMPLES_MANIFEST": int(preference.get("bootstrap_reward_model_examples", preference["reward_model_examples"])),
    "DPO_PAIRS_MANIFEST": int(preference.get("bootstrap_target_pairs", preference["target_pairs"])),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

TOKENIZER_SAMPLES="${METIS15_TOKENIZER_SAMPLES:-$TOKENIZER_SAMPLE_DOCS_MANIFEST}"
TOKENIZER_VOCAB_SIZE="${METIS15_TOKENIZER_VOCAB_SIZE:-$TOKENIZER_VOCAB_SIZE_MANIFEST}"
TOKENIZER_MIN_FREQUENCY="${METIS15_TOKENIZER_MIN_FREQUENCY:-$TOKENIZER_MIN_FREQUENCY_MANIFEST}"
ENCODE_BATCH_SIZE="${METIS15_ENCODE_BATCH_SIZE:-1024}"
PRETRAIN_MAX_DOCS="${METIS15_PRETRAIN_MAX_DOCS:-$PRETRAIN_MAX_DOCS_MANIFEST}"
CONTINUED_MAX_DOCS="${METIS15_CONTINUED_MAX_DOCS:-$CONTINUED_MAX_DOCS_MANIFEST}"
CHAT_EXAMPLES="${METIS15_CHAT_EXAMPLES:-$CHAT_EXAMPLES_MANIFEST}"
REASONING_EXAMPLES="${METIS15_REASONING_EXAMPLES:-$REASONING_EXAMPLES_MANIFEST}"
REASONING_MAX_THINK_CHARS="${METIS15_REASONING_MAX_THINK_CHARS:-$REASONING_MAX_THINK_CHARS_MANIFEST}"
REASONING_MAX_ANSWER_CHARS="${METIS15_REASONING_MAX_ANSWER_CHARS:-$REASONING_MAX_ANSWER_CHARS_MANIFEST}"
REWARD_EXAMPLES="${METIS15_REWARD_EXAMPLES:-$REWARD_EXAMPLES_MANIFEST}"
DPO_PAIRS="${METIS15_DPO_PAIRS:-$DPO_PAIRS_MANIFEST}"

mkdir -p \
  "$STATE_ROOT" \
  "$HF_CACHE_ROOT/datasets" \
  "$HF_CACHE_ROOT/transformers" \
  "$PARQUET_CACHE_ROOT" \
  "$NORMALIZED_PRETRAIN_DIR" \
  "$NORMALIZED_CONTINUED_DIR" \
  "$(dirname "$TOKENIZER_SAMPLE_PATH")" \
  "$HF_ASSETS_DIR" \
  "$PRETRAIN_DIR" \
  "$CONTINUED_DIR" \
  "$CHAT_DIR" \
  "$REASONING_DIR" \
  "$REWARD_PREF_DIR" \
  "$DPO_PREF_DIR" \
  "$(dirname "$PLAN_PATH")"

acquire_lock
write_status bootstrapping

log "Metis-1.5 CPU prep mode: $PREP_MODE"
log "Manifest: $MANIFEST"
log "Local root: $LOCAL_ROOT"
log "S3 root: $S3_ROOT"
if [[ -n "$STOP_AFTER_STAGE" ]]; then
  log "Stop after stage: $STOP_AFTER_STAGE"
fi

if [[ "$(stage_index "$START_STAGE")" -ge "$(stage_index pretrain_data)" ]]; then
  ensure_dir_from_s3_if_missing "$HF_ASSETS_DIR" "tokenizer.json" "$TOKENIZER_S3" "Metis-1.5 tokenizer assets"
  ensure_dir_from_s3_if_missing "$NORMALIZED_PRETRAIN_DIR" "manifest.json" "$NORMALIZED_PRETRAIN_S3" "normalized pretrain shards"
fi

if [[ "$(stage_index "$START_STAGE")" -ge "$(stage_index continued_pretrain_data)" ]]; then
  ensure_dir_from_s3_if_missing "$NORMALIZED_CONTINUED_DIR" "manifest.json" "$NORMALIZED_CONTINUED_S3" "normalized continued-pretrain shards"
fi

if should_run_stage setup; then
  write_status setup
  install_cpu_dependencies
  if [[ "${METIS15_S3_CREATE_BUCKET:-0}" == "1" ]]; then
    log "Ensuring the S3 bucket exists."
    "$PY" "$ROOT_DIR/scripts/s3_artifacts.py" ensure-bucket --s3-uri "$S3_ROOT" --region "${METIS15_S3_REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-}}}" >/dev/null
  fi
  if [[ -f "$IDENTITY_SFT_LOCAL" ]]; then
    log "Uploading manual identity SFT source to $IDENTITY_SFT_S3."
    upload_file_to_s3 "$IDENTITY_SFT_LOCAL" "$IDENTITY_SFT_S3"
  fi
  mark_stage_complete setup
fi

if should_run_stage normalized_pretrain; then
  write_status normalized_pretrain
  log "Building normalized pretrain shards on local NVMe and mirroring them to S3."
  run_with_stage_cache normalized_pretrain \
    "$PY" "$ROOT_DIR/scripts/prepare_normalized_shards.py" \
      --mixture-config "$PRETRAIN_MIX" \
      --output-dir "$NORMALIZED_PRETRAIN_DIR" \
      --max-docs "$PRETRAIN_MAX_DOCS" \
      --shard-docs "$NORMALIZED_SHARD_DOCS" \
      --zstd-level "$NORMALIZED_ZSTD_LEVEL" \
      --workers "$NORMALIZE_WORKERS" \
      --source-partitions "$SOURCE_PARTITIONS" \
      --partition-target-docs "$PARTITION_TARGET_DOCS" \
      --partition-min-docs "$PARTITION_MIN_DOCS" \
      $( [[ "$PURGE_HF_CACHE" == "1" ]] && printf '%s' "--purge-hf-cache-between-sources" ) \
      --s3-prefix "$NORMALIZED_PRETRAIN_S3"
  mark_stage_complete normalized_pretrain
fi

if should_run_stage tokenizer_sample; then
  write_status tokenizer_sample
  log "Building tokenizer sample from normalized pretrain shards."
  "$PY" "$ROOT_DIR/scripts/build_tokenizer_sample.py" \
    --mixture-config "$PRETRAIN_MIX" \
    --normalized-root "$NORMALIZED_PRETRAIN_DIR" \
    --jsonl-glob "shard-*.jsonl.zst" \
    --jsonl-text-field text \
    --max-samples "$TOKENIZER_SAMPLES" \
    --output-path "$TOKENIZER_SAMPLE_PATH"
  upload_file_to_s3 "${TOKENIZER_SAMPLE_PATH}.meta.json" "$MANIFESTS_S3/tokenizer_sample.meta.json"
  mark_stage_complete tokenizer_sample
fi

if should_run_stage tokenizer_assets; then
  write_status tokenizer_assets
  log "Training tokenizer and rendering HF assets."
  run_with_stage_cache tokenizer_assets \
    "$PY" "$ROOT_DIR/scripts/train_tokenizer.py" \
      --jsonl-path "$TOKENIZER_SAMPLE_PATH" \
      --max-samples "$TOKENIZER_SAMPLES" \
      --vocab-size "$TOKENIZER_VOCAB_SIZE" \
      --min-frequency "$TOKENIZER_MIN_FREQUENCY" \
      --output-dir "$HF_ASSETS_DIR"
  "$PY" "$ROOT_DIR/scripts/render_metis13_hf_assets.py" \
    --manifest "$MANIFEST" \
    --tokenizer-dir "$HF_ASSETS_DIR" \
    --output-dir "$HF_ASSETS_DIR"
  "$PY" - "$HF_ASSETS_DIR" <<'PY'
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

assets_dir = Path(sys.argv[1])
tokenizer = Tokenizer.from_file(str(assets_dir / "tokenizer.json"))
config = json.loads((assets_dir / "config.json").read_text(encoding="utf-8"))
tokenizer_size = tokenizer.get_vocab_size()
config_size = int(config["vocab_size"])
if tokenizer_size != config_size:
    raise SystemExit(
        f"Tokenizer/config vocab mismatch before S3 upload: {tokenizer_size} != {config_size}"
    )
print(f"Validated Metis-1.5 tokenizer assets: vocab_size={tokenizer_size}", flush=True)
PY
  upload_dir_to_s3 "$HF_ASSETS_DIR" "$TOKENIZER_S3"
  cleanup_tokenizer_sample
  mark_stage_complete tokenizer_assets
fi

if should_run_stage pretrain_data; then
  write_status pretrain_data
  log "Building base pretraining memmaps from normalized shards."
  run_with_stage_cache pretrain_data \
    "$PY" "$ROOT_DIR/scripts/prepare_streaming_data.py" \
      --mixture-config "$PRETRAIN_MIX" \
      --normalized-root "$NORMALIZED_PRETRAIN_DIR" \
      --jsonl-glob "shard-*.jsonl.zst" \
      --tokenizer-path "$HF_ASSETS_DIR/tokenizer.json" \
      --output-dir "$PRETRAIN_DIR" \
      --max-docs "$PRETRAIN_MAX_DOCS" \
      --target-train-tokens "$PRETRAIN_TARGET_TOKENS_MANIFEST" \
      --target-val-tokens "$PRETRAIN_VAL_TOKENS_MANIFEST" \
      --val-ratio 0.01 \
      --encode-batch-size "$ENCODE_BATCH_SIZE"
  upload_dir_to_s3 "$PRETRAIN_DIR" "$PRETRAIN_S3"
  mark_stage_complete pretrain_data
fi

if should_run_stage normalized_continued; then
  write_status normalized_continued
  log "Building normalized continued-pretrain shards on local NVMe and mirroring them to S3."
  run_with_stage_cache normalized_continued \
    "$PY" "$ROOT_DIR/scripts/prepare_normalized_shards.py" \
      --mixture-config "$CONTINUED_MIX" \
      --output-dir "$NORMALIZED_CONTINUED_DIR" \
      --max-docs "$CONTINUED_MAX_DOCS" \
      --shard-docs "$NORMALIZED_SHARD_DOCS" \
      --zstd-level "$NORMALIZED_ZSTD_LEVEL" \
      --workers "$NORMALIZE_WORKERS" \
      --source-partitions "$SOURCE_PARTITIONS" \
      --partition-target-docs "$PARTITION_TARGET_DOCS" \
      --partition-min-docs "$PARTITION_MIN_DOCS" \
      $( [[ "$PURGE_HF_CACHE" == "1" ]] && printf '%s' "--purge-hf-cache-between-sources" ) \
      --s3-prefix "$NORMALIZED_CONTINUED_S3"
  mark_stage_complete normalized_continued
fi

if should_run_stage continued_pretrain_data; then
  write_status continued_pretrain_data
  log "Clearing parquet prefetch cache before continued-pretraining packing."
  rm -rf "$PARQUET_CACHE_ROOT"/*
  log "Building continued-pretraining memmaps from normalized shards."
  run_with_stage_cache continued_pretrain_data \
    "$PY" "$ROOT_DIR/scripts/prepare_streaming_data.py" \
      --mixture-config "$CONTINUED_MIX" \
      --normalized-root "$NORMALIZED_CONTINUED_DIR" \
      --jsonl-glob "shard-*.jsonl.zst" \
      --tokenizer-path "$HF_ASSETS_DIR/tokenizer.json" \
      --output-dir "$CONTINUED_DIR" \
      --max-docs "$CONTINUED_MAX_DOCS" \
      --target-train-tokens "$CONTINUED_TARGET_TOKENS_MANIFEST" \
      --target-val-tokens "$CONTINUED_VAL_TOKENS_MANIFEST" \
      --val-ratio 0.01 \
      --encode-batch-size "$ENCODE_BATCH_SIZE"
  upload_dir_to_s3 "$CONTINUED_DIR" "$CONTINUED_S3"
  mark_stage_complete continued_pretrain_data
fi

if should_run_stage chat_sft_data; then
  write_status chat_sft_data
  log "Preparing chat SFT JSONL on local NVMe."
  run_with_stage_cache chat_sft_data \
    "$PY" "$ROOT_DIR/scripts/prepare_metis13_sft_data.py" \
      --mixture-config "$CHAT_MIX" \
      --output-dir "$CHAT_DIR" \
      --total-examples "$CHAT_EXAMPLES"
  upload_dir_to_s3 "$CHAT_DIR" "$CHAT_S3"
  mark_stage_complete chat_sft_data
fi

if should_run_stage reasoning_sft_data; then
  write_status reasoning_sft_data
  log "Preparing reasoning SFT JSONL on local NVMe."
  run_with_stage_cache reasoning_sft_data \
    "$PY" "$ROOT_DIR/scripts/prepare_metis13_sft_data.py" \
      --mixture-config "$REASONING_MIX" \
      --output-dir "$REASONING_DIR" \
      --total-examples "$REASONING_EXAMPLES" \
      --max-think-chars "$REASONING_MAX_THINK_CHARS" \
      --max-answer-chars "$REASONING_MAX_ANSWER_CHARS" \
      --max-code-fences 0
  upload_dir_to_s3 "$REASONING_DIR" "$REASONING_S3"
  mark_stage_complete reasoning_sft_data
fi

if should_run_stage reward_prefs; then
  write_status reward_prefs
  log "Preparing reward-model preference pairs."
  run_with_stage_cache reward_prefs \
    "$PY" "$ROOT_DIR/scripts/prepare_metis15_preference_data.py" \
      --mixture-config "$PREFERENCE_MIX" \
      --output-dir "$REWARD_PREF_DIR" \
      --total-pairs "$REWARD_EXAMPLES"
  upload_dir_to_s3 "$REWARD_PREF_DIR" "$REWARD_PREF_S3"
  mark_stage_complete reward_prefs
fi

if should_run_stage dpo_prefs; then
  write_status dpo_prefs
  log "Preparing DPO preference pairs."
  run_with_stage_cache dpo_prefs \
    "$PY" "$ROOT_DIR/scripts/prepare_metis15_preference_data.py" \
      --mixture-config "$PREFERENCE_MIX" \
      --output-dir "$DPO_PREF_DIR" \
      --total-pairs "$DPO_PAIRS"
  upload_dir_to_s3 "$DPO_PREF_DIR" "$DPO_PREF_S3"
  mark_stage_complete dpo_prefs
fi

if should_run_stage planning; then
  write_status planning
  log "Writing derived plan and S3 storage manifest."
  "$PY" "$ROOT_DIR/scripts/plan_metis13.py" \
    --manifest "$MANIFEST" \
    --pretrain-meta "$PRETRAIN_DIR/meta.json" \
    --chat-meta "$CHAT_DIR/meta.json" \
    --reasoning-meta "$REASONING_DIR/meta.json" \
    --output-path "$PLAN_PATH"
  cat > "$STORAGE_MANIFEST_PATH" <<EOF
{
  "s3_root": "$S3_ROOT",
  "artifacts": {
    "normalized_pretrain": "$NORMALIZED_PRETRAIN_S3",
    "normalized_continued": "$NORMALIZED_CONTINUED_S3",
    "tokenizer": "$TOKENIZER_S3",
    "pretrain_base": "$PRETRAIN_S3",
    "pretrain_continued": "$CONTINUED_S3",
    "chat_sft": "$CHAT_S3",
    "identity_sft_source": "$IDENTITY_SFT_S3",
    "reasoning_sft": "$REASONING_S3",
    "reward_prefs": "$REWARD_PREF_S3",
    "dpo_prefs": "$DPO_PREF_S3",
    "plan": "$MANIFESTS_S3/metis15_plan.json"
  }
}
EOF
  upload_file_to_s3 "$PLAN_PATH" "$MANIFESTS_S3/metis15_plan.json"
  upload_file_to_s3 "$STORAGE_MANIFEST_PATH" "$MANIFESTS_S3/storage_manifest.json"
  mark_stage_complete planning
fi

mark_stage_complete complete
log "Metis-1.5 CPU prep complete."
echo "Metis-1.5 CPU prep complete."
echo "Prep mode:    $PREP_MODE"
echo "Local root:   $LOCAL_ROOT"
echo "S3 root:      $S3_ROOT"
echo "Plan file:    $PLAN_PATH"
