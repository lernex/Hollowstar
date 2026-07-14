#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_ROOT="${METIS15_LOCAL_ROOT:-/workspace/metis15_cpu_prep}"
LOG_DIR="${METIS15_LOG_DIR:-/workspace/metis15_logs}"
SUPERVISOR_LOG="$LOG_DIR/metis15_cpu_prep_supervisor.log"
PID_FILE="$LOG_DIR/metis15_cpu_prep_tokenizer_forward.pid"
STATUS_PATH="$LOCAL_ROOT/state/metis15_cpu_prep_status.env"
LAST_COMPLETED_STAGE_PATH="$LOCAL_ROOT/state/metis15_cpu_prep_last_completed_stage.txt"
LOCK_DIR="$LOCAL_ROOT/state/metis15_cpu_prep.lock"
PARQUET_CACHE_ROOT="$LOCAL_ROOT/cache/parquet"

S3_ROOT="${METIS15_S3_ROOT:?METIS15_S3_ROOT is required}"
PREP_MODE="${METIS15_PREP_MODE:-runpod}"
NORMALIZE_WORKERS="${METIS15_NORMALIZE_WORKERS:-48}"
SOURCE_PARTITIONS="${METIS15_SOURCE_PARTITIONS:-16}"
PARTITION_TARGET_DOCS="${METIS15_PARTITION_TARGET_DOCS:-60000}"
PARTITION_MIN_DOCS="${METIS15_PARTITION_MIN_DOCS:-120000}"
NORMALIZED_ZSTD_LEVEL="${METIS15_NORMALIZED_ZSTD_LEVEL:-1}"
PARQUET_CACHE_GB="${METIS15_PARQUET_CACHE_GB:-1}"
PARQUET_PREFETCH_COUNT="${METIS15_PARQUET_PREFETCH_COUNT:-1}"

mkdir -p "$LOG_DIR" "$LOCAL_ROOT/state"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$SUPERVISOR_LOG"
}

last_completed_stage() {
  if [[ -f "$LAST_COMPLETED_STAGE_PATH" ]]; then
    cat "$LAST_COMPLETED_STAGE_PATH"
  else
    printf 'unknown\n'
  fi
}

current_stage() {
  if [[ -f "$STATUS_PATH" ]]; then
    grep '^METIS15_STAGE=' "$STATUS_PATH" | tail -n 1 | cut -d= -f2-
  else
    printf 'unknown\n'
  fi
}

prep_running() {
  pgrep -f 'scripts/metis15_cpu_prep.sh' >/dev/null 2>&1
}

start_prep() {
  local start_stage="$1"
  log "Starting metis15_cpu_prep.sh from $start_stage."
  rm -rf "$LOCK_DIR"
  (
    cd "$ROOT_DIR"
    nohup env \
      METIS15_PREP_MODE="$PREP_MODE" \
      METIS15_LOCAL_ROOT="$LOCAL_ROOT" \
      METIS15_S3_ROOT="$S3_ROOT" \
      METIS15_START_STAGE="$start_stage" \
      METIS15_STOP_AFTER_STAGE=continued_pretrain_data \
      METIS15_NORMALIZE_WORKERS="$NORMALIZE_WORKERS" \
      METIS15_SOURCE_PARTITIONS="$SOURCE_PARTITIONS" \
      METIS15_PARTITION_TARGET_DOCS="$PARTITION_TARGET_DOCS" \
      METIS15_PARTITION_MIN_DOCS="$PARTITION_MIN_DOCS" \
      METIS15_NORMALIZED_ZSTD_LEVEL="$NORMALIZED_ZSTD_LEVEL" \
      METIS15_PARQUET_CACHE_GB="$PARQUET_CACHE_GB" \
      METIS15_PARQUET_PREFETCH_COUNT="$PARQUET_PREFETCH_COUNT" \
      bash scripts/metis15_cpu_prep.sh >> "$LOG_DIR/metis15_cpu_prep_tokenizer_forward.log" 2>&1 &
    echo "$!" > "$PID_FILE"
  )
}

cleanup_safe_cache() {
  rm -rf "$LOCAL_ROOT/data/metis15_base" "$LOCAL_ROOT/normalized/pretrain"
  find "$PARQUET_CACHE_ROOT" -type f -mmin +10 -delete 2>/dev/null || true
  find "$PARQUET_CACHE_ROOT" -type d -empty -delete 2>/dev/null || true
}

log "Supervisor started."
while true; do
  cleanup_safe_cache
  last_stage="$(last_completed_stage)"
  stage="$(current_stage)"

  if [[ "$last_stage" == "continued_pretrain_data" ]]; then
    log "continued_pretrain_data is complete. Supervisor exiting."
    exit 0
  fi

  if prep_running; then
    sleep 120
    continue
  fi

  if [[ "$last_stage" == "normalized_continued" || "$stage" == "continued_pretrain_data" ]]; then
    start_prep continued_pretrain_data
  else
    start_prep normalized_continued
  fi

  sleep 120
done

