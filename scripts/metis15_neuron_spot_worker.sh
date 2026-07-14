#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METIS15_ROOT_DIR:-/mnt/trn1/src/metis}"
RUN_ID="${METIS15_RUN_ID:-metis15-base-neuron}"
LOCK_TABLE="${METIS15_LOCK_TABLE:-metis15-training-locks}"
LOCK_TTL_SECONDS="${METIS15_LOCK_TTL_SECONDS:-600}"
LOCK_HEARTBEAT_SECONDS="${METIS15_LOCK_HEARTBEAT_SECONDS:-60}"
LOCK_WAIT_SECONDS="${METIS15_LOCK_WAIT_SECONDS:-60}"
CHECKPOINT_SYNC_SECONDS="${METIS15_CHECKPOINT_SYNC_SECONDS:-60}"
NEURON_CACHE_SYNC_SECONDS="${METIS15_NEURON_CACHE_SYNC_SECONDS:-300}"
LOG_SYNC_SECONDS="${METIS15_LOG_SYNC_SECONDS:-60}"
CHECKPOINT_S3_URI="${METIS15_S3_CHECKPOINTS_URI:-}"
NEURON_CACHE_S3_URI="${METIS15_S3_NEURON_CACHE_URI:-}"
LOG_S3_URI="${METIS15_S3_LOGS_URI:-}"
AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
export AWS_REGION
export AWS_DEFAULT_REGION="$AWS_REGION"
OUT_DIR="${METIS15_OUT_DIR:-/mnt/trn1/checkpoints/metis15_base_neuron}"
NEURON_CACHE_DIR="${NEURON_CC_CACHE:-/mnt/trn1/neuron_cc_cache}"
LOG_DIR="${METIS15_LOG_DIR:-/mnt/trn1/logs/metis15}"
LOG_FILE="$LOG_DIR/trainium-worker.log"
NEURON_CACHE_SYNC_EXCLUDES=(--exclude "*.lock" --exclude "*.tmp" --exclude "*/lock")
NEURON_CACHE_FAILURE_PATTERNS=(
  "NCC_EOOM"
  "Maximum peak HBM usage"
  "Failed compilation"
  "failed compilation"
  "cached failed"
  "error condition"
  "INTERNAL_ERROR"
  "TCTransform"
  "CompilerInternalError"
  "oom_checker failed"
  "Assertion failure"
  "Traceback"
  "RuntimeError"
)
DEFAULT_PYTHON_BIN="/mnt/trn1/venvs/aws_neuron_venv_pytorch/bin/python"
if [[ -d "/mnt/trn1/venvs/aws_neuron_venv_pytorch/bin" ]]; then
  export PATH="/mnt/trn1/venvs/aws_neuron_venv_pytorch/bin:$PATH"
fi
if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
  PYTHON_BIN="${METIS15_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="${METIS15_PYTHON_BIN:-python3}"
fi

imds_get() {
  local path="$1"
  local token
  token="$(curl -fsS --max-time 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
  if [[ -n "$token" ]]; then
    curl -fsS --max-time 2 -H "X-aws-ec2-metadata-token: $token" "http://169.254.169.254/latest/meta-data/$path" 2>/dev/null || true
  else
    curl -fsS --max-time 2 "http://169.254.169.254/latest/meta-data/$path" 2>/dev/null || true
  fi
}

INSTANCE_ID="${METIS15_INSTANCE_ID:-$(imds_get instance-id)}"
INSTANCE_TYPE="${METIS15_INSTANCE_TYPE:-$(imds_get instance-type)}"
LOCAL_IPV4="${METIS15_LOCAL_IPV4:-$(imds_get local-ipv4)}"
PUBLIC_IPV4="${METIS15_PUBLIC_IPV4:-$(imds_get public-ipv4)}"
AVAILABILITY_ZONE="${METIS15_AVAILABILITY_ZONE:-$(imds_get placement/availability-zone)}"
OWNER_ID="${METIS15_OWNER_ID:-${INSTANCE_ID:-$(hostname)}-$$}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

STATUS_S3_URI="${METIS15_S3_STATUS_URI:-}"
STATUS_STAGE="starting"
STATUS_EXTRA=""
shutdown_requested=0

json_escape() {
  sed 's/\\/\\\\/g; s/"/\\"/g' <<<"${1:-}"
}

write_status() {
  [[ -n "$STATUS_S3_URI" ]] || return 0
  local stage="${1:-$STATUS_STAGE}"
  local extra="${2:-$STATUS_EXTRA}"
  local tmp_status
  local log_bytes=0
  local checkpoint_files=0
  local checkpoint_bytes=0
  local completed_cache_modules=0
  tmp_status="$(mktemp /tmp/metis15_status.XXXXXX.json)"
  if [[ -f "$LOG_FILE" ]]; then
    log_bytes="$(stat -c '%s' "$LOG_FILE" 2>/dev/null || echo 0)"
  fi
  if [[ -d "$OUT_DIR" ]]; then
    checkpoint_files="$(find "$OUT_DIR" -maxdepth 1 -type f 2>/dev/null | wc -l | awk '{print $1}')"
    checkpoint_bytes="$(find "$OUT_DIR" -maxdepth 1 -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
  fi
  if [[ -d "$NEURON_CACHE_DIR" ]]; then
    completed_cache_modules="$(find "$NEURON_CACHE_DIR" -mindepth 2 -maxdepth 2 -type d -name "MODULE_*" -exec test -f "{}/model.done" \; -exec test -s "{}/model.neff" \; -print 2>/dev/null | wc -l | awk '{print $1}')"
  fi
  cat >"$tmp_status" <<JSON
{
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "stage": "$(json_escape "$stage")",
  "extra": "$(json_escape "$extra")",
  "run_id": "$(json_escape "$RUN_ID")",
  "owner_id": "$(json_escape "$OWNER_ID")",
  "hostname": "$(json_escape "$(hostname)")",
  "instance_id": "$(json_escape "$INSTANCE_ID")",
  "instance_type": "$(json_escape "$INSTANCE_TYPE")",
  "availability_zone": "$(json_escape "$AVAILABILITY_ZONE")",
  "local_ipv4": "$(json_escape "$LOCAL_IPV4")",
  "public_ipv4": "$(json_escape "$PUBLIC_IPV4")",
  "log_s3_uri": "$(json_escape "$LOG_S3_URI")",
  "checkpoint_s3_uri": "$(json_escape "$CHECKPOINT_S3_URI")",
  "neuron_cache_s3_uri": "$(json_escape "$NEURON_CACHE_S3_URI")",
  "log_bytes": $log_bytes,
  "checkpoint_files": $checkpoint_files,
  "checkpoint_bytes": $checkpoint_bytes,
  "completed_cache_modules": $completed_cache_modules
}
JSON
  aws s3 cp "$tmp_status" "$STATUS_S3_URI/status.json" --only-show-errors || true
  rm -f "$tmp_status"
}

module_has_failed_neuron_cache_artifact() {
  local module_dir="$1"
  local pattern
  for pattern in "${NEURON_CACHE_FAILURE_PATTERNS[@]}"; do
    if LC_ALL=C grep -RIl --exclude='*.neff' -- "$pattern" "$module_dir" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

prune_incomplete_neuron_cache_modules() {
  [[ -d "$NEURON_CACHE_DIR" ]] || return 0
  while IFS= read -r -d '' module_dir; do
    if [[ ! -f "$module_dir/model.done" || ! -s "$module_dir/model.neff" ]]; then
      echo "neuron_cache_prune_incomplete module=${module_dir#$NEURON_CACHE_DIR/}"
      rm -rf "$module_dir"
    elif module_has_failed_neuron_cache_artifact "$module_dir"; then
      echo "neuron_cache_prune_failed module=${module_dir#$NEURON_CACHE_DIR/}"
      rm -rf "$module_dir"
    fi
  done < <(find "$NEURON_CACHE_DIR" -mindepth 2 -maxdepth 2 -type d -name "MODULE_*" -print0 2>/dev/null)
  find "$NEURON_CACHE_DIR" -type f \( -name "*.lock" -o -name "*.tmp" \) -delete 2>/dev/null || true
}

sync_completed_neuron_cache_modules() {
  [[ -n "$NEURON_CACHE_S3_URI" && -d "$NEURON_CACHE_DIR" ]] || return 0
  prune_incomplete_neuron_cache_modules
  while IFS= read -r -d '' module_dir; do
    [[ -f "$module_dir/model.done" && -s "$module_dir/model.neff" ]] || continue
    module_has_failed_neuron_cache_artifact "$module_dir" && continue
    rel="${module_dir#$NEURON_CACHE_DIR/}"
    aws s3 sync "$module_dir/" "${NEURON_CACHE_S3_URI%/}/$rel/" "${NEURON_CACHE_SYNC_EXCLUDES[@]}" --only-show-errors || true
  done < <(find "$NEURON_CACHE_DIR" -mindepth 2 -maxdepth 2 -type d -name "MODULE_*" -print0 2>/dev/null)
}

completed_neuron_cache_token() {
  [[ -d "$NEURON_CACHE_DIR" ]] || return 0
  while IFS= read -r -d '' module_dir; do
    [[ -f "$module_dir/model.done" && -s "$module_dir/model.neff" ]] || continue
    module_has_failed_neuron_cache_artifact "$module_dir" && continue
    find "$module_dir" -maxdepth 1 -type f ! -name "*.lock" ! -name "*.tmp" -printf '%P:%s:%T@\n' 2>/dev/null | sed "s#^#${module_dir#$NEURON_CACHE_DIR/}/#"
  done < <(find "$NEURON_CACHE_DIR" -mindepth 2 -maxdepth 2 -type d -name "MODULE_*" -print0 2>/dev/null) | sort | sha256sum | awk '{print $1}'
}

lock_helper="$(mktemp /tmp/metis15_lock.XXXXXX.py)"
cat >"$lock_helper" <<'PY'
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

table_name = os.environ["METIS15_LOCK_TABLE"]
run_id = os.environ["METIS15_RUN_ID"]
owner = os.environ["METIS15_OWNER_ID"]
ttl = int(os.environ.get("METIS15_LOCK_TTL_SECONDS", "600"))
now = int(time.time())
expires = now + ttl
ddb = boto3.client("dynamodb")


def acquire() -> int:
    try:
        ddb.update_item(
            TableName=table_name,
            Key={"run_id": {"S": run_id}},
            UpdateExpression=(
                "SET #owner = :owner, expires_at = :expires, updated_at = :now, "
                "host = :host"
            ),
            ConditionExpression=(
                "attribute_not_exists(run_id) OR expires_at < :now OR #owner = :owner"
            ),
            ExpressionAttributeNames={"#owner": "owner"},
            ExpressionAttributeValues={
                ":owner": {"S": owner},
                ":expires": {"N": str(expires)},
                ":now": {"N": str(now)},
                ":host": {"S": os.uname().nodename},
            },
        )
        print(f"lock_acquired run_id={run_id} owner={owner} expires_at={expires}", flush=True)
        return 0
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            item = ddb.get_item(
                TableName=table_name,
                Key={"run_id": {"S": run_id}},
                ConsistentRead=True,
            ).get("Item", {})
            holder = item.get("owner", {}).get("S", "unknown")
            held_until = item.get("expires_at", {}).get("N", "unknown")
            print(f"lock_busy run_id={run_id} holder={holder} expires_at={held_until}", flush=True)
            return 1
        raise


def release() -> int:
    try:
        ddb.delete_item(
            TableName=table_name,
            Key={"run_id": {"S": run_id}},
            ConditionExpression="#owner = :owner",
            ExpressionAttributeNames={"#owner": "owner"},
            ExpressionAttributeValues={":owner": {"S": owner}},
        )
        print(f"lock_released run_id={run_id} owner={owner}", flush=True)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
    return 0


if sys.argv[1] == "acquire":
    raise SystemExit(acquire())
if sys.argv[1] == "heartbeat":
    raise SystemExit(acquire())
if sys.argv[1] == "release":
    raise SystemExit(release())
raise SystemExit(f"unknown command: {sys.argv[1]}")
PY

export METIS15_LOCK_TABLE="$LOCK_TABLE"
export METIS15_RUN_ID="$RUN_ID"
export METIS15_OWNER_ID="$OWNER_ID"
export METIS15_LOCK_TTL_SECONDS="$LOCK_TTL_SECONDS"

request_shutdown() {
  shutdown_requested=1
  exit 143
}

cleanup() {
  set +e
  for child_pid in "${notice_pid:-}" "${log_sync_pid:-}" "${cache_sync_pid:-}" "${sync_pid:-}" "${heartbeat_pid:-}"; do
    if [[ -n "$child_pid" ]]; then
      kill "$child_pid" >/dev/null 2>&1 || true
    fi
  done
  if [[ "${STATUS_STAGE:-}" != "finished" ]]; then
    STATUS_STAGE="cleanup"
    write_status "cleanup" "signal_or_exit"
  fi
  if [[ "${shutdown_requested:-0}" == "1" && -n "${train_pid:-}" && -d "/proc/$train_pid" ]]; then
    pkill -USR1 -f train_metis15_neuron.py >/dev/null 2>&1 || true
    wait "$train_pid" >/dev/null 2>&1 || true
  fi
  if [[ -n "$CHECKPOINT_S3_URI" && -d "$OUT_DIR" ]]; then
    aws s3 sync "$OUT_DIR/" "$CHECKPOINT_S3_URI/" --only-show-errors || true
  fi
  if [[ -n "$NEURON_CACHE_S3_URI" && -d "$NEURON_CACHE_DIR" ]]; then
    prune_incomplete_neuron_cache_modules
    sync_completed_neuron_cache_modules
  fi
  if [[ -n "$LOG_S3_URI" && -f "$LOG_FILE" ]]; then
    aws s3 cp "$LOG_FILE" "$LOG_S3_URI/trainium-worker.log" --only-show-errors || true
  fi
  "$PYTHON_BIN" "$lock_helper" release >/dev/null 2>&1 || true
  rm -f "$lock_helper"
}
trap cleanup EXIT
trap request_shutdown INT TERM

while ! "$PYTHON_BIN" "$lock_helper" acquire; do
  write_status "waiting_for_lock" "lock_busy"
  sleep "$LOCK_WAIT_SECONDS"
done

export METIS15_S3_ROOT="${METIS15_S3_ROOT:-s3://lernex-metis-artifacts-151025633969-us-east-1/metis15}"
export METIS15_S3_PRETRAIN_URI="${METIS15_S3_PRETRAIN_URI:-$METIS15_S3_ROOT/pretrain-shards/base}"
export METIS15_S3_CHECKPOINTS_URI="${METIS15_S3_CHECKPOINTS_URI:-$METIS15_S3_ROOT/checkpoints/base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master}"
export METIS15_S3_NEURON_CACHE_URI="${METIS15_S3_NEURON_CACHE_URI:-$METIS15_S3_ROOT/neuron-cc-cache/base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master}"
export METIS15_S3_LOGS_URI="${METIS15_S3_LOGS_URI:-$METIS15_S3_ROOT/logs/$RUN_ID/$OWNER_ID}"
export METIS15_S3_STATUS_URI="${METIS15_S3_STATUS_URI:-$METIS15_S3_ROOT/status/$RUN_ID/$OWNER_ID}"
CHECKPOINT_S3_URI="$METIS15_S3_CHECKPOINTS_URI"
NEURON_CACHE_S3_URI="$METIS15_S3_NEURON_CACHE_URI"
LOG_S3_URI="$METIS15_S3_LOGS_URI"
STATUS_S3_URI="$METIS15_S3_STATUS_URI"

write_status "lock_acquired" "starting_background_sync"

(
  while true; do
    sleep "$LOCK_HEARTBEAT_SECONDS"
    "$PYTHON_BIN" "$lock_helper" heartbeat || exit 1
  done
) &
heartbeat_pid=$!

(
  last_sync_token=""
  while true; do
    sleep "$CHECKPOINT_SYNC_SECONDS"
    [[ -n "$CHECKPOINT_S3_URI" && -d "$OUT_DIR" ]] || continue
    if [[ -f "$OUT_DIR/latest.pt" ]]; then
      sync_token="$(find "$OUT_DIR" -maxdepth 1 -type f -printf '%f:%s:%T@\n' | sort | sha256sum | awk '{print $1}')"
      if [[ "$sync_token" != "$last_sync_token" ]]; then
        echo "checkpoint_sync_start uri=$CHECKPOINT_S3_URI"
        aws s3 sync "$OUT_DIR/" "$CHECKPOINT_S3_URI/" --only-show-errors
        echo "checkpoint_sync_done uri=$CHECKPOINT_S3_URI"
        last_sync_token="$sync_token"
      fi
    fi
  done
) &
sync_pid=$!

(
  last_sync_token=""
  while true; do
    sleep "$NEURON_CACHE_SYNC_SECONDS"
    [[ -n "$NEURON_CACHE_S3_URI" && -d "$NEURON_CACHE_DIR" ]] || continue
    sync_token="$(completed_neuron_cache_token)"
    if [[ -n "$sync_token" && "$sync_token" != "$last_sync_token" ]]; then
      echo "neuron_cache_sync_start uri=$NEURON_CACHE_S3_URI"
      sync_completed_neuron_cache_modules
      echo "neuron_cache_sync_done uri=$NEURON_CACHE_S3_URI"
      last_sync_token="$sync_token"
    fi
  done
) &
cache_sync_pid=$!

(
  while true; do
    token="$(curl -fsS --max-time 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 30' 2>/dev/null || true)"
    if [[ -n "$token" ]]; then
      action="$(curl -fsS --max-time 2 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null || true)"
      rebalance="$(curl -fsS --max-time 2 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/events/recommendations/rebalance 2>/dev/null || true)"
      if [[ -n "$action" || -n "$rebalance" ]]; then
        echo "spot_interruption_or_rebalance_notice action=${action:-none} rebalance=${rebalance:-none}"
        write_status "spot_interruption_or_rebalance_notice" "action=${action:-none} rebalance=${rebalance:-none}"
        pkill -USR1 -f train_metis15_neuron.py >/dev/null 2>&1 || true
        exit 0
      fi
    fi
    sleep 5
  done
) &
notice_pid=$!

export METIS15_S3_ROOT="${METIS15_S3_ROOT:-s3://lernex-metis-artifacts-151025633969-us-east-1/metis15}"
export METIS15_S3_PRETRAIN_URI="${METIS15_S3_PRETRAIN_URI:-$METIS15_S3_ROOT/pretrain-shards/base}"
export METIS15_S3_CHECKPOINTS_URI="${METIS15_S3_CHECKPOINTS_URI:-$METIS15_S3_ROOT/checkpoints/base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master}"
export METIS15_S3_NEURON_CACHE_URI="${METIS15_S3_NEURON_CACHE_URI:-$METIS15_S3_ROOT/neuron-cc-cache/base-neuron-groupstatic-cf2-hidden-lr1p5e4-sched-master}"
export METIS15_S3_LOGS_URI="${METIS15_S3_LOGS_URI:-$METIS15_S3_ROOT/logs/$RUN_ID/$OWNER_ID}"
export METIS15_S3_STATUS_URI="${METIS15_S3_STATUS_URI:-$METIS15_S3_ROOT/status/$RUN_ID/$OWNER_ID}"
export METIS15_DATA_DIR="${METIS15_DATA_DIR:-/mnt/trn1/data/metis15_base}"
export METIS15_OUT_DIR="$OUT_DIR"
export METIS15_NEURON_LOCAL_BATCH_SIZE="${METIS15_NEURON_LOCAL_BATCH_SIZE:-1}"
export METIS15_NEURON_GRAD_ACCUM_STEPS="${METIS15_NEURON_GRAD_ACCUM_STEPS:-16}"
export METIS15_NEURON_LR="${METIS15_NEURON_LR:-1.5e-4}"
export METIS15_NEURON_WARMUP_STEPS="${METIS15_NEURON_WARMUP_STEPS:-200}"
export METIS15_NEURON_CONSTANT_LR="${METIS15_NEURON_CONSTANT_LR:-0}"
export METIS15_NEURON_OVERRIDE_LR_ON_RESUME="${METIS15_NEURON_OVERRIDE_LR_ON_RESUME:-1}"
export METIS15_NEURON_DISPATCH_PACK_IMPL="${METIS15_NEURON_DISPATCH_PACK_IMPL:-group_static}"
export METIS15_NEURON_BALANCED_STATIC_LAYOUT="${METIS15_NEURON_BALANCED_STATIC_LAYOUT:-indexed}"
export METIS15_NEURON_BALANCED_STATIC_ROUTER_WEIGHTS="${METIS15_NEURON_BALANCED_STATIC_ROUTER_WEIGHTS:-uniform}"
export METIS15_NEURON_BALANCED_STATIC_ROUTER_INPUT="${METIS15_NEURON_BALANCED_STATIC_ROUTER_INPUT:-hidden}"
export METIS15_NEURON_ROUTER_OVERRIDE="${METIS15_NEURON_ROUTER_OVERRIDE:-learned}"
export METIS15_NEURON_EXPERT_CAPACITY_FACTOR="${METIS15_NEURON_EXPERT_CAPACITY_FACTOR:-2.0}"
export METIS15_NEURON_ATTENTION_KERNEL="${METIS15_NEURON_ATTENTION_KERNEL:-nki_flash_1k}"
export METIS15_NEURON_NKI_FLASH_LSE_DTYPE="${METIS15_NEURON_NKI_FLASH_LSE_DTYPE:-bfloat16}"
export METIS15_NEURON_GRAD_SYNC_MODE="${METIS15_NEURON_GRAD_SYNC_MODE:-all_reduce_staged}"
export METIS15_NEURON_GRAD_SYNC_BUCKET_MB="${METIS15_NEURON_GRAD_SYNC_BUCKET_MB:-32}"
export METIS15_NEURON_MARK_STEP_EACH_MICROBATCH="${METIS15_NEURON_MARK_STEP_EACH_MICROBATCH:-0}"
export METIS15_NEURON_LOCAL_LOG_METRICS="${METIS15_NEURON_LOCAL_LOG_METRICS:-0}"
export METIS15_NEURON_TIE_EMBEDDINGS="${METIS15_NEURON_TIE_EMBEDDINGS:-0}"
export METIS15_NEURON_GRAD_CLIP="${METIS15_NEURON_GRAD_CLIP:-0}"
export METIS15_NEURON_EXPERT_ACTIVATION_SAFETY="${METIS15_NEURON_EXPERT_ACTIVATION_SAFETY:-clamp}"
export METIS15_NEURON_PREINIT_OPTIMIZER_STATE="${METIS15_NEURON_PREINIT_OPTIMIZER_STATE:-1}"
export METIS15_NEURON_OPTIMIZER_MASTER_WEIGHTS="${METIS15_NEURON_OPTIMIZER_MASTER_WEIGHTS:-1}"
export METIS15_NEURON_CE_IMPL="${METIS15_NEURON_CE_IMPL:-cross_entropy}"
export METIS15_NEURON_CE_LOGITS_DTYPE="${METIS15_NEURON_CE_LOGITS_DTYPE:-bfloat16}"
export METIS15_NEURON_PERF_WARMUP_STEPS="${METIS15_NEURON_PERF_WARMUP_STEPS:-20}"
export METIS15_NEURON_PROFILE_COMPONENTS="${METIS15_NEURON_PROFILE_COMPONENTS:-1}"
export METIS15_NEURON_CHECKPOINT_INTERVAL="${METIS15_NEURON_CHECKPOINT_INTERVAL:-5000}"
NEURON_CACHE_S3_URI="$METIS15_S3_NEURON_CACHE_URI"
LOG_S3_URI="$METIS15_S3_LOGS_URI"
STATUS_S3_URI="$METIS15_S3_STATUS_URI"
write_status "configured" "runtime_env_ready"

if [[ -n "$NEURON_CACHE_S3_URI" ]]; then
  mkdir -p "$NEURON_CACHE_DIR"
  echo "Hydrating Neuron compiler cache from S3: $NEURON_CACHE_S3_URI"
  write_status "hydrating_neuron_cache" "$NEURON_CACHE_S3_URI"
  aws s3 sync "$NEURON_CACHE_S3_URI/" "$NEURON_CACHE_DIR/" "${NEURON_CACHE_SYNC_EXCLUDES[@]}" --only-show-errors || true
  prune_incomplete_neuron_cache_modules
  write_status "hydrated_neuron_cache" "$NEURON_CACHE_S3_URI"
fi

(
  last_log_size=-1
  while true; do
    sleep "$LOG_SYNC_SECONDS"
    [[ -n "$LOG_S3_URI" && -f "$LOG_FILE" ]] || continue
    log_size="$(stat -c '%s' "$LOG_FILE" 2>/dev/null || echo 0)"
    if [[ "$log_size" != "$last_log_size" ]]; then
      aws s3 cp "$LOG_FILE" "$LOG_S3_URI/trainium-worker.log" --only-show-errors || true
      write_status "training_or_compiling" "log_bytes=$log_size"
      last_log_size="$log_size"
    fi
  done
) &
log_sync_pid=$!

write_status "launching_training" "$ROOT_DIR/scripts/metis15_neuron_pretrain.sh"
"$ROOT_DIR/scripts/metis15_neuron_pretrain.sh" 2>&1 | tee -a "$LOG_FILE" &
train_pid=$!
wait "$train_pid"
train_status=$?
write_status "training_exited" "exit_code=$train_status"

kill "$heartbeat_pid" "$sync_pid" "$cache_sync_pid" "$notice_pid" "$log_sync_pid" >/dev/null 2>&1 || true
if [[ -n "$CHECKPOINT_S3_URI" && -d "$OUT_DIR" ]]; then
  echo "final_checkpoint_sync_start uri=$CHECKPOINT_S3_URI"
  write_status "final_checkpoint_sync_start" "$CHECKPOINT_S3_URI"
  aws s3 sync "$OUT_DIR/" "$CHECKPOINT_S3_URI/" --only-show-errors
  echo "final_checkpoint_sync_done uri=$CHECKPOINT_S3_URI"
  write_status "final_checkpoint_sync_done" "$CHECKPOINT_S3_URI"
fi
if [[ -n "$NEURON_CACHE_S3_URI" && -d "$NEURON_CACHE_DIR" ]]; then
  echo "final_neuron_cache_sync_start uri=$NEURON_CACHE_S3_URI"
  write_status "final_neuron_cache_sync_start" "$NEURON_CACHE_S3_URI"
  prune_incomplete_neuron_cache_modules
  sync_completed_neuron_cache_modules
  echo "final_neuron_cache_sync_done uri=$NEURON_CACHE_S3_URI"
  write_status "final_neuron_cache_sync_done" "$NEURON_CACHE_S3_URI"
fi
if [[ -n "$LOG_S3_URI" && -f "$LOG_FILE" ]]; then
  echo "final_log_sync_start uri=$LOG_S3_URI"
  aws s3 cp "$LOG_FILE" "$LOG_S3_URI/trainium-worker.log" --only-show-errors || true
  echo "final_log_sync_done uri=$LOG_S3_URI"
fi
STATUS_STAGE="finished"
write_status "finished" "exit_code=$train_status"
exit "$train_status"
