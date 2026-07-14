#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${METIS15_MANIFEST:-$ROOT_DIR/configs/metis15_manifest.json}"
ASSETS_DIR="${METIS15_ASSETS_DIR:-$ROOT_DIR/artifacts/metis15_hf_assets}"

PRETRAIN_DATA_DIR="${METIS15_DATA_DIR:-$ROOT_DIR/data/metis15_base}"
CONTINUED_DATA_DIR="${METIS15_CONTINUED_DATA_DIR:-$ROOT_DIR/data/metis15_continued_pretrain}"
CHAT_DATA_DIR="${METIS15_CHAT_DATA_DIR:-$ROOT_DIR/data/metis15_chat_sft}"
REASONING_DATA_DIR="${METIS15_REASONING_DATA_DIR:-$ROOT_DIR/data/metis15_reasoning_sft}"
REWARD_PREF_DIR="${METIS15_REWARD_PREF_DIR:-$ROOT_DIR/data/metis15_reward_prefs}"
DPO_PREF_DIR="${METIS15_DPO_PREF_DIR:-$ROOT_DIR/data/metis15_dpo_prefs}"

RUNS_ROOT="${METIS15_RUNS_ROOT:-$ROOT_DIR/checkpoints}"
RELEASES_ROOT="${METIS15_RELEASES_ROOT:-$ROOT_DIR/releases/metis15}"
S3_ROOT="${METIS15_S3_ROOT:-}"
DISABLE_UPLOADS="${METIS15_DISABLE_UPLOADS:-0}"

BASE_RUN="${METIS15_BASE_OUT:-$RUNS_ROOT/metis15_base}"
CONTINUED_RUN="${METIS15_CONTINUED_OUT:-$RUNS_ROOT/metis15_continued}"
CHAT_RUN="${METIS15_CHAT_OUT:-$RUNS_ROOT/metis15_chat}"
THINK_RUN="${METIS15_THINK_OUT:-$RUNS_ROOT/metis15_think}"
REWARD_RUN="${METIS15_REWARD_OUT:-$RUNS_ROOT/metis15_reward}"
THINK_DPO_RUN="${METIS15_THINK_DPO_OUT:-$RUNS_ROOT/metis15_think_dpo}"

BASE_RELEASE="${METIS15_BASE_RELEASE:-$RELEASES_ROOT/base}"
THINK_RELEASE="${METIS15_THINK_RELEASE:-$RELEASES_ROOT/think}"
EVAL_REPORT="${METIS15_EVAL_REPORT:-$RELEASES_ROOT/eval_comparison.json}"
TOKENIZER_S3_URI="${METIS15_S3_TOKENIZER_URI:-${S3_ROOT:+$S3_ROOT/tokenizer}}"
PRETRAIN_S3_URI="${METIS15_S3_PRETRAIN_URI:-${S3_ROOT:+$S3_ROOT/pretrain-shards/base}}"
CONTINUED_S3_URI="${METIS15_S3_CONTINUED_URI:-${S3_ROOT:+$S3_ROOT/pretrain-shards/continued}}"
CHAT_S3_URI="${METIS15_S3_CHAT_URI:-${S3_ROOT:+$S3_ROOT/chat-sft}}"
REASONING_S3_URI="${METIS15_S3_REASONING_URI:-${S3_ROOT:+$S3_ROOT/reasoning-sft}}"
REWARD_PREF_S3_URI="${METIS15_S3_REWARD_PREF_URI:-${S3_ROOT:+$S3_ROOT/reward-prefs}}"
DPO_PREF_S3_URI="${METIS15_S3_DPO_PREF_URI:-${S3_ROOT:+$S3_ROOT/dpo-prefs}}"
CHECKPOINTS_S3_ROOT="${METIS15_S3_CHECKPOINTS_URI:-${S3_ROOT:+$S3_ROOT/checkpoints}}"
RELEASES_S3_ROOT="${METIS15_S3_RELEASES_URI:-${S3_ROOT:+$S3_ROOT/releases}}"
MANIFESTS_S3_ROOT="${METIS15_S3_MANIFESTS_URI:-${S3_ROOT:+$S3_ROOT/manifests}}"

ENABLE_PREP="${METIS15_ENABLE_PREP:-1}"
ENABLE_FP8="${METIS15_FP8:-0}"
ENABLE_NVFP4="${METIS15_NVFP4:-0}"
ENABLE_CUSTOM_NEGATIVES="${METIS15_ENABLE_CUSTOM_NEGATIVES:-1}"
RUN_THINK_DPO="${METIS15_RUN_THINK_DPO:-1}"
UPLOAD_RELEASES="${METIS15_UPLOAD_RELEASES:-0}"
RESUME_TRAINING="${METIS15_RESUME:-1}"
START_STAGE="${METIS15_START_STAGE:-pretrain}"
STOP_AFTER_STAGE="${METIS15_STOP_AFTER_STAGE:-}"
SKIP_PRIOR_STAGE_CHECKS="${METIS15_SKIP_PRIOR_STAGE_CHECKS:-0}"
NPROC="${METIS15_NPROC:-1}"
SFT_NUM_WORKERS="${METIS15_SFT_NUM_WORKERS:-4}"
PREF_NUM_WORKERS="${METIS15_PREF_NUM_WORKERS:-4}"
TF32_FLAG="${METIS15_TF32:-0}"
MATMUL_PRECISION="${METIS15_MATMUL_PRECISION:-highest}"
LM_LOSS_IMPL="${METIS15_LM_LOSS_IMPL:-standard}"
RETAIN_STANDARD_CE_LOGITS="${METIS15_RETAIN_STANDARD_CE_LOGITS:-1}"
OPTIMIZER_NAME="${METIS15_OPTIMIZER:-muon_adamw}"
PREFETCH_BATCHES="${METIS15_PREFETCH_BATCHES:-4}"
FP8_EXPERT_PRECISION="${METIS15_FP8_EXPERT_PRECISION:-}"
MOE_DISPATCH_MODE="${METIS15_MOE_DISPATCH_MODE:-}"
MOE_BACKEND="${METIS15_MOE_BACKEND:-}"
MOE_STATIC_CAPACITY="${METIS15_MOE_STATIC_CAPACITY:-}"
MOE_CAPACITY_FACTOR="${METIS15_MOE_CAPACITY_FACTOR:-}"
MOE_CAPACITY_ALIGNMENT="${METIS15_MOE_CAPACITY_ALIGNMENT:-}"
MOE_OVERFLOW_MODE="${METIS15_MOE_OVERFLOW_MODE:-}"
MOE_GRAPHABLE="${METIS15_MOE_GRAPHABLE:-0}"
MOE_FUSED_COMBINE="${METIS15_MOE_FUSED_COMBINE:-1}"
MOE_MEMORY_EFFICIENT_PERMUTATION="${METIS15_MOE_MEMORY_EFFICIENT_PERMUTATION:-}"
MOE_PERMUTE_FUSION="${METIS15_MOE_PERMUTE_FUSION:-}"
NVFP4_FINAL_EXPERT_LAYERS="${METIS15_NVFP4_FINAL_EXPERT_LAYERS:-}"
NVFP4_FINAL_EXPERT_PRECISION="${METIS15_NVFP4_FINAL_EXPERT_PRECISION:-}"
BASE_TE_FUSED_MLP="${METIS15_BASE_TE_FUSED_MLP:-0}"
CONTINUED_TE_FUSED_MLP="${METIS15_CONTINUED_TE_FUSED_MLP:-0}"
BASE_TRAINING_MODE="${METIS15_BASE_TRAINING_MODE:-${METIS15_TRAINING_MODE:-static_dense_pretrain}}"
CONTINUED_TRAINING_MODE="${METIS15_CONTINUED_TRAINING_MODE:-dynamic_token_mor}"
COMPILE_FLAG="${METIS15_COMPILE:-0}"
COMPILE_MODE="${METIS15_COMPILE_MODE:-default}"
COMPILE_LOW_PRECISION="${METIS15_COMPILE_LOW_PRECISION:-0}"
PRETRAIN_MAX_DOCS="${METIS15_PRETRAIN_MAX_DOCS:-25000000}"
CONTINUED_MAX_DOCS="${METIS15_CONTINUED_MAX_DOCS:-3000000}"
CUSTOM_NEGATIVE_DEVICE="${METIS15_CUSTOM_NEGATIVE_DEVICE:-}"
CUSTOM_NEGATIVE_TOP_K="${METIS15_CUSTOM_NEGATIVE_TOP_K:-60}"
CUSTOM_NEGATIVE_CHAT_TEMPERATURE="${METIS15_CUSTOM_NEGATIVE_CHAT_TEMPERATURE:-0.9}"
CUSTOM_NEGATIVE_THINK_TEMPERATURE="${METIS15_CUSTOM_NEGATIVE_THINK_TEMPERATURE:-0.65}"
CUSTOM_NEGATIVE_CHAT_MAX_NEW_TOKENS="${METIS15_CUSTOM_NEGATIVE_CHAT_MAX_NEW_TOKENS:-220}"
CUSTOM_NEGATIVE_THINK_MAX_NEW_TOKENS="${METIS15_CUSTOM_NEGATIVE_THINK_MAX_NEW_TOKENS:-260}"
CUSTOM_NEGATIVE_MAX_SEED_ROWS="${METIS15_CUSTOM_NEGATIVE_MAX_SEED_ROWS:-}"
REQUIRED_CHAT_IDENTITY_SOURCE="${METIS15_CHAT_IDENTITY_SOURCE:-metis15_identity_manual}"
REQUIRED_CHAT_IDENTITY_COUNT="${METIS15_CHAT_IDENTITY_COUNT:-150}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_nonempty_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty $label at $path" >&2
    exit 1
  fi
}

require_release_dir() {
  local dir="$1"
  local label="$2"
  local required=(
    "model.safetensors"
    "config.json"
    "generation_config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "README.md"
  )
  local filename
  for filename in "${required[@]}"; do
    require_nonempty_file "$dir/$filename" "$label artifact $filename"
  done
}

stage_index() {
  case "$1" in
    pretrain) echo 10 ;;
    continued_pretrain) echo 20 ;;
    export_base) echo 30 ;;
    chat_sft) echo 40 ;;
    reasoning_sft) echo 50 ;;
    preferences) echo 60 ;;
    reward) echo 70 ;;
    think_dpo) echo 80 ;;
    eval) echo 90 ;;
    *)
      echo "Unknown Metis-1.5 pipeline stage: $1" >&2
      exit 1
      ;;
  esac
}

should_run_stage() {
  local stage="$1"
  local start_idx
  local stage_idx
  start_idx="$(stage_index "$START_STAGE")"
  stage_idx="$(stage_index "$stage")"
  if [[ "$stage_idx" -lt "$start_idx" ]]; then
    return 1
  fi
  if [[ -n "$STOP_AFTER_STAGE" ]]; then
    local stop_idx
    stop_idx="$(stage_index "$STOP_AFTER_STAGE")"
    if [[ "$stage_idx" -gt "$stop_idx" ]]; then
      return 1
    fi
  fi
  return 0
}

stage_before_start() {
  if [[ "$SKIP_PRIOR_STAGE_CHECKS" == "1" ]]; then
    return 1
  fi
  local stage="$1"
  local start_idx
  local stage_idx
  start_idx="$(stage_index "$START_STAGE")"
  stage_idx="$(stage_index "$stage")"
  [[ "$stage_idx" -lt "$start_idx" ]]
}

download_dir_from_s3() {
  local s3_uri="$1"
  local local_dir="$2"
  local optional="${3:-0}"
  if [[ -z "$s3_uri" ]]; then
    return 0
  fi
  if [[ "$optional" == "1" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$s3_uri" --local-dir "$local_dir" --optional >/dev/null || true
  else
    "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" download-dir --s3-uri "$s3_uri" --local-dir "$local_dir" >/dev/null
  fi
}

upload_dir_to_s3() {
  local local_dir="$1"
  local s3_uri="$2"
  local label="$3"
  if [[ "$DISABLE_UPLOADS" == "1" ]]; then
    log "Skipping $label upload because METIS15_DISABLE_UPLOADS=1"
    return 0
  fi
  if [[ -z "$s3_uri" || ! -d "$local_dir" ]]; then
    return 0
  fi
  log "Uploading $label to $s3_uri"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" upload-dir --local-dir "$local_dir" --s3-uri "$s3_uri" >/dev/null
}

upload_file_to_s3() {
  local local_path="$1"
  local s3_uri="$2"
  local label="$3"
  if [[ "$DISABLE_UPLOADS" == "1" ]]; then
    log "Skipping $label upload because METIS15_DISABLE_UPLOADS=1"
    return 0
  fi
  if [[ -z "$s3_uri" || ! -f "$local_path" ]]; then
    return 0
  fi
  log "Uploading $label to $s3_uri"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/s3_artifacts.py" upload-file --local-path "$local_path" --s3-uri "$s3_uri" >/dev/null
}

ensure_dir_from_s3_if_missing() {
  local local_dir="$1"
  local required_rel="$2"
  local s3_uri="$3"
  local label="$4"
  if [[ -f "$local_dir/$required_rel" || -z "$s3_uri" ]]; then
    return 0
  fi
  log "Hydrating $label from S3."
  rm -rf "$local_dir"
  download_dir_from_s3 "$s3_uri" "$local_dir" 1
}

sft_source_count() {
  local local_dir="$1"
  local source_name="$2"
  "$PYTHON_BIN" - "$local_dir/meta.json" "$source_name" <<'PY'
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
source_name = sys.argv[2]
if not meta_path.is_file():
    print(0)
    raise SystemExit

meta = json.loads(meta_path.read_text(encoding="utf-8"))
print(int((meta.get("source_counts") or {}).get(source_name, 0)))
PY
}

ensure_chat_identity_source() {
  local output_dir="$1"
  if [[ -z "$REQUIRED_CHAT_IDENTITY_SOURCE" || "$REQUIRED_CHAT_IDENTITY_COUNT" -le 0 ]]; then
    return 0
  fi
  if [[ ! -f "$output_dir/meta.json" ]]; then
    return 0
  fi
  local actual_count
  actual_count="$(sft_source_count "$output_dir" "$REQUIRED_CHAT_IDENTITY_SOURCE")"
  if [[ "$actual_count" -ge "$REQUIRED_CHAT_IDENTITY_COUNT" ]]; then
    return 0
  fi
  log "Existing chat SFT data is missing required identity source $REQUIRED_CHAT_IDENTITY_SOURCE ($actual_count/$REQUIRED_CHAT_IDENTITY_COUNT); rebuilding."
  rm -rf "$output_dir"
}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Metis manifest not found: $MANIFEST" >&2
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is required for the Metis AWS launcher." >&2
  exit 1
fi

MANIFEST_ATTENTION_BACKEND="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(manifest.get("model", {}).get("attention_backend", "auto"))
PY
)"
MANIFEST_FP8_ENABLED="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(bool(manifest.get("hardware", {}).get("fp8", {}).get("enabled", False))))
PY
)"
MANIFEST_NVFP4_ENABLED="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(manifest.get("model", {}).get("low_precision_mode") == "nvfp4" or bool(manifest.get("hardware", {}).get("nvfp4", {}).get("enabled", False))))
PY
)"
MANIFEST_NVFP4_DISABLE_RHT="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
model = manifest.get("model", {})
hardware = manifest.get("hardware", {}).get("nvfp4", {})
print(int(bool(model.get("nvfp4_disable_rht", hardware.get("disable_rht", False)))))
PY
)"
MANIFEST_NVFP4_DISABLE_2D="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
model = manifest.get("model", {})
hardware = manifest.get("hardware", {}).get("nvfp4", {})
print(int(bool(model.get("nvfp4_disable_2d_quantization", hardware.get("disable_2d_quantization", False)))))
PY
)"
MANIFEST_NVFP4_DISABLE_STOCHASTIC="$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
model = manifest.get("model", {})
hardware = manifest.get("hardware", {}).get("nvfp4", {})
print(int(bool(model.get("nvfp4_disable_stochastic_rounding", hardware.get("disable_stochastic_rounding", False)))))
PY
)"
if [[ -z "${METIS15_NVFP4+x}" ]]; then
  ENABLE_NVFP4="$MANIFEST_NVFP4_ENABLED"
fi
if [[ "$ENABLE_FP8" != "0" ]]; then
  ENABLE_NVFP4=0
  FP8_EXPERT_PRECISION="${FP8_EXPERT_PRECISION:-bf16}"
  MOE_DISPATCH_MODE="${MOE_DISPATCH_MODE:-bucketed}"
fi
if [[ -n "$MOE_STATIC_CAPACITY" || "$MOE_GRAPHABLE" == "1" ]]; then
  MOE_DISPATCH_MODE="${MOE_DISPATCH_MODE:-bucketed}"
fi

if [[ "$MANIFEST_ATTENTION_BACKEND" == "flash_attention_3" ]] && ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    try:
        from flash_attn_3 import flash_attn_interface  # noqa: F401
    except ImportError:
        from hopper import flash_attn_interface  # noqa: F401
PY
then
  cat >&2 <<'EOF'
FlashAttention-3 is not installed.

Install a FlashAttention build that explicitly supports the active GPU before
launching, or leave the manifest on the default SDPA/Transformer Engine sweep path.
EOF
  exit 1
fi

if [[ ( "$ENABLE_FP8" != "0" && "$MANIFEST_FP8_ENABLED" == "1" ) || ( "$ENABLE_NVFP4" != "0" && "$MANIFEST_NVFP4_ENABLED" == "1" ) ]]; then
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import transformer_engine.pytorch  # noqa: F401
PY
  then
    cat >&2 <<'EOF'
Transformer Engine is not installed.

Install the official PyTorch package before launching low-precision training:
  pip install --no-build-isolation transformer_engine[pytorch]
EOF
    exit 1
  fi
fi

if [[ -n "$S3_ROOT" ]]; then
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import boto3  # noqa: F401
PY
  then
    cat >&2 <<'EOF'
The Metis S3-backed launcher requires boto3 in the active environment.

Install the GPU requirements bundle before launching:
  pip install -r requirements-gpu-train.txt
EOF
    exit 1
  fi
fi

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVTE_DEBUG="${NVTE_DEBUG:-0}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
export METIS_TORCH_GROUPED_SAFE_SYNC="${METIS_TORCH_GROUPED_SAFE_SYNC:-0}"
export METIS_ASYNC_METRICS="${METIS_ASYNC_METRICS:-1}"

eval "$("$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json
import math
import os
import shlex
import sys

manifest = json.load(open(sys.argv[1], "r", encoding="utf-8"))
model = manifest["model"]
hardware = manifest["hardware"]
data_manifests = manifest["data_manifests"]
strategy = manifest["selected_data_strategy"]
pretrain = manifest["pretrain"]
continued = manifest["continued_pretrain"]
chat = manifest["chat_sft"]
reasoning = manifest["reasoning_sft"]
preference = manifest["preference_optimization"]

world_size = int(hardware.get("world_size", 1))
seq_len = int(model["block_size"])

pretrain_mix = data_manifests["pretrain_best_research"]
if strategy.get("pretrain") == "release_clean":
    pretrain_mix = data_manifests["pretrain_release_clean"]

def token_stage(target_tokens: int, local_batch: int, grad_accum: int, warmup_ratio: float):
    tokens_per_step = world_size * local_batch * grad_accum * seq_len
    steps = max(1, math.ceil(target_tokens / tokens_per_step))
    warmup_steps = max(1, round(steps * warmup_ratio))
    return tokens_per_step, steps, warmup_steps

def example_stage(target_examples: int, epochs: float, local_batch: int, grad_accum: int, warmup_ratio: float):
    global_batch = world_size * local_batch * grad_accum
    steps = max(1, math.ceil((target_examples * epochs) / global_batch))
    warmup_steps = max(1, round(steps * warmup_ratio))
    return global_batch, steps, warmup_steps

pretrain_tps, pretrain_steps, pretrain_warmup = token_stage(
    int(pretrain["target_train_tokens"]),
    int(pretrain["local_batch_size"]),
    int(pretrain["grad_accum_steps"]),
    float(pretrain["warmup_ratio"]),
)
continued_tps, continued_steps, continued_warmup = token_stage(
    int(continued["target_train_tokens"]),
    int(continued["local_batch_size"]),
    int(continued["grad_accum_steps"]),
    float(continued["warmup_ratio"]),
)
chat_gbs, chat_steps, chat_warmup = example_stage(
    int(chat["target_examples"]),
    float(chat["epochs"]),
    int(chat["local_batch_size"]),
    int(chat["grad_accum_steps"]),
    float(chat["warmup_ratio"]),
)
reasoning_gbs, reasoning_steps, reasoning_warmup = example_stage(
    int(reasoning["target_examples"]),
    float(reasoning["epochs"]),
    int(reasoning["local_batch_size"]),
    int(reasoning["grad_accum_steps"]),
    float(reasoning["warmup_ratio"]),
)
reward_gbs, reward_steps, reward_warmup = example_stage(
    int(preference["reward_model_examples"]),
    float(preference["epochs"]),
    int(preference["local_batch_size"]),
    int(preference["grad_accum_steps"]),
    float(preference["warmup_ratio"]),
)
dpo_gbs, dpo_steps, dpo_warmup = example_stage(
    int(preference["target_pairs"]),
    float(preference["epochs"]),
    int(preference["local_batch_size"]),
    int(preference["grad_accum_steps"]),
    float(preference["warmup_ratio"]),
)

on_policy_in_target = bool(preference.get("on_policy_in_target", False))
reward_chat_share = float(preference.get("on_policy_chat_negative_share", 0.05))
reward_think_share = float(preference.get("on_policy_think_negative_share", 0.03))
dpo_chat_share = float(preference.get("on_policy_chat_negative_share", 0.05))
dpo_think_share = float(preference.get("on_policy_think_negative_share", 0.03))
reward_chat_negative_pairs_default = max(0, round(int(preference["reward_model_examples"]) * reward_chat_share))
reward_think_negative_pairs_default = max(0, round(int(preference["reward_model_examples"]) * reward_think_share))
dpo_chat_negative_pairs_default = max(0, round(int(preference["target_pairs"]) * dpo_chat_share))
dpo_think_negative_pairs_default = max(0, round(int(preference["target_pairs"]) * dpo_think_share))
if on_policy_in_target:
    reward_bootstrap_examples_default = max(
        1,
        int(preference.get(
            "bootstrap_reward_model_examples",
            int(preference["reward_model_examples"]) - reward_chat_negative_pairs_default - reward_think_negative_pairs_default,
        )),
    )
    dpo_bootstrap_pairs_default = max(
        1,
        int(preference.get(
            "bootstrap_target_pairs",
            int(preference["target_pairs"]) - dpo_chat_negative_pairs_default - dpo_think_negative_pairs_default,
        )),
    )
else:
    reward_bootstrap_examples_default = int(preference["reward_model_examples"])
    dpo_bootstrap_pairs_default = int(preference["target_pairs"])

pretrain_val_tokens = max(1, round(int(pretrain["target_train_tokens"]) * float(pretrain["val_ratio"]) / max(1.0 - float(pretrain["val_ratio"]), 1e-9)))
continued_val_tokens = max(1, round(int(continued["target_train_tokens"]) * float(continued["val_ratio"]) / max(1.0 - float(continued["val_ratio"]), 1e-9)))
data_prep = manifest.get("data_prep", {})
release_repos = manifest.get("release", {}).get("repos", {})
model_name_slug = str(manifest.get("name", model.get("name", "Metis-1.5"))).replace(" ", "-")
if release_repos:
    base_release_repo = release_repos.get("base", "")
    think_release_repo = release_repos.get("think", "")
else:
    base_release_repo = f"Lernex/{model_name_slug}-base"
    think_release_repo = f"Lernex/{model_name_slug}-think"

values = {
    "MODEL_BLOCK_SIZE": seq_len,
    "WORLD_SIZE": world_size,
    "PRETRAIN_MIX": pretrain_mix,
    "CONTINUED_MIX": data_manifests["continued_pretrain"],
    "CHAT_MIX": data_manifests["chat_sft"],
    "REASONING_MIX": data_manifests["reasoning_sft"],
    "PREFERENCE_MIX": data_manifests["preference"],
    "PRETRAIN_MAX_DOCS_MANIFEST": int(data_prep.get("pretrain_max_docs", 25000000)),
    "CONTINUED_MAX_DOCS_MANIFEST": int(data_prep.get("continued_max_docs", 3000000)),
    "PRETRAIN_TARGET_TOKENS": int(pretrain["target_train_tokens"]),
    "PRETRAIN_VAL_TOKENS": pretrain_val_tokens,
    "PRETRAIN_BATCH": int(pretrain["local_batch_size"]),
    "PRETRAIN_GRAD_ACCUM": int(pretrain["grad_accum_steps"]),
    "PRETRAIN_STEPS": pretrain_steps,
    "PRETRAIN_WARMUP": pretrain_warmup,
    "PRETRAIN_LR": pretrain["base_lr"],
    "PRETRAIN_WEIGHT_DECAY": pretrain["weight_decay"],
    "PRETRAIN_BETA1": pretrain["optimizer_beta1"],
    "PRETRAIN_BETA2": pretrain["optimizer_beta2"],
    "PRETRAIN_LOG_INTERVAL": pretrain.get("log_interval", 20),
    "PRETRAIN_EVAL_INTERVAL": pretrain.get("eval_interval", 1000),
    "PRETRAIN_CHECKPOINT_INTERVAL": pretrain["checkpoint_interval"],
    "CONTINUED_TARGET_TOKENS": int(continued["target_train_tokens"]),
    "CONTINUED_VAL_TOKENS": continued_val_tokens,
    "CONTINUED_BATCH": int(continued["local_batch_size"]),
    "CONTINUED_GRAD_ACCUM": int(continued["grad_accum_steps"]),
    "CONTINUED_STEPS": continued_steps,
    "CONTINUED_WARMUP": continued_warmup,
    "CONTINUED_LR": continued["base_lr"],
    "CONTINUED_WEIGHT_DECAY": continued["weight_decay"],
    "CONTINUED_BETA1": continued["optimizer_beta1"],
    "CONTINUED_BETA2": continued["optimizer_beta2"],
    "CONTINUED_CHECKPOINT_INTERVAL": continued["checkpoint_interval"],
    "CHAT_EXAMPLES": int(chat["target_examples"]),
    "CHAT_BATCH": int(chat["local_batch_size"]),
    "CHAT_GRAD_ACCUM": int(chat["grad_accum_steps"]),
    "CHAT_EPOCHS": chat["epochs"],
    "CHAT_LR": chat["base_lr"],
    "CHAT_WARMUP": chat_warmup,
    "CHAT_CHECKPOINT_INTERVAL": chat["checkpoint_interval"],
    "CHAT_MAX_LENGTH": chat["max_length"],
    "REASONING_EXAMPLES": int(reasoning["target_examples"]),
    "REASONING_BATCH": int(reasoning["local_batch_size"]),
    "REASONING_GRAD_ACCUM": int(reasoning["grad_accum_steps"]),
    "REASONING_EPOCHS": reasoning["epochs"],
    "REASONING_LR": reasoning["base_lr"],
    "REASONING_WARMUP": reasoning_warmup,
    "REASONING_CHECKPOINT_INTERVAL": reasoning["checkpoint_interval"],
    "REASONING_MAX_LENGTH": reasoning["max_length"],
    "REASONING_MAX_THINK_CHARS": reasoning.get("recommended_max_think_chars", 320),
    "REASONING_MAX_ANSWER_CHARS": reasoning.get("recommended_max_answer_chars", 320),
    "REWARD_EXAMPLES": int(preference["reward_model_examples"]),
    "REWARD_BATCH": int(preference["local_batch_size"]),
    "REWARD_GRAD_ACCUM": int(preference["grad_accum_steps"]),
    "REWARD_LR": preference["base_lr"],
    "REWARD_WARMUP": reward_warmup,
    "REWARD_CHECKPOINT_INTERVAL": preference["checkpoint_interval"],
    "REWARD_BOOTSTRAP_EXAMPLES": reward_bootstrap_examples_default,
    "REWARD_CHAT_NEGATIVE_PAIRS_DEFAULT": reward_chat_negative_pairs_default,
    "REWARD_THINK_NEGATIVE_PAIRS_DEFAULT": reward_think_negative_pairs_default,
    "DPO_PAIRS": int(preference["target_pairs"]),
    "DPO_BOOTSTRAP_PAIRS": dpo_bootstrap_pairs_default,
    "DPO_BATCH": int(preference["local_batch_size"]),
    "DPO_GRAD_ACCUM": int(preference["grad_accum_steps"]),
    "DPO_LR": preference["base_lr"],
    "DPO_WARMUP": dpo_warmup,
    "DPO_CHECKPOINT_INTERVAL": preference["checkpoint_interval"],
    "DPO_CHAT_NEGATIVE_PAIRS_DEFAULT": dpo_chat_negative_pairs_default,
    "DPO_THINK_NEGATIVE_PAIRS_DEFAULT": dpo_think_negative_pairs_default,
    "DPO_BETA": preference["beta"],
    "MODEL_DISPLAY_NAME": manifest.get("name", "Metis-1.5"),
    "MOE_EXPERT_PARALLEL_SIZE": os.environ.get(
        "METIS15_MOE_EXPERT_PARALLEL_SIZE",
        model.get("moe_expert_parallel_size", 1),
    ),
    "MANIFEST_MOE_MEMORY_EFFICIENT_PERMUTATION": int(bool(model.get("moe_memory_efficient_permutation", False))),
    "MANIFEST_MOE_PERMUTE_FUSION": int(bool(model.get("moe_permute_fusion", True))),
    "BASE_RELEASE_REPO": base_release_repo,
    "THINK_RELEASE_REPO": think_release_repo,
    "MODEL_PREFERRED_PRECISION": hardware.get("preferred_precision", "fp8_hybrid"),
    "MODEL_FALLBACK_PRECISION": hardware.get("fallback_precision", "bf16"),
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
MOE_MEMORY_EFFICIENT_PERMUTATION="${MOE_MEMORY_EFFICIENT_PERMUTATION:-$MANIFEST_MOE_MEMORY_EFFICIENT_PERMUTATION}"
MOE_PERMUTE_FUSION="${MOE_PERMUTE_FUSION:-$MANIFEST_MOE_PERMUTE_FUSION}"

NPROC="${METIS15_NPROC:-$WORLD_SIZE}"
CHAT_CHECKPOINT_INTERVAL="${METIS15_CHAT_CHECKPOINT_INTERVAL:-$CHAT_CHECKPOINT_INTERVAL}"
REASONING_CHECKPOINT_INTERVAL="${METIS15_REASONING_CHECKPOINT_INTERVAL:-$REASONING_CHECKPOINT_INTERVAL}"
REWARD_BATCH="${METIS15_REWARD_BATCH:-$REWARD_BATCH}"
REWARD_GRAD_ACCUM="${METIS15_REWARD_GRAD_ACCUM:-$REWARD_GRAD_ACCUM}"
REWARD_CHECKPOINT_INTERVAL="${METIS15_REWARD_CHECKPOINT_INTERVAL:-$REWARD_CHECKPOINT_INTERVAL}"
DPO_BATCH="${METIS15_DPO_BATCH:-$DPO_BATCH}"
DPO_GRAD_ACCUM="${METIS15_DPO_GRAD_ACCUM:-$DPO_GRAD_ACCUM}"
DPO_CHECKPOINT_INTERVAL="${METIS15_DPO_CHECKPOINT_INTERVAL:-$DPO_CHECKPOINT_INTERVAL}"
PRETRAIN_MAX_DOCS="${METIS15_PRETRAIN_MAX_DOCS:-$PRETRAIN_MAX_DOCS_MANIFEST}"
CONTINUED_MAX_DOCS="${METIS15_CONTINUED_MAX_DOCS:-$CONTINUED_MAX_DOCS_MANIFEST}"

mkdir -p "$RUNS_ROOT" "$RELEASES_ROOT"

REWARD_BOOTSTRAP_EXAMPLES="${METIS15_REWARD_BOOTSTRAP_EXAMPLES:-$REWARD_BOOTSTRAP_EXAMPLES}"
DPO_BOOTSTRAP_PAIRS="${METIS15_DPO_BOOTSTRAP_PAIRS:-$DPO_BOOTSTRAP_PAIRS}"
REWARD_CHAT_NEGATIVE_PAIRS="${METIS15_REWARD_CUSTOM_CHAT_NEGATIVE_PAIRS:-$REWARD_CHAT_NEGATIVE_PAIRS_DEFAULT}"
REWARD_THINK_NEGATIVE_PAIRS="${METIS15_REWARD_CUSTOM_THINK_NEGATIVE_PAIRS:-$REWARD_THINK_NEGATIVE_PAIRS_DEFAULT}"
DPO_CHAT_NEGATIVE_PAIRS="${METIS15_DPO_CUSTOM_CHAT_NEGATIVE_PAIRS:-$DPO_CHAT_NEGATIVE_PAIRS_DEFAULT}"
DPO_THINK_NEGATIVE_PAIRS="${METIS15_DPO_CUSTOM_THINK_NEGATIVE_PAIRS:-$DPO_THINK_NEGATIVE_PAIRS_DEFAULT}"

ensure_dir_from_s3_if_missing "$ASSETS_DIR" "tokenizer.json" "$TOKENIZER_S3_URI" "Metis-1.5 tokenizer assets"

if [[ ! -f "$ASSETS_DIR/tokenizer.json" ]]; then
  echo "Tokenizer assets not found at $ASSETS_DIR/tokenizer.json" >&2
  exit 1
fi

if [[ ! -f "$ASSETS_DIR/config.json" ]]; then
  log "Rendering Metis-1.5 HF assets."
  "$PYTHON_BIN" "$ROOT_DIR/scripts/render_metis13_hf_assets.py" \
    --manifest "$MANIFEST" \
    --tokenizer-dir "$ASSETS_DIR" \
    --output-dir "$ASSETS_DIR"
fi

run_train() {
  if [[ "$NPROC" -gt 1 ]]; then
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

append_common_train_flags() {
  local -n ref_array="$1"
  ref_array+=(--dtype "${METIS15_DTYPE:-$MODEL_FALLBACK_PRECISION}" --fused-adamw --matmul-precision "$MATMUL_PRECISION")
  if [[ "$TF32_FLAG" == "1" ]]; then
    ref_array+=(--tf32)
  fi
  if [[ "$COMPILE_FLAG" == "1" ]]; then
    ref_array+=(--compile --compile-mode "$COMPILE_MODE")
  fi
  if [[ -n "$OPTIMIZER_NAME" ]]; then
    ref_array+=(--optimizer "$OPTIMIZER_NAME")
  fi
  if [[ "${METIS15_MUON_INCLUDE_ROUTED_EXPERTS:-0}" == "1" ]]; then
    ref_array+=(--muon-include-routed-experts)
  fi
  if [[ -n "${METIS15_MUON_BETA:-}" ]]; then
    ref_array+=(--muon-beta "$METIS15_MUON_BETA")
  fi
  if [[ -n "${METIS15_MUON_NS_STEPS:-}" ]]; then
    ref_array+=(--muon-ns-steps "$METIS15_MUON_NS_STEPS")
  fi
  if [[ -n "${METIS15_MUON_LR_SCALE:-}" ]]; then
    ref_array+=(--muon-lr-scale "$METIS15_MUON_LR_SCALE")
  fi
  if [[ "$ENABLE_NVFP4" != "0" && "$MANIFEST_NVFP4_ENABLED" == "1" ]]; then
    ref_array+=(--nvfp4)
    if [[ "$MANIFEST_NVFP4_DISABLE_RHT" == "1" ]]; then
      ref_array+=(--nvfp4-disable-rht)
    fi
    if [[ "$MANIFEST_NVFP4_DISABLE_2D" == "1" ]]; then
      ref_array+=(--nvfp4-disable-2d-quantization)
    fi
    if [[ "$MANIFEST_NVFP4_DISABLE_STOCHASTIC" == "1" ]]; then
      ref_array+=(--nvfp4-disable-stochastic-rounding)
    fi
  elif [[ "$ENABLE_FP8" != "0" && "$MANIFEST_FP8_ENABLED" == "1" ]]; then
    ref_array+=(--fp8 --fp8-format HYBRID --fp8-margin 0 --fp8-amax-history-len 16 --fp8-amax-compute-algo max)
    if [[ -n "$FP8_EXPERT_PRECISION" ]]; then
      ref_array+=(--fp8-expert-precision "$FP8_EXPERT_PRECISION")
    fi
  fi
}

prepare_streaming_stage() {
  local label="$1"
  local mixture_config="$2"
  local output_dir="$3"
  local max_docs="$4"
  local target_tokens="$5"
  local val_tokens="$6"
  local s3_uri="${7:-}"

  if [[ -f "$output_dir/meta.json" ]]; then
    log "$label data already prepared at $output_dir"
    return 0
  fi
  ensure_dir_from_s3_if_missing "$output_dir" "meta.json" "$s3_uri" "$label data"
  if [[ -f "$output_dir/meta.json" ]]; then
    log "$label data hydrated from S3 into $output_dir"
    return 0
  fi
  if [[ "$ENABLE_PREP" == "0" ]]; then
    echo "$label data missing at $output_dir and METIS15_ENABLE_PREP=0" >&2
    exit 1
  fi

  log "Preparing $label data with $mixture_config"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_streaming_data.py" \
    --mixture-config "$mixture_config" \
    --tokenizer-path "$ASSETS_DIR/tokenizer.json" \
    --output-dir "$output_dir" \
    --max-docs "$max_docs" \
    --target-train-tokens "$target_tokens" \
    --target-val-tokens "$val_tokens" \
    --val-ratio 0.01
  require_nonempty_file "$output_dir/meta.json" "$label meta.json"
  upload_dir_to_s3 "$output_dir" "$s3_uri" "$label data"
}

prepare_sft_stage() {
  local label="$1"
  local mixture_config="$2"
  local output_dir="$3"
  local total_examples="$4"
  local s3_uri="$5"
  shift 5

  if [[ "$label" == "chat SFT" ]]; then
    ensure_chat_identity_source "$output_dir"
  fi
  if [[ -f "$output_dir/meta.json" ]]; then
    log "$label data already prepared at $output_dir"
    return 0
  fi
  ensure_dir_from_s3_if_missing "$output_dir" "meta.json" "$s3_uri" "$label data"
  if [[ "$label" == "chat SFT" ]]; then
    ensure_chat_identity_source "$output_dir"
  fi
  if [[ -f "$output_dir/meta.json" ]]; then
    log "$label data hydrated from S3 into $output_dir"
    return 0
  fi
  if [[ "$ENABLE_PREP" == "0" ]]; then
    echo "$label data missing at $output_dir and METIS15_ENABLE_PREP=0" >&2
    exit 1
  fi

  log "Preparing $label data with $mixture_config"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_metis13_sft_data.py" \
    --mixture-config "$mixture_config" \
    --output-dir "$output_dir" \
    --total-examples "$total_examples" \
    "$@"
  require_nonempty_file "$output_dir/meta.json" "$label meta.json"
  upload_dir_to_s3 "$output_dir" "$s3_uri" "$label data"
}

prepare_preference_stage() {
  local label="$1"
  local output_dir="$2"
  local total_pairs="$3"
  local s3_uri="$4"

  if [[ -f "$output_dir/meta.json" ]]; then
    log "$label data already prepared at $output_dir"
    return 0
  fi
  ensure_dir_from_s3_if_missing "$output_dir" "meta.json" "$s3_uri" "$label data"
  if [[ -f "$output_dir/meta.json" ]]; then
    log "$label data hydrated from S3 into $output_dir"
    return 0
  fi
  if [[ "$ENABLE_PREP" == "0" ]]; then
    echo "$label data missing at $output_dir and METIS15_ENABLE_PREP=0" >&2
    exit 1
  fi

  log "Preparing $label data with $PREFERENCE_MIX"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_metis15_preference_data.py" \
    --mixture-config "$PREFERENCE_MIX" \
    --output-dir "$output_dir" \
    --total-pairs "$total_pairs"
  require_nonempty_file "$output_dir/meta.json" "$label meta.json"
  upload_dir_to_s3 "$output_dir" "$s3_uri" "$label data"
}

augment_preference_stage() {
  local label="$1"
  local output_dir="$2"
  local s3_uri="$3"
  local chat_pairs="$4"
  local think_pairs="$5"

  if [[ "$ENABLE_CUSTOM_NEGATIVES" == "0" ]]; then
    return 0
  fi
  if [[ "$chat_pairs" -le 0 && "$think_pairs" -le 0 ]]; then
    return 0
  fi

  local cmd=(
    "$PYTHON_BIN"
    "$ROOT_DIR/scripts/mine_metis15_custom_negatives.py"
    --output-dir "$output_dir"
    --tokenizer-path "$ASSETS_DIR/tokenizer.json"
    --chat-seed-dir "$CHAT_DATA_DIR"
    --reasoning-seed-dir "$REASONING_DATA_DIR"
    --chat-checkpoint "$CHAT_RUN/best.pt"
    --think-checkpoint "$THINK_RUN/best.pt"
    --chat-target-pairs "$chat_pairs"
    --think-target-pairs "$think_pairs"
    --chat-max-new-tokens "$CUSTOM_NEGATIVE_CHAT_MAX_NEW_TOKENS"
    --think-max-new-tokens "$CUSTOM_NEGATIVE_THINK_MAX_NEW_TOKENS"
    --chat-temperature "$CUSTOM_NEGATIVE_CHAT_TEMPERATURE"
    --think-temperature "$CUSTOM_NEGATIVE_THINK_TEMPERATURE"
    --top-k "$CUSTOM_NEGATIVE_TOP_K"
  )
  if [[ -n "$CUSTOM_NEGATIVE_DEVICE" ]]; then
    cmd+=(--device "$CUSTOM_NEGATIVE_DEVICE")
  fi
  if [[ -n "$CUSTOM_NEGATIVE_MAX_SEED_ROWS" ]]; then
    cmd+=(--max-seed-rows-per-split "$CUSTOM_NEGATIVE_MAX_SEED_ROWS")
  fi

  log "Mining custom negatives for $label"
  "${cmd[@]}"
  require_nonempty_file "$output_dir/custom_negatives_meta.json" "$label custom negative meta"
  upload_dir_to_s3 "$output_dir" "$s3_uri" "$label data"
}

train_base_stage() {
  local stage_name="$1"
  local data_dir="$2"
  local out_dir="$3"
  local batch_size="$4"
  local grad_accum="$5"
  local max_steps="$6"
  local warmup_steps="$7"
  local lr="$8"
  local weight_decay="$9"
  local beta1="${10}"
  local beta2="${11}"
  local eval_interval="${12}"
  local checkpoint_interval="${13}"
  local init_checkpoint="${14:-}"
  local checkpoint_s3_uri="${15:-}"
  local train_stage="${16:-pretrain}"
  local training_mode="${17:-}"
  local use_te_fused_mlp="${18:-0}"

  if [[ "$RESUME_TRAINING" != "0" && ! -f "$out_dir/latest.pt" ]]; then
    download_dir_from_s3 "$checkpoint_s3_uri" "$out_dir" 1
  fi

  local cmd=(
    "$ROOT_DIR/scripts/train_mamba_lm.py"
    --manifest "$MANIFEST"
    --train-stage "$train_stage"
    --data-dir "$data_dir"
    --out-dir "$out_dir"
    --batch-size "$batch_size"
    --grad-accum-steps "$grad_accum"
    --max-steps "$max_steps"
    --warmup-steps "$warmup_steps"
    --lr "$lr"
    --weight-decay "$weight_decay"
    --beta1 "$beta1"
    --beta2 "$beta2"
    --log-interval "$PRETRAIN_LOG_INTERVAL"
    --eval-interval "$eval_interval"
    --checkpoint-interval "$checkpoint_interval"
  )
  if [[ -n "$init_checkpoint" ]]; then
    cmd+=(--init-checkpoint "$init_checkpoint")
  fi
  if [[ "$RESUME_TRAINING" != "0" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "$training_mode" ]]; then
    cmd+=(--training-mode "$training_mode")
  fi
  if [[ "$use_te_fused_mlp" == "1" ]]; then
    cmd+=(--te-fused-mlp)
  fi
  if [[ "${METIS15_TE_DOT_PRODUCT_ATTENTION:-0}" == "1" ]]; then
    cmd+=(--te-dot-product-attention)
  fi
  if [[ "${METIS15_DISABLE_NATIVE_GQA_ATTENTION:-0}" == "1" ]]; then
    cmd+=(--disable-native-gqa-attention)
  fi
  if [[ -n "$LM_LOSS_IMPL" ]]; then
    cmd+=(--lm-loss-impl "$LM_LOSS_IMPL")
  fi
  if [[ "$RETAIN_STANDARD_CE_LOGITS" == "1" ]]; then
    cmd+=(--retain-standard-ce-logits)
  fi
  if [[ "$COMPILE_FLAG" == "1" && "$COMPILE_LOW_PRECISION" == "1" ]]; then
    cmd+=(--allow-low-precision-compile)
  fi
  cmd+=(--prefetch-batches "$PREFETCH_BATCHES")
  if [[ -n "$MOE_DISPATCH_MODE" ]]; then
    cmd+=(--moe-dispatch-mode "$MOE_DISPATCH_MODE")
  fi
  if [[ -n "$MOE_BACKEND" ]]; then
    cmd+=(--moe-backend "$MOE_BACKEND")
  fi
  if [[ -n "$MOE_STATIC_CAPACITY" ]]; then
    cmd+=(--moe-static-capacity "$MOE_STATIC_CAPACITY")
  fi
  if [[ -n "$MOE_CAPACITY_FACTOR" ]]; then
    cmd+=(--moe-capacity-factor "$MOE_CAPACITY_FACTOR")
  fi
  if [[ -n "$MOE_CAPACITY_ALIGNMENT" ]]; then
    cmd+=(--moe-capacity-alignment "$MOE_CAPACITY_ALIGNMENT")
  fi
  if [[ -n "$MOE_OVERFLOW_MODE" ]]; then
    cmd+=(--moe-overflow-mode "$MOE_OVERFLOW_MODE")
  fi
  if [[ "$MOE_FUSED_COMBINE" == "0" ]]; then
    cmd+=(--disable-moe-fused-combine)
  fi
  if [[ "$MOE_GRAPHABLE" == "1" ]]; then
    cmd+=(--moe-graphable)
  fi
  if [[ "$MOE_MEMORY_EFFICIENT_PERMUTATION" == "1" ]]; then
    cmd+=(--moe-memory-efficient-permutation)
  fi
  if [[ "$MOE_PERMUTE_FUSION" == "0" ]]; then
    cmd+=(--disable-moe-permute-fusion)
  fi
  if [[ "$MOE_EXPERT_PARALLEL_SIZE" != "1" ]]; then
    cmd+=(--moe-expert-parallel-size "$MOE_EXPERT_PARALLEL_SIZE")
  fi
  if [[ -n "$NVFP4_FINAL_EXPERT_LAYERS" ]]; then
    cmd+=(--nvfp4-final-expert-layers "$NVFP4_FINAL_EXPERT_LAYERS")
  fi
  if [[ -n "$NVFP4_FINAL_EXPERT_PRECISION" ]]; then
    cmd+=(--nvfp4-final-expert-precision "$NVFP4_FINAL_EXPERT_PRECISION")
  fi
  append_common_train_flags cmd
  log "Training $stage_name"
  run_train "${cmd[@]}"
  require_nonempty_file "$out_dir/best.pt" "$stage_name best checkpoint"
  require_nonempty_file "$out_dir/latest.pt" "$stage_name latest checkpoint"
  upload_dir_to_s3 "$out_dir" "$checkpoint_s3_uri" "$stage_name checkpoints"
}

train_sft_stage() {
  local stage_name="$1"
  local base_checkpoint="$2"
  local train_jsonl="$3"
  local val_jsonl="$4"
  local out_dir="$5"
  local max_length="$6"
  local batch_size="$7"
  local grad_accum="$8"
  local epochs="$9"
  local lr="${10}"
  local warmup_steps="${11}"
  local checkpoint_interval="${12}"
  local checkpoint_s3_uri="${13:-}"

  if [[ "$RESUME_TRAINING" != "0" && ! -f "$out_dir/latest.pt" ]]; then
    download_dir_from_s3 "$checkpoint_s3_uri" "$out_dir" 1
  fi

  local cmd=(
    "$ROOT_DIR/scripts/train_mamba_sft.py"
    --base-checkpoint "$base_checkpoint"
    --train-jsonl "$train_jsonl"
    --val-jsonl "$val_jsonl"
    --tokenizer-path "$ASSETS_DIR/tokenizer.json"
    --out-dir "$out_dir"
    --max-length "$max_length"
    --batch-size "$batch_size"
    --grad-accum-steps "$grad_accum"
    --epochs "$epochs"
    --lr "$lr"
    --warmup-steps "$warmup_steps"
    --eval-interval "$checkpoint_interval"
    --checkpoint-interval "$checkpoint_interval"
    --num-workers "$SFT_NUM_WORKERS"
  )
  if [[ "$RESUME_TRAINING" != "0" ]]; then
    cmd+=(--resume)
  fi
  append_common_train_flags cmd
  log "Training $stage_name"
  run_train "${cmd[@]}"
  require_nonempty_file "$out_dir/best.pt" "$stage_name best checkpoint"
  require_nonempty_file "$out_dir/latest.pt" "$stage_name latest checkpoint"
  upload_dir_to_s3 "$out_dir" "$checkpoint_s3_uri" "$stage_name checkpoints"
}

train_reward_stage() {
  local base_checkpoint="$1"
  local out_dir="$2"
  local checkpoint_s3_uri="$3"
  if [[ "$RESUME_TRAINING" != "0" && ! -f "$out_dir/latest.pt" ]]; then
    download_dir_from_s3 "$checkpoint_s3_uri" "$out_dir" 1
  fi
  local cmd=(
    "$ROOT_DIR/scripts/train_mamba_reward.py"
    --base-checkpoint "$base_checkpoint"
    --train-jsonl "$REWARD_PREF_DIR/train.jsonl"
    --val-jsonl "$REWARD_PREF_DIR/val.jsonl"
    --tokenizer-path "$ASSETS_DIR/tokenizer.json"
    --out-dir "$out_dir"
    --max-length "$MODEL_BLOCK_SIZE"
    --batch-size "$REWARD_BATCH"
    --grad-accum-steps "$REWARD_GRAD_ACCUM"
    --epochs 1.0
    --lr "$REWARD_LR"
    --warmup-steps "$REWARD_WARMUP"
    --eval-interval "$REWARD_CHECKPOINT_INTERVAL"
    --checkpoint-interval "$REWARD_CHECKPOINT_INTERVAL"
    --num-workers "$PREF_NUM_WORKERS"
  )
  if [[ "$RESUME_TRAINING" != "0" ]]; then
    cmd+=(--resume)
  fi
  append_common_train_flags cmd
  log "Training reward model"
  run_train "${cmd[@]}"
  require_nonempty_file "$out_dir/best.pt" "reward model best checkpoint"
  require_nonempty_file "$out_dir/latest.pt" "reward model latest checkpoint"
  upload_dir_to_s3 "$out_dir" "$checkpoint_s3_uri" "reward model checkpoints"
}

train_dpo_stage() {
  local stage_name="$1"
  local base_checkpoint="$2"
  local out_dir="$3"
  local checkpoint_s3_uri="$4"
  if [[ "$RESUME_TRAINING" != "0" && ! -f "$out_dir/latest.pt" ]]; then
    download_dir_from_s3 "$checkpoint_s3_uri" "$out_dir" 1
  fi
  local cmd=(
    "$ROOT_DIR/scripts/train_mamba_dpo.py"
    --base-checkpoint "$base_checkpoint"
    --reference-checkpoint "$base_checkpoint"
    --train-jsonl "$DPO_PREF_DIR/train.jsonl"
    --val-jsonl "$DPO_PREF_DIR/val.jsonl"
    --tokenizer-path "$ASSETS_DIR/tokenizer.json"
    --out-dir "$out_dir"
    --max-length "$MODEL_BLOCK_SIZE"
    --batch-size "$DPO_BATCH"
    --grad-accum-steps "$DPO_GRAD_ACCUM"
    --epochs 1.0
    --lr "$DPO_LR"
    --warmup-steps "$DPO_WARMUP"
    --dpo-beta "$DPO_BETA"
    --eval-interval "$DPO_CHECKPOINT_INTERVAL"
    --checkpoint-interval "$DPO_CHECKPOINT_INTERVAL"
    --num-workers "$PREF_NUM_WORKERS"
    --sequence-score-mode mean
  )
  if [[ "$RESUME_TRAINING" != "0" ]]; then
    cmd+=(--resume)
  fi
  append_common_train_flags cmd
  log "Training $stage_name"
  run_train "${cmd[@]}"
  require_nonempty_file "$out_dir/best.pt" "$stage_name best checkpoint"
  require_nonempty_file "$out_dir/latest.pt" "$stage_name latest checkpoint"
  upload_dir_to_s3 "$out_dir" "$checkpoint_s3_uri" "$stage_name checkpoints"
}

full_checkpoint_path() {
  local checkpoint="$1"
  if [[ "$MOE_EXPERT_PARALLEL_SIZE" == "1" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  local rank1="${checkpoint%.pt}.rank001.pt"
  if [[ ! -f "$rank1" ]]; then
    printf '%s\n' "$checkpoint"
    return 0
  fi
  local merged="${checkpoint%.pt}.full.pt"
  if [[ ! -f "$merged" || "$rank1" -nt "$merged" || "$checkpoint" -nt "$merged" ]]; then
    log "Merging expert-parallel shards for $(basename "$checkpoint")" >&2
    "$PYTHON_BIN" "$ROOT_DIR/scripts/merge_metis15_expert_parallel_checkpoint.py" \
      --checkpoint "$checkpoint" \
      --out "$merged" \
      --world-size "$MOE_EXPERT_PARALLEL_SIZE" >&2
  fi
  printf '%s\n' "$merged"
}

export_release() {
  local checkpoint="$1"
  local out_dir="$2"
  local stage_name="$3"
  local repo_id="$4"
  local release_s3_uri="${5:-}"
  log "Exporting $stage_name release"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/export_mamba_checkpoint.py" \
    --manifest "$MANIFEST" \
    --checkpoint "$checkpoint" \
    --assets-dir "$ASSETS_DIR" \
    --out-dir "$out_dir" \
    --stage-name "$stage_name" \
    --repo-id "$repo_id"
  require_release_dir "$out_dir" "$stage_name release"
  upload_dir_to_s3 "$out_dir" "$release_s3_uri" "$stage_name release"
}

if should_run_stage pretrain; then
  prepare_streaming_stage "pretrain" "$PRETRAIN_MIX" "$PRETRAIN_DATA_DIR" "$PRETRAIN_MAX_DOCS" "$PRETRAIN_TARGET_TOKENS" "$PRETRAIN_VAL_TOKENS" "$PRETRAIN_S3_URI"
  train_base_stage "base pretrain" "$PRETRAIN_DATA_DIR" "$BASE_RUN" "$PRETRAIN_BATCH" "$PRETRAIN_GRAD_ACCUM" "$PRETRAIN_STEPS" "$PRETRAIN_WARMUP" "$PRETRAIN_LR" "$PRETRAIN_WEIGHT_DECAY" "$PRETRAIN_BETA1" "$PRETRAIN_BETA2" "$PRETRAIN_EVAL_INTERVAL" "$PRETRAIN_CHECKPOINT_INTERVAL" "" "${CHECKPOINTS_S3_ROOT:+$CHECKPOINTS_S3_ROOT/base}" "pretrain" "$BASE_TRAINING_MODE" "$BASE_TE_FUSED_MLP"
elif stage_before_start pretrain; then
  require_nonempty_file "$BASE_RUN/best.pt" "base pretrain best checkpoint"
fi

if should_run_stage continued_pretrain; then
  prepare_streaming_stage "continued pretrain" "$CONTINUED_MIX" "$CONTINUED_DATA_DIR" "$CONTINUED_MAX_DOCS" "$CONTINUED_TARGET_TOKENS" "$CONTINUED_VAL_TOKENS" "$CONTINUED_S3_URI"
  train_base_stage "continued pretrain" "$CONTINUED_DATA_DIR" "$CONTINUED_RUN" "$CONTINUED_BATCH" "$CONTINUED_GRAD_ACCUM" "$CONTINUED_STEPS" "$CONTINUED_WARMUP" "$CONTINUED_LR" "$CONTINUED_WEIGHT_DECAY" "$CONTINUED_BETA1" "$CONTINUED_BETA2" "$CONTINUED_CHECKPOINT_INTERVAL" "$CONTINUED_CHECKPOINT_INTERVAL" "$BASE_RUN/best.pt" "${CHECKPOINTS_S3_ROOT:+$CHECKPOINTS_S3_ROOT/continued}" "continued_pretrain" "$CONTINUED_TRAINING_MODE" "$CONTINUED_TE_FUSED_MLP"
elif stage_before_start continued_pretrain; then
  require_nonempty_file "$CONTINUED_RUN/best.pt" "continued pretrain best checkpoint"
fi

if should_run_stage export_base; then
  CONTINUED_FULL_CHECKPOINT="$(full_checkpoint_path "$CONTINUED_RUN/best.pt")"
  export_release "$CONTINUED_FULL_CHECKPOINT" "$BASE_RELEASE" "base" "$BASE_RELEASE_REPO" "${RELEASES_S3_ROOT:+$RELEASES_S3_ROOT/base}"
elif stage_before_start export_base; then
  require_release_dir "$BASE_RELEASE" "base release"
fi

if should_run_stage chat_sft; then
  prepare_sft_stage "chat SFT" "$CHAT_MIX" "$CHAT_DATA_DIR" "$CHAT_EXAMPLES" "$CHAT_S3_URI"
  CONTINUED_FULL_CHECKPOINT="$(full_checkpoint_path "$CONTINUED_RUN/best.pt")"
  train_sft_stage "chat SFT" "$CONTINUED_FULL_CHECKPOINT" "$CHAT_DATA_DIR/train.jsonl" "$CHAT_DATA_DIR/val.jsonl" "$CHAT_RUN" "$CHAT_MAX_LENGTH" "$CHAT_BATCH" "$CHAT_GRAD_ACCUM" "$CHAT_EPOCHS" "$CHAT_LR" "$CHAT_WARMUP" "$CHAT_CHECKPOINT_INTERVAL" "${CHECKPOINTS_S3_ROOT:+$CHECKPOINTS_S3_ROOT/chat}"
elif stage_before_start chat_sft; then
  require_nonempty_file "$CHAT_RUN/best.pt" "chat SFT best checkpoint"
fi

if should_run_stage reasoning_sft; then
  prepare_sft_stage "reasoning SFT" "$REASONING_MIX" "$REASONING_DATA_DIR" "$REASONING_EXAMPLES" "$REASONING_S3_URI" --max-think-chars "$REASONING_MAX_THINK_CHARS" --max-answer-chars "$REASONING_MAX_ANSWER_CHARS" --max-code-fences 0
  train_sft_stage "reasoning SFT" "$CHAT_RUN/best.pt" "$REASONING_DATA_DIR/train.jsonl" "$REASONING_DATA_DIR/val.jsonl" "$THINK_RUN" "$REASONING_MAX_LENGTH" "$REASONING_BATCH" "$REASONING_GRAD_ACCUM" "$REASONING_EPOCHS" "$REASONING_LR" "$REASONING_WARMUP" "$REASONING_CHECKPOINT_INTERVAL" "${CHECKPOINTS_S3_ROOT:+$CHECKPOINTS_S3_ROOT/think}"
elif stage_before_start reasoning_sft; then
  require_nonempty_file "$THINK_RUN/best.pt" "reasoning SFT best checkpoint"
fi

if should_run_stage preferences; then
  prepare_preference_stage "reward preference" "$REWARD_PREF_DIR" "$REWARD_BOOTSTRAP_EXAMPLES" "$REWARD_PREF_S3_URI"
  augment_preference_stage "reward preference" "$REWARD_PREF_DIR" "$REWARD_PREF_S3_URI" "$REWARD_CHAT_NEGATIVE_PAIRS" "$REWARD_THINK_NEGATIVE_PAIRS"
  prepare_preference_stage "dpo preference" "$DPO_PREF_DIR" "$DPO_BOOTSTRAP_PAIRS" "$DPO_PREF_S3_URI"
  augment_preference_stage "dpo preference" "$DPO_PREF_DIR" "$DPO_PREF_S3_URI" "$DPO_CHAT_NEGATIVE_PAIRS" "$DPO_THINK_NEGATIVE_PAIRS"
fi

if should_run_stage reward; then
  train_reward_stage "$CHAT_RUN/best.pt" "$REWARD_RUN" "${CHECKPOINTS_S3_ROOT:+$CHECKPOINTS_S3_ROOT/reward}"
elif stage_before_start reward; then
  require_nonempty_file "$REWARD_RUN/best.pt" "reward model best checkpoint"
fi

if [[ "$RUN_THINK_DPO" == "1" ]]; then
  if should_run_stage think_dpo; then
    train_dpo_stage "think DPO" "$THINK_RUN/best.pt" "$THINK_DPO_RUN" "${CHECKPOINTS_S3_ROOT:+$CHECKPOINTS_S3_ROOT/think_dpo}"
    export_release "$THINK_DPO_RUN/best.pt" "$THINK_RELEASE" "think" "$THINK_RELEASE_REPO" "${RELEASES_S3_ROOT:+$RELEASES_S3_ROOT/think}"
  elif stage_before_start think_dpo; then
    require_release_dir "$THINK_RELEASE" "think release"
  fi
else
  if should_run_stage think_dpo; then
    export_release "$THINK_RUN/best.pt" "$THINK_RELEASE" "think" "$THINK_RELEASE_REPO" "${RELEASES_S3_ROOT:+$RELEASES_S3_ROOT/think}"
  elif stage_before_start think_dpo; then
    require_release_dir "$THINK_RELEASE" "think release"
  fi
fi

if [[ "${METIS15_RUN_EVAL:-0}" == "1" ]] && should_run_stage eval; then
  log "Running evaluation suite."
  eval_cmd=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/eval_model_suite.py" \
    --suite "$ROOT_DIR/configs/metis15_eval_prompts.json" \
    --model base="$BASE_RELEASE" \
    --model think="$THINK_RELEASE" \
    --output-path "$EVAL_REPORT"
  )
  "${eval_cmd[@]}"
  require_nonempty_file "$EVAL_REPORT" "evaluation report"
  upload_file_to_s3 "$EVAL_REPORT" "${MANIFESTS_S3_ROOT:+$MANIFESTS_S3_ROOT/eval_comparison.json}" "evaluation report"
fi

if [[ "$UPLOAD_RELEASES" == "1" && -n "${HF_TOKEN:-}" ]]; then
  log "Uploading release folders to Hugging Face."
  if [[ -n "$BASE_RELEASE_REPO" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/upload_hf_model.py" --create-repo --private --repo-id "$BASE_RELEASE_REPO" --artifact-dir "$BASE_RELEASE" --message "Upload $MODEL_DISPLAY_NAME base release"
  fi
  if [[ -n "$THINK_RELEASE_REPO" ]]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/upload_hf_model.py" --create-repo --private --repo-id "$THINK_RELEASE_REPO" --artifact-dir "$THINK_RELEASE" --message "Upload $MODEL_DISPLAY_NAME think release"
  fi
fi

log "$MODEL_DISPLAY_NAME full pipeline complete."
echo "$MODEL_DISPLAY_NAME full pipeline complete."
echo "Base release:  $BASE_RELEASE"
echo "Think release: $THINK_RELEASE"
echo "Eval report:   $EVAL_REPORT"
