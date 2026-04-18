#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

PY="./.venv/bin/python"
PIP="./.venv/bin/pip"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoints/fast/best.pt}"
CHAT_CHECKPOINT="${CHAT_CHECKPOINT:-checkpoints/chat_sft/best.pt}"
APP_PORT="${APP_PORT:-7860}"
METIS_BASE_CHECKPOINT="${METIS_BASE_CHECKPOINT:-checkpoints/metis_base/best.pt}"
METIS_THINK_CHECKPOINT="${METIS_THINK_CHECKPOINT:-checkpoints/metis_think/best.pt}"
METIS11_BASE_CHECKPOINT="${METIS11_BASE_CHECKPOINT:-checkpoints/metis11_base/best.pt}"
METIS11_CHAT_CHECKPOINT="${METIS11_CHAT_CHECKPOINT:-checkpoints/metis11_chat/best.pt}"
METIS11_THINK_CHECKPOINT="${METIS11_THINK_CHECKPOINT:-checkpoints/metis11_think/best.pt}"
METIS100_BASE_CHECKPOINT="${METIS100_BASE_CHECKPOINT:-checkpoints/metis100_base/best.pt}"
METIS11_PRETRAIN_MIX="${METIS11_PRETRAIN_MIX:-configs/metis11_pretrain_mix.json}"
METIS11_TOKENIZER_DIR="${METIS11_TOKENIZER_DIR:-artifacts/metis11_tokenizer}"
METIS11_TOKENIZER_PATH="${METIS11_TOKENIZER_PATH:-$METIS11_TOKENIZER_DIR/tokenizer.json}"
METIS11_DATA_DIR="${METIS11_DATA_DIR:-data/metis11_base}"
METIS11_CHAT_DATA_DIR="${METIS11_CHAT_DATA_DIR:-data/metis11_chat_sft}"
METIS11_REASONING_DATA_DIR="${METIS11_REASONING_DATA_DIR:-data/metis11_reasoning}"
METIS11_TOKENIZER_SAMPLES="${METIS11_TOKENIZER_SAMPLES:-120000}"
METIS11_DATA_DOCS="${METIS11_DATA_DOCS:-80000}"
METIS11_CHAT_EXAMPLES="${METIS11_CHAT_EXAMPLES:-60000}"
METIS11_REASONING_EXAMPLES="${METIS11_REASONING_EXAMPLES:-18000}"
METIS11_BASE_STEPS="${METIS11_BASE_STEPS:-5000}"
METIS11_H100_DTYPE="${METIS11_H100_DTYPE:-bf16}"
METIS11_H100_COMPILE_MODE="${METIS11_H100_COMPILE_MODE:-default}"
VENV_FLAGS="${VENV_FLAGS:-}"

usage() {
  cat <<'EOF'
Usage: ./run.sh <command>

Commands:
  setup       install/update dependencies and editable package
  tokenizer   train the BPE tokenizer
  prepare     tokenize TinyStories into train.bin / val.bin
  train       train the base 10M model
  generate    sample from the base model
  chat-data   prepare chat fine-tuning tensors
  sft         fine-tune the base model on chat examples
  chat        sample from the chat fine-tuned model
  chat-fast   run UltraChat prep + SFT from checkpoints/fast/best.pt
  app         launch the local browser chat app
  metis-tokenizer train a tokenizer on FineWeb-Edu
  metis-data   stream FineWeb-Edu into binary token files
  metis-base   train the new Metis 1.0 base model
  metis-think-data prepare short-form reasoning SFT data
  metis-think  fine-tune Metis into a reasoning model
  metis-full   run the full Metis base + reasoning pipeline
  metis11-tokenizer train the Metis 1.1 tokenizer on the mixed small-model corpus
  metis11-data  write the mixed Metis 1.1 pretraining binary data
  metis11-base train a ~55M Metis 1.1 base model
  metis11-chat-data prepare SmolTalk-based conversational SFT data
  metis11-chat conversationally fine-tune Metis 1.1
  metis11-think-data prepare compact OpenThoughts reasoning SFT data
  metis11-think train the Metis 1.1 thinking checkpoint
  metis11-full run the full Metis 1.1 base + chat + thinking pipeline
  metis11-h100-full run the optimized H100 Metis 1.1 path
  metis11-h100-pod run the optimized GPU path and mirror logs into pod logs
  metis100-base train a ~104M experimental Metis base model
  metis100-full run tokenizer + data + 100M base training
  base        run setup -> tokenizer -> prepare -> train
  full-chat   run the full base + chat fine-tuning pipeline
  fast        quick learning run, good first pass
  standard    balanced default run
  overnight   longer run for better quality
EOF
}

cmd="${1:-help}"

case "$cmd" in
  setup)
    if [[ ! -x "$PY" ]]; then
      python3 -m venv ${VENV_FLAGS:+$VENV_FLAGS } .venv
    fi
    "$PIP" install -U pip
    "$PIP" install -r requirements.txt
    "$PIP" install -e .
    ;;
  tokenizer)
    "$PY" scripts/train_tokenizer.py --vocab-size 8192 --max-samples 200000
    ;;
  prepare)
    "$PY" scripts/prepare_data.py
    ;;
  train)
    "$PY" scripts/train.py --max-steps 3000 --batch-size 8 --grad-accum-steps 4
    ;;
  generate)
    "$PY" scripts/generate.py --checkpoint "$BASE_CHECKPOINT" --prompt "Hello, how are you?" --max-new-tokens 120
    ;;
  chat-data)
    "$PY" scripts/prepare_chat_data.py --hf-dataset HuggingFaceH4/ultrachat_200k --hf-train-split train_sft --hf-val-split test_sft --hf-train-limit 20000 --hf-val-limit 2000 --output-dir data/chat_sft
    ;;
  sft)
    "$PY" scripts/train_sft.py --base-checkpoint "$BASE_CHECKPOINT" --train-data data/chat_sft/train.pt --val-data data/chat_sft/val.pt --out-dir checkpoints/chat_sft
    ;;
  chat)
    "$PY" scripts/generate.py --checkpoint "$CHAT_CHECKPOINT" --prompt "What is a tokenizer?" --max-new-tokens 80
    ;;
  chat-fast)
    "$0" setup
    "$0" chat-data
    "$PY" scripts/train_sft.py --base-checkpoint checkpoints/fast/best.pt --train-data data/chat_sft/train.pt --val-data data/chat_sft/val.pt --out-dir checkpoints/chat_fast
    ;;
  app)
    "$PY" scripts/chat_app.py --port "$APP_PORT"
    ;;
  metis-tokenizer)
    "$0" setup
    "$PY" scripts/train_tokenizer.py --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --split train --streaming --max-samples 200000 --vocab-size 8192 --output-dir artifacts/metis_tokenizer
    ;;
  metis-data)
    "$PY" scripts/prepare_streaming_data.py --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --split train --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --output-dir data/metis_base --max-docs 60000 --val-ratio 0.02
    ;;
  metis-base)
    "$PY" scripts/train.py --resume --data-dir data/metis_base --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --out-dir checkpoints/metis_base --max-steps 3000 --batch-size 8 --grad-accum-steps 4 --block-size 384 --n-layer 10 --n-head 8 --n-embd 256 --eval-interval 250
    ;;
  metis-think-data)
    "$PY" scripts/prepare_reasoning_sft.py --dataset-name open-thoughts/OpenThoughts-114k --split train --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --output-dir data/metis_reasoning --max-examples 12000 --val-ratio 0.05 --max-length 384 --max-thought-chars 700 --max-solution-chars 300
    ;;
  metis-think)
    "$PY" scripts/train_sft.py --resume --base-checkpoint "$METIS_BASE_CHECKPOINT" --train-data data/metis_reasoning/train.pt --val-data data/metis_reasoning/val.pt --out-dir checkpoints/metis_think --epochs 3 --lr 5e-5
    ;;
  metis-full)
    "$0" setup
    "$0" metis-tokenizer
    "$0" metis-data
    "$0" metis-base
    "$0" metis-think-data
    "$0" metis-think
    ;;
  metis11-tokenizer)
    "$0" setup
    "$PY" scripts/train_tokenizer.py --mixture-config "$METIS11_PRETRAIN_MIX" --max-samples "$METIS11_TOKENIZER_SAMPLES" --vocab-size 8192 --output-dir "$METIS11_TOKENIZER_DIR"
    ;;
  metis11-data)
    "$PY" scripts/prepare_streaming_data.py --mixture-config "$METIS11_PRETRAIN_MIX" --tokenizer-path "$METIS11_TOKENIZER_PATH" --output-dir "$METIS11_DATA_DIR" --max-docs "$METIS11_DATA_DOCS" --val-ratio 0.02
    ;;
  metis11-base)
    "$PY" scripts/train.py --resume --data-dir "$METIS11_DATA_DIR" --tokenizer-path "$METIS11_TOKENIZER_PATH" --out-dir checkpoints/metis11_base --max-steps "$METIS11_BASE_STEPS" --batch-size 4 --grad-accum-steps 8 --block-size 512 --n-layer 16 --n-head 8 --n-embd 512 --lr 2.5e-4 --eval-interval 500
    ;;
  metis11-chat-data)
    "$PY" scripts/prepare_chat_data.py --tokenizer-path "$METIS11_TOKENIZER_PATH" --hf-dataset HuggingFaceTB/smol-smoltalk --hf-split train --hf-limit "$METIS11_CHAT_EXAMPLES" --hf-streaming --val-ratio 0.05 --output-dir "$METIS11_CHAT_DATA_DIR" --max-length 384
    ;;
  metis11-chat)
    "$PY" scripts/train_sft.py --resume --base-checkpoint "$METIS11_BASE_CHECKPOINT" --train-data "$METIS11_CHAT_DATA_DIR/train.pt" --val-data "$METIS11_CHAT_DATA_DIR/val.pt" --out-dir checkpoints/metis11_chat --epochs 2 --lr 8e-5 --batch-size 6
    ;;
  metis11-think-data)
    "$PY" scripts/prepare_reasoning_sft.py --dataset-name open-thoughts/OpenThoughts-114k --split train --tokenizer-path "$METIS11_TOKENIZER_PATH" --output-dir "$METIS11_REASONING_DATA_DIR" --max-examples "$METIS11_REASONING_EXAMPLES" --val-ratio 0.05 --max-length 512 --max-user-chars 550 --max-thought-chars 700 --max-solution-chars 260 --assistant-format think_tags --system-prompt "You are Metis, a compact reasoning assistant. Use a short <think> block when it helps, then answer clearly."
    ;;
  metis11-think)
    "$PY" scripts/train_sft.py --resume --base-checkpoint "$METIS11_CHAT_CHECKPOINT" --train-data "$METIS11_REASONING_DATA_DIR/train.pt" --val-data "$METIS11_REASONING_DATA_DIR/val.pt" --out-dir checkpoints/metis11_think --epochs 2 --lr 4e-5 --batch-size 4
    ;;
  metis11-full)
    "$0" metis11-tokenizer
    "$0" metis11-data
    "$0" metis11-base
    "$0" metis11-chat-data
    "$0" metis11-chat
    "$0" metis11-think-data
    "$0" metis11-think
    ;;
  metis11-h100-base)
    "$PY" scripts/train.py --resume --dtype "$METIS11_H100_DTYPE" --compile --compile-mode "$METIS11_H100_COMPILE_MODE" --fused-adamw --tf32 --matmul-precision high --data-dir "$METIS11_DATA_DIR" --tokenizer-path "$METIS11_TOKENIZER_PATH" --out-dir checkpoints/metis11_base --max-steps "$METIS11_BASE_STEPS" --batch-size 16 --grad-accum-steps 2 --block-size 512 --n-layer 16 --n-head 8 --n-embd 512 --lr 2.5e-4 --eval-interval 500
    ;;
  metis11-h100-chat)
    "$PY" scripts/train_sft.py --resume --dtype "$METIS11_H100_DTYPE" --compile --compile-mode "$METIS11_H100_COMPILE_MODE" --fused-adamw --tf32 --matmul-precision high --pad-to-block-size --base-checkpoint "$METIS11_BASE_CHECKPOINT" --train-data "$METIS11_CHAT_DATA_DIR/train.pt" --val-data "$METIS11_CHAT_DATA_DIR/val.pt" --out-dir checkpoints/metis11_chat --epochs 2 --lr 8e-5 --batch-size 24
    ;;
  metis11-h100-think)
    "$PY" scripts/train_sft.py --resume --dtype "$METIS11_H100_DTYPE" --compile --compile-mode "$METIS11_H100_COMPILE_MODE" --fused-adamw --tf32 --matmul-precision high --pad-to-block-size --base-checkpoint "$METIS11_CHAT_CHECKPOINT" --train-data "$METIS11_REASONING_DATA_DIR/train.pt" --val-data "$METIS11_REASONING_DATA_DIR/val.pt" --out-dir checkpoints/metis11_think --epochs 2 --lr 4e-5 --batch-size 16
    ;;
  metis11-h100-full)
    "$0" metis11-tokenizer
    "$0" metis11-data
    "$0" metis11-h100-base
    "$0" metis11-chat-data
    "$0" metis11-h100-chat
    "$0" metis11-think-data
    "$0" metis11-h100-think
    ;;
  metis11-h100-pod)
    ./scripts/pod_launch.sh "$0" metis11-h100-full
    ;;
  metis100-base)
    "$PY" scripts/train.py --data-dir data/metis_base --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --out-dir checkpoints/metis100_base --max-steps 3000 --batch-size 2 --grad-accum-steps 8 --block-size 512 --n-layer 20 --n-head 8 --n-embd 640 --lr 2e-4 --eval-interval 250
    ;;
  metis100-full)
    "$0" setup
    "$0" metis-tokenizer
    "$0" metis-data
    "$0" metis100-base
    ;;
  base)
    "$0" setup
    "$0" tokenizer
    "$0" prepare
    "$0" train
    ;;
  full-chat)
    "$0" setup
    "$0" tokenizer
    "$0" prepare
    "$0" train
    "$0" chat-data
    "$PY" scripts/train_sft.py --base-checkpoint checkpoints/default/best.pt --train-data data/chat_sft/train.pt --val-data data/chat_sft/val.pt --out-dir checkpoints/chat_sft
    ;;
  fast)
    "$0" setup
    "$PY" scripts/train_tokenizer.py --vocab-size 8192 --max-samples 100000
    "$PY" scripts/prepare_data.py --train-limit 500000 --val-limit 10000
    "$PY" scripts/train.py --max-steps 1000 --batch-size 8 --grad-accum-steps 4 --block-size 128 --eval-interval 250 --out-dir checkpoints/fast
    ;;
  standard)
    "$0" setup
    "$PY" scripts/train_tokenizer.py --vocab-size 8192 --max-samples 200000
    "$PY" scripts/prepare_data.py --train-limit 1000000 --val-limit 20000
    "$PY" scripts/train.py --max-steps 3000 --batch-size 8 --grad-accum-steps 4 --block-size 256 --eval-interval 250 --out-dir checkpoints/standard
    ;;
  overnight)
    "$0" setup
    "$PY" scripts/train_tokenizer.py --vocab-size 8192 --max-samples 300000
    "$PY" scripts/prepare_data.py --train-limit 1500000 --val-limit 25000
    "$PY" scripts/train.py --max-steps 8000 --batch-size 8 --grad-accum-steps 4 --block-size 256 --eval-interval 500 --out-dir checkpoints/overnight
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
