SHELL := /bin/bash
VENV_DIR ?= .venv
PYTHON ?= python3
VENV_FLAGS ?=
PY := ./$(VENV_DIR)/bin/python
PIP := ./$(VENV_DIR)/bin/pip
ENV_FILE := .env
BASE_CHECKPOINT ?= checkpoints/fast/best.pt
CHAT_CHECKPOINT ?= checkpoints/chat_sft/best.pt
APP_PORT ?= 7860
METIS_BASE_CHECKPOINT ?= checkpoints/metis_base/best.pt
METIS_THINK_CHECKPOINT ?= checkpoints/metis_think/best.pt
METIS11_BASE_CHECKPOINT ?= checkpoints/metis11_base/best.pt
METIS11_CHAT_CHECKPOINT ?= checkpoints/metis11_chat/best.pt
METIS11_THINK_CHECKPOINT ?= checkpoints/metis11_think/best.pt
METIS100_BASE_CHECKPOINT ?= checkpoints/metis100_base/best.pt
HF_NAMESPACE ?= Lernex
METIS11_PRETRAIN_MIX ?= configs/metis11_pretrain_mix.json
METIS11_TOKENIZER_DIR ?= artifacts/metis11_tokenizer
METIS11_TOKENIZER_PATH ?= $(METIS11_TOKENIZER_DIR)/tokenizer.json
METIS11_DATA_DIR ?= data/metis11_base
METIS11_CHAT_DATA_DIR ?= data/metis11_chat_sft
METIS11_REASONING_DATA_DIR ?= data/metis11_reasoning
METIS11_TOKENIZER_SAMPLES ?= 120000
METIS11_DATA_DOCS ?= 80000
METIS11_CHAT_EXAMPLES ?= 60000
METIS11_REASONING_EXAMPLES ?= 18000
METIS11_BASE_STEPS ?= 5000
METIS11_H100_DTYPE ?= bf16
METIS11_H100_COMPILE_MODE ?= default
METIS12_SHARED_ROOT ?= $(PWD)/.runpod/metis12
METIS13_SHARED_ROOT ?= $(PWD)/.runpod/metis13
METIS15_MANIFEST ?= configs/metis15_manifest.json
METIS15_DATA_DIR ?= data/metis15_base
METIS15_CONTINUED_DATA_DIR ?= data/metis15_continued_pretrain
METIS15_BASE_OUT ?= checkpoints/metis15_base
METIS15_CONTINUED_OUT ?= checkpoints/metis15_continued_pretrain
METIS15_S3_ROOT ?= s3://lernex-metis-artifacts-151025633969-us-east-1/metis15
METIS15_SMOKE_RECIPES ?= fp8,fp8_block,nvfp4,bf16

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export HF_TOKEN
endif

.PHONY: help setup tokenizer prepare train generate chat-data sft chat chat-fast app base full-chat fast standard overnight metis-tokenizer metis-data metis-base metis-think-data metis-think metis-full metis11-tokenizer metis11-data metis11-base metis11-chat-data metis11-chat metis11-think-data metis11-think metis11-full metis11-h100-base metis11-h100-chat metis11-h100-think metis11-h100-full metis11-h100-pod metis12-cpu-prep metis12-gpu-full metis13-cpu-memory-prep metis13-cpu-compute-prep metis13-cpu-prep metis13-gpu-full metis15-validate-data-plan metis15-tokenizer-assets-aws metis15-cpu-prep-aws metis15-blackwell-smoke metis15-training-contracts metis15-rtx-benchmark-matrix metis15-megatron-profile metis15-a100-pretrain metis15-a100-continued-pretrain metis15-rtx-pretrain metis15-rtx-fp8-pretrain metis15-rtx-continued-pretrain metis15-neuron-pretrain metis15-full metis15-p5-pretrain metis100-base metis100-full hf-upload-metis11-base hf-upload-metis11-chat hf-upload-metis11-think clean-bench

help:
	@echo "Targets:"
	@echo "  make setup       - install/update dependencies and editable package"
	@echo "  make tokenizer   - train the BPE tokenizer"
	@echo "  make prepare     - tokenize the legacy TinyStories starter corpus into train.bin / val.bin"
	@echo "  make train       - train the base 10M model"
	@echo "  make generate    - sample from the base model"
	@echo "  make chat-data   - prepare chat fine-tuning tensors"
	@echo "  make sft         - fine-tune the base model on chat examples"
	@echo "  make chat        - sample from the chat fine-tuned model"
	@echo "  make chat-fast   - run UltraChat prep + SFT from checkpoints/fast/best.pt"
	@echo "  make app         - launch the local browser chat app"
	@echo "  make metis-tokenizer - train a tokenizer on FineWeb-Edu"
	@echo "  make metis-data      - stream FineWeb-Edu into binary token files"
	@echo "  make metis-base      - train the new Metis 1.0 base model"
	@echo "  make metis-think-data - prepare short-form reasoning SFT data"
	@echo "  make metis-think     - fine-tune Metis into a reasoning model"
	@echo "  make metis-full      - run the full Metis base + reasoning pipeline"
	@echo "  make metis11-tokenizer - train the Metis 1.1 tokenizer on the mixed small-model corpus"
	@echo "  make metis11-data      - write the mixed Metis 1.1 pretraining binary data"
	@echo "  make metis11-base      - train the ~55M Metis 1.1 base model"
	@echo "  make metis11-chat-data - prepare SmolTalk-based conversational SFT data"
	@echo "  make metis11-chat      - conversationally fine-tune Metis 1.1"
	@echo "  make metis11-think-data - prepare compact OpenThoughts reasoning SFT data"
	@echo "  make metis11-think     - train the Metis 1.1 thinking checkpoint"
	@echo "  make metis11-full      - run the full Metis 1.1 base + chat + thinking pipeline"
	@echo "  make metis11-h100-full - run the optimized H100 Metis 1.1 path"
	@echo "  make hf-upload-metis11-base  - upload the Metis 1.1 base checkpoint to Hugging Face"
	@echo "  make hf-upload-metis11-chat  - upload the Metis 1.1 chat checkpoint to Hugging Face"
	@echo "  make hf-upload-metis11-think - upload the Metis 1.1 think checkpoint to Hugging Face"
	@echo "  make metis12-cpu-prep    - run the shared-volume CPU preparation flow for Metis-1.2"
	@echo "  make metis12-gpu-full    - run the TorchTitan FP8 GPU flow for Metis-1.2"
	@echo "  make metis13-cpu-memory-prep  - run the RAM-heavy tokenizer/assets pod for Metis-1.3"
	@echo "  make metis13-cpu-compute-prep - run the compute-heavy data-build pod for Metis-1.3"
	@echo "  make metis13-cpu-prep         - run both Metis-1.3 CPU halves sequentially on one pod"
	@echo "  make metis13-gpu-full    - run the Mamba2-hybrid BF16 GPU flow for Metis-1.3"
	@echo "  make metis15-validate-data-plan - validate Metis-1.5 bucket totals, caps, and release targets"
	@echo "  make metis15-tokenizer-assets-aws - run Metis-1.5 CPU prep through the 32k tokenizer upload"
	@echo "  make metis15-cpu-prep-aws - run the Metis-1.5 CPU prep flow with its sparse-MoE manifest"
	@echo "  make metis15-blackwell-smoke - legacy exact-shape FP8/NVFP4 kernel smoke on RTX PRO 6000"
	@echo "  make metis15-rtx-benchmark-matrix - run the RTX PRO 6000 optimization benchmark matrix"
	@echo "  make metis15-megatron-profile - print the native-vs-Megatron MoE optimization profile"
	@echo "  make metis15-a100-pretrain - launch the Metis-1.5 8xA100 BF16 expert-parallel baseline"
	@echo "  make metis15-neuron-pretrain - launch Metis-1.5 on AWS Trainium/Neuron static expert parallelism"
	@echo "  make metis15-rtx-pretrain - legacy alias for the current Metis-1.5 BF16 pretrain launcher"
	@echo "  make metis15-rtx-fp8-pretrain - launch Metis-1.5 with expert-only H100 FP8"
	@echo "  make metis15-rtx-continued-pretrain - continue Metis-1.5 pretraining from the base checkpoint in BF16"
	@echo "  make metis15-full      - run the full Metis-1.5 base -> think pipeline through DPO and release export"
	@echo "  make metis100-base   - train a ~104M experimental Metis base model"
	@echo "  make metis100-full   - run tokenizer + data + 100M base training"
	@echo "  make base        - run setup -> tokenizer -> prepare -> train"
	@echo "  make full-chat   - run the full base + chat fine-tuning pipeline"
	@echo "  make fast        - quick learning run, good first pass"
	@echo "  make standard    - balanced default run"
	@echo "  make overnight   - longer run for better quality"

setup:
	test -x $(PY) || $(PYTHON) -m venv $(VENV_FLAGS) $(VENV_DIR)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

tokenizer:
	$(PY) scripts/train_tokenizer.py --vocab-size 8192 --max-samples 200000

prepare:
	$(PY) scripts/prepare_data.py

train:
	$(PY) scripts/train.py --max-steps 3000 --batch-size 8 --grad-accum-steps 4

generate:
	$(PY) scripts/generate.py --checkpoint $(BASE_CHECKPOINT) --prompt "Hello, how are you?" --max-new-tokens 120

chat-data:
	$(PY) scripts/prepare_chat_data.py --hf-dataset HuggingFaceH4/ultrachat_200k --hf-train-split train_sft --hf-val-split test_sft --hf-train-limit 20000 --hf-val-limit 2000 --output-dir data/chat_sft

sft:
	$(PY) scripts/train_sft.py --base-checkpoint $(BASE_CHECKPOINT) --train-data data/chat_sft/train.pt --val-data data/chat_sft/val.pt --out-dir checkpoints/chat_sft

chat:
	$(PY) scripts/generate.py --checkpoint $(CHAT_CHECKPOINT) --prompt "What is a tokenizer?" --max-new-tokens 80

chat-fast: setup chat-data
	$(PY) scripts/train_sft.py --base-checkpoint checkpoints/fast/best.pt --train-data data/chat_sft/train.pt --val-data data/chat_sft/val.pt --out-dir checkpoints/chat_fast

app:
	$(PY) scripts/chat_app.py --port $(APP_PORT)

metis-tokenizer: setup
	$(PY) scripts/train_tokenizer.py --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --split train --streaming --max-samples 200000 --vocab-size 8192 --output-dir artifacts/metis_tokenizer

metis-data:
	$(PY) scripts/prepare_streaming_data.py --dataset-name HuggingFaceFW/fineweb-edu --dataset-config sample-10BT --split train --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --output-dir data/metis_base --max-docs 60000 --val-ratio 0.02

metis-base:
	$(PY) scripts/train.py --resume --data-dir data/metis_base --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --out-dir checkpoints/metis_base --max-steps 3000 --batch-size 8 --grad-accum-steps 4 --block-size 384 --n-layer 10 --n-head 8 --n-embd 256 --eval-interval 250

metis-think-data:
	$(PY) scripts/prepare_reasoning_sft.py --dataset-name open-thoughts/OpenThoughts-114k --split train --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --output-dir data/metis_reasoning --max-examples 12000 --val-ratio 0.05 --max-length 384 --max-thought-chars 700 --max-solution-chars 300

metis-think:
	$(PY) scripts/train_sft.py --resume --base-checkpoint $(METIS_BASE_CHECKPOINT) --train-data data/metis_reasoning/train.pt --val-data data/metis_reasoning/val.pt --out-dir checkpoints/metis_think --epochs 3 --lr 5e-5

metis-full: setup metis-tokenizer metis-data metis-base metis-think-data metis-think

metis11-tokenizer: setup
	$(PY) scripts/train_tokenizer.py --mixture-config $(METIS11_PRETRAIN_MIX) --max-samples $(METIS11_TOKENIZER_SAMPLES) --vocab-size 8192 --output-dir $(METIS11_TOKENIZER_DIR)

metis11-data:
	$(PY) scripts/prepare_streaming_data.py --mixture-config $(METIS11_PRETRAIN_MIX) --tokenizer-path $(METIS11_TOKENIZER_PATH) --output-dir $(METIS11_DATA_DIR) --max-docs $(METIS11_DATA_DOCS) --val-ratio 0.02

metis11-base:
	$(PY) scripts/train.py --resume --data-dir $(METIS11_DATA_DIR) --tokenizer-path $(METIS11_TOKENIZER_PATH) --out-dir checkpoints/metis11_base --max-steps $(METIS11_BASE_STEPS) --batch-size 4 --grad-accum-steps 8 --block-size 512 --n-layer 16 --n-head 8 --n-embd 512 --lr 2.5e-4 --eval-interval 500

metis11-chat-data:
	$(PY) scripts/prepare_chat_data.py --tokenizer-path $(METIS11_TOKENIZER_PATH) --hf-dataset HuggingFaceTB/smol-smoltalk --hf-split train --hf-limit $(METIS11_CHAT_EXAMPLES) --hf-streaming --val-ratio 0.05 --output-dir $(METIS11_CHAT_DATA_DIR) --max-length 384

metis11-chat:
	$(PY) scripts/train_sft.py --resume --base-checkpoint $(METIS11_BASE_CHECKPOINT) --train-data $(METIS11_CHAT_DATA_DIR)/train.pt --val-data $(METIS11_CHAT_DATA_DIR)/val.pt --out-dir checkpoints/metis11_chat --epochs 2 --lr 8e-5 --batch-size 6

metis11-think-data:
	$(PY) scripts/prepare_reasoning_sft.py --dataset-name open-thoughts/OpenThoughts-114k --split train --tokenizer-path $(METIS11_TOKENIZER_PATH) --output-dir $(METIS11_REASONING_DATA_DIR) --max-examples $(METIS11_REASONING_EXAMPLES) --val-ratio 0.05 --max-length 512 --max-user-chars 550 --max-thought-chars 700 --max-solution-chars 260 --assistant-format think_tags --system-prompt "You are Metis, a compact reasoning assistant. Use a short <think> block when it helps, then answer clearly."

metis11-think:
	$(PY) scripts/train_sft.py --resume --base-checkpoint $(METIS11_CHAT_CHECKPOINT) --train-data $(METIS11_REASONING_DATA_DIR)/train.pt --val-data $(METIS11_REASONING_DATA_DIR)/val.pt --out-dir checkpoints/metis11_think --epochs 2 --lr 4e-5 --batch-size 4

metis11-full:
	$(MAKE) metis11-tokenizer
	$(MAKE) metis11-data
	$(MAKE) metis11-base
	$(MAKE) metis11-chat-data
	$(MAKE) metis11-chat
	$(MAKE) metis11-think-data
	$(MAKE) metis11-think

metis11-h100-base:
	$(PY) scripts/train.py --resume --dtype $(METIS11_H100_DTYPE) --compile --compile-mode $(METIS11_H100_COMPILE_MODE) --fused-adamw --tf32 --matmul-precision high --data-dir $(METIS11_DATA_DIR) --tokenizer-path $(METIS11_TOKENIZER_PATH) --out-dir checkpoints/metis11_base --max-steps $(METIS11_BASE_STEPS) --batch-size 16 --grad-accum-steps 2 --block-size 512 --n-layer 16 --n-head 8 --n-embd 512 --lr 2.5e-4 --eval-interval 500

metis11-h100-chat:
	$(PY) scripts/train_sft.py --resume --dtype $(METIS11_H100_DTYPE) --compile --compile-mode $(METIS11_H100_COMPILE_MODE) --fused-adamw --tf32 --matmul-precision high --pad-to-block-size --base-checkpoint $(METIS11_BASE_CHECKPOINT) --train-data $(METIS11_CHAT_DATA_DIR)/train.pt --val-data $(METIS11_CHAT_DATA_DIR)/val.pt --out-dir checkpoints/metis11_chat --epochs 2 --lr 8e-5 --batch-size 24

metis11-h100-think:
	$(PY) scripts/train_sft.py --resume --dtype $(METIS11_H100_DTYPE) --compile --compile-mode $(METIS11_H100_COMPILE_MODE) --fused-adamw --tf32 --matmul-precision high --pad-to-block-size --base-checkpoint $(METIS11_CHAT_CHECKPOINT) --train-data $(METIS11_REASONING_DATA_DIR)/train.pt --val-data $(METIS11_REASONING_DATA_DIR)/val.pt --out-dir checkpoints/metis11_think --epochs 2 --lr 4e-5 --batch-size 16

metis11-h100-full:
	$(MAKE) metis11-tokenizer
	$(MAKE) metis11-data
	$(MAKE) metis11-h100-base
	$(MAKE) metis11-chat-data
	$(MAKE) metis11-h100-chat
	$(MAKE) metis11-think-data
	$(MAKE) metis11-h100-think

metis11-h100-pod:
	./scripts/pod_launch.sh $(MAKE) metis11-h100-full

metis12-cpu-prep:
	METIS12_SHARED_ROOT="$(METIS12_SHARED_ROOT)" ./scripts/runpod_metis12_cpu.sh

metis12-gpu-full:
	METIS12_SHARED_ROOT="$(METIS12_SHARED_ROOT)" ./scripts/runpod_metis12_gpu.sh

metis13-cpu-memory-prep:
	METIS13_SHARED_ROOT="$(METIS13_SHARED_ROOT)" METIS13_CPU_ROLE="memory" ./scripts/runpod_metis13_cpu.sh

metis13-cpu-compute-prep:
	METIS13_SHARED_ROOT="$(METIS13_SHARED_ROOT)" METIS13_CPU_ROLE="compute" ./scripts/runpod_metis13_cpu.sh

metis13-cpu-prep:
	METIS13_SHARED_ROOT="$(METIS13_SHARED_ROOT)" METIS13_CPU_ROLE="full" ./scripts/runpod_metis13_cpu.sh

metis13-gpu-full:
	METIS13_SHARED_ROOT="$(METIS13_SHARED_ROOT)" ./scripts/runpod_metis13_gpu.sh

metis15-validate-data-plan:
	$(PYTHON) scripts/validate_metis15_data_plan.py --manifest $(METIS15_MANIFEST)

metis15-tokenizer-assets-aws:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_PREP_MODE="aws" METIS15_STOP_AFTER_STAGE="tokenizer_assets" ./scripts/metis15_cpu_prep.sh

metis15-cpu-prep-aws:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_PREP_MODE="aws" ./scripts/metis15_cpu_prep.sh

metis15-blackwell-smoke:
	$(PYTHON) scripts/smoke_metis15_blackwell_kernels.py --recipes "$(METIS15_SMOKE_RECIPES)" --nvfp4-disable-rht --nvfp4-disable-2d-quantization --nvfp4-disable-stochastic-rounding

metis15-training-contracts:
	$(PYTHON) scripts/smoke_metis15_training_contracts.py

metis15-rtx-benchmark-matrix:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_DATA_DIR="$(METIS15_DATA_DIR)" ./scripts/metis15_rtx_benchmark_matrix.sh

metis15-megatron-profile:
	$(PY) scripts/metis15_megatron_super_profile.py --manifest "$(METIS15_MANIFEST)" --check-imports

metis15-a100-pretrain:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_DATA_DIR="$(METIS15_DATA_DIR)" METIS15_OUT_DIR="$(METIS15_BASE_OUT)" METIS15_FP8=0 METIS15_NVFP4=0 METIS15_FP8_EXPERT_PRECISION=bf16 METIS15_MOE_DISPATCH_MODE=bucketed ./scripts/metis15_pretrain.sh

metis15-neuron-pretrain:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_DATA_DIR="$(METIS15_DATA_DIR)" METIS15_OUT_DIR="$(METIS15_BASE_OUT)_neuron" ./scripts/metis15_neuron_pretrain.sh

metis15-rtx-pretrain: metis15-a100-pretrain

metis15-rtx-fp8-pretrain:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_DATA_DIR="$(METIS15_DATA_DIR)" METIS15_OUT_DIR="$(METIS15_BASE_OUT)" METIS15_FP8=1 METIS15_NVFP4=0 METIS15_FP8_EXPERT_PRECISION=fp8 METIS15_MOE_DISPATCH_MODE=bucketed ./scripts/metis15_pretrain.sh

metis15-a100-continued-pretrain:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_TRAIN_STAGE="continued_pretrain" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_S3_PRETRAIN_URI="$(METIS15_S3_ROOT)/pretrain-shards/continued" METIS15_S3_CHECKPOINTS_URI="$(METIS15_S3_ROOT)/checkpoints/continued" METIS15_DATA_DIR="$(METIS15_CONTINUED_DATA_DIR)" METIS15_OUT_DIR="$(METIS15_CONTINUED_OUT)" METIS15_INIT_CHECKPOINT="$(METIS15_BASE_OUT)/best.pt" METIS15_FP8=0 METIS15_NVFP4=0 METIS15_FP8_EXPERT_PRECISION=bf16 METIS15_MOE_DISPATCH_MODE=bucketed ./scripts/metis15_pretrain.sh

metis15-rtx-continued-pretrain: metis15-a100-continued-pretrain

metis15-full:
	METIS15_MANIFEST="$(METIS15_MANIFEST)" METIS15_S3_ROOT="$(METIS15_S3_ROOT)" METIS15_DATA_DIR="$(METIS15_DATA_DIR)" METIS15_BASE_OUT="$(METIS15_BASE_OUT)" METIS15_FP8=0 METIS15_NVFP4=0 METIS15_FP8_EXPERT_PRECISION=bf16 METIS15_MOE_DISPATCH_MODE=bucketed ./scripts/metis15_full.sh

metis15-p5-pretrain: metis15-rtx-pretrain

metis100-base:
	$(PY) scripts/train.py --data-dir data/metis_base --tokenizer-path artifacts/metis_tokenizer/tokenizer.json --out-dir checkpoints/metis100_base --max-steps 3000 --batch-size 2 --grad-accum-steps 8 --block-size 512 --n-layer 20 --n-head 8 --n-embd 640 --lr 2e-4 --eval-interval 250

hf-upload-metis11-base:
	$(PY) scripts/upload_hf_model.py --create-repo --private --repo-id $(HF_NAMESPACE)/Metis-1.1-base --checkpoint checkpoints/metis11_base/best.pt --tokenizer-path $(METIS11_TOKENIZER_PATH) --message "Upload Metis 1.1 base checkpoint"

hf-upload-metis11-chat:
	$(PY) scripts/upload_hf_model.py --create-repo --private --repo-id $(HF_NAMESPACE)/Metis-1.1-chat --checkpoint checkpoints/metis11_chat/best.pt --tokenizer-path $(METIS11_TOKENIZER_PATH) --message "Upload Metis 1.1 chat checkpoint"

hf-upload-metis11-think:
	$(PY) scripts/upload_hf_model.py --create-repo --private --repo-id $(HF_NAMESPACE)/Metis-1.1-think --checkpoint checkpoints/metis11_think/best.pt --tokenizer-path $(METIS11_TOKENIZER_PATH) --message "Upload Metis 1.1 think checkpoint"

metis100-full: setup metis-tokenizer metis-data metis100-base

base: setup tokenizer prepare train

full-chat: setup tokenizer prepare train chat-data
	$(PY) scripts/train_sft.py --base-checkpoint checkpoints/default/best.pt --train-data data/chat_sft/train.pt --val-data data/chat_sft/val.pt --out-dir checkpoints/chat_sft

fast: setup
	$(PY) scripts/train_tokenizer.py --vocab-size 8192 --max-samples 100000
	$(PY) scripts/prepare_data.py --train-limit 500000 --val-limit 10000
	$(PY) scripts/train.py --max-steps 1000 --batch-size 8 --grad-accum-steps 4 --block-size 128 --eval-interval 250 --out-dir checkpoints/fast

standard: setup
	$(PY) scripts/train_tokenizer.py --vocab-size 8192 --max-samples 200000
	$(PY) scripts/prepare_data.py --train-limit 1000000 --val-limit 20000
	$(PY) scripts/train.py --max-steps 3000 --batch-size 8 --grad-accum-steps 4 --block-size 256 --eval-interval 250 --out-dir checkpoints/standard

overnight: setup
	$(PY) scripts/train_tokenizer.py --vocab-size 8192 --max-samples 300000
	$(PY) scripts/prepare_data.py --train-limit 1500000 --val-limit 25000
	$(PY) scripts/train.py --max-steps 8000 --batch-size 8 --grad-accum-steps 4 --block-size 256 --eval-interval 500 --out-dir checkpoints/overnight

clean-bench:
	rm -rf checkpoints/benchmark
