# Metis Local Trainer

This is a minimal, learn-by-building setup for training small GPT-style language models locally, with the main focus now on the Metis model line.

The pipeline is:

1. Download or stream a text corpus with Hugging Face `datasets`
2. Train a BPE tokenizer with `tokenizers`
3. Encode the dataset into `train.bin` / `val.bin`
4. Train a roughly 10M-parameter causal language model with PyTorch
5. Sample text from a saved checkpoint

## Metis-1.6 login2/Rhea data factory

Metis-1.6's production data recipe is separate from the local teaching flow below. It locks exactly
1T final-tokenizer-measured exposures across 700B/250B/50B phases, including a 90B freshness layer.
The exact recipe and operational gates are documented in
[`docs/metis16_pretraining_data_plan.md`](docs/metis16_pretraining_data_plan.md); the machine-readable
source of truth is [`manifests/metis-1.6.yaml`](manifests/metis-1.6.yaml).
The same acquisition command retrieves the extra long-document reserve for an
exact 18B-token context extension with 6B/12B/18B promotion gates; see
[`docs/metis16_context_extension_data.md`](docs/metis16_context_extension_data.md).
Every planned source also has an ordered, fail-closed automatic shortfall path in
[`manifests/replacements.yaml`](manifests/replacements.yaml); see the
[`replacement research and policy`](docs/metis16_replacement_data_research.md).

Split login2/Rhea/Portage operator interface:

```bash
git clone git@github.com:lernex/Metis.git
cd Metis
./ops/start-acquisition.sh \
  --lustre-root /lus/lustre1/vollmerc/metis-1.6 \
  --quota-acknowledgement administrator-confirmed
```

That command securely prompts for the Hugging Face token and, when no `gh` credential is available,
a read-only GitHub metadata token. It runs acquisition inside GNU Screen on `login2` and may safely outlive SSH. Rhea later
runs CPU preparation with `--profile rhea`. After Rhea has sealed both the
tokenized base release and the required post-training umbrella, Portage runs
the simultaneous Praxis/Logos base, context-extension, and post-training
campaign with:

```bash
./ops/start-portage-training.sh
```

Tokenized shards alone are not a complete Portage handoff: the post-training
DeepSeek capability, verifiers, evaluation bundles, sealed generation
adapters, and a compatible Portage runtime are mandatory fail-closed inputs.
The production campaign performs mixed 8K-131K SFT, cross-tokenizer sequence
DPD, five independent Metis specialists, same-tokenizer OPD consolidation,
preference alignment, and dynamic thinking-length control. The profiles refuse an
implicit path or `/lus/lustre1` itself, and Rhea stays sealed until
its scheduler facts and direct view of the acquisition directory are confirmed. See
[`docs/metis16_operator_runbook.md`](docs/metis16_operator_runbook.md).

## Why this setup

This is intentionally simple and educational, not production-scale:

- one dataset
- one tokenizer
- one small decoder-only transformer
- one plain PyTorch training loop

That makes it much easier to see how the pieces actually fit together.

## Environment

Use the existing local virtualenv in this folder:

```bash
source .venv/bin/activate
python --version
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Optional Hugging Face token:

Create a local `.env` file with:

```bash
HF_TOKEN=your_hugging_face_token_here
```

The provided `Makefile` and `run.sh` load `.env` automatically.

## Quick Start

Train the tokenizer:

```bash
python scripts/train_tokenizer.py \
  --vocab-size 8192 \
  --max-samples 200000
```

Tokenize the legacy TinyStories starter corpus into binary files:

```bash
python scripts/prepare_data.py
```

Train a ~10M parameter model:

```bash
python scripts/train.py \
  --max-steps 3000 \
  --batch-size 8 \
  --grad-accum-steps 4
```

Generate text:

```bash
python scripts/generate.py \
  --prompt "Once upon a time" \
  --max-new-tokens 120
```

Or use the shortcuts:

```bash
make setup
make tokenizer
make prepare
make train
make generate
make chat-data
make sft
make chat
make chat-fast
make app
make base
make full-chat
make fast
make standard
make overnight
```

or

```bash
./run.sh setup
./run.sh tokenizer
./run.sh prepare
./run.sh train
./run.sh generate
./run.sh chat-data
./run.sh sft
./run.sh chat
./run.sh chat-fast
./run.sh app
./run.sh base
./run.sh full-chat
./run.sh fast
./run.sh standard
./run.sh overnight
```

Prepare chat fine-tuning data:

```bash
python scripts/prepare_chat_data.py \
  --hf-dataset HuggingFaceH4/ultrachat_200k \
  --hf-train-split train_sft \
  --hf-val-split test_sft \
  --hf-train-limit 20000 \
  --hf-val-limit 2000
```

Fine-tune the base model into a simple chat model:

```bash
python scripts/train_sft.py \
  --base-checkpoint checkpoints/fast/best.pt \
  --train-data data/chat_sft/train.pt \
  --val-data data/chat_sft/val.pt \
  --out-dir checkpoints/chat_sft
```

Generate from the chat checkpoint:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/chat_sft/best.pt \
  --prompt "How does a tokenizer work?" \
  --max-new-tokens 80
```

One-command chat fine-tune from your `fast` checkpoint:

```bash
make chat-fast
```

Local browser app:

```bash
make app
```

Then open [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Metis Run

For a fresh 10M model aimed more at explanation and short reasoning than story completion, this repo now includes a second recipe:

- base corpus: `HuggingFaceFW/fineweb-edu` (`sample-10BT`, streamed subset)
- reasoning tune: `open-thoughts/OpenThoughts-114k` (short-form filtered subset)

One command for the full new line:

```bash
make metis-full
```

Or stage by stage:

```bash
make metis-tokenizer
make metis-data
make metis-base
make metis-think-data
make metis-think
make metis11-base
make metis11-full
make metis100-base
make metis100-full
```

The checkpoints land at:

- `checkpoints/metis_base/best.pt`
- `checkpoints/metis_think/best.pt`

## Legacy Starter Recipe

The original TinyStories starter flow is still in the repo for comparison and quick smoke tests, but it is no longer the main path for current Metis work.

## Metis 1.1 Presets

If you want a stronger local model without immediately jumping into truly painful runtimes, this repo now includes a fuller Metis 1.1 recipe built around higher-quality small-model data:

- pretraining mix:
  - `HuggingFaceTB/smollm-corpus` / `fineweb-edu-dedup`
  - `HuggingFaceTB/smollm-corpus` / `cosmopedia-v2`
  - `HuggingFaceTB/smollm-corpus` / `python-edu`
  - `HuggingFaceTB/finemath` / `finemath-4plus`
- chat SFT: `HuggingFaceTB/smol-smoltalk`
- reasoning SFT: `open-thoughts/OpenThoughts-114k` (shortened into compact `<think>` traces)

The pretraining mix lives in [configs/metis11_pretrain_mix.json](/Users/giulianno/Documents/10M%20model/configs/metis11_pretrain_mix.json).

The two larger presets are:

- `make metis11-base`: about `54.9M` params (`Metis 1.1`)
- `make metis100-base`: about `104.0M` params

Recommended next step on a MacBook Air:

```bash
make metis11-full
```

That now runs:

1. a mixed-corpus tokenizer build
2. mixed pretraining data prep
3. the `~55M` Metis 1.1 base run
4. a SmolTalk conversational SFT stage
5. a compact OpenThoughts reasoning SFT stage

The base preset uses:

- `block_size=512`
- `n_layer=16`
- `n_head=8`
- `n_embd=512`
- `batch_size=4`
- `grad_accum_steps=8`
- `max_steps=5000`

The experimental 100M preset uses:

- `block_size=512`
- `n_layer=20`
- `n_head=8`
- `n_embd=640`
- `batch_size=2`
- `grad_accum_steps=8`

Rough runtime expectations on this machine, using the earlier Metis 1.0 run as a reference:

- `Metis 1.1` (`~55M`): roughly `10-14 hours` for the 5000-step base run, plus extra time for tokenizer/data prep and the two SFT stages
- `Metis 100` (`~104M`): roughly `12-16 hours` for the 3000-step base run

Those are still local-toy scales, not Chinchilla-optimal training budgets, but the Metis 1.1 recipe is now much more data-aware than the old single-dataset path.

## Metis 1.2 Runpod Flow

Metis 1.2 is the new Runpod-oriented path for the next model line:

## Metis 1.3 Runpod Flow

Metis 1.3 is the new large-step recipe:

- backbone: `Mamba2` + sparse attention hybrid
- size: about `200M` parameters
- context: `4096`
- pretraining budget: `12B` tokens
- precision: `BF16`

CPU prep:

```bash
make metis13-cpu-memory-prep
make metis13-cpu-compute-prep
```

GPU train/export/eval:

```bash
make metis13-gpu-full
```

Recommended split-pod flow:

1. RAM-heavy CPU pod:
   - `make metis13-cpu-memory-prep`
   - builds the tokenizer sample on fast local disk
   - trains the tokenizer on fast local disk
   - renders final HF/tokenizer assets
   - syncs only final tokenizer/config assets to the network volume

2. Compute-heavy CPU pod:
   - `make metis13-cpu-compute-prep`
   - hydrates tokenizer/config assets from the network volume to fast local disk
   - builds pretraining memmaps on local disk, then syncs them
   - builds chat/reasoning JSONL on local disk, then syncs them
   - writes the final derived training plan and syncs it

3. GPU pod:
   - `make metis13-gpu-full`
   - trains base/chat/think and exports release folders

If you ever want the old single-pod fallback, you can still run:

```bash
make metis13-cpu-prep
```

Final artifacts written to the network volume after the two CPU pods:

- tokenizer assets to `artifacts/metis13_hf_assets`
- pretraining memmaps to `data/metis13_base`
- chat SFT JSONL to `data/metis13_chat_sft`
- reasoning SFT JSONL to `data/metis13_reasoning_sft`
- derived plan to `plans/metis13_plan.json`

Intermediate work stays on fast local disk whenever possible:

- tokenizer sample JSONL
- tokenizer fitting workspace
- memmap build workspace
- SFT dataset build workspace

The GPU flow trains:

- `checkpoints` under the shared run root for base/chat/think
- clean export folders for base/chat/think releases
- an eval comparison JSON for the three stages

- name: `Metis-1.2`
- size target: about `100M-110M`
- context: `1024`
- target stack: `TorchTitan` + `torchao.float8` on the GPU pod
- release format: Hugging Face `model.safetensors` directories, not raw `.pt` checkpoints

The new config files live here:

- [configs/metis12_manifest.json](/Users/giulianno/Documents/10M%20model/configs/metis12_manifest.json)
- [configs/metis12_pretrain_mix.json](/Users/giulianno/Documents/10M%20model/configs/metis12_pretrain_mix.json)
- [configs/metis12_chat_mix.json](/Users/giulianno/Documents/10M%20model/configs/metis12_chat_mix.json)
- [configs/metis12_reasoning_mix.json](/Users/giulianno/Documents/10M%20model/configs/metis12_reasoning_mix.json)
- [configs/metis12_eval_prompts.json](/Users/giulianno/Documents/10M%20model/configs/metis12_eval_prompts.json)

The workflow is intentionally split across two pods that share the same network volume:

1. CPU pod:
   - build the tokenizer
   - render the local HF asset folder
   - stream and tokenize the pretraining corpus into memmap binaries with a `4B` train-token target plus a small held-out val split
   - prepare the chat and reasoning JSONL SFT sets
   - write a derived run plan with step counts and warmups
2. GPU pod:
   - install a CUDA `12.8+` PyTorch nightly plus TorchTitan and torchao
   - run base, chat, and think stages with the prepared data
   - export BF16 Hugging Face releases for `base`, `chat`, and `think`
   - compare the three stages with the manual eval suite

The main entrypoints are:

```bash
make metis12-cpu-prep
make metis12-gpu-full
```

or

```bash
./run.sh metis12-cpu-prep
./run.sh metis12-gpu-full
```

The Runpod scripts now bake in the operational lessons from the first real Metis 1.2 pod bring-up:

- both CPU and GPU flows write caches and temp files to the shared volume instead of local `/tmp`
- both flows refuse to start a second overlapping run by default via shared-volume locks
- both flows write a live stage file and status file under the shared volume so you can tell where the run is without guessing from `ps`
- fresh pods prefer a modern bootstrap interpreter like `python3.12` when creating `.venv`

The status files live here by default:

- CPU prep stage: `$METIS12_SHARED_ROOT/state/metis12_cpu_prep_stage.txt`
- CPU prep status: `$METIS12_SHARED_ROOT/state/metis12_cpu_prep_status.env`
- GPU stage: `$METIS12_SHARED_ROOT/state/metis12_gpu_stage.txt`
- GPU status: `$METIS12_SHARED_ROOT/state/metis12_gpu_status.env`

If a pod dies and leaves behind a stale lock, you can intentionally clear it on the next start:

```bash
METIS12_FORCE_UNLOCK=1 ./run.sh metis12-cpu-prep
METIS12_FORCE_UNLOCK=1 ./run.sh metis12-gpu-full
```

Practical Runpod notes we want to keep for Metis 1.3 and later:

- `/workspace` is the network volume, but `/` is usually a much smaller local container disk
- `pip` and big wheel installs will happily fill local `/tmp` unless `TMPDIR` is redirected
- do not launch a second prep or training shell against the same shared root unless you mean to replace the first one
- after a fresh CPU pod comes up, verify the interpreter with `python3.12 --version` before assuming `python3` is new enough

The key supporting scripts are:

- `scripts/runpod_metis12_cpu.sh`
- `scripts/runpod_metis12_gpu.sh`
- `scripts/render_metis12_hf_assets.py`
- `scripts/prepare_metis12_sft_data.py`
- `scripts/plan_metis12.py`
- `scripts/assemble_hf_release.py`
- `scripts/eval_model_suite.py`
- `src/metis_titan/`

## Metis 1.5 Release Eval

After the Metis-1.5 releases are exported, run the local release sanity suite
against the base and think folders:

```bash
python -m pip install -r requirements-eval.txt
python scripts/eval_model_suite.py \
  --suite configs/metis15_eval_prompts.json \
  --model base=releases/metis15/base \
  --model think=releases/metis15/think \
  --output-path releases/metis15/eval_comparison.json
```

The main pipeline can run the same check with `METIS15_RUN_EVAL=1 make metis15-full`.

## Recommended first run

For your first pass, keep it short:

```bash
python scripts/train_tokenizer.py --vocab-size 8192 --max-samples 50000
python scripts/prepare_data.py --train-limit 200000 --val-limit 5000
python scripts/train.py --max-steps 500 --batch-size 6 --grad-accum-steps 4
python scripts/generate.py --prompt "Hello, how are you?"
```

That gives you a fast end-to-end sanity check before you commit to a longer run.

## Files

- `scripts/train_tokenizer.py`: trains a BPE tokenizer on text data
- `scripts/prepare_data.py`: tokenizes the legacy starter dataset and writes binary token files
- `scripts/train.py`: trains the transformer
- `scripts/prepare_chat_data.py`: turns JSONL chat examples into SFT tensors
- `make chat-data` / `./run.sh chat-data`: by default uses a 20k/2k subset of `HuggingFaceH4/ultrachat_200k`
- `scripts/train_sft.py`: fine-tunes a base checkpoint on chat data
- `scripts/generate.py`: loads a checkpoint and samples text
- `scripts/chat_app.py`: runs a local browser UI for chatting with checkpoints
- `scripts/prepare_streaming_data.py`: writes binary training data from a streaming text corpus
- `scripts/prepare_reasoning_sft.py`: builds short-form reasoning SFT data from OpenThoughts
- `scripts/data_mixture.py`: streams the weighted Metis 1.1 corpus mix, including Python-Edu blob fetches
- `scripts/runpod_metis12_cpu.sh`: shared-volume CPU prep for Metis 1.2
- `scripts/runpod_metis12_gpu.sh`: TorchTitan FP8 GPU training flow for Metis 1.2
- `src/metis_titan/`: custom TorchTitan configs and memmap dataloader for Metis 1.2
- `src/tinylm/model.py`: model definition

## Notes for Apple Silicon

- The scripts prefer `mps` automatically when available.
- Training on a fanless MacBook Air is fine for learning, but keep expectations modest.
- Start with shorter runs and a smaller batch size if memory gets tight.
- If `mps` behaves oddly, run with `--device cpu` to compare.

## What “10M parameters” means here

The default config is close to 10M parameters:

- `n_layer=10`
- `n_head=8`
- `n_embd=256`
- `block_size=256`
- `vocab_size=8192`

That is a useful “small enough to train, big enough to learn something” size.

## Next Steps

After this works, good learning upgrades are:

- add cosine LR decay + warmup
- add validation perplexity tracking
- try a larger context window
- compare BPE vs character tokenization
- try a slightly larger model after the pipeline is stable
- add better instruction/chat data and compare base vs SFT checkpoints
