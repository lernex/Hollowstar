# TinyStories Local Trainer

This is a minimal, learn-by-building setup for training a small GPT-style language model locally on Apple silicon.

The pipeline is:

1. Download TinyStories with Hugging Face `datasets`
2. Train a BPE tokenizer with `tokenizers`
3. Encode the dataset into `train.bin` / `val.bin`
4. Train a roughly 10M-parameter causal language model with PyTorch
5. Sample text from a saved checkpoint

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

Tokenize TinyStories into binary files:

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

- `scripts/train_tokenizer.py`: trains a BPE tokenizer on TinyStories
- `scripts/prepare_data.py`: tokenizes TinyStories and writes binary token files
- `scripts/train.py`: trains the transformer
- `scripts/prepare_chat_data.py`: turns JSONL chat examples into SFT tensors
- `make chat-data` / `./run.sh chat-data`: by default uses a 20k/2k subset of `HuggingFaceH4/ultrachat_200k`
- `scripts/train_sft.py`: fine-tunes a base checkpoint on chat data
- `scripts/generate.py`: loads a checkpoint and samples text
- `scripts/chat_app.py`: runs a local browser UI for chatting with checkpoints
- `scripts/prepare_streaming_data.py`: writes binary training data from a streaming text corpus
- `scripts/prepare_reasoning_sft.py`: builds short-form reasoning SFT data from OpenThoughts
- `scripts/data_mixture.py`: streams the weighted Metis 1.1 corpus mix, including Python-Edu blob fetches
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
