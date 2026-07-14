# Metis-1.3 Base

Metis-1.3 Base is a 201M-parameter English-first decoder model from the Metis family. It uses a hybrid Mamba2 plus attention backbone and was trained as the base model for later chat and reasoning post-training.

## Model summary

- Family: Metis
- Stage: Base pretraining checkpoint exported for Hugging Face
- Parameters: 201,490,560
- Architecture: Mamba2-attention hybrid decoder
- Context length: 4096
- Vocabulary size: 8192
- Dtype: bfloat16 weights

## Architecture

Metis-1.3 uses 28 decoder blocks:

- 21 Mamba2 blocks
- 7 attention blocks at layers 3, 7, 11, 15, 19, 23, and 27
- Grouped-query attention with 18 query heads and 6 KV heads
- Tied input and output embeddings
- RMSNorm

This release is intended as the raw pretrained base for further instruction tuning and reasoning tuning.

## Training

- Pretraining target: 12B train tokens
- Tokenizer: custom 8k tokenizer trained on an 8M-document sample
- Intended data mix: English-first web, educational, math, and code-heavy mixture

## Intended use

This checkpoint is mainly useful for:

- research and experimentation on small hybrid Mamba models
- base-model comparison against the chat and think variants
- continued fine-tuning or alignment work

It is not the most user-friendly conversational variant. For direct assistant use, the chat or think releases are likely better starting points.

## Limitations

- This is a small model and will still make factual, reasoning, and instruction-following mistakes.
- The eval setup used in this project is lightweight and should be treated as a smoke test rather than a comprehensive benchmark.
- English is the intended primary language.

## Files

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

## License

This release inherits the licensing and attribution obligations of the upstream training data sources used in the Metis pipeline. Review dataset licenses and usage constraints before production use.
