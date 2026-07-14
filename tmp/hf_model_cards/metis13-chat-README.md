# Metis-1.3 Chat

Metis-1.3 Chat is the instruction-tuned conversational variant of the 201M-parameter Metis-1.3 base model. It starts from the pretrained hybrid Mamba2-attention base and is post-trained for more helpful chat behavior.

## Model summary

- Family: Metis
- Stage: Chat SFT
- Parameters: 201,490,560
- Architecture: Mamba2-attention hybrid decoder
- Context length: 4096
- Vocabulary size: 8192
- Dtype: bfloat16 weights

## What changed from base

This release starts from Metis-1.3 Base and applies supervised fine-tuning on a chat-style dataset designed to improve:

- instruction following
- conversational formatting
- general helpfulness
- cleaner assistant behavior

## Intended use

This is the best general-purpose conversational checkpoint in the Metis-1.3 line for:

- lightweight assistant experiments
- browser or app integration tests
- comparison against the raw base model and the think model

## Limitations

- This remains a small model and can still hallucinate, overfit to style, or miss nuance.
- The SFT data in this run appears relatively easy and narrow, so low training loss should not be overinterpreted as broad capability.
- English is the intended primary language.

## Recommended variants

- Use **Metis-1.3 Base** for raw pretrained behavior.
- Use **Metis-1.3 Chat** for general conversational use.
- Use **Metis-1.3 Think** when you want the reasoning-tuned variant.

## Files

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

## License

This release inherits the licensing and attribution obligations of the upstream training and post-training data sources used in the Metis pipeline. Review dataset licenses and usage constraints before production use.
