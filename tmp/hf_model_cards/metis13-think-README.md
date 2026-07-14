# Metis-1.3 Think

Metis-1.3 Think is the reasoning-tuned variant of the 201M-parameter Metis-1.3 family. It starts from the chat-tuned model and applies an additional reasoning-focused supervised fine-tuning stage.

## Model summary

- Family: Metis
- Stage: Reasoning SFT
- Parameters: 201,490,560
- Architecture: Mamba2-attention hybrid decoder
- Context length: 4096
- Vocabulary size: 8192
- Dtype: bfloat16 weights

## What changed from chat

This release starts from Metis-1.3 Chat and adds a reasoning-focused SFT stage aimed at improving:

- step-by-step problem solving
- structured answers
- deliberate response style on reasoning-heavy prompts

## Intended use

This is the most reasoning-oriented checkpoint in the Metis-1.3 line and is the best candidate for:

- small-model reasoning experiments
- side-by-side evals against the base and chat models
- testing whether extra reasoning-oriented post-training helps downstream behavior

## Limitations

- This is still a small model and can fail on complex reasoning, factuality, and long-horizon consistency.
- The reasoning SFT dataset in this run saturated very quickly, so this release should be treated as an early reasoning-tuned checkpoint rather than a fully mature post-training result.
- English is the intended primary language.

## Recommended variants

- Use **Metis-1.3 Base** for raw pretrained behavior.
- Use **Metis-1.3 Chat** for the more balanced conversational variant.
- Use **Metis-1.3 Think** when you specifically want the reasoning-tuned release.

## Files

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

## License

This release inherits the licensing and attribution obligations of the upstream training and post-training data sources used in the Metis pipeline. Review dataset licenses and usage constraints before production use.
