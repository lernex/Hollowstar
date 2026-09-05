---
license: cc0-1.0
language:
- en
pipeline_tag: text-generation
inference: false
base_model: Lernex/Metis-1.5-base
tags:
- metis
- mixture-of-experts
- moe
- latent-moe
- jax
- tpu
- instruction-tuned
- chat
- reasoning
- text-generation
---

# Metis-1.5-think

**Metis-1.5-think** is the **instruction-tuned** variant of [`Metis-1.5-base`](https://huggingface.co/Lernex/Metis-1.5-base) — an **898M-parameter (≈340M active/token) single-latent Mixture-of-Experts** model that was pretrained from scratch on 50B tokens and then supervised-fine-tuned to **follow instructions, chat, and reason**. Trained end to end in **pure JAX on a single TPU v6e-8**.

Unlike the base model (which only continues text), Metis-1.5-think **answers**: give it an instruction or a question and it responds directly. It answers concisely and does **not** emit a visible `<think>` chain-of-thought — its instruction data was trained without retained reasoning traces, so it states answers rather than narrating its reasoning.

> [!NOTE]
> This is a **small** model (≈340M active params). It follows instructions and reasons at a level appropriate to its size — capable and surprisingly coherent for sub-1B, but not competitive with multi-billion-parameter assistants. It has **no RLHF**; alignment comes only from a safety subset in the SFT mix.

---

## At a glance

| | |
|---|---|
| **Base** | [Lernex/Metis-1.5-base](https://huggingface.co/Lernex/Metis-1.5-base) (50B-token pretrain) |
| **Post-training** | Supervised fine-tuning (SFT), ~1.15M curated examples, ~1.2 epochs |
| **Total / active params** | 898M / ≈340M per token |
| **Architecture** | Single-latent MoE decoder — 19 layers, `d_model` 1536, 32 experts (top-4) + 1 shared |
| **Attention** | GQA, 24 query / 8 KV heads, head_dim 64, RoPE (NeoX) |
| **Context length** | 1024 tokens |
| **Chat format** | `User:` / `Assistant:` template (see below) |
| **Precision** | bf16 weights |
| **Hardware** | 1× TPU v6e-8, JAX/XLA |
| **License** | CC0-1.0 (public domain — zero restrictions) |

---

## Chat format

Metis-1.5-think was fine-tuned with a simple role-tagged template:

```
User: {your message}
Assistant: {model reply}
```

For a single turn, prompt the model with (a leading `<bos>` token, then):

```
User: <your message>
Assistant: 
```

and let it generate until `<eos>`. Multi-turn conversations stack `User:`/`Assistant:` lines in order. A system message may be prepended as `System: {content}\n`. The model answers directly — it does **not** emit `<think>` reasoning blocks.

---

## Training data (SFT mix)

~**1.15M examples** (fixed 1024-token windows, prompt tokens loss-masked), assembled from open instruction & reasoning datasets, then **English-filtered, deduplicated, and decontaminated** against the standard benchmark suite (MMLU, GSM8K, MATH, ARC, HellaSwag, …). Roughly:

**Chat & instruction (~0.75M)** — Tülu-3 (general instruction, writing/summarization, safety), WildChat (cleaned real prompts), SmolTalk (everyday, system-chats, rewrite, summarize, magpie, system-constraints), OpenHermes, SciRIFF (science instruction & reference tasks), OpenStax reference QA, and a small identity set.

**Reasoning & math (~0.38M)** — OpenR1-Math (verified), NuminaMath (CoT medium/olympiad, 1.5 verified), OpenThoughts-3 (math, science, general non-code), Bespoke-Stratos (proofs), s1K (symbolic proofs), TemplateGSM, and reasoning-style SmolTalk. These were used as **direct-answer** supervision — the source `<think>` reasoning traces were stripped during data prep — so the model learns to *answer* math/reasoning prompts rather than to narrate a visible chain-of-thought. (A future revision could retain the traces to make it an explicit chain-of-thought model.)

**Safety** — a Tülu-3 safety subset (~60k) for basic refusal/safety behavior.

Credit for the underlying datasets belongs to their original authors (see Acknowledgements).

---

## Training procedure

- **SFT from `Metis-1.5-base`** (initialized from base params; fresh optimizer/schedule).
- ~**1.2 epochs**, ≈2,685 steps, global batch 64 × 8 grad-accum (≈524k tokens/step), sequence length 1024.
- **Optimizer: AdaMuon** (Newton–Schulz-orthogonalized momentum + Adam), **fp32 master weights / bf16 compute**, base LR 4e-5 with short warmup + cosine decay.
- Prompt tokens are **loss-masked** (`-100`); the model only learns to produce the assistant turns.
- Pure JAX/XLA on one **TPU v6e-8**, ≈245k tokens/sec. SFT cross-entropy fell from ≈1.70 to ≈1.11 over the run.
- **Identity-alignment patch:** a short follow-up SFT pass (identity Q&A + general-data rehearsal) so the model identifies as **Metis, created by Lernex** — rather than inheriting the identity of the assistants its instruction data was partly distilled from.

---

## Evaluation

0-shot accuracy on the **full test split** of each benchmark, scored with a custom JAX harness — multiple-choice by length-normalized loglikelihood (`acc_norm`) or plain loglikelihood (`acc`); GSM8K by greedy generation (chat template + flexible numeric extraction). Training data was decontaminated against these benchmarks, so these are clean held-out numbers.

| Benchmark | Metric | Random | Metis-1.5-base | Metis-1.5-think |
|---|---|:---:|:---:|:---:|
| ARC-Easy | acc_norm | 25.0 | 41.3 | 41.7 |
| ARC-Challenge | acc_norm | 25.0 | 25.9 | 28.2 |
| HellaSwag | acc_norm | 25.0 | 30.4 | 31.0 |
| PIQA | acc_norm | 50.0 | 54.7 | 54.6 |
| WinoGrande | acc | 50.0 | 51.5 | 51.8 |
| OpenBookQA | acc_norm | 25.0 | 29.6 | 28.6 |
| BoolQ | acc | ~62¹ | 47.7 | 57.2 |
| MMLU | acc | 25.0 | 23.6 | 23.3 |
| GSM8K | acc | ~0 | — | 7.6 |

¹ BoolQ majority-class baseline ≈ 62%. GSM8K not run for the base model (non-instruct).

**How to read these.** Metis-1.5 is a **~340M-active** model (898M total, MoE) trained on **only 50B tokens** — far fewer than the 0.3–18T behind modern sub-2B models — so it lands around **GPT-2-medium tier**: clearly above chance on ARC-Easy, modestly so on HellaSwag / PIQA / WinoGrande, and at chance on MMLU. Supervised fine-tuning leaves raw knowledge roughly unchanged (base ≈ think on multiple choice) while adding instruction-following — visible on **BoolQ (+9.5)** and **GSM8K (7.6%)**, the latter notably strong for the scale (TinyLlama-1.1B / Pythia-1B sit ~2–3%), reflecting the math/reasoning-heavy data mix. The clearest lever for higher scores is **more training tokens**, not a different architecture.

---

## Intended uses & limitations

**Use it for** — instruction following, lightweight chat, and math/reasoning demonstrations at small scale; research on efficient MoE post-training; a base for further fine-tuning or preference optimization.

**Limitations**
- **Small** model — limited world knowledge and reasoning depth; will confidently hallucinate.
- **No RLHF / preference optimization** — only SFT, with a modest safety subset; **not** a safety-aligned assistant.
- **English-only**, **1024-token** context.
- Answers on reasoning/math prompts are frequently wrong even when fluent, and the model does **not** show its work (no chain-of-thought) — verify anything important.

---

## How to use

> [!NOTE]
> Weights use **JAX-native tensor names** and a custom architecture, so this does **not** load via `transformers.AutoModel`. It ships as a self-describing safetensors release (`config.json` has all dims; the base model card documents the forward pass).

Load the raw tensors with `safetensors` and run the forward described in the [base model card](https://huggingface.co/Lernex/Metis-1.5-base), wrapping inputs in the chat template above:

```python
from safetensors import safe_open
weights = {}
with safe_open("model.safetensors", framework="numpy") as f:
    for k in f.keys():
        weights[k] = f.get_tensor(k)

# Prompt format (prepend <bos>, id 1):
#   "User: <message>\nAssistant: "
# Generate to <eos> (id 2); stop early if "\nUser:" appears.
```

A reference JAX/CPU chat implementation (KV-cache decode + this template) is part of the Metis training stack; a standalone single-file loader is a planned follow-up.

## License

Released under **[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/)** — public-domain dedication. Use, modify, redistribute, fine-tune, or build on Metis-1.5-think for **any purpose, with no restrictions and no attribution required**. Provided **as-is, without warranty**. (Underlying SFT datasets retain their own licenses; CC0 applies to these released weights.)

## Citation

```bibtex
@misc{metis15think2026,
  title  = {Metis-1.5-think: An instruction-tuned single-latent MoE language model},
  author = {Lernex},
  year   = {2026},
  howpublished = {Hugging Face},
  note   = {898M params (340M active); SFT of Metis-1.5-base on ~1.15M curated examples, JAX/TPU v6e-8}
}
```

## Acknowledgements

Thanks to the open instruction/reasoning-data community — Tülu-3 (AI2), WildChat, SmolTalk, OpenHermes, SciRIFF, OpenStax, OpenR1, NuminaMath, OpenThoughts, Bespoke-Stratos, s1K, and TemplateGSM — and to the JAX and Cloud TPU teams.
