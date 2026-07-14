# Metis-1.5 RTX PRO 6000 NVFP4 Research Packet

Date: 2026-05-13

Purpose: give a stronger research model a grounded, codebase-specific packet for finding throughput optimizations on the actual Metis-1.5 training stack. This is not a generic FP4/MoE note. It is a snapshot of the current repo, launch path, precision policy, model math, data pipeline, and the most likely places where RTX PRO 6000 Blackwell NVFP4 performance could be won or lost.

User-reported live state: the 32k tokenizer has finished training. Base 50B pretrain memmap tokenization is running now. Continued pretrain will then be normalized, sharded, and tokenized with the same tokenizer. The main training target is an RTX PRO 6000 96GB box using NVFP4, MXFP8 on selected surfaces, BF16 on the most sensitive surfaces, and Muon-AdamW hybrid optimization.

Primary source files inspected:

- `configs/metis15_manifest.json`
- `configs/metis15_pretrain_mix.json`
- `configs/metis15_continued_pretrain_mix.json`
- `docs/metis15_multihead_latent_moe.md`
- `docs/metis15_data_plan.md`
- `src/metis_mamba/config.py`
- `src/metis_mamba/model.py`
- `src/metis_mamba/fp8.py`
- `src/metis_mamba/optim.py`
- `scripts/train_mamba_lm.py`
- `scripts/metis15_pretrain.sh`
- `scripts/metis15_full.sh`
- `scripts/metis15_cpu_prep.sh`
- `scripts/prepare_streaming_data.py`
- `scripts/prepare_normalized_shards.py`
- `scripts/data_mixture.py`
- `scripts/normalized_shard_mixture.py`
- `docker/runpod-metis-gpu/Dockerfile`
- `docker/runpod-metis-gpu/Dockerfile.runtime-liger`
- `docker/runpod-metis-gpu/README.md`
- `Makefile`

## One-Page Summary For The Research Model

Metis-1.5 is currently configured as:

- 1B-class sparse decoder, `950,973,312` actual named parameters.
- Active params including embedding/final norm: `297,251,712`.
- Active transformer params excluding embedding/final norm: `246,860,160`.
- `d_model=1536`, `n_layer=19`, `block_size=1024`, `vocab_size=32768`.
- Attention: 24 query heads, 8 KV heads, head dim 64, GQA, RoPE.
- FFN: multi-head Latent MoE, 4 feature heads of 384 dims each.
- Experts: 32 routed experts, top-4 selected per token-head, 1 shared expert always active.
- Routed expert payload is compressed from 384 to 320 dims.
- Expert hidden size is 1280 with fused gate/up projection and separate down projection.
- Router latent size is 128.
- Base pretrain is static dense in the sense of no Dynamic MoR, but still sparse MoE in the FFN.
- CPT switches on Dynamic token MoR with target average depth warmup from 1.05 to 1.65 over 1B CPT tokens.
- Precision: NVFP4 global Transformer Engine recipe, MXFP8 on QKV and LatentMoE projections, BF16 embeddings and tied LM head.
- Optimizer: custom local Muon-AdamW hybrid. Routed expert matrices are AdamW by default, not Muon.

Likely top optimization targets:

1. MoE dispatch is Python-loop based. The routed expert path loops over all 32 experts per layer, uses `topk_indices == expert_index`, `torch.nonzero`, `index_select`, expert forward, and `index_add_`. This is probably the biggest performance opportunity: grouped GEMM, fused scatter/gather, CUTLASS/TE grouped GEMM, MegaBlocks-style block sparse dispatch, or Triton kernels should be investigated first.

2. Current RTX/Blackwell target is not reflected in the Docker build. The GPU image is still named/described as a Hopper NVFP4 image, hard-codes `TORCH_CUDA_ARCH_LIST=9.0` and `NVTE_CUDA_ARCHS=90`, and builds FlashAttention from the Hopper path. For RTX PRO 6000, verify the actual compute capability and rebuild Transformer Engine/attention kernels for the card. If the card is not sm90, this is a critical mismatch.

3. The direct `make metis15-rtx-pretrain` path calls `scripts/metis15_pretrain.sh`, which does not pass `--lm-loss-impl liger_fused_linear_ce`. The full-pipeline wrapper defaults to Liger fused linear CE, but the standalone RTX pretrain wrapper defaults to standard LM head plus PyTorch cross entropy unless env/script changes are made. With 32k vocab this may be a real throughput leak.

4. The manifest has `attention_backend: "sdpa"` and `te_dot_product_attention` is false by default. FlashAttention-3 and TE DotProductAttention code paths exist, but the manifest excludes FA3 and the launcher only enables TE attention if `METIS15_TE_DOT_PRODUCT_ATTENTION=1`. Attention is about half the active transformer parameter budget per block, so this deserves direct benchmarking.

5. `torch.compile` is explicitly disabled when Transformer Engine low precision is active. There is no CUDA graph path despite config fields. If TE/NVFP4 can be made graph-compatible, launch overhead and Python dispatch might improve, but the current MoE Python routing probably blocks a clean graph.

6. The optimizer is custom Python. AdamW groups are stepped manually in Python with FP32 moment buffers; Muon groups run Newton-Schulz in Python over each 2D matrix. `--fused-adamw` is ignored for `muon_adamw`. The default policy leaves `747,110,400` routed expert parameters in AdamW, so optimizer step cost and state bandwidth should be profiled.

7. Data loading for LM pretrain uses random NumPy memmap indexing, CPU-side widening to `int64`, and direct `.to(device)`. No pinned prefetch, async loader, or GPU-side decode pipeline exists. It may not be the first bottleneck, but at a 600-800 TFLOP/s target it should be measured.

8. CPT Dynamic MoR adds more Python/token routing overhead: active-token packing loops by batch row, uses `torch.nonzero`, scatters with `scatter_add`, and calls the shared stack repeatedly. Base pretrain avoids this, but CPT performance will likely be worse unless this path is optimized or run in a dense-active mode.

9. Current TFLOP logging has two estimates. `est_tflops` uses total params, which is a dense-equivalent number for a sparse MoE. `est_active_tflops` uses active params and is closer to actual routed compute. Research should define the target metric before optimizing.

10. Quality/data hygiene gaps are still present: validation confirms bucket totals/caps, but the implementation does not yet perform the full promised fastText/CLD3-style English LID, global MinHash near-dedup, or benchmark decontamination pass. This is quality risk more than GPU throughput risk.

## Current Architecture Snapshot

Canonical manifest: `configs/metis15_manifest.json`.

Core model:

- `model_type`: `metis_multihead_latent_moe`
- `architecture`: `metis_multihead_latent_moe_decoder`
- vocab size: 32768
- sequence length: 1024
- `d_model`: 1536
- layers: 19
- query heads: 24
- KV heads: 8
- head dim: 64
- attention backend: `sdpa`
- hidden activation: SwiGLU
- tied embeddings: true
- RMSNorm: true
- attention dropout: 0
- attention bias: false
- MLP bias: false
- RoPE theta: 10000

MoE config:

- `ffn_type`: `multi_head_latent_moe`
- routed experts: 32
- top-k: 4
- shared experts: 1
- MoE heads: 4
- per-MoE-head width: 1536 / 4 = 384
- routed latent payload width: 320
- router latent width: 128
- expert intermediate size: 1280
- router score: sigmoid
- router temperature: 1.0
- balance strategy: `aux_loss_free_bias`
- balance bias update rate: 0.001
- balance bias clamp: 5.0
- `moe_aux_loss_coef`: 0.0
- balance updates scale by recursive active token fraction: true

Important implementation detail: the 32 routed experts are global `MetisHeadExpert` modules that process token-head rows. The router has per-MoE-head expert embeddings/biases, but the expert modules themselves are not duplicated per MoE feature head.

Per block parameter math from the current config:

- Attention block params:
  - Q: `1536 * 1536 = 2,359,296`
  - KV: `1536 * 1024 = 1,572,864`
  - O: `1536 * 1536 = 2,359,296`
  - Total attention per block: `6,291,456`
- One routed expert:
  - gate/up/down SwiGLU matrices over routed dim 320 and hidden 1280.
  - `3 * 320 * 1280 = 1,228,800`
- Shared expert:
  - original head dim 384 and hidden 1280.
  - `3 * 384 * 1280 = 1,474,560`
- Router per block:
  - latent projection: `384 * 128 = 49,152`
  - per-head expert embeddings: `4 * 32 * 128 = 16,384`
  - expert bias: `4 * 32 = 128`
  - total router-ish params: `65,664`
- Routed down/up projections:
  - `2 * 384 * 320 = 245,760`
- Active MoE MLP params per block:
  - `top4 * 1,228,800 + 1,474,560 + 65,664 + 245,760 = 6,701,184`
- Total MoE MLP params per block:
  - `32 * 1,228,800 + 1,474,560 + 65,664 + 245,760 = 41,107,584`
- Active transformer params:
  - `19 * (6,291,456 + 6,701,184) = 246,860,160`
- Total transformer params:
  - `19 * (6,291,456 + 41,107,584) = 900,581,760`
- Embedding table:
  - `32768 * 1536 = 50,331,648`
- Actual named params from meta-device model construction:
  - `950,973,312`

The active block split is important: attention and active MoE are similar magnitude. This means optimizing only experts may still leave attention as a major limiter, but the current expert dispatch is much less optimized than attention.

## Current Forward Pass Details

### Attention

Implementation: `MetisSelfAttention` in `src/metis_mamba/model.py`.

Current behavior:

- QKV is a single fused projection `qkv_proj`.
- Q, K, V are reshaped into GQA layout: Q has 24 heads, K/V have 8 heads.
- RoPE is applied to Q/K.
- Backend selection order:
  1. TE DotProductAttention if explicitly enabled and compatible.
  2. FlashAttention-3 if `attention_backend` is `auto` or `flash_attention_3`, no attention mask, CUDA, BF16/FP16, device capability major >= 9, and head dim <= 256.
  3. PyTorch `F.scaled_dot_product_attention`.
  4. Manual eager attention.
- Current manifest pins `attention_backend` to `sdpa`, so FA3 is not selected.
- `te_dot_product_attention` is false unless `--te-dot-product-attention` is passed.
- `native_gqa_attention` defaults true and SDPA tries `enable_gqa` if available.

Research questions:

- On RTX PRO 6000, is SDPA, FA3, TE fused attention, or another Blackwell path fastest for BF16 attention with head dim 64 and GQA 24q/8kv?
- Does the Hopper FlashAttention-3 package used in the Dockerfile produce correct/fast kernels on RTX PRO 6000, or should this be replaced with a Blackwell-appropriate attention stack?
- Should manifest `attention_backend` be changed from `sdpa` to `auto` or `flash_attention_3` after verifying correctness?
- Should `METIS15_TE_DOT_PRODUCT_ATTENTION=1` be used and benchmarked?
- Are `NVTE_FLASH_ATTN=1` and `NVTE_FUSED_ATTN=1` doing anything in the current path while TE attention is off?

### MoE Routing And Expert Dispatch

Implementation: `MetisMultiHeadLatentMoE` in `src/metis_mamba/model.py`.

Forward path:

1. Hidden states `[B, S, 1536]` reshape to `[B, S, 4, 384]`.
2. Flatten token-head rows to `[B*S*4, 384]`.
3. Optional routed down projection maps 384 -> 320.
4. Router latent projection maps each token-head 384 -> 128.
5. Router latents and expert embeddings are L2-normalized in float then cast back.
6. Router logits are `einsum("bshd,hnd->bshn")`, plus per-head expert bias, divided by temperature.
7. Sigmoid router scores are computed.
8. For aux-loss-free bias mode, non-gradient `balance_bias` is added only to selection scores.
9. Top-k experts are selected.
10. Top-k weights come from unbiased sigmoid scores, normalized over selected experts.
11. Shared expert processes all flattened token-head rows in original 384d space.
12. For each of 32 routed experts:
    - Build mask `topk_indices == expert_index`.
    - Skip if no selected rows.
    - Use `torch.nonzero(selected, as_tuple=True)`.
    - `index_select` routed inputs.
    - Run that expert's gate/up and down linears.
    - Weight outputs.
    - Accumulate via `routed_output.index_add_`.
13. Optional routed up projection maps 320 -> 384.
14. Sum shared plus routed output and reshape back to `[B, S, 1536]`.

This is correctness-readable but likely not peak-throughput:

- There is a Python loop over 32 experts per MoE layer.
- There are many small dynamic dispatches per layer.
- Expert input shapes vary by routing distribution.
- `nonzero`, `index_select`, and `index_add_` introduce launch overhead and memory movement.
- Expert GEMMs are not grouped across experts.
- Shared expert and routed experts are separate calls.
- No capacity padding or block-sparse layout is used.
- No all-to-all exists, which is fine for world_size=1, but the single-GPU dispatch still needs fused local grouping.

Expected base-pretrain row counts:

- Microbatch: 24 sequences.
- Sequence length: 1024.
- MoE heads: 4.
- Token-head rows per layer per microbatch: `24 * 1024 * 4 = 98,304`.
- Top-k assignments per layer per microbatch: `98,304 * 4 = 393,216`.
- Mean assignments per routed expert if balanced: about `12,288`.
- Expert input matrix around `[~12k, 320]` for each selected expert under ideal balance, but shape varies dynamically.

This shape is large enough that grouped expert GEMM should matter. The current path likely spends avoidable time in routing/scatter/gather and per-expert launch overhead.

Research targets:

- Transformer Engine grouped GEMM or grouped MLP for experts.
- CUTLASS grouped GEMM with FP4/NVFP4 support for 320x2560 and 1280x320 expert matrices.
- Triton fused MoE dispatch kernels for:
  - top-k packing,
  - grouped gate/up projection,
  - fused SwiGLU,
  - grouped down projection,
  - weighted accumulation.
- MegaBlocks/block-sparse expert dispatch ideas adapted to single-GPU top-k.
- DeepSpeed/Megatron/Tutel-style local MoE kernels, if compatible with NVFP4 or adaptable.
- Capacity-padded experts to regularize matrix shapes and enable static grouped GEMMs.
- Persistent routing buffers to avoid repeated allocation.
- Sorting token-head rows by expert once, then contiguous ranges per expert.
- Fusing routed down projection with dispatch, or moving routed down/up around grouped expert kernels if possible.
- Whether top-k=4 should be implemented as four expert streams or combined grouped batches.
- Whether router `einsum` is optimal; it is small but called every MoE layer.

### Aux-Loss-Free Balance Bias

Implementation details:

- `balance_bias` is a persistent buffer of shape `[moe_num_heads, moe_num_experts]`.
- It only affects top-k selection, not final gate weights.
- Final weights use unbiased router scores.
- During training, `_maybe_update_balance_bias` counts selected experts using `index_put_`, computes load fractions, and moves bias up/down by a fixed `update_rate`.
- In Dynamic MoR, update rate can scale by active token fraction for recursive passes.

Research questions:

- Is fixed-sign bias update too crude compared with overload-magnitude updates?
- Should updates run every step/layer, or be smoothed with EMA?
- Is the bias buffer handled correctly if world_size ever becomes >1? Current target is world_size=1.
- Does bias update become noisy with top-k=4 and 4 MoE heads?
- Does balance bias create extra sync/launch overhead worth fusing or delaying?

### Dynamic MoR

Base pretrain:

- Manifest stage `pretrain.training_mode` is `static_dense_pretrain`.
- `apply_training_mode_overrides` disables MoR and sets:
  - `mor_enabled=false`
  - `mor_train_router=false`
  - `mor_runtime_mode=disabled`
  - `mor_max_depth=1`
  - `attention_mask_mode=causal_none`
  - `disable_depth_stack=true`
  - `disable_token_packing=true`
  - `disable_token_scatter=true`

CPT:

- Manifest stage `continued_pretrain.training_mode` is `dynamic_token_mor`.
- Router depth budget: 1..3.
- Target average depth warms from 1.05 to 1.65 over first 1B CPT tokens.
- Router aux starts 0.01 and ends 0.02.
- Entropy coefficient: 0.002.
- Z-loss coefficient: 0.0001.
- Token packing/scatter are enabled by default.

Implementation:

- `MetisMoRModel._route_tokens` computes soft and hard depth routes.
- Training uses Gumbel softmax hard route selection.
- For each recursion step:
  - Build active mask for tokens with chosen depth >= step.
  - Either run dense-active mode or pack active tokens.
  - Packed mode loops over batch rows in `pack_active_tokens`, uses `torch.nonzero`, pads active tokens, runs shared stack, then scatters back.
- Env `METIS_MOR_TRAIN_ROUTING_MODE` can be `token_pack` or `dense_active`. Default is `token_pack`.

Research questions:

- For CPT on one RTX PRO 6000, is `token_pack` actually faster than `dense_active` after accounting for dynamic shape overhead and MoE dispatch?
- Can pack/scatter be moved to Triton?
- Can active token counts be bucketed/padded to static shapes to improve TE kernel reuse?
- Is the first 1B-token depth warmup conservative enough for stability while still allowing throughput?
- Should the Dynamic MoR path be delayed until after the base run is fully benchmarked?

## Current Precision Map

Implementation files: `src/metis_mamba/fp8.py`, `src/metis_mamba/config.py`, `src/metis_mamba/model.py`, and `scripts/train_mamba_lm.py`.

Launch flag:

- `METIS15_NVFP4=1` by default in `scripts/metis15_pretrain.sh`.
- This passes `--nvfp4` to `scripts/train_mamba_lm.py`.
- `--fp8` and `--nvfp4` are mutually exclusive.

NVFP4 requirements enforced by code:

- CUDA required.
- Transformer Engine required.
- `NVFP4BlockScaling` required.
- BF16 dtype required for master weights.
- If any configured surface uses MXFP8, `MXFP8BlockScaling` is also required.

Current manifest precision settings:

- `low_precision_mode`: `nvfp4`
- `nvfp4_disable_rht`: false
- `nvfp4_disable_2d_quantization`: false
- `nvfp4_disable_stochastic_rounding`: false
- `nvfp4_keep_embeddings_bf16`: true
- `nvfp4_keep_qkv_bf16`: false
- `nvfp4_keep_latent_moe_projections_bf16`: false
- `nvfp4_keep_lm_head_bf16`: true
- `nvfp4_qkv_precision`: `mxfp8`
- `nvfp4_latent_moe_projection_precision`: `mxfp8`
- `nvfp4_lm_head_precision`: `bf16`

Effective map in current code:

- Token embeddings: PyTorch `nn.Embedding` in BF16 after `model.to(dtype=torch.bfloat16)`.
- Tied LM head: PyTorch `nn.Linear` in BF16 because low precision is disabled for `lm_head`; weight is tied to embedding.
- QKV projection: TE Linear with local MXFP8 recipe.
- Attention output projection: TE Linear under outer NVFP4 recipe.
- Latent MoE router projection: TE Linear with local MXFP8 recipe.
- Routed down/up projections: TE Linear with local MXFP8 recipe.
- Routed expert gate/up/down: TE Linear under outer NVFP4 recipe.
- Shared expert gate/up/down: TE Linear under outer NVFP4 recipe.
- Dense SwiGLU MLP path, if used: TE Linear under outer NVFP4 recipe.
- RMSNorm/LayerNorm: TE RMSNorm if `use_fp8` is true. Not explicitly pinned in config; relies on TE behavior.
- Attention matmul/softmax itself: currently SDPA on BF16 Q/K/V outputs. QKV projection is MXFP8, but attention is not an explicit NVFP4 DPA path.

Potential precision questions:

- Is MXFP8 on QKV optimal, or should QKV be BF16 for stability or NVFP4 for speed?
- Should attention O projection remain NVFP4?
- Should norms be explicitly kept in BF16 if TE RMSNorm under NVFP4 causes instability or hidden casts?
- Are routed down/up projections truly routing-sensitive enough for MXFP8, or can they move to NVFP4 after warmup?
- Is the tied LM head BF16 enough, or should Liger fused CE be mandatory to avoid a huge BF16 vocab projection bottleneck?
- Do RHT, 2D quantization, and stochastic rounding materially affect quality/throughput on RTX PRO 6000?
- Does TE v2.8 in the current Docker image expose the same NVFP4/MXFP8 APIs expected by the code? The error text recommends TE 2.14+, but the Docker ARG is `TRANSFORMER_ENGINE_REF="v2.8"`.

## Current Optimizer Policy

Implementation: `src/metis_mamba/optim.py`.

Default optimizer from manifest:

- `name`: `muon_adamw`
- AdamW eps: `1e-8`
- Muon beta: `0.95`
- Muon Newton-Schulz steps: `5`
- Muon LR scale: `1.0`
- Muon Nesterov: true
- Include routed experts in Muon: false

Classification from a meta-device construction of the actual Metis-1.5 model:

| Group | Optimizer | Tensors | Params |
| --- | --- | ---: | ---: |
| bias/vector/scalar/norm-ish | AdamW | 58 | 62,336 |
| embedding or LM head | AdamW | 1 | 50,331,648 |
| latent MoE router projection | AdamW | 19 | 933,888 |
| routed expert projections | AdamW | 1216 | 747,110,400 |
| router/gate control | AdamW | 19 | 311,296 |
| attention projections | Muon | 38 | 119,537,664 |
| latent MoE payload projections | Muon | 38 | 4,669,440 |
| shared expert projections | Muon | 38 | 28,016,640 |

Total:

- AdamW params by default: about `798.75M`.
- Muon params by default: about `152.22M`.
- Routed experts dominate total parameters and are AdamW by default.

Important implementation details:

- AdamW is custom Python, not `torch.optim.AdamW`.
- Muon is custom Python.
- `--fused-adamw` is only used if optimizer is plain AdamW. It does nothing for `muon_adamw`.
- AdamW state uses FP32 `exp_avg` and `exp_avg_sq`.
- Muon state uses FP32 momentum buffer and Newton-Schulz orthogonalization over each matrix.
- Muon update scale is `sqrt(rows / cols)` clamped at min 1.0.
- Routed expert matrices can be moved to Muon via `--muon-include-routed-experts` or `METIS15_MUON_INCLUDE_ROUTED_EXPERTS=1`, but this is treated as an ablation, not default.

Research questions:

- Is the custom optimizer step a measurable bottleneck every 8 microbatches?
- Should AdamW groups use fused/foreach AdamW while Muon groups stay custom?
- Would including routed experts in Muon reduce memory enough to allow larger batch or improve quality per token, despite extra Newton-Schulz cost?
- Should Muon state/update be implemented with fused kernels or batched matrix orthogonalization?
- Do we need BF16/FP32 master weight handling that is more explicit than current `model.to(bf16)` plus TE recipe?
- Does TE keep FP4 module master weights in BF16 internally, or are PyTorch parameters themselves BF16 only?

Approximate optimizer memory from the current policy:

- BF16 parameters: `950,973,312 * 2 bytes = 1.90 GB`.
- BF16 gradients, if all present: another about `1.90 GB`.
- AdamW state for about `798.75M` params with two FP32 buffers: about `6.39 GB`.
- Muon state for about `152.22M` params with one FP32 buffer: about `0.61 GB`.
- Optimizer state total rough order: about `7.0 GB`, excluding allocator overhead and TE metadata.

Memory probably fits on 96GB, but optimizer bandwidth/launch overhead still deserves profiling.

## Current Training Launch Path

Make targets:

- `make metis15-rtx-pretrain`
- `make metis15-rtx-continued-pretrain`
- `make metis15-full`
- `make metis15-p5-pretrain` currently aliases `metis15-rtx-pretrain`.

Standalone pretrain launcher: `scripts/metis15_pretrain.sh`.

Defaults:

- Manifest: `configs/metis15_manifest.json`
- Data dir: `data/metis15_base`
- Output dir: `checkpoints/metis15_base`
- Stage: `pretrain`
- Resume: on
- FP8: off
- NVFP4: on
- TF32 flag: off
- matmul precision: `highest`
- optimizer: from manifest unless `METIS15_OPTIMIZER` set, so `muon_adamw`
- world size: manifest hardware world size, currently 1
- OMP threads: 8
- NCCL debug: WARN
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `NVTE_DEBUG=0`
- `NVTE_FLASH_ATTN=1`
- `NVTE_FUSED_ATTN=1`

Pretrain stage derived values:

- Local batch size: 24.
- Grad accumulation: 8.
- Sequence length: 1024.
- World size: 1.
- Tokens per optimizer step: `24 * 8 * 1024 = 196,608`.
- Target train tokens: 50B.
- Max steps: `254,314`.
- Warmup steps: `5,086`.
- LR: `1.2e-4`.
- Weight decay: `0.1`.
- Betas: `0.9`, `0.95`.
- Log interval: 20.
- Eval interval: 1000.
- Checkpoint interval: 2500.

CPT stage derived values:

- Local batch size: 16.
- Grad accumulation: 8.
- Sequence length: 1024.
- Tokens per optimizer step: `131,072`.
- Target train tokens: 10B.
- Max steps: `76,294`.
- Warmup steps: `2,289`.
- LR: `6e-5`.
- Weight decay: `0.1`.
- Betas: `0.9`, `0.95`.
- Checkpoint interval: 2500.

Important launch gap:

- `scripts/train_mamba_lm.py` defaults to `--lm-loss-impl standard`.
- `scripts/metis15_full.sh` defaults `METIS15_LM_LOSS_IMPL=liger_fused_linear_ce` and passes it.
- `scripts/metis15_pretrain.sh` does not pass `--lm-loss-impl` at all.
- Therefore the direct `make metis15-rtx-pretrain` route will use standard CE unless the script is changed or command path changes.

Research should test standard CE vs Liger fused linear CE on the actual RTX PRO 6000. For a 32k vocab, fused CE could matter.

## Current Docker/Image Situation

Files:

- `docker/runpod-metis-gpu/Dockerfile`
- `docker/runpod-metis-gpu/Dockerfile.runtime-liger`
- `docker/runpod-metis-gpu/README.md`

Current base image:

- `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Torch version asserted: 2.8.0.
- CUDA path: `/usr/local/cuda`.
- Transformer Engine built from source with `TRANSFORMER_ENGINE_REF="v2.8"`.
- FlashAttention built from `Dao-AILab/flash-attention.git`, `FLASH_ATTENTION_REF="v2.8.3"`, hopper path.
- Runtime-liger image installs `liger-kernel==0.8.0`.

Potential critical mismatch for RTX PRO 6000:

- Docker hard-codes `TORCH_CUDA_ARCH_LIST=9.0`.
- Docker hard-codes `NVTE_CUDA_ARCHS=90`.
- README says image is optimized for "Hopper NVFP4" and constrains CUDA builds to `sm_90`.
- RTX PRO 6000 is the target Blackwell card in this request, so research must verify whether sm90-only compiled kernels run, fall back, or fail on the actual GPU.
- If RTX PRO 6000 uses a different compute capability, rebuild TE/FlashAttention/custom kernels for that architecture.
- The code error text says TE 2.14+ recommended for NVFP4, while the Dockerfile pins TE v2.8. This may be stale or wrong, but it needs live verification on the target image.

Smoke checks to run on the target box before real training:

```bash
python - <<'PY'
import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("te", getattr(te, "__file__", "missing"))
print("has NVFP4", hasattr(recipe, "NVFP4BlockScaling"))
print("has MXFP8", hasattr(recipe, "MXFP8BlockScaling"))
PY
```

Also verify the actual architecture flags used to build TE and any attention packages, not just Python imports.

## Current LM Training Loop

Implementation: `scripts/train_mamba_lm.py`.

Runtime:

- Supports single GPU and DDP, but manifest target is world_size 1.
- Sets CUDA device if distributed.
- Seeds Python, NumPy, Torch, CUDA.
- Loads manifest and applies stage training overrides.
- Resolves precision.
- Loads memmap `train.bin` and `val.bin`.
- Builds model, optionally loads init checkpoint.
- Converts model to target dtype/device.
- Disables `torch.compile` if distributed or low precision/TE is enabled.
- Builds optimizer and scheduler.
- Resumes `latest.pt` if requested.
- Randomly samples batches from memmap each microstep.
- Runs forward with `return_logits=False`.
- Scales loss by grad accumulation and backprops.
- Clips gradients.
- Optimizer step and scheduler step.
- Logs loss, LR, depth, aux losses, tok/s, step seconds, tokens seen, estimated total TFLOP/s, estimated active TFLOP/s, and perf counters.
- Evaluates every configured interval.
- Saves `best.pt`, `latest.pt`, and gate checkpoints.

Batch loader:

```python
positions = np.random.randint(0, max_start, size=(batch_size,), dtype=np.int64)
offsets = positions[:, None] + np.arange(block_size, dtype=np.int64)[None, :]
batch = torch.from_numpy(np.asarray(data[offsets], dtype=np.int64))
x = batch.contiguous()
y = x.clone()
return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)
```

Possible issues:

- Data is stored as `uint16` because vocab is 32768, but training widens to `int64` on CPU every batch.
- No pinned memory is used.
- No background prefetch thread/process is used for LM pretrain.
- `non_blocking=True` may not help without pinned memory.
- Random advanced indexing over memmap may produce scattered CPU reads.
- At very high tok/s, CPU-side batch preparation could become visible.

Metric caveat:

The code estimates TFLOP/s as:

```python
6.0 * param_count * tokens_per_second / 1e12
```

It logs:

- `est_tflops` with total params, currently `950,973,312`.
- `est_active_tflops` with active params at the current mean MoR depth.

For sparse MoE, total-param TFLOP/s is dense-equivalent and not actual active compute. Active TFLOP/s is closer to actual routed work. If the target is "beat H100 450 TFLOP/s", define whether that was measured with total dense params, active params, profiler FLOPs, or a vendor counter.

Token/s needed under the current estimator:

| Target | Total-param tok/s | Active-param tok/s | Active-transformer tok/s |
| ---: | ---: | ---: | ---: |
| 450 TFLOP/s | 78,867 | 252,311 | 303,816 |
| 600 TFLOP/s | 105,155 | 336,415 | 405,088 |
| 800 TFLOP/s | 140,207 | 448,554 | 540,117 |

Those token/s requirements show why the metric definition matters.

## Current Data Pipeline Snapshot

CPU prep wrapper: `scripts/metis15_cpu_prep.sh`.

Stage order:

1. `setup`
2. `normalized_pretrain`
3. `tokenizer_sample`
4. `tokenizer_assets`
5. `pretrain_data`
6. `normalized_continued`
7. `continued_pretrain_data`
8. `chat_sft_data`
9. `reasoning_sft_data`
10. `reward_prefs`
11. `dpo_prefs`
12. `planning`
13. `complete`

Current run per user:

- Tokenizer has finished.
- Base pretrain memmap tokenization is running.
- CPT normalization/tokenization comes after base tokenization.

Data sizes from manifest:

- Base pretrain target: 50B train tokens, 1 percent validation.
- CPT target: 10B train tokens, 1 percent validation.
- Tokenizer sample target: 12M docs.
- Tokenizer vocab: 32768.
- Base normalized max docs: 70M in manifest.
- Continued normalized max docs: 16M in manifest.

Base bucket mix:

- 12B high-quality DCLM/DCLM-edu.
- 9B FineWeb-Edu/FineWeb-HQ.
- 5B newer quality web.
- 3B reference/encyclopedic.
- 5B academic STEM/science.
- 6B math/proof/symbolic.
- 3B open textbooks/educational reference.
- 2B long-form books.
- 2B knowledge QA/explanations.
- 2B synthetic educational prose.
- 1B reserve balancing pool.

CPT bucket mix:

- 1.2B general replay.
- 2.0B academic STEM.
- 2.0B math/proof documents.
- 1.2B verifiable math problem-solution prose.
- 1.2B reference/wiki/StackExchange.
- 0.8B FinePDFs technical OCR.
- 0.6B science instruction/literature as plain text.
- 0.5B long-form/book replay.
- 0.5B hard general decontaminated examples.

Prep implementation:

- `prepare_normalized_shards.py` normalizes each source into zstd JSONL shards and can mirror each shard to S3.
- It supports multi-process source workers.
- It reuses completed source manifests.
- It removes stale source shards before rewriting an incomplete source.
- It supports incremental HF parquet/jsonl/jsonl.zst/text/gzip loaders for several large datasets.
- `prepare_streaming_data.py` tokenizes normalized shards into flat `train.bin`/`val.bin`.
- Vocab <=65535 yields `uint16` memmap dtype.
- Tokenizer post-processor adds `<bos>` and `<eos>` by default when encoding documents.

Data hygiene caveat:

`docs/metis15_data_plan.md` explicitly says full data build still needs stronger fastText/CLD3-style LID, global MinHash near-dedup, and benchmark contamination pass before serious GPU spend. Current code enforces simpler filters:

- Latin letter ratio / alpha ratio.
- Repeat character run limits.
- URL count.
- max line length.
- selected row numeric/label filters when metadata exists.
- bucket totals and same-bucket fallback validation.

The validation command passed, but it validates plan structure and caps, not all promised hygiene.

## Things Already Optimized Or Partly Optimized

- QKV is fused into one projection.
- Dense SwiGLU uses fused gate/up projection.
- Expert SwiGLU also uses fused gate/up projection.
- TE Linear is used for dimensions that satisfy multiples of 16.
- All current key dimensions are multiples of 16:
  - 1536
  - 2560 QKV output
  - 384
  - 320
  - 1280
  - 128
  - 32768 vocab
- Outer TE autocast wraps the shared stack.
- MXFP8 local recipe is used for QKV and LatentMoE projections.
- Liger fused CE is wired in model code and used by the full-pipeline wrapper.
- Perf counters exist for attention backend, MoR packing/scatter, and MoE dispatch.
- NVTX ranges exist around optimizer zero, forward, backward, grad clip, optimizer step, LM head, cross entropy, and fused CE.
- Gate checkpoints exist at 10B, 25B, 40B, 50B for base and 2B, 5B, 8B, 10B for CPT.
- S3 hydration/resume exists for data and checkpoints.

## Things Not Present Today

No evidence found for:

- Triton kernels.
- Custom CUDA kernels.
- CUTLASS grouped GEMM wrappers.
- MegaBlocks-style block sparse MoE.
- Tutel/DeepSpeed/Megatron MoE dispatch.
- Expert-parallel all-to-all, which is fine for one GPU.
- Activation checkpointing.
- CUDA graph capture in the actual training loop.
- Real use of `config.cuda_graphs`.
- Fused add+norm residual kernels.
- Fused MoE SwiGLU expert kernels.
- Fused/foreach AdamW inside the Muon-AdamW hybrid optimizer.
- Async checkpoint writing.
- Pinned-memory prefetch for LM pretrain memmaps.
- Full implemented global MinHash dedup/decontam/LID pass.

## Highest-Value Experiments To Propose

### A. Establish The Real Baseline

Before changing kernels, run a short real target benchmark and capture:

- torch/CUDA/TE versions.
- GPU name and compute capability.
- Whether TE exposes NVFP4 and MXFP8 recipes.
- Whether actual modules are TE Linear vs PyTorch Linear.
- `tok/s`, `step_s`, `est_tflops`, `est_active_tflops`.
- Perf counters: `fa3`, `sdpa`, `te_attention`, `moe_routed_expert_dispatches`.
- Nsight Systems trace for 20-50 optimizer steps.
- Nsight Compute on representative expert GEMMs, attention, LM head, optimizer step.

Minimal run should use the same launch path as the real run, not a synthetic architecture.

### B. Fix The Blackwell Image Target

Investigate:

- Correct `TORCH_CUDA_ARCH_LIST` for RTX PRO 6000.
- Correct `NVTE_CUDA_ARCHS` for RTX PRO 6000.
- Transformer Engine release/branch that fully supports NVFP4/MXFP8 on the card.
- Whether the current Hopper FlashAttention build is appropriate.
- Whether a current NGC/PyTorch container already has better Blackwell TE kernels than building TE v2.8.

This is likely prerequisite work.

### C. Make Fused CE Mandatory For Pretrain Benchmark

Compare:

- `--lm-loss-impl standard`
- `--lm-loss-impl liger_fused_linear_ce`

Use identical batch, sequence length, precision, and manifest. If Liger wins, update `scripts/metis15_pretrain.sh` to pass `--lm-loss-impl` or add a real `METIS15_LM_LOSS_IMPL` hook there.

### D. Attention Backend Sweep

Compare:

- Current `sdpa`.
- `attention_backend=auto`.
- `attention_backend=flash_attention_3`.
- `METIS15_TE_DOT_PRODUCT_ATTENTION=1`.
- Native GQA on/off if needed.

Keep QKV precision fixed first. Capture both speed and correctness/stability.

### E. MoE Expert Dispatch Kernel

This is the largest structural opportunity.

Candidate approaches:

- Sort token-head rows by expert, then run grouped GEMM for all experts.
- Capacity-pad expert batches to static shapes.
- Use grouped GEMM for gate/up, fused SwiGLU, grouped down, and fused weighted scatter.
- Keep shared expert as a dense TE MLP or fold it into grouped flow.
- Avoid per-expert Python `nonzero`.
- Avoid repeated allocation of `output`, `routed_output`, selected masks, row indices, slot indices.
- Benchmark both exact dynamic routing and capacity-padded routing.

Target matrix shapes:

- Gate/up per routed expert: `[tokens_for_expert, 320] x [320, 2560]`.
- Down per routed expert: `[tokens_for_expert, 1280] x [1280, 320]`.
- Shared gate/up: `[98,304, 384] x [384, 2560]` per layer/microbatch in base pretrain.
- Shared down: `[98,304, 1280] x [1280, 384]`.

### F. Optimizer Step Optimization

Profile optimizer step separately. If it is visible:

- Use fused/foreach AdamW for AdamW groups inside the hybrid optimizer.
- Batch Muon Newton-Schulz where matrix shapes match.
- Consider routed experts in Muon as memory/quality/speed ablation.
- Consider lower optimizer frequency only if accumulation/lr math is adjusted.
- Consider decoupling routed expert optimizer policy from router/shared expert policy.

### G. Data Loader Prefetch

If CPU data prep appears in traces:

- Use pinned staging buffers.
- Use a background prefetch thread/process.
- Keep memmap dtype `uint16` longer and widen on GPU if embedding path permits `int32`/`int64` conversion there.
- Precompute random offsets in chunks.
- Consider sequential shard streaming for cache locality if random flat sampling is too expensive.

### H. CPT MoR Routing

After base:

- Benchmark `METIS_MOR_TRAIN_ROUTING_MODE=token_pack` vs `dense_active`.
- Profile `pack_active_tokens` and `scatter_active_tokens`.
- Consider Triton active-token packing/scatter.
- Consider bucketed active shapes.
- Verify MoE balance-bias scaling by active ratio is stable.

## Suggested Profiling Command Sketches

These are sketches for the target machine after data is present.

Environment introspection:

```bash
python - <<'PY'
import torch
import transformer_engine.pytorch as te
import transformer_engine.common.recipe as recipe
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("te", getattr(te, "__file__", "missing"))
print("NVFP4", hasattr(recipe, "NVFP4BlockScaling"))
print("MXFP8", hasattr(recipe, "MXFP8BlockScaling"))
PY
```

Short pretrain with direct launcher after adding a `METIS15_LM_LOSS_IMPL` hook to `scripts/metis15_pretrain.sh`:

```bash
METIS15_DATA_DIR=/path/to/metis15_base \
METIS15_OUT_DIR=/path/to/bench_metis15_base \
METIS15_NVFP4=1 \
METIS15_FP8=0 \
METIS15_RESUME=0 \
METIS15_LOCAL_BATCH_SIZE=24 \
METIS15_GRAD_ACCUM_STEPS=8 \
METIS15_LM_LOSS_IMPL=liger_fused_linear_ce \
./scripts/metis15_pretrain.sh
```

Note: `METIS15_LM_LOSS_IMPL` is not currently consumed by `scripts/metis15_pretrain.sh`. It is consumed by `scripts/metis15_full.sh`. For direct pretrain today, use the manual `torchrun` command below or patch the launcher before using the wrapper sketch above.

Manual direct command for controlled benchmarking:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_mamba_lm.py \
  --manifest configs/metis15_manifest.json \
  --data-dir /path/to/metis15_base \
  --out-dir /path/to/bench_metis15_base \
  --train-stage pretrain \
  --batch-size 24 \
  --grad-accum-steps 8 \
  --max-steps 200 \
  --warmup-steps 10 \
  --lr 1.2e-4 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --log-interval 10 \
  --eval-interval 0 \
  --checkpoint-interval 0 \
  --dtype bf16 \
  --matmul-precision highest \
  --optimizer muon_adamw \
  --nvfp4 \
  --lm-loss-impl liger_fused_linear_ce
```

Nsight Systems:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=nvtx \
  --capture-range-end=stop \
  -o metis15_base_nvfp4_profile \
  torchrun --standalone --nproc_per_node=1 scripts/train_mamba_lm.py ...
```

The script already emits NVTX ranges, but it does not currently call a start/stop capture marker. If using capture-range=nvtx, add a small marker or use time/step-based capture.

## Verification Performed In This Repo Snapshot

Commands run locally:

```bash
./.venv/bin/python scripts/validate_metis15_data_plan.py --manifest configs/metis15_manifest.json
PYTHONPATH=src ./.venv/bin/python -m py_compile src/metis_mamba/config.py src/metis_mamba/model.py src/metis_mamba/fp8.py src/metis_mamba/optim.py scripts/train_mamba_lm.py
bash -n scripts/metis15_pretrain.sh
bash -n scripts/metis15_full.sh
bash -n scripts/metis15_cpu_prep.sh
PYTHONPATH=src ./.venv/bin/python - <<'PY'
import json
from pathlib import Path
from metis_mamba.config import MetisMambaConfig
manifest=json.loads(Path('configs/metis15_manifest.json').read_text())
config=MetisMambaConfig.from_dict(manifest['model'])
config.validate()
print(config.to_dict()['estimated_params'])
print(config.to_dict()['estimated_active_params'])
print(config.nvfp4_surface_precision('qkv'))
print(config.nvfp4_surface_precision('latent_moe_projection'))
print(config.nvfp4_surface_precision('lm_head'))
PY
```

Results:

- Data plan validation passed with no errors.
- Python compile passed for the key Metis model/precision/optimizer/train files.
- Shell syntax checks passed for the Metis-1.5 CPU prep, full pipeline, and pretrain launchers.
- Config validation passed.
- Current config estimates:
  - total params: `950,973,312`
  - active params: `297,251,712`
  - QKV precision: `mxfp8`
  - LatentMoE projection precision: `mxfp8`
  - LM head precision: `bf16`

One false start during verification: I accidentally tried to `py_compile` a shell script, which correctly produced a Python syntax error. The proper `bash -n` shell checks then passed.

## Bottom Line

The current Metis-1.5 code is a strong correctness-first sparse MoE implementation, but it is not yet a throughput-specialized RTX PRO 6000 implementation. The biggest likely gap is local MoE dispatch: Python routing plus per-expert dynamic GEMMs will probably leave a lot of Blackwell NVFP4 performance unused. The second biggest risk is that the GPU image and attention stack still look Hopper/sm90-oriented while the requested target is RTX PRO 6000. The third is that the standalone pretrain launcher may miss fused linear CE even though the model code and full wrapper support it.

If the research model only has time for a few recommendations, ask it to prioritize:

1. Correct Blackwell TE/attention image build.
2. MoE grouped/fused expert dispatch for top-4 32-expert token-head routing.
3. Fused CE on the actual pretrain launch.
4. Attention backend sweep on BF16 Q/K/V with GQA.
5. Hybrid optimizer step profiling and fused AdamW integration.
6. CPT token-pack/scatter optimization after the base benchmark is stable.
