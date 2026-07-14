# Metis-1.5 RTX PRO 6000 FP8 Throughput Research Packet

Date: 2026-05-14  
Repo: `/Users/giulianno/Documents/10M model`  
Live test instance: `54.144.73.7`, EC2 instance id `i-0d3c96f9acc371c1f`  
Purpose: handoff packet for a stronger research model to find the fastest possible RTX PRO 6000 FP8 training path for Metis-1.5.

## 1. Executive Summary

Metis-1.5 is intended to be a sparse ~1B total parameter model with about 0.29B to 0.31B active parameters per token. The current local manifest describes it as a multi-head LatentMoE decoder with 32 routed experts, top-4 routing, 1 shared expert, 4 MoE heads, and Dynamic MoR for continued pretraining. Base pretraining is static dense, so Dynamic MoR is not active in the current throughput tests.

The important live finding is harsh: the current best verified RTX PRO 6000 Blackwell FP8 path is only about 28.7k tok/s, not the expected 170k to 200k tok/s. The training log's own active-param target estimate says 450 active TFLOP/s would correspond to about 240,956 tok/s. The best live run is about 53 active TFLOP/s. Matching the previous Metis-1.4 H100 FP8 reference of about 170k tok/s requires roughly a 5.9x throughput increase from the current best. Reaching 200k tok/s requires about a 7.0x increase.

The profile strongly suggests the current bottleneck is not data loading and not primarily the optimizer. The main issues are launch/API overhead, extreme GEMM fragmentation, TE grouped GEMM overhead, MoE sort/gather/unpermute/scatter work, unfused SwiGLU elementwise kernels, and a currently inefficient all-FP8 expert path. Custom CUDA/Triton/CUTLASS kernels, CUDA graphing, persistent buffers, fused MoE dispatch, and/or a different grouped expert backend are all in scope if they get throughput up.

The canonical repo is also still partially stale from the NVFP4/MXFP8 exploration. Several launch paths and manifest fields still default to or describe NVFP4, even though the target should now be true FP8 with BF16 where stability requires it. The fastest live logs are explicitly `Low precision mode: fp8`, `Precision path: FP8 compute with BF16 master weights`, and `FP8 precision map: experts=bf16`.

## 2. Verified Baseline Facts

### 2.1 Target and expectation

User-provided target context:

- Previous Metis-1.4 H100 FP8 training reference: about 170k tok/s.
- RTX PRO 6000 dense FP8 compute is expected to be about half of H100 dense FP8 compute.
- Metis-1.5 has sparse active experts and should have lower active expert compute than Metis-1.4's 500M dense model.
- User target: at least match 170k tok/s, ideally push into the 200k tok/s range.
- Custom kernels are acceptable. The goal is maximum practical throughput on RTX PRO 6000.

Current live best:

| Metric | Current best live value |
| --- | ---: |
| Best max throughput | 28,668 tok/s |
| Best last throughput | 28,511 tok/s |
| Best log | `train-fp8-b16g24-bf16experts-adamwloop-src-te216main.log` |
| Best stable b16/g12 max | 28,301 tok/s |
| Active TFLOP/s at best | about 53.25 |
| Active-param target printed by trainer for 450 TFLOP/s | 240,956 tok/s |
| Ratio to 170k tok/s | 16.9% |
| Required speedup to 170k | 5.9x |
| Ratio to 200k tok/s | 14.3% |
| Required speedup to 200k | 7.0x |

### 2.2 Live instance state

Verified over SSH on 2026-05-14:

- Public IPv4: `54.144.73.7`
- EC2 instance id: `i-0d3c96f9acc371c1f`
- Instance type: `g7e.2xlarge`
- Region/AZ: `us-east-1`, `us-east-1b`
- AMI id: `ami-082ecb0714b440c33`
- IAM profile: `lernex-p5-ssm-role`
- Local IPv4: `172.31.86.44`
- Hostname: `ip-172-31-86-44`
- OS/kernel: Ubuntu 24.04 era kernel, `Linux 6.17.0-1013-aws #13~24.04.1-Ubuntu SMP Fri Apr 24 21:50:45 UTC 2026 x86_64`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Driver: `595.58.03`
- CUDA runtime reported by driver: `13.2`
- GPU memory: 97,887 MiB total
- Power cap: 600 W
- Persistence mode: on
- At inspection time: no running Docker containers, no tmux sessions, no GPU processes.
- Fast local disk: `/opt/dlami/nvme`, 1.7T total, about 1.5T available at inspection.

SSH command used:

```bash
ssh -i ~/.ssh/aws_codex_builder.pem ubuntu@54.144.73.7
```

The old RunPod key path was not accepted by this EC2 instance. The AWS builder key worked.

### 2.3 Live Docker images

Images present on the instance:

| Image | Image id prefix | Size | Notes |
| --- | --- | ---: | --- |
| `metis15-blackwell-fp8-ngc2604:sm120a` | `78be6680c99c` | 35.6 GB | Current best FP8/TE-main image |
| `metis15-ngc2604-fp8-runtime:te214` | `2f323c968a7d` | 35.2 GB | Earlier NGC 26.04 FP8 runtime |
| `nvcr.io/nvidia/pytorch:26.04-py3` | `192d749b4d77` | 34.9 GB | Base NGC image |

Build log for `metis15-blackwell-fp8-ngc2604:sm120a` reports:

- PyTorch: `2.12.0a0+0291f960b6.nv26.04.48445190`
- CUDA: `13.2`
- Transformer Engine: `2.16.0.dev0+76c2a9e`
- TE exposes `NVFP4BlockScaling`: true
- TE exposes `MXFP8BlockScaling`: true
- Liger fused linear cross entropy installed.
- Final image manifest list sha: `78be6680c99c337b519263e151aeff1d15db3bb7fef2f41322f78bae0e1e060f`

Important: TE exposing NVFP4/MXFP8 does not mean those recipes are viable for the RTX PRO 6000 training path. The current target is true FP8, not NVFP4/MXFP8.

### 2.4 Live code parity

The live remote code under `/opt/dlami/nvme/metis/10M-model` matches the local repo for the key files checked by SHA-256:

| File | SHA-256 |
| --- | --- |
| `configs/metis15_manifest.json` | `eed81ea70765df529b75d4913b4df197c0d926e6cbef1c4df2a21272c0da3ef4` |
| `scripts/metis15_pretrain.sh` | `604a58adfe57e33b9f670aa5e0b30c08aaaea2c9adbf8743d36afb62a9df318a` |
| `scripts/metis15_full.sh` | `49d12da33fe5bba6f32a17577323cad2fa4b78c36c67e43f9fb759f8458d82d6` |
| `scripts/metis15_rtx_benchmark_matrix.sh` | `edbc967ce2cb07a035cb2b6216b154c2891493860e10c093b1f634f880ddddf5` |
| `scripts/train_mamba_lm.py` | `f95e64ba978a87d6f7b792404ac622daa3e4a34c8f742419e008a866c8c97a23` |
| `src/metis_mamba/model.py` | `3a7204524db42c1b7c0b864b4caed5c6d95dcca5a801eb07bd66e9550286b509` |
| `src/metis_mamba/fp8.py` | `2b2670e75bbfeef02e1c9d8e30278dd0267ada1bd9a449f756306ae0584d3537` |
| `src/metis_mamba/optim.py` | `34e393efaeac6ea5d3fb758eb73886b85c32c11f5a475b9786f7101abede054f` |
| `Makefile` | `37ee9bd13e291c8082e74f6b247b1bf10c2892ea89f23d21bcd807e55a62cc7c` |

So the live logs can be interpreted against the current local source.

## 3. Data Assets on Instance

Main live root:

```text
/opt/dlami/nvme/metis
```

Important files:

```text
/opt/dlami/nvme/metis/data/metis15_base/meta.json
/opt/dlami/nvme/metis/logs/*.log
/opt/dlami/nvme/metis/profiles/metis15_fp8_b16g12_bf16experts_src_te216.nsys-rep
/opt/dlami/nvme/metis/profiles/metis15_fp8_b16g12_bf16experts_src_te216.sqlite
/opt/dlami/nvme/metis/profiles/stats/*.csv
```

Data meta facts:

- Source mode: mixture
- Mixture config: `configs/metis15_pretrain_mix.json`
- Tokenizer path recorded in meta: `/workspace/metis15_cpu_prep/artifacts/metis15_hf_assets/tokenizer.json`
- Vocab size: 32,768
- Sequence block size used by training config: 1,024
- Max docs for base prep: 70,000,000
- Validation ratio: 0.01
- Target train tokens: 50,000,000,000
- Target validation tokens: 505,050,506
- Packed dtype: `uint16`
- Encode batch size: 1,024
- Hydrated local data path: `/opt/dlami/nvme/metis/data/metis15_base`

Source count examples from `meta.json`:

- `dclm_baseline_hq`: about 11.196B train tokens
- `fineweb_edu`: about 6.481B train tokens
- `nemotron_math`: about 4.139B train tokens
- `dclm_edu`: about 3.672B train tokens
- `common_corpus_reference`: about 2.974B train tokens
- `proof_pile2_arxiv_science`: about 2.775B train tokens
- `finewiki`: about 2.343B train tokens
- `finemath`: about 2.125B train tokens
- `essential_web`: about 1.823B train tokens

Batch fetch was effectively zero in the NSYS NVTX profile, so the hydrated memmap input pipeline is not the first throughput suspect.

## 4. Model Architecture in Current Manifest

Source: `configs/metis15_manifest.json`

Top-level description says:

- Metis-1.5
- Sparse Metis model
- About 1B-class total parameters
- A0.24B transformer-active in the manifest description
- Multi-head LatentMoE decoder
- 32 routed experts
- Top-4 routing per head
- 1 shared expert

Core model dimensions:

| Field | Value |
| --- | ---: |
| `vocab_size` | 32,768 |
| `block_size` | 1,024 |
| `d_model` | 1,536 |
| `n_layer` | 19 |
| `n_heads` | 24 |
| `n_kv_heads` | 8 |
| `head_dim` | 64 |
| Attention | GQA, 24 Q heads, 8 KV heads |
| Attention backend | `sdpa` |
| Training mode for base | `static_dense_pretrain` |
| MoR enabled for base | false |
| FFN type | `multi_head_latent_moe` |
| Hidden act | SwiGLU |
| RMSNorm | true |
| Tie embeddings | true |
| Attention bias | false |
| MLP bias | false |
| Dropout | 0.0 |

MoE dimensions:

| Field | Value |
| --- | ---: |
| MoE heads | 4 |
| MoE head dim | 384 (`1536 / 4`) |
| Routed latent dim | 384 |
| Router latent size | 128 |
| Routed experts | 32 |
| Top-k | 4 |
| Shared experts | 1 |
| Expert intermediate size | 1,280 |
| Router score | sigmoid |
| Balance strategy | aux-loss-free bias |
| Balance update rate | 0.001 |
| Balance clamp | 5.0 |
| Dispatch mode | grouped |
| Capacity factor | 0.0 |
| Capacity alignment | 128 |

Parameter estimates from manifest/current code:

| Estimate | Value |
| --- | ---: |
| Total params | 1,095,725,952 |
| Active params | 311,260,032 |
| Active transformer params | 260,868,480 |

Important architectural detail from the code:

- There are 4 MoE routing heads, but the grouped expert weight bank is `num_experts=32`, not `num_heads * num_experts`.
- Routing embeddings and routing bias are per MoE head: shape `[4, 32, 128]` and `[4, 32]`.
- Expert computation is done on flattened token-head rows, and the same 32 expert modules serve all 4 MoE heads.
- With batch 16 and sequence length 1024:
  - token rows = `16 * 1024 = 16,384`
  - token-head rows = `16,384 * 4 = 65,536`
  - routed assignments per layer = `65,536 * top_k 4 = 262,144`
  - average assignments per expert per layer = `262,144 / 32 = 8,192`
  - per optimizer step at grad accum 12 and 19 layers: `262,144 * 19 * 12 = 59,768,832` routed assignments
- The logs confirm this arithmetic for b16/g12: `moe_grouped 228` and `assign 59768832`, where `228 = 19 layers * 12 grad accum`.

## 5. Precision State: Intended vs Current Code

### 5.1 What the target should be now

Current target based on failed RTX PRO 6000 NVFP4/MXFP8 attempts:

- Use true FP8 training similar in spirit to H100/Hopper delayed scaling.
- Keep master weights in BF16.
- Keep stability-sensitive pieces in BF16 where needed.
- Stop relying on NVFP4/MXFP8 for RTX PRO 6000.

### 5.2 What the manifest still says

The manifest still contains stale NVFP4-first fields:

- `model.low_precision_mode`: `nvfp4`
- `hardware.preferred_precision`: `nvfp4_sm120_safe_with_fp8_block_surfaces`
- `hardware.nvfp4.enabled`: false
- `hardware.fp8.enabled`: true
- NVFP4 fields for keeping embeddings/lm head BF16 and using `fp8_block` surfaces for qkv/latent projections.

This is confusing and risky. For the actual main run, the canonical config and launch targets should be made unambiguous: true FP8 target, NVFP4/MXFP8 disabled unless explicitly running old experiments.

### 5.3 How the live FP8 runs are actually configured

Live FP8 logs report:

```text
Low precision mode: fp8
Precision path: FP8 compute with BF16 master weights
FP8 precision map: experts=bf16
```

The fastest currently verified path is not all-FP8. It is FP8 autocast around TE-supported linears, but routed/shared experts are forced BF16 via `--fp8-expert-precision bf16`. The BF16 expert path still uses Transformer Engine GroupedLinear where available, but under an FP8-disabled context for expert modules.

In source:

- `src/metis_mamba/fp8.py` builds a TE `DelayedScaling` recipe for true FP8.
- `build_linear` uses TE `Linear` if `use_fp8` is true and shapes are multiples of 16.
- `build_grouped_linear` uses TE `GroupedLinear` when available, even for BF16 expert mode.
- `MetisGroupedLinear` disables FP8 autocast when `force_bf16` is true.
- `config.expert_precision_for_layer()` returns `fp8_expert_precision` when `low_precision_mode == "fp8"`.

### 5.4 NVFP4 and MXFP8 guardrails in code

`src/metis_mamba/fp8.py` has runtime guards:

- TE may expose NVFP4 on RTX PRO 6000 / SM120, but default production NVFP4 recipe uses SM100-oriented RHT/2D/stochastic-rounding kernels that fail exact Metis GEMMs. Runtime support on SM120 is only considered true when all reduced flags are set:
  - `disable_rht`
  - `disable_2d_quantization`
  - `disable_stochastic_rounding`
- MXFP8 runtime support returns false for capability >= 12.0 because TE 2.15 raised that MXFP8 is not supported on 12.0+ architectures yet.

This matches the overall operational pivot: treat NVFP4/MXFP8 as failed/not viable for the current RTX PRO 6000 main training path. Use true FP8.

## 6. Launchers and Configuration Risk

### 6.1 `scripts/metis15_pretrain.sh`

Key defaults:

- `METIS15_FP8` defaults to `0`
- `METIS15_NVFP4` defaults from manifest unless explicitly set
- If `METIS15_FP8=1` and `METIS15_NVFP4` was not explicitly set, the script disables NVFP4
- `METIS15_LM_LOSS_IMPL` defaults to `liger_fused_linear_ce`
- `METIS15_PREFETCH_BATCHES` defaults to `4`
- `MATMUL_PRECISION` defaults to `highest`
- `TF32` defaults to `0`
- Exports:
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `NVTE_FLASH_ATTN=1`
  - `NVTE_FUSED_ATTN=1`

The torchrun command uses:

- `--dtype bf16`
- `--fused-adamw`
- `--matmul-precision highest`
- `--lm-loss-impl liger_fused_linear_ce`
- `--prefetch-batches 4`
- `--fp8` only when enabled
- `--nvfp4` only when enabled

### 6.2 `scripts/metis15_full.sh`

This script is still dangerous for the new target:

- `ENABLE_FP8="${METIS15_FP8:-0}"`
- `ENABLE_NVFP4="${METIS15_NVFP4:-1}"`

So the full pipeline still defaults to NVFP4 unless overridden. It appends `--nvfp4` first if enabled, otherwise appends FP8 args.

Before a main FP8 run, this needs to be flipped or guarded so the full flow cannot accidentally launch NVFP4.

### 6.3 `Makefile`

Relevant targets:

- `metis15-rtx-pretrain`: forces `METIS15_FP8=0 METIS15_NVFP4=1`
- `metis15-rtx-fp8-pretrain`: forces `METIS15_FP8=1 METIS15_NVFP4=0`
- `metis15-rtx-continued-pretrain`: still forces NVFP4
- `metis15-full`: still forces NVFP4
- `metis15-rtx-benchmark-matrix`: runs `scripts/metis15_rtx_benchmark_matrix.sh`

Recommendation: make the FP8 target canonical before the main run. Leave NVFP4 as an explicitly named legacy experiment only.

### 6.4 Benchmark matrix

`scripts/metis15_rtx_benchmark_matrix.sh` supports:

- `LOW_PRECISION=nvfp4` by default, but can be set to `fp8`
- `--fp8-expert-precision`
- `--fp8-pad-multiple`
- SDPA vs TE attention
- Liger vs standard CE
- native GQA toggles
- capacity factor
- no-prefetch case

The script is useful, but its default `LOW_PRECISION=nvfp4` should be changed for this hardware target.

## 7. Training Loop and Runtime Details

Source: `scripts/train_mamba_lm.py`

Important behavior:

- `--fp8` and `--nvfp4` are mutually exclusive.
- FP8 requires CUDA, TE, BF16 dtype, and capability major >= 9.
- The error text says Hopper-class, but the code permits SM120 because capability `(12, 0)` passes `>= (9, 0)`.
- `config.low_precision_mode` is set to `fp8` when `--fp8` is used.
- `--fp8-expert-precision` can be `fp8` or `bf16`; the fastest live path uses `bf16`.
- `--fp8-pad-multiple` controls grouped expert split padding in all-FP8 expert mode.
- Compile is disabled for low precision unless `--allow-low-precision-compile` is passed. Current best runs do not use `torch.compile`.
- `CudaBatchPrefetcher` uses a CPU worker thread, pinned memory, and async H2D copy. It widens labels to long on GPU after copy.
- Training calls the model with `return_logits=False`, allowing Liger fused linear CE to avoid materializing full logits during training.
- Throughput logging prints:
  - `tok/s`
  - `step_s`
  - estimated total TFLOP/s
  - estimated active TFLOP/s
  - attention counters
  - MoE counters
  - MoR counters
  - capacity/padding counters

## 8. MoE Dispatch Anatomy

Source: `src/metis_mamba/model.py`

### 8.1 Routing

Per MoE layer:

1. Reshape hidden states `[B, S, 1536]` to `[B, S, 4, 384]`.
2. Flatten token-head rows for computation.
3. Latent router projection maps 384 to 128.
4. Router latents and expert embeddings are normalized.
5. Router logits are produced with `einsum("bshd,hnd->bshn", ...)`.
6. Sigmoid scores are used.
7. If aux-loss-free balancing is enabled, `balance_bias` is added to selection scores.
8. `torch.topk` selects 4 experts per token-head row.
9. Top-k scores are normalized by sum.
10. Balance bias is updated with `counts.index_put_(..., accumulate=True)`.

Potential issue:

- Routing uses several standalone PyTorch ops: normalize, einsum, sigmoid, topk, gather, sum, counts/index_put.
- The profile's top direct bottlenecks are not the router alone, but routing contributes to the launch count and dynamic shapes.

### 8.2 Grouped routed expert dispatch

Current grouped path:

1. Flatten top-k expert ids.
2. `torch.bincount(flat_experts)` to get tokens per expert.
3. `torch.argsort(flat_experts)` to group assignments by expert.
4. Compute `row_indices = order // top_k`.
5. Gather weights and routed hidden rows with `index_select`.
6. Optionally pad per-expert splits.
7. Call `MetisGroupedHeadExperts`, which does:
   - grouped gate/up projection
   - `F.silu(gate) * up`
   - grouped down projection
8. If padded, select only valid positions.
9. Allocate `routed_output = torch.zeros_like(routed_heads)`.
10. `index_add_` weighted expert outputs back to token-head rows.

The NSYS profile says this path matters:

- `moe_routed_grouped`: 8.8% NVTX time
- `moe_grouped_sort`: 5.4% NVTX time
- `moe_grouped_experts`: 3.1% NVTX time
- `moe_grouped_gather`: 0.1% NVTX time
- `moe_grouped_unpermute`: 0.1% NVTX time by NVTX, but GPU kernel summary has `indexFuncLargeIndex ReduceAdd` at 3.1%

Note: the NVTX numbers are not the full story because many TE/cuBLAS ranges and PyTorch kernels overlap conceptually with the MoE ranges.

### 8.3 Expert MLP implementation

Each grouped expert bank has:

- `gate_up_proj`: grouped linear 384 -> 2560
- split into gate and up, each 1280
- `F.silu(gate) * up`
- `down_proj`: grouped linear 1280 -> 384

Current important point:

- `TE fused MLP: False` in logs.
- SwiGLU is not fused. In the NSYS kernel summary, elementwise multiply and SiLU forward/backward kernels are prominent.

### 8.4 All-FP8 expert path

All-FP8 experts have not won so far:

- `train-fp8-b16g12-allfp8.log`: about 13.9k tok/s
- `train-fp8-b16g12-allfp8-pad16.log`: about 14.3k tok/s
- `train-fp8-b16g12-allfp8-pad1.log`: fails before step logging with TE grouped GEMM error:

```text
Leading dimension requirement on A for FP8 GEMM. Caller must pad.
```

This implies TE grouped FP8 experts need alignment/padding, and the current aligned all-FP8 expert path is much slower than BF16 experts with no padding. This may be due to padding overhead, FP8 quantization overhead, small/fragmented grouped GEMMs, cublasLt plan/heuristic overhead, or a suboptimal TE GroupedLinear path for these shapes on SM120.

## 9. Attention Path

Current manifest pins `attention_backend: "sdpa"`.

Logs for the current best report:

- `Attention backend: sdpa`
- `Native GQA attention: True`
- `TE dot-product attention: False`
- `TE fused MLP: False`
- `fa3 0`
- `sdpa 228` for b16/g12, which again equals `19 layers * 12 grad accum`

Older benchmark rows show:

- BF16 experts with TE attention: about 14.5k tok/s
- BF16 experts no-GQA: about 14.6k tok/s
- BF16 experts SDPA/native GQA: about 14.7k tok/s

Those older comparisons were in the slower padded/foreach family, so they are not a final attention conclusion. Still, attention is not the first visible bottleneck in the profile. MoE dispatch/GEMM fragmentation and launch overhead are much more prominent.

Research model should still check whether Blackwell-supported FlashAttention/FA3 or TE DotProductAttention can help after the MoE path is fixed, but do not spend first effort there.

## 10. Optimizer Path

Source: `src/metis_mamba/optim.py`

Current optimizer:

- `MuonAdamWHybrid`
- AdamW epsilon: `1e-8`
- Manifest default AdamW implementation: `foreach`
- Live fastest logs use `adamw_impl=loop`
- Muon beta: `0.95`
- Newton-Schulz steps: `5`
- Nesterov: true
- `include_routed_experts`: false

Live optimizer param split from logs:

| Group | Params | Notes |
| --- | ---: | --- |
| Muon | 147,554,304 | attention matrices, shared expert matrices, selected dense matrices |
| AdamW | 948,171,648 | embeddings, routed experts, router/gate/MoR, norms/scalars |
| Total | 1,095,725,952 | matches manifest |

AdamW breakdown:

- `adamw_decay`: 1,255 tensors, 948,109,312 params
- `adamw_no_decay`: 58 tensors, 62,336 params
- `muon`: 76 tensors, 147,554,304 params

Profile:

- `optimizer_step`: 1.5% NVTX time
- `optimizer_muon`: 0.8%
- `optimizer_adamw`: 0.7%
- `grad_clip`: 0.1%

Conclusion:

- Optimizer time is not the leading throughput bottleneck in the short NSYS capture.
- Optimizer memory may be important. Routed experts are in AdamW by default, so they carry two FP32 state tensors. This likely contributes to the batch 17/18 OOM boundary.
- Including routed experts in Muon is an ablation-only policy in the manifest, but it might reduce optimizer state memory. Research should evaluate stability and memory tradeoffs before recommending it for main training.

## 11. Live Benchmark Results

The following table was parsed from logs in `/opt/dlami/nvme/metis/logs`.

| Log | Max tok/s | Last tok/s | Steps | Last step_s | Active TFLOP/s | pad_tok | Experts | AdamW impl | Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `train-fp8-b16g24-bf16experts-adamwloop-src-te216main.log` | 28,668 | 28,511 | 3 | 13.79 | 53.25 | 0 | BF16 | loop |  |
| `train-fp8-b16g16-bf16experts-adamwloop-src-te216main.log` | 28,468 | 28,223 | 4 | 9.29 | 52.71 | 0 | BF16 | loop |  |
| `train-fp8-b16g12-bf16experts-routeidx-src-te216main.log` | 28,301 | 28,261 | 5 | 6.96 | 52.78 | 0 | BF16 | loop |  |
| `train-fp8-b16g12-bf16experts-adamwloop-src-te216main.log` | 28,277 | 27,946 | 5 | 7.04 | 52.19 | 0 | BF16 | loop |  |
| `train-fp8-b16g12-bf16experts-adamwloop.log` | 28,196 | 28,196 | 4 | 6.97 | 52.66 | 0 | BF16 | loop |  |
| `train-fp8-b15g13-bf16experts-unstable-sort.log` | 28,137 | 27,819 | 4 | 7.18 | 51.95 | 0 | BF16 | foreach |  |
| `train-fp8-b16g12-bf16experts-noenv-srcpath-te216main.log` | 27,969 | 27,969 | 2 | 7.03 | 52.23 | 0 | BF16 | loop |  |
| `nsys-fp8-b16g12-bf16experts-src-te216.log` | 27,860 | 27,860 | 3 | 7.06 | 52.03 | 0 | BF16 | loop |  |
| `train-fp8-b17g12-bf16experts-adamwloop-src-te216main.log` | 19,606 | 19,606 | 1 | 10.65 | 36.61 | 0 | BF16 | loop | OOM |
| `train-fp8-b17g12-bf16experts-adamwloop.log` | 19,390 | 19,390 | 1 | 10.77 | 36.21 | 0 | BF16 | loop | OOM |
| `train-fp8-b16g12-bf16experts-no-pad.log` | 19,093 | 19,093 | 1 | 10.30 | 35.66 | 0 | BF16 | foreach | OOM |
| `train-fp8-b15g13-bf16experts.log` | 14,655 | 14,571 | 5 | 13.70 | 27.21 | 247,808 | BF16 | foreach |  |
| `train-fp8-b14g14-bf16experts.log` | 14,640 | 14,635 | 6 | 13.71 | 27.33 | 266,880 | BF16 | foreach |  |
| `train-fp8-b15g13-bf16experts-no-gqa.log` | 14,596 | 14,596 | 3 | 13.68 | 27.26 | 248,384 | BF16 | foreach |  |
| `train-fp8-b15g13-bf16experts-teattn.log` | 14,546 | 14,546 | 4 | 13.73 | 27.17 | 248,000 | BF16 | foreach |  |
| `train-fp8-b16g12-bf16experts-adamwloop-te216main.log` | 14,433 | 14,433 | 5 | 13.62 | 26.95 | 229,440 | BF16 | loop |  |
| `train-fp8-b16g12-allfp8-unstable-sort.log` | 14,326 | 14,326 | 4 | 13.72 | 26.75 | 230,336 | FP8 | foreach |  |
| `train-fp8-b16g12-allfp8-pad16.log` | 14,283 | 14,283 | 4 | 13.77 | 26.67 | 54,704 | FP8 | foreach |  |
| `train-fp8-b16g12-allfp8.log` | 13,945 | 13,945 | 6 | 14.10 | 26.04 | 231,744 | FP8 | foreach |  |
| `train-fp8-b16g12-bf16experts.log` | 11,566 | 11,566 | 1 | 17.00 | 21.60 | 230,144 | BF16 | foreach | OOM |
| `train-fp8-b18g11-allfp8.log` | 11,237 | 11,237 | 1 | 18.04 | 20.99 | 210,944 | FP8 | foreach | OOM |
| `train-fp8-b1g1-contract.log` | 2,074 | 2,074 | 2 | 0.49 | 3.87 | 17,472 | FP8 | foreach |  |
| `train-fp8-b16g12-allfp8-pad1.log` |  |  | 0 |  |  |  | FP8 | foreach | TE FP8 GEMM `lda % 16` |
| `train-fp8-b17g12-bf16experts-inplace-weight.log` |  |  | 0 |  |  |  | BF16 | loop | In-place view from TE GroupedLinear backward |
| `train-fp8-b18g11-bf16experts-adamwloop.log` |  |  | 0 |  |  |  | BF16 | loop | OOM |

### 11.1 What worked

- FP8 mode with BF16 master weights.
- BF16 experts under FP8 global training.
- Batch 16.
- Grad accum 12, 16, or 24. Throughput is roughly the same; larger grad accum just makes fewer optimizer steps per token.
- AdamW loop implementation in the source/TE216main family.
- No MoE padding for BF16 experts (`pad_tok 0`).
- Liger fused linear CE.
- SDPA with native GQA.
- Prefetch depth 4.

### 11.2 What failed or underperformed

- All-FP8 experts are slower at the tested settings.
- All-FP8 with `fp8_pad_multiple=1` fails TE grouped GEMM alignment:

```text
Leading dimension requirement on A for FP8 GEMM. Caller must pad.
```

- Batch 17 and 18 hit OOM on BF16-expert FP8 path.
- Batch 18 all-FP8 also OOM after a slow first step.
- A test variant using in-place weighting after TE GroupedLinear failed:

```text
Output 0 of _GroupedLinearBackward is a view and is being modified inplace.
```

- Earlier padded runs around 14.5k tok/s were much slower and carried hundreds of thousands of padding tokens per logging interval.

## 12. Kernel Smoke Results

File: `/opt/dlami/nvme/metis/logs/kernel-smoke-fp8-bf16-te216main.log`

Environment:

```text
torch 2.12.0a0+0291f960b6.nv26.04.48445190
cuda 13.2
gpu NVIDIA RTX PRO 6000 Blackwell Server Edition
capability (12, 0)
has NVFP4 True
has MXFP8 True
```

Selected FP8 recipe smoke results:

| Shape test | Shape | Avg ms | Approx TFLOP/s | Peak GiB |
| --- | --- | ---: | ---: | ---: |
| qkv | `[24576,1536] x [1536,2560]` | 3.66 | 158.27 | 1.56 |
| attn_o | `[24576,1536] x [1536,1536]` | 2.23 | 155.82 | 1.19 |
| routed_down | `[98304,384] x [384,384]` | 1.79 | 48.45 | 1.13 |
| expert_gate_up | `[12288,384] x [384,2560]` | 1.36 | 53.24 | 0.90 |
| expert_down | `[12288,1280] x [1280,384]` | 0.80 | 45.57 | 0.25 |
| shared_gate_up | `[98304,384] x [384,2560]` | 11.76 | 49.30 | 5.77 |
| shared_down | `[98304,1280] x [1280,384]` | 2.53 | 114.75 | 1.77 |
| lm_head | `[24576,1536] x [1536,32768]` | 43.71 | 169.79 | 18.32 |
| grouped_moe_experts | `32 x [12288,384] -> [12288,384]` | 30.57 | 113.79 | 8.96 |

Selected BF16 recipe smoke results:

| Shape test | Avg ms | Approx TFLOP/s | Notes |
| --- | ---: | ---: | --- |
| qkv | 4.05 | 143.03 | FP8 modestly faster |
| attn_o | 2.44 | 142.83 | FP8 modestly faster |
| expert_gate_up | 1.36 | 53.12 | Same as FP8 |
| expert_down | 0.40 | 90.99 | BF16 faster than FP8 in this smoke |
| shared_gate_up | 11.36 | 51.04 | BF16 slightly faster |
| shared_down | 2.35 | 123.46 | BF16 faster |
| lm_head | 49.50 | 149.94 | FP8 faster |
| grouped_moe_experts | 29.47 | 118.04 | BF16 faster |

Liger fused linear CE smoke:

```text
ok loss=162.319672 avg_ms=28.07 peak_gib=2.13
```

Interpretation:

- The isolated shape smoke does not show all-FP8 experts as obviously faster.
- Several expert shapes are faster in BF16 than FP8 on this setup.
- The full training result agrees: BF16 experts under FP8 global mode are currently faster than all-FP8 experts.
- The grouped expert smoke throughput is far below ideal dense FP8 peak, suggesting small/fragmented grouped GEMM inefficiency.

## 13. NSYS Profile Findings

Profile:

```text
/opt/dlami/nvme/metis/profiles/metis15_fp8_b16g12_bf16experts_src_te216.nsys-rep
/opt/dlami/nvme/metis/profiles/metis15_fp8_b16g12_bf16experts_src_te216.sqlite
/opt/dlami/nvme/metis/profiles/stats/metis15_fp8_b16g12_bf16experts_src_te216_*.csv
```

This profile corresponds to the b16/g12 BF16-expert FP8 path, around 27.9k tok/s.

### 13.1 CUDA API summary

Top API rows:

| API | Time % | Calls | Notes |
| --- | ---: | ---: | --- |
| `cudaLaunchKernel` | 26.6% | 248,586 | huge launch count |
| `cudaMemcpyAsync` | 20.9% | 5,427 | includes internal transfers/copies, not necessarily data loader |
| `cudaStreamSynchronize` | 17.3% | 4,094 | sync overhead is visible |
| `cuLaunchKernelEx` | 12.3% | 110,166 | huge launch count |
| `cudaLaunchKernelExC` | 10.5% | 52,080 | huge launch count |
| `cuKernelSetAttribute` | 3.2% | 109,518 | likely dynamic/kernel setup overhead |
| `cudaEventRecord` | 2.7% | 20,520 | event overhead |
| `cuLibraryLoadData` | 2.5% | 149 | JIT/library loading overhead in short run |

Interpretation:

- The training step is launch/API dominated.
- This is exactly where CUDA graphs, fewer/fused kernels, persistent buffers, fewer dynamic per-call decisions, and less fragmented TE/cuBLAS dispatch could matter.

### 13.2 GPU kernel summary

Top kernel rows:

| Time % | Instances | Kernel family | Likely source |
| ---: | ---: | --- | --- |
| 12.2% | 9,576 | BF16 elementwise mul | SwiGLU / weighted operations |
| 7.1% | 22,497 | `nvjet_sm120_tst_mma_128x176x64...` | TE/cuBLAS FP8/BF16 GEMM |
| 6.4% | 1,368 | `CatArrayBatchedCopy_vectorized` | cat/copy from MLP or grouped path |
| 5.9% | 2,736 | vectorized elementwise mul | SwiGLU / weights |
| 4.8% | 48,888 | cublasLt splitK reduce | GEMM overhead |
| 4.8% | 1,368 | SiLU backward | expert/shared MLP |
| 4.3% | 21,950 | `nvjet_sm120_tst_mma_128x128x64...` | TE/cuBLAS GEMM |
| 3.7% | 12,618 | CUTLASS BF16 GEMM relu kernel | TE grouped/BF16 path |
| 3.5% | 1,368 | SiLU forward | expert/shared MLP |
| 3.1% | 1,368 | `indexFuncLargeIndex ReduceAdd` | MoE unpermute / `index_add_` |
| 2.6% | 57,657 | vectorized add | residuals / accumulation |

Interpretation:

- SwiGLU is visibly unfused.
- `index_add_` scatter/unpermute is visible.
- Cat/copy kernels are visible.
- There are many small GEMM and splitK/reduction kernels.

### 13.3 NVTX summary

Top NVTX rows:

| Range | Time % | Instances | Notes |
| --- | ---: | ---: | --- |
| `:backward` | 21.8% | 36 | 36 microbatches in profile |
| `:forward` | 18.8% | 36 | 36 microbatches in profile |
| `:nvte_multi_tensor_gemm` | 10.6% | 4,104 | TE grouped/multi-tensor GEMM |
| `:nvte_cublas_gemm_v2` | 10.1% | 137,484 | enormous count |
| `:moe_routed_grouped` | 8.8% | 684 | 19 layers * 36 microbatches |
| `cuBLAS:cublasLtMatmul` | 7.9% | 137,484 | same enormous GEMM count |
| `:moe_grouped_sort` | 5.4% | 684 | sorting assignments per layer/microbatch |
| `:fused_linear_ce` | 4.6% | 36 | Liger CE |
| `:moe_grouped_experts` | 3.1% | 684 | grouped expert compute |
| `:nvte_quantize_v2` | 2.4% | 4,275 | quantization overhead |
| `cuBLAS:cublasLtMatmulAlgoGetHeuristic` | 1.7% | 137,484 | heuristic overhead |
| `:optimizer_step` | 1.5% | 3 | optimizer not dominant |
| `:optimizer_muon` | 0.8% | 3 | not dominant |
| `:optimizer_adamw` | 0.7% | 6 | not dominant |
| `:moe_shared_experts` | 0.3% | 684 | surprisingly small as NVTX range |
| `:batch_fetch` | about 0.0% | - | data not bottleneck |

Interpretation:

- 137,484 cublasLt matmul ranges across 36 microbatches means about 3,819 cublasLt matmul calls per microbatch.
- That is far too fragmented for the throughput target.
- `moe_grouped_sort` alone costs more than optimizer time.
- The model likely needs a fused/persistent MoE implementation, not just launch flag tuning.

## 14. Current Bottleneck Hypothesis

Primary bottlenecks:

1. Too many launches and tiny/cuBLAS calls.
2. TE GroupedLinear is not delivering enough effective throughput for the MoE expert shapes.
3. The MoE path sorts/gathers/scatters every layer and every microbatch.
4. The expert SwiGLU activation is unfused and shows up heavily in GPU kernels.
5. All-FP8 experts need padding and are slower or unstable, so the current fastest path leaves the biggest parameter block in BF16.
6. Batch size is memory-capped at 16 for the current BF16-expert FP8 path, and batch 17/18 OOM.

Secondary bottlenecks:

1. Liger fused CE still shows up at 4.6% NVTX time. It is probably better than unfused CE, but not free.
2. Attention is not first-order based on current profile, but should be revisited once MoE is improved.
3. Optimizer is not first-order in time but may be first-order in memory.
4. CPT Dynamic MoR is not active yet. Its current pack/scatter implementation is likely to introduce new overhead later.

## 15. High-Value Research Questions

The research model should answer these in the context of RTX PRO 6000 Blackwell Server Edition, SM120, CUDA 13.2, PyTorch NGC 26.04, Transformer Engine 2.16 dev, and a single GPU.

### 15.1 MoE dispatch/kernel questions

1. What is the fastest known implementation strategy for top-k sparse MoE on a single RTX PRO 6000 / SM120 with this shape?
   - token-head rows per layer at b16: 65,536
   - assignments per layer: 262,144
   - experts: 32
   - average rows per expert: 8,192
   - expert gate/up: 384 -> 2560
   - expert down: 1280 -> 384
2. Should we replace the current `torch.argsort + index_select + TE GroupedLinear + index_add_` path with:
   - custom Triton kernels
   - CUTLASS grouped GEMM
   - MegaBlocks-style block sparse MoE
   - Tutel-style dispatcher
   - TE ops APIs beyond current `GroupedLinear`
   - a custom fused CUDA extension
3. Can routing sort/gather/unpermute be fused with grouped GEMM or implemented as a block-sparse layout to avoid `argsort` and `index_add_`?
4. Can we use fixed capacity per expert to get static shapes and unlock CUDA graphs without too much padding?
5. What capacity/padding multiple is optimal for SM120 FP8 grouped GEMM for this exact shape?
6. Is it better to group by expert with token-head rows, or restructure around MoE heads to improve locality and reduce scatter?
7. Can top-k weighted accumulation be fused into expert down output or unpermute kernel?
8. Can the aux-loss-free balance-bias update be moved off critical path or fused/approximated?

### 15.2 FP8 expert questions

1. Why are all-FP8 experts slower than BF16 experts in both smoke tests and training?
2. Is TE GroupedLinear using the optimal Blackwell kernels for grouped FP8 GEMM?
3. Is the `lda % 16` requirement the only hard alignment requirement, or do we need `m`, `n`, `k`, and split alignment choices that better match Blackwell Tensor Cores?
4. Would E4M3-only, HYBRID, or different `amax_history_len`/margin settings reduce quantization overhead or improve speed?
5. Is per-expert FP8 quantization overhead dominating because expert GEMMs are too fragmented?
6. Would BF16 experts plus FP8 attention/lm/head projections remain the fastest practical path?
7. If all-FP8 experts are required for memory, can custom grouped GEMM avoid TE's slow path?

### 15.3 Launch and graph questions

1. Can a whole train microbatch or multi-microbatch region be CUDA graph captured?
2. What dynamic pieces block graph capture?
   - top-k indices
   - tokens per expert
   - `m_splits` list
   - TE amax updates
   - Liger CE
   - optimizer
3. Would fixed expert capacity make MoE graphable?
4. Can cublasLt heuristics be cached or preselected to avoid 137,484 heuristic/matmul invocations in the profile?
5. Can persistent buffers remove allocation/copy overhead from gather/padding/unpermute?
6. Are there TE environment flags for reducing cublasLt heuristic overhead or using better SM120 kernels?

### 15.4 Fused MLP/SwiGLU questions

1. Can expert gate/up, SiLU, multiply, and down be fused for this grouped expert shape?
2. Can Liger, TE, or Triton provide a fused SwiGLU MLP for grouped MoE experts?
3. Can shared expert MLP be switched to a fused MLP path?
4. Is the high BF16 elementwise mul/Silu profile caused by expert MLP, shared MLP, or both?

### 15.5 Optimizer/memory questions

1. Can routed expert weights safely move from AdamW to Muon to reduce optimizer state memory?
2. If yes, can that permit batch 17/18 and improve throughput?
3. Would BF16 optimizer states or 8-bit optimizer states be safe enough for this experimental run?
4. Is the OOM boundary mostly optimizer states, activations, TE workspaces, Liger CE, or grouped expert buffers?
5. Should checkpointing/activation recompute be used to allow larger batch, or would it reduce tok/s too much?

### 15.6 Attention questions

1. Is current SDPA/native GQA the right Blackwell path?
2. Should FA3 or TE DotProductAttention be installed/enabled for SM120 and retested?
3. Can attention QKV/O projections and attention kernels be graph captured or fused better?
4. After MoE dispatch is fixed, does attention become a top bottleneck?

### 15.7 Dynamic MoR questions for CPT

Base pretraining does not use MoR, but continued pretraining will.

Current Dynamic MoR pack/scatter code:

- loops over batch rows with Python-level list building
- uses `torch.nonzero`
- pads each batch row to max active length
- uses `torch.stack`
- scatters back with `scatter_add`

Research questions:

1. Will Dynamic MoR destroy throughput if left as-is?
2. Should active-token packing/scattering be rewritten as custom kernels before CPT?
3. Can MoR be run in dense-active mode until the base throughput issue is solved?
4. Can routing and MoR token packing share infrastructure with MoE dispatch?

## 16. Recommended Experiment Queue

### Experiment 0: make the FP8 target canonical

Before long training:

- Change manifest `low_precision_mode` and hardware precision text to unambiguous FP8.
- Change `scripts/metis15_full.sh` default away from NVFP4.
- Change Makefile main RTX targets away from NVFP4.
- Keep NVFP4/MXFP8 only as explicitly named legacy experiments.

Reason: too easy to launch the wrong recipe.

### Experiment 1: reproduce the current best for longer

Run a 500 to 1000 step baseline with:

- `METIS15_FP8=1`
- `METIS15_NVFP4=0`
- batch 16
- grad accum 12 and/or 24
- `--fp8-expert-precision bf16`
- `--hybrid-adamw-impl loop`
- Liger fused CE
- SDPA/native GQA
- prefetch 4

Collect:

- stable tok/s after warmup
- peak memory
- per-step memory trend
- expert load balance histograms
- full NSYS after warmup, not only first few steps
- torch profiler if not too intrusive

### Experiment 2: isolate padding vs AdamW loop vs source path

The jump from about 14.5k to about 28k appears correlated with zero MoE padding and/or source TE216main loop variants. Confirm the causal factor.

Matrix:

- BF16 experts, AdamW loop, pad 0
- BF16 experts, AdamW foreach, pad 0
- BF16 experts, AdamW loop, forced old padding
- all-FP8 experts, pad 16/32/64
- capacity factor fixed 1.05/1.10 with static splits

Goal: know exactly whether the speedup came from eliminating padding, AdamW loop, image/runtime, or a code path.

### Experiment 3: custom/fused MoE dispatch prototype

This is likely the highest upside.

Prototype objective:

- Replace per-layer `argsort/index_select/TE GroupedLinear/index_add_` with a fused or lower-launch MoE path.
- Keep model semantics identical:
  - 32 experts
  - top-4
  - 4 token-head routing heads
  - shared expert
  - sigmoid router scores
  - aux-loss-free balance bias

Candidate implementation directions:

- Triton fused dispatcher plus grouped GEMM.
- CUTLASS grouped GEMM wrapper with persistent split buffers.
- MegaBlocks-style block sparse MoE.
- TE lower-level grouped GEMM APIs if they expose better plan reuse.
- Custom CUDA extension for pack/unpack/weighted scatter, leaving GEMMs to cuBLASLt/CUTLASS.

Minimum win target:

- reduce `moe_grouped_sort` from 5.4% NVTX to near-zero
- reduce `indexFuncLargeIndex ReduceAdd`
- reduce cublasLt matmul call count
- reduce launch count

### Experiment 4: fused expert SwiGLU

Current expert path does:

```python
gate, up = gate_up.split(intermediate_size, dim=-1)
down = down_proj(F.silu(gate) * up)
```

Profile shows SiLU and multiply are very visible.

Try:

- Liger fused SwiGLU where compatible
- TE fused MLP for shared expert
- Triton fused SwiGLU for grouped expert output
- combine activation and top-k weight multiply before scatter

### Experiment 5: make FP8 experts actually competitive

Because BF16 experts are currently faster, do not assume all-FP8 is a win.

Investigate:

- FP8 grouped expert alignment requirements
- optimal `fp8_pad_multiple`
- fixed capacity vs dynamic splits
- TE grouped GEMM internals on SM120
- HYBRID vs E4M3
- amax history and quantization overhead
- whether custom grouped GEMM can avoid TE's current slow/fragile path

### Experiment 6: graph capture once shapes are static

Try CUDA graph capture only after reducing dynamic shape churn.

Likely prerequisites:

- fixed capacity per expert or block-sparse layout
- persistent m_split buffers
- no Python list of dynamic splits on critical path
- stable TE autocast behavior
- fixed Liger CE shape

Goal:

- reduce launch/API overhead, which is currently enormous.

### Experiment 7: optimizer memory ablations

Try carefully:

- `--muon-include-routed-experts`
- optimizer state dtype changes if supported
- reducing AdamW state memory for routed experts
- memory snapshots before/after first step

Success condition:

- batch 17/18 becomes stable without losing throughput or training stability.

Warning:

- Optimizer time is not the leading profile bottleneck, so this is mainly a memory unlock experiment.

### Experiment 8: attention after MoE fixes

Only after MoE/launch issues are improved:

- test TE DotProductAttention on current NGC/TE main
- test FA3 if a Blackwell-compatible build is available
- compare SDPA native GQA vs alternatives with a fixed MoE implementation

## 17. Concrete Code/Config Gaps Found

1. Manifest is stale/confusing:
   - `low_precision_mode` still says `nvfp4`
   - `preferred_precision` still says `nvfp4_sm120_safe_with_fp8_block_surfaces`
   - hardware FP8 enabled and NVFP4 disabled are contradictory with model-level low precision

2. Full pipeline defaults are stale:
   - `scripts/metis15_full.sh` defaults `METIS15_NVFP4=1`
   - Makefile full and continued-pretrain RTX targets still force NVFP4

3. Benchmark defaults are stale:
   - `scripts/metis15_rtx_benchmark_matrix.sh` defaults `LOW_PRECISION=nvfp4`

4. MoE dispatch is PyTorch-op heavy:
   - `bincount`
   - `argsort`
   - `index_select`
   - optional Python-loop padding
   - TE GroupedLinear
   - `index_add_`

5. Expert MLP activation is unfused:
   - SiLU forward/backward and multiply are high in profile

6. All-FP8 experts are currently not a valid fast path:
   - require padding
   - slower than BF16 experts
   - can hit TE alignment failure
   - can OOM at larger batch

7. Dynamic MoR pack/scatter path is likely too Python/list/op-heavy for CPT:
   - not currently active in base pretraining, but needs attention before continued pretraining throughput work

8. No local hand-written CUDA/Triton kernels were found for this path:
   - current stack is PyTorch + Transformer Engine + Liger + SDPA
   - custom kernel work is a real opportunity, not a small tweak

## 18. Files Inspected

Core architecture/config:

- `configs/metis15_manifest.json`
- `docs/metis15_multihead_latent_moe.md`
- `src/metis_mamba/config.py`
- `src/metis_mamba/model.py`

Precision/runtime:

- `src/metis_mamba/fp8.py`
- `scripts/train_mamba_lm.py`
- `docker/runpod-metis-gpu/Dockerfile.ngc-blackwell`
- `docker/runpod-metis-gpu/README.md`
- `requirements-gpu-train.txt`

Launchers/benchmarks:

- `scripts/metis15_pretrain.sh`
- `scripts/metis15_full.sh`
- `scripts/metis15_rtx_benchmark_matrix.sh`
- `Makefile`

Optimizer:

- `src/metis_mamba/optim.py`

Live evidence:

- `/opt/dlami/nvme/metis/logs/*.log`
- `/opt/dlami/nvme/metis/profiles/stats/*.csv`
- `/opt/dlami/nvme/metis/data/metis15_base/meta.json`

## 19. Suggested Prompt to the Research Model

Use this packet as ground truth. We need to maximize single-GPU training throughput for Metis-1.5 on NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120, CUDA 13.2, PyTorch NGC 26.04, Transformer Engine 2.16 dev.

Current model: ~1.096B total params, ~311M active params, 19 layers, d_model 1536, block 1024, 24 Q heads, 8 KV heads, multi-head LatentMoE with 4 MoE heads, 32 routed experts, top-4, 1 shared expert, expert hidden 1280, router latent 128. Base pretraining uses static dense mode, not Dynamic MoR. Continued pretraining later adds Dynamic MoR.

Current best RTX PRO 6000 FP8 live run is only ~28.7k tok/s, about 53 active TFLOP/s. Target is at least 170k tok/s, ideally 200k+ tok/s. H100 Metis-1.4 reference was ~170k tok/s. Custom CUDA/Triton/CUTLASS kernels are allowed if they preserve model semantics and improve throughput.

Find the most likely optimization path. Prioritize changes that can plausibly produce multi-x gains, especially MoE dispatch, grouped GEMM, fused SwiGLU, CUDA graphing, and FP8 expert implementation. Be explicit about what to try, what code to replace, what libraries/kernels to use, what measurements to collect, and what risks exist for training stability.

## 20. Bottom Line

The current system is not close to the desired throughput. It is not a matter of one flag. The evidence points to structural inefficiency in the sparse MoE implementation on this hardware/runtime:

- too many launches
- too many tiny/fragmented cublasLt/TE calls
- dynamic per-layer MoE sort/gather/scatter
- unfused expert SwiGLU
- all-FP8 experts currently slower/fragile
- stale NVFP4 defaults that should be removed from the main path

The highest-upside work is to build or adopt a real high-performance MoE kernel path for this exact shape and SM120 runtime, then use static capacities/persistent buffers/CUDA graphs to reduce launch overhead. Only after that should attention and smaller launch flags be the main focus.
