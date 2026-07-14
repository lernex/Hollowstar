# Metis-1.5 RTX PRO 6000 Blackwell Testing Handoff

Date: 2026-05-14

Purpose: give a research model a complete, codebase-specific packet for diagnosing why the current Metis-1.5 RTX PRO 6000 Blackwell training stack is only reaching about 15.4k tokens/sec and about 88 dense-equivalent TFLOP/s, despite the intended NVFP4/MXFP8/FP8-block low-precision training design.

This file is deliberately blunt. The current stack does run real-data next-token pretraining, but it is not close to the target performance. The most important unresolved issue is that RTX PRO 6000 is SM120, and current Transformer Engine behavior on SM120 is not equivalent to a clean B100/B200-style low-precision path. NVFP4 is exposed but the default recipe fails. MXFP8 is exposed in the final image after a guard patch, but exact Metis training GEMMs still fail. The current viable path is a reduced NVFP4 recipe plus Float8BlockScaling fallback surfaces, and that path is slow.

Post-handoff architecture update: after this packet was written, `configs/metis15_manifest.json` was changed from `moe_routed_latent_size=320` to `384` so the routed expert path is 128-aligned. Because the routed dim now equals the 384d MoE head dim, the model skips the routed down/up projection pair. New code-derived estimates are `1,095,725,952` total params, `311,260,032` active params, and `260,868,480` active transformer params. The benchmark results and failure analysis below are still the observed `320`-latent RTX PRO 6000 baseline unless a section explicitly says otherwise.

## Executive Summary

Instance and runtime:

- AWS instance public IPv4: `98.94.18.6`
- Instance ID: `i-0da4154f5638861cc`
- Instance ARN: `arn:aws:ec2:us-east-1:151025633969:instance/i-0da4154f5638861cc`
- IAM role: `lernex-p5-ssm-role`
- User-selected AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.11 (Ubuntu 24.04)`
- Hostname observed over SSH: `ip-172-31-91-14`
- Kernel: `Linux 6.17.0-1013-aws #13~24.04.1-Ubuntu SMP Fri Apr 24 21:50:45 UTC 2026 x86_64`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Compute capability from PyTorch: `(12, 0)`
- VRAM from `nvidia-smi`: `97,887 MiB`
- Driver: `595.58.03`
- Host CUDA version shown by `nvidia-smi`: `13.2`
- NVMe data mount: `/opt/dlami/nvme`, about 1.7 TB, about 1.5 TB free after data/image work

Final Docker image currently used:

- Image: `metis15-blackwell-ngc2604-te-main:sm120a`
- Base: `nvcr.io/nvidia/pytorch:26.04-py3`
- PyTorch: `2.12.0a0+0291f960b6.nv26.04.48445190`
- CUDA runtime: `13.2`
- Transformer Engine: `2.16.0.dev0+76c2a9e`
- TE recipe exposure:
  - `NVFP4BlockScaling`: present
  - `MXFP8BlockScaling`: present
  - `Float8BlockScaling`: present
- `aws-cli`: `1.45.7`, Python `3.12.3`, botocore `1.43.7`
- Build arch env:
  - `TORCH_CUDA_ARCH_LIST=12.0a`
  - `NVTE_CUDA_ARCHS=120a`

Real data:

- Local data path on instance: `/opt/dlami/nvme/metis/data/metis15_base`
- Files:
  - `train.bin`: about 94 GiB
  - `val.bin`: about 964 MiB
  - `meta.json`: about 91 KiB
- Metadata verified previously:
  - `train_tokens`: `50,000,000,000`
  - `val_tokens`: `505,050,506`
  - `vocab_size`: `32768`
  - `dtype`: `uint16`
- File-size-derived byte counts:
  - `train.bin`: `100,000,000,000` bytes expected from 50B uint16 tokens, shown by `ls -lh` as about `94G`
  - `val.bin`: `1,010,101,012` bytes expected from 505,050,506 uint16 tokens, shown by `ls -lh` as about `964M`
- Important correction: the live `meta.json` does not contain `train_bytes` or `val_bytes` fields. Those byte counts are inferred from token count times uint16 width and from file size.

Current stable real-data setting:

- Batch: `18`
- Gradient accumulation: `11`
- Sequence length: `1024`
- Tokens per optimizer step: `202,752`
- Estimated steps for 50B tokens: `246,607`
- Precision: reduced NVFP4 plus FP8-block surfaces, not true all-surface NVFP4/MXFP8
- Attention: SDPA native GQA
- LM loss: Liger fused linear CE
- Optimizer: Muon-AdamW hybrid with foreach AdamW path
- Data: pinned async CUDA batch prefetch depth 4
- MoE dispatch: grouped dispatch, capacity factor 0

Best observed stable real-data throughput so far:

- About `15.4k tok/s`
- About `87.8` dense-equivalent total-param TFLOP/s
- About `27.5` active-param TFLOP/s

User expectation / target framing:

- This is not considered remotely acceptable.
- Prior Metis-1.4 reference: dense 500M-ish FP8 H100 training reached about `170k tok/s`.
- The RTX PRO 6000's advertised dense FP4 tensor compute is expected to be in the same broad class as H100 dense FP8 tensor compute.
- Metis-1.5 has about `297M` active params including embedding/final norm and about `246.9M` active transformer params, much less active compute than the Metis-1.4 dense 500M reference.
- User expectation: minimum target around `420k tok/s`, not `15k tok/s`.
- Current `15.4k tok/s` is about `27x` below the `420k tok/s` target.
- Therefore the research question is not "can we squeeze 10 percent more out of the current path"; it is "what is catastrophically preventing RTX PRO 6000 Blackwell from running this workload like a real low-precision sparse model."

ETA if kept as-is:

- `50,000,000,000 / 15,396 tok/s = 37.59 days` of pure training time
- With eval/checkpoint/upload/interrupt overhead, a realistic full run would likely be about `39-43 days`
- This is not acceptable relative to the intended RTX PRO 6000 target

450 TFLOP/s token-rate targets printed by the launcher:

- Total-param dense-equivalent target: about `78,867 tok/s`
- Active-param target: about `252,311 tok/s`
- Active-transformer-only target from the model math: about `303,816 tok/s`

Additional target math:

- Metis-1.4-style dense reference math: `6 * 500M * 170k tok/s = 510 TFLOP/s`.
- If Metis-1.5 achieved the same useful active TFLOP/s on `297,251,712` active params, expected token rate would be about `286k tok/s`.
- If using active-transformer-only params `246,860,160`, the same `510 TFLOP/s` implies about `344k tok/s`.
- The user's `420k tok/s` target implies:
  - about `749 TFLOP/s` on the `297,251,712` active-param basis
  - about `622 TFLOP/s` on the `246,860,160` active-transformer basis
  - about `2.4 PFLOP/s` on the misleading total-param dense-equivalent basis
- Hitting `420k tok/s` requires the MoE routing/dispatch/experts/loss/optimizer stack to behave like a real sparse low-precision stack, not like dense compute surrounded by sort/gather/scatter/kernel fallback overhead.

## The Important Low-Precision Finding

The problem is not simply that the AMI's packaged PyTorch is too old. The final image bypasses the AMI framework stack by using an NVIDIA 26.04 PyTorch container with PyTorch 2.12/CUDA 13.2 and Transformer Engine main. The host driver is also new enough to expose CUDA 13.2.

The problem is that Transformer Engine's practical SM120 support for the exact Metis low-precision training shapes appears incomplete:

1. Default NVFP4 recipe is exposed but fails exact Metis GEMMs on SM120.
2. Reduced NVFP4 recipe works only when disabling RHT, 2D quantization, and stochastic rounding.
3. MXFP8 recipe is exposed in the final patched TE image, but exact training GEMMs still fail during backward with cuBLASLt unsupported errors.
4. Float8BlockScaling works and is currently used as the practical substitute for the originally intended MXFP8 surfaces.
5. Grouped MoE experts under NVFP4 emit an unfused quantization fallback warning because some input inner dimensions, especially `320`, are not multiples of `128`.

This means the current code path is not full BF16, but it is also not the intended clean NVFP4/MXFP8 Blackwell path. It is a compromise:

```text
embeddings: BF16
lm_head: BF16
qkv projection: Float8BlockScaling, not MXFP8
latent MoE projections: Float8BlockScaling, not MXFP8
routed/shared experts: reduced NVFP4 for most blocks
final 2 expert blocks: Float8BlockScaling
attention matmul/softmax: BF16 SDPA
```

The manifest names this preferred precision as:

```text
nvfp4_sm120_safe_with_fp8_block_surfaces
```

## Source Files Inspected

Primary repo files:

- `configs/metis15_manifest.json`
- `scripts/metis15_pretrain.sh`
- `scripts/metis15_full.sh`
- `scripts/smoke_metis15_blackwell_kernels.py`
- `scripts/train_mamba_lm.py`
- `scripts/metis15_rtx_benchmark_matrix.sh`
- `src/metis_mamba/config.py`
- `src/metis_mamba/model.py`
- `src/metis_mamba/fp8.py`
- `src/metis_mamba/optim.py`
- `docker/runpod-metis-gpu/Dockerfile.ngc-blackwell`
- `Makefile`
- Earlier packet: `docs/metis15_rtx_pro_6000_nvfp4_research_packet_2026-05-13.md`

Remote instance probes:

- `nvidia-smi`
- `docker images`
- image-local Python imports for torch, Transformer Engine, recipe availability, and awscli
- remote data path/file-size checks
- previous interactive benchmark and smoke-test outputs from this setup

Note: some benchmark outputs were observed in interactive terminal sessions and were not persisted as log files under `/opt/dlami/nvme/metis`. The numbers below were transcribed from those runs.

## Current Model Architecture

Canonical manifest: `configs/metis15_manifest.json`

Core model:

- Name: `Metis-1.5`
- Architecture: `metis_multihead_latent_moe_decoder`
- Model type: `metis_multihead_latent_moe`
- Vocab size: `32768`
- Sequence length: `1024`
- `d_model`: `1536`
- Layers: `19`
- Query heads: `24`
- KV heads: `8`
- Head dim: `64`
- Intermediate size field: `4096`
- Activation: `swiglu`
- Tied embeddings: true
- RMSNorm: true
- Attention bias: false
- MLP bias: false
- Attention dropout: `0.0`
- RoPE theta: `10000.0`
- Attention backend: `sdpa`
- Native GQA enabled unless disabled by CLI

MoE:

- FFN type: `multi_head_latent_moe`
- MoE feature heads: `4`
- Per-feature-head width: `1536 / 4 = 384`
- Routed experts: `32`
- Top-k: `4`
- Shared experts: `1`
- Expert intermediate size: `1280`
- Router latent size: `128`
- Routed latent size: `320`
- Router score: `sigmoid`
- Router temperature: `1.0`
- Balance strategy: `aux_loss_free_bias`
- Balance bias update rate: `0.001`
- Balance bias clamp: `5.0`
- Aux loss coef: `0.0`
- Dispatch mode: `grouped`
- Capacity factor: `0.0`
- Capacity alignment: `128`

Parameter counts from manifest/model construction:

- Estimated total params: `950,973,312`
- Estimated active params: `297,251,712`
- Estimated active transformer params: `246,860,160`

Training stage:

- Base pretrain mode: `static_dense_pretrain`
- Dynamic MoR disabled for base pretrain:
  - `mor_enabled=false`
  - `mor_train_router=false`
  - `mor_runtime_mode=disabled`
- Continued pretrain is separate and later enables Dynamic token MoR, but this packet is focused on base pretrain throughput.

## Current Precision Policy

Manifest low-precision fields:

```json
{
  "low_precision_mode": "nvfp4",
  "nvfp4_disable_rht": true,
  "nvfp4_disable_2d_quantization": true,
  "nvfp4_disable_stochastic_rounding": true,
  "nvfp4_keep_embeddings_bf16": true,
  "nvfp4_keep_qkv_bf16": false,
  "nvfp4_keep_latent_moe_projections_bf16": false,
  "nvfp4_keep_lm_head_bf16": true,
  "nvfp4_qkv_precision": "fp8_block",
  "nvfp4_latent_moe_projection_precision": "fp8_block",
  "nvfp4_lm_head_precision": "bf16",
  "nvfp4_final_expert_layers": 2,
  "nvfp4_final_expert_precision": "fp8_block",
  "fp8_pad_multiple": 64
}
```

Important code behavior in `src/metis_mamba/fp8.py`:

- `transformer_engine_runtime_supports_nvfp4(...)` returns true on SM120 only if all three reduced-recipe flags are set:
  - `disable_rht`
  - `disable_2d_quantization`
  - `disable_stochastic_rounding`
- The comment says TE exposes NVFP4 on RTX PRO 6000/SM120, but default production recipe kernels fail exact Metis GEMMs.
- `transformer_engine_runtime_supports_mxfp8()` returns false on capability >= `(12, 0)` because TE 2.15 raises:

```text
MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet
```

- `transformer_engine_runtime_supports_fp8_block_scaling()` is used as the practical block-scaled FP8 fallback check.

Important nuance:

- The final NGC/TE-main image patched the TE MXFP8 SM120 guard so the recipe reports available.
- Even after that guard patch, exact Metis MXFP8 GEMMs still fail at runtime.
- So the repo intentionally treats MXFP8 as unsupported on SM120 for Metis training until a real GEMM path is proven.

## Docker Image Chronology And Findings

### Original issue

The earlier Hopper-oriented image/build path was wrong for RTX PRO 6000:

- Hard-coded Hopper-ish arch assumptions existed in older Docker context.
- RTX PRO 6000 Blackwell reports compute capability `(12, 0)`, not SM90.
- The image path needed a real Blackwell rebuild.

### CUDA 12.8 / TE 2.15 image

Image/tag observed:

- `metis15-blackwell-sm120:local`

Runtime:

- PyTorch `2.8.0+cu128`
- CUDA `12.8`
- Transformer Engine `2.15`
- Arch targeted as `12.0`

Results:

- BF16 exact shapes passed.
- Default NVFP4 failed at the QKV shape with an invalid-argument CUDA error in a Hadamard/RHT-related kernel:

```text
row_cast_col_hadamard_transform_cast_fusion.cu:1200 CUDA Error: invalid argument
```

- MXFP8 rejected before useful dispatch with:

```text
MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet.
```

### CUDA 12.8 / TE 2.15 with `12.0a` / `120a`

Image/tag observed:

- `metis15-blackwell-sm120a:cu128-te215`

Build:

- `TORCH_CUDA_ARCH_LIST=12.0a`
- `NVTE_CUDA_ARCHS=120a`
- `cuobjdump` verified `sm_120a` cubins.

Results:

- Default NVFP4 still failed in the Hadamard/RHT path.
- NVFP4 passed exact shapes only with:
  - `--nvfp4-disable-rht`
  - `--nvfp4-disable-2d-quantization`
  - `--nvfp4-disable-stochastic-rounding`
- MXFP8 guard patch got past the high-level check, but backward still failed on QKV with cuBLAS unsupported status.

### NVIDIA NGC PyTorch 26.04 base

Base image:

- `nvcr.io/nvidia/pytorch:26.04-py3`

Runtime:

- PyTorch `2.12.0a0+0291f960b6.nv26.04`
- CUDA `13.2`
- Transformer Engine base version around `2.14.0+f031cf87`
- Capability `(12, 0)`

Results:

- TE's high-level MXFP8 guard was not active in the same way, or could be patched, but runtime MXFP8 still failed.
- Reduced NVFP4 QKV passed.

### Final image: NGC 26.04 plus TE main

Image/tag:

- `metis15-blackwell-ngc2604-te-main:sm120a`

Dockerfile:

- `docker/runpod-metis-gpu/Dockerfile.ngc-blackwell`

Key build details:

```dockerfile
FROM nvcr.io/nvidia/pytorch:26.04-py3

ARG TRANSFORMER_ENGINE_REF="main"
ARG PATCH_TE_SM120_MXFP8_GUARD=1

ENV TORCH_CUDA_ARCH_LIST=12.0a
ENV NVTE_CUDA_ARCHS=120a
```

The Dockerfile:

- Uninstalls pre-existing TE wheels.
- Installs `pybind11` and `nvidia-mathdx==25.6.0`.
- Clones Transformer Engine main.
- Patches the SM120 MXFP8 guard in `transformer_engine/pytorch/quantization.py`.
- Builds TE from source with PyTorch framework support.
- Installs the Metis repo editable.
- Runs import checks for torch, CUDA, TE, NVFP4, MXFP8, and Liger.
- Runs `train_mamba_lm.py --help`, `train_mamba_sft.py --help`, `train_mamba_reward.py --help`, and `train_mamba_dpo.py --help`.

Runtime probe in this image:

```text
torch 2.12.0a0+0291f960b6.nv26.04.48445190
cuda 13.2
cap (12, 0)
gpu NVIDIA RTX PRO 6000 Blackwell Server Edition
te 2.16.0.dev0+76c2a9e
has_nvfp4 True
has_mxfp8 True
has_float8_block True
aws-cli/1.45.7 Python/3.12.3 Linux/6.17.0-1013-aws botocore/1.43.7
```

Final low-precision result:

- Reduced NVFP4 works for exact Metis shapes.
- Float8BlockScaling works for exact Metis shapes.
- MXFP8 still fails for exact Metis training GEMMs even though it is exposed.
- Full/default NVFP4 recipe still not viable.

## Exact-Shape Kernel Smoke Test

Script:

- `scripts/smoke_metis15_blackwell_kernels.py`

Make target:

```bash
make metis15-blackwell-smoke
```

The Make target runs:

```bash
python scripts/smoke_metis15_blackwell_kernels.py \
  --recipes "fp8_block,nvfp4,bf16" \
  --nvfp4-disable-rht \
  --nvfp4-disable-2d-quantization \
  --nvfp4-disable-stochastic-rounding
```

Shapes tested:

```text
qkv:            [24576, 1536] x [1536, 2560]
attn_o:         [24576, 1536] x [1536, 1536]
routed_down:    [98304, 384]  x [384, 320]
expert_gate_up: [12288, 320]  x [320, 2560]
expert_down:    [12288, 1280] x [1280, 320]
shared_gate_up: [98304, 384]  x [384, 2560]
shared_down:    [98304, 1280] x [1280, 384]
lm_head:        [24576, 1536] x [1536, 32768]
grouped MoE:    32 experts, each about [12288, 320] -> [12288, 320]
```

Final NGC/TE-main image exact-shape smoke numbers:

| Recipe | Shape | Avg ms | Approx TFLOP/s | Notes |
|---|---:|---:|---:|---|
| fp8_block | qkv | 4.08 | 142.01 | works |
| fp8_block | attn_o | 2.36 | 147.49 | works |
| fp8_block | routed_down | 1.70 | 42.69 | works |
| fp8_block | expert_gate_up | 1.67 | 36.18 | works |
| fp8_block | expert_down | 0.80 | 37.81 | works |
| fp8_block | shared_gate_up | 11.73 | 49.43 | works |
| fp8_block | shared_down | 2.63 | 110.16 | works |
| fp8_block | lm_head | 44.11 | 168.25 | works |
| fp8_block | grouped_moe_experts | 31.89 | 90.90 | works |
| nvfp4 reduced | qkv | 3.53 | 164.30 | works |
| nvfp4 reduced | attn_o | 2.16 | 160.79 | works |
| nvfp4 reduced | routed_down | 1.73 | 41.93 | works |
| nvfp4 reduced | expert_gate_up | 1.64 | 36.74 | works |
| nvfp4 reduced | expert_down | 0.99 | 30.43 | works, slower than BF16 here |
| nvfp4 reduced | shared_gate_up | 11.67 | 49.69 | works |
| nvfp4 reduced | shared_down | 2.57 | 112.89 | works |
| nvfp4 reduced | lm_head | 39.60 | 187.40 | works |
| nvfp4 reduced | grouped_moe_experts | 31.06 | 93.35 | works, but warning |
| bf16 | qkv | 4.21 | 137.58 | works |
| bf16 | attn_o | 2.57 | 135.23 | works |
| bf16 | routed_down | 1.64 | 44.07 | works, faster than low precision here |
| bf16 | expert_gate_up | 1.49 | 40.57 | works, faster than low precision here |
| bf16 | expert_down | 0.54 | 55.49 | works, much faster than low precision here |
| bf16 | shared_gate_up | 11.51 | 50.37 | works |
| bf16 | shared_down | 2.48 | 116.94 | works |
| bf16 | lm_head | 49.77 | 149.12 | works |
| bf16 | grouped_moe_experts | 28.18 | 102.88 | works, faster than low precision grouped MoE |

Warning seen in the NVFP4 grouped MoE path:

```text
Unfused NVFP4 quantization fallback because input inner dim not multiple of 128, disabling NVFP4 grouped kernel fusion.
```

This warning is probably very important. The routed expert input dimension is `320`, and the expert down output dimension is `320`. Those are multiples of 64 but not 128. The current model dimensions are shape-friendly for many kernels but may be hostile to the specific fused NVFP4 grouped path TE wants on SM120.

Interpretation:

- Reduced NVFP4 is not clearly faster than BF16 for the most important expert shapes.
- Grouped MoE in BF16 measured faster than grouped MoE in reduced NVFP4/FP8-block in the smoke test.
- If these numbers are real and representative, the model is paying low-precision complexity without getting low-precision tensor-core speed on the expert path.
- The research model should investigate whether dimensions `320` and `1280`, TE grouped-linear layout, padding, or SM120 kernel selection are forcing bad kernels.

## MXFP8 Failure Summary

MXFP8 was a hard requirement in the intended precision map, especially for:

- QKV projection
- latent MoE router projection
- latent MoE routed down projection
- latent MoE routed up projection

Observed reality:

- TE 2.15 explicitly rejected MXFP8 on compute capability >= `(12, 0)` with a guard message.
- TE main still contained/contained-like logic warning that MXFP8 was not supported on 12.0+ for all GEMM layouts.
- We patched the guard in the final image to test whether it was only an overly conservative check.
- After patching, MXFP8 still failed in actual training-shape backward calls.

Representative failure:

```text
cublaslt_gemm.cu:771 CUBLAS_STATUS_NOT_SUPPORTED
```

Conclusion:

- Do not assume MXFP8 is usable merely because the RTX PRO 6000 is Blackwell or because `recipe.MXFP8BlockScaling` exists.
- The exact train forward/backward GEMMs need to pass.
- For the current repo, MXFP8 should be treated as not proven and not usable on SM120 until the failing cuBLASLt layout is fixed or bypassed.

## NVFP4 Failure Summary

Default NVFP4 recipe failure:

```text
row_cast_col_hadamard_transform_cast_fusion.cu:1200 CUDA Error: invalid argument
```

This was seen on exact Metis dimensions, especially the QKV-like surfaces, when using default `recipe.NVFP4BlockScaling()`.

Reduced NVFP4 recipe required:

```python
recipe.NVFP4BlockScaling(
    disable_rht=True,
    disable_2d_quantization=True,
    disable_stochastic_rounding=True,
)
```

Code has been changed so SM120 only passes the NVFP4 runtime gate if these three flags are active.

Risk:

- This reduced recipe is not the original quality/stability recipe. Disabling RHT, 2D quantization, and stochastic rounding may change convergence and quality.
- It also does not seem to deliver the expected speed on the grouped MoE/expert shapes.
- It may be a "works enough to run" path, not a "correct final training" path.

## Current Training Launcher

Script:

- `scripts/metis15_pretrain.sh`

Important defaults:

```bash
LM_LOSS_IMPL="${METIS15_LM_LOSS_IMPL:-liger_fused_linear_ce}"
PREFETCH_BATCHES="${METIS15_PREFETCH_BATCHES:-4}"
ENABLE_NVFP4="${METIS15_NVFP4:-1}"
MATMUL_PRECISION="${METIS15_MATMUL_PRECISION:-highest}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-1}"
```

The launcher passes:

```bash
--lm-loss-impl "$LM_LOSS_IMPL"
--prefetch-batches "$PREFETCH_BATCHES"
--fused-adamw
--matmul-precision "$MATMUL_PRECISION"
--nvfp4
--nvfp4-disable-rht
--nvfp4-disable-2d-quantization
--nvfp4-disable-stochastic-rounding
```

It also hydrates from S3 if `METIS15_S3_ROOT` or specific S3 env vars are set and local data/checkpoints are missing.

The current fused CE issue from the first research pass has been patched. The standalone pretrain launcher now defaults to Liger fused linear CE.

## Correctness: Next-Token Prediction

Previous Metis-1.4 mistake was training on an incorrect shifted target. The current Metis-1.5 path has explicit next-token checks.

Relevant behavior:

- `CudaBatchPrefetcher` creates `x` from the memmap window and sets `y = x.to(torch.long)`.
- It includes this code comment:

```python
# MetisMoRLMHeadModel shifts labels internally, so labels stay aligned.
# Keep CPU staging narrow and widen labels after the asynchronous copy.
```

- The LM model/loss path shifts internally rather than pre-shifting the memmap labels.
- Liger fused linear CE was tested against a manual next-token calculation.

Observed baked image loss-contract smoke output:

```text
standard_ce_next_token_ok loss=4.844723 two_token_loss=4.865763
liger_fused_linear_ce_next_token_ok loss=4.845042 manual=4.845043
```

Interpretation:

- Standard CE and Liger fused linear CE agree on the next-token objective.
- The two-token comparison differs, which is good: it catches the prior class of off-by-one/off-by-two error.
- Next-token correctness appears specifically verified in the image.

## Current Data Loader

Code:

- `scripts/train_mamba_lm.py`
- Class: `CudaBatchPrefetcher`

Current behavior:

- Reads from NumPy `memmap`.
- Randomly samples positions.
- Uses a precomputed `np.arange(block_size)` offset base.
- Builds CPU staging tensors as `torch.int32` in pinned memory.
- Transfers to GPU on a separate CUDA stream with `non_blocking=True`.
- Keeps labels aligned and widens labels to `torch.long` on GPU after async copy.
- Prefetch depth default from launcher: `4`.

Important snippet behavior:

```python
arr = np.asarray(self.data[offsets], dtype=np.int32)
x_cpu = torch.empty((batch_size, block_size), dtype=torch.int32, pin_memory=True)
x_cpu.copy_(torch.from_numpy(arr), non_blocking=False)
...
x = x_cpu.to(self.device, non_blocking=True)
y = x.to(torch.long)
```

This is a meaningful improvement over the earlier CPU-side int64/no-prefetch path. Current low throughput is unlikely to be mostly data starvation unless profiling proves otherwise.

## Current MoE Dispatch Implementation

Code:

- `src/metis_mamba/model.py`
- `MetisMultiHeadLatentMoE`
- `MetisGroupedHeadExperts`

The first research packet correctly identified the old per-expert Python loop as a major issue. That has been replaced with grouped dispatch.

Current grouped dispatch flow:

1. Flatten token-head rows.
2. Compute top-k experts and weights.
3. Flatten expert assignments.
4. Sort assignments by expert with `torch.argsort(flat_experts, stable=True)`.
5. Build `tokens_per_expert` with `torch.bincount`.
6. Gather routed heads into expert-contiguous `x_perm`.
7. Optional padding to TE split alignment/capacity.
8. Run `self.grouped_experts(x_perm, tokens_per_expert, is_first_microbatch=...)`.
9. If padded, trim with `index_select(valid_positions)`.
10. Accumulate with `routed_output.index_add_(0, row_indices, y_perm * weights.unsqueeze(-1))`.

Counters logged:

- `moe_grouped_expert_dispatches`
- `moe_grouped_assignments`
- `moe_routed_expert_dispatches`
- `moe_capacity_padded_dispatches`
- `moe_capacity_overflow_fallbacks`
- `moe_capacity_padded_tokens`

Critical fix already made:

- An earlier padding trim used `masked_select(valid_mask.unsqueeze(-1))`, which created huge temporary memory pressure.
- It was changed to a direct index-select style trim.
- This fixed one OOM class but did not make batch 24 viable.

Remaining issue:

- Even with grouped dispatch, the TE grouped MoE expert path is not fast.
- Smoke tests show grouped BF16 faster than grouped reduced NVFP4.
- Real training is still only about 15.4k tok/s.
- The sort/gather/index_add route may still be expensive.
- TE grouped NVFP4 may be using unfused fallback kernels due `320` not being a multiple of `128`.

## Optimizer

Code:

- `src/metis_mamba/optim.py`

Current optimizer:

- `muon_adamw`
- Custom `MuonAdamWHybrid`
- Routed experts stay in AdamW by default.
- Muon is used for selected 2D hidden matrices.
- AdamW path now supports a foreach implementation.
- NVTX ranges exist around:
  - `optimizer_adamw`
  - `optimizer_muon`

Observed optimizer grouping:

```text
Muon-AdamW hybrid
muon params: 152,223,744
adamw params: 798,749,568
adamw_impl: foreach
routed_experts_muon: False
```

Important design choices:

- Routed expert matrices are not included in Muon by default. This is probably correct for now because routed expert params are huge and Muon would add Newton-Schulz cost.
- AdamW states are FP32, so optimizer-state memory is large.
- Several OOMs happened only after or near optimizer state allocation, meaning benchmark runs that pass step 1 can still fail at step 2.

OOM evidence:

- Batch 20 grad-accum 10 completed step 1 and then OOMed on step 2 after optimizer state had materialized.
- Batch 18 grad-accum 11 appears stable for at least 3 steps and eval/checkpoint.

Research questions:

- Can AdamW state be reduced safely for routed experts?
- Can TE/NVFP4 weight/gradient handling avoid part of the FP32 state pressure?
- Is foreach AdamW still too slow or memory-heavy?
- Would fused AdamW work with the actual TE parameter layouts?
- Is optimizer time a meaningful fraction after 100 warm steps?

## Attention Backend Sweep

Current default:

- `attention_backend = "sdpa"`
- `native_gqa_attention = True`
- `te_dot_product_attention = False`

TE DotProductAttention was tested at the stable `b18 g11` setting:

| Mode | Step 1 tok/s | Step 2 tok/s | Notes |
|---|---:|---:|---|
| SDPA native GQA | 13,222 | 15,396 | baseline stable |
| TE DotProductAttention | 12,559 | 15,268 | slightly slower, `sdpa 0` counter |

Interpretation:

- TE attention did not materially improve throughput in the short smoke.
- SDPA native GQA currently looks slightly better or roughly tied.
- Attention is not the first obvious bottleneck at these numbers.
- A full Nsight run could still find attention issues after MoE/precision fixes.

## Real Training Benchmarks

All benchmark rows below are real-data base-pretrain unless noted. Sequence length is `1024`.

### Tiny real-data low-precision proof

Setting:

- Batch `1`
- Grad accumulation `1`
- Real memmap data
- Reduced NVFP4/FP8-block path

Observed:

```text
step 1 train 10.7566 | tok/s 403  | step_s 2.54 | est_tflops 2.30 | active 0.72
step 2 train 10.8147 | tok/s 1646 | step_s 0.62 | est_tflops 9.39 | active 2.94
val 10.7108
saved best/latest checkpoint about 8.29 GiB
```

Counters included:

```text
sdpa 19
moe_grouped 19
assign 311296 on step 1
cap_pad 0
cap_over 19
pad_tok around 19328-19776
```

Interpretation:

- Real-data pipeline, next-token loss, grouped MoE, checkpoint save, and validation work.
- Tiny batch is only a correctness smoke.

### Batch 24 grad-accum 8

Tokens per step:

- `24 * 8 * 1024 = 196,608`

First tried with capacity padding:

- Capacity factor `1.05`
- OOM before first optimizer step.
- Process around `93.19 GiB` in use.
- Tried to allocate `1.88 GiB`.
- Failure came from the old padded trim path using a large `masked_select(valid_mask.unsqueeze(-1))` temporary.

After trim fix and capacity factor `0`:

- Still OOMed in TE grouped linear down projection on first step.
- Process around `94.79 GiB` in use.
- Tried to allocate about `272 MiB`.

Interpretation:

- Batch 24 is not currently viable in this model/image/precision path on 96GB.
- Memory is extremely tight even before considering longer stable runs.

### Batch 20 grad-accum 10

Tokens per step:

- `20 * 10 * 1024 = 204,800`

Observed:

```text
step 1 train 10.7233
tok/s 13,298
step_s 15.40
est_tflops 75.87
active_tflops 23.72
sdpa 190
moe_grouped 190
assign 62,259,200
cap_pad 0
cap_over 0
pad_tok 191,360
```

Then OOMed on step 2:

```text
in use 94.68 GiB
allocated 87.00 GiB
reserved 3.16 GiB
tried to allocate 414 MiB
```

Interpretation:

- Step 1 can be misleading because optimizer state is not fully materialized until after the first backward/step.
- Batch 20 is not stable.

### Batch 19 grad-accum 10

Tokens per step:

- `19 * 10 * 1024 = 194,560`

Observed:

- Started but did not emit a completed step for several minutes.
- `nvidia-smi` showed about 100 percent GPU utilization and about `89,177 MiB` allocated.
- The run was killed rather than waiting longer.

Interpretation:

- Something pathological or just very slow may happen at this shape.
- It was not selected as stable.

### Batch 16 grad-accum 12

Tokens per step:

- `16 * 12 * 1024 = 196,608`

Observed stable 3-step run:

```text
step 1 train about 10.7
tok/s 13,074
step_s 15.04
est_tflops 74.60
active_tflops 23.32
sdpa 228
moe_grouped 228
assign 59,768,832
pad_tok 229,888

step 2 tok/s 15,291
step_s 12.86
est_tflops 87.25
active_tflops 27.27

step 3 tok/s 15,271
step_s 12.87
est_tflops 87.14
active_tflops 27.24

val 10.5336
ppl 37556.18
saved checkpoint about 8.29 GiB
```

Interpretation:

- Stable enough for smoke.
- Slightly lower than batch 18/grad 11.

### Batch 18 grad-accum 11: best current stable setting

Tokens per step:

- `18 * 11 * 1024 = 202,752`

Observed:

```text
step 1 train 10.7245
lr 1.2e-4
tok/s 13,222
step_s 15.33
est_tflops 75.44
active_tflops 23.58
sdpa 209
moe_grouped 209
assign 61,636,608
cap_pad 0
cap_over 0
pad_tok 210,112

step 2 train 10.6460
lr 6e-5
tok/s 15,396
step_s 13.17
est_tflops 87.85
active_tflops 27.46
pad_tok 212,864

step 3 train 10.5443
lr 0
tok/s 15,377
step_s 13.19
est_tflops 87.74
active_tflops 27.42
pad_tok 209,344

validation/train estimate 10.5817
val 10.4968
ppl 36199.94
saved checkpoint about 8.29 GiB
```

Interpretation:

- This is the best stable setting found so far.
- It is still far too slow.
- The first step is slower due warmup/state initialization; steps 2-3 are the relevant short-run steady estimate.

### TE DotProductAttention at batch 18 grad-accum 11

Setting:

- Same as stable run, but `METIS15_TE_DOT_PRODUCT_ATTENTION=1`

Observed:

```text
step 1 tok/s 12,559
step_s 16.14
est_tflops 71.66
active_tflops 22.40
sdpa 0

step 2 tok/s 15,268
step_s 13.28
est_tflops 87.12
active_tflops 27.23
```

Interpretation:

- No meaningful win over SDPA in this short test.

### All-expert reduced NVFP4 at batch 18 grad-accum 11

Setting:

- `METIS15_NVFP4_FINAL_EXPERT_LAYERS=0`
- This means no final-2 expert FP8-block safety override.

Observed:

```text
step 1 tok/s 13,230
step_s 15.32
est_tflops 75.49
active_tflops 23.60

step 2 tok/s 15,421
step_s 13.15
est_tflops 87.99
active_tflops 27.50
```

Interpretation:

- Only about 0.3 percent faster than final-2 FP8-block expert safety.
- Not worth the possible quality/stability risk unless a longer pilot proves it.

## Benchmark Command Templates

### Environment/image sanity

```bash
ssh -i /Users/giulianno/.ssh/aws_codex_builder.pem ubuntu@98.94.18.6
nvidia-smi
docker run --rm --gpus all metis15-blackwell-ngc2604-te-main:sm120a bash -lc '
python - <<PY
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cap", torch.cuda.get_device_capability(0))
print("gpu", torch.cuda.get_device_name(0))
import transformer_engine as te
print("te", getattr(te, "__version__", None))
import transformer_engine.common.recipe as recipe
print("has_nvfp4", hasattr(recipe, "NVFP4BlockScaling"))
print("has_mxfp8", hasattr(recipe, "MXFP8BlockScaling"))
print("has_float8_block", hasattr(recipe, "Float8BlockScaling"))
PY
aws --version
'
```

### Exact-shape smoke

```bash
docker run --rm --gpus all --ipc=host \
  -v /opt/dlami/nvme/metis/10M-model:/opt/metis \
  -w /opt/metis \
  metis15-blackwell-ngc2604-te-main:sm120a \
  make metis15-blackwell-smoke
```

### Stable real-data benchmark

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /opt/dlami/nvme/metis/10M-model:/opt/metis \
  -v /opt/dlami/nvme/metis/data/metis15_base:/opt/dlami/nvme/metis/data/metis15_base \
  -v /opt/dlami/nvme/metis/checkpoints:/opt/dlami/nvme/metis/checkpoints \
  -w /opt/metis \
  metis15-blackwell-ngc2604-te-main:sm120a \
  bash -lc '
    export METIS15_DATA_DIR=/opt/dlami/nvme/metis/data/metis15_base
    export METIS15_OUT_DIR=/opt/dlami/nvme/metis/checkpoints/metis15_base_b18g11_bench
    export METIS15_LOCAL_BATCH_SIZE=18
    export METIS15_GRAD_ACCUM_STEPS=11
    export METIS15_MOE_CAPACITY_FACTOR=0
    export METIS15_LM_LOSS_IMPL=liger_fused_linear_ce
    export METIS15_PREFETCH_BATCHES=4
    export METIS15_RESUME=0
    python3 scripts/train_mamba_lm.py \
      --manifest configs/metis15_manifest.json \
      --data-dir "$METIS15_DATA_DIR" \
      --out-dir "$METIS15_OUT_DIR" \
      --train-stage pretrain \
      --batch-size 18 \
      --grad-accum-steps 11 \
      --max-steps 3 \
      --warmup-steps 2 \
      --lr 1.2e-4 \
      --weight-decay 0.1 \
      --beta1 0.9 \
      --beta2 0.95 \
      --log-interval 1 \
      --eval-interval 3 \
      --checkpoint-interval 3 \
      --dtype bf16 \
      --matmul-precision highest \
      --optimizer muon_adamw \
      --fused-adamw \
      --prefetch-batches 4 \
      --nvfp4 \
      --nvfp4-disable-rht \
      --nvfp4-disable-2d-quantization \
      --nvfp4-disable-stochastic-rounding \
      --lm-loss-impl liger_fused_linear_ce
  '
```

### Attention test

Same as above, plus:

```bash
export METIS15_TE_DOT_PRODUCT_ATTENTION=1
```

or direct CLI:

```bash
--te-dot-product-attention
```

## Current Bottleneck Hypotheses

These are not proven in Nsight yet, but they are the strongest leads.

### 1. TE low-precision expert kernels are not actually fast on SM120 with Metis dimensions

Evidence:

- BF16 grouped MoE smoke: `28.18 ms`, `102.88 TFLOP/s`
- Reduced NVFP4 grouped MoE smoke: `31.06 ms`, `93.35 TFLOP/s`
- FP8-block grouped MoE smoke: `31.89 ms`, `90.90 TFLOP/s`
- Warning about unfused NVFP4 fallback because input inner dim is not a multiple of 128.

Potential causes:

- Routed latent dimension `320` may be bad for NVFP4 grouped kernels.
- TE GroupedLinear may not have a mature SM120 path for this exact dtype/layout/shape.
- Reduced NVFP4 disables the mechanisms expected by the optimized kernels.
- Fallback quantization and dequantization overhead may dominate.

Research asks:

- Is changing routed latent size from `320` to `384`, `256`, or another multiple-of-128 value worth considering?
- Can weights stay at 320 while padding GEMMs to 384 or 512 internally without changing architecture semantics?
- Does CUTLASS have a better SM120 grouped GEMM path than TE GroupedLinear here?
- Would MegaBlocks/block-sparse style kernels outperform TE GroupedLinear for 32 experts top-4 single GPU?

### 2. Grouped dispatch still has expensive sort/gather/scatter overhead

Evidence:

- Current code still uses `argsort`, `index_select`, optional padding, grouped experts, and `index_add_`.
- This is much better than the old 32-expert Python loop but not a fused MoE kernel.
- Real training step time remains about `13.17 s` for only `202,752` tokens.

Research asks:

- Can token count, permutation, and weighted unpermutation be fused with Triton?
- Can row indices/topk be bucketized or preallocated to reduce allocator pressure?
- Can capacity padding be made static without causing OOM?
- Can `torch.argsort` be replaced by a specialized histogram/prefix-sum/counting sort for 32 experts?
- Can the entire route/permute/unpermute path be captured or fused?

### 3. Memory pressure is severe

Evidence:

- 96GB card is almost full on larger batches.
- Batch 20 can pass one step and then OOM on step 2 after optimizer state allocation.
- Batch 24 OOMs even after major padding fix.
- Checkpoints are about `8.29 GiB`.
- AdamW params are about `798.75M`, with FP32 optimizer state.

Research asks:

- Can AdamW state be BF16/FP32 mixed safely for routed experts?
- Can optimizer state be allocated lazily or sharded even on one GPU?
- Can activation checkpointing be improved without destroying speed?
- Are TE FP8/NVFP4 tensors causing extra hidden caches/buffers?
- Does `is_first_microbatch` weight caching in TE GroupedLinear help or hurt with grad accumulation?

### 4. Current "low precision" may be mostly overhead

Evidence:

- Reduced NVFP4 exact-shape expert/down kernels are sometimes slower than BF16.
- All-expert reduced NVFP4 barely improves throughput versus final-2 FP8-block safety.
- MXFP8 is not actually usable.
- The low-precision path still stores BF16 master weights and large FP32 optimizer states.

Research asks:

- Is reduced NVFP4 worthwhile for training, or is BF16 expert compute faster until TE SM120 improves?
- Is there a different FP8 recipe/layout that works on SM120 and beats BF16?
- Is the target card's RTX PRO 6000 GDDR7 bandwidth/tensor-core behavior fundamentally different from B200-style expectations for these kernels?

### 5. Attention is not the first bottleneck, but should still be profiled

Evidence:

- TE DotProductAttention did not beat SDPA in the short test.
- Attention still accounts for a large active compute fraction.

Research asks:

- Which SDPA backend is actually selected on this exact PyTorch 2.12/CUDA 13.2 build?
- Is native GQA avoiding repeat-interleave materialization?
- Is FA4 available or relevant to RTX PRO 6000 SM120?
- Is attention compute hidden by MoE dispatch/optimizer overhead in the current short benchmarks?

## What Has Already Been Implemented

Implemented since the first research packet:

- New Blackwell Docker image based on NGC PyTorch 26.04.
- Rebuilt Transformer Engine main for `12.0a` / `120a`.
- Patched TE MXFP8 SM120 guard experimentally.
- Added exact-shape Blackwell kernel smoke script.
- Added reduced NVFP4 runtime gate for SM120.
- Added Float8BlockScaling fallback precision surfaces.
- Changed manifest from direct MXFP8 surfaces to `fp8_block` surfaces.
- Added final-2 expert higher precision option.
- Patched standalone pretrain launcher to use Liger fused linear CE.
- Added grouped MoE dispatch replacing the old per-expert Python loop.
- Added grouped expert module path using TE GroupedLinear when available.
- Added MoE capacity/padding controls.
- Fixed a major grouped-MoE padding trim OOM.
- Added pinned CUDA batch prefetcher.
- Added foreach AdamW logic inside the Muon-AdamW hybrid.
- Added optimizer NVTX ranges.
- Added logging for grouped MoE counters, active TFLOPs, and 450 TFLOP/s token targets.
- Verified next-token loss alignment with standard CE and Liger CE.
- Hydrated real 50B-token data onto NVMe.
- Ran real-data smoke/benchmark passes.

This is not an untouched stack. It has already had the obvious first-round fixes. The remaining problem is deeper: kernel/runtime/layout efficiency on SM120 and the actual MoE training dataflow.

## Current Open Questions For The Research Model

1. Is RTX PRO 6000 SM120 currently capable of fast training NVFP4 for TE GroupedLinear at dimensions involving `K=320`, `K=384`, `K=1280`, and `K=1536`, or are we hitting incomplete consumer/pro RTX Blackwell kernel coverage?

2. Is there a known TE branch, cuBLASLt version, CUTLASS version, or NGC image newer than 26.04 that fixes:
   - default NVFP4 Hadamard/RHT invalid-argument failures on SM120
   - MXFP8 SM120 guard/failure
   - NVFP4 grouped-kernel fallback for non-128 input dimensions

3. Is the routed latent dimension `320` a bad choice for Blackwell NVFP4 grouped GEMMs? If yes, what is the least architecture-invasive fix:
   - internal padding to 384
   - internal padding to 512
   - change routed latent size to 384
   - change expert hidden size
   - use a custom kernel that supports 320 efficiently

4. Can the MoE route/permute/unpermute be replaced with a fused Triton/CUTLASS/MegaBlocks-like local kernel for one GPU, top-4, 32 experts?

5. Is TE GroupedLinear the right abstraction for this exact workload, or should expert weights be packed into a custom grouped GEMM format?

6. Can the optimizer state/memory layout be reduced enough to make larger batches viable without losing training correctness?

7. Are current TFLOP estimates misleadingly low because the dense-equivalent formula is not appropriate, or is the GPU genuinely underutilized? The active TFLOP/s number is also low, so this is probably not only a logging issue.

8. What is the expected throughput for RTX PRO 6000 Blackwell Server Edition on this workload if kernels were ideal? The current 15.4k tok/s implies about 37.6 days for 50B tokens.

9. Should we continue trying to make reduced NVFP4 work, or should the next attempt be a completely different low-precision stack such as:
   - FP8-block everywhere possible
   - BF16 experts but optimized dispatch
   - CUTLASS FP4 custom kernels
   - a newer TE nightly
   - a different NVIDIA container/toolchain

10. Could a full BF16 run be faster than the current reduced low-precision run even though it uses more memory? This is unacceptable as a final precision plan from the user's perspective, but it may be a diagnostic benchmark to isolate low-precision overhead.

## Research Model Warnings

- Do not answer "RTX PRO 6000 supports Blackwell FP4, so NVFP4/MXFP8 should work" without checking exact training GEMMs. The card may support the data types while the public TE/cuBLASLt path fails or falls back for specific SM120 layouts.
- Do not assume B100/B200 performance results transfer to RTX PRO 6000 Server Edition. This is SM120 with GDDR7 and different ecosystem maturity.
- Do not optimize only the dense-equivalent total-param TFLOP/s metric. Active TFLOP/s and token/s matter more.
- Do not treat step 1 success as stability. Batch 20 passed step 1 and OOMed on step 2 after optimizer state.
- Do not recommend full BF16 as the final answer. It may be useful as a diagnostic, but the user explicitly wants NVFP4/MXFP8 or another genuinely low-precision Blackwell path working.
- Do not ignore the warning about NVFP4 grouped fallback for input inner dim not multiple of 128. This is one of the strongest clues in the whole packet.
- Do not propose Dynamic MoR/CPT optimization before base pretrain is fast. Base pretrain has Dynamic MoR disabled and is already too slow.

## Most Useful Next Experiments

Highest-signal experiments for the next round:

1. Run Nsight Systems on the stable `b18 g11` real-data command for about 20 warm steps plus 5 profiled steps.
2. Record time breakdown:
   - attention
   - MoE route/sort/gather/scatter
   - grouped expert GEMMs
   - shared expert
   - LM head/loss
   - optimizer AdamW
   - optimizer Muon
   - data wait
3. Run exact-shape microbench variants with dimensions padded or changed:
   - routed latent 320 vs padded 384 vs padded 512
   - expert hidden 1280 unchanged vs alternate alignments
   - grouped experts with and without TE GroupedLinear
4. Test newest possible TE/NGC/nightly stack if available after 26.04.
5. Test CUTLASS or custom Triton grouped MoE kernels for the exact shape:
   - 32 experts
   - top-4
   - about 98,304 token-head rows per microbatch at b24
   - about 73,728 token-head rows per microbatch at b18
6. Compare BF16 grouped MoE versus reduced NVFP4 grouped MoE in the full training loop, not only microbench, to quantify whether low precision is currently a slowdown.
7. Try `moe_routed_latent_size=384` in a disposable architecture branch only as a kernel-shape diagnostic. This changes parameter count/model shape, so it is not a casual patch.
8. Investigate whether a true SM120 MXFP8 training path exists in TE main/nightly or whether the guard is correct and it is not ready.

## Raw Context Appendices

These appendices intentionally include broad context, not only suspected causes. The point is to let another model notice details that may look boring locally but matter in combination.

### Appendix A: Local Repo State At Handoff

The local repo is dirty and mid-Metis-1.5 transition. Do not assume committed main contains this exact state. Relevant `git status --short` excerpt from `/Users/giulianno/Documents/10M model`:

```text
 M .dockerignore
 M Makefile
 M README.md
 D configs/metis14_chat_mix.json
 D configs/metis14_continued_pretrain_mix.json
 D configs/metis14_eval_prompts.json
 D configs/metis14_manifest.json
 D configs/metis14_preference_mix.json
 D configs/metis14_pretrain_mix_best_research.json
 D configs/metis14_pretrain_mix_release_clean.json
 D configs/metis14_real_benchmarks.json
 D configs/metis14_reasoning_mix.json
 D configs/metis14_static_block_mor.json
 D configs/metis14_static_dense_pretrain.json
 D configs/metis14_static_sequence_mor.json
 M docker/runpod-metis-gpu/Dockerfile
 M docker/runpod-metis-gpu/README.md
 D docs/metis14_static_dense_perf_notes.md
 D examples/metis14_identity_sft.jsonl
 D metis14_data_plan.md
 D metis14_h100_perf_research_packet_2026-05-05.md
 D metis14_plan.txt
 M requirements-gpu-train.txt
 D scripts/aws_p5_metis14_auto_handoff.sh
 D scripts/aws_p5_metis14_full.sh
 D scripts/aws_p5_metis14_pretrain.sh
 D scripts/aws_p5_metis14_real_benchmarks.sh
 D scripts/aws_p5_metis14_smoke.sh
 D scripts/aws_p5_metis14_static_dense_pretrain.sh
 D scripts/aws_p5_metis14_static_dense_pretrain_optimized.sh
 D scripts/aws_p5_metis14_static_sequence_continued_pretrain.sh
 D scripts/benchmark_metis14_static_dense_sweep.sh
 D scripts/benchmark_metis14_step.py
 M scripts/chat_app.py
 M scripts/data_mixture.py
 M scripts/eval_model_suite.py
 M scripts/export_mamba_checkpoint.py
 D scripts/metis14_cpu_prep.sh
 D scripts/mine_metis14_custom_negatives.py
 M scripts/normalized_shard_mixture.py
 M scripts/plan_metis13.py
 M scripts/prepare_metis13_sft_data.py
 D scripts/prepare_metis14_preference_data.py
 M scripts/prepare_normalized_shards.py
 M scripts/prepare_streaming_data.py
 D scripts/profile_metis14_ncu.sh
 D scripts/profile_metis14_nsys.sh
 M scripts/render_metis13_hf_assets.py
 D scripts/run_metis14_benchmarks.py
 D scripts/smoke_metis14_static_modes.py
 M scripts/train_mamba_dpo.py
 M scripts/train_mamba_lm.py
 M scripts/train_mamba_reward.py
 M scripts/train_mamba_sft.py
 M src/metis_mamba/checkpoint_compat.py
 M src/metis_mamba/config.py
 M src/metis_mamba/fp8.py
 M src/metis_mamba/model.py
 M src/metis_mamba/runtime.py
 M src/metis_runtime.py
 M src/tinylm/inference.py
 M web/static/app.css
 M web/static/app.js
 M web/templates/index.html
?? configs/metis15_chat_mix.json
?? configs/metis15_continued_pretrain_mix.json
?? configs/metis15_eval_prompts.json
?? configs/metis15_manifest.json
?? configs/metis15_preference_mix.json
?? configs/metis15_pretrain_mix.json
?? configs/metis15_pretrain_mix_proofpile_arxiv_chunked_rebuild.json
?? configs/metis15_reasoning_mix.json
?? docker/runpod-metis-gpu/Dockerfile.ngc-blackwell
?? docker/runpod-metis-gpu/Dockerfile.runtime-liger
?? docs/metis15_data_plan.md
?? docs/metis15_multihead_latent_moe.md
?? docs/metis15_rtx_pro_6000_blackwell_testing_handoff_2026-05-14.md
?? docs/metis15_rtx_pro_6000_nvfp4_research_packet_2026-05-13.md
?? docs/metis16_data_prep_throughput_plan.md
?? examples/metis15_identity_sft.jsonl
?? scripts/metis15_cpu_prep.sh
?? scripts/metis15_cpu_prep_supervisor.sh
?? scripts/metis15_full.sh
?? scripts/metis15_pretrain.sh
?? scripts/metis15_rtx_benchmark_matrix.sh
?? scripts/mine_metis15_custom_negatives.py
?? scripts/prepare_metis15_preference_data.py
?? scripts/smoke_metis15_blackwell_kernels.py
?? scripts/smoke_metis15_training_contracts.py
?? scripts/validate_metis15_data_plan.py
?? src/metis_mamba/optim.py
```

The Metis-1.5 files are mostly untracked in this checkout. That matters for reproducibility: the remote instance had a synced working tree, not a clean release tag.

### Appendix B: Live Host Raw Probe

Raw SSH probe on 2026-05-14:

```text
### HOST
ip-172-31-91-14
Thu May 14 13:42:05 UTC 2026
Linux ip-172-31-91-14 6.17.0-1013-aws #13~24.04.1-Ubuntu SMP Fri Apr 24 21:50:45 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

### NVIDIA
NVIDIA-SMI 595.58.03
Driver Version: 595.58.03
CUDA Version: 13.2
GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition
Persistence-M: On
Bus-Id: 00000000:2F:00.0
ECC volatile uncorrected: 0
Temperature: 25C
Perf state: P8
Power: 29W / 600W
Memory: 0MiB / 97887MiB
GPU-Util: 0%
Compute mode: Default
MIG mode: Disabled
No running processes found

### DOCKER IMAGES
metis15-blackwell-ngc2604-te-main:sm120a 87681c8ac578 35.6GB
metis15-blackwell-sm120a:cu128-te215 37466697228e 31.3GB
metis15-blackwell-sm120:local caa774230d5b 31.3GB
nvcr.io/nvidia/pytorch:26.04-py3 192d749b4d77 34.9GB
nvidia/cuda:12.8.1-base-ubuntu24.04 133c78a05753 411MB

### DISK
/dev/root                        29G   14G   15G  48% /
/dev/mapper/vg.01-lv_ephemeral  1.7T  162G  1.5T  10% /opt/dlami/nvme

### DATA
total 95G
meta.json 91K
train.bin 94G
val.bin 964M

### META KEYS
dtype uint16
vocab_size 32768
train_tokens 50000000000
val_tokens 505050506
target_train_tokens 50000000000
target_val_tokens 505050506
train_docs 52887626
val_docs 535354
mixture_config configs/metis15_pretrain_mix.json
source_counts_len 26
```

### Appendix C: Real Data Source Counts

The base pretrain memmap metadata includes 26 source-count entries:

```text
dclm_baseline_hq train_docs=9702544 val_docs=97456 train_tokens=11195689299 val_tokens=112305274
finewiki_english_structured train_docs=3174529 val_docs=32101 train_tokens=2343144103 val_tokens=23655038
essential_web_v1_curated_high_quality train_docs=2771925 val_docs=28075 train_tokens=1823247497 val_tokens=18575396
fineweb_edu_score3plus train_docs=6930205 val_docs=69795 train_tokens=6481245229 val_tokens=64222559
openstax_textbooks train_docs=50114 val_docs=466 train_tokens=39701888 val_tokens=360892
cosmopedia_v2_capped train_docs=2079064 val_docs=20936 train_tokens=1527249891 val_tokens=15369423
pg19_books_reduced train_docs=1108753 val_docs=11247 train_tokens=708195805 val_tokens=7181652
pes2o_stem train_docs=3464805 val_docs=35195 train_tokens=980532261 val_tokens=9986303
nemotron_cc_math_4plus train_docs=3880809 val_docs=39191 train_tokens=4139329455 val_tokens=41671495
common_pile_stackexchange_filtered train_docs=1663201 val_docs=16799 train_tokens=1278339010 val_tokens=12858485
reserve_underrepresented_stem train_docs=554293 val_docs=5707 train_tokens=156958947 val_tokens=1622945
common_corpus_educational_reference train_docs=2446480 val_docs=24520 train_tokens=2974104960 val_tokens=29663479
common_pile_project_gutenberg_filtered train_docs=969996 val_docs=10004 train_tokens=1144317443 val_tokens=11805643
txt360_bestofweb_english_cc train_docs=1732732 val_docs=17268 train_tokens=2198258940 val_tokens=21787805
reserve_underrepresented_reference train_docs=415856 val_docs=4144 train_tokens=333324614 val_tokens=3277284
finemath_4plus_reduced train_docs=1662994 val_docs=17006 train_tokens=2125094590 val_tokens=21680732
proof_pile2_arxiv_science train_docs=1823634 val_docs=18598 train_tokens=2775111655 val_tokens=28336949
fineweb_hq_score_filtered train_docs=2589444 val_docs=26160 train_tokens=1806702398 val_tokens=18576901
dclm_edu_score_filtered train_docs=2990384 val_docs=30737 train_tokens=3671998529 val_tokens=38010154
roots_en_no_code_stackexchange train_docs=453360 val_docs=4771 train_tokens=345468088 val_tokens=3714729
common_pile_educational_filtered train_docs=676682 val_docs=6839 train_tokens=158550752 val_tokens=1605276
zyda2_novelty_heavy_sample train_docs=783448 val_docs=8274 train_tokens=905907890 val_tokens=9498801
openwebmath_equation_rich train_docs=802430 val_docs=8190 train_tokens=705703286 val_tokens=7216329
reserve_underrepresented_math train_docs=86922 val_docs=886 train_tokens=103771029 val_tokens=1018128
synthetic_textbook_explainer_selected train_docs=36331 val_docs=514 train_tokens=26619119 val_tokens=368403
common_corpus_books_wikisource train_docs=36691 val_docs=475 train_tokens=51433322 val_tokens=680431
```

### Appendix D: Full Current Manifest Snapshot

This is the current relevant manifest subset from `configs/metis15_manifest.json`. It includes general info even when not obviously performance-related.

```json
{
  "name": "Metis-1.5",
  "model": {
    "name": "Metis-1.5",
    "architecture": "metis_multihead_latent_moe_decoder",
    "model_type": "metis_multihead_latent_moe",
    "vocab_size": 32768,
    "block_size": 1024,
    "d_model": 1536,
    "n_layer": 19,
    "n_heads": 24,
    "n_kv_heads": 8,
    "head_dim": 64,
    "intermediate_size": 4096,
    "hidden_act": "swiglu",
    "tie_embeddings": true,
    "rms_norm": true,
    "residual_in_fp32": false,
    "fused_add_norm": false,
    "pad_vocab_size_multiple": 16,
    "initializer_range": 0.02,
    "torch_dtype": "bfloat16",
    "attention_bias": false,
    "mlp_bias": false,
    "attention_dropout": 0.0,
    "rope_theta": 10000.0,
    "attention_backend": "sdpa",
    "training_mode": "static_dense_pretrain",
    "mor_enabled": false,
    "mor_train_router": false,
    "mor_runtime_mode": "disabled",
    "mor_max_depth": 3,
    "mor_router_hidden_dim": 384,
    "mor_router_aux_loss_coef": 0.0,
    "mor_router_entropy_coef": 0.0,
    "mor_router_z_loss_coef": 0.0,
    "mor_target_avg_depth": 1.0,
    "ffn_type": "multi_head_latent_moe",
    "moe_num_experts": 32,
    "moe_top_k": 4,
    "moe_shared_experts": 1,
    "moe_num_heads": 4,
    "moe_expert_intermediate_size": 1280,
    "moe_router_latent_size": 128,
    "moe_routed_latent_size": 320,
    "moe_router_temperature": 1.0,
    "moe_aux_loss_coef": 0.0,
    "moe_router_score": "sigmoid",
    "moe_balance_strategy": "aux_loss_free_bias",
    "moe_balance_bias_update_rate": 0.001,
    "moe_balance_bias_clamp": 5.0,
    "moe_balance_scale_by_token_fraction": true,
    "moe_dispatch_mode": "grouped",
    "moe_capacity_factor": 0.0,
    "moe_capacity_alignment": 128,
    "low_precision_mode": "nvfp4",
    "nvfp4_disable_rht": true,
    "nvfp4_disable_2d_quantization": true,
    "nvfp4_disable_stochastic_rounding": true,
    "nvfp4_keep_embeddings_bf16": true,
    "nvfp4_keep_qkv_bf16": false,
    "nvfp4_keep_latent_moe_projections_bf16": false,
    "nvfp4_keep_lm_head_bf16": true,
    "nvfp4_qkv_precision": "fp8_block",
    "nvfp4_latent_moe_projection_precision": "fp8_block",
    "nvfp4_lm_head_precision": "bf16",
    "nvfp4_final_expert_layers": 2,
    "nvfp4_final_expert_precision": "fp8_block",
    "fp8_pad_multiple": 64,
    "estimated_params": 950973312,
    "estimated_active_params": 297251712,
    "estimated_active_transformer_params": 246860160
  },
  "hardware": {
    "target_cluster": "rtx_pro_6000_workstation",
    "accelerator": "1x NVIDIA RTX PRO 6000 96GB",
    "world_size": 1,
    "preferred_precision": "nvfp4_sm120_safe_with_fp8_block_surfaces",
    "fallback_precision": "bf16",
    "attention_backend": "sdpa",
    "flash_attention_source": "optional",
    "flash_attention_cuda_min": "optional",
    "flash_attention_cuda_recommended": "optional",
    "fp8": {
      "enabled": true,
      "format": "HYBRID",
      "margin": 0,
      "amax_history_len": 16,
      "amax_compute_algo": "max"
    },
    "nvfp4": {
      "enabled": false,
      "master_weights": "bf16",
      "recipe": "NVFP4BlockScaling",
      "disable_rht": true,
      "disable_2d_quantization": true,
      "disable_stochastic_rounding": true,
      "higher_precision_modules": [
        "embeddings",
        "lm_head",
        "final_2_expert_blocks_fp8_block"
      ],
      "fp8_block_modules": [
        "attention_qkv",
        "latent_moe_router_projection",
        "latent_moe_routed_down_projection",
        "latent_moe_routed_up_projection"
      ]
    },
    "launcher": {
      "torchrun_nproc_per_node": 1,
      "omp_num_threads": 8,
      "nccl_debug": "WARN"
    }
  },
  "optimizer": {
    "name": "muon_adamw",
    "adamw_eps": 1e-08,
    "hybrid_adamw_impl": "foreach",
    "muon_beta": 0.95,
    "muon_ns_steps": 5,
    "muon_lr_scale": 1.0,
    "muon_nesterov": true,
    "include_routed_experts": false,
    "policy": {
      "adamw": [
        "token_embeddings",
        "lm_head",
        "rmsnorm_layernorm_weights",
        "router_gate_networks",
        "mor_recursion_router_control",
        "moe_router_embeddings_and_biases",
        "latent_moe_router_projection",
        "positional_scalars_biases"
      ],
      "muon": [
        "attention_qkv_o_matrices",
        "dense_mlp_matrices",
        "shared_expert_matrices",
        "latent_moe_routed_payload_down_up_projections"
      ],
      "ablation_only": [
        "routed_expert_up_gate_down_matrices"
      ]
    }
  },
  "pretrain": {
    "training_mode": "static_dense_pretrain",
    "target_train_tokens": 50000000000,
    "gates": [
      {"label": "gate_10b", "tokens": 10000000000},
      {"label": "gate_25b", "tokens": 25000000000},
      {"label": "gate_40b", "tokens": 40000000000},
      {"label": "gate_50b", "tokens": 50000000000}
    ],
    "val_ratio": 0.01,
    "base_lr": 0.00012,
    "warmup_ratio": 0.02,
    "weight_decay": 0.1,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.95,
    "local_batch_size": 18,
    "grad_accum_steps": 11,
    "log_interval": 20,
    "eval_interval": 1000,
    "checkpoint_interval": 2500
  },
  "continued_pretrain": {
    "training_mode": "dynamic_token_mor",
    "mor": {
      "enabled": true,
      "train_router": true,
      "runtime_mode": "dynamic_token",
      "max_depth": 3,
      "router_hidden_dim": 384,
      "router_temperature": 1.0,
      "router_aux_loss_coef_start": 0.01,
      "router_aux_loss_coef": 0.02,
      "router_aux_loss_coef_end": 0.02,
      "router_entropy_coef": 0.002,
      "router_z_loss_coef": 0.0001,
      "target_avg_depth_start": 1.05,
      "target_avg_depth": 1.65,
      "target_avg_depth_end": 1.65,
      "target_avg_depth_warmup_tokens": 1000000000,
      "disable_depth_stack": false,
      "disable_token_packing": false,
      "disable_token_scatter": false,
      "moe_balance_scale_by_token_fraction": true
    },
    "target_train_tokens": 10000000000,
    "val_ratio": 0.01,
    "base_lr": 0.00006,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "optimizer_beta1": 0.9,
    "optimizer_beta2": 0.95,
    "local_batch_size": 16,
    "grad_accum_steps": 8,
    "checkpoint_interval": 2500
  }
}
```

### Appendix E: Python Requirements And Dependency Policy

`requirements-gpu-train.txt` currently contains:

```text
flask>=3.0
datasets>=2.19,<4
tokenizers>=0.15
sentencepiece>=0.2
tqdm>=4.66
numpy>=1.26
requests>=2.31
huggingface_hub>=0.29
transformers>=4.51,<5
safetensors>=0.4
einops>=0.8
packaging>=24.0
psutil>=5.9
ninja>=1.11
boto3>=1.34
awscli>=1.32
zstandard>=0.22
liger-kernel==0.8.0
```

Transformer Engine is intentionally not installed from this requirements file. It is built separately from source in `docker/runpod-metis-gpu/Dockerfile.ngc-blackwell` to match the PyTorch/CUDA runtime.

### Appendix F: Benchmark Matrix Script

`scripts/metis15_rtx_benchmark_matrix.sh` exists as a broader sweep wrapper. Its defaults:

```bash
STEPS="${METIS15_BENCH_STEPS:-300}"
WARMUP_STEPS="${METIS15_BENCH_WARMUP_STEPS:-20}"
BATCH_SIZE="${METIS15_BENCH_BATCH_SIZE:-18}"
GRAD_ACCUM="${METIS15_BENCH_GRAD_ACCUM:-11}"
LR="${METIS15_BENCH_LR:-1.2e-4}"
WEIGHT_DECAY="${METIS15_BENCH_WEIGHT_DECAY:-0.1}"
PREFETCH_BATCHES="${METIS15_PREFETCH_BATCHES:-4}"
RUN_BASELINES="${METIS15_BENCH_BASELINES:-1}"
LOW_PRECISION="${METIS15_BENCH_LOW_PRECISION:-nvfp4}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
```

It runs exact kernel smoke first unless `METIS15_SKIP_KERNEL_SMOKE=1`, then benchmarks:

```text
loop_sdpa_standard_ce_no_prefetch
grouped_sdpa_standard_ce
grouped_sdpa
grouped_sdpa_final_all_nvfp4
grouped_capacity_110
grouped_te_attention
grouped_sdpa_no_native_gqa
```

Important caveat: the full matrix was not completed because the early real-data exploratory tests already showed OOMs and low throughput. The stable short runs in this packet are the actually observed numbers.

### Appendix G: Things Not Yet Proven

These are not necessarily suspicious; they are simply not proven enough to assume:

- No full 300-step Nsight Systems benchmark has been completed on the final image.
- No Nsight Compute kernel-level analysis has been completed for the TE GroupedLinear expert kernels.
- No long 1B-token stability/quality pilot has been run with reduced NVFP4.
- No true MXFP8 training pass has completed on exact Metis shapes.
- No default NVFP4 training pass has completed on exact Metis shapes.
- No FA4-specific path has been proven on this RTX PRO 6000 instance.
- No CUDA Graph path is active.
- No `torch.compile` path is active for the low-precision run.
- No custom Triton/CUTLASS MoE route/permute/unpermute kernel has been implemented.
- No architecture-changing latent-dim alignment experiment has been run.
- No BF16 full-training diagnostic baseline has been run to completion at the same stable token step, because BF16-only is not acceptable as the final plan and memory was already tight.
- No longer-run validation exists yet for whether disabling NVFP4 RHT/2D quantization/stochastic rounding hurts convergence.
- No CPT Dynamic MoR benchmark has been run on the final Blackwell image. Base pretrain is the current blocker.
- No remote checkpoint upload endurance test was done beyond short checkpoint saves.

## Bottom Line

The current Metis-1.5 RTX PRO 6000 stack is launch-capable but not production-fast.

It is running:

- real 50B-token S3-hydrated data
- next-token prediction
- Liger fused linear CE
- grouped MoE dispatch
- pinned prefetch
- foreach AdamW
- reduced NVFP4 plus FP8-block fallback surfaces
- SDPA native GQA attention

But it is only achieving:

```text
about 15.4k tokens/sec
about 87.8 dense-equivalent TFLOP/s
about 27.5 active TFLOP/s
about 37.6 days for 50B tokens before overhead
```

The biggest red flag is that the exact expert/MoE shapes do not show a low-precision speed win. BF16 grouped MoE microbench was faster than reduced NVFP4 and FP8-block grouped MoE. That suggests the RTX PRO 6000/SM120 low-precision kernel path is currently falling back, misaligned, or otherwise not doing the thing we need it to do.

The research model should focus first on SM120 Transformer Engine/CUTLASS support for NVFP4/MXFP8 grouped GEMMs at Metis dimensions, especially the `320` routed latent dimension and TE's unfused fallback warning. If that cannot be fixed, the next best path is a custom grouped MoE kernel or an architecture-preserving internal padding strategy that unlocks real Blackwell tensor-core kernels without changing the model's external semantics.
