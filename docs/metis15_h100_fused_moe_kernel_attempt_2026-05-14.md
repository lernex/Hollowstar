# Metis-1.5 H100 fused MoE kernel attempt - 2026-05-14

## Executive summary

The H100 stack is not fundamentally dead. A reconstructed dense Metis-1.4 FP8 synthetic training benchmark reaches about **146k-147k tok/s** on this exact instance/image with SDPA. The current Metis-1.5 BF16 MoE path remains around **33.5k-34k tok/s** stable. That means the main tax is still Metis-1.5 sparse MoE architecture/backend, but the old 170k dense comparison also likely depended on a better attention/kernel stack because this image is missing FlashAttention-3.

The parameter-application audit also found that the old active-param throughput target is misleading: `config.estimate_active_params()` reports about **311M**, while direct multi-head top-4 MoE math is roughly **734M parameter-applications/token**. The trainer's active TFLOP/s is therefore undercounting actual work for Metis-1.5.

## Environment

- Instance: H100 host `18.183.61.208`
- Remote repo: `/opt/dlami/nvme/metis`
- Docker image: `metis15-h100:torch280-te215-torchgrouped`
- GPU: NVIDIA H100 80GB HBM3
- PyTorch: `2.8.0+cu128`
- Transformer Engine: `2.15.0+42b8400`
- Real Metis-1.5 data: `/opt/dlami/nvme/metis/data/metis15_base`
- Important missing piece: FlashAttention-3 import fails in this image:
  - `No module named 'flash_attn_3'`
  - `No module named 'flash_attn_interface'`
  - `No module named 'hopper'`

## Code added in this pass

- `src/metis_mamba/moe_kernels.py`
  - Added Triton fused SwiGLU forward kernel.
  - Added Triton SwiGLU backward kernel, but it is **not safe** in full training.
  - Added safe manual PyTorch backward fallback via `METIS_TRITON_SWIGLU_BACKWARD=torch`.
  - Exported `fused_swiglu(...)`.
- `src/metis_mamba/model.py`
  - Added `_apply_swiglu(...)` shared helper.
  - Added env-controlled fused SwiGLU surfaces:
    - `METIS_DISABLE_TRITON_SWIGLU=1` disables by default.
    - `METIS_TRITON_SWIGLU_SURFACES=grouped_experts` opts into grouped routed experts.
    - `METIS_TRITON_SWIGLU_BACKWARD=triton` opts into the fast unsafe backward.
    - `METIS_DEBUG_SWIGLU_FINITE=1` prints first non-finite fused activation details.
- `scripts/smoke_triton_swiglu.py`
  - Correctness check against PyTorch forward/backward.
- `scripts/benchmark_metis15_expert_path.py`
  - Isolated grouped expert MLP benchmark.
- `scripts/benchmark_metis14_h100_synthetic.py`
  - Reconstructed dense Metis-1.4 synthetic training benchmark.
- `scripts/audit_metis_compute.py`
  - Direct parameter-application accounting.
- `configs/metis14_h100_dense_manifest.json`
  - Reconstructed Metis-1.4 dense H100 sanity config.
- `src/metis_mamba/config.py`
  - Fixed `estimate_active_params()` for multi-head top-k MoE so active TFLOP/s logs account for per-head expert applications.

## Compute accounting audit

Command:

```bash
python3 scripts/audit_metis_compute.py --manifest configs/metis15_manifest.json
```

Result:

```text
config_estimate_params: 1,095,725,952
config_estimate_active_params_depth1: 734,311,296
embedding_params: 50,331,648
attention_param_apps_per_layer: 6,291,456
moe_heads: 4
top_k_per_head: 4
shared_experts_per_head: 1
expert_param_apps_per_assignment: 1,474,560
routed_expert_param_apps_per_layer: 23,592,960
shared_expert_param_apps_per_layer: 5,898,240
router_projection_and_match_apps_per_layer: 212,992
rough_moe_param_apps_per_layer: 35,995,648
rough_total_param_apps_per_token: 734,248,960
```

Interpretation: the previous `active-param` throughput log was too optimistic for Metis-1.5. The code now reports roughly 734M active parameter-applications/token for this architecture.

## Triton fused SwiGLU results

Correctness smoke:

```bash
python scripts/smoke_triton_swiglu.py --rows 512 --hidden-size 1280 --dtype fp32
```

Result:

```text
forward_max_abs=9.536743e-07 backward_max_abs=9.536743e-07
```

BF16 max absolute error was about `6.25e-02`, consistent with BF16 worst-case quantization on random tensors.

### Isolated expert MLP benchmark

Shape:

```text
32 experts
4096 rows/expert
head_dim=384
expert_hidden=1280
BF16
fwd+bwd+AdamW step
```

Fast Triton backward:

```text
mean_step_s=0.004202
mean_rows_s=31,189,627
```

PyTorch SwiGLU baseline:

```text
mean_step_s=0.006048
mean_rows_s=21,671,664
```

This is a real isolated speedup: about **1.44x** for the grouped expert MLP slice.

Safe PyTorch backward fallback:

```text
mean_step_s=0.030584
mean_rows_s=4,285,663
```

This is numerically correct but far too slow. It is not useful as a throughput fix.

### Full Metis-1.5 training with fast Triton backward

Command shape:

```bash
METIS15_BATCH_SIZE=8
METIS15_MAX_STEPS=20
METIS15_LR=1e-5
METIS_DISABLE_TRITON_SWIGLU=0
METIS_TRITON_SWIGLU_SURFACES=grouped_experts
METIS_TRITON_SWIGLU_BACKWARD=triton
./scripts/metis15_h100_benchmark.sh
```

Observed:

```text
stable-ish speed before failure: ~39k-41k tok/s
failure: train nan by step 5-8 depending on exact kernel variant
```

With `METIS15_LR=0`, the same failure still happens. That rules out normal optimizer LR drift.

Debug output with `METIS_DEBUG_SWIGLU_FINITE=1`:

```text
nonfinite_swiglu surface=grouped_experts gate_up_finite=True gate_up_min=-3.593069e+36 gate_up_max=2.866148e+36 hidden_min=-3.402823e+38 hidden_max=3.402823e+38
nonfinite_swiglu surface=grouped_experts gate_up_finite=False ...
```

Interpretation: the fast Triton backward likely corrupts/poisons training state in full-model execution. It passes isolated random fwd/bwd but is not safe in the routed model. The implementation is therefore left opt-in only, and disabled by default.

## Stable Metis-1.5 baseline control

Command:

```bash
METIS15_BATCH_SIZE=8
METIS15_MAX_STEPS=20
METIS_DISABLE_TRITON_SWIGLU=1
./scripts/metis15_h100_benchmark.sh
```

Result:

```text
step_s ~0.24
tok/s ~33.5k-33.9k
stable through 20 steps
```

This matches the prior stable H100 BF16 result.

## Static capacity test

Command:

```bash
python scripts/train_mamba_lm.py ... \
  --batch-size 8 \
  --moe-backend torch_grouped \
  --moe-dispatch-mode bucketed \
  --moe-static-capacity 4608 \
  --moe-overflow-mode drop \
  --moe-graphable
```

Result:

```text
assignments/step rose from 2,490,368 to 2,801,664
pad_tok 311,296
tok/s ~31.2k after warmup
train nan from step 2 onward
```

Interpretation: static capacity/drop is not a safe shortcut. It pads extra expert work and changes semantics. Graphability still needs a correct overflow/semantic plan.

## Dense Metis-1.4 reconstruction

Reconstructed specs from the old manifest:

```text
vocab_size=16384
block_size=1024
d_model=1536
n_layer=19
n_heads=24
n_kv_heads=8
head_dim=64
intermediate_size=4096
tie_embeddings=true
rms_norm=true
attention_backend originally flash_attention_3
estimated params ~503M
```

This repo no longer has the old Metis-1.4 data wired, so the benchmark used synthetic tokens. That is enough for kernel/throughput sanity, not loss-quality validation.

### Metis-1.4 BF16 dense

Batch 16:

```text
mean_step_s=0.457909
mean_tok_s=35,780
mean_total_tflops=108.1
```

Batch 32:

```text
mean_step_s=0.374409
mean_tok_s=87,519
mean_total_tflops=264.3
```

### Metis-1.4 FP8 dense, SDPA

Batch 32:

```text
mean_step_s=0.224754
mean_tok_s=145,795
mean_total_tflops=440.3
```

Batch 40:

```text
mean_step_s=0.277797
mean_tok_s=147,446
mean_total_tflops=445.3
```

TE fused dense MLP did not materially improve this:

```text
batch=32 fp8 te_fused_mlp=True
mean_tok_s=147,069
mean_total_tflops=444.2
```

FlashAttention-3 failed because it is not installed in the image. That likely accounts for part of the gap between this run and any old 170k dense FP8 number.

## What this proves

1. The H100 image can run dense FP8 around **146k-147k tok/s** for a reconstructed 503M Metis-1.4 model with SDPA. The stack is not catastrophically broken.
2. Metis-1.5 stable BF16 at **33.5k-34k tok/s** is bad, but it is also doing far more work than the old active-param estimate says.
3. Fusing SwiGLU is a real speed lever in isolation and in full-model pre-failure speed, but the custom Triton backward written in this pass is unsafe in full training.
4. A safe backward fallback removes the numerical issue but is too slow to use.
5. Static capacity/drop is not currently viable.
6. The current image lacks FlashAttention-3, so dense and MoE attention paths are not at the old intended Hopper kernel stack.

## Next recommended research target

Do not spend more time on the current hand-written Triton SwiGLU backward without a memory-safety audit. The symptom looks like full-model state corruption, not ordinary numeric instability.

The next serious backend routes are:

1. Install/validate FlashAttention-3 on the H100 image and rerun Metis-1.4 FP8 dense. This checks whether 170k is recoverable on the same stack.
2. Use Nsight Compute on `_grouped_mm` FC1/FC2 fwd/bwd/wgrad for Metis-1.5. The grouped GEMMs are still the core wall.
3. Replace the hand-written Triton SwiGLU backward with a generated/verified Triton implementation or a CUTLASS/CUDA extension with cuda-memcheck/racecheck coverage.
4. Investigate FP8 instability separately; 250k is not plausible from the current BF16 lane.
5. Continue validating compute accounting against profiler FLOPs, but the obvious missing `moe_num_heads` multiplier is now fixed.
