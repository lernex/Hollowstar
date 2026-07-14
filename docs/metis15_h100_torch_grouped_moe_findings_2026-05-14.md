# Metis-1.5 H100 MoE Backend Findings - 2026-05-14

This is a research handoff for the H100 pivot and MoE expert-backend replacement pass.

Bottom line: moving from TE `GroupedLinear` to PyTorch's H100 grouped GEMM primitive did remove a real wall, but overall throughput is still terrible relative to the 250k tok/s target. The stable real-data result is about 34k tok/s, not 250k. The current run is now less obviously blocked by TE's Python split-list expert path, but the model is still launch-heavy, BF16-only stable, and likely missing the real fused MoE kernels needed for this architecture.

## Executive Summary

### What Was Done

- Connected to the H100 instance at `18.183.61.208`.
- Verified the machine:
  - User: `ubuntu`
  - Hostname: `ip-172-31-8-8`
  - GPU: `NVIDIA H100 80GB HBM3`
  - Driver: `580.126.16`
  - Docker installed and working with NVIDIA runtime.
- Synced the current Metis repo to `/opt/dlami/nvme/metis`.
- Built a new H100 runtime image:
  - `metis15-h100:torch280-te215-torchgrouped`
  - Base: existing `metis15-h100-bench:torch280-te-source`
  - PyTorch: `2.8.0+cu128`
  - CUDA runtime: `12.8`
  - Transformer Engine: `2.15.0+42b8400`
  - Liger installed
  - `torch._grouped_mm` available
- Hydrated real Metis-1.5 base pretrain memmaps from S3:
  - `/opt/dlami/nvme/metis/data/metis15_base/train.bin`: `94G`
  - `/opt/dlami/nvme/metis/data/metis15_base/val.bin`: `964M`
  - total local base data dir: `95G`
- Implemented a new `torch_grouped` MoE backend using `torch._grouped_mm`.
- Added a H100 benchmark wrapper:
  - `scripts/metis15_h100_benchmark.sh`
- Changed the Metis-1.5 manifest default MoE backend from `te_grouped` to `torch_grouped`.

### Main Result

On real base pretrain data:

| Setup | Batch | Precision | Backend | Stability | Throughput |
|---|---:|---|---|---|---:|
| Old TE grouped expert path | 8 | BF16 | `te_grouped` | stable short run | ~2.1k-2.4k tok/s |
| New grouped GEMM expert path | 8 | BF16 | `torch_grouped` | stable 50 steps | ~33.5k-34.0k tok/s |
| New grouped GEMM expert path | 10 | BF16 | `torch_grouped` | `nan` around step 6 on real data | ~34.5k tok/s before failure |
| New grouped GEMM expert path | 8 | global FP8 / BF16 experts | `torch_grouped` | unstable, `nan` by step 3 or immediately in train-mode stress test | ~40k-47k tok/s before failure |
| New grouped GEMM + `torch.compile` | 8 | BF16 | `torch_grouped` | failed | unusable |

The expert-backend change is a real improvement: BF16 `torch_grouped` is about 14x faster than BF16 TE `GroupedLinear` in the tested training loop. But the absolute number is still far below target.

## Important Code Changes

### New PyTorch Grouped GEMM Expert Backend

Files:

- `src/metis_mamba/model.py`
- `src/metis_mamba/moe_kernels.py`
- `src/metis_mamba/config.py`
- `scripts/train_mamba_lm.py`
- `configs/metis15_manifest.json`

New backend:

```text
--moe-backend torch_grouped
```

New implementation class:

```python
class _AtenGroupedLinear(nn.Module):
    ...
    return torch._grouped_mm(hidden_states.contiguous(), self.weight, offsets, bias)
```

Weight layout for `torch._grouped_mm`:

```text
hidden_states: [sum_m, in_features]
weight:        [num_experts, in_features, out_features]
offsets:       [num_experts] int32 CUDA tensor of cumulative end offsets
output:        [sum_m, out_features]
```

Discovered schema on the H100 image:

```text
aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
```

Key point: `torch._grouped_mm` wants device-side int32 end offsets, not a Python `List[int]`.

### Dispatcher Change For Device Counts

Added:

```python
def bucket_dispatch_counts(...) -> tuple[x_perm, assignment_ids, reverse_positions, weights_perm, counts]
```

This returns the Triton bucket dispatch counts tensor directly instead of doing:

```python
counts.cpu().tolist()
```

For `torch_grouped`, the bucketed path now passes the CUDA counts tensor through to the expert backend, where it becomes int32 cumulative end offsets on-device.

### Current Stable H100 Benchmark Wrapper

File:

```text
scripts/metis15_h100_benchmark.sh
```

Default stable settings:

```bash
METIS15_BATCH_SIZE=10
METIS15_MAX_STEPS=20
METIS15_LR=1e-5
METIS15_OPTIMIZER=adamw
METIS15_LM_LOSS_IMPL=liger_fused_linear_ce
METIS15_PREFETCH_BATCHES=1
```

But the observed stable real-data setting is batch 8:

```bash
cd /opt/dlami/nvme/metis
METIS15_BATCH_SIZE=8 METIS15_MAX_STEPS=50 ./scripts/metis15_h100_benchmark.sh
```

## Environment Details

### H100 Instance

```text
Public IP: 18.183.61.208
SSH user: ubuntu
Instance hostname: ip-172-31-8-8
GPU: NVIDIA H100 80GB HBM3
GPU memory: 81559 MiB
Disk root for Metis: /opt/dlami/nvme/metis
```

### Docker Images Present

```text
metis15-h100:torch280-te215-torchgrouped
metis15-h100-bench:torch280-te-source
metis15-h100-bench:torch280-te-pypi
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
nvidia/cuda:13.0.2-base-ubuntu24.04
```

### Working Runtime Stack

From image inspection:

```text
torch 2.8.0+cu128
CUDA 12.8
Transformer Engine 2.15.0+42b8400
triton available
liger_kernel available
torch._grouped_mm available
torch._scaled_grouped_mm available
```

The image did not have a Python cuDNN Frontend or CUTLASS integration ready to call directly from the Metis model. The practical backend available immediately was PyTorch's private grouped GEMM op.

## Exact Tests And Results

### 1. `torch._grouped_mm` API Probe

Confirmed operator schemas:

```text
aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
aten::_scaled_grouped_mm(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? offs=None, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False) -> Tensor
```

Findings:

- Offsets must be `int32`, not `int64`.
- Offsets are cumulative end offsets of shape `[num_groups]`, not `[num_groups + 1]`.
- Weight layout that works is `[G, K, N]`.
- Empty groups are supported if repeated offsets are used.
- Autograd works for input and weight.
- Bias is not differentiable for `_grouped_mm`, but Metis expert MLP bias is false in the manifest, so this is fine.

Minimal working shape test:

```text
x:      [M, K] BF16 CUDA
weight: [G, K, N] BF16 CUDA
offs:   [G] int32 CUDA
y:      [M, N] BF16 CUDA
```

### 2. Expert Module Smoke

Module:

```python
MetisGroupedHeadExperts(
    num_experts=32,
    head_dim=384,
    intermediate_size=1280,
    bias=False,
    use_fp8=True,
    precision_kwargs={"force_bf16": True, "low_precision_allowed": False},
    backend="torch_grouped",
)
```

Result:

```text
use_fp8 False fwd 0.0736s, bwd 0.0153s
use_fp8 True  fwd 0.0334s, bwd 0.0010s
expert_module_ok
```

Standalone expert module gradients were finite:

```text
loss 0.003346
xgrad finite True
gate_up_proj.impl.weight grad finite True
down_proj.impl.weight grad finite True
```

Interpretation: the grouped GEMM op itself can forward/backward cleanly on expert-shaped inputs.

### 3. Tiny Model Smoke

One-layer tiny Metis model with `torch_grouped`, bucketed dispatch:

```text
fwd 2.2357s loss 8.6875
bwd 0.8211s
torch_grouped_model_smoke_ok
```

Later reduced tests with a smaller block confirmed finite loss, finite grads, and finite params over multiple optimizer steps.

### 4. First Full Training Test: Global FP8 + BF16 Experts

Command shape:

```bash
python scripts/train_mamba_lm.py \
  --data-dir data/metis15_h100_synth \
  --batch-size 8 \
  --grad-accum-steps 1 \
  --max-steps 8 \
  --dtype bf16 \
  --fp8 \
  --fp8-expert-precision bf16 \
  --moe-backend torch_grouped \
  --moe-dispatch-mode bucketed \
  --lm-loss-impl liger_fused_linear_ce \
  --optimizer muon_adamw \
  --hybrid-adamw-impl foreach \
  --fused-adamw \
  --prefetch-batches 2 \
  --matmul-precision high \
  --tf32
```

Result:

```text
step 1: train 10.7108, tok/s 1,464
step 2: train nan, tok/s 31,499
```

Then process stalled and had to be killed.

Interpretation: global FP8 is not stable in this H100 full-model training path.

### 5. TE GroupedLinear BF16 Baseline

Real H100, same general training loop, BF16, `te_grouped`, bucketed dispatch, batch 8:

```text
step 1: tok/s 1,027
step 2: tok/s 2,080
step 3: tok/s 2,213
step 4: tok/s 2,366
```

Interpretation: TE `GroupedLinear` path is catastrophically slow on H100 in this setup, even in BF16.

### 6. New `torch_grouped` BF16, Synthetic Data, Batch 8

Stable six-step synthetic run:

```text
step 1: tok/s 2,490
step 2: tok/s 33,777
step 3: tok/s 34,021
step 4: tok/s 34,044
step 5: tok/s 34,141
step 6: tok/s 33,970
```

No `nan`.

### 7. New `torch_grouped` BF16, Synthetic Data, Batch 10

Stable short synthetic run:

```text
step 1: tok/s 3,024
step 2: tok/s 34,695
step 3: tok/s 35,048
step 4: tok/s 35,155
```

No `nan` in that 4-step synthetic test.

### 8. New `torch_grouped` BF16, Real Base Data, Batch 10

Real data path:

```text
/opt/dlami/nvme/metis/data/metis15_base
```

Result:

```text
step 1: train 10.7105, tok/s 3,026
step 2: train 10.6844, tok/s 33,964
step 3: train 10.6817, tok/s 34,633
step 4: train 10.6229, tok/s 34,537
step 5: train 10.6471, tok/s 34,539
step 6: train nan,     tok/s 34,038
```

Interpretation: batch 10 is slightly faster but not stable enough for a real run.

### 9. New `torch_grouped` BF16, Real Base Data, Batch 8

Real data, 10-step run:

```text
step 1:  tok/s 2,479
step 2:  tok/s 33,213
step 3:  tok/s 33,680
step 4:  tok/s 33,779
step 5:  tok/s 33,722
step 6:  tok/s 33,723
step 7:  tok/s 33,741
step 8:  tok/s 33,700
step 9:  tok/s 33,683
step 10: tok/s 33,698
```

No `nan`.

Real data, 50-step run:

```text
step 2:  tok/s 33,150
step 10: tok/s 33,533
step 20: tok/s 33,668
step 30: tok/s 33,826
step 40: tok/s 34,007
step 50: tok/s 33,883
```

No `nan`.

This is the current best stable H100 setting.

### 10. Global FP8 Isolation

FP8 eval-style repeated forward with no grad and small batch/block:

```text
iter 1 loss 10.6875 finite True
...
iter 7 loss 10.6875 finite True
```

But full model train-mode batch 8 / block 1024 with FP8:

```text
iter 1 loss nan finite False
bad_grad ['backbone.embed_tokens.weight']
```

Also FP8 with `lr=0` and standard CE:

```text
step 1 train 10.6875
step 2 train 10.6875
step 3 train nan
```

Interpretation: the FP8 failure is not only caused by optimizer update magnitude or Liger CE. Full training-mode FP8 can generate bad values even before a useful update. Suspects include TE FP8 autocast state/amax behavior, training-mode activation path, batch/sequence-size-dependent FP8 kernel behavior, or interaction with tied embedding/lm head gradients.

### 11. TE Fused SwiGLU In BF16

I temporarily allowed TE `ops.SwiGLU` even when global FP8 was disabled.

Smoke test:

```text
swiglu_module SwiGLU
loss 8.625
bf16_fused_swiglu_smoke_ok
```

Short training:

```text
step 1: tok/s 2,507
step 2: tok/s 39,307
step 3: tok/s 41,896
```

Then the process hung after step 3 with about 41GB still allocated. Because it was not stable, this change was reverted.

Interpretation: fused BF16 SwiGLU may be a speed win, but TE `ops.SwiGLU` is not safe enough in this model/training loop as-is.

### 12. `torch.compile`

Command included:

```bash
--compile --compile-mode reduce-overhead
```

Problems:

- Dynamo recompiled because perf counters mutate on modules:

```text
self.perf_counters['sdpa_calls'] == ...
```

- Liger fused CE graph-broke on `.item()`:

```text
target_mask.sum().item()
```

- Then run failed with CUDA graph/rotary cache overwrite:

```text
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
```

Interpretation: `torch.compile` is not currently viable. The model has mutable counters, cached rotary tensors, Liger graph breaks, and dynamic MoE routing. It needs a separate compile/graphability cleanup pass.

## Current Best Known H100 Launch

Use BF16, batch 8:

```bash
cd /opt/dlami/nvme/metis
METIS15_BATCH_SIZE=8 METIS15_MAX_STEPS=50 ./scripts/metis15_h100_benchmark.sh
```

Equivalent direct core settings:

```bash
python scripts/train_mamba_lm.py \
  --data-dir data/metis15_base \
  --manifest configs/metis15_manifest.json \
  --train-stage pretrain \
  --out-dir checkpoints/h100_sweeps/real_base_b8 \
  --device cuda \
  --batch-size 8 \
  --grad-accum-steps 1 \
  --max-steps 50 \
  --lr 1e-5 \
  --warmup-steps 1 \
  --weight-decay 0.1 \
  --eval-interval 1000000 \
  --eval-iters 2 \
  --log-interval 1 \
  --checkpoint-interval 1000000 \
  --skip-final-eval \
  --skip-final-checkpoint \
  --dtype bf16 \
  --fp8-expert-precision bf16 \
  --moe-backend torch_grouped \
  --moe-dispatch-mode bucketed \
  --lm-loss-impl liger_fused_linear_ce \
  --optimizer adamw \
  --fused-adamw \
  --prefetch-batches 1 \
  --matmul-precision high \
  --tf32
```

Current steady-state:

```text
~0.24 seconds/step
batch_size = 8
seq_len = 1024
tokens/step = 8192
throughput = ~33,500-34,000 tok/s
estimated total TFLOP/s = ~220-223
estimated active TFLOP/s = ~63
```

## Why This Is Still Bad

The backend replacement proved TE `GroupedLinear` was a real wall, but it did not produce the target-class speed. The likely reasons:

### 1. Expert GEMM Backend Is Better, But Not A Full Fused MoE Kernel

The current expert path is now roughly:

```text
Triton bucket dispatch
torch._grouped_mm FC1
unfused F.silu(gate) * up
torch._grouped_mm FC2
Triton reverse weighted combine
```

This is much better than TE `GroupedLinear`, but it is still not:

```text
fused dispatch + grouped FC1 + SwiGLU + FC2 + combine
```

It still launches separate kernels for dispatch, FC1, activation, FC2, combine, and all backward components.

### 2. SwiGLU Is Still Unfused In The Stable Path

Stable BF16 path uses:

```python
gate, up = gate_up.split(...)
hidden = F.silu(gate) * up
```

TE `ops.SwiGLU` looked faster but hung. So the stable path still has extra activation kernels and memory traffic.

### 3. Global FP8 Is Not Stable

The target math assumed H100 FP8 should be a big win. In this actual full training loop, global FP8 is currently unusable:

- `nan` by step 2-3 in training loop.
- `nan` immediately in a full-shape train-mode stress test.
- Not explained away by optimizer LR or Liger alone.

This keeps the stable H100 lane in BF16, which caps attainable throughput.

### 4. Batch Size Is Constrained By Stability/OOM

Batch 8 is stable.

Batch 10:

- Slightly faster.
- Real data went `nan` around step 6.

Batch 12:

- OOM or `nan` depending on concurrent container/memory state.

This is an 80GB H100. The previous RTX PRO 6000 had 96GB and could test larger batch sizes despite being slower per kernel. The H100 has much better compute/memory bandwidth, but less VRAM.

### 5. `torch.compile`/CUDA Graphs Are Blocked

Launch overhead is probably still significant, but compile/graphs are not currently usable because of:

- mutable perf counters,
- dynamic MoE routing,
- Liger `.item()` graph break,
- rotary cache CUDAGraph overwrite,
- likely other dynamic routing breaks after those are fixed.

### 6. Optimizer State Is Heavy

For full model AdamW:

```text
1.095B params
AdamW states are large
batch 10/12 stability and OOM behavior may be partly memory-pressure related
```

Muon-AdamW hybrid was tested in FP8 early but not established as a stable high-throughput H100 path in BF16 during this pass. The stable H100 wrapper currently uses AdamW because the goal was to isolate the backend and avoid extra optimizer variables while debugging `nan`.

## Suspicious / Important Failure Modes

### FP8 Failure

Most important unresolved issue.

Evidence:

- Small no-grad FP8 forward is fine.
- Full train-mode FP8 at production batch/sequence can produce `nan` immediately.
- FP8 with `lr=0` still goes `nan` by step 3.
- BF16 with the same `torch_grouped` expert backend is stable at batch 8 for 50 real-data steps.

Research questions:

- Is TE FP8 autocast being applied around tied embedding/lm head in a way that corrupts gradients or activations?
- Are FP8 amax/scale histories invalid across the multi-layer, multi-expert path?
- Does `is_first_microbatch` handling interact badly with grad accumulation = 1 or with TE weight caching?
- Does Liger CE interact with FP8 hidden states or tied lm head in training mode?
- Is a specific module first producing non-finite outputs under full shape? Need hooks with full-shape train forward, possibly layer-by-layer.
- Does disabling FP8 for embeddings/lm head/attention while keeping selected linears FP8 stabilize?

### Batch 10 BF16 Failure

Evidence:

- Batch 10 synthetic 4-step run was stable.
- Batch 10 real data went `nan` around step 6.
- Batch 8 real data was stable for 50 steps.

Research questions:

- Is batch 10 simply exposing rare bad token windows?
- Is Liger fused CE producing NaNs for some targets/logits?
- Is optimizer update too aggressive with this initialization/data at batch 10?
- Is memory pressure causing some kernel/workspace behavior to change?
- Would `lr=3e-6`, no fused AdamW, or grad clipping changes stabilize batch 10?
- Would Muon-AdamW or BF16 optimizer state change behavior?

### TE `ops.SwiGLU` BF16 Hang

Evidence:

- Smoke test passed.
- Training reached ~42k tok/s by step 3.
- Process then hung with GPU memory still allocated.

Research questions:

- Is TE `ops.SwiGLU` unsafe outside TE FP8/autocast context?
- Does it have backward/autograd issues for this shape?
- Is it stream-syncing or deadlocking after repeated calls?
- Would a custom Triton SwiGLU kernel be safer than TE ops?

### `torch.compile` Failure

Evidence:

- Recompile limit hit because perf counters mutate.
- Liger CE breaks graph with `.item()`.
- Rotary cache fails with CUDAGraph output overwrite.

Research questions:

- Mark perf counter mutation disabled under compile?
- Clone rotary cache slices or precompute fixed position cos/sin outside graph?
- Disable Liger under compile or patch Liger `.item()`?
- Capture only MoE expert block instead of whole model?

## What The Research AI Should Not Conclude

Do not conclude that H100 is weak. The H100 is not the obvious issue.

Do not conclude that replacing TE `GroupedLinear` did nothing. It did a lot:

```text
TE grouped BF16:      ~2.3k tok/s
torch_grouped BF16: ~33.8k tok/s
```

Do not conclude that this is now fixed. The absolute throughput is still only about 14% of the 250k target.

Do not conclude that FP8 experts/global FP8 are bad in principle. The current TE/PyTorch training path is unstable; fused block-scaled MoE kernels might still be the right answer.

Do not trust batch 10 throughput as usable yet. It is slightly faster but unstable on real data.

Do not trust `torch.compile` without significant graphability cleanup.

## Most Likely Next Work

### Priority 1: Profile The New Stable Path

Run NSYS/NCU on the H100 stable setting:

```bash
cd /opt/dlami/nvme/metis
METIS15_BATCH_SIZE=8 METIS15_MAX_STEPS=20 ./scripts/metis15_h100_benchmark.sh
```

Profile questions:

- How many launches per step remain?
- How much time is in:
  - Triton bucket dispatch,
  - `_grouped_mm` FC1,
  - SiLU/mul,
  - `_grouped_mm` FC2,
  - reverse combine,
  - backward Wgrad/Dgrad,
  - attention,
  - Liger CE,
  - optimizer?
- Does `_grouped_mm` use efficient kernels for these shapes?
- Are cublasLt heuristic calls still happening?
- Are grouped GEMMs under-occupied because groups are too small/ragged?

### Priority 2: Stabilize FP8 Or Prove It Cannot Be Used With Current TE

Suggested ablations:

- BF16 everything except attention/projections FP8.
- FP8 without tied lm head.
- FP8 with standard CE and no Liger, full-shape train-mode.
- FP8 with `is_first_microbatch=None` always.
- FP8 with delayed scaling recipe variants:
  - history len,
  - margin,
  - E4M3 only,
  - HYBRID.
- Disable FP8 for lm head/embedding specifically.
- Hook every major layer output in full-shape train mode to find first non-finite tensor.

### Priority 3: Fused SwiGLU Without TE `ops.SwiGLU`

Stable path still pays unfused activation overhead.

A custom Triton BF16 SwiGLU kernel for `[sum_assignments, 2560] -> [sum_assignments, 1280]` may be a useful near-term step:

```text
gate_up: [M, 2560]
hidden:  [M, 1280]
hidden = silu(gate) * up
```

Need custom backward too for real training benefit, or rely on autograd if forward-only does not help enough.

### Priority 4: Real Fused MoE Kernel Path

The current path is still a stitched path, just much better stitched:

```text
dispatch kernel
grouped FC1 GEMM
SwiGLU kernels
grouped FC2 GEMM
combine kernel
separate backward pieces
```

The target path is closer to:

```text
device offsets
grouped FC1 + SwiGLU fusion
grouped FC2 + gating/score fusion
grouped Wgrad
reverse combine without atomics
static buffers / graphable shapes
```

Candidates:

- cuDNN Frontend grouped GEMM + SwiGLU / Wgrad APIs if usable on H100 or only Blackwell.
- CUTLASS grouped GEMM / custom extension.
- Triton persistent grouped GEMM plus custom SwiGLU/combine.
- DeepGEMM only if H100 support and integration are realistic.

### Priority 5: Make Static Capacity Safe

Static capacity was not used in the stable H100 runs. The bucketed dynamic count path still means dynamic offsets.

Potential path:

- Log max/p95/p99 counts per expert at batch 8 and 10.
- Pick static capacity with overflow fallback or no-drop guarantee.
- Preallocate buffers.
- Remove dynamic allocation/syncs.
- Then revisit CUDA graphs/module capture.

## Open Numerical Context

Current stable real-data H100:

```text
batch_size = 8
seq_len = 1024
tokens/step = 8192
step time = ~0.24s
throughput = ~33,500-34,000 tok/s
estimated active TFLOP/s = ~63
```

Goal:

```text
250,000 tok/s
```

Needed speedup over stable H100 result:

```text
250,000 / 33,800 ~= 7.4x
```

Batch 10 only gives:

```text
~34,500 tok/s
```

So batch size alone is not the answer. The remaining gap requires major kernel/path changes, not launch flag tuning.

## Final Current State

As of this handoff:

- H100 is set up.
- Docker image is built.
- Real data is hydrated.
- `torch_grouped` backend is implemented and tested.
- Stable H100 run exists: BF16, batch 8, real data, 50 steps, ~34k tok/s.
- H100 memory is free after tests.
- No training container was intentionally left running.

Current best command:

```bash
ssh -i ~/.ssh/aws_codex_builder.pem ubuntu@18.183.61.208
cd /opt/dlami/nvme/metis
METIS15_BATCH_SIZE=8 METIS15_MAX_STEPS=50 ./scripts/metis15_h100_benchmark.sh
```

The current implementation is a successful removal of one very bad wall, not a successful 250k tok/s solution.
