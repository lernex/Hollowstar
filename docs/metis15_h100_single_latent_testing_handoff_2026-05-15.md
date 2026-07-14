# Metis-1.5 H100 Single-Latent MoE Testing Handoff

Date: 2026-05-15  
Machine: AWS Ubuntu host `13.206.201.56`  
GPU: 1x NVIDIA H100 80GB HBM3, driver `580.126.16`  
Remote repo: `/opt/dlami/nvme/metis`  
Image: `lernex/metis-gpu:metis15-h100-single-latent-v1`  
Image digest/id observed: `sha256:3730e92f945044af245132dc0a57e5f28d806e813801b8d7df9346eb375253f9`

## Executive Summary

The new single-latent MoE Metis-1.5 path is training on H100 in BF16, but the first stable baseline is extremely slow: the best short smoke observed was about `8.2k tok/s` at batch size 4 after warmup. This is a stability floor, not an optimized result.

Critical findings:

- `torch_grouped` / `torch._grouped_mm` is not usable as the H100 launch backend right now. Forward can complete, but backward stalls indefinitely.
- `te_grouped` completes forward/backward and can train real data in BF16.
- Batch size 4 is stable for 20 real-data steps.
- Batch size 6 is stable for 10 real-data steps, but short-run throughput was not clearly better than batch size 4.
- Batch size 8 goes non-finite after the first optimizer update, even at `lr=1e-6`.
- FP8 was not tested after this reset. Current priority should be BF16 stability and throughput first.

## Current Architecture Under Test

From `configs/metis15_manifest.json`:

- Model name: `Metis-1.5`
- Model type: `metis_single_latent_moe`
- FFN type: `single_latent_moe`
- Context window: `1024` tokens
- Layers: `19`
- `d_model`: `1536`
- Experts: `32`
- Top-k: `4`
- Shared experts: `1`
- MoE heads: `1`
- Router latent size: `512`
- Routed latent size: `512`
- Expert hidden size: `1024`
- MoE activation: `squared_relu`
- Attention backend: `sdpa`
- Native GQA attention: enabled at runtime
- Training precision: BF16
- LM loss: `liger_fused_linear_ce`
- Current stable MoE backend: `te_grouped`
- Current dispatch mode: `bucketed`

Compute audit:

```text
config_estimate_params: 897,428,576
config_estimate_active_params_depth1: 339,586,144
attention_apps_per_layer: 6,291,456
latent_dim: 512
num_experts: 32
top_k: 4
routing_units_per_token: 4
expert_hidden: 1,024
expert_param_apps_per_assignment: 1,048,576
routed_expert_param_apps_per_layer: 4,194,304
shared_expert_param_apps_per_layer: 3,145,728
latent_projection_apps_per_layer: 1,572,864
router_projection_and_match_apps_per_layer: 16,416
rough_total_param_apps_per_token: 339,526,240
estimated_train_flops_per_token: 2,037,157,440
```

The script prints target rates for a hypothetical 450 TFLOP/s effective training path:

```text
total-param target: 83,572 tok/s
active-param target: 220,857 tok/s
```

Current stable observed throughput is far below either target.

## Environment And Data

Remote data was hydrated from S3:

```text
/opt/dlami/nvme/metis/data/metis15_base/train.bin 100000000000 bytes
/opt/dlami/nvme/metis/data/metis15_base/val.bin   1010101012 bytes
/opt/dlami/nvme/metis/data/metis15_base/meta.json  92869 bytes
```

Tokenizer assets were pulled to:

```text
/opt/dlami/nvme/metis/artifacts/metis15_tokenizer
```

Container smoke passed:

```text
torch 2.8.0+cu128
cuda 12.8
device NVIDIA H100 80GB HBM3
grouped_mm True
transformer_engine 2.15.0+42b8400
standard_ce_next_token_ok loss=4.844694 two_token_loss=4.865734
single_latent_moe_training_ok loss=4.208083 assignments=640 rough_param_apps=49,288
```

Docker/containerd had to be moved to NVMe. Root disk is now healthy:

```text
/dev/root                        29G   14G   15G  48% /
/dev/mapper/vg.01-lv_ephemeral  3.5T  125G  3.2T   4% /opt/dlami/nvme
```

## Backend Findings

### `torch_grouped` Failure

The intended fast backend was `torch_grouped`, using PyTorch `torch._grouped_mm`. It is present in this runtime:

```text
grouped_mm aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
```

But it is not usable for training this model on the current H100 runtime.

Observed failure:

- A full training smoke using `torch_grouped` initialized correctly.
- It printed model setup and compute audit.
- It never emitted step 1.
- GPU stayed at ~100% utilization with only ~4.2GB allocated.
- A smaller debug run with `block_size=256`, batch 1, max step 1 reproduced the stall.
- A surgical `block_size=32` probe showed forward could complete, but backward did not return and had to be killed.

Interpretation: the likely failure is in the backward path for the grouped expert GEMM route, not data loading, model construction, attention, or LM loss.

### `te_grouped` Success

Switching the exact same single-latent model to Transformer Engine grouped GEMM completed forward and backward.

Surgical `block_size=32` probe:

```text
backend: te_grouped
forward done: ~1.74s
backward done: ~0.88s
```

This is why the current H100 baseline was changed to:

```text
--moe-backend te_grouped
--moe-dispatch-mode bucketed
```

## Real-Data BF16 Results

All real-data runs used:

```text
dtype: bf16
fp8-expert-precision: bf16
moe-backend: te_grouped
moe-dispatch-mode: bucketed
lm-loss-impl: liger_fused_linear_ce
attention: sdpa/native GQA
optimizer: AdamW fused
prefetch-batches: 1
context: 1024
```

### Batch 1

Command shape:

```text
METIS15_BATCH_SIZE=1
METIS15_MAX_STEPS=3
METIS15_LR=1e-5
```

Result: stable but slow.

```text
step 1 train 10.6333 tok/s 241
step 2 train 10.7322 tok/s 990
step 3 train 10.6875 tok/s 1,156
```

### Batch 4

Short run:

```text
METIS15_BATCH_SIZE=4
METIS15_MAX_STEPS=5
METIS15_LR=1e-5
```

Result: stable.

```text
step 1 train 10.7242 tok/s 715
step 2 train 10.6326 tok/s 1,926
step 3 train 10.6490 tok/s 2,548
step 4 train 10.5694 tok/s 2,849
step 5 train 10.5669 tok/s 3,256
```

Longer smoke:

```text
METIS15_BATCH_SIZE=4
METIS15_MAX_STEPS=20
METIS15_LR=1e-5
```

Result: stable for all 20 steps.

```text
step  2 train 10.6784 tok/s 1,051
step  4 train 10.6025 tok/s 2,537
step  6 train 10.5572 tok/s 3,553
step  8 train 10.4059 tok/s 4,361
step 10 train 10.4886 tok/s 4,636
step 12 train 10.3754 tok/s 5,573
step 14 train 10.4119 tok/s 6,426
step 16 train 10.4047 tok/s 6,867
step 18 train 10.4083 tok/s 7,946
step 20 train 10.3759 tok/s 8,187
```

This is the current safe default.

### Batch 6

Command shape:

```text
METIS15_BATCH_SIZE=6
METIS15_MAX_STEPS=10
METIS15_LR=1e-5
```

Result: stable for 10 steps, but not clearly better than batch 4 in the short run.

```text
step  1 train 10.7114 tok/s 970
step  2 train 10.6659 tok/s 2,609
step  3 train 10.5987 tok/s 3,232
step  4 train 10.6201 tok/s 3,443
step  5 train 10.4885 tok/s 3,969
step  6 train 10.5194 tok/s 4,248
step  7 train 10.5381 tok/s 4,909
step  8 train 10.4833 tok/s 5,066
step  9 train 10.4937 tok/s 6,087
step 10 train 10.5542 tok/s 5,830
```

### Batch 8

Command shape:

```text
METIS15_BATCH_SIZE=8
METIS15_MAX_STEPS=5
METIS15_LR=1e-5
```

Result: non-finite after first optimizer step.

```text
step 1 train 10.7044 moe_aux 0.0004 tok/s 1,241
step 2 train nan     moe_aux nan    tok/s 3,220
step 3 train nan     moe_aux nan    tok/s 46,329
step 4 train nan     moe_aux nan    tok/s 46,853
step 5 train nan     moe_aux nan    tok/s 46,959
```

The very high post-NaN tok/s is invalid throughput. Once NaN appears, the run is poisoned.

Lowering LR did not fix it:

```text
METIS15_BATCH_SIZE=8
METIS15_MAX_STEPS=5
METIS15_LR=1e-6
```

Result:

```text
step 1 train 10.7042 moe_aux 0.0004 tok/s 1,253
step 2 train nan     moe_aux nan    tok/s 3,107
step 3 train nan     moe_aux nan    tok/s 46,318
step 4 train nan     moe_aux nan    tok/s 47,128
step 5 train nan     moe_aux nan    tok/s 47,085
```

Interpretation: batch 8 instability is probably not a simple LR magnitude issue. It may be a gradient/optimizer-state corruption, grouped expert backward edge case, expert-load issue, or MoE aux/router numerical issue triggered by larger assignment count.

## Code Changes Made Because Of Testing

Current H100 wrapper defaults were changed:

```text
scripts/metis15_h100_benchmark.sh
BATCH_SIZE="${METIS15_BATCH_SIZE:-4}"
--moe-backend te_grouped
```

A fail-fast guard was added to training so future NaN runs stop immediately instead of continuing to print fake throughput:

```python
if not torch.isfinite(loss.detach()):
    raise FloatingPointError(
        f"Non-finite loss at step {step}, micro_step {micro_step}: {loss.detach().float().item()}"
    )
```

The rebuilt image contains these defaults.

## Strong Suspicions / Optimization Leads

1. `torch_grouped` backward needs to be avoided or isolated.

   It is tempting because it should be a lower-overhead grouped GEMM path, but the current H100 runtime stalls in backward. Do not use it for training until a minimal reproducer proves it is fixed.

2. `te_grouped` is stable but likely has too much dispatch/GEMM overhead at this shape.

   The model is doing 19 MoE grouped dispatches per step, and with batch 4 each 2-step log interval showed:

   ```text
   moe_grouped 38
   assign 622592
   ```

   For one step at batch 4:

   ```text
   assignments = 4 batch * 1024 tokens * top_k 4 * 19 layers = 311,296
   ```

   That contract is correct, but the backend is not turning it into H100-class utilization.

3. Batch 8 NaN after first update needs instrumentation before any larger batch sweep.

   Suggested checks:

   - Detect non-finite gradients before `clip_grad_norm_`.
   - Detect non-finite parameters immediately after `optimizer.step()`.
   - Log which module first becomes non-finite.
   - Compare fused AdamW vs non-fused AdamW.
   - Compare Liger fused CE vs standard CE.
   - Disable MoE aux loss temporarily to see whether `moe_aux` is source or symptom.
   - Freeze router or shared expert for a tiny run to localize corruption.
   - Test batch 8 with grad accumulation equivalent to batch 4 microbatches, e.g. microbatch 4, grad accum 2.

4. Batch size 6 may deserve a longer run, but batch 4 is the current safe default.

   Batch 6 did not NaN in 10 steps, but the short-run throughput was noisy and not clearly better than batch 4. A 100-step run with stable LR and no cosine-to-zero artifact would be more meaningful.

5. The current benchmarks are too short for final throughput claims.

   The 20-step batch 4 run warmed from `1k tok/s` to `8.2k tok/s`. This suggests one-time overheads, cache effects, optimizer state materialization, or CUDA/TE warmup are heavily distorting short tests. Future sweeps should use:

   - Warmup steps ignored in throughput.
   - Fixed LR schedule that does not decay to zero over the entire smoke.
   - At least 100 measured steps once stability is established.
   - Median and p95 step time, not final line only.

## Recommended Next Experiments

Run these in order.

### 1. Add Non-Finite Gradient/Parameter Localization

Goal: explain batch 8 NaN.

Instrumentation points:

- After backward, before grad clipping.
- After grad clipping.
- After optimizer step.
- Include module/parameter name and max absolute value.

Expected result:

- If gradients are finite but params become NaN, inspect fused AdamW and optimizer state.
- If gradients are NaN before step, inspect MoE aux/router/expert activations.
- If only MoE aux is NaN but LM loss is finite before update, router/balance math is likely source.

### 2. Batch 8 With Microbatch 4 x Grad Accum 2

Goal: separate effective batch size from per-microbatch kernel shape.

Command idea:

```bash
METIS15_BATCH_SIZE=4 \
METIS15_GRAD_ACCUM=2 \
METIS15_MAX_STEPS=20 \
METIS15_LR=1e-5 \
METIS_ASYNC_METRICS=0 \
bash scripts/metis15_h100_benchmark.sh
```

If this is stable, batch 8 failure is per-microbatch shape/load, not effective batch.

### 3. Fused AdamW Off

Goal: check whether fused optimizer is corrupting state at batch 8.

Need wrapper option or manual command without:

```text
--fused-adamw
```

### 4. Standard CE Instead Of Liger

Goal: verify Liger fused CE is not involved in NaN.

Run:

```bash
METIS15_LM_LOSS_IMPL=standard ...
```

This may be slower but should clarify stability.

### 5. Static Capacity / Padding Sweep For TE Grouped

Goal: reduce grouped GEMM shape churn and improve TE grouped performance.

Potential knobs already exposed:

```text
--moe-static-capacity
--moe-capacity-factor
--moe-capacity-alignment
--moe-overflow-mode
--disable-moe-fused-combine
```

Try capacity aligned to stable per-expert loads for batch 4 and batch 6. Watch for overflow, padding overhead, and throughput.

### 6. Longer Batch 4 And Batch 6 Throughput Runs

Goal: get real steady-state numbers.

Use 100+ steps, log every 10 steps, ignore warmup. Keep BF16 only.

## Current Safe Launch Command

```bash
cd /opt/dlami/nvme/metis

METIS15_REPO_DIR=/opt/dlami/nvme/metis \
METIS15_DATA_DIR=data/metis15_base \
METIS15_OUT_DIR=checkpoints/h100_single_latent_bf16_safe \
METIS15_BATCH_SIZE=4 \
METIS15_GRAD_ACCUM=1 \
METIS15_MAX_STEPS=100 \
METIS15_LOG_INTERVAL=10 \
METIS15_LR=1e-5 \
METIS15_WARMUP_STEPS=1 \
METIS15_PREFETCH_BATCHES=1 \
METIS15_OPTIMIZER=adamw \
METIS_ASYNC_METRICS=0 \
bash scripts/metis15_h100_benchmark.sh
```

Current wrapper defaults already use batch 4 and `te_grouped`.

## Do Not Trust

Do not trust any throughput line after the first NaN. The `46k-47k tok/s` lines in the batch 8 logs are invalid because the model had already gone non-finite.

Do not assume `torch._grouped_mm` is good just because import/runtime detection passes. The H100 failure is in actual training backward behavior.

Do not begin FP8 work until BF16 has a stable and instrumented throughput baseline.

