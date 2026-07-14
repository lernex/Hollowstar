# Metis-1.5 H100 Torch Grouped Safe Follow-up

Date: 2026-05-15  
Machine: AWS Ubuntu host `13.206.201.56`  
GPU: 1x NVIDIA H100 80GB HBM3  
Remote repo: `/opt/dlami/nvme/metis`  
Image tag under test: `lernex/metis-gpu:metis15-h100-single-latent-v1`

## Summary

The original `~8.2k tok/s` single-latent result was not an architecture limit. It was the slow `te_grouped` fallback. After implementing a guarded native grouped path, the best stable H100 BF16 run is now:

```text
backend: torch_grouped_safe
loss: standard CE
batch: 8
context: 1024
steps: 100
throughput after warmup: ~30.7k-31.1k tok/s
```

That is roughly `3.7x` faster than the `te_grouped` fallback, but still below the optimistic `35k-60k` target. The remaining bottleneck is not the single-latent design; it is the native grouped-GEMM stability workaround stack.

## Implemented Changes

- Added `moe_backend=torch_grouped_safe`.
- Added `moe_torch_grouped_min_m`, defaulting the H100 path to `8`.
- Added no-empty expert padding before `torch._grouped_mm`.
- Wrapped `_grouped_mm` calls with autocast disabled and explicit input casts.
- Added padded dummy-row masking inside grouped experts.
- Added grouped-output `nan_to_num` plus conservative `[-64, 64]` activation clamp for the `torch_grouped_safe` path.
- Added squared-ReLU guards for shared and grouped experts.
- Added `METIS_TORCH_GROUPED_SAFE_SYNC`, default `1`, for the safe backend.
- Added `torch_bmm` fallback backend.
- Added `torch_looped` diagnostic backend. It is not useful for launch.
- Added non-finite debug hooks:
  - scalar loss component checks
  - final hidden/logit checks
  - grad/param/optimizer finite checks
  - optional forward module hooks
  - optional top-k tensor absmax logging
- Changed Metis-1.5 H100/pretrain/full defaults to `standard` CE for now.
- Added `--retain-standard-ce-logits`; H100 benchmark/pretrain/full default it on via `METIS15_RETAIN_STANDARD_CE_LOGITS=1`.

## Source Checks

The report's PyTorch diagnosis matches the runtime behavior:

- PyTorch issue `#152668`: BF16 grouped GEMM backward can hang with empty/repeated offsets. The no-empty padding is directly based on this.  
  https://github.com/pytorch/pytorch/issues/152668
- PyTorch issue `#174763`: `_grouped_mm` is not autocast-compatible. The implementation now disables autocast and casts explicitly.  
  https://github.com/pytorch/pytorch/issues/174763
- Runtime schema confirmed in the container:

```text
torch 2.8.0+cu128
aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor
```

## Compute Audit

Unchanged after the backend patch:

```text
config_estimate_params: 897,428,576
config_estimate_active_params_depth1: 339,586,144
rough_total_param_apps_per_token: 339,526,240
estimated_train_flops_per_token: 2,037,157,440
```

## Final Working H100 Baseline

Recommended current command shape:

```bash
METIS15_MOE_BACKEND=torch_grouped_safe \
METIS15_MOE_TORCH_GROUPED_MIN_M=8 \
METIS15_BATCH_SIZE=8 \
METIS15_MAX_STEPS=100 \
METIS15_LR=1e-5 \
METIS15_LM_LOSS_IMPL=standard \
METIS15_RETAIN_STANDARD_CE_LOGITS=1 \
METIS_TORCH_GROUPED_SAFE_SYNC=1 \
METIS_ASYNC_METRICS=0 \
bash scripts/metis15_h100_benchmark.sh
```

Observed:

```text
step  20 tok/s 30,428
step  40 tok/s 30,834
step  60 tok/s 31,057
step  80 tok/s 31,014
step 100 tok/s 30,737
```

Loss stayed finite through 100 steps:

```text
step 100 train 9.4625
```

## Batch Matrix

All runs below used BF16, `torch_grouped_safe`, min-M 8, standard CE, retained logits, fused AdamW, SDPA/native GQA.

```text
batch 4, grad_accum 1, 50 steps: stable, ~20.4k-21.0k tok/s after warmup
batch 4, grad_accum 2, 50 steps: stable, ~20.8k-21.4k tok/s after warmup
batch 6, grad_accum 1, 50 steps: stable, ~27.3k-27.5k tok/s after warmup
batch 8, grad_accum 1, 100 steps: stable, ~30.7k-31.1k tok/s after warmup
```

Batch 8 is the current best throughput/stability point.

## Important Failures

These are important because they explain why the final baseline looks a little weird.

### `torch_grouped_safe` Without Retained Standard-CE Logits

No-debug runs could fail even after grouped-output sanitization. A debug run with `return_logits=True` for standard CE passed. Making retained standard-CE logits explicit reproduced stability without the expensive debug scans.

Interpretation: this looks like a lifetime/async/memory-reuse sensitivity in the current stack, not a normal optimizer overshoot.

### Liger CE

Still fails:

```text
METIS15_LM_LOSS_IMPL=liger_fused_linear_ce
FloatingPointError: Non-finite loss at step 4
```

Do not use Liger CE for the next BF16 H100 run.

### Fused AdamW Off

Turning fused AdamW off did not fix instability during earlier localization and could fail sooner. Keep fused AdamW on for the current baseline.

### `min_m=16`

`torch_grouped_safe` with min-M 16 was slower and still failed in short tests. Keep min-M 8.

### `torch_bmm`

Stable but too slow:

```text
batch 8, 25 steps: ~8.6k-8.9k tok/s
```

It is a correctness fallback, not a launch backend.

### `torch_looped`

Not useful:

```text
batch 8: ~3.5k tok/s and later non-finite
```

## Diagnosis

The native `_grouped_mm` path is still suspicious. Deep hooks caught raw grouped expert gate-up outputs around `1e37` with row-sized non-finite counts. Some of those appear to be padded dummy rows, but the backend also showed shape/lifetime sensitivity beyond simple empty experts.

The working baseline therefore uses all of these together:

```text
no-empty expert padding
dummy row masking
grouped-output sanitize/clamp
squared-ReLU sanitize/clamp
sync after safe grouped expert GEMMs
standard CE
retained standard-CE logits
```

This should be treated as a guarded PyTorch grouped-GEMM baseline, not as proof that `_grouped_mm` is healthy.

## Next Recommendations

1. Run a longer 1k-step BF16 batch-8 soak with the exact final baseline.
2. If 1k steps pass, profile the sync and retained-logit overhead separately.
3. Install/test FlashAttention-3 only after the MoE baseline survives the longer soak.
4. Do not test FP8 yet.
5. For real speed beyond ~31k tok/s, the next backend should be a real fused/segmented MoE kernel, not TE GroupedLinear and not Python-looped experts.
