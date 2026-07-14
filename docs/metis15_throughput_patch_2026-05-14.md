# Metis-1.5 Throughput Patch Notes - 2026-05-14

## What changed

- Canonical RTX PRO 6000 launch defaults now use delayed FP8 with BF16 expert weights instead of NVFP4-first launcher defaults.
- The default MoE route is `moe_dispatch_mode=bucketed`, keeping the Triton count/pack dispatcher in front of TE GroupedLinear.
- Bucketed dispatch now records reverse packed positions and can combine top-k expert outputs with `reverse_weighted_combine`, avoiding the old `index_add_`-style atomic combine.
- TE `ops.SwiGLU` is used when Transformer Engine is available, replacing the explicit `F.silu(gate) * up` expert activation.
- New experimental controls:
  - `--moe-backend {te_grouped,cudnnfe,triton,cutlass}`
  - `--moe-static-capacity N`
  - `--moe-overflow-mode {fallback,drop,error}`
  - `--moe-graphable`
  - `--disable-moe-fused-combine`

`cudnnfe`, `triton`, and `cutlass` are exposed as rejected backend choices for now rather than silently mapped to TE. The only working grouped-GEMM backend in this checkout is still `te_grouped`.

## Remote validation

Instance:

- `54.144.73.7`
- RTX PRO 6000 Blackwell Server Edition, SM120
- Docker image: `metis15-blackwell-fp8-ngc2604:sm120a`

Checks run:

- Local syntax: `python3 -m py_compile`, `bash -n`, `python3 -m json.tool`
- Remote syntax in `/opt/dlami/nvme/metis/10M-model`
- Container import check: Torch 2.12.0a0 / CUDA 13.2 / Triton 3.6 / Transformer Engine GroupedLinear present
- Triton bucket dispatcher + reverse combine CUDA correctness smoke
- `scripts/smoke_metis15_training_contracts.py --device cuda --check-liger`
- `MetisGroupedHeadExperts` forward/backward smoke with TE `SwiGLU`

## Throughput observed

Short real-data runs on `/opt/dlami/nvme/metis/data/metis15_base`:

| Run | Final logged tok/s | Notes |
| --- | ---: | --- |
| b16 g12, bucketed + reverse combine before TE SwiGLU | 29.7k | Stable, only small gain over old bucketed family |
| b16 g12, old atomic combine | 29.6k | Confirms combine is not the main remaining wall |
| b16 g12, reverse combine + TE SwiGLU | 37.2k | Main observed gain |
| b18 g11, reverse combine + TE SwiGLU | 37.0k | Similar to b16/g12 |

The patch improves the current family, but it does not make 180k tok/s plausible with TE GroupedLinear still in the hot path.

## Static capacity warning

The `--moe-graphable` path intentionally sets `overflow=drop` to avoid host overflow checks. A tiny smoke with `batch=2`, `--moe-static-capacity 1152`, and drop reached step 2 but produced `nan`, and a checked `--moe-static-capacity 2048 --moe-overflow-mode error` overflowed during a real model forward.

Do not use graphable/drop for a real training run until we have a measured per-layer count envelope after the router has balanced, or a real overflow slow path that does not sync every layer.

## Next blocker

The remaining bottleneck is still the grouped expert GEMM backend. NVIDIA's cuDNN Frontend grouped GEMM + SwiGLU APIs use device-side padded offsets for Blackwell MoE, which is the right shape of solution. The current Python environment has `cudnn` and `cutlass`, but no ready `cudnn_frontend` Python package and no Metis extension wrapping those kernels yet.

Next implementation pass should build a real expert backend, not another TE flag sweep:

1. Add a C++/CUDA or pybind extension that exposes Blackwell grouped GEMM + SwiGLU / grouped GEMM down-proj for the exact Metis expert shapes.
2. Use device-side offsets/padded offsets and persistent work buffers.
3. Add backward kernels or bind cuDNN FE grouped Wgrad/dgrad paths.
4. Re-enable static capacity only after overflow behavior is measured and safe.
