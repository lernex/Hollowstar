# Metis-1.5 Nemotron / Megatron MoE Optimization Notes

Date: 2026-05-17

## Sources Checked

- NVIDIA Megatron-Bridge MoE Training Optimization.
- NVIDIA Megatron-Core MoE package docs.
- NVIDIA Megatron-Bridge EP overlap skill.
- NVIDIA Megatron-Bridge MoE optimization workflow skill.
- NVIDIA Nemotron 3 Super pretraining guide.
- NVIDIA Nemotron 3 Super research page and technical report link.

## What Nemotron / Megatron Does That Matters Here

The relevant optimization stack is not just "use DeepSpeed." Nemotron 3 Super is trained through Megatron-Bridge/Megatron-Core, and NVIDIA's current MoE guidance clusters the wins into:

- Expert parallelism first, with EP kept inside the fastest interconnect domain.
- Flex token dispatch with DeepEP on H100/B200-style nodes or HybridEP on GB200/GB300 NVLink domains.
- EP all-to-all overlap plus optional delayed expert wgrad compute.
- Grouped GEMM whenever each rank owns more than one local expert.
- Router fusion and permutation fusion to cut launch and token-movement overhead.
- Memory-efficient permutation: move routing weights into the activation before expert `down_proj`, avoiding saved weighted expert outputs for router backward.
- Distributed optimizer, grad-reduce overlap, and param-gather overlap for replicated state.
- Selective recompute/offload only after the simpler memory tools are exhausted.

## What Landed In Native Metis

- Native 8-way expert parallel remains the correctness bring-up path.
- `moe_memory_efficient_permutation` is now a first-class config/CLI option and is enabled in the Metis-1.5 manifest.
- The routed expert path passes routing weights into local grouped experts and does an unweighted combine afterward.
- The same identity works in the expert-parallel path by all-to-alling routing weights to owner ranks with autograd support.
- Startup logs and the compute audit now expose the MoE optimization profile.
- Unsupported Megatron-only knobs fail fast instead of silently pretending to work.
- `scripts/metis15_megatron_super_profile.py` prints the native-vs-Megatron profile and probes target image dependencies.

## What Did Not Land Natively

DeepEP, HybridEP, EP overlap, delayed expert wgrad, router fusion, and Megatron's distributed optimizer are framework-level pieces. They require moving the model definition/training loop into Megatron-Core/Bridge or writing a custom compatibility layer against those internals. The native Metis trainer now makes that boundary explicit.

## Recommended Next Target-Box Sequence

1. Run `make metis15-a100-pretrain` for a short 8xA100 correctness/profile smoke.
2. Run `make metis15-megatron-profile` on the same image to check Megatron/DeepEP/HybridEP availability.
3. If all-to-all dominates, port the model to Megatron-Bridge with `moe_token_dispatcher_type=flex` and `moe_flex_dispatcher_backend=deepep`.
4. If expert GEMM dominates, sweep `torch_grouped_safe` vs `torch_grouped` vs `torch_bmm`, then decide whether the Megatron grouped-GEMM path is worth the port before custom kernels.
5. If activation memory dominates, keep memory-efficient permutation enabled before reaching for full recompute.
