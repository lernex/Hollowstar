# Metis-1.5 8xA100 Expert-Parallel BF16 Training

Date: 2026-05-17

## Why This Exists

The single-H100 Metis-1.5 runs proved that dense data parallelism is the wrong shape for this model. Replicating all 32 routed experts on every GPU kept the run bottlenecked on MoE dispatch, optimizer state, and grouped-GEMM launch overhead. The 8xA100 lane now treats routed experts as sharded model state instead of replicated data-parallel state.

## Training Shape

- Hardware target: 8x NVIDIA A100 80GB.
- Precision: BF16 master weights and BF16 compute. FP8 and NVFP4 are disabled in the manifest.
- Expert parallel size: 8.
- Routed experts: 32 global, 4 local experts per rank.
- Shared expert: replicated on every rank.
- Router, attention, latent projections, embeddings, final norm, and LM head: replicated and gradient-synchronized globally.
- Routed expert parameters: rank-local, not DDP-reduced.

## Runtime Path

The model uses a native all-to-all expert-parallel dispatcher:

1. Every rank routes its local token batch against the full replicated router.
2. Top-k routed assignments are packed by owning expert rank.
3. BF16 latent token payloads move through `torch.distributed.all_to_all_single`.
4. Each rank runs only its 4 local routed experts with the existing grouped expert MLP backend.
5. With `moe_memory_efficient_permutation=true`, routing weights are sent to the owning expert rank and multiplied into the expert activation before the down projection. Expert outputs then move back through all-to-all and are summed on the source rank. This is the Megatron-style memory-efficient permutation identity for bias-free experts.
6. The training loop all-reduces only replicated parameter gradients; local expert shard gradients are scaled by world size but not averaged with unrelated expert shards.

The implementation is intentionally native PyTorch distributed instead of DeepSpeed MoE. That keeps the existing Metis single-latent router, aux-loss-free balance bias, grouped expert kernels, Muon-AdamW policy, and checkpoint format under repo control.

## Nemotron / Megatron Alignment

NVIDIA's current Nemotron 3 Super / Megatron-Core direction attacks MoE throughput with flex dispatchers, DeepEP or HybridEP, EP communication overlap, delayed expert weight-gradient compute, router and permutation fusion, grouped GEMM, memory-efficient permutation, distributed optimizer overlap, and selective activation memory tools.

What is now native in this repo:

- Expert parallel sharding across 8 ranks.
- Torch all-to-all token dispatch in latent space.
- Grouped expert GEMM through `torch_grouped_safe`.
- Fused local bucket dispatch when Triton kernels are available.
- Megatron-style memory-efficient permutation for the routed experts.
- Rank-sharded checkpoints plus merge for export.

What still requires a Megatron-Core/Bridge port:

- `flex` dispatcher with DeepEP or HybridEP.
- Batch-level EP all-to-all overlap.
- Delayed expert wgrad compute.
- Transformer Engine router fusion.
- Megatron distributed optimizer and Bridge distributed checkpointing.

The native trainer rejects those unsupported knobs if they are enabled, instead of silently pretending to use them. Use this probe on the target image before starting a port:

```bash
make metis15-megatron-profile
```

## Default Launch

```bash
make metis15-a100-pretrain
```

The manifest drives the launch as `WORLD_SIZE=8`, `moe_expert_parallel_size=8`, `torch_grouped_safe`, bucketed dispatch, BF16, and no low-precision Transformer Engine recipe.

Useful overrides:

```bash
METIS15_LOCAL_BATCH_SIZE=6 METIS15_GRAD_ACCUM_STEPS=10 make metis15-a100-pretrain
METIS15_MOE_BACKEND=torch_bmm make metis15-a100-pretrain
METIS_TORCH_GROUPED_SAFE_SYNC=1 make metis15-a100-pretrain
```

Use `torch_bmm` only as a correctness fallback; it is expected to be slower.

## Checkpointing

Expert-parallel checkpoints are sharded:

- Rank 0 writes `best.pt`, `latest.pt`, and gate checkpoints.
- Ranks 1..7 write `best.rank001.pt` through `best.rank007.pt`, and the matching `latest.rankNNN.pt` / gate files.

Resume on the same 8-rank topology loads each rank's own shard. Loading from an older full-expert checkpoint is supported: routed expert tensors are sliced by global expert id into each rank's local 4-expert shard.

For export or single-process inference, merge the rank shards into a full-expert checkpoint:

```bash
python scripts/merge_metis15_expert_parallel_checkpoint.py \
  --checkpoint checkpoints/metis15_base/best.pt \
  --out checkpoints/metis15_base/best.full.pt \
  --world-size 8
```

## Batch Defaults

Base pretrain now uses local batch 8 and grad accumulation 8:

```text
8 ranks * 8 local sequences * 8 accumulation * 1024 tokens = 524,288 tokens / optimizer step
```

Continued pretrain uses local batch 6 and grad accumulation 6:

```text
8 ranks * 6 local sequences * 6 accumulation * 1024 tokens = 294,912 tokens / optimizer step
```

These are launch defaults, not final proof. The first real A100 run should watch memory after optimizer state materializes, then sweep local batch before touching model architecture.
