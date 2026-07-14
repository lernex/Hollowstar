# Metis-1.5 JAX TPU v6e Rewrite

This is the concrete replacement direction for the TPU training lane. The old Trainium and
PyTorch/XLA paths are retained only as historical/debug references while the JAX lane proves out.

## Target Contract

- Hardware: one Google Cloud TPU `v6e-8` host, eight chips.
- Runtime: JAX/libTPU with fixed-shape `jit`/SPMD arrays, not PyTorch/XLA or Neuron.
- Model: Metis-1.5 single LatentMoE decoder.
- Experts: 32 routed experts, top-4 routing, 4 experts per TPU chip.
- Latent path: 1536d hidden -> 512d latent -> routed squared-ReLU experts -> 1536d hidden.
- Shared path: one BF16 full-dim shared expert per layer.
- Base PT: `static_dense_pretrain`, MoR disabled.
- CPT: `dynamic_token_mor` with `static_packed_hard` compute, so every token gets depth 1 and only hard-routed active tokens enter fixed-capacity packed buffers for recursive depth 2/3 passes.
- Optimizer: Muon/AdamW hybrid. Routed experts, routers, embeddings, norms, and biases stay on AdamW. Attention, shared expert matrices, and latent down/up projections use Muon.

## Research Decisions

Primary-source checks:

- Google Cloud’s v6e training guide presents MaxText as the JAX LLM training example for v6e and recommends the JAX TPU stack, XPK/TPU VM launch flow, profiling, and TPU network/memory tuning. Source: `https://cloud.google.com/tpu/docs/v6e-training`.
- Google’s JAX AI stack guidance frames MaxText/MaxDiffusion as reusable JAX TPU production stacks with Orbax, Optax, Qwix/Tunix, goodput monitoring, and MFU-oriented model code. Source: `https://docs.cloud.google.com/tpu/docs/jax-ai-stack`.
- JAX `shard_map` is an SPMD API where each function instance receives explicit shards and uses explicit collectives. This is the right mental model for expert-parallel Metis once the single-host proof is stable. Source: `https://docs.jax.dev/en/latest/notebooks/shard_map.html`.
- JAX `jax.Array` sharding and `PartitionSpec` make array layout part of the value, which is the contract we want instead of relying on Python dispatch becoming graphable after the fact. Source: `https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html`.
- Orbax is the JAX-native checkpoint direction for PyTrees of `jax.Array`s, including large sharded arrays; Metis uses it as the default checkpoint backend and keeps NPZ only as a local fallback. Source: `https://orbax.readthedocs.io/en/stable/guides/checkpoint/checkpointing_pytrees.html`.
- NVIDIA’s LatentMoE reports and Nemotron 3 Super material motivate Metis’s design choice: route in a compressed latent width to lower routing/all-to-all cost and reinvest savings in expert diversity/top-k. Sources: `https://research.nvidia.com/labs/nemotron/LatentMoE/`, `https://arxiv.org/abs/2601.18089`, `https://arxiv.org/abs/2604.12374`.
- NVIDIA’s Nemotron/Megatron material still matters as a caution: efficient MoE is about dispatcher/combine layout and memory movement, not just FLOPs. Metis’s JAX lane should avoid the old ragged Python dispatcher and later graduate from local sort-pack to explicit `shard_map` expert shards.
- Muon is a matrix-only optimizer for hidden-layer 2D parameters using momentum plus Newton-Schulz orthogonalization. That supports Metis’s current policy: do not put embeddings, norms, routers, or routed expert tensors on Muon by default. Source: `https://kellerjordan.github.io/posts/muon/`.
- Kimi/Moonshot’s MuonClip lesson maps to Metis’s existing QK-clip stability guardrail: Muon efficiency needs attention-logit safety and FP32 master/state handling rather than blind optimizer swaps. Public Kimi K2 material and report pointers: `https://github.com/MoonshotAI/Kimi-K2`, `https://arxiv.org/pdf/2507.20534`.

## Trainium Mistakes To Avoid

- Do not use Python-side ragged token lists, dynamic all-to-all splits, or host-visible overflow decisions inside the model step.
- Do not mistake static synthetic FFN throughput for real Metis training health.
- Do not allow capacity factor changes to hide a broken router/combine path.
- Do not claim throughput readiness until the log proves real forward/backward/optimizer steps after compile.
- Do not let QK clipping mutate BF16 params without synchronizing the FP32 optimizer state.
- Do not let checkpoint resume advance counters while replaying sampler positions.

## JAX Implementation Shape

The first local implementation uses static sort-pack routing:

1. Compute router logits as arrays.
2. Select top-k with `lax.top_k`.
3. Flatten assignments to a fixed `tokens * top_k` axis.
4. Sort by expert id.
5. Pack each expert into `[num_experts, capacity, latent_dim]` using static `searchsorted` slot ranges.
6. Run all experts with `einsum` on unweighted latent payloads.
7. Apply route weights only at expert-output combine, then scatter-add back into `[tokens, latent_dim]`.
8. Apply the latent up-projection and shared expert path.

This avoids the Trainium failure pattern while also avoiding an impossible dense `[assignments, experts, capacity]` mask. It is still a reference implementation, not the final throughput ceiling. On v6e, the next optimization layer is explicit expert-axis sharding with `shard_map` or named-sharding constraints so each chip owns four experts.

The CPT MoR path now uses the same bounded-static idea for recursive depth. The depth router picks a hard depth per token. Depth 1 runs as a normal full-sequence decoder pass. For depth 2 and 3, active query tokens are sorted into a fixed `[depth_capacity, d_model]` buffer, run through packed-query causal attention that still attends to the full sequence for that batch, routed through mask-aware LatentMoE, and scattered back to `[batch, seq, d_model]`. Padding slots are masked so inactive slots do not consume expert assignment accounting.

## Local Proof Gates Now In Tree

- `scripts/metis15_jax_tpu_v6e_preflight.py --skip-device-check`
  - verifies manifest shape, v6e-8 expert layout, JAX import, optimizer policy, and stage-specific MoR rules.
- `scripts/smoke_metis15_jax_contracts.py`
  - proves finite JIT loss, fixed-shape LatentMoE assignment accounting, v6e expert partition specs, Muon/AdamW grouping, tiny overfit movement for base PT, deterministic memmap sampler resume, and local checkpoint restore.
- `scripts/smoke_metis15_jax_contracts.py --mor`
  - proves the CPT dynamic-token MoR path compiles as static packed hard routing, keeps the MoR control router replicated, proves forced depth-1 tokens skip recursive MoE assignments, proves forced recursive tokens use packed depth buffers, and still learns on a tiny local batch.
- `scripts/smoke_metis15_jax_full_shape_contracts.py`
  - proves the abstract full Metis-1.5 parameter tree has 898,051,168 params, 32 routed experts, 4 experts per v6e chip, manifest-derived effective train steps that include grad accumulation, and the expected Muon/AdamW plus partition-spec policies without allocating the full model on CPU.
- `scripts/train_metis15_jax_tpu.py --tiny-config --data-dir ... --resume`
  - proves the real trainer CLI can consume `train.bin`/`meta.json`, save params/optimizer/sampler state with Orbax, and resume without replaying the already-consumed windows.
- `XLA_FLAGS=--xla_force_host_platform_device_count=8 scripts/smoke_metis15_jax_mesh.py`
  - proves `shard_map` expert execution over an 8-way `expert` mesh matches the reference expert math, can run a JIT train step, and can save/restore sharded params plus optimizer state through Orbax.
- `XLA_FLAGS=--xla_force_host_platform_device_count=8 scripts/train_metis15_jax_tpu.py --tiny-config --expert-execution shard_map ...`
  - proves the trainer CLI can execute the shard-map expert path with the same fixed-shape learning contract and checkpoint path.
- `scripts/analyze_metis15_jax_tpu_logs.py` and `scripts/summarize_metis15_jax_tpu_sweep.py`
  - audit JAX trainer logs for finite loss, loss movement, logged `total_assign`-based valid assignment fraction, expert drop fraction, QK clip metrics, MoR-active/disabled stage checks, post-warmup throughput, and p95 step time without relying on old PyTorch/XLA log formats.
- `scripts/metis15_jax_tpu_v6e_compile_probe.sh`
  - runs the fixed-shape JAX trainer with `JAX_LOG_COMPILES=1`, verifies compile markers are present, and then audits the resulting step log. This is the first real-v6e gate after preflight because compile completion alone is not training health.
- `scripts/metis15_jax_tpu_v6e_sharding_report.py`
  - reports the abstract full-shape partition contract and, with eight visible devices, validates runtime `NamedSharding` for expert weights, router tensors, optimizer state, and replicated batches.
- `scripts/metis15_jax_tpu_v6e_quality_canary.sh`
  - runs a fixed-batch JAX canary and rejects it unless loss moves, assignment/drop/QK gates pass, and the log proves real steps after compile.
- `scripts/metis15_jax_tpu_v6e_perf_sweep.sh`
  - sweeps local batch, fixed-rank grad accumulation, block size, expert capacity factor, remat mode, dtype, and expert execution settings while ranking only candidates that pass the JAX log-health gates.
- `scripts/metis15_jax_tpu_v6e_local_readiness.sh`
  - runs the local proof ladder end-to-end: py_compile, preflight, full-shape abstract contracts, base/MoR smokes, 8-device CPU mesh proof, tiny compile probe, tiny pretrain canary, trainer-level CPT/MoR canary, CPT/MoR checkpoint-resume schedule proof, tiny sweep, real-data resume, and shard-map resume. Logs are retained under `tmp/metis15_jax_tpu_v6e_local_readiness`.

## Implemented Training Surfaces

- `JaxMemmapTokenData`: deterministic fixed-window memmap loader for `train.bin` / `meta.json` with explicit `cursor`, `epoch`, `tokens_emitted`, split, and data fingerprint.
- `save_training_checkpoint` / `restore_training_checkpoint`: Orbax-first JAX PyTree checkpoint hook that saves params, AdamW moments, Muon momentum, optimizer step, manifest hash, metrics, and sampler state. A local NPZ fallback remains for debugging.
- Gradient accumulation: fixed-rank microbatch stacking inside `train_step`, so larger effective batch sizes do not introduce Python-side dynamic control flow.
- Static LatentMoE combine semantics: routed expert payloads are unweighted latent vectors; top-k weights are applied once at combine after the nonlinear expert path.
- CPT MoR schedule, hard routing, and router controls: `mor_schedule_for_tokens` turns the manifest's target-depth and router-aux warmups into scalar JAX math from optimizer step and effective tokens per step. The MoR depth router consumes `router_temperature` inside the softmax, then `static_packed_hard` chooses a hard per-token depth for compute. PT logs prove MoR stays disabled; CPT trainer canaries prove MoR aux is active and target/coef increase without Python-side routing.
- Static packed recursive MoR: depth-1 tokens stop after the first decoder pass; depth-2/3 tokens are packed into fixed-capacity arrays, processed as active queries with full causal context, and scattered back. Trainer logs include packed active/valid/overflow telemetry.
- `parameter_partition_specs`: first v6e layout contract. Routed expert tensors shard over `P("expert", None, None)`, layer router outputs/biases shard over the expert axis, replicated tensors remain `P()`, and the CPT MoR depth router deliberately stays replicated.
- `sharded_squared_relu_experts`: explicit `jax.shard_map` expert MLP kernel. The expert dimension is sharded over the 8-chip `expert` mesh; for full Metis-1.5, each chip owns four routed experts.
- `shard_batch_for_v6e`: trainer batches are explicitly placed as replicated arrays on the expert mesh before the JIT step, avoiding implicit host/single-device placement in the `shard_map` path.
- `remat_layers`: full Metis-1.5 loads with rematerialized decoder layers enabled so first v6e HBM tests start from the memory-conservative graph.
- `qk_clip_with_optimizer_state`: Q/K clipping mirrors the scale into optimizer state, including Muon momentum and AdamW moments, so the stability transform does not leave stale optimizer tensors behind.
- `METIS15_JAX_GCS_CHECKPOINT_DIR`: optional launcher/trainer hook for `gcloud storage rsync -r` after checkpoint saves. The local checkpoint remains the primary Orbax write target; the hook mirrors it to GCS for TPU-host durability.
- `METIS15_JAX_DATA_GCS_URI`: optional launcher hook for hydrating local TPU-VM data from GCS before training. The recommended reservation flow is Storage Transfer Service from S3 to GCS before access starts, then `gcloud storage rsync -r` from GCS to local SSD at launch.
- Tuning overrides: `train_metis15_jax_tpu.py`, `metis15_jax_tpu_v6e_pretrain.sh`, the compile probe, and the quality canary expose block size, expert capacity factor, remat mode, activation dtype, local batch, grad accumulation, and expert execution as launch-time knobs. The perf sweep writes a `best.env` that can be sourced into the pretrain launcher, and local readiness clears generated gate logs before each proof run so stale sweep candidates cannot be promoted accidentally.
- CPU guard in `train_metis15_jax_tpu.py`: full Metis-1.5 cannot accidentally initialize on CPU unless explicitly overridden.

## Remaining v6e-Only Gates

- Run Orbax checkpoint save/restore on real v6e sharded arrays and verify `METIS15_JAX_GCS_CHECKPOINT_DIR` mirrors checkpoints from the TPU VM to GCS.
- Run `scripts/metis15_jax_tpu_v6e_sharding_report.py --require-runtime` on v6e and keep the report as placement evidence.
- Run `scripts/metis15_jax_tpu_v6e_compile_probe.sh` on v6e, keep the full `JAX_LOG_COMPILES=1` log as compile evidence, and record first-step compile time separately from steady-state step time.
- Inspect real-v6e sharding/resharding logs after the report/probe to confirm no hidden all-gathers appear around the expert path.
- Sweep local batch, capacity factor, remat, dtype, and effective batch via fixed-rank grad accumulation after HBM numbers are known.
- Promote throughput only when fixed-batch loss, valid assignments, drop fraction, qk clip, and real-data step logs all pass.

## v6e Launch Ladder

0. Before renting v6e, run `scripts/metis15_jax_tpu_v6e_local_readiness.sh` locally.
0.5. Pre-stage data with `docs/metis15_jax_tpu_v6e_launch_checklist.md`; use Storage Transfer Service for S3 -> GCS and reserve runtime copying for GCS -> local TPU SSD.
1. `scripts/metis15_jax_tpu_v6e_bootstrap.sh`
2. `scripts/metis15_jax_tpu_v6e_preflight.py`
3. `scripts/metis15_jax_tpu_v6e_sharding_report.py --require-runtime`
4. `scripts/metis15_jax_tpu_v6e_compile_probe.sh`
5. `scripts/metis15_jax_tpu_v6e_quality_canary.sh`
6. `scripts/metis15_jax_tpu_v6e_perf_sweep.sh`
7. Promote only the best sweep candidate that also passes `scripts/analyze_metis15_jax_tpu_logs.py` and `scripts/summarize_metis15_jax_tpu_sweep.py`.

The compile probe must show JAX/XLA compile markers and real step logs. The first step after
compile is not throughput evidence. Use `METIS15_JAX_PERF_WARMUP_STEPS` to exclude
compile-heavy steps, then compare post-warmup median tok/s, p95 step time, loss, assignment
drop, and QK clip metrics together.
