# Metis-1.5 TPU v6e-8 Training Handoff

> Historical note: this document describes the older PyTorch/XLA TPU path.
> The active June 2026 reservation path is the JAX/libTPU launcher in
> `docs/metis15_jax_tpu_v6e_rewrite_plan.md` and
> `docs/metis15_jax_tpu_v6e_launch_checklist.md`.
> Use `scripts/metis15_jax_tpu_v6e_pretrain.sh`, not the `metis15_tpu_*`
> PyTorch/XLA launchers, for Metis-1.5 v6e-8 training.

This is the Google TPU/PJRT path for Metis-1.5 base and continued pretraining. The Trainium/Neuron stack is not used.

## Target

- Accelerator: Google Cloud TPU `v6e-8` / 8 chips on one host.
- Runtime: PyTorch/XLA PJRT with `PJRT_DEVICE=TPU`.
- Entry point: `scripts/metis15_tpu_v6e_pretrain.sh`.
- Trainer: `scripts/train_metis15_tpu.py`.
- Bootstrap: `scripts/metis15_tpu_v6e_bootstrap.sh`.
- Preflight: `scripts/metis15_tpu_v6e_preflight.py`.
- Default optimizer: AdamW/Muon hybrid with routed experts left on AdamW, latent payload/shared/attention matrices on Muon, FP32 master weights enabled.

Google documents v6e as Trillium with 32 GB HBM per chip, 8 chips per host, BF16-oriented transformer training support, and PyTorch/XLA training examples on `v6e-8`. Their current PyTorch TPU VM guide verifies TPU visibility with `PJRT_DEVICE=TPU python3 -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('TPU'))"`, expecting `xla:0` through `xla:7`. PyTorch/XLA documents PJRT TPU runtime through `PJRT_DEVICE=TPU`, and supports `torchrun` with `init_method="xla://"`.

## Research Decisions Baked Into This Port

### LatentMoE

The TPU trainer keeps the NVIDIA LatentMoE design contract:

- The router sees the full hidden state.
- The routed payload is projected to the smaller latent dimension before all-to-all expert dispatch.
- Routed expert computation stays in latent space.
- Shared experts stay in full hidden space.

That is why the TPU launcher defaults `METIS15_TPU_BALANCED_STATIC_ROUTER_INPUT=hidden` and the model manifest keeps `moe_single_latent_router_input=hidden`.

### MoE Stability

The loss-stuck-at-7 failure mode gets treated as a training-correctness issue, not just hardware noise. The TPU path keeps these guardrails on by default:

- Next-token CE objective only, with logits cast to FP32 before CE.
- Learned router on hidden states, not force-balanced routing except for canaries.
- Aux-loss-free bias balancing enabled.
- Expert capacity factor default `4.0` to avoid early token dropping while the router is untrained.
- Expert activation clamp for squared-ReLU experts.
- Routed expert matrices stay on AdamW by default while Muon is used for attention, dense/shared experts, and latent down/up projections.
- QK clipping is enabled by default to stop Muon-driven attention-logit explosion.

### AdamW/Muon Hybrid

Muon is used only where it is expected to help and where the update is a true 2D matrix:

- Muon: attention QKV/O, dense MLP matrices, shared expert matrices, latent payload down/up projections.
- AdamW: embeddings, LM head, normalization/scalars, router/gate controls, latent router projection, and routed expert matrices.

This mirrors Moonshot/Kimi guidance: use Muon with weight decay and AdamW-matched update RMS for scalable LLM training, but keep non-matrix/control parameters on AdamW.

### MuonClip / QK Clip

`scripts/train_metis15_tpu.py` records the maximum scaled attention logit per layer/head during forward/backward and, after each optimizer step, rescales the packed Q/K projection rows when a head exceeds the default threshold `100`.

Because the TPU lane uses FP32 optimizer master weights, the trainer immediately syncs the clipped QKV parameters back into the optimizer's master copies. Without that sync, the next optimizer step could overwrite the clipped BF16 model weights and make QK clip look active in logs while not persisting.

The trainer logs:

- `qk_clip max_logit`
- `qk_clip min_scale`
- `qk_clip scaled_heads`
- `profile_components qk_clip_s`

That makes the stabilizer auditable: if it saves training but costs too much throughput, the TPU run will show it.

## Loss-Stuck-at-7 Investigation Guardrails

The old bad-learning symptom should be treated as a failed correctness gate until a TPU log proves otherwise. The TPU path attacks the plausible failure modes directly:

- Objective cannot silently degrade to a dummy loss: launcher default is `METIS15_TPU_LOSS_MODE=real_ce`, and the analyzer can require loss descent.
- BF16 CE numerical risk is reduced: shifted logits are cast to FP32 before cross entropy by default.
- Router cannot be hidden by a forced-balance canary: production default is `METIS15_TPU_ROUTER_OVERRIDE=learned`; forced-balanced routing remains an explicit ablation only.
- Early router drops are visible: capacity factor defaults to `4.0`, `valid_assign` is logged every step, and the analyzer fails if valid assignments fall below the expected fraction.
- Expert collapse/load imbalance is auditable: `METIS15_TPU_LOG_EXPERT_HISTOGRAMS=1` records per-layer counts, drop fractions, and destination counts during canaries.
- Muon is not applied to the brittle routed expert payload matrices by default; routed experts, routers, embeddings, norms, and scalars stay on AdamW.
- Attention-logit blowups from Muon are bounded by QK clip and logged every optimizer step.
- Resume runs do not silently replay a mismatched sampler stream: checkpoints carry the batch RNG state, and older checkpoints advance the sampler by the expected number of draws.

For the specific "loss not below 7" regression, use `scripts/metis15_tpu_v6e_quality_canary.sh` first, then audit a longer real-data log with `--max-final-loss 7.0` only after enough steps/tokens for that threshold to be meaningful. A synthetic compile sweep is not enough evidence for this bug.

## TPU VM Setup

Install a TPU-compatible PyTorch/XLA image or environment, then clone/sync this repo and data to local disk on the TPU VM.

Typical VM creation shape:

```bash
gcloud alpha compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$ZONE" \
  --accelerator-type=v6e-8 \
  --version=v2-alpha-tpuv6e
```

Then on the TPU VM:

```bash
export PJRT_DEVICE=TPU
export METIS15_GCS_ROOT=gs://YOUR_BUCKET/metis15
export METIS15_TRAIN_STAGE=pretrain
scripts/metis15_tpu_v6e_bootstrap.sh
scripts/metis15_tpu_v6e_pretrain.sh
```

The bootstrap script creates `.venv-tpu`, installs `requirements-tpu-train.txt`, and runs strict preflight. The launcher also runs preflight by default before it hydrates data or starts `torchrun`; set `METIS15_TPU_PREFLIGHT=0` only for deliberate local dry-runs.

The launcher writes the trainer stream to `$METIS15_OUT_DIR/train.log` by default, and the checkpoint GCS sync uploads it with the rest of the run artifacts. The TPU dependency file intentionally has no Neuron, AWS, CUDA-only, Liger, or GPU-kernel packages.

For a compile/performance smoke without real data:

```bash
METIS15_TPU_SYNTHETIC=1 \
METIS15_TPU_MAX_STEPS=8 \
METIS15_TPU_SKIP_CHECKPOINT=1 \
METIS15_TPU_LOG_INTERVAL=1 \
scripts/metis15_tpu_v6e_pretrain.sh
```

For a controlled throughput sweep that keeps training-quality guardrails enabled:

```bash
METIS15_TPU_SWEEP_LOCAL_BATCHES="1 2" \
METIS15_TPU_SWEEP_GRAD_ACCUMS="8 16" \
METIS15_TPU_SWEEP_ATTENTION_KERNELS="sdpa eager_gqa" \
METIS15_TPU_SWEEP_GRAD_BUCKET_MB="16 32" \
scripts/metis15_tpu_v6e_perf_sweep.sh
```

The sweep writes a ranked summary and `best.env` under `tmp/metis15_tpu_v6e_perf_sweep` only for candidates that pass the log audit gates. You still need to run the fixed-batch quality canary before promoting the fastest setting:

```bash
scripts/metis15_tpu_v6e_quality_canary.sh
```

To audit any single run log after a smoke or real-data launch:

```bash
scripts/analyze_metis15_tpu_logs.py checkpoints/metis15_base_tpu_v6e/train.log \
  --min-logged-steps 10 \
  --require-profile \
  --require-qk-clip \
  --require-loss-decrease \
  --min-loss-drop-frac 0.01 \
  --min-valid-assign-frac 0.99 \
  --max-qk-logit 1000
```

## Throughput-Safe Tuning Loop

Do not optimize tok/s by changing the training objective or hiding router drops. Use this order:

1. Establish a baseline with defaults:
   - `METIS15_TPU_SYNTHETIC=1`
   - `METIS15_TPU_MAX_STEPS=12`
   - `METIS15_TPU_LOG_INTERVAL=1`
   - `METIS15_TPU_PROFILE_COMPONENTS=1`

2. Repeat with real data for at least 50 optimizer steps:
   - Loss must move in the right direction on a fixed-batch overfit canary.
   - `valid_assign` should be near total expected routed assignments.
   - Expert histogram drop fraction should stay near zero during the first run.
   - `qk_clip max_logit` must not explode.
   - If evaluating the historical loss-stuck-at-7 failure, run enough real-data steps to use `--max-final-loss 7.0`; do not claim it fixed from a compile-only synthetic sweep.

3. Sweep only throughput-safe knobs first:
   - `METIS15_TPU_LOCAL_BATCH_SIZE`
   - `METIS15_TPU_GRAD_ACCUM_STEPS`
   - `METIS15_TPU_GRAD_SYNC_BUCKET_MB`
   - `METIS15_TPU_ATTENTION_KERNEL=sdpa|eager_gqa`
   - `METIS15_TPU_PREMAPPED_BUFFER_SIZE`

4. Keep quality guardrails fixed while sweeping:
   - Do not disable QK clipping until an ablation shows no logit growth.
   - Do not reduce expert capacity until expert histograms show stable low drop rates.
   - Do not move routed expert matrices to Muon until the AdamW-routed baseline learns cleanly.

5. Promote a faster setting only when it wins all of these at once:
   - Higher median tok/s after perf warmup.
   - Similar or lower p95 step time.
   - No higher expert drop fraction.
   - No worse fixed-batch loss descent.
   - No attention-logit explosion.
   - `scripts/analyze_metis15_tpu_logs.py` passes on the candidate log.
   - `scripts/metis15_tpu_v6e_quality_canary.sh` passes after sourcing the candidate `best.env`.

## Current Source Anchors

- Google Cloud TPU v6e training guide: `https://docs.cloud.google.com/tpu/docs/v6e-training`.
- Google Cloud PyTorch TPU VM guide: `https://docs.cloud.google.com/tpu/docs/run-calculation-pytorch`.
- PyTorch/XLA PJRT runtime guide: `https://docs.pytorch.org/xla/release/r2.5/runtime.html`.
- NVIDIA LatentMoE paper/page: `https://arxiv.org/abs/2601.18089`.
- Kimi K2 / MuonClip paper: `https://arxiv.org/abs/2507.20534`.
- Moonshot Muon scaling paper: `https://arxiv.org/abs/2502.16982`.

## Known Cloud-Only Validation Boundary

Mac-local CPU tests can prove contracts, optimizer grouping, CE shape, QK clipping mechanics, finite gradients, and that tiny MoE models can learn. They cannot prove TPU compiler behavior, real v6e memory headroom, ICI all-to-all throughput, or true tok/s. The handoff is not complete until the TPU VM run records compile success and steady-state logs with `profile_components`, `qk_clip`, and optionally `expert_hist`.
