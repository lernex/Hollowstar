# Metis GPU Image

This folder builds the custom GPU image for the current `Metis-1.5` H100 training stack on top of:

- `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, pinned in `Dockerfile.h100` to the linux/amd64 manifest digest

It preinstalls:

- the repo's GPU-side Python dependencies from `requirements-gpu-train.txt`
- NVIDIA Transformer Engine v2.15 built for Hopper / `sm_90a`
- Liger Kernel for fused linear cross entropy
- the current Metis repo as an editable package

This image is intentionally optimized for the reset `Metis-1.5` single LatentMoE path:

- `scripts/metis15_pretrain.sh`
- `scripts/metis15_full.sh`
- BF16 baseline training by default
- optional expert-only FP8 after the BF16 path is stable
- TF32-enabled kernels
- SDPA / native GQA attention without making FlashAttention a launch blocker
- MoR transformer checkpoints
- CUDA builds constrained to `sm_90a` to avoid wasting image-build time and memory on architectures this image is not targeting

It deliberately does not preinstall the older `TorchTitan` + `torchao` stack, the old `Metis-1.3` Mamba CUDA extensions, or Blackwell-only NVFP4/MXFP8 assumptions because the current H100 path does not use them by default.

Build from the repo root so Docker can see the whole project:

```bash
cd /Users/giulianno/Documents/10M\ model
docker buildx build \
  --platform linux/amd64 \
  -f docker/runpod-metis-gpu/Dockerfile.h100 \
  -t YOUR_DOCKERHUB_USER/metis-gpu:metis15-h100-single-latent-v1 \
  --push \
  .
```

If we decide to test TE's newer `te.ops` fused grouped MLP path on the live
card, rebuild with:

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg INSTALL_TE_FUSED_GROUPED_MLP_DEPS=1 \
  -f docker/runpod-metis-gpu/Dockerfile.h100 \
  -t YOUR_DOCKERHUB_USER/metis-gpu:metis15-h100-single-latent-v1-teops \
  --push \
  .
```

Recommended image tag:

- `lernex/metis-gpu:metis15-h100-single-latent-v1`

The H100 tag is the preferred RunPod launch image for the current Metis-1.5
scripts because the launchers default to BF16, use the new single-latent compute
audit gate, and keep `--lm-loss-impl liger_fused_linear_ce` available.

The legacy runtime-liger Dockerfile is kept as a tiny compatibility layer for
older automation that still references it:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/runpod-metis-gpu/Dockerfile.runtime-liger \
  -t lernex/metis-gpu:metis15-h100-single-latent-v1-runtime-liger \
  --push \
  .
```

This image sets these defaults for you:

- `METIS13_SKIP_ENV_SETUP=1`
- `METIS13_PYTHON_BIN=/opt/metis-venv/bin/python`
- `METIS15_SKIP_ENV_SETUP=1`
- `METIS15_PYTHON_BIN=/opt/metis-venv/bin/python`

So on the GPU box, the launchers can use the prebuilt environment immediately instead of spending time reinstalling Python dependencies at startup.

Quick smoke test after building locally:

```bash
docker run --rm --platform linux/amd64 YOUR_DOCKERHUB_USER/metis-gpu:metis15-h100-single-latent-v1 \
  bash -lc 'python -c "import torch, transformer_engine.pytorch as te; from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss; print(torch.__version__, torch.version.cuda, getattr(te, \"__file__\", \"missing\"), LigerFusedLinearCrossEntropyLoss)" && python scripts/audit_metis_compute.py --manifest configs/metis15_manifest.json && python scripts/train_mamba_lm.py --help'
```
