# Metis-1.5 JAX TPU v6e-8 Launch Checklist

This is the source-of-truth launch checklist for the Google Cloud TPU v6e-8 reservation.
Use this JAX/libTPU path, not the older PyTorch/XLA TPU scripts.

## Decision: Pre-stage S3 to GCS

Pre-stage the training files from AWS S3 to Google Cloud Storage before TPU access starts.
Do not spend the paid TPU window pulling directly from S3 unless this pre-stage fails.

Why:

- Google recommends Storage Transfer Service for transfers from another cloud storage provider to Cloud Storage.
- Storage Transfer Service can copy Amazon S3 to Cloud Storage as a managed job, with source prefix filtering, manifests, logging, and monitoring.
- `gcloud storage rsync` can mirror across providers, but cross-provider traffic flows through the machine running the command, so it is a fallback for small repair copies, not the main 50B-token staging path.
- The JAX trainer consumes local `train.bin`/`meta.json` memmaps. The TPU VM should hydrate from nearby GCS to local disk, then train locally and mirror checkpoints back to GCS.

Authoritative references:

- Storage Transfer Service transfer options: `https://docs.cloud.google.com/storage-transfer/docs/transfer-options`
- Amazon S3 to Cloud Storage transfer guide: `https://docs.cloud.google.com/storage-transfer/docs/create-transfers/agentless/s3`
- `gcloud storage rsync`: `https://docs.cloud.google.com/sdk/gcloud/reference/storage/rsync`
- TPU v6e training guide: `https://docs.cloud.google.com/tpu/docs/v6e-training`

## Pre-stage Before The Reservation

Pick a regional GCS bucket close to the TPU zone.

Current live setup:

- Google Cloud project: `calm-spring-478700-b9`
- TPU future reservation: `future-reservation-20260530-053311`
- TPU zone: `us-east1-d`
- Reservation window: `June 4, 2026, 4:30 AM` to `June 5, 2026, 4:30 AM`
- GCS bucket: `gs://metis15-tpu-v6e-calm-spring-478700-b9`
- Base data transfer job: `9147819145504621172`
- Continued data transfer job: `142258592209547308`

```bash
export GCP_PROJECT="calm-spring-478700-b9"
export GCS_BUCKET="metis15-tpu-v6e-calm-spring-478700-b9"
export METIS15_S3_ROOT="s3://lernex-metis-artifacts-151025633969-us-east-1/metis15"
export METIS15_GCS_ROOT="gs://$GCS_BUCKET/metis15"

gcloud config set project "$GCP_PROJECT"
gcloud storage buckets create "gs://$GCS_BUCKET" --location="us-east1"
```

Create a one-time managed transfer for base pretrain data.

```bash
cat > /tmp/metis15-aws-source-creds.json <<'JSON'
{
  "accessKeyId": "AWS_ACCESS_KEY_ID",
  "secretAccessKey": "AWS_SECRET_ACCESS_KEY"
}
JSON

gcloud transfer jobs create \
  "$METIS15_S3_ROOT/pretrain-shards/base" \
  "$METIS15_GCS_ROOT/pretrain-shards/base/" \
  --source-creds-file=/tmp/metis15-aws-source-creds.json \
  --description="metis15-base-pretrain-s3-to-gcs"
```

Repeat for continued pretrain if the base phase will finish inside this reservation.

```bash
gcloud transfer jobs create \
  "$METIS15_S3_ROOT/pretrain-shards/continued" \
  "$METIS15_GCS_ROOT/pretrain-shards/continued/" \
  --source-creds-file=/tmp/metis15-aws-source-creds.json \
  --description="metis15-cpt-s3-to-gcs"
```

Verify the transferred object set before TPU access starts.

```bash
gcloud storage ls "$METIS15_GCS_ROOT/pretrain-shards/base/"
gcloud storage ls "$METIS15_GCS_ROOT/pretrain-shards/base/meta.json"
gcloud storage ls "$METIS15_GCS_ROOT/pretrain-shards/base/train.bin"
```

Use direct cross-provider rsync only for small fallback repair copies.

```bash
gcloud storage rsync \
  "$METIS15_S3_ROOT/pretrain-shards/base" \
  "$METIS15_GCS_ROOT/pretrain-shards/base" \
  --recursive
```

## Local Gate Before Access

Run this before the reservation starts.

```bash
scripts/metis15_jax_tpu_v6e_local_readiness.sh
```

Expected final line:

```text
metis15_jax_tpu_v6e_local_readiness_ok
```

## TPU VM Bring-up

On the TPU VM, sync the repo, then run:

```bash
export METIS15_GCS_ROOT="gs://metis15-tpu-v6e-calm-spring-478700-b9/metis15"
export METIS15_TRAIN_STAGE=pretrain
export METIS15_DATA_DIR=/mnt/disks/localssd/metis15/pretrain
export METIS15_GCS_PRETRAIN_URI="$METIS15_GCS_ROOT/pretrain-shards/base"
export METIS15_JAX_DATA_GCS_URI="$METIS15_GCS_PRETRAIN_URI"
export METIS15_JAX_GCS_CHECKPOINT_DIR="$METIS15_GCS_ROOT/checkpoints/base-jax-v6e"
export METIS15_OUT_DIR=/mnt/disks/localssd/metis15/checkpoints/base-jax-v6e
export METIS15_JAX_EXPERT_EXECUTION=shard_map

scripts/metis15_jax_tpu_v6e_bootstrap.sh
```

The pretrain launcher hydrates `METIS15_DATA_DIR` from `METIS15_JAX_DATA_GCS_URI`
when `meta.json` or `train.bin` is missing. It also pulls `METIS15_JAX_GCS_CHECKPOINT_DIR`
into `METIS15_OUT_DIR` before `--resume` when local `latest/` is missing.

## Paid TPU Gate Order

Run these in order once the v6e-8 is visible.

```bash
scripts/metis15_jax_tpu_v6e_preflight.py
scripts/metis15_jax_tpu_v6e_sharding_report.py --require-runtime
scripts/metis15_jax_tpu_v6e_compile_probe.sh
scripts/metis15_jax_tpu_v6e_quality_canary.sh
scripts/metis15_jax_tpu_v6e_perf_sweep.sh
```

Promote the best safe sweep candidate only after the analyzer accepts it.

```bash
source tmp/metis15_jax_tpu_v6e_perf_sweep/best.env
```

## Full Base Launch

Leave `METIS15_JAX_MAX_STEPS` unset for the manifest-derived full base run.
Set it only for deliberate short runs.

```bash
unset METIS15_JAX_MAX_STEPS
unset METIS15_JAX_SYNTHETIC
scripts/metis15_jax_tpu_v6e_pretrain.sh
```

For base pretraining, the manifest effective step count is derived from
`50B / (local_batch_size * grad_accum_steps * block_size)`.

## Promotion Rules

Do not call the run healthy from compile completion alone. Promote only when logs show:

- eight TPU devices visible;
- runtime sharding report passes;
- JAX compile probe has compile markers and real post-compile step logs;
- fixed-batch quality canary passes loss, assignment, drop, and QK gates;
- perf sweep improves post-warmup throughput without worse p95 step time or quality gates;
- checkpoints save locally and mirror to GCS;
- resume from the GCS-mirrored `latest/` checkpoint does not replay sampler state.

## Current Architecture Contract

- Runtime: JAX/libTPU.
- Hardware: one v6e-8 host, eight chips.
- Model: Metis-1.5 single LatentMoE decoder.
- Routed experts: 32 total, top-4, four experts per TPU chip.
- Latent routed path: 1536d hidden to 512d latent, squared-ReLU experts, then back to 1536d.
- Shared path: one full-dim BF16 shared expert per layer.
- Base PT: `static_dense_pretrain`, MoR disabled.
- CPT: `dynamic_token_mor`, `static_packed_hard`, hard packed recursive depths 2/3.
- Optimizer: Muon/AdamW hybrid; routed experts, routers, embeddings, norms, biases, and control tensors stay on AdamW.
