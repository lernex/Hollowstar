# Metis-1.5 Single LatentMoE Reset

Metis-1.5 now uses a single NVIDIA-style LatentMoE FFN path instead of the
previous multi-head LatentMoE path.

The old production shape routed four feature heads independently:

```text
4 MoE heads * top-4 experts = 16 routed expert applications per token per layer
```

The reset shape routes each token once:

```text
x_full [1536]
  -> latent_down [512]
  -> one top-k router over 32 experts
  -> routed experts operate in 512d latent space
  -> latent_up [1536]
  + full-dim shared BF16 expert
```

Current manifest defaults:

```text
ffn_type: single_latent_moe
latent_dim: 512
num_experts: 32
top_k: 4
shared_experts: 1
expert_hidden: 1024
moe_activation: squared_relu
router_score: sigmoid
moe_balance_strategy: aux_loss_free_bias
moe_balance_bias_update_rate: 1e-3
moe_aux_loss_coef: 1e-4
```

Precision policy:

```text
Default launch: BF16 baseline
Protected BF16 surfaces: embeddings, lm_head, attention QKV/O, router, latent down/up, shared expert
Optional later path: routed expert GEMMs only under H100 FP8
Disabled by default: NVFP4, MXFP8, global FP8
```

The mandatory launch gate is the compute audit:

```bash
python scripts/audit_metis_compute.py --manifest configs/metis15_manifest.json
```

The current reset audit is about `339.5M` rough parameter-applications per token,
below the `450M` H100 gate. If that rises above `450M`, do not start paid H100
training until the architecture is resized.
