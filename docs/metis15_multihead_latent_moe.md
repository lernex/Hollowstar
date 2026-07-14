# Metis-1.5 Multi-Head Latent MoE

Metis-1.5 is configured as the first sparse Metis decoder:

- Total parameters: about 1.096B with the 32,768-token vocabulary.
- Active transformer parameters: about 261M per token, reported as A0.26B.
- Active parameters including embeddings/final norm: about 311M.
- Routed experts: 32.
- Routing: top-4 per MoE head.
- Shared experts: 1 always-active expert.
- MoE heads: 4 feature heads, each 384 dimensions.
- Router latent size: 128.
- Routed payload/expert latent size: 384.

The implementation follows two research ideas without importing a new framework:

- DeepSeekMoE-style shared expert isolation: the shared expert is always active so common knowledge does not have to be duplicated across routed experts.
- Multi-Head MoE-style sub-token routing: each token channel is split into multiple feature heads and routed independently.
- NVIDIA LatentMoE-style routed payload separation: routed expert compute runs in a per-head latent width, then projects back to the model width; shared experts remain in the original head width. The current 384d routed width is intentionally 128-aligned for Blackwell low-precision kernels.
- DeepSeek-V3-style auxiliary-loss-free balancing: non-gradient expert bias affects top-k selection only, while gate weights are still computed from unbiased router scores.
- Dynamic token MoR during continued pretraining: base pretraining is dense, then CPT introduces token-level recursive depth selection over the shared MoE stack.
- Blackwell mixed NVFP4 training: expert MLP TE linears run under `NVFP4BlockScaling`, QKV and LatentMoE projections use MXFP8, and token/logit surfaces stay BF16.
- Muon-AdamW hybrid optimization: large hidden matrices use Muon, while embeddings, routers, norms, biases, and unstable control surfaces stay AdamW.

## Code Paths

- `src/metis_mamba/config.py` adds `ffn_type = "multi_head_latent_moe"` plus sparse expert sizing, latent router, and active parameter estimates.
- `src/metis_mamba/model.py` adds `MetisMultiHeadLatentMoE`, a sparse FFN replacement for `MetisSwiGLU`.
- `configs/metis15_manifest.json` is the canonical Metis-1.5 architecture and training manifest.
- `make metis15-rtx-pretrain` launches base pretraining with the Metis-1.5 manifest.
- `make metis15-rtx-continued-pretrain` launches continued pretraining from the base checkpoint with the same mixed NVFP4 policy.

## Routing Contract

For each transformer block:

1. RMS-normalized token states enter the FFN path.
2. The hidden channel is reshaped from `d_model = 1536` to 4 heads of 384 dimensions.
3. Each head is projected to a 128d latent router space.
4. Latents are scored against per-head expert embeddings for 32 routed experts.
5. The top 4 experts per token-head are selected with auxiliary-loss-free balance bias and normalized sigmoid gates.
6. The routed payload stays in a 384d expert latent space, processed by routed experts, and combined without a routed down/up compression pair.
7. One shared expert always processes every token-head in the original 384d space.
8. Routed and shared outputs are summed and reshaped back to `d_model`.

Metis-1.5 disables MoR by default. It also defaults to auxiliary-loss-free MoE load balancing, so the MoE aux-loss coefficient is zero unless explicitly changed for ablation.

## Dynamic MoR + MoE

Metis-1.5 does not use sequence-level or static bucket MoR. The `pretrain` stage runs `static_dense_pretrain`, and the `continued_pretrain` stage switches to `dynamic_token_mor`.

The continued-pretrain router assigns each token a recursive depth from 1 to 3. The target average depth warms from 1.05 to 1.65 over the first 1B CPT tokens, with token packing enabled so later recursive passes only process still-active tokens. This keeps the first CPT phase close to the dense base checkpoint, then gradually teaches the model to spend extra computation on harder tokens.

To keep MoR and MoE from fighting each other, Metis-1.5 scales auxiliary-loss-free MoE balance-bias updates by the active token fraction for each recursive pass. A second pass over 45% of tokens now updates expert balance at 45% of the normal bias-update rate instead of overcorrecting as if it saw a full batch. The MoR router also gets a small entropy reward and z-loss so the depth router does not immediately collapse to one depth or explode its logits.

## Mixed NVFP4

Metis-1.5 defaults to `low_precision_mode = "nvfp4"` for RTX PRO 6000 training. The NVFP4 recipe keeps BF16 master weights and uses Transformer Engine's `NVFP4BlockScaling` autocast path. Metis-1.5 also has an explicit MXFP8 tier through `MXFP8BlockScaling`.

The precision map is:

- BF16: token embeddings and the tied LM head.
- MXFP8: attention QKV projection.
- MXFP8: LatentMoE router projection.
- MXFP8: LatentMoE routed down/up projections.
- NVFP4: routed and shared expert MLP linears plus the remaining TE linears.

The routed and shared expert MLP linears are the main NVFP4 compute target. QKV and LatentMoE projections are still cheaper than BF16, but avoid jumping straight to FP4 for routing-sensitive surfaces.

## Muon-AdamW Hybrid

Metis-1.5 uses a local `muon_adamw` optimizer for LM pretraining and exposes the same optimizer option for SFT, reward-model, and DPO stages. The default policy is intentionally conservative around routing:

- AdamW: token embeddings and LM head/unembedding.
- AdamW: RMSNorm/LayerNorm weights, scalars, and biases.
- AdamW: MoR routers/control params and MoE router/gate params.
- AdamW: LatentMoE `latent_proj`, expert embeddings, and expert bias.
- Muon: attention QKV/O matrices.
- Muon: dense MLP matrices.
- Muon: shared expert gate/up/down matrices.
- Muon: LatentMoE routed payload down/up projections.
- AdamW by default: routed expert gate/up/down matrices.

Routed expert matrices can be moved into Muon for an ablation with `METIS15_MUON_INCLUDE_ROUTED_EXPERTS=1` or `--muon-include-routed-experts`. They are not Muon by default because routed experts are the highest-risk surface for sparse-MoE instability.

## Launch

```bash
make metis15-cpu-prep-aws
make metis15-rtx-pretrain
make metis15-rtx-continued-pretrain
```

The Metis-1.5 manifest now points at its own bucketed 60B data plan:

- `configs/metis15_pretrain_mix.json`: 50B base pretrain tokens across high-quality web, reference/wiki, academic STEM, math/proof, textbooks, books, QA, synthetic education, and reserve buckets.
- `configs/metis15_continued_pretrain_mix.json`: 10B CPT/midtrain tokens focused on STEM, math/proof, reference, verified problem-solution prose, FinePDFs, science tasks as text, and replay.
- `configs/metis15_chat_mix.json`: 1.2M internal chat SFT examples.
- `configs/metis15_reasoning_mix.json`: 600K think/reasoning SFT examples.
- `configs/metis15_preference_mix.json`: 400K reward-model pairs and 300K DPO pairs, with 18% on-policy Metis-generated negatives included in the final target.

The release target remains base plus think only; chat is an internal bridge stage.
