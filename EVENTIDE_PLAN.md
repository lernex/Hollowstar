# Eventide — Working Architecture and Serving Plan

Status: **research specification, not an executable manifest**
Last reconciled: **2026-08-29**

This is the durable source for the architecture currently being designed. Values marked
**working default** are the configuration to size and prototype first. Values marked **open** must
not be repeated as settled facts until an executable parameter ledger, a training canary, and a
representative RTX 5070 Ti kernel benchmark close them.

## 0. North star

Eventide is a roughly **50B-total / 2B-active-per-MoRE-pass** hybrid recurrent LatentMoE intended
to serve from one 16 GiB RTX 5070 Ti using packed ternary weights, a standard 262,144-token
context, durable prefix/state caching, and an exact speculative verifier. The deployment target is:

- batch 32;
- at least 13,000 aggregate accepted tokens/s;
- at least 400 accepted tokens/s per stream;
- no change to the target model distribution in the exact-serving mode;
- approximately 1.6 bits/weight for the large ternary matrices, without pretending the GPU has a
  native ternary tensor-core instruction.

This target requires an Eventide-specific inference engine. DFlash, DARTree, CALM, prefix caching, and
packed ternary kernels are ingredients, not a stack of independently multiplicative speedups.

## 1. Current working architecture

| Surface | Working specification | Status |
|---|---|---|
| Total neural parameters | approximately 50B | Target; exact ledger open |
| Average active parameters | approximately 2B per MoRE pass | Target; exact ledger open |
| MoRE depth | 1–5 passes, target mean approximately 2 | Working default |
| Model width | `d_model = 2048` | Working default |
| Routed latent width | `d_latent = 512` | Working default |
| Unique backbone blocks | 32 | Working default |
| Backbone layout | `7 Mamba-3 SISO -> HCA -> 7 Mamba-3 SISO -> CSA`, repeated twice | Working default |
| Mixer counts | 28 Mamba-3 SISO, 2 HCA, 2 CSA | Working default |
| Residual topology | four-stream, recursion-aware mHC | Working default |
| Routed experts per expert-bearing block | **1,024 logical micro-experts** | Preferred candidate; see section 2 |
| Physical expert packing | 512 adjacent two-micro-expert groups, independently addressable | Preferred engine layout |
| Routed K | integer `K in [0, 32]`, target `E[K] = 16.5` | Working default |
| Shared path | one full-width shared expert per block, `d_model=2048`, `d_ff=1536` | Working default |
| Vocabulary | 131,072 | Working default |
| Embedding/head | untied candidate; ternary head if it passes parity gates | Open ablation |
| Weight storage | packed ternary target for large matrices, approximately 1.6 bpw | Target, not measured execution rate |
| Higher-precision islands | smallest validated type for state, scales, reductions, normalization, and fragile controls | Tensor-by-tensor gate |
| Standard context | 262,144 original tokens | Working default |
| Extended context | 524,288 and 1,048,576 evaluation tiers | Open release gate |
| N-gram memory | large system-RAM-resident learned table plus exact suffix-continuation sidecar | Capacity and layout open |
| MTP | CALM-style training signal and proposal feature | Exact-serving integration open |
| Serving | batch-32 exact tree speculation with durable multi-tier state cache | Custom implementation required |

The matched system control is **512 double-width experts at average K 8.25**. The earlier
512/7.5 and 1,024/15 pair remains a historical throughput reference, not the current A2B ledger.

## 2. Expert-count decision

### 2.1 The normalization that must not drift

Let a 512-expert baseline expert contain `P_e` parameters. Then:

```text
512 experts, mean K = 8.25:
  stored routed bank per layer = 512 * P_e
  active routed parameters     = 8.25 * P_e

1024 half-sized experts, mean K = 16.5:
  stored routed bank per layer = 1024 * (P_e / 2) = 512 * P_e
  active routed parameters     = 16.5 * (P_e / 2) = 8.25 * P_e
```

Therefore the exact iso-total, iso-active conversion is:

```text
N:       512 -> 1024
size:    P_e -> P_e / 2
mean K:  8.25 -> 16.5
max K:   16 -> 32
```

Mean `K = 16.5` never means executing half an expert. K is integer for every token at every pass;
16.5 is the mean over routed token-pass decisions. It could arise from half of decisions selecting
16 experts and half selecting 17, or from a learned wider distribution with the same budget.

Relative to the historical 1,024/15 design, the new mean raises routed activity by 10%:

```text
16.5 / 15 - 1 = 10%
```

This is paid for by shrinking the full-width shared expert from `d_ff=2048` to `d_ff=1536`, a 25%
reduction. It is a deliberate transfer of the A2B budget from universal computation into routed
specialization, not free compute.

### 2.2 Why 512 was chosen originally

The 512 configuration was the conservative single-GPU systems point:

1. A full expert has twice the intermediate width, so each selected expert supplies a larger
   nonlinear transform and maps to healthier GEMM tiles.
2. It halves router logits, top-k work, assignment descriptors, expert groups, and load-balancing
   targets relative to 1,024 logical experts.
3. It halves average dispatch fan-out: 8.25 selections instead of 16.5.
4. It lowers the risk that tiny per-expert token counts turn the MoE into hundreds of inefficient
   GEMVs or very small grouped GEMMs during decode.
5. It was enough specialization to make the 50B/A2B ledger plausible while leaving the custom
   ternary and recurrent kernels as the primary unknowns.

Those are implementation-risk arguments. They do **not** establish that 512 produces the best
quality at equal total and active parameters.

### 2.3 Why more logical experts can still be better

At an equal budget, 1,024 half-experts do not create more stored capacity. They partition the same
capacity more finely. The possible benefit is greater routing granularity: a token composes roughly
16 or 17 smaller nonlinear pieces instead of roughly 8 or 9 larger pieces, allowing more
combinations and more precise allocation.

Recent evidence supports taking that possibility seriously:

- **May 2025:** *The Power of Fine-Grained Experts* shows theoretical expressivity separations and
  controlled experiments favoring matched expert granularity while holding total and active
  parameters roughly fixed. This is evidence for a challenger, not proof at Eventide scale.
- **January 2026:** NVIDIA's LatentMoE work explicitly treats expert diversity and higher top-k as
  accuracy levers, while warning that inference cost is governed by bytes moved and effective
  nonlinear budget rather than sparsity alone.
- **May 2026:** *Slicing and Dicing* reports more than 2,000 pretraining runs and finds expert count
  and granularity among the dominant MoE choices. Its largest reported scale is 6.6B total, so it
  does not settle the Eventide point.

The counterargument is equally real: splitting one SwiGLU FFN into two narrower SwiGLU FFNs is not
an algebraic identity. Nonlinearity occurs inside each expert before their weighted outputs are
combined. More pieces may improve conditional composition, while narrower pieces may lose useful
per-expert feature rank. Only a matched canary can resolve that trade at this width.

### 2.4 Why 1,024 does not automatically hurt weight bandwidth

For `n` independently routed token positions presented to one expert layer, a useful first-order
estimate of the number of unique experts touched is:

```text
U(N, k, n) = N * (1 - (1 - k/N)^n)
```

The 512/8.25 and 1,024/16.5 configurations have identical `k/N`. Because the latter expert is half the
size, their expected unique expert **bytes** are identical under this approximation:

| Verification load at one expert layer | 512 full, K=8.25 | 1,024 half, K=16.5 |
|---|---:|---:|
| `n=32` unique experts | 207.552 full | 415.104 half = 207.552 full-equivalent |
| `n=512` unique experts | 511.875 full | 1,023.750 half = 511.875 full-equivalent |
| Active expert size/token | `8.25 P_e` | `8.25 P_e` |

So doubling logical expert count is not a bandwidth magic trick, but neither does it inherently
double ideal expert-weight bytes. It primarily increases routing, dispatch, metadata, and small-GEMM
costs. At wide speculative verification loads, both configurations nearly exhaust the same total
routed bank in byte-equivalent terms.

This is what “we are limited by the number of experts” meant in the throughput analysis: a large
batch/tree touches most of the expert bank, destroying additional weight reuse. Adding twice as
many half-sized labels does not remove that working-set saturation; it reaches twice as many experts
whose weights are half as large.

### 2.5 Current recommendation

Prototype **1,024 logical micro-experts at `E[K] = 16.5`** as the preferred Eventide candidate, but
co-design the engine around them:

- store them in 512 adjacent two-micro-expert groups while preserving independent addressing;
- use a persistent grouped kernel that coalesces many micro-expert jobs into full hardware tiles;
- fuse paired halves when both receive work, without requiring the router to select pairs;
- make the router and speculative candidate scorer aware of marginal expert bytes and already-hot
  expert groups;
- regularize route locality only as a soft serving cost, never as a rule that collapses the logical
  experts back into 512 fixed pairs;
- retain dropless routing and measure specialization, imbalance, expert starvation, route entropy,
  and tokens per launched tile.

Train a matched **512 experts / `E[K]=8.25` / double-width expert** control. Freeze the winner only
after comparing held-out loss and downstream quality at matched total parameters, active parameters,
tokens, data order, optimizer, and wall-clock serving measurements.

### 2.6 Dynamic K, including K=0

`K=0` is coherent only because the shared path remains active. It lets easy token-pass decisions
skip routed experts. It should be available, but rare unless evidence says otherwise.

The budget distribution matters more than the range printed in a config. With maximum K=32, a mean
of 16.5 makes it mathematically impossible for half the token-pass decisions to use K=0: even if
every remaining decision used K=32, the maximum zero fraction would be 48.4375%. The desired zero
fraction should be far smaller so the controller does not create a gratuitously bimodal policy and
large hard-token working sets. The controller therefore needs:

- an explicit mean routed-parameter constraint, not merely a maximum K;
- per-pass and per-domain K histograms;
- a tail penalty for gratuitous K=32 use;
- calibration against marginal quality gain from each additional micro-expert;
- separate reporting for K, depth, and their product.

## 3. Provisional expert dimensions and exact ledger gate

For a bias-free SwiGLU-like latent expert:

```text
P_expert = 3 * d_latent * d_ff_expert
```

With `d_latent=512` and 32 expert-bearing blocks:

| Configuration | `d_ff_expert` | Parameters/expert | Routed bank | Mean routed/pass |
|---|---:|---:|---:|---:|
| 512 full experts, K=8.25 | 1,792 | 2,752,512 | 45.097B | 0.727B |
| 1,024 half experts, K=16.5 | 896 | 1,376,256 | 45.097B | 0.727B |
| 512 full experts, K=8.25 | 1,856 | 2,850,816 | 46.708B | 0.753B |
| 1,024 half experts, K=16.5 | 928 | 1,425,408 | 46.708B | 0.753B |
| 512 full experts, K=8.25 | 1,920 | 2,949,120 | 48.318B | 0.779B |
| 1,024 half experts, K=16.5 | 960 | 1,474,560 | 48.318B | 0.779B |

The 896/1,792, 928/1,856, and 960/1,920 pairs leave approximately 4.903B, 3.292B, and 1.682B
respectively below a nominal 50B ceiling for the shared experts, latent projections, 28 Mamba-3
blocks, four attention blocks, routers/controllers, mHC, recurrent memory, normalization, CALM/MTP,
and embeddings/head. The expert width cannot be selected from total parameters alone; it must also
close the active-pass ledger below.

### 3.1 What A2B counts

The `0.753B` figure for the 928-wide candidate is **routed experts only**. A2B is the complete set of
weights applied during one recurrent body pass:

```text
active/pass = routed experts
            + always-on shared experts
            + sequence mixers
            + latent down/up projections
            + expert routers and K/depth controls
            + recurrent mHC and memory machinery
            + normalization and other per-pass weights
```

The embedding lookup, final output head, and any truly once-per-generated-token CALM/correction
module are reported separately and are not multiplied by MoRE depth.

The following is a **bridge ledger**, not the final HCA/CSA/mHC ledger. It uses the official
Mamba-3 SISO defaults `expand=2`, `d_state=128`, `headdim=64`, and `ngroups=1`. From the official
March 2026 implementation, one `d_model=2048` SISO mixer has 26,165,632 parameters. The
attention line temporarily uses `4*d_model^2` per attention block and must be replaced by the actual
Eventide-scaled HCA/CSA definitions.

| Per-pass component | Provisional parameters |
|---|---:|
| Routed micro-experts, `d_ff=928`, `E[K]=16.5` | 752,615,424 |
| 32 full-width shared SwiGLU experts, `d_ff=1536` | 301,989,888 |
| 28 Mamba-3 SISO mixers | 732,637,696 |
| Four attention projection placeholders | 67,108,864 |
| 32 latent down/up projection pairs | 67,108,864 |
| 32 hidden-to-1,024 expert routers | 67,108,864 |
| **Bridge subtotal** | **1,988,569,600** |
| mHC, depth/K controllers, memory, norms, exact HCA/CSA delta | **not yet included** |

The new bridge subtotal leaves 11,430,400 parameters below A2B for mHC, depth/K controllers,
memory, norms, and the difference between the attention placeholder and exact HCA/CSA definitions.
If those terms exceed that envelope, lower the learned K budget slightly rather than silently
exceeding A2B. At the current dimensions, each `+1.0` in mean K adds exactly 45,613,056 active
parameters per pass.

The routed bank contains 46,707,769,344 stored parameters and the 32 shared experts contain
301,989,888, leaving 2,990,240,768 below a nominal 50B ceiling for every other stored component.
The final ledger must close both the stored and active budgets simultaneously.

The shared expert now consumes 15.10% of the A2B target, while routed experts consume 37.63% at
mean K=16.5. Shared-to-routed active FFN parameters are 40.13%, down from 58.85% in the former
`d_ff=2048`, K=15 design. This retains a full-width path around the 4x latent bottleneck without
letting universal computation dominate MoRE's specialization budget.

### 3.2 Dynamic active-pass envelope

Using the bridge ledger's fixed 1,235,954,176 non-routed parameters, the current K envelope is:

| Routed K | Routed parameters | Bridge active/pass |
|---:|---:|---:|
| 0 | 0 | 1,235,954,176 |
| 16.5 mean | 752,615,424 | 1,988,569,600 |
| 32 | 1,459,617,792 | 2,695,571,968 |

These values still exclude the explicitly open mHC/controller/memory/norm/HCA/CSA delta. At mean
depth 2, the bridge body applies 3,977,139,200 parameters per generated token on average. At the
absolute `K=32`, depth-5 corner it applies 13,477,859,840, before once-per-token head and CALM
correction work. Recurrence may reuse the same unique weights, but every pass is another parameter
application.

The ledger must separately report:

- stored parameters and packed bytes by tensor family;
- parameters applied once per generated token, once per block, and once per MoRE pass;
- minimum, mean, p95, and maximum active parameters under joint K/depth routing;
- tied and untied embedding/head variants;
- ternary payload, scale/metadata overhead, padding, alignment, and higher-precision islands;
- CALM/MTP and multimodal parameters rather than hiding them in “other.”

## 4. Precision contract

The storage objective is not “force every scalar to ternary.” It is “minimize end-to-end bytes and
runtime while passing quality and stability gates.” The working order is:

1. Start every large affine weight in ternary-aware training and deployment experiments: routed and
   shared experts, Mamba projections, attention projections, HCA/CSA projections, latent projections,
   embedding, and output head.
2. Promote only tensors that fail a paired loss, stability, or downstream-quality gate, using the
   lowest precision that fixes the failure.
3. Keep reductions, accumulators, dynamic scales, state updates, normalization statistics, and
   fragile controller logits at their lowest validated higher precision.
4. Report effective whole-model bits/weight including scales, packing slack, alignment, and every
   non-ternary tensor. “98% ternary” is not a byte ledger.

Packed radix-3 storage at 20 trits per 32 bits is approximately 1.6 payload bits/weight. This is a
storage fact, not a claim of native 1.6-bit arithmetic. The RTX 5070 Ti has native FP4 paths, not a
native ternary MMA; decode and prefill require different custom kernels and measured useful rates.

The embedding/head choice remains open. Untying costs one additional vocabulary matrix in storage,
but the output head must still execute whether the weights are tied or untied. The decision is a
quality/storage ablation; it is not a reason to remove the head from throughput accounting.

## 5. Exact custom serving stack

The intended serving path is:

1. Restore the longest exact model-state prefix from a token-keyed radix tree/block DAG.
2. Keep hot state in VRAM, warm state in pinned system RAM, and dormant checkpoints on NVMe.
3. Generate a small topology-aware candidate tree from the CALM/MTP signal, N-gram suffix sidecar,
   and recent accepted path.
4. Score branches by probability **and** marginal Eventide cost: new expert bytes, expected MoRE depth,
   Mamba/HCA/CSA state work, and output-head work.
5. Verify the tree with the full target model using branch-aware Mamba scans, copy-on-write HCA/CSA
   and mHC state, and packed active-token layouts.
6. Apply an exact rejection/correction rule and commit only the accepted branch state.

CALM is not assumed lossless by itself, and its K is not multiplied naively by a tree-speculation
acceptance length. In exact mode it supplies training signal, proposals, or a draft distribution;
the full target verifier determines the final samples.

The previous 512-expert roofline is a sensitivity model, not a performance claim. The 1,024
micro-expert design must rerun the ledger with measured router, top-k, dispatch, unpack, grouped
GEMM, state-scan, and output-head kernels. Thirteen thousand accepted tokens/s remains a target
until those measurements and end-to-end acceptance traces exist.

The routing input to that rerun is now `E[K]=16.5`: under independent routing, B32 touches an
expected 415.104 half-experts per layer, while a 512-node verification load touches 1,023.750 and
therefore nearly saturates the bank. The smaller shared expert saves deterministic weight traffic,
but ordinary B32 autoregression touches more routed expert weights than the historical K=15 model.

## 6. Cache contract

Conversation state is durable and multi-entry, not limited to the most recent call:

- stable global/system prefixes can be shared when model version, tokenizer, template, permissions,
  adapters, and multimodal inputs all match;
- every user thread owns an immutable chain of exact state checkpoints;
- a new turn restores the checkpoint after the previous assistant response and prefills only the
  new suffix;
- branches are retained for regeneration, editing, and undo;
- semantic response caching is separate from exact hidden-state reuse.

Each checkpoint must validate component-specific state: exact attention prefixes, Mamba recurrent
state at the precise boundary, HCA/CSA index state, mHC streams, CALM configuration, N-gram version,
precision recipe, and tenant/user salt.

## 7. Decision gates before the architecture is frozen

1. Produce an executable parameter/byte ledger for both 1,024/16.5 and the matched 512/8.25 control,
   with `d_ff=1536` full-width shared experts in both.
2. Benchmark representative ternary expert GEMV and grouped-GEMM shapes on the actual RTX 5070 Ti.
3. Benchmark router + top-k + dispatch separately at N=512 and N=1,024.
4. Run matched small-scale pretraining canaries for 512/full and 1,024/half, including K/depth joint
   routing stability.
5. Reconcile `d_ff=928`, shared `d_ff=1536`, and the learned K target against the exact non-routed
   ledger; adjust mean K before changing the stored expert bank.
6. Ablate tied versus untied embedding/head and ternary versus NVFP4 head on identical held-out data.
7. Validate every proposed ternary tensor family independently before setting the final precision
   allowlist.
8. Measure DARTree-style acceptance and expert union at 262K context with real Eventide route traces.
9. Treat the 13k batch-32 target as unproven until an end-to-end exact-serving benchmark reaches it.

## 8. Primary research anchors

- **May 2025:** Boix-Adsera and Rigollet, [The Power of Fine-Grained Experts](https://arxiv.org/abs/2505.06839).
- **December 2025 / January 2026:** NVIDIA, [Nemotron 3 white paper](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf) and [LatentMoE](https://research.nvidia.com/labs/nemotron/LatentMoE/).
- **March 2026:** [Mamba-3 paper](https://arxiv.org/abs/2603.15569) and
  [official SISO implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py).
- **May 2026:** Li et al., [Slicing and Dicing: Configuring Optimal Mixtures of Experts](https://arxiv.org/abs/2605.11689).
- **February 2026:** [DFlash](https://arxiv.org/abs/2602.06036).
- **June 2026:** [CaDDTree](https://arxiv.org/abs/2606.01813).
- **July 2026:** [EcoSpec](https://arxiv.org/abs/2607.12696).
- **August 2026:** [DARTree](https://arxiv.org/abs/2608.13524).
