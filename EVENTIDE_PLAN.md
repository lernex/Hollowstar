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
| Total neural-core parameters | **50,131,285,109** with untied embedding/head and N-gram fusion, before CALM and multimodal auxiliaries | Reconciled core ledger |
| External N-gram memory | **200,000,000,000** learned sparse values in system RAM/NVMe | Working default; separate from neural core |
| Average active parameters | **2,000,000,000 unique parameters per MoRE pass** at `E[K] = 15.56547` | Reconciled core ledger |
| MoRE depth | 1–5 passes, target mean approximately 2 | Working default |
| Model width | `d_model = 2048` | Working default |
| Routed latent width | `d_latent = 512` | Working default |
| Unique backbone blocks | 32 | Working default |
| Backbone layout | `7 Mamba-3 SISO -> HCA -> 7 Mamba-3 SISO -> CSA`, repeated twice | Working default |
| Mixer counts | 28 Mamba-3 SISO, 2 HCA, 2 CSA | Working default |
| Residual topology | four-stream, recursion-aware mHC | Working default |
| Routed experts per expert-bearing block | **1,024 logical micro-experts** | Preferred candidate; see section 2 |
| Physical expert packing | 512 adjacent two-micro-expert groups, independently addressable | Preferred engine layout |
| Routed expert width | `d_ff = 960` | Reconciled, 64-aligned working default |
| Routed K | integer `K in [0, 32]`, target `E[K] = 15.56547` | Reconciled A2B working default |
| Shared path | one full-width shared expert per block, `d_model=2048`, `d_ff=1536` | Working default |
| Vocabulary | 131,072 | Working default |
| Embedding/head | untied working default; ternary head if it passes parity gates | Quality/storage ablation remains required |
| Weight storage | packed ternary target for large matrices, approximately 1.6 bpw | Target, not measured execution rate |
| Prefill matrix activations | **NVFP4 A4 target** with Hadamard outlier control and BF16 state boundaries | Preferred candidate; Eventide QAT gate required |
| Higher-precision islands | smallest validated type for state, scales, reductions, normalization, and fragile controls | Tensor-by-tensor gate |
| Standard context | 262,144 original tokens | Working default |
| Extended context | 524,288 and 1,048,576 evaluation tiers | Open release gate |
| N-gram memory | 200B-value, 16-table deterministic suffix-hash memory; four HCA/CSA injection sites; exact sidecar | Working capacity; storage layout remains gated |
| MTP | CALM-style training signal and proposal feature | Exact-serving integration open |
| Serving | batch-32 exact tree speculation with durable multi-tier state cache | Custom implementation required |

The routed-capacity-matched system control is **512 experts at `d_ff=1920` and average
K 7.78273**. Because its dense router is smaller, that control lands at 1.98694B active/pass; a
strict A2B compute-matched control instead uses average K 7.92108. The earlier 512/7.5 and
1,024/15 pair remains a historical throughput reference, not the current reconciled ledger.

## 2. Expert-count decision

### 2.1 The normalization that must not drift

Let a 512-expert baseline expert contain `P_e` parameters. At the reconciled routed-compute point:

```text
512 experts, mean K = 7.78273:
  stored routed bank per layer = 512 * P_e
  active routed parameters     = 7.78273 * P_e

1024 half-sized experts, mean K = 15.56547:
  stored routed bank per layer = 1024 * (P_e / 2) = 512 * P_e
  active routed parameters     = 15.56547 * (P_e / 2) = 7.78273 * P_e
```

Therefore the exact iso-total, iso-active conversion is:

```text
N:       512 -> 1024
size:    P_e -> P_e / 2
mean K:  7.78273 -> 15.56547
max K:   16 -> 32
```

Mean `K = 15.56547` never means executing a fractional expert. K is integer for every token at
every pass; the fractional value is the mean over routed token-pass decisions. It must arise from
a learned distribution over integer K values with that budget.

Relative to the historical 1,024/15 design, the reconciled mean raises routed activity by 3.77%:

```text
15.56547 / 15 - 1 = 3.770%
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
3. It halves average dispatch fan-out: 7.78273 selections instead of 15.56547.
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

The 512/7.78273 and 1,024/15.56547 configurations have identical `k/N`. Because the latter expert is half the
size, their expected unique expert **bytes** are identical under this approximation:

| Verification load at one expert layer | 512 full, K=7.78273 | 1,024 half, K=15.56547 |
|---|---:|---:|
| `n=32` unique experts | 198.384 full | 396.768 half = 198.384 full-equivalent |
| `n=512` unique experts | 511.799 full | 1,023.598 half = 511.799 full-equivalent |
| Active expert size/token | `7.78273 P_e` | `7.78273 P_e` |

So doubling logical expert count is not a bandwidth magic trick, but neither does it inherently
double ideal expert-weight bytes. It primarily increases routing, dispatch, metadata, and small-GEMM
costs. At wide speculative verification loads, both configurations nearly exhaust the same total
routed bank in byte-equivalent terms.

This is what “we are limited by the number of experts” meant in the throughput analysis: a large
batch/tree touches most of the expert bank, destroying additional weight reuse. Adding twice as
many half-sized labels does not remove that working-set saturation; it reaches twice as many experts
whose weights are half as large.

### 2.5 Current recommendation

Prototype **1,024 logical micro-experts at `d_ff=960`, `E[K] = 15.56547`** as the preferred
Eventide candidate, but co-design the engine around them:

- store them in 512 adjacent two-micro-expert groups while preserving independent addressing;
- use a persistent grouped kernel that coalesces many micro-expert jobs into full hardware tiles;
- fuse paired halves when both receive work, without requiring the router to select pairs;
- make the router and speculative candidate scorer aware of marginal expert bytes and already-hot
  expert groups;
- regularize route locality only as a soft serving cost, never as a rule that collapses the logical
  experts back into 512 fixed pairs;
- retain dropless routing and measure specialization, imbalance, expert starvation, route entropy,
  and tokens per launched tile.

Train a routed-capacity-matched **512 experts / `d_ff=1920` / `E[K]=7.78273`** control, and
separately report the strict-A2B `E[K]=7.92108` systems point. Freeze the winner only
after comparing held-out loss and downstream quality at matched total parameters, active parameters,
tokens, data order, optimizer, and wall-clock serving measurements.

### 2.6 Dynamic K, including K=0

`K=0` is coherent only because the shared path remains active. It lets easy token-pass decisions
skip routed experts. It should be available, but rare unless evidence says otherwise.

The budget distribution matters more than the range printed in a config. With maximum K=32, a mean
of 15.56547 caps the zero fraction at 51.3579% even under the pathological distribution where every
nonzero decision uses K=32. The desired zero fraction should be far smaller so the controller does
not create a gratuitously bimodal policy and
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

| Configuration | `d_ff_expert` | Parameters/expert | Routed bank | Routed/pass at reconciled matched K |
|---|---:|---:|---:|---:|
| 512 full experts, K=7.78273 | 1,792 | 2,752,512 | 45.097B | 0.686B |
| 1,024 half experts, K=15.56547 | 896 | 1,376,256 | 45.097B | 0.686B |
| 512 full experts, K=7.78273 | 1,856 | 2,850,816 | 46.708B | 0.710B |
| 1,024 half experts, K=15.56547 | 928 | 1,425,408 | 46.708B | 0.710B |
| 512 full experts, K=7.78273 | 1,920 | 2,949,120 | 48.318B | 0.734B |
| 1,024 half experts, K=15.56547 | 960 | 1,474,560 | 48.318B | 0.734B |

The 896/1,792, 928/1,856, and 960/1,920 pairs leave approximately 4.903B, 3.292B, and 1.682B
respectively below a nominal 50B ceiling for the shared experts, latent projections, 28 Mamba-3
blocks, four attention blocks, routers/controllers, mHC, recurrent memory, normalization, CALM/MTP,
and embeddings/head. The expert width cannot be selected from total parameters alone; it must also
close the active-pass ledger below.

### 3.1 Reconciled A2B ledger

The `0.734B` figure for the 960-wide candidate is **routed experts only**. A2B is the complete set
of weights applied during one recurrent body pass:

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
module are reported separately and are not multiplied by MoRE depth. The calculations below are
reproducible with `python3 scripts/eventide_parameter_ledger.py`.

#### Eventide-scaled HCA and CSA

The attention dimensions now scale the official **June 2026** DeepSeek-V4 design instead of using
`4*d_model^2` placeholders:

```text
d_model=2048, shared-KV head c=256, partial-RoPE=32, 64 query heads, query rank=512
8 grouped-output groups, output rank/group=512
CSA m=4, index heads=32, index head dim=128, top-k=512
HCA m'=128, local window=128
```

The common attention core is 26,739,520 parameters. HCA adds a 1,081,600-parameter compressor,
giving **27,821,120 per HCA block**. CSA adds a 2,099,456-parameter overlapping compressor and a
3,212,416-parameter Lightning Indexer, giving **32,051,392 per CSA block**. Two of each total
**119,745,024** parameters. Learned core/compressor/indexer norms are already included in those
numbers.

#### mHC, routers, memory, and norms

The mHC count uses four streams and the full input-dependent DeepSeek mapping rather than the much
smaller historical pass-only controller. For each of the two mHC sites in every block:

```text
dynamic fn:       24 outputs * (4 * 2048) inputs = 196,608
static base:                                         24
three scales:                                          3
pass controller:            64 * 24 + 24 =         1,560
per connection:                                  198,195
64 connections:                              12,684,480
global pass + stream embeddings:                  8,512
total mHC:                                    12,692,992
```

The controller and recurrent-memory arithmetic is:

| Family | Stored parameters | Active/pass treatment |
|---|---:|---:|
| Expert-logit routers, `768 -> 1024`, 32 sites | 25,198,592 | all applied |
| K routers, `768 -> 33`, 32 sites | 812,064 | all applied |
| Expert route embeddings, `1024 x 256`, 32 sites | 8,388,608 | only `32*K*256` selected rows |
| Global expert-aware recurrent depth memory | 3,423,236 | 3,422,212 unique parameters touched; weights reused within pass |
| Global continuation/depth controller, `6400 -> 256 -> 1` | 1,638,913 | all applied |
| N-gram projection, `1024 -> 2048` | 2,099,200 | once before recurrent passes |
| N-gram injection gates, four sites x five passes | 163,920 | 32,784 selected gate values/pass |
| Two learned block RMSNorms per block | 131,072 | all applied |
| Final RMSNorm | 2,048 | once after recursion |

The depth-memory total includes state and metadata writers, pass and attention-anchor embeddings,
query/key/value/output projections, the `6400 -> 256` route-feature projection, and four stream
gates. Exactly: state write 524,544; metadata write 66,816; pass embeddings 1,280; anchor
embeddings 1,280; query 524,544; key 65,792; value 65,792; output 526,336; route projection
1,638,656; and stream gate 8,196. The continuation controller is a 1,638,656-parameter hidden
projection plus a 257-parameter output projection. The learned norm inventory is 252,160
parameters in the whole core: 131,072 block norms,
114,688 Mamba internal norms, 4,352 HCA/CSA norms, and the 2,048 final norm. The latter three
families are already included in their parent rows and must not be added twice. mHC's flattened
input RMSNorm is unweighted and contributes no learned parameters.

One pass touches **12,684,544** of the stored mHC parameters: all 64 connection modules and one
64-value pass-embedding row. The 8,192 stream-seed values are applied once when the token enters the
recurrent body. Likewise, one pass touches one of the five 256-value depth-memory pass embeddings,
which is why its unique-active count is 1,024 below its stored count.

Non-neural runtime control state is also explicit: 131,072 bytes of FP32 expert selection biases,
128 bytes of K-budget multipliers, and one 4-byte depth-budget multiplier. Those bytes matter to an
implementation ledger but are not model parameters.

#### Exact per-pass result

| Per-pass component | Active parameters |
|---|---:|
| Routed micro-experts, `d_ff=960`, `E[K]=15.56547` | 734,470,835 |
| Selected expert route-embedding rows | 127,512 |
| 32 full-width shared SwiGLU experts, `d_ff=1536` | 301,989,888 |
| 28 Mamba-3 SISO mixers | 732,637,696 |
| 2 HCA + 2 CSA blocks | 119,745,024 |
| 32 latent down/up projection pairs | 67,108,864 |
| Expert-logit routers | 25,198,592 |
| K routers | 812,064 |
| Recursion-aware mHC | 12,684,544 |
| Recurrent depth memory | 3,422,212 |
| Continuation/depth controller | 1,638,913 |
| Selected N-gram injection gates | 32,784 |
| Per-block learned norms not included above | 131,072 |
| **Average body active/pass** | **2,000,000,000** |

The fixed K-independent body is **1,265,401,653** unique parameters. Every `+1.0` in K adds
**47,194,112** active parameters: 47,185,920 expert weights plus 8,192 selected route-embedding
values. Solving `(2B - fixed) / slope` gives **E[K] = 15.5654660268**.

The 960 width is preferred over 928 because it is a clean multiple of 64 and closes both budgets.
With untied embedding/head matrices and the N-gram fusion path, the neural core contains
**50,131,285,109 stored parameters**; tied would contain 49,862,849,653. This is the first
configuration here that reconciles stored capacity
and A2B without an unexplained “other” bucket. CALM and multimodal auxiliaries remain outside that
number until they have concrete executable definitions; they must be reported as additions, not
silently absorbed.

| Stored core family | Parameters |
|---|---:|
| Routed expert bank | 48,318,382,080 |
| Shared experts | 301,989,888 |
| Mamba-3 SISO | 732,637,696 |
| HCA | 55,642,240 |
| CSA | 64,102,784 |
| Latent down/up | 67,108,864 |
| Expert-logit routers | 25,198,592 |
| K routers | 812,064 |
| Expert route embeddings | 8,388,608 |
| Recursion-aware mHC | 12,692,992 |
| Recurrent depth memory | 3,423,236 |
| Continuation/depth controller | 1,638,913 |
| N-gram projection | 2,099,200 |
| N-gram injection gates | 163,920 |
| Per-block learned norms | 131,072 |
| Token embedding | 268,435,456 |
| Untied output head | 268,435,456 |
| Final norm | 2,048 |
| **Untied neural-core total** | **50,131,285,109** |

The untied 131,072-by-2,048 output head applies **268,435,456 parameters once per output token**,
regardless of MoRE depth. The once-per-token N-gram lookup and projection add 2,100,224 parameter
applications; an input embedding row, final norm, and four stream seeds add another 12,288.
Therefore conventional A2B accounting gives 4.000B summed unique-active body parameters at mean
depth 2 and **4.270548B** after the model interface, before
CALM proposal/correction work.

That still is not the literal invocation count. The 3.422M unique-active depth-memory module is
reused throughout a pass: its projections account for 86,801,508 parameter applications on the
first pass and 96,326,788 on a later pass. At mean depth 2, the invocation-weighted body count is
therefore **4,176,283,872**, and the head/interface-inclusive count is **4,446,831,840**. These
extra applications are mostly small repeated matrices that should remain cache-resident; they
matter to FLOPs and kernel launches but should not be mistaken for another 176M of unique streamed
weights.

### 3.2 Dynamic active-pass envelope

| Routed K | Body active/pass | Body applications at indicated depth |
|---:|---:|---:|
| 0 | 1,265,401,653 | 1,265,401,653 at depth 1 |
| 15.56547 mean | 2,000,000,000 | 4,000,000,000 unique-active sum at mean depth 2 |
| 32 | 2,775,613,237 | 13,878,066,185 unique-active sum at depth 5 |

Adding the once-per-token interface gives 4,270,547,968 average and 14,148,614,153 worst-corner
conventional counts. Including every repeated depth-memory invocation raises those to 4,446,831,840
and 14,603,611,753 respectively. Recurrence may reuse the same unique weights, but every invocation
still costs arithmetic.

### 3.3 External 200B N-gram memory

The working N-gram capacity is **200B learned scalar values**, not 200B rows. If “200B” instead
means rows, the 64-value layout becomes 12.8T values and approximately 2.692 TiB with radix-3
payload plus one BF16 scale per row. Retain the Eventide
layout of suffix orders 2 and 3, eight independently hashed tables per order, and 64-value rows:

```text
tables                               = 2 * 8 = 16
logical rows                         = 200B / 64 = 3.125B
mean rows/table                      = 195,312,500
rows retrieved/token                 = 16
learned table values retrieved/token = 16 * 64 = 1,024
concatenated vector                  = 1,024
fusion projection                    = 1,024 -> 2,048
injection sites                      = HCA/CSA blocks 7, 15, 23, 31
```

Suffix hashes directly address the rows, so there is no learned 200B-way router. The learned
decision surface is the 2,099,200-parameter fusion projection and 163,920 stored stream-gate
parameters. The lookup and projection run once per input token and are cached across MoRE passes;
only 32,784 gate values are selected during one pass. Those gates are already included in the exact
A2B result above.

The lossless radix-3 payload is exactly 40,000,000,000 bytes, or **37.253 GiB**. Giving every
64-value row an independent BF16 scale adds 6,250,000,000 bytes, producing **43.074 GiB** before
page alignment, checksums, collision metadata, and the exact suffix sidecar. Packing scales more
coarsely can approach the 37.253-GiB floor, but per-row scaling is the safer quality assumption for
random row retrieval.

Therefore the complete learned system is **250,131,285,109 logical parameters/values**: a 50.131B
neural core plus a 200B sparse conditional memory. The table is not part of GPU weight residency and
does not turn A2B into A202B; only 1,024 table values are retrieved per token. It does, however,
require a real RAM/NVMe hierarchy. A 64-GiB host can hold the scaled table but leaves little room for
the OS, prefix cache, and file cache; **128 GiB host RAM is the comfortable all-resident target**.
With less RAM, hot rows must remain resident and SSD misses must be batched/coalesced by page, because
sixteen uncoalesced 4-KiB reads per generated token would erase the intended serving speedup.

### 3.4 What mHC and recurrent memory are for

mHC and recurrent memory solve different information-preservation problems.

The four-stream mHC path is the residual transport system. Instead of forcing every sublayer to
read and overwrite one residual vector, it maintains four parallel streams, dynamically composes
the sublayer input from them, and distributes the sublayer output back across them. The
manifold-constrained mixing keeps that transport stable rather than allowing arbitrary recurrent
gain. The intended intelligence benefit is not 12.7M parameters of additional knowledge. It is
less destructive feature interference: lexical evidence, retrieved context, an evolving reasoning
state, and routing/control evidence can remain partially separated and be recombined differently
by layer and pass. For Eventide, making the small controller pass-aware should also let a later MoRE
pass use the same backbone as a refinement pass rather than merely repeating pass one. These are
architectural hypotheses that require an mHC-versus-single-residual ablation.

The 256-dimensional recurrent depth memory is the within-token scratchpad and control trace. Four
attention anchors plus the end of each pass write compressed state and routing metadata; later
blocks and later passes retrieve it. Its route features combine the current state, retrieved
memory, their difference, and route history, while the continuation controller sees the memory
summary when choosing whether another pass is worthwhile. Its intended jobs are therefore:

- remember what the previous pass attended to, attempted, and left unresolved;
- change expert choice and K on the next pass instead of replaying the same route;
- let difficult tokens request more routed width or another pass while easy tokens exit;
- carry compact plan/progress signals across the Mamba/HCA/CSA backbone without placing another
  full residual sequence in memory.

It is **not** the 262K context store, the cross-token Mamba state, a conversation cache, or the
200B N-gram table. The failure modes are equally explicit: the model may ignore it, always choose
maximum depth, collapse onto repetitive routes, or homogenize the four streams. Training must log
memory use, route change by pass, depth calibration, and task quality, and compare against a
parameter-matched controller without recurrent memory.

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
5. Target W1.58A4-NVFP4 for large prefill GEMMs: keep weights packed as ternary in VRAM, expand
   them losslessly into FP4 tensor-core tiles, quantize the corresponding matrix activation to
   block-scaled NVFP4, accumulate in FP32, and return to a BF16 state boundary.

Packed radix-3 storage at 20 trits per 32 bits is approximately 1.6 payload bits/weight. This is a
storage fact, not a claim of native 1.6-bit arithmetic. The RTX 5070 Ti has native FP4 paths, not a
native ternary MMA; decode and prefill require different custom kernels and measured useful rates.

Untied embedding/head weights are the working default, not a frozen conclusion. Untying costs one
additional vocabulary matrix in storage, but the output head must still execute whether the weights
are tied or untied. The release decision remains a quality/storage ablation; it is not a reason to
remove the head from throughput accounting.

The A4 decision does **not** make every live value four-bit. mHC residual streams, Mamba recurrent
state, recurrent depth memory, normalization and softmax reductions, controller logits, and other
stateful feedback remain BF16/FP32 until individual evidence supports promotion. A4 describes the
operands of the large matrix multiplications; those operands are transient tiles rather than the
authoritative residual representation.

Microsoft's **June 2025** BitNet-v2 evidence supports the direction but not an unconditional
NVFP4-quality claim. It tested ternary weights with **INT4** activations, per-token absmean scaling,
and online Hadamard transforms for outlier-prone attention-output and FFN-down inputs. Models were
trained for 95B tokens in A8 and continue-trained for 5B tokens in A4. At 7B, A4 reported average
downstream accuracy 58.30 versus 58.12 for BitNet b1.58, while perplexity moved from 9.09 to 9.24;
relative to BitNet-v2 A8, average accuracy moved from 58.73 to 58.30. That is small, not literally
zero, and the largest experiment was 7B rather than a recurrent 50B hybrid.

An **August 2026** literature check finds an active ternary field, but the newer work is centered on
post-training quantization and inference rather than native pretraining at Eventide's scale:

- **February 2026:** TernaryLM trains ternary weights natively, but only in a 132M-parameter
  TinyStories-scale Transformer.
- **June 2026:** TWLA reports W1.58A4 through post-training quantization, orthogonal distribution
  shaping, and activation mixed precision; it does not establish native NVFP4 training.
- **June 2026:** CAT-Q ternarizes pretrained 14B-to-235B models, but its result is post-training
  conversion from higher-precision checkpoints rather than ternary pretraining.
- **August 2026:** ScaleQ-1.58 extends ternary post-training quantization to reasoning and MoE models
  up to 235B, again without validating a native ternary recurrent model.

BitNet-v2 therefore remains the closest published precedent for native A4 continuation despite its
age. None of these results closes the Eventide-specific gate for NVFP4 activations, a recurrent
LatentMoE body, 50B-scale native training, or fused execution on an RTX 5070 Ti.

Therefore Eventide should include the Hadamard-aware topology from the beginning, retain A8 for the
main training phase, then run an NVFP4-QAT continuation phase over at least the final 5% of training
tokens. INT4 and NVFP4 must be evaluated separately. The fallback checkpoint remains A8 until the
paired loss, long-context retrieval, routing/depth calibration, and downstream suite pass.

### 4.1 B32 x 262K decode-residency estimate

The NVIDIA reference specification is 16 GB of GDDR7. The following uses a more generous 16 GiB
arithmetic ceiling, so it is a design budget rather than a promise of allocatable VRAM. It assumes
compressed attention state: partial-RoPE key dimensions in BF16, the remaining main-cache values in
FP8, FP4 index rows with grouped scales, HCA compression 128, CSA compression 4, and a local window
of 128.

| Resident family | GiB |
|---|---:|
| 50.131B core, pure 1.6-bpw payload floor | 9.338 |
| Core with BF16 scale per 128 ternary weights, all tensors ternary | 10.067 |
| Conservative core: 98% at 1.725 bpw, 1% NVFP4 at 4.5 bpw, 1% FP8 | **10.595** |
| B32 x 262K HCA/CSA/index/local attention state | **1.454** |
| B32 Mamba SSM and convolution state | **0.904** |
| Conservative weights plus sequence state | **12.953** |
| Arithmetic room below 16 GiB | **3.047** |

Consequently, the target model itself still fits in this optimistic deployment recipe. The exact
CALM/DFlash hybrid does **not yet have a truthful final fit number**, because its executable draft
architecture and verifier workspace are not specified. The **October 2025** CALM K=4 recipe
contains a 75M autoencoder and separate 371M, 735M, and 1.82B continuous models; **February 2026**
DFlash is likewise a real lightweight block-diffusion draft, not a zero-parameter scheduling trick.
The Eventide implementation must keep all draft/autoencoder weights, live speculative-tree state,
verification scratch, unpack workspace, CUDA context, and allocator slack inside the remaining
3.047 GiB.

The working engineering gate is therefore **at most 1.5 GiB for CALM/DFlash weights plus dynamic
verification workspace**, leaving roughly 1.55 GiB for CUDA/allocator slack. A 2.0-GiB auxiliary
budget is a red-line prototype, not the production target. B32 can fit on paper under that cap, but
it remains tight and may fail on a display-attached GPU or with fragmentation. B16 halves the
sequence-state portion to 1.179 GiB and leaves 4.226 GiB before auxiliary/workspace allocations, so
it is the required fallback. The 200B N-gram table stays in system RAM/NVMe;
placing it in VRAM would make this configuration impossible.

### 4.2 B32 x 262K W1.58A4 prefill roofline

The RTX 5070 Ti's official dense peak is **703 FP4 TFLOP/s with FP32 accumulation**; the advertised
1,406 AI TOPS is the 2:4 sparse figure. Ordinary ternary zeros are not automatically structured 2:4
sparsity, so this ledger uses 703 and gives no sparse credit.

For one completely uncached 262,144-token prefix in each of 32 streams, mean MoRE depth 2, and
inference logits computed only at the final prompt position, the counted work per input token is:

| Work per input token | GFLOP |
|---|---:|
| Recurrent body and once-per-token learned projections | 8.357 |
| HCA compressed plus local attention | 0.302 |
| CSA Lightning Indexer scan | 1.074 |
| CSA selected plus local core attention | 0.168 |
| Mamba selective-scan estimate | 0.147 |
| **Counted total** | **10.047** |

The vocabulary head is not applied to every prompt token during serving, which is why the prefill
count is lower than the decode head-inclusive invocation count. The attention terms include causal
average history at 262K: HCA compression 128, CSA compression 4, CSA top-k 512, and local window
128. The executable ledger deliberately leaves online Hadamard transforms, NVFP4 quantization,
ternary tile expansion, softmax, top-k, synchronization, and kernel launches in the end-to-end
utilization discount instead of pretending they are FP4 tensor-core work.

| Equivalent end-to-end share of dense FP4 peak | B32 aggregate input tok/s | Fair-share tok/s per stream | Time for 32 full 262K prefixes |
|---:|---:|---:|---:|
| 100% mathematical ceiling | 69,970 | 2,187 | 119.9 s |
| 98% optimistic ceiling | **68,571** | **2,143** | **122.3 s** |
| 85% exceptional engine target | **59,475** | **1,859** | **141.0 s** |
| 75% strong engine target | 52,478 | 1,640 | 159.9 s |
| 60% conservative target | 41,982 | 1,312 | 199.8 s |
| 50% | 34,985 | 1,093 | 239.8 s |

The honest target band is **52K–59K aggregate raw prefill tok/s**, or **1.64K–1.86K per stream at
B32**, until the fused kernel is measured. The 68.6K result is a roofline, not a forecast. Prefix
cache hits can avoid most of this work; CALM/DFlash do not multiply prompt-prefill throughput.
A4 halves transient matrix-activation tile bytes relative to A8 and should improve batched kernel
occupancy, but it does not halve the persistent HCA/CSA or Mamba sequence state. Consequently, it
does not make B64 x 262K fit in 16 GiB under the current state layout.

MoRE depth remains a first-order sensitivity: at the same artificial 98% equivalent utilization,
mean prefill depth 1.5 gives approximately 91.5K aggregate / 2.86K per stream, and depth 1 gives
137.3K / 4.29K. These are not the working forecast; depth 2 remains the budget until real routing
traces prove prompt tokens exit earlier.

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

The routing input to that rerun is now `E[K]=15.56547`: under independent routing, B32 touches an
expected 396.768 half-experts per layer, while a 512-node verification load touches 1,023.598 and
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

1. Extend the executable parameter ledger into a tensor-level byte ledger for both
   1,024/15.56547 and the matched 512/7.78273 control, with `d_ff=1536` full-width shared experts
   in both.
2. Benchmark representative ternary expert GEMV and grouped-GEMM shapes on the actual RTX 5070 Ti.
3. Benchmark router + top-k + dispatch separately at N=512 and N=1,024.
4. Run matched small-scale pretraining canaries for 512/full and 1,024/half, including K/depth joint
   routing stability.
5. Keep `d_ff=960`, shared `d_ff=1536`, and the learned K target synchronized with
   `scripts/eventide_parameter_ledger.py`; adjust mean K before changing the stored expert bank.
6. Ablate tied versus untied embedding/head and ternary versus NVFP4 head on identical held-out data.
7. Validate every proposed ternary tensor family independently before setting the final precision
   allowlist.
8. Measure DARTree-style acceptance and expert union at 262K context with real Eventide route traces.
9. Treat the 13k batch-32 target as unproven until an end-to-end exact-serving benchmark reaches it.

## 8. Primary research anchors

- **June 2025, retained by Microsoft's current July 2026 BitNet index:**
  [BitNet v2 native A4](https://arxiv.org/abs/2504.18415) and the
  [current official BitNet repository](https://github.com/microsoft/BitNet).
- **February 2026:** [TernaryLM native 1.5-bit training](https://arxiv.org/abs/2602.07374).
- **June 2026:** [TWLA W1.58A4 post-training quantization](https://arxiv.org/abs/2606.13054) and
  [CAT-Q ternary post-training quantization](https://arxiv.org/abs/2606.26650).
- **August 2026:** [ScaleQ-1.58 reasoning-model post-training quantization](https://arxiv.org/abs/2608.01078).
- **May 2025:** Boix-Adsera and Rigollet, [The Power of Fine-Grained Experts](https://arxiv.org/abs/2505.06839).
- **December 2025 / January 2026:** NVIDIA, [Nemotron 3 white paper](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-White-Paper.pdf) and [LatentMoE](https://research.nvidia.com/labs/nemotron/LatentMoE/).
- **March 2026:** [Mamba-3 paper](https://arxiv.org/abs/2603.15569) and
  [official SISO implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py).
- **June 2026:** DeepSeek-AI, [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348) and
  [official Flash configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base/blob/main/config.json).
- **May 2026:** Li et al., [Slicing and Dicing: Configuring Optimal Mixtures of Experts](https://arxiv.org/abs/2605.11689).
- **February 2026:** [DFlash](https://arxiv.org/abs/2602.06036).
- **October 2025:** [Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688)
  and the [official CALM implementation](https://github.com/shaochenze/calm).
- **June 2026:** [CaDDTree](https://arxiv.org/abs/2606.01813).
- **July 2026:** [EcoSpec](https://arxiv.org/abs/2607.12696).
- **August 2026:** [DARTree](https://arxiv.org/abs/2608.13524).
- **RTX 5070 Ti hardware:** [NVIDIA reference specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/).
- **RTX 5070 Ti precision peaks:** [NVIDIA RTX Blackwell architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
