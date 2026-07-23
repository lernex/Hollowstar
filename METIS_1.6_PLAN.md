# Metis-1.6 — MoRE Architecture & Training Plan

Status: architecture draft; **pretraining data manifest v1.0 locked at 1T tokens**.
Supersedes 1.5 (single-latent MoE, dense). Folds in the 1.5 post-mortem findings
(`METIS_1.6_NOTES.md`). The canonical data recipe and executable release gates
live in `manifests/metis-1.6.yaml` and `docs/metis16_pretraining_data_plan.md`.

## 0. North star
**MoRE = per-token, three-axis dynamic routing:** recursion **depth** (how many passes), expert
**width** (how many experts per pass), and expert **pathway** (which experts at each pass). Depth and
width are the two compute-budget axes; pathway dynamically changes the composition of that compute.
Hard tokens get more depth and width, while every continuing token can re-route as its recurrent
hidden state evolves — nothing over- or under-shot. Goal: 1.5's efficiency instincts, but with real
reasoning depth, **1T final-tokenizer-measured pretraining exposures, locked** (the fused additions
below and the Portage topology still require a fresh throughput probe),
and a deliberate strategic bet — **a tool-using reasoner, not an
encyclopedia** (§1.5). Backbone shifts to **hybrid Mamba-2 + attention** for cheap O(n) long-context/RAG.

## 1. Architecture (what MoRE is)

Three dynamic decisions, implemented by the continuation router and the per-pass expert router, determine
each token's compute and processing path:

1. **Depth (per-pass continuation MoR).** Every token receives the first pass. After pass `r`, a
   continuation router decides whether that token halts or enters pass `r+1`, conditioned on its
   current recurrent state, retrieved depth memory, and prior route metadata. The active set is
   monotonic (`A_{r+1} ⊆ A_r`) and packed at every pass. Depth remains **capped** at 5 passes
   (4 recursions), with **target mean ≈2 passes**. The cap and monotonic packed schedule are what
   keep execution bounded and compilable (see §4). Convention (pinned, matches
   Ouro's `F^(t)`, arXiv 2510.25741): **depth = passes = recursions + 1**; `depth=1` is the
   non-looped base case. Ouro's own data (§2 Decisions) shows diminishing returns exactly in this
   range — the big win is depth 1→2, real-but-shrinking gains to ~4, and their trained ceiling was
   `t=4` (our depth≈5) — going further hurts once you exceed what training covered. We target a
   *lower* mean (2, not 2.5) than the previous draft specifically because most tokens are
   easy/medium and the curve says the bulk of the value sits early — not a compute-saving guess,
   directly evidenced.
2. **Width (token-choice adaptive-k MoE), evaluated per recursion step.** At each pass, a router
   picks **k** for the token (min 1 routed + 1 always-on shared = 2 total; max 8 routed) and then the
   top-k experts. Budget-steered to **avg ≈ 4 routed + 1 shared**. k can vary across a token's own
   recursion steps.
3. **Pathway (per-pass expert identity).** The expert router is re-evaluated from the token's updated
   recurrent hidden state at every pass, so the selected expert coalition can change even when `k`
   stays constant. Shared physical weights therefore do not imply repeated identical computation:
   successive passes can specialize into stages such as parse → manipulate → verify → revise.

Per-token compute ≈ Σ_steps k(step). Experts specialize **by reasoning stage** (parse → manipulate →
verify), because routing is re-decided each pass — a qualitative win, not just efficiency.

### Naming boundary for the paper (locked)

**MoRE-Core names the original architecture:** a weight-shared recurrent stack with an evolving
hidden state, per-pass continuation (depth), variable expert count (width), and per-pass expert
re-routing (pathway), together with the routing/budget objectives required to train those decisions.
The three axes are the headline abstraction, not the entire implementation. Do not redefine MoRE as
every component in Metis-1.6; that would blur the novel claim and incorrectly absorb independently
sourced techniques.

**MoRE-RM is the paper's named native extension:** MoRE-Core plus the dual-use, expert-aware
recurrent depth memory and the per-pass continuation controller that consumes it. This belongs in
the MoRE paper because it is designed around MoRE's evolving state and route history, but it is not
required for the clean three-axis definition or the selected axis-isolating ablations.

**Metis-1.6 is the complete MoRE-based model family:** MoRE-RM + the hybrid Mamba-2/attention backbone +
recursion-aware mHC + concatenated N-gram conditional memory. mHC and N-gram memory are
complementary integrations rather than MoRE inventions and must be cited and reported as such. In
the paper, report `Full MoRE-Core`, `MoRE-RM`, and `Full Metis-1.6` separately.

### Fused recursion-aware mHC residual path (locked)

The single residual stream is replaced by **four persistent manifold-constrained Hyper-Connection
(mHC) streams**. Every Mamba-2 mixer, attention mixer, and shared/routed-expert sublayer reads a
learned mixture of the streams, executes the expensive sublayer once, and writes its output back
across all four streams. The residual mixing matrix is projected to the doubly stochastic manifold
(Sinkhorn-normalized), preserving bounded signal propagation across physical layers and recursion.

The mHC controller is **recursion-aware**: shared controller projections receive a learned pass
embedding and small pass-specific biases/gates, allowing pass 1 through pass 5 to use different
residual topologies without duplicating the Mamba, attention, or expert weights. This is a required
**fused path**, with fused read/mix/write kernels, selective recomputation, and packed-active-token
support; an unfused four-stream implementation is not an acceptable production endpoint.

### Dual-use expert-aware recurrent depth memory (locked)

Each active token writes typed memory entries at the two attention anchors and the end of every
completed pass. An entry contains a learned state representation plus its pass/anchor identity,
weighted expert-coalition embedding, routed `k`, expert-router entropy/confidence, and continuation
confidence. At later passes, every current mHC stream attends over only that token's valid entries.

The retrieved memory has **two simultaneous consumers**:

1. **Representation path:** gated fusion back into the current mHC streams before the attention/MoE
   path, so computation can recover, compare, verify, or revise earlier reasoning states.
2. **Routing path:** the continuation, adaptive-`k`, and expert-identity heads jointly consume the
   current state, retrieved memory, learned state-difference features, and route history. No fixed
   cosine threshold or hand-written halt rule decides depth.

Memory is bounded by `R_max=5`, stored and addressed in the packed token layout, and masked per
token. The production implementation keeps all valid typed anchors rather than heuristically
discarding intermediate states. Gates initialize near zero, but remain fully trainable end to end.

### Concatenated N-gram conditional memory (locked)

This is **not the tokenizer itself**. The 65,536-token BPE remains the tokenizer; a deterministic
canonicalization/compression map over its token IDs supplies suffix **2-gram and 3-gram** keys to a
separate learned conditional-memory table. Multiple independent prime-sized hash tables reduce
collisions. Their retrieved vectors are **concatenated, then projected once**, preserving N-gram
order/hash-head identity instead of destroying it through averaging.

Target **0.60B additional stored parameters** for Praxis while retaining all **128 routed experts +
1 shared expert**. This raises total stored parameters from ~2.94B to **~3.54B** while
before small controllers, while adding only the retrieved rows, projection, gates, and memory
traffic to active computation. Use
separate sparse-embedding optimizer settings (Adam-style states, higher embedding LR, no weight
decay) rather than applying the dense-matrix AdaMuon policy blindly.

The parameter count is dominated by table rows, not projection matrices. For illustration, two
N-gram orders × eight hash heads × approximately 0.586M slots/head × 64 values/slot is about 0.60B
parameters; exact prime slot counts are frozen in the executable manifest. Each token retrieves only
16 rows (1,024 values total in this illustration), concatenates them, and applies a small projection.
At BF16 the table weights occupy about 1.2GB, but FP32 master weights plus two FP32 Adam moments bring
the training-state footprint to roughly 8.4GB before sparse-gradient metadata and buffers.

The 0.60B target is a conditional-memory allocation, not a claim that lookup parameters equal neural
compute parameters. It is justified by the Engram sparsity-allocation result and by Praxis already
having substantial routed computation; monitor slot occupancy, collision rate, per-row update count,
gate utilization, and loss contribution. Before the 1T launch, run one short architecture-selection
canary comparing the locked hashed/concatenated design against a parameter-efficient Tensorized
Engram implementation. This is a production design check, not a required MoRE paper ablation.

Retrieve each token's N-gram rows once and cache them across recursion passes. Inject them through
branch-specific, pass-aware gates at two points: after approximately physical block 2 and after the
first attention anchor. The same static vector is never added unconditionally on every pass; the
current recurrent state decides whether and how strongly to use it.

Together the architecture has four complementary capacity primitives: static local memory
(N-grams), persistent multi-stream workspace (mHC), iterative reasoning memory (depth attention),
and conditional computation (MoRE depth × width × pathway).

### Expected quality effects and falsification criteria

These are architectural hypotheses, not promised benchmark deltas:

- **mHC:** preserve several simultaneous candidate features instead of repeatedly compressing all
  computation into one residual stream; improve gradient/signal stability through up to five uses
  of shared weights; permit pass-specific processing topology. Expected effect: broader reasoning
  representations, fewer destructive overwrites, and more reliable benefit from later passes.
- **Recurrent depth memory:** give later passes direct access to earlier intermediate results and
  route context instead of requiring the recurrent hidden state to carry everything losslessly.
  Expected effect: better verification/revision, multi-step consistency, and recovery of useful
  partial work rather than repeated reconstruction.
- **Per-pass continuation:** allocate another pass using evidence produced by the current pass,
  rather than predicting the whole depth budget before reasoning begins. Expected effect: less
  under-thinking on unexpectedly hard tokens and less redundant looping after convergence.
- **Expert-aware routing memory:** distinguish states produced by different expert coalitions and
  use prior route success/confidence in the next decision. Expected effect: stronger stage-wise
  specialization and more deliberate recruitment of complementary or verification experts.
- **N-gram memory:** retrieve recurring entities, phrases, code fragments, and local syntactic
  patterns directly, freeing early Mamba/attention/expert capacity for contextual reasoning.
  Expected effect: lower perplexity, better local/closed-book knowledge and code exactness, and
  more effective depth for reasoning and long-context attention.

The combined target is not merely additive: static lookup handles memorized local structure, mHC
keeps multiple live features, recurrent memory preserves work across passes, and MoRE chooses the
next computation. The plan must falsify the thesis if later passes stop improving loss/accuracy,
mHC streams collapse to identical representations, depth memory bypasses later experts, N-gram
hash collisions dominate rare patterns, or continuation confidence is not calibrated. Required
telemetry includes per-pass quality gain, stream diversity/CKA, memory-attention entropy and anchor
mass, gradients through later experts, expert-route transition matrices, halt calibration, N-gram
hit/collision frequency, and matched-compute quality against the pre-addition MoRE core.

### Paper ablation contract

Use the small, interpretable model list below. A complete 2³ statistical factorial is not required;
the selected variants isolate the claims directly without training every binary interaction.

| Model | Adaptive depth | Variable expert count | Routed expert identity | What it tests |
|---|---:|---:|---:|---|
| Dense | No | No | No | Total-parameter-matched dense reference |
| Vanilla MoE | No | No | Yes | Sparse routing without recurrence |
| Fixed LoopMoE | No | No | Yes | Recurrence with fixed depth and fixed `k` |
| MoR + dense FFN | Yes | No | No | Adaptive depth without sparse experts |
| MoR + fixed-top-`k` MoE | Yes | No | Yes | Adaptive depth plus pathway, fixed `k=4` |
| Fixed-depth variable-`k` MoE | No | Yes | Yes | Adaptive width without adaptive depth |
| **Full MoRE-Core** | **Yes** | **Yes** | **Yes** | All three axes together |
| **MoRE-RM** | **Yes** | **Yes** | **Yes** | Full MoRE-Core plus expert-aware recurrent memory |

Fixed-depth models use the measured mean depth of Full MoRE; fixed-`k` models use `k=4`. The dense
model matches the proxy's total stored core parameters as requested, so it is deliberately much more
expensive per token. Report quality versus tokens and executed FLOPs rather than pretending it is a
compute-matched control. mHC, N-gram combination/placement, and LoopSplit are **not required paper
ablations**. They remain parts of full Metis or possible follow-up studies; Praxis and Logos can be
reported as full-architecture scale demonstrations without decomposing every integration.

**Recommended proxy manifest:** 10 physical layers (8 Mamba + 2 attention), `d_model=1792`, latent
width `896`, expert intermediate `448`, 96 routed + 1 shared expert, dynamic `k=1–8` with mean near
4, and the same `R_max=5`. This is approximately **1.5B core stored / 0.29B core active per pass**
before the lightweight controllers. It is roughly half-Praxis class without naively halving every width, which would shrink matrix
capacity much faster than twofold.

Use **10B tokens as the primary proxy-paper budget**. Ten billion tokens is enough to test
optimization, routing, and relative quality at this A0.29B active scale, but it is not enough to
claim final scaling-law superiority; the full Praxis/Logos runs supply scaling evidence. Continue
Dense, Full MoRE-Core, or MoRE-RM beyond 10B only if their curves have not separated or reviewers
would otherwise be left with an ambiguous headline comparison. The proxy uses
the same sampled corpus mixture and curriculum logic as the 1T release, not literally the first
contiguous 10B tokens.

### Executable paper experiment ledger

All architecture comparisons use **base pretrained checkpoints**. RL and full post-training are not
required per ablation: they would multiply cost and introduce policy/reward confounders. This is
**8 unique models and 8 required 10B training runs**. If time permits, repeat only Dense, Full
MoRE-Core, and MoRE-RM with a second paired seed, producing **11 total runs** while remaining eight
unique architectures. A seed is just a repeat with a different initialization and shuffle; it is
insurance against one lucky headline result, not a different model design. No third seeds are
required by this plan. Reserve the complete SFT/RL pipeline for Praxis/Logos.

### Recursion-budget scaling test

`R_max=5` means five passes: one initial pass plus up to four recursions. Evaluate the same trained
checkpoint under natural continuation caps 1–5 and forced uniform depths 1–5, reporting quality
against executed FLOPs. Train with stochastic maximum-depth budgets or an explicit budget embedding
so caps 1–5 are supported operating points rather than out-of-distribution overrides; retain the
learned continuation loss inside each sampled cap. For the proxy, approximately A0.29B/pass × 5 = **1.45B executed active-
parameter equivalents** at mean `k`; depth 5 with `k=8` approaches **1.70B**. That brackets the
roughly 1.5B fully active stored-parameter-matched dense control. It does not create new parameters,
and weight reuse can have diminishing returns, so the paper claim is conditional: if depth-5 MoRE
matches or beats dense at equal executed FLOPs while adaptive mean depth remains near 2, MoRE has
learned to concentrate dense-class compute selectively. Extend to `R_max=8` only after depth-5
stability, non-saturated quality gains, healthy gradients, and continued expert use are demonstrated.

### Worked example (`2x + 3 = 11, so x = 4`)
Matches the spec: `2x`→depth4, k=[2,2,1,1] (6 calls); `+`→depth2 (2); `3`→depth3 (4); `so`→depth1
(1); `4`→depth4, k=[3,2,1,1] (7). Cheap tokens cost ~1 expert call; hard tokens cost ~7.

## 1.5 Product thesis — a tool-using reasoner, not an encyclopedia

We can't out-knowledge the frontier (they brute-force trillions of tokens for the long tail), so we
don't try. Metis-1.6 is built to **reason, then research**: minimal parametric facts, maximal
reasoning + **faithful retrieval-grounding** + **abstention**. It distrusts its own memory — for
anything factual it calls **web search / RAG**, reasons over the results, and grounds its answer in
sources. This directly kills 1.5's #1 failure (confident fabrication), plays to MoRE's reasoning
strength. The 1T schedule is intentionally knowledge-dense rather than a raw crawl mirror.

- **Not "zero knowledge"** — enough to *formulate queries, comprehend results, and know when to
  search*. Drop only long-tail-fact *memorization*; keep reasoning + reading-comprehension + grounding.
- **Trained skills:** when/how to call tools, query formulation, reasoning over retrieved context,
  **grounding faithfulness**, and **abstention** ("the sources don't cover this") instead of guessing.
- **Eval shifts:** measure *tool-augmented* QA + grounding faithfulness, not closed-book recall.
  Closed-book MMLU staying ~random is *expected and fine* under this thesis.
- **Existence proof:** two related Nanbeige papers, both verified directly (full-text fetch + grep,
  not summary): **Nanbeige4-3B Technical Report** (arXiv 2512.06266, the base recipe — cold-start
  SFT, DPD, multi-domain RLVR, pairwise-RM-last; source of most of §5's pipeline) and
  **Nanbeige4.1-3B** (arXiv 2602.13367, built on top of 4-3B-Base — extended-context SFT + a
  lightweight agentic-RL stage for tool-use/search, reaching reliable multi-step agentic search).
  Both are dense 3B models, not MoE — worth remembering when borrowing their RL algorithm choices
  (see §5's GSPO note).
- **External validation (Ouro, arXiv 2510.25741):** their own ablation found recursion **"does not
  increase knowledge capacity nor improve capacity scaling"** — looped and non-looped models hold
  the same ~2 bits/parameter of memorized facts. The gain from recursion shows up specifically in
  **"knowledge manipulation"** (their term — reasoning over/applying what's already stored), not in
  storing more of it. This is independent confirmation that the MoRE depth axis is pointed at
  exactly the right target for this thesis: we were never trying to buy knowledge with recursion,
  and the one paper that measured this says that's the correct read of what recursion actually does.

## 2. Specs (Praxis target and family scaling)

| Field | 1.5 | **Metis-1.6 Praxis** |
|---|---|---|
| Architecture | single-latent MoE, dense | **MoRE** (per-pass continuation × adaptive-k × expert re-routing) + recursion-aware mHC + recurrent depth memory + N-gram conditional memory |
| Backbone | transformer (attention) | **hybrid Mamba-2 + attention** (O(n) long-context for RAG) |
| Vocab | 32,768 | **65,536** (new BPE; canonicalized ID sidecar for N-gram hashing) |
| Pretrain context | 1024 | **4096** (packed stream; add EOS separators + doc masking / SSD state reset) |
| Final context | 1024 | **131k** (NoPE attention; single-jump post-training extension, not staged — §5) |
| `d_model` | 1536 | **2048** |
| Layers (physical) | 19 | **12 = 10 Mamba-2 + 2 attention** @ indices ~4, ~8 (× recursion → larger *effective* depth) |
| Latent dim | 512 | **1024** |
| Experts | 32 | **128 routed + 1 shared** (fine-grained, **G=16** — confirmed by scaling laws) |
| Expert intermediate (`d_expert`) | 1024 | **512** |
| top_k | 4 (fixed) | **dynamic 1–8 routed + 1 shared** (avg ≈ 4+1) |
| Residual topology | standard single stream | **4-stream recursion-aware mHC**, fused around every mixer and expert sublayer |
| Recurrent memory | none | **dual-use expert-aware depth memory** at both attention anchors + pass end |
| Conditional memory | none | **0.60B concatenated 2/3-gram parameters**, two gated injection points |
| Recursion depth | (cut) | **1–5 passes, per-pass continue/halt, monotonic packed active set**, target mean ≈2 |
| PT tokens | 50B | **1T, locked** (700B foundation + 250B capability build + 50B premium cooldown) |
| Params | 0.9B total / 340M active | **~3.54B stored before small controllers / ~0.464B core active per pass** (avg k), plus lightweight mHC/depth-memory/N-gram projections and retrieved rows |

**Param estimate (active, non-embedding, per pass):** mixers **~316M** — 10 Mamba-2 × ~29.5M
(**expand 2.0**, restored to the paper-default: d_inner 4096, 64 heads, ngroups 8, d_state 128:
in_proj+out_proj ≈ 29.5M/block) + 2 attention × 10.5M (QKVO, 32Q/8KV×64) — + latent proj **50.3M**
(12 × 2 × 2048×1024) + router **3.1M** (12 × 2048×128) + active experts **94.4M** at avg k
(5 × 12 × 1.573M; SwiGLU 1024↔512) = **~0.464B active** at avg k (4 routed + 1 shared);
**~0.407B at min k** (1 routed + 1 shared) — **~0.539B at max k** (8 routed + 1 shared).
**Core total:** experts 128 × 12 × 1.573M ≈ 2.42B + shared 0.019B + mixers 0.316B + latent/router
0.053B + tied embedding 0.134B (65,536 × 2048) ≈ **~2.94B**. Add **0.60B N-gram conditional
memory** for **~3.54B before** small mHC/depth-memory/controller parameters; the final manifest must
report the exact controller/projection delta rather than rounding it away.
*Provenance (this spec went through several revisions in one sitting — recorded so the numbers
don't look inconsistent across the doc):* 18 layers/expand-1.5/0.61B active → cut to 14
layers/expand-1.0/0.396B active (to hit a ~0.4B active target without touching the just-validated
G=16 MoE) → **settled at 12 layers/expand-2.0/0.464B active** once research showed the
Mamba-2 expand ratio has never been ablated at LLM scale in either direction, and the one real
ablation that exists anywhere (an unrelated small-scale OCR task) found expand mattering
substantially in the "more helps" direction — cutting the paper-default expand=2 down to 1.0 to
save compute was judged too risky relative to just trading physical layers for it instead.
**Total params were also revisited**: bumping expert *count* (holding d_expert=512 fixed, so G=16
is untouched) is a free way to buy back total params with ~zero active-compute cost, but a
2-per-cent-of-benchmark-accuracy gain (per the closest real reference, OLMoE's 32→64-expert
ablation, itself a bigger jump than what was on the table here) wasn't judged worth the systems
complexity — **kept at 128 experts**. Storage flexibility is now spent instead on the complementary
0.60B N-gram memory, bringing the architecture to ~3.54B stored before small controllers while preserving the
physical Mamba-2/attention layers and restored expand=2.0.

### Metis-1.6 family contract

`Praxis` and `Logos` are **size classes of the same Metis-1.6 generation**, not different
architectures. They use the same 65,536-token tokenizer, verified 1T-exposure pretraining release,
phase boundaries and data order, MoRE/mHC/depth-memory/N-gram mechanisms, loss definitions,
curriculum, context extension, and post-training sequence. Width/depth/expert counts, batch and
parallelism schedules, optimizer hyperparameters, and N-gram table capacity scale by class.

| Field | **Praxis** | **Logos target** |
|---|---:|---:|
| Stored parameters | ~3.54B before small controllers | **12.0B** (table slots tuned after exact controller count) |
| Core active/pass @ avg `k` | ~0.464B | **~1.183B** before control/memory projections; ~1.2B with controllers |
| `d_model` | 2,048 | **2,560** |
| Physical layers | 12 (10 Mamba + 2 attention) | **20 (17 Mamba + 3 attention)** |
| Attention indices (0-based) | ~4, ~8 | **5, 10, 15** |
| Attention geometry | 32Q / 8KV × 64 | **40Q / 10KV × 64** |
| Mamba inner geometry | 4,096; 64 heads; 8 groups | **5,120; 80 heads; 10 groups** |
| Latent width | 1,024 | **1,024** |
| Routed experts | 128 | **192** |
| Shared experts | 1 | **1** |
| Expert intermediate | 512 | **768** |
| Dynamic routed `k` | 1–8, mean ~4 | **1–8, mean ~4** |
| N-gram memory | **0.60B** | **~1.60–1.70B initial range**, tuned to the exact 12B manifest |
| Recursion | 1–5, mean ~2 | **1–5, mean ~2** |

**Why Logos uses `d_model=2560` but keeps latent 1,024:** shared width and MoE bottleneck width solve
different problems. Wider shared Mamba/attention/mHC state gives every token a richer recurrent
workspace, while the latent only needs to preserve the subspace required by the expert bank. NVIDIA
Nemotron 3 Super provides a strong precedent: `d_model=4096`, latent `1024`, expert intermediate
`2688`. Logos therefore keeps Praxis's 1,024 latent, raises the expert intermediate from 512 to
**768**, and keeps 192 routed experts. The 768 dimension is a multiple of 128 for the target FP8
GEMMs and shifts capacity from the shared-to-expert bottleneck into the nonlinear expert transform.
Buying only more routed identities would add stored specialization without raising active capacity
at fixed mean `k≈4`, reduce tokens per expert GEMM, and break the clean `EP=192 × replica 2`
mapping if the count does not divide the 384-APU allocation.

Adding physical layers is less attractive than the 768 expert intermediate for the locked target.
One additional Logos layer carries roughly **0.45B stored routed-expert parameters** and about
**59M active parameters per pass** once its mixer, latent projections, router, and five average
expert calls are included. It also extends the serial path on every recursion, so one layer executes
approximately twice per average token and five times for a maximum-depth token. Raising the expert
intermediate instead increases nonlinear routed capacity throughout the existing 20-stage stack,
keeps the clean topology, and leaves controller headroom near A1.2B. Benchmark a 21-layer/640-expert-
intermediate sizing pilot before the final manifest if desired, but do not silently change the paper
proxy or production class on that result alone.

**Logos sizing derivation (planning estimate):** each routed expert in a layer is
`3 × 1024 × 768 = 2.3593M` parameters. Across 20 layers and 192 routed experts this is **9.060B**;
the shared expert contributes **0.047B**. Scaling the locked Praxis mixer geometry gives
approximately **0.833B mixers**, plus **0.105B latent projections**, **0.010B expert routers**, and
**0.168B tied embeddings**, for a **~10.222B core stored** estimate. At mean routed `k=4`, the
active experts contribute **0.236B/pass**, producing **~1.183B core active/pass before the mHC,
memory, continuation, and fusion controllers**. Those controllers should bring the real active
manifest close to the A1.2B target. The remaining **~1.778B** to the 12B stored target is their
envelope plus the N-gram tables; tune deterministic table slots only after exact controller counts
are instantiated. `Logos-A1.2B` remains a rounded active-class label, and the manifest must publish
both core-active and all-control-active counts rather than hiding the delta.

**Decisions (locked):**
- **Total stored params ~3.54B before small controllers**, retaining 128 routed experts; **~0.464B core active per
  *pass*** (avg k) plus mHC/depth-memory/N-gram control work. The previous **6.5 GF/token** estimate
  is the base-core lower bound, not the final fused-architecture measurement. §6 carries it only as
  a reference until the fused single-GPU probe supplies real tokens/sec and MFU.
- **Recursion-aware 4-stream mHC, dual-use expert-aware recurrent depth memory, per-pass
  continuation routing, and 0.60B concatenated 2/3-gram memory are locked architecture**, not
  optional post-hoc experiments. Their internal sizes and kernel layouts may be tuned without
  removing the capabilities.
- **Hybrid ratio: 10 Mamba-2 + 2 full attention (16.7% of mixers), attention at block indices ~4 and
  ~8; block 0 and block 11 are Mamba-2.** Research consensus (Waleffe et al. 2406.07887: loss
  minimized at ~8% attention, evenly dispersed, Mamba-first enables NoPE; Jamba ablation: 1:3 vs 1:7
  "virtually no difference"; Nemotron-H/Nano ~1:6 of mixers; Granite 4.0 9:1; Falcon-H1: extra
  attention *hurts*) puts the sweet spot around 8–15%; 16.7% is a touch richer than the tightest
  end but still well inside the validated band, and gives a real margin of safety given the smaller
  physical layer count. Recursion favors even a lean ratio: at the target mean depth (2), 2 attn ×
  2 passes = **4 effective attention applications on average — matching Jamba's proven 4-layer,
  zero-recursion baseline that got clean 256K retrieval, exactly**; hard tokens (up to depth 5) get
  up to 2×5=10 effective applications, well above it. Attention KV cost scales ×passes at
  inference while Mamba state doesn't, which is the other reason to keep this ratio lean rather
  than richer. **Zamba2 precedent** for weight-shared looped attention — adopt their fix: a cheap
  **per-pass LoRA or gate on the shared attention** so each recursion pass can specialize; optional
  ablation: KV-share across passes (see the corrected Ouro finding below on why this is worth
  testing, not assuming).
- **Expert granularity G=16 confirmed (128 × d_expert 512, avg k 4+1).** Krajewski et al. (2402.07871):
  compute-optimal G=8 at 100M active, G=16 at 1B — we sit between; Ling scaling study (2507.17702,
  300+ models): optimal band includes this config, and imbalanced routing shifts the optimum coarser
  (→ invest in the aux-loss-free bias balancing from 1.5, monitor expert-utilization entropy);
  frontier convergence d_expert/d_model ≈ 0.29–0.38 (DeepSeek-V3/K2/Qwen3/GLM-4.5 = G 11–14);
  vs the 1024 latent the experts actually see, d_expert/d_latent = 0.5 = OLMoE-exact. **Do not go
  finer; 64×1024 would cost quality.** Systems guardrail: keep **≥9–16k tokens per grouped-GEMM
  call** so 512-wide experts stay compute-bound (SonicMoE-style fused/persistent grouped GEMM);
  expect ~10–15% M-tile fragmentation vs a 64-expert design — bounded, acceptable.
- **Optimizer: AdaMuon** (carried from 1.5 — Newton–Schulz-orthogonalized momentum + Adam-style second
  moment, fp32 states, fp32 masters). Previously only implied by the §6 memory budget; now explicit.
- **Tokenizer: train a new 65,536-vocab BPE** (1.5's 32k penalized exact-match tasks like LAMBADA; the
  data scale supports a bigger vocab). Decide/train early — it gates the whole data pipeline.
- **Pretrain context 4096** (was 1024 in 1.5). Attention-quadratic cost of 1024→4096 is ~2% of
  training FLOPs (2 attn layers of 12; SSM+MoE are flat per token) — no meaningful token dock. Win:
  the bulk of long docs stop being fragmented (a 7k-token doc = 2 windows, not 7) → long-range
  coherence the CPT stage can build on. Keep continuous packing but add **EOS separators +
  document-aware masking + Mamba-2 state reset at doc boundaries** (1.5's packer glued unrelated docs).
- **Recursion granularity: full-stack loop (Ouro-validated).** Loop all 12 layers (weight-shared); only
  embedding + lm_head sit outside the loop — `F^(t) = lmhead ∘ [Hᴸ]^t ∘ emb`, exactly Ouro's design
  (Ouro-1.4B: 24 unique layers at `t=4`; Ouro-2.6B: 48 unique layers at `t=4` — note "24L×R4" isn't
  the paper's own notation, that's shorthand introduced during our own research; the underlying
  numbers are right, the string isn't a direct quote). 12 layers × up to 5 passes = up to 60
  effective layers at the max, 24 at the target mean (depth 2). (Revised off the earlier "middle
  block" lean — Ouro shows full-stack sharing works.)
- **Mamba-2 expand ratio: 2.0, the paper default — NOT the 1.0–1.5 this draft used mid-session.**
  No paper in the Mamba/Mamba-2/Nemotron-H/Falcon-H1/Jamba lineage has ever ablated this
  hyperparameter at LLM pretraining scale — every one just inherits `expand=2` from the original
  Mamba paper, which itself picked it only to param-match a Transformer's combined MHA+MLP block,
  not from a sweep. The one real ablation found anywhere (a small-scale OCR adapter paper) showed
  expand mattering substantially, in the "more helps" direction (5× error reduction going 2→6).
  Given that signal argues against complacency, not for it, we chose to preserve the validated
  default and pay for it with physical layer count (18→14→12 over this design pass) rather than
  gamble an unquantified quality hit on the SSM's core working dimension.
- **Attention-converges-after-recursion claim, corrected.** An earlier draft of this plan cited
  "Ouro's attention patterns converge after the first iteration" — that's wrong. The paper's actual
  finding (§7.2, an answer-agreement matrix over 1,000 QQP pairs, not raw attention maps) is that
  consecutive-step agreement stays low through their trained depth (e.g. only ~55% agreement
  between recursion steps 2→3) and only converges to near-100% **once you go *past* their trained
  ceiling (t≥4)** — because the model never learned to keep changing its answer beyond the depth it
  was trained at, not because it settles quickly. Relevant to us because it's a caution, not a
  green light: whatever max depth we deploy (5) needs to be the depth we actually *train* to, not a
  ceiling we discover post-hoc doesn't generalize.

**Still open:**
- ~~Hybrid ratio / G=16~~ — **resolved above** (research pass 2026-07-05). Attention config locked:
  `n_heads=32`, `head_dim=64`, GQA `n_kv_heads=8` (Granite-exact at d_model 2048), **NoPE** (Mamba-2
  block 0 supplies position — the Waleffe finding that makes NoPE safe). Mamba-2 config: **expand
  2.0** (restored to default), 64 heads × 64, `ngroups=8`, d_state 128 (Nemotron/Falcon-H1 defaults).
- ~~Depth/k off-by-one convention~~ — **resolved.** `depth (passes) = recursions + 1`, matching
  Ouro's `F^(1) ≡ F` (non-looped = 1 pass, 0 recursions) exactly. Pinned throughout the doc.
- Training-time depth strategy: **MoR-style packed** (GPU-native — §4, §6); revisit only if the
  packed path fails parity.

## 3. The central systems risk: packed dynamic compute across expert-parallel ROCm ranks

This remains the make-or-break issue. The original failure mode was XLA/TPU static shapes; Portage
removes that specific constraint but adds a harder distributed one: per-token continuation,
variable `k`, expert re-routing, mHC streams, and typed memory create ragged traffic across up to
192 EP ranks. The implementation must make dynamic compute cheaper in measured wall-clock while
providing useful Slingshot evidence, not merely move the saved FLOPs into dispatch overhead.

The historical two execution forms remain useful references:

- **(a) Envelope + mask** — run max depth and max k for everyone, mask the inactive. Trivial to
  compile and train, **but saves no compute** (you pay the maximum). Useful only for *adaptive
  quality*, not efficiency. Fine for **training** (you compute the envelope for gradients anyway).
- **(b) Recursion-wise token packing** — at each recursion step, **gather only the still-active
  tokens** into a capacity-bounded dense buffer, process, scatter back; same for expert dispatch
  (capacity-factor buffers). This is where the **real inference savings** live.

We already have the machinery from the 1.5 codebase: `mor_compute_mode="static_packed_hard"`,
`_decoder_layer_packed_queries`, `_pack_assignments_cumsum`, and `mor_pack_active/valid/overflow`
metrics. **1.6's core engineering job is productionizing the packed path in fused HIP/ROCm kernels**,
integrating it with expert all-to-all, and proving both numerical parity and wall-clock benefit from
1 APU through EP=192. The TPU machinery remains a correctness reference; the RTX PRO 6000 remains
a CUDA canary/fallback, while Portage §6B is the launch target.

## 4. Training the discrete decisions (continuation + k + pathway)

**Depth — per-pass continuation is the production strategy:**

- **Ouro-style diagnostic baseline (best for TPU): fixed-depth train + early-exit inference.**
  Train every token at the *full* max depth — static shapes, **no packing needed during training** —
  and learn **early-exit gates** that make depth adaptive *at inference only*. Ouro's proven two-stage
  recipe: Stage I entropy-regularized depth exploration (uniform-prior KL); Stage II gate trained on
  observed per-step loss improvement; at inference exit when CDF(t) > threshold q. Lowest engineering
  risk, fully static on TPU. It remains an ablation and fallback, not the target architecture. Cost:
  ~R× training compute per token (prohibitive at 1T — see §6).
- **MoRE production path (GPU): packed per-pass continuation during training and inference.** After
  each pass, the continuation head reads the current mHC streams, recurrent depth memory, and route
  history. Hard continue decisions produce a monotonic active set for the next packed pass; training
  uses a differentiable continuation estimator plus hard-path straight-through/parity checks. The
  model never commits to a complete depth bucket from its initial state.

**k (adaptive width)** is capacity-based MoE regardless (solved in 1.5): predict per-token k, compute
top-`k_max`, mask down to k.

Common to both:

- **Soft survival objective + hard packed execution.** Let each pass predict a continuation hazard;
  cumulative survival probabilities define the differentiable expected loss across exits. Execute
  the hard monotonic packed path with a straight-through estimator during the routed phase, and
  maintain a soft-envelope reference for numerical/gradient parity. Experts remain weighted by
  router probabilities in the soft reference and top-`k` in the hard path.
- **Compute-budget control.** Aux loss targeting an *average* depth and *average* k (**mean depth
  ≈ 2**, mean routed-k ≈ 4) so the model doesn't collapse to all-min (free, dumb) or all-max
  (expensive). This is the steering wheel for the efficiency/quality tradeoff. Mean depth 2 (not
  the earlier 2.5) is directly evidenced by Ouro's own per-step accuracy curve (§2 Decisions) —
  the biggest gain is 1→2, real-but-smaller gains to ~4, marginal beyond that.
- **Separate control regularization.** Use aux-loss-free bias balancing plus a small telemetry loss
  for **expert load** at every pass; use softmax logits + z-loss for expert identity. The continuation
  head instead receives survival-profile, compute-budget, entropy-floor, and calibration losses—do
  not force a Bernoulli halt decision into the expert-router loss. Monitor halt rate and correctness
  conditioned on pass, route-history entropy, and expert utilization after memory reads.
- **Curriculum / ramp.** Warm-start **dense** (depth 1, fixed k) for the first ~5–10% of tokens so
  the backbone learns basic LM, then ramp in adaptive depth/k. Always-init both routers (even in the
  dense phase) so checkpoints stay stage-flip compatible — a 1.5 lesson.
- **k-predictor design.** Token-choice adaptive-k: predict a per-token k (small head → integer bucket
  1–8, or a learned probability threshold over expert scores so k emerges naturally). For static
  compile: always compute top-`k_max`, mask down to the chosen k.

## 5. Data & post-training (fold in every 1.5 lesson)

The 1.5 eval (`METIS_1.6_NOTES.md`) said the bottleneck was tokens + post-training choices, not
architecture. So:

- **1T PT exposures, locked** — measured only with the final Metis tokenizer after filtering,
  cross-source deduplication, and benchmark decontamination. The source of truth is
  `manifests/metis-1.6.yaml`: 525B web, 160B code, 85B math, 125B science/technical,
  70B synthetic pedagogical/factual, 25B books/reference/legal, and 10B translated/native
  multilingual. The 90B freshness layer (35B web + 35B software + 10B science + 10B official
  documentation) is inside the trillion. Target 875B unique tokens plus 125B controlled replay;
  no generated data is eligible for Phase C. New 65,536-vocab byte-level BPE; packed at 4096 with
  EOS separators, document boundaries, and Mamba state reset at document boundaries.
- **Keep `<think>` chain-of-thought traces** in SFT (`keep_think=True` on OpenThoughts/OpenR1/
  Bespoke-Stratos/s1K). 1.5-think emitted *zero* CoT because prep stripped them — the single biggest
  post-training miss. MoRE's depth axis is the natural substrate for reasoning; pair it with visible CoT.
- **Scrub distilled-assistant artifacts at the source**: identity ("OpenAI"), "as an AI language
  model I cannot," refusal boilerplate. Ship a **native identity/persona set** from the start (no
  post-hoc patch like 1.5 needed).
- **Strong, calibrated safety data** — 1.5 *under*-refused harm and *over*-refused benign requests.
  Train clean refusals + benign compliance.
- **Abstention / "I don't know" data** to curb the confident hallucination 1.5 showed (fabricated
  bios, "gold named after the Greek god").
- **Add DPO** (cut from 1.5) for helpfulness + refusal calibration + less fabrication.
- **Dedup + length/EOS diversity** to kill the degeneration/looping 1.5 showed ("2,4,6,8 → …46").

**Post-training pipeline (final, research-verified 2026-07-06 — synthesizes both Nanbeige papers'
recipes, not a single paper's literal sequence; see the per-step notes on which parts come from
where and which parts are our own addition):**

1. **SFT — hybrid mix, not pure-CoT or pure-chat.** Every SFT set must contain *both* legitimate
   short direct-answer exemplars *and* long-thinking exemplars, tagged so the model represents both
   modes explicitly. This is the detail that makes step 4's dynamic-length RL possible: if SFT never
   shows a legitimate short answer, RL can't later teach brevity — it has nowhere to collapse to
   except degenerate-but-wrong short answers. (Grounded in the "Reasoning Models Can Be Effective
   Without Thinking" finding, arXiv 2504.09858 — skipping CoT is competitive specifically in
   low-token-budget settings, which tells you the short-answer capability already sits latent in a
   properly-SFT'd model; RL's job is allocation between modes that already exist, not inventing a
   new one.) Structurally follows Nanbeige4-3B's two-stage split: **cold-start SFT** (~30M curated
   reasoning samples, 32K ctx, math/science/code — builds the CoT foundation) → **overall SFT**
   (64K ctx, broadens to general/agent/tool-use/code, adds function-calling support; their
   "Solution Refinement" — iterative teacher-critique-revision against a dynamic per-instruction
   checklist — plus "CoT Reconstruction" to re-attach a clean reasoning trace after refinement
   disrupts it, are worth replicating). Keep every 1.5 lesson from below (identity scrub, safety,
   abstention, dedup).
2. **DPD (Dual-Level Preference Distillation)** — verified directly from Nanbeige4-3B (arXiv
   2512.06266, §3.3), not inference: token-level distillation from a strong teacher's probability
   distribution on **both** positive samples (best teacher rollouts) *and* negative samples (worse
   rollouts sampled from the student itself, filtered to be clearly inferior), combined with a
   sequence-level DPO margin loss. Reported gains from DPD alone: ~8% AIME24/25, ~10% GPQA, ~30%
   BFCL-V4, ~8% Arena-Hard V2. Critically, their paper states directly (verbatim): **"incorporating
   an RL phase on top of this distillation framework yields substantially larger gains compared to
   initiating RL directly from the SFT baseline"** — this is why DPD sits *before* RL, not a
   stylistic choice.
   - **Purpose, narrowed (our own framing, not Nanbeige's):** three candidate objectives exist inside
     DPD — (A) reasoning-quality uplift on RLVR's own domains, raising the baseline pass-rate so more
     prompts land in step 3's productive 10-90% filtering band instead of the "everything wrong, zero
     signal" zone; (B) self-correction, via the negative-sample loss training the model to recognize
     *its own* characteristic errors (this specifically needs negatives generated fresh from the
     post-Overall-SFT checkpoint, not a pre-baked dataset — a real sequencing dependency, DPD can't be
     data-collected before Overall SFT finishes); (C) style/preference polish. We prioritize A+B —
     C is already owned by step 5's dedicated pairwise RM, so DPD's dataset should be domain-weighted
     toward STEM/code/agentic (matching step 3), not a broad general-chat mix.
   - **Teacher: DeepSeek-V4-Flash** (284B total/13B active MoE, MIT-licensed — verified directly, no
     distillation restriction), via **DeepInfra** ($0.09/1M input, $0.018/1M cached input, $0.18/1M
     output — confirmed exact). For STEM/code specifically, positives are selected by the *same free
     verifiers* step 3's RLVR will use (Python-interpreter equivalence for math, sandboxed execution
     for code) — no LLM-judge scoring cost needed for those two domains. Target scale: roughly
     30-60k instructions per domain (STEM/code/agentic), 4-6 samples each — a few hundred dollars at
     real DeepInfra pricing with per-instruction input caching, not thousands.
   - **Open risk, deliberately accepted, not resolved:** DeepSeek-V4-Flash (13B active) is ~28× our
     student's active params (0.464B). Dense-model distillation literature (Qwen2.5-family study,
     arXiv 2502.12143; Apple's Distillation Scaling Laws, arXiv 2502.08606; Zhang et al.'s "optimal
     teacher ≈2.5× student" law, arXiv 2311.07052) flags a real "capacity gap" failure mode at large
     teacher:student ratios — specifically for the *token-level* distillation half, not the DPO-margin
     half (every mechanism described is about matching the teacher's full probability distribution,
     which a DPO-style preference margin doesn't require). None of that literature examined a
     **recursive (MoRE) or MoE** student, so it's not known whether Metis-1.6's effective capacity
     (up to ~2.3B core capacity exposure via max-depth recursion; ~3.54B stored before controllers including
     static N-gram memory) mitigates this the way raw
     dense active-params would predict — this is genuinely untested territory, not just for us but
     for anyone. **Decision: proceed anyway** — betting on MoRE's capacity being real is the point of
     testing a novel architecture, not something provable in advance. **Cheap mitigations kept in
     reserve, not pre-committed:** DeepSeek-V4-Flash's reasoning effort is configurable — dial it down
     for the token-level distillation component specifically if a small pilot shows degradation
     (mirrors the literature's validated "Mix Distillation" fix — blending long/short or
     large/small-teacher traces recovered +7-8 points in the closest analog study). Run a small pilot
     (a modest slice, checkpoint-and-eval) before committing the full DPD budget, rather than assuming
     either outcome.
3. **Multi-domain RLVR — GSPO (not GRPO), with DAPO-derived stabilizers and on-policy filtering.**
   - **GSPO over GRPO, and this is architecture-driven, not a style preference** — verified directly
     from the GSPO paper (arXiv 2507.18071, §5.3), tested on Qwen3-30B-A3B (a real MoE model): GRPO's
     token-level importance ratio breaks under MoE because after each gradient update, ~10% of the
     experts activated for the *same* rollout change between old and new policy, making the
     per-token ratio "fluctuate drastically" and destabilizing training. GRPO's fix (Routing
     Replay — forcing the same experts to fire for ratio computation) works but adds real
     memory/comms overhead and artificially caps the model's actual capacity. GSPO computes the
     ratio on **sequence-level** likelihood instead, which doesn't depend on which expert fired per
     token, eliminating the instability at the root. **This is why Nanbeige's plain GRPO doesn't
     transfer to us** — both Nanbeige models are dense 3B, so they never hit this failure mode; we
     are MoE (128 experts), so we would. One honest caveat: GSPO's demonstration is at Qwen3-30B-A3B
     scale (much bigger active params than our 0.464B) — the mechanism is architectural so it should
     transfer, but hasn't been validated at our tiny active-param scale specifically.
   - **DAPO-derived stabilizers** (verified from Nanbeige4-3B §3.4.2, which layers these onto GRPO —
     port them onto GSPO instead): remove the KL penalty term, mask the loss for truncated/overlong
     sequences. (DAPO's own full recipe, arXiv 2503.14476, also has Clip-Higher and Dynamic Sampling
     — Dynamic Sampling filters only exact-0/exact-1 accuracy, a different, stricter criterion than
     Nanbeige's own on-policy filtering below; worth an A/B, not assumed.)
   - **On-policy pass-rate filtering, verified exact number: strictly 10–90%.** Using the *preceding*
     stage's checkpoint, compute avg@16 accuracy per question, keep only questions with pass rate
     strictly between 10% and 90% (Nanbeige4-3B §3.4.1, verbatim). This band is Nanbeige's own
     method, not literally DAPO's (DAPO filters only the two extremes, 0% and 100%) — a real,
     confirmed difference between the two; we're using Nanbeige's softer band.
   - **Domain order** (Nanbeige4-3B's STEM→coding, extended with 4.1-3B's agentic addendum — our own
     synthesis, not one paper's literal sequence): **STEM** (math + science, with a tool-augmented
     agentic verifier calling a Python interpreter for exact symbolic/numeric equivalence checking —
     avoids false negatives from differently-formatted correct answers) → **coding** (synthetic
     problems paired with sandboxed executable test functions, binary pass/fail reward; reverse-
     generation — synthesize solution+tests first, then the natural-language problem — to guarantee
     correctness) → **agentic/tool-use** (search → read → reason → answer; synth multi-hop QA via
     knowledge-graph random walks, a real search env with search API + page extractor + sandbox,
     turn-level rewards for tool-call accuracy/info-gain + full-trajectory credit; kept "lightweight"
     per 4.1-3B's own framing — the compute-hungry stage, prioritize if budget tightens).
4. **Dynamic-thinking-length RL — layered onto step 3's reward function, not a separate stage.**
   Field consensus (AdaptThink arXiv 2505.13417, DAST arXiv 2503.04472, SelfBudgeter arXiv
   2505.11274, AnytimeReasoner arXiv 2505.13438 — all independently verified): a naive flat length
   penalty **collapses accuracy** — the model discovers shorter is rewarded and starts truncating on
   hard problems where it needed the tokens (DAST and SelfBudgeter both name specific baselines,
   "L1"/"E1," that collapse this way on AIME2025). Every serious method uses a **dual reward**:
   correctness + a difficulty-conditioned length term, and the length term only has purchase once
   correctness reward already exists — which is why this rides on top of step 3's RLVR, not before
   it. **The design win specific to this pipeline:** the difficulty signal needed for length-shaping
   is the *same* avg@16 pass-rate number already being computed for step 3's on-policy filtering.
   DAST's own mechanism (verified, exact formula) is built exactly this way — Token Length Budget
   `L_budget = p·L_mean + (1−p)·L_max`, where `p` is the sampling accuracy on that question. Reward
   shorter completions on questions the running model already solves reliably (high pass rate);
   don't penalize length (or mildly reward it) on questions it doesn't (low pass rate). One piece of
   infrastructure, two jobs. Concretely: `reward = correctness_reward + λ · length_shaping(pass_rate)`,
   with λ small enough a wrong-but-short answer never beats a correct-but-long one — correctness
   stays the dominant term, always. (AnytimeReasoner's dense multi-truncation-budget reward is the
   more sophisticated version of this — better credit assignment, meaningfully more implementation
   work — noted as a stretch goal, not the first-run default.)
5. **Pairwise reward model / human-preference alignment — last.** Matches Nanbeige4-3B's own
   ordering (not 4.1-3B's, which sequences its point-wise/pair-wise stage second — the two papers
   genuinely differ here; we're following the base paper's order since it's the one with DPD as an
   antecedent). Rationale: this is the one non-verifiable stage, so it runs after every verifiable
   stage (STEM, coding, agentic) is locked in, so the reward model's softer signal can't disturb
   hard-won verifiable gains. Nanbeige's own reasoning for training a *dedicated* pairwise RM rather
   than using a general LM-as-judge: a general judge needs a lengthy CoT before verdicting (slow) and
   is prone to reward hacking; a small dedicated pairwise model expresses preference in a few tokens
   and resists hacking better.
6. **Eval** — tool-augmented QA + grounding faithfulness (not closed-book).

> RL is the budget swing factor — agentic rollouts (search-in-the-loop, many samples/prompt) are
> compute-hungry. Keep the agentic stage "lightweight" (4.1-3B's word); prioritize if budget tightens.

**Context extension → 131k — single direct jump, not a staged ladder (corrected 2026-07-06).**
Verified directly across five NVIDIA papers (Nemotron-H arXiv 2504.03624, Nemotron Nano 2 arXiv
2508.14444, Nemotron 3 whitepaper arXiv 2512.20856, Nemotron 3 Nano arXiv 2512.20848, Nemotron 3
Super arXiv 2604.12374, Nemotron 3 Ultra arXiv 2606.15007): **every one of them jumps directly from
an 8192-token base context straight to the long target — none stage through intermediate lengths
like 8K→32K→128K.** Quoted directly from the Nemotron 3 whitepaper: *"In CPT, we did not observe the
need to follow a staged increase of training sequence length from 8k to 512k."* The explicit enabler
is **NoPE**, which we already have locked (§2): *"Since Mamba layers provide implicit positional
information, Nemotron 3 models do not use RoPE in attention layers and therefore do not suffer from
out-of-distribution RoPE issues during context extension."* This directly overturns the earlier draft
of this plan, which assumed a staged curriculum (4096→32k→131k) — drop the staging, do a **single
jump from 4096 straight to 131k** over the ~10–15B token budget (closest real precedent by budget
size: Nemotron Nano 2's dedicated extension phase, 18.9B tokens for their 12B model — our budget
sits just below that, consistent with our much smaller active-param count).

Two more things every one of these papers do that we should copy:
- **Mix in a small fraction of base-length (4096) sequences alongside the long ones, not 100% long.**
  Nemotron Nano 2's first attempt (pure long-context batches) regressed short-context/math
  benchmarks; every subsequent model mixes in short sequences (Nemotron 3 Ultra: 92% long / 8%
  short, run concurrently) specifically to prevent that regression. Exact ratio is tuned per model,
  not universal — treat 90/10 to 80/20 (long/short) as the starting range to sweep.
- **Data mix ~80% downscaled pretrain-style data / ~20% new synthetic long-context data** (long-
  document QA, retrieval-style tasks) is the NVIDIA default across Nano 2, Nemotron 3 Nano, and
  Super; Ultra used a richer 46% new-data mix when specifically pushing for stronger long-context
  ability — our smaller budget argues for the ~80/20 default, not Ultra's richer mix.
- **Overshoot is a real, evidenced choice, but sized differently for us.** Nemotron-H/Nano 2 train
  at 512K to deploy at 128K (4× overshoot) specifically because "longer training sequences lower the
  chance of long coherent documents being cut" by their data-loading chunking — a data-mechanics
  reason, not just a robustness margin. Given our 131k target is already much smaller than
  Nemotron's 512K–1M training lengths, a modest overshoot (e.g. train to ~164k–200k, deploy at
  131k) is worth doing for the same data-chunking reason, but doesn't need Nemotron's 4× ratio at
  our scale — size this against our actual long-document length distribution once the data
  pipeline (§7) is built, not a borrowed ratio.

## 6. Historical 300B Azure sizing — superseded for the 1T Portage run

> **Superseded 2026-07-21.** The section below is retained as architecture/precision research
> history only. It is not the current Metis-1.6 launch plan, does not size the 1T schedule, and its
> Azure cost table must not be used for Portage reservations. The current operational contract is
> §6A and `docs/metis16_pretraining_data_plan.md`.

**Hardware:** **RTX PRO 6000 Blackwell Workstation Edition** (GB202, **sm_120** — a distinct target
from datacenter sm_100; expect software-lag potholes) — 96 GB GDDR7 @ ~1.79 TB/s, PCIe 5.0 x16
(~63 GB/s per direction, "128 GB/s" bidirectional; **no NVLink**), 600 W.
**Corrected peak specs (2026-07-05):** **~1000 TFLOPS dense FP8, ~2000 TFLOPS dense FP4**
(NVIDIA's own datasheet; "4,000 AI TOPS" is the *sparse* FP4 marketing headline, not the dense
number). An earlier draft of this plan cited ~1.9/3.8 PF here — that was a mix-up: those numbers
are from an academic microbenchmark of **B200** (the datacenter GB100 die), not this card's GB202
die. **This card's FP4 dense peak (~2000 TF) is almost exactly H100 SXM's FP8 dense peak
(~1979 TF)** — the whole economic case for training in NVFP4 here, if the sm_120 kernel-maturity
risk (below) resolves.

**Committed to Azure only** (compute credits live there) — this rules out the cheaper third-party
spot pricing found during scouting (Verda ~$0.66/GPU-hr, Nebius ~$0.95, kept below only as
context) and rules out a genuine single-node 4-GPU rig: **Azure's RTX PRO 6000 series (NC v6,
`Standard_NC{n}ds_xl_RTXPRO6000BSE_v6`) caps at 2 GPUs per VM** (SR-IOV vGPU, not passthrough).
Given §6's budget table below shows 300B tokens fits comfortably on **1–2 GPUs** within the
required RL reserve, this cap isn't actually a constraint here — **the 2-GPU VM
(`NC288ds_xl_RTXPRO6000BSE_v6`) is the practical target for the full run**, giving real local
(single-node) interconnect rather than a cross-VM bridge.

**Precision plan (research pass 2026-07-05 — MXFP8 is the MAIN lane; NVFP4 is the experiment):**
Nemotron 3 Super/Ultra prove NVFP4 pretraining at 25T/20T tokens, but two findings flip the default:
(1) on **sm_120** (GB202 — this card, vs sm_100 datacenter) NVFP4 **grouped GEMM** — the exact kernel
the MoE lives on — is the least-mature path in the stack (CUTLASS #3096 / flashinfer #2577-class
silent-garbage bugs; vLLM SM120 MoE kernels missing); (2) the MXFP8 recipe (arXiv 2506.08027: E4M3
for weights+acts+grads, ceil-rounded UE8M0 scales) needs **zero layer exemptions** and reliably gives
~20–30% over BF16, matching BF16 loss within 0.5% at 15T tokens. At our small GEMM sizes NVFP4's
quantization overhead (RHT, 2D scaling) can even make it *slower* than FP8 (nanochat data point).
- **MXFP8 (E4M3)** — all linear GEMMs by default: routed+shared experts (grouped), latent down/up,
  attention QKVO, Mamba in_proj and **out_proj** (Nemotron 3 finding: out_proj *underflows in NVFP4*
  — MXFP8 explicitly, never FP4).
- **BF16** — token/N-gram embeddings and lm_head, norms, mHC streams, recurrent-depth-memory values,
  continuation gates, and residual state.
- **FP32** — **router logits + softmax/top-k** (DeepSeek/Megatron practice; it's 2048×128 — free),
  mHC Sinkhorn projection and depth-attention logits/softmax,
  Mamba-2 SSD scan/conv/dt internals (fused-kernel default — don't touch), master weights, AdaMuon
  states, CE-loss accumulation (**chunk the 65,536-wide logits** — full bf16 logits at 4096 ctx ≈
  0.5 GB/seq).
- **NVFP4 experiment lane (upside, not baseline):** routed-expert GEMMs only, with Nemotron's exact
  kit — 2D 16×16 weight scaling (transpose-consistent), RHT on **wgrad inputs only**, stochastic
  rounding on **gradient quantization only** (RTN for weights/acts), final ~15% of blocks BF16.
  Gate on: pin TE ≥ 2.14 + CUDA 13.x, smoke-test every grouped-GEMM shape vs BF16 reference for
  exact-zero/garbage outputs on sm_120, then A/B loss curves. Adopt only if it's both correct AND
  faster; Nemotron 3 Super saw no gain switching NVFP4→MXFP8 at 19T tokens, so staying MXFP8 is
  not a quality sacrifice.

**Base-core throughput reference (@ target mean depth 2, corrected 0.464B active):**
- Before mHC/depth-memory/N-gram overhead, FLOPs/token ≈ 6 × (0.464B × 2 + 0.134B lm_head) + ~2% attention-quadratic buffer at 4096 ctx
  (2 attn layers × ~2 average effective applications) ≈ **6.5 GF/token** → 300B ≈ **1.95 × 10²¹
  FLOPs total**. The new architecture adds little dense parameter compute but meaningful residual
  bandwidth, depth-memory attention, projections, and sparse lookup traffic. Until the fused probe,
  use **+10–30% wall-clock** as a planning range—not as a measured result.
- **Base-core precision paths, using the corrected peaks (1000 TF FP8 dense / 2000 TF FP4 dense):**

  | Path | Peak dense | MFU | Effective TF | tok/s (1 GPU) |
  |---|---|---|---|---|
  | MXFP8 conservative | 1000 TF | 20% | 200 | 30.8k |
  | MXFP8 mid | 1000 TF | 30% | 300 | 46.2k |
  | NVFP4 conservative | 2000 TF | 20% | 400 | 61.5k |
  | NVFP4 mid | 2000 TF | 25% | 500 | 76.9k |
  | NVFP4 optimistic | 2000 TF | 30% | 600 | 92.3k |

- **Why MoR-style adaptive depth still matters:** Ouro-style fixed-depth training (everyone pays the
  max) costs meaningfully more per token than adaptive packing — it's what keeps 300B tokens inside
  a sane wall-clock at this budget.

**Pricing & budget (checked 2026-07-05; $5,000 total run budget; committed to Azure — §6 top):**
- **Azure NC v6**: **$5.50/GPU-hr PAYG, $1.10/GPU-hr spot** (flat 80% off), max 2 GPUs/VM
  (`NC288ds_xl_RTXPRO6000BSE_v6`), SR-IOV vGPU. **Spot is required** — PAYG is 5× the cost.
- *(Context only, not usable — GPU credits are Azure-locked):* Verda ~$0.66 spot, Nebius ~$0.95
  preemptible, Vast.ai from ~$1.03 — all cheaper, but off the table given the credit constraint.
- **Base-core budget table, 300B tokens, at $1.10/GPU-hr spot** (GPU-hrs = 1.95×10²¹ / (eff TF × 3.6×10¹⁵)):

  | Path | Effective TF | GPU-hrs (300B) | Cost @ $1.10 | % of $5,000 |
  |---|---|---|---|---|
  | MXFP8 conservative | 200 | 2,709 | **$2,980** | 59.6% |
  | MXFP8 mid | 300 | 1,806 | **$1,986** | 39.7% |
  | NVFP4 conservative | 400 | 1,354 | **$1,490** | 29.8% |
  | NVFP4 mid | 500 | 1,083 | **$1,192** | 23.8% |
  | NVFP4 optimistic | 600 | 903 | **$993** | 19.9% |

- **Architecture-adjusted planning range:** applying +10/+20/+30% to the conservative MXFP8 core
  case yields approximately **$3,278 / $3,576 / $3,874**, leaving **$1,722 / $1,424 / $1,126** of
  the $5,000 budget before post-training. Therefore the historical 300B proposal fit that old
  budget, but it is no longer the active Metis-1.6 token schedule; the old claim of a
  guaranteed 40% RL reserve is retired until the fused path is measured. Better MFU or NVFP4 grows
  the reserve; an unfused implementation that falls outside this range blocks the full run.
- Calendar time at a representative middle scenario (300 TF, MXFP8-mid): **~75 days on 1 GPU,
  ~38 days on the 2-GPU VM before the +10–30% architecture range**, or roughly 83–98 / 42–49 days.
  Same total dollar cost either way—2 GPUs primarily reduce calendar time. **Preemption-safe
  checkpointing is mandatory on spot** (frequent async saves + auto-resume).

**Multi-GPU over PCIe (2× on Azure) — data-parallel core + replicated sparse memory:**
- The dense/MoE core still communicates gradients for **~2.94B parameters**, not only the active
  0.464B, because a large accumulated batch reaches every expert: **~5.9 GB bf16** per step. The
  0.60B N-gram table is replicated but uses custom sparse touched-row synchronization; do not
  materialize/all-reduce another dense 0.9–1.2 GB gradient buffer. If sparse synchronization is not
  correct and faster in measurement, shard the table lookup rather than silently accepting dense
  traffic.
- **Amortize via grad accumulation:** at a large accumulated global batch, comms are a low
  single-digit-% overhead, overlappable with backward → effectively free. Comm per optimizer step
  is fixed; make the step big.
- **ZeRO-1** (shard masters + optimizer states): identical wire traffic (reduce-scatter + all-gather
  ≡ all-reduce), halves the masters+optimizer memory footprint across the 2 GPUs. Worth using even
  at just 2×.
- **Do NOT use over PCIe:** ZeRO-3/FSDP (re-gathers all weights every microbatch — a real tax),
  expert parallelism (all-to-all per MoE layer per pass per microbatch — dead without NVLink), or
  tensor parallelism (per-layer activation all-reduces). None are needed — the model fits on one card.
- **Platform:** Azure's `NC288ds_xl_RTXPRO6000BSE_v6` (2 GPUs, SR-IOV vGPU, real single-node/local
  interconnect — not a cross-VM bridge). Budget ~1.2 kW GPU power for the pair.

**Memory per GPU (96 GB):**
- The original ~2.94B core occupied **~45 GB static** in 1× pure DP. A BF16 0.60B N-gram
  table plus FP32 master/Adam states adds approximately **6.3–8.4 GB**, before sparse-gradient
  transients and small controller weights: plan **~52–55 GB static**. Four mHC streams and typed
  depth memory increase activation pressure substantially, so aggressive selective rematerialization
  and measured microbatch sizing are mandatory; “very comfortable” is no longer assumed.
- 2× with ZeRO-1 and sharded N-gram optimizer states plans roughly **~31–34 GB static per GPU** plus
  sparse-gradient transients, leaving the preferred activation headroom for the fused architecture.
- Full training checkpoint planning range becomes approximately **42–55 GB**—write async;
  spot preemption makes checkpoint cadence a correctness requirement, not a nicety.
- **Scale context (historical):** 300B was 6× 1.5's 50B. The current 1T Portage schedule is 20×
  Metis-1.5's 50B and is governed by the measured-token release contract in §6A.

**De-risk order:** prove the complete fused architecture—packed continuation, recursion-aware mHC,
typed recurrent memory, and cached N-gram lookup—on one card before extrapolating the base-core
table. Then confirm sparse table synchronization and 2-GPU scaling; NVFP4 remains upside, not a
gate. This probe still decides the training topology and honest calendar estimate, but not the
now-locked 1T data schedule.

## 6A. Login2/Rhea data and Portage training launch plan — current

Metis-1.6 now uses **exactly 1,000,000,000,000 final-tokenizer-measured training exposures**. The
release contract is declarative in `manifests/metis-1.6.yaml`; detailed source rows are under
`manifests/sources/`. The three phases are fixed at **700B / 250B / 50B**. The first 875B exposures
are unique after global exact and near-duplicate removal. Phase B contains 75B controlled replay,
and Phase C is a 50B premium cooldown made entirely from high-priority, non-generated records.

The embedded freshness layer is **90B**, not an addition to the trillion: 35B fresh general web,
35B fresh software, 10B recent open science, and 10B current official documentation. Every target
is enforced after filtering, global deduplication, benchmark decontamination, and counting with the
accepted 65,536-token Metis tokenizer. Published source-token estimates never satisfy a target.

The production science/reference recipe uses only pinned, auditable reservoirs. A third-party S2ORC
snapshot was removed because a database-level ODC-By label did not resolve the licenses of its
underlying papers; its 15B allocation remains in the same phases across FinePDFs (7B), peS2o (4B),
and licensed Common Pile PubMed/PMC (4B). The pinned historical CC-BY OpenStax snapshot is capped at
39M tokens, with its displaced allocation moved to FinePDFs in the same phases. The bounded pinned
formal repositories are capped at 150M formal-math and 180M systems-code exposures. A pinned-commit
materialization canary measured about 186.5M and 231.4M accepted byte-estimated tokens respectively,
leaving deterministic filtering headroom; the displaced allocations move to Proof-Pile-2 math and
NVIDIA repository code without changing category or phase totals.
Wholesale Pile of Law was
replaced by pinned Common Pile Caselaw Access Project and USGPO primary-law snapshots. These are
source substitutions only: science remains 90B / 30B / 5B and reference remains 25B / 0 / 0.

The operator interface is split by environment:

1. after the Lustre administrator confirms sufficient capacity, on `login2`, run
   `./ops/start-acquisition.sh --lustre-root /lus/lustre1/vollmerc/metis-1.6 --quota-acknowledgement administrator-confirmed`;
2. the launcher bootstraps and runs restart-safe acquisition inside GNU Screen, then emits the
immutable `ACQUISITION_READY.json` handoff; the launcher obtains both gated Hugging Face access and
read-only GitHub repository metadata credentials without placing tokens in argv or logs;
3. after `build_ready: true`, enter Rhea, run `./metisctl doctor --profile rhea --role compute` and
   `./metisctl verify-handoff --profile rhea`, then `./metisctl submit build --profile rhea`;
4. Portage consumes the final verified release for model training and does not run CPU data prep;
5. monitor acquisition with
   `METIS_LUSTRE_ROOT=/lus/lustre1/vollmerc/metis-1.6 ./metisctl status --profile login2` and resume
   by rerunning the same Screen launcher command.

Packaged Hugging Face, Common Crawl WARC-range, pinned GitHub/repository, and canonical-source
acquisition paths are implemented and must produce hashed local payloads; a URL, repository index,
or generation recipe is never counted as downloaded training text. The login2 launch still requires
confirmed usable Lustre capacity and a clean cloneable commit. Rhea's later build remains sealed
until its mount path and scheduler facts are known, and the license-review attestation remains a
separate fail-closed release gate. Keep credentials outside the repository.

The Lustre-server acquisition supervisor resolves and hydrates pinned sources plus evaluation
holdouts. The later Slurm graph normalizes canonical records, applies fail-closed quality and license
gates, runs full-SHA-256 global exact deduplication, two-pass externally sorted repeated-span
deduplication, disk-backed partitioned MinHash winner resolution, 128-bit code-structural
deduplication, and a final SHA-256 audit, then decontaminates
against 63 benchmark registry entries across 36 family labels/203 pinned jobs with exact, long,
short, code-token, and identifier/literal-normalized code-skeleton matching bound to individual
benchmark rows,
trains and validates the tokenizer, performs exact source and phase selection, writes 1,000 one-
billion-token uint16 shards, rehashes them, and emits `RELEASE.json`. Semantic deduplication is
explicitly disabled.
Training must refuse to start without that verified release. The operational and data details live in
`docs/metis16_pretraining_data_plan.md`.

## 6B. Portage model-training topology — current

**Portage is the primary Metis-1.6 training platform; Azure RTX PRO 6000 is fallback/canary only.**
The operating allocation is 512 AMD Instinct MI300A APUs on HPE Cray EX255a with Slingshot-11.
The public system record reports 129,024 aggregate CPU+GPU cores, which is exactly 512 ×
(24 Zen 4 CPU cores + 228 CDNA 3 GPU compute units), and HPE identifies Portage as its real-world
HPC/AI benchmarking system. Each MI300A supplies 128 GB coherent HBM3 and ~5.3 TB/s local peak
bandwidth. Sources: `https://top500.org/site/51014/`, HPE's 2025 Portage announcement, and AMD's
MI300A data sheet. Exact reservation, Slingshot group layout, node/NIC mapping, software modules,
filesystem paths, and job limits remain live site facts and must be captured before launch.

### Simultaneous family allocation

| Run | APUs | Likely nodes at 4 APUs/node | Logical expert topology | Purpose |
|---|---:|---:|---|---|
| **Metis-1.6 Praxis** | **128** | **32** | **EP=128, expert replica count=1** | one routed expert per APU; maximum cross-node routing exposure |
| **Metis-1.6 Logos** | **384** | **96** | **EP=192, expert replica count=2** | two 192-APU expert replicas; one routed expert per APU per replica |

Do not inflate Logos to 384 experts merely to mirror the device count. The 192-expert, two-replica
layout preserves compute-efficient 768-wide experts, hits the 12B/A1.2B model target, uses all 384
APUs, and adds both expert all-to-all and cross-replica gradient synchronization—the communication
surfaces needed for the Slingshot study. No tensor or pipeline parallelism is planned initially;
the model states fit comfortably in 128 GB/APU. Shared experts and dense Mamba/attention/controller
weights are replicated within the logical data groups; routed experts are partitioned by EP rank.
N-gram tables start replicated per rank because deterministic rows and optimizer states fit; a
sharded-table lane is a measured systems ablation, not an unverified launch dependency.

Concretely, the 384 APU allocation contains **two independent 192-rank expert-parallel groups**.
Within each group, rank `e` owns routed expert `e`, so every one of the 192 experts has two physical
copies across the job. Each replica dispatches its own token shard with a 192-way all-to-all/all-to-
all-v; corresponding expert copies then synchronize gradients across the two replicas. Dense,
shared-expert, controller, and embedding gradients use their appropriate data-parallel groups. Thus
`EP=192, replica=2` uses all 384 APUs and creates both the expert-dispatch traffic and replica-sync
traffic relevant to the Slingshot study; it does not leave 192 APUs idle.

### ROCm precision and kernel contract

The CUDA/Transformer-Engine MXFP8/NVFP4 implementation assumptions in historical §6 do not transfer
unchanged to MI300A; the **math and architecture do**. Portage's production target is **FP8-first**:
use native CDNA 3 FP8 for every numerically validated, throughput-positive GEMM, with a BF16 reference
lane for parity and fallback. NVFP4 is not a Portage target.

- **FP8 by default where supported:** routed/shared expert GEMMs; latent down/up projections;
  attention Q/K/V/O projections; Mamba input/output projections; recurrent-memory Q/K/V/output;
  mHC controller projections; N-gram fusion projections; and the LM-head matrix multiplication if
  the loss path remains high precision. Every exact shape must be benchmarked because library
  alignment/layout restrictions and small grouped GEMMs can erase the nominal FP8 advantage.
- **BF16 where state or reductions are sensitive:** embeddings and retrieved table rows, residual
  and mHC streams, Mamba recurrent state/scan tensors, normalization and gates, attention state when
  the installed FP8 attention kernel lacks parity, activation outputs between GEMMs, and initially
  gradients/collectives. FP8 collectives are enabled only if Portage's installed RCCL stack supports
  them and end-to-end tests beat BF16 communication without quality loss.
- **FP32:** master weights and optimizer states; router/continuation logits; Sinkhorn normalization;
  depth-attention logits, softmax and reductions; loss/logit accumulation; and numerically sensitive
  Mamba parameters or reductions identified by parity tests.

Fused kernels must be HIP/ROCm-native (hipBLASLt/Composable Kernel or the validated site stack), and
the packed active-token path may not fall back silently to host or unfused Python loops. “BF16
reference lane” means the oracle used to validate FP8, not the intended precision of the full run.

### Dynamic execution and dropless routing contract

MI300A/HIP supports runtime control flow, dynamic kernel launches, compaction, gather/scatter, and
variable-count collectives; MoRE does **not** require token dropping. HIP execution graphs are an
optional launch-overhead optimization, not a requirement for dynamic execution. Capture stable
segments or a small set of bounded-shape pass buckets; execute changing dispatch/compaction regions
with streams or update/re-instantiate graph templates when topology changes.

Production routing is **dropless**. Use exact packed token counts with RCCL all-to-all-v where the
site stack performs well, or preallocated bounded buffers plus validity masks and a deterministic
overflow slow path. Active-token compaction is not token dropout: a halted token preserves its final
state and simply skips later recurrent passes. Track three distinct quantities and never conflate
them: training-data token dropout (none), MoE overflow drop rate (target exactly zero), and learned
continuation halting (intended MoRE behavior).

mHC, recurrent depth memory, and packed continuation therefore work on AMD hardware. The porting
constraint is that CUDA-specific Transformer Engine, CUTLASS, Triton-CUDA, and CUDA-graph kernels do
not run unchanged; equivalent HIP/ROCm kernels must be implemented and fused. The risk is lost
efficiency or missing optimized shapes, not architectural incompatibility.

Before model work, record `rocminfo`, ROCm/PyTorch/hipBLASLt/CK versions, Slurm topology, visible
APUs, NUMA/HBM placement, `fi_info`, RCCL/OFI/Cray-MPICH modules, Lustre mounts, and scheduler limits.
Run single-APU GEMM/attention/Mamba canaries, then intra-node collectives, then multi-node RCCL
all-reduce and all-to-all sweeps. Checkpoint/resume, deterministic token accounting, and optimizer
state restoration must pass before the two long jobs start.

### Slingshot research matrix

The interconnect study and model training share one instrumentation contract. Before the full runs,
sweep short fixed-token jobs over:

- Praxis EP sizes **4, 8, 16, 32, 64, 128**; Logos EP sizes **12, 24, 48, 96, 192**, with expert
  replica count adjusted to consume the selected allocation.
- Topology-aware contiguous placement versus randomized/cross-group expert placement.
- Expert dispatch with overlap on/off, dropless versus bounded-capacity packing, and controlled
  router-balance distributions.
- Fixed depths 1/2/5 and adaptive continuation; fixed `k` 1/4/8 and learned variable `k`.
- Uniform synthetic routes, learned routes, and intentionally skewed routes as labeled network
  stress tests—never mix intentionally skewed stress routing into the production checkpoint.

For every point capture tokens/s, achieved BF16/FP8 throughput, MFU, per-layer all-to-all bytes and
time, dispatch/combine latency p50/p95/p99, RCCL collective timelines, Slingshot link/NIC counters,
congestion and retransmission indicators, expert token-count CV/max-min, straggler gap, overlap
efficiency, packed-token padding/overflow/drop rate, HBM bandwidth, power, and loss parity. The
production 1T runs use the fastest numerically correct topology found; the deliberately inefficient
placements remain separate research evidence for the Slingshot paper.

### Fast ablation allocation (separate from the Slingshot study)

The approximately 1.5B-stored proxy fits on one 128GB MI300A with FP32 master weights and optimizer
state, so paper ablations do not need wide expert parallelism. Before the campaign, race three
fixed-token canaries at 4/16/32 APUs: fully replicated local experts + data parallelism, intra-node
`EP=4` + data parallelism, and the validated sharded-optimizer/HSDP path. Select solely by
end-to-end tokens/second at matched loss and zero token drops. Do not use deliberately congested
placements, wide EP, or routing skew in the architecture paper campaign.

Assuming four APUs per node is confirmed, train all eight required models in **one concurrent
wave**. Allocate 128 APUs to Dense, 64 each to Full MoRE-Core and MoRE-RM, and 48 each to the other
five models: `128 + 64 + 64 + 5×48 = 496`, leaving 16 APUs for evaluation/canaries. For optional
second seeds of Dense/MoRE-Core/MoRE-RM, run a short second wave and allocate the full machine by
measured throughput rather than preserving these initial ratios.

Compute-only planning uses approximately 3.5 GFLOP/token for the A0.29B, mean-depth-2 proxy, or
about **35 EFLOP per normal 10B run**; the total-parameter dense model is roughly **90 EFLOP**.
With the allocation above and an unproven end-to-end 3–15% of the official FP8 peak, budget roughly
**1–6 hours of compute and one practical day** for the required eight-model wave after the stack is
stable. The optional three-repeat wave should fit inside another day. BF16 fallback or immature
dynamic kernels can roughly double that; queue time and initial ROCm/kernel commissioning are
excluded. Replace every estimate with measured tokens/second from the canary before promising a
calendar date.

### Two-model comparability contract

Praxis and Logos consume the same immutable 1T release and 700B/250B/50B phase boundaries. Use the
same tokenizer, document boundaries, curriculum events, objective, and evaluation checkpoints.
Record tokens—not steps—as the comparison axis; global batch and learning-rate scale by active
class. Publish matched-token, matched-FLOP, and matched-wall-clock comparisons so the family scaling
result does not confuse larger active capacity with more data or compute. Simultaneous execution
must isolate each job's Slingshot counters and placement; if the jobs share fabric groups, record
cross-job interference as an explicit experimental condition.

## 7. Suggested build order
1. Implement + unit-test the **four-stream recursion-aware mHC wrapper** around Mamba, attention,
   and MoE, including Sinkhorn constraints, pass conditioning, recomputation, and fused HIP/ROCm
   kernels, while retaining the Azure/CUDA canary backend.
2. Implement the **typed expert-aware recurrent depth memory + per-pass continuation router** in the
   packed token layout; prove monotonic active sets, masking, route metadata, and representation/router
   dual use through pass 5.
3. Implement the **canonicalized concatenated 2/3-gram memory**, cached per sequence across passes,
   with branch/pass-aware gates, sparse optimizer states, and replicated/sharded Portage table modes.
4. Verify **parity and health**: soft survival/envelope vs hard packed execution; Sinkhorn marginal
   error; non-collapsed streams/experts/halts; target mean depth/k; gradients through later experts;
   memory gates and N-gram collisions.
5. Freeze executable **Praxis and Logos manifests** from §2, including exact controller counts,
   N-gram table slots, active-parameter audit, optimizer memory, and checkpoint schema.
6. Execute the §6B Portage bring-up and Slingshot scaling matrix; throughput-prove the **complete
   fused path** from 1 APU → 1 node → multi-node EP before reserving all 512 APUs. Reject any
   extrapolation based only on component microbenchmarks.
7. Tokenizer + **1T-token login2/Rhea data pipeline** (with the §5 fixes baked in — hybrid short/long SFT tagging
   scheme decided early since it constrains data collection, not just training recipe).
8. Launch simultaneous Praxis-128 and Logos-384 curriculum PT runs
   (depth-1/fixed-k/near-zero memory gates warm-start → staged joint ramp); all components exist from
   initialization so the optimizer/checkpoint contract never changes mid-run.
9. **Context extension**: single jump 4096→~164-200k (modest overshoot; deploy at 131k), 80/20
   old/new data mix, 90/10-ish long/short sequence mix — size exact ratios against real long-doc
   data once collected (§5).
10. Post-training (§5, final order): cold-start SFT → overall SFT (hybrid short/long mix) → DPD →
   multi-domain RLVR (GSPO + DAPO stabilizers + 10–90% on-policy filtering; STEM → coding →
   agentic, dynamic-thinking-length reward layered throughout) → pairwise RM/human-preference →
   eval (reuse `eval_metis.py`) → publish.

## 8. Open questions for you
- ✅ Params **~3.54B stored before small controllers / ~0.464B core active per pass plus fused control/memory work**;
  128 routed + 1 shared expert retained; **0.60B additional concatenated 2/3-gram memory** ·
  ✅ Tokenizer: new 65,536 · ✅ Recursion: full-stack loop (Ouro-style), **depth = passes =
  recursions+1**, per-pass continuation, target mean 2, max 5 · ✅ **four-stream recursion-aware
  fused mHC** · ✅ **dual-use expert-aware recurrent depth memory** · ✅ PT context 4096 ·
  ✅ Extension target 131k ·
  ✅ **Hybrid ratio: 10 Mamba-2 + 2 attention @ ~4/~8 (12 layers)** · ✅ **G=16 granularity
  (128×512) confirmed** · ✅ Core optimizer: AdaMuon; N-gram tables: separate sparse Adam-style
  policy — all locked.
- ✅ Data preparation platform: **login2 acquisition onto Lustre, then Rhea CPU Slurm**, using pinned
  manifests, restartable arrays, and an immutable verified release. Portage is trainer-only. Keep
  site-specific profiles private and credentials out of Git. Exact Rhea partition/account/QoS values
  remain site configuration and must be supplied by the HPE account owner.
- ✅ Primary training allocation: **Praxis on 128 MI300A APUs (EP=128, one expert replica)** and
  **Logos on 384 MI300A APUs (EP=192, two expert replicas)**, running simultaneously after the
  staged Portage/RCCL/Slingshot canaries. Azure RTX PRO 6000 is fallback/canary, not the launch plan.
- ✅ Logos planning class: **20 layers, d_model 2560, latent 1024, 192 routed + 1 shared expert,
  d_expert 768, ~10.222B core stored, ~1.183B core active/pass before controllers, and ~1.60–1.70B
  N-gram memory**;
  tune table slots against exact controller counts to land at 12.0B stored.
- ✅ **1T PT exposures, locked** — 700B broad foundation, 250B capability intensification, 50B
  premium cooldown; 875B unique plus 125B controlled replay; 90B embedded freshness; no generated
  data in Phase C. The historical Azure/300B cost table in §6 is not a launch estimate.
- Per-pass LoRA/gate on the shared looped attention (Zamba2 finding) — implement as config flag; A/B.
- Continuation estimator inside the locked per-pass halting architecture: cumulative hazard vs
  ACT/PonderNet-style relaxation — A/B for calibration, quality, and hard-packed gradient health.
- ✅ **Post-training pipeline locked** (research-verified 2026-07-06, §5): cold-start SFT (hybrid
  short/long, tagged) → overall SFT → **DPD** (Nanbeige4-3B, arXiv 2512.06266) → multi-domain RLVR
  on **GSPO** (not GRPO — MoE-specific stability argument, arXiv 2507.18071) + DAPO-derived
  no-KL/loss-masking + 10–90% on-policy filtering (STEM → coding → agentic) with
  **dynamic-thinking-length reward layered on top** (DAST-style, reusing the same pass-rate number)
  → pairwise RM last. Two real open items inside this: (a) GSPO's stability claim is demonstrated at
  Qwen3-30B-A3B scale, not either our Praxis-A0.46B or Logos-A1.2B scale—worth early class-specific
  sanity checks; (b) exact λ for the length-shaping term, and exact GSPO clip ranges, need tuning,
  not just adoption.
- ✅ **Context extension locked as single-jump, not staged** (research-verified 2026-07-06, §5):
  4096 → ~164-200k (modest overshoot) → deploy 131k, over ~10–15B tokens, mixing in a small
  fraction of base-length sequences and an ~80/20 old/new data split. Exact overshoot ratio and
  short-sequence fraction need sizing against real long-document data once collected, not just
  borrowed from Nemotron's ratios at a much larger target length.
- Exact Portage Slingshot group/NIC placement, measured end-to-end tokens/second, final BF16/FP8
  choice, and post-training allocation remain open operational inputs. They do not change the
  locked family architecture or 1T data release contract.
