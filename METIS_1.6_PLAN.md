# Metis-1.6 — MoRE Architecture & Training Plan

Status: design draft. Supersedes 1.5 (single-latent MoE, dense). Folds in the 1.5
post-mortem findings (`METIS_1.6_NOTES.md`).

## 0. North star
**MoRE = per-token, two-axis adaptive compute:** dynamic recursion **depth** (Mixture-of-Recursions)
× dynamic expert **width** (adaptive-k MoE). Hard tokens get more of both; easy tokens get less —
nothing over- or under-shot. Goal: 1.5's efficiency instincts, but with real reasoning depth, ~6× the
training data (**300B tokens, locked** — budget analysis in §6 shows this fits even in the
conservative-precision case), and a deliberate strategic bet — **a tool-using reasoner, not an
encyclopedia** (§1.5). Backbone shifts to **hybrid Mamba-2 + attention** for cheap O(n) long-context/RAG.

## 1. Architecture (what MoRE is)

Two routers decide a token's compute envelope:

1. **Depth (token-depth MoR).** A depth router maps each token to a recursion-depth bucket. The
   **recursive block** (weight-shared) is re-applied that many times. Bucketed and **capped**:
   1 pass (no recursion) … up to 5 passes (4 recursions), **target mean ≈2 passes**. *Capped +
   bucketed is essential* — it's what lets us compile it (see §4). Convention (pinned, matches
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

Per-token compute ≈ Σ_steps k(step). Experts specialize **by reasoning stage** (parse → manipulate →
verify), because routing is re-decided each pass — a qualitative win, not just efficiency.

### Worked example (`2x + 3 = 11, so x = 4`)
Matches the spec: `2x`→depth4, k=[2,2,1,1] (6 calls); `+`→depth2 (2); `3`→depth3 (4); `so`→depth1
(1); `4`→depth4, k=[3,2,1,1] (7). Cheap tokens cost ~1 expert call; hard tokens cost ~7.

## 1.5 Product thesis — a tool-using reasoner, not an encyclopedia

We can't out-knowledge the frontier (they brute-force trillions of tokens for the long tail), so we
don't try. Metis-1.6 is built to **reason, then research**: minimal parametric facts, maximal
reasoning + **faithful retrieval-grounding** + **abstention**. It distrusts its own memory — for
anything factual it calls **web search / RAG**, reasons over the results, and grounds its answer in
sources. This directly kills 1.5's #1 failure (confident fabrication), plays to MoRE's reasoning
strength, and is why 300B tokens (not trillions) suffices.

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

## 2. Specs (target)

| Field | 1.5 | **1.6 (target)** |
|---|---|---|
| Architecture | single-latent MoE, dense | **MoRE** (token-depth MoR × adaptive-k MoE) |
| Backbone | transformer (attention) | **hybrid Mamba-2 + attention** (O(n) long-context for RAG) |
| Vocab | 32,768 | **65,536** (new BPE) |
| Pretrain context | 1024 | **4096** (packed stream; add EOS separators + doc masking / SSD state reset) |
| Final context | 1024 | **131k** (NoPE attention; single-jump post-training extension, not staged — §5) |
| `d_model` | 1536 | **2048** |
| Layers (physical) | 19 | **12 = 10 Mamba-2 + 2 attention** @ indices ~4, ~8 (× recursion → larger *effective* depth) |
| Latent dim | 512 | **1024** |
| Experts | 32 | **128 routed + 1 shared** (fine-grained, **G=16** — confirmed by scaling laws) |
| Expert intermediate (`d_expert`) | 1024 | **512** |
| top_k | 4 (fixed) | **dynamic 1–8 routed + 1 shared** (avg ≈ 4+1) |
| Recursion depth | (cut) | **1–5 passes, bucketed/capped** (0–4 recursions), target mean ≈2 |
| PT tokens | 50B | **300B, locked** (spot-priced — §6; fits even the conservative-precision case) |
| Params | 0.9B total / 340M active | **~2.94B total / ~0.464B active per pass** (avg k); ~0.41B at min k, ~0.54B at max k |

**Param estimate (active, non-embedding, per pass):** mixers **~316M** — 10 Mamba-2 × ~29.5M
(**expand 2.0**, restored to the paper-default: d_inner 4096, 64 heads, ngroups 8, d_state 128:
in_proj+out_proj ≈ 29.5M/block) + 2 attention × 10.5M (QKVO, 32Q/8KV×64) — + latent proj **50.3M**
(12 × 2 × 2048×1024) + router **3.1M** (12 × 2048×128) + active experts **94.4M** at avg k
(5 × 12 × 1.573M; SwiGLU 1024↔512) = **~0.464B active** at avg k (4 routed + 1 shared);
**~0.407B at min k** (1 routed + 1 shared) — **~0.539B at max k** (8 routed + 1 shared).
**Total:** experts 128 × 12 × 1.573M ≈ 2.42B + shared 0.019B + mixers 0.316B + latent/router
0.053B + tied embedding 0.134B (65,536 × 2048) ≈ **~2.94B**.
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
complexity — **kept at 128 experts, ~2.94B total**, prioritizing physical Mamba-2/attention layers
and the restored expand=2.0 over a marginal total-param bump.

**Decisions (locked):**
- **Total params ~2.94B**, **~0.464B active per *pass*** (avg k) — FLOPs/token ≈
  6 × (0.464B × avg depth + 0.134B lm_head) + a small (~2%) attention-quadratic buffer at 4096 ctx
  ≈ **6.5 GF/token** @ target mean depth 2 (see §6 for the full budget this produces).
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

## 3. The central risk: dynamic compute on a static-graph TPU

This is the make-or-break, and the reason MoR was cut from 1.5. XLA/TPU compile **static shapes**;
per-token dynamic depth + k is data-dependent ragged control flow. Two ways to realize it:

- **(a) Envelope + mask** — run max depth and max k for everyone, mask the inactive. Trivial to
  compile and train, **but saves no compute** (you pay the maximum). Useful only for *adaptive
  quality*, not efficiency. Fine for **training** (you compute the envelope for gradients anyway).
- **(b) Recursion-wise token packing** — at each recursion step, **gather only the still-active
  tokens** into a capacity-bounded dense buffer, process, scatter back; same for expert dispatch
  (capacity-factor buffers). This is where the **real inference savings** live.

We already have the machinery from the 1.5 codebase: `mor_compute_mode="static_packed_hard"`,
`_decoder_layer_packed_queries`, `_pack_assignments_cumsum`, and `mor_pack_active/valid/overflow`
metrics. **1.6's core engineering job is productionizing the packed path** and proving it both trains
and yields wall-clock savings on the GPU. (Section kept for history: the packing machinery was
designed under TPU constraints; on the RTX PRO 6000 the dynamic-compute win is native — the packed
path ports, minus the XLA static-shape gymnastics.)

## 4. Training the discrete decisions (depth + k)

**Depth — two strategies, pick with the hardware (§6):**

- **Ouro-style (recommended baseline; best for TPU): fixed-depth train + early-exit inference.**
  Train every token at the *full* max depth — static shapes, **no packing needed during training** —
  and learn **early-exit gates** that make depth adaptive *at inference only*. Ouro's proven two-stage
  recipe: Stage I entropy-regularized depth exploration (uniform-prior KL); Stage II gate trained on
  observed per-step loss improvement; at inference exit when CDF(t) > threshold q. Lowest engineering
  risk, fully static on TPU. Cost: ~R× training compute per token (prohibitive at 300B — see §6).
- **MoR-style (for GPU / compute-constrained): packed adaptive-depth during training.** Per-token
  depth router + recursion-wise packing so training only pays each token's chosen depth — saves
  training compute, but needs the dynamic/packed path (hard on TPU, native on GPU).

**k (adaptive width)** is capacity-based MoE regardless (solved in 1.5): predict per-token k, compute
top-`k_max`, mask down to k.

Common to both:

- **Soft-expected → hard at inference.** Where a decision must stay differentiable, train with expected
  outputs (depth = Σ_d p_d · block_d; experts weighted by router probs) and switch to hard at inference —
  the `soft_fixed_depth` ↔ `static_packed_hard` split 1.5 prototyped.
- **Compute-budget control.** Aux loss targeting an *average* depth and *average* k (**mean depth
  ≈ 2**, mean routed-k ≈ 4) so the model doesn't collapse to all-min (free, dumb) or all-max
  (expensive). This is the steering wheel for the efficiency/quality tradeoff. Mean depth 2 (not
  the earlier 2.5) is directly evidenced by Ouro's own per-step accuracy curve (§2 Decisions) —
  the biggest gain is 1→2, real-but-smaller gains to ~4, marginal beyond that.
- **Dual load balancing.** Separate aux losses for **expert load** (per recursion step) and **depth
  distribution**. Anchor router scores by construction — 1.5's sigmoid router with a gameable aux
  loss collapsed (drops hit 38%); **use softmax scores + z-loss** on both routers from day one.
- **Curriculum / ramp.** Warm-start **dense** (depth 1, fixed k) for the first ~5–10% of tokens so
  the backbone learns basic LM, then ramp in adaptive depth/k. Always-init both routers (even in the
  dense phase) so checkpoints stay stage-flip compatible — a 1.5 lesson.
- **k-predictor design.** Token-choice adaptive-k: predict a per-token k (small head → integer bucket
  1–5, or a learned probability threshold over expert scores so k emerges naturally). For static
  compile: always compute top-`k_max`, mask down to the chosen k.

## 5. Data & post-training (fold in every 1.5 lesson)

The 1.5 eval (`METIS_1.6_NOTES.md`) said the bottleneck was tokens + post-training choices, not
architecture. So:

- **300B PT tokens, locked** — the smaller active size (~0.464B) + cheap spot GPU-hours are being
  spent on *more tokens*, not saved; §6's budget table shows this fits even in the conservative
  precision/MFU case while preserving the required 30–40% post-training/RL reserve. Since facts
  come from retrieval (§1.5), **bias the mix toward reasoning / math / code / reading-comprehension
  / instruction** over encyclopedic crawl — enough world model to *research*, not to *memorize*.
  English-filtered, deduped, benchmark-decontaminated; new 65,536 vocab; packed at 4096 with EOS
  separators + doc masking.
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
     (up to ~2.3B via max-depth recursion; ~2.94B total stored params) mitigates this the way raw
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

## 6. Compute plan — RTX PRO 6000 Blackwell on Azure (1× probe → 2× for the full run)

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
- **BF16** — embeddings/lm_head, norms, depth gates, residual stream.
- **FP32** — **router logits + softmax/top-k** (DeepSeek/Megatron practice; it's 2048×128 — free),
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

**Throughput & wall-clock (@ target mean depth 2, corrected 0.464B active):**
- FLOPs/token ≈ 6 × (0.464B × 2 + 0.134B lm_head) + ~2% attention-quadratic buffer at 4096 ctx
  (2 attn layers × ~2 average effective applications) ≈ **6.5 GF/token** → 300B ≈ **1.95 × 10²¹
  FLOPs total**.
- **Precision paths, using the corrected peaks (1000 TF FP8 dense / 2000 TF FP4 dense):**

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
- **Budget table, 300B tokens, at $1.10/GPU-hr spot** (GPU-hrs = 1.95×10²¹ / (eff TF × 3.6×10¹⁵)):

  | Path | Effective TF | GPU-hrs (300B) | Cost @ $1.10 | % of $5,000 |
  |---|---|---|---|---|
  | MXFP8 conservative | 200 | 2,709 | **$2,980** | 59.6% |
  | MXFP8 mid | 300 | 1,806 | **$1,986** | 39.7% |
  | NVFP4 conservative | 400 | 1,354 | **$1,490** | 29.8% |
  | NVFP4 mid | 500 | 1,083 | **$1,192** | 23.8% |
  | NVFP4 optimistic | 600 | 903 | **$993** | 19.9% |

- **Verdict: 300B tokens is locked, GO on spot pricing — even the worst case fits.** MXFP8-
  conservative (no NVFP4 required) costs $2,980, leaving **$2,020 (40.4%) for post-training/RL** —
  right at the top of the required 30–40% reserve. Every better-than-worst-case outcome (better
  MFU, or NVFP4 landing) only grows that reserve — NVFP4-mid alone frees ~77% of the budget for
  post-training. This is a materially better position than earlier in this design pass, when 300B
  *depended* on NVFP4 working out — now it's pure upside, not a requirement.
- Calendar time at a representative middle scenario (300 TF, MXFP8-mid): **~75 days on 1 GPU,
  ~38 days on the 2-GPU VM** (same total dollar cost either way — 2 GPUs just runs it in roughly
  half the wall-clock). **Preemption-safe checkpointing is mandatory on spot** (frequent async
  orbax-style saves + auto-resume — infra exists from 1.3/1.5).

**Multi-GPU over PCIe (2× on Azure) — data-parallel only, and it's fine:**
- The wire carries **gradients of TOTAL params (~2.94B), not active (0.464B)** — over a big batch
  every expert gets tokens, so the full grad buffer ships. bf16 grads = **5.9 GB**; a 2-GPU
  exchange moves that once per optimizer step ≈ **0.1–0.2 s** at realistic NCCL-over-PCIe
  (25–50 GB/s eff.) — small relative to a real compute step.
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
- 1× pure-DP: fp32 masters 11.8 + AdaMuon ~23.5 + quantized compute weights ~3.5 (MXFP8 experts +
  bf16 rest) + bf16 grads 5.9 ≈ **~45 GB static** → ~51 GB for activations at 4096 (remat the
  recursion loop; chunked CE) — very comfortable.
- 2× with ZeRO-1: sharded masters+opt ~17.6 + weights 3.5 + grads 5.9 ≈ **~27 GB static** → large
  activation headroom → bigger microbatches → better MFU.
- Full training checkpoint ≈ 35–45 GB (masters + opt states + quantized weights) — write async;
  spot preemption makes checkpoint cadence a correctness requirement, not a nicety.
- **Scale context:** 300B = 6× 1.5's 50B — but facts come from *retrieval*, not parameters (§1.5), so
  the tokens buy reasoning / comprehension / grounding, not encyclopedic recall.

**De-risk order:** prove MXFP8 throughput + the packed MoRE path on a single card first (NVFP4 A/B
after — it's upside, not a gate); confirm the 2-GPU VM's real-world scaling before committing to
the full multi-week run. 300B tokens is locked regardless of the outcome (§6 budget table above) —
what the probe decides is calendar time and how much of the reserve gets spent on RL vs banked.

## 7. Suggested build order
1. Implement + unit-test the **packed MoRE block** (depth + adaptive-k) on tiny configs, CPU/1-chip.
2. Verify **parity**: soft-expected (train) vs hard-packed (infer) agree; routers don't collapse
   (softmax + z-loss); compute-budget aux loss hits target mean depth/k.
3. Throughput-prove the packed path on the RTX PRO 6000 (does it actually beat dense-envelope wall-clock?).
4. Tokenizer + 300B-token data pipeline (with the §5 fixes baked in — hybrid short/long SFT tagging
   scheme decided early since it constrains data collection, not just training recipe).
5. Curriculum PT run (dense warm-start → ramp) on the big allocation.
6. **Context extension**: single jump 4096→~164-200k (modest overshoot; deploy at 131k), 80/20
   old/new data mix, 90/10-ish long/short sequence mix — size exact ratios against real long-doc
   data once collected (§5).
7. Post-training (§5, final order): cold-start SFT → overall SFT (hybrid short/long mix) → DPD →
   multi-domain RLVR (GSPO + DAPO stabilizers + 10–90% on-policy filtering; STEM → coding →
   agentic, dynamic-thinking-length reward layered throughout) → pairwise RM/human-preference →
   eval (reuse `eval_metis.py`) → publish.

## 8. Open questions for you
- ✅ Params ~2.94B total / 0.464B active (Mamba **expand 2.0**, the paper default, restored) ·
  ✅ Tokenizer: new 65,536 · ✅ Recursion: full-stack loop (Ouro-style), **depth = passes =
  recursions+1**, target mean 2, max 5 · ✅ PT context 4096 · ✅ Extension target 131k ·
  ✅ **Hybrid ratio: 10 Mamba-2 + 2 attention @ ~4/~8 (12 layers)** · ✅ **G=16 granularity
  (128×512) confirmed** · ✅ Optimizer: AdaMuon — all locked.
- ✅ Hardware: **RTX PRO 6000 Blackwell on Azure, 1× probe → 2× (`NC288ds_xl`) for the full run**
  (Azure caps at 2 GPU/VM; committed to Azure for the compute credits) → **MoR-style packed
  adaptive-depth training**, **MXFP8 main lane** (NVFP4 = gated experiment on expert GEMMs, high
  upside if the sm_120 grouped-GEMM kernel risk resolves — its FP4 peak ≈ H100's FP8 peak) +
  fp32 masters.
- ✅ **300B PT tokens, locked** — §6's budget table shows this fits even in the conservative
  MXFP8/20%-MFU case ($2,980 of $5,000, leaving the required 30–40% reserve for post-training/RL).
  No longer contingent on NVFP4 landing — that's now pure upside.
- Per-pass LoRA/gate on the shared looped attention (Zamba2 finding) — implement as config flag; A/B.
- Depth gate mechanism: Ouro's two-stage early-exit vs a halting head (ACT/PonderNet) — A/B.
- ✅ **Post-training pipeline locked** (research-verified 2026-07-06, §5): cold-start SFT (hybrid
  short/long, tagged) → overall SFT → **DPD** (Nanbeige4-3B, arXiv 2512.06266) → multi-domain RLVR
  on **GSPO** (not GRPO — MoE-specific stability argument, arXiv 2507.18071) + DAPO-derived
  no-KL/loss-masking + 10–90% on-policy filtering (STEM → coding → agentic) with
  **dynamic-thinking-length reward layered on top** (DAST-style, reusing the same pass-rate number)
  → pairwise RM last. Two real open items inside this: (a) GSPO's stability claim is demonstrated at
  Qwen3-30B-A3B scale, not our 0.464B-active scale — worth an early small-scale sanity check; (b)
  exact λ for the length-shaping term, and exact GSPO clip ranges, need tuning, not just adoption.
- ✅ **Context extension locked as single-jump, not staged** (research-verified 2026-07-06, §5):
  4096 → ~164-200k (modest overshoot) → deploy 131k, over ~10–15B tokens, mixing in a small
  fraction of base-length sequences and an ~80/20 old/new data split. Exact overshoot ratio and
  short-sequence fraction need sizing against real long-document data once collected, not just
  borrowed from Nemotron's ratios at a much larger target length.
- RL costing — no longer decides token count (300B is locked), but still decides how much of the
  ~$1.0–2.0k reserve goes to which post-training stage (DPD data collection vs which RLVR domain
  gets the most rollouts vs agentic-RL depth).
