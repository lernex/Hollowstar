# Metis-1.7 Logos — Token-Superposition pretraining data plan

Status: **research draft, nothing locked.** Target generation: Metis-1.7 Logos.
Effective date: 2026-07-25

## 0. Scope and relationship to Metis-1.6

**This document changes nothing about Metis-1.6.** The 1.6 pretraining release
(`manifests/metis-1.6.yaml`, [`metis16_pretraining_data_plan.md`](metis16_pretraining_data_plan.md))
is locked at exactly 1,000,000,000,000 final-tokenizer exposures and is substantially acquired.
Improvements discovered during the 1.6 build are deliberately banked here rather than folded back
into a generation already in flight. A generation that keeps absorbing its own findings never
ships; the correct unit of improvement is the next generation.

What is banked here:

1. **Token-Superposition Training (TST)** as a first-phase pretraining regime for 1.7 Logos.
2. A **substantially larger token budget**, made affordable by TST's compute↔data exchange.

**Headline configuration (selected 2026-07-25, detail in §3.3):** full TST at `s=16`, **6.0T premium
recovery tokens + 24–25T medium superposition tokens = 30–31T total data**, costing **7.5T** of
compute in baseline-NTP-token units, for an estimated **~16.5T-NTP-equivalent** of quality. `r`
falls out at **0.200**; it is derived from the data split, not chosen.

Everything else about 1.7 (MoRE-Core, mHC, depth memory, N-gram conditional memory, hybrid
Mamba-2/attention, post-training sequence) is assumed inherited from the 1.6 architecture family
unless a section below says otherwise.

**Two 1.6 assumptions that are explicitly *not* inherited:**

- **The §1.5 retrieval-first product thesis is dropped.** Metis-1.6's "tool-using reasoner, not an
  encyclopedia" framing — minimal parametric facts, closed-book MMLU near random treated as expected
  and fine — was written against a 1T budget. At 30T it is obsolete and actively misleading:
  30T buys real parametric knowledge, and 1.7 should be evaluated closed-book as well as
  tool-augmented. Grounding, citation, and abstention remain first-class goals; *deliberate
  knowledge minimalism* does not. Do not carry the RAG-substitution argument forward.
- **The 65,536 vocabulary is superseded.** 1.7 targets ~131,072, which breaks the uint16 shard
  packing 1.6 relies on and forces uint32 (doubling token storage). See §5.6 for the interaction
  with `s`, and §10 for the open SuperBPE decision.

## 1. What TST is

Source: [Efficient Pre-Training with Token Superposition](https://arxiv.org/abs/2605.06546)
(Peng, Gigant, Quesnelle; Nous Research; arXiv 2605.06546v1, 7 May 2026). Full text read directly,
not summarized from secondary coverage.

TST is a two-phase pretraining schedule that leaves the inference-time architecture untouched:

1. **Superposition phase.** The tokenized stream is cut into non-overlapping contiguous bags of
   `s` tokens. In the embedding layer each bag is collapsed to one latent "s-token" by **averaging
   the token embeddings**. The target is the *next bag*, scored with a multi-hot cross-entropy
   (MCE) loss that is simply the mean of the `s` per-target standard CE terms:
   `L_MCE(z, y) = (1/|y|) Σ_{y ∈ y} L_CE(z, y)`. Labels are shifted left by `s − 1` before bagging
   so bag `[t, t+s−1]` predicts bag `[t+s, t+2s−1]`.
2. **Recovery phase.** Training resumes from the checkpoint with the TST code fully removed —
   ordinary next-token prediction, ordinary CE.

Equal-FLOPs per step is maintained by lengthening the **data** sequence by `s×`; the number of
*processed* positions is unchanged at `⌊L/s⌋`. So the model ingests `s×` more raw text per step at
identical per-step cost. `r` is the fraction of total steps spent in the superposition phase.

Two hyperparameters. The paper's robust bands are `s ∈ [4, 8]` and `r ∈ [0.2, 0.4]`, but optimal
`s` grows with model scale (3–8 at 270M, 6–10 at 600M, **16 at 10B**). Power-law `1/i` bag
weighting outperforms uniform averaging for `s ≥ 8` and is more stable there.

A model left in the superposition regime produces nonsense — it emits a mixed distribution over the
`s` future tokens. The recovery phase is not optional polish; it is what makes the model a language
model.

## 2. The evidence, verbatim from the paper's Table 1

All runs TorchTitan + FSDP on B200. Evals 0-shot, LM-Eval harness.

| Model | Params | TST steps | Total steps | Bag | TST tokens | Total tokens | B200-hrs | Final loss | HellaSwag | ARC-E | ARC-C | MMLU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense Baseline | 270M | – | 20,000 | – | – | 42B | 34 | 3.212 | 36.3 | 46.7 | 24.9 | – |
| Dense TST | 270M | 6,000 | 20,000 | 6× | 75B | 105B | 34 | 3.142 | 38.6 | 47.6 | 26.4 | – |
| Dense Baseline | 270M | – | 100,000 | – | – | 209B | 170 | 3.092 | 40.2 | 47.5 | 26.2 | – |
| Dense TST | 270M | 30,000 | 100,000 | 6× | 377B | 524B | 170 | 3.048 | 42.6 | 50.3 | 25.5 | – |
| Dense Baseline | 600M | – | 20,000 | – | – | 42B | 61 | 3.019 | 43.5 | 51.7 | 25.5 | – |
| Dense TST | 600M | 6,000 | 20,000 | 6× | 75B | 105B | 61 | 2.943 | 48.2 | 52.5 | 26.9 | – |
| Dense Baseline | 3B | – | 20,000 | – | – | 42B | 247 | 2.808 | 57.6 | 60.6 | 31.9 | 31.2 |
| Dense Baseline | 3B | – | 36,000 | – | – | 75B | 443 | 2.677 | 62.3 | 65.9 | 34.9 | 32.7 |
| Dense Baseline | 3B | – | 50,000 | – | – | 105B | 622 | 2.640 | 63.9 | 67.3 | 36.8 | 33.3 |
| Dense TST | 3B | 6,000 | 20,000 | 6× | 75B | 105B | 247 | 2.676 | 62.4 | 66.3 | 36.0 | 32.8 |
| **MoE Baseline** | **10B A1B** | – | 125,000 | – | – | **1.05T** | 12,311 | 2.252 | 70.1 | 73.8 | 46.3 | 37.4 |
| **MoE TST** | **10B A1B** | 12,483 | 49,983 | **16×** | 1.68T | **2T** | **4,768** | **2.236** | 71.2 | 74.2 | 47.3 | 39.0 |

The 10B A1B row is the directly relevant reference point for Logos (12B stored / A1.2B active).
It is a Qwen3-architecture MoE scaled to 10B total / 1B active, trained on a 50/50 mix of
FineWeb-Edu and DCLM at a constant 8.4M-token global batch, LR 3e-4, AdamW, WSD with 2,000-step
warmup and last-10% decay.

Expanded evals for that pair (paper Table 3): BoolQ 66.2 → 69.4, OpenBookQA 44.0 → **43.2**,
PIQA 77.2 → 77.4, Winogrande 61.3 → 63.0. Seven of eight benchmarks improve; OpenBookQA regresses
0.8. The correct summary is *slightly better across the board*, not *uniformly better*.

**The trade is not a quality tax.** At 10B A1B, TST reached a lower loss and better downstream
scores using **1.90× the data tokens** and **0.40× the compute**. This is a compute↔data exchange
rate, not a degradation.

### 2.1 The finding that constrains how far we can lean on this

Same 270M model, same `s=6`, same `r=0.3`, two total budgets:

| Budget | Baseline loss | TST loss | Δ |
|---|---:|---:|---:|
| 42B tokens (20k steps) | 3.212 | 3.142 | **0.070** |
| 209B tokens (100k steps) | 3.092 | 3.048 | **0.044** |

**The advantage fell 37% at 5× the token budget.** This is consistent with the paper's own
mechanistic reading of input superposition as "pre-pre-training" (§5.2): a coarse distributional
prior over local topic and co-occurrence, which a longer standard run learns anyway. The paper
concedes there is no scaling law — *"Future work could investigate scaling laws of token
superposition, in order to predict the best TST settings for larger model sizes."*

**Consequence for this plan: every multiplier in §3 is measured at ≤1.05T-baseline-equivalent and
must be treated as an upper bound when extrapolated to the selected ~16.5T-equivalent target.** The
1.7 plan must remain viable if the multiplier decays toward 1.0; §3.5 shows what that costs.

### 2.2 The failure mode that dictates the integration contract

Paper Table 2, Dense TST 3B, `s=6`, `r=0.3`, with the input embedding and output LM head randomly
re-initialized at the phase boundary:

| Run | Final loss |
|---|---:|
| Dense TST 3B | 2.676 |
| Dense TST 3B with randomization | **2.938** |
| Dense Baseline 3B (20k, matched steps) | 2.808 |

Perturbing the input/output representations at the boundary **eliminated the gains entirely and
landed below baseline** — the 6,000 superposition steps became wasted compute. The paper's
hypothesis (§5.3) is that representation continuity across the boundary is the reason TST works
where prior compressive methods needed explicit adapter alignment.

**This is the single most important constraint on the 1.7 integration.** Anything that changes
what the input embedding or LM head means at the phase boundary is a candidate for the same
failure. §5 is written around it.

## 3. Exchange rates and sizing

### 3.1 Derived multipliers

Both rows are derived arithmetically from Table 1 equal-loss pairs. Every step is equal-FLOPs, so
compute ratio is exactly the step ratio.

**`s=6`, `r=0.30` (3B dense, batch 2.1M):** TST 20,000 steps, loss 2.676 ≈ baseline 36,000 steps,
loss 2.677.
- Recovery NTP tokens: 14,000 × 2.1M = **29.4B**
- Superposition tokens: 6,000 × 2.1M × 6 = **75.6B**
- Baseline-equivalent: 75.6B NTP. So superposition supplied 75.6 − 29.4 = 46.2B NTP-equivalent.
- **Exchange rate: 1.64 superposition tokens ≈ 1 NTP token.**
- Compute multiplier 36,000/20,000 = **1.80×**. Data multiplier 105B/75.6B = **1.39×**.

**`s=16`, `r=0.25` (10B A1B, batch 8.4M):** TST 49,983 steps, loss 2.236 ≈ baseline 125,000 steps,
loss 2.252 (TST marginally ahead).
- Recovery NTP tokens: 37,500 × 8.4M = **315B**
- Superposition tokens: 12,483 × 8.4M × 16 = **1.678T** (matches the paper's reported 1.68T)
- Baseline-equivalent: 1.05T NTP. So superposition supplied 1.05T − 315B = 735B NTP-equivalent.
- **Exchange rate: 2.28 superposition tokens ≈ 1 NTP token.**
- Compute multiplier 125,000/49,983 = **2.50×**. Data multiplier 2.0T/1.05T = **1.90×**.

| Config | Compute multiplier | Data multiplier | Superposition tokens per NTP token | Data per NTP-equivalent |
|---|---:|---:|---:|---:|
| pure NTP | 1.00× | 1.00× | – | 1.00 |
| `s=6`, `r=0.30` (3B, verified) | 1.80× | 1.39× | 1.64 | 1.39 |
| `s=12`, `r=0.25` (interpolated, **unvalidated**) | ~2.24× | ~1.68× | ~2.02 | ~1.68 |
| `s=16`, `r=0.25` (10B A1B, verified) | 2.50× | 1.90× | 2.28 | 1.90 |

Larger bags buy compute and cost data. Smaller bags are gentler on the data budget. The `s=12`
row is linear interpolation of the exchange rate between the two measured points and carries no
empirical support; it is included because Logos sits at the scale where `s=16` was validated but
with a smaller vocabulary (§5.6).

### 3.2 The structural constraint

`r` is a **step** ratio. Because the superposition phase consumes `s×` tokens per step, hitting a
paper-band `r` forces a fixed *token* ratio between the phases:

`M / P = (r · s) / (1 − r)`

| `s` | `r` | Superposition : recovery token ratio |
|---:|---:|---:|
| 6 | 0.30 | 2.57 : 1 |
| 12 | 0.25 | 4.00 : 1 |
| 16 | 0.25 | 5.33 : 1 |

**This is not tunable independently.** A corpus that is mostly premium NTP data cannot also have a
meaningful `r`; pushing 90% of tokens into the recovery phase drives `r` to ~2% and TST becomes a
rounding error. TST commits the corpus to being majority-superposition by token count. §4 turns
that constraint into the design rather than fighting it.

### 3.3 Selected configuration

**Metis-1.7 Logos is sized by its data budget, not by a compute target.** The acquirable corpus is
the input; compute is whatever falls out. Selected 2026-07-25:

| Parameter | Value |
|---|---|
| Regime | **full TST** (input + output superposition) |
| Bag size `s` | **16** |
| Premium recovery tokens `P` | **6.0T** |
| Medium superposition tokens `M` | **24–25T** |
| **Total data** | **30–31T** |
| Step ratio `r` | **0.200** at `M=24T`; **0.207** at `M=25T` — *derived, not chosen* |
| **Compute** | **7.5T** baseline-NTP-token equivalents |
| Estimated quality | **~16.5T-NTP-equivalent** (`6T + 24T/2.28`) |
| Effective compute multiplier | **~2.20×** |

**`r` is not a free parameter here.** Fixing `s`, `P`, and `M` pins it through
`M/P = r·s/(1−r)`. At `s=16` and `M/P = 4.0`, `r = 0.200`. This sits at the **bottom edge** of the
paper's validated band `r ∈ [0.2, 0.4]` — inside it, but with no margin below. If `M` comes in
under 24T during acquisition, `r` falls out of the validated band and the configuration must be
re-derived (lower `s`, or less premium held back for recovery), not silently run out of band.

The multiplier is 2.20× rather than the paper's 2.50× precisely because `r=0.20 < 0.25` — less of
the run sits in the cheap phase. That is the correct trade, not a loss: premium tokens convert to
quality at 1:1, which beats superposition's 2.28:1, so spending more of the budget on premium buys
more quality per token even though it buys less compute leverage. Leverage is not the objective.

### 3.4 Sensitivity to the medium-tier budget — and why 24–25T stays

At the operating point, one extra medium token costs `1/16` token of compute and yields `1/2.28`
NTP-equivalent — roughly **7× leverage** against a premium token's 1:1. That is a real local
property and it is why medium-tier shortfall is much less alarming than premium shortfall.

**It does not license scaling `M` upward.** The table below holds the 2.28 rate constant, which is
precisely the extrapolation §3.5 warns against, and it becomes less defensible with every row:

| `M` | `r` | Total data | Compute | Nominal quality (constant-rate) |
|---:|---:|---:|---:|---:|
| 20T | 0.172 | 26T | 7.25T | 14.8T-equiv |
| **24T** | **0.200** | **30T** | **7.50T** | **16.5T-equiv** |
| **25T** | **0.207** | **31T** | **7.56T** | **17.0T-equiv** |
| 30T | 0.238 | 36T | 7.88T | 19.2T-equiv |
| 40T | 0.294 | 46T | 8.50T | 23.5T-equiv |
| 64T | 0.400 | 70T | 10.0T | 34.1T-equiv |

**Read the lower rows as an upper bound that will not be achieved**, for four compounding reasons:

1. **The exchange rate decays with budget.** §2.1 measures this directly. A superposition phase of
   40T is far deeper into diminishing returns than one of 24T, so the marginal token is worth
   progressively less than 1/2.28.
2. **Higher `r` means more to undo.** At `r=0.40`, 40% of steps produce a model that emits nonsense
   by construction. Recovery has proportionally more work to do, and the only large-scale evidence
   for the whole method sits at `r=0.25`.
3. **The quality floor drops.** Going 24T → 40T means reaching further down the web quality ladder.
   The leverage calculation assumes the marginal medium token is as good as the average one; that
   is false, and increasingly so.
4. **Pipeline cost is not free.** Global deduplication is the one stage that cannot stream, and its
   cost scales with corpus size. 30T already forces a conveyor-belt design; 46T or 70T changes the
   engineering problem rather than just the download bill.

**Recommendation: hold at 24–25T.** It puts `r` inside the validated band, sits close to the only
large-scale configuration anyone has published, and keeps the data pipeline tractable. Chasing the
lower rows trades a measured operating point for a speculative one.

**The table's real use is the other direction.** If premium acquisition lands short of 6.0T, this
shows how far medium can be pushed to partially compensate while keeping `r` in band — e.g. 5T
premium needs `M ≈ 20T` to hold `r ≈ 0.20`. That is the contingency it exists for, not an
aspiration.

### 3.5 Downside bound

The 2.28 exchange rate is measured at 1.05T-baseline-equivalent on a 10B-A1B model. This
configuration extrapolates it to ~16.5T-equivalent on a 12B model — roughly **13× beyond its
measurement point in overtraining ratio**, against a trend (§2.1) that says it decays. Assume decay
and check the cost:

| Actual exchange rate | Quality delivered | Comment |
|---|---:|---|
| 2.28 (measured at 10B-A1B) | 16.5T-equiv | plan estimate |
| 3.0 | 14.0T-equiv | mild decay |
| 4.0 | 12.0T-equiv | substantial decay |
| ∞ (superposition worthless) | 6.0T-equiv | total failure |

**The downside is bounded and mild.** Total failure means a 6T premium-NTP model that cost 7.5T of
compute — a 25% compute tax for nothing, with the premium corpus intact and unharmed. That is a
cheap bet, and it is why this configuration is acceptable despite the extrapolation. It is also why
§9's fallback is not a fallback in the usual sense: **the 6T premium NTP run is a subset of this
run, not an alternative to it.** Nothing is staked that is not recoverable by simply continuing.

The exchange rate is additionally assumed `r`-invariant when applied at `r=0.20` rather than the
`r=0.25` at which it was measured. Untested. The likely direction is favourable — a shorter
superposition phase sits earlier on a diminishing-returns curve — but it is an assumption.

## 4. Two-tier corpus design

The superposition phase's MCE objective destroys within-bag ordering. It can teach topic,
co-occurrence, vocabulary, and coarse local structure. It cannot teach anything for which order is
the semantics. This is a mechanical property of the loss, not a hypothesis, and it dictates the
corpus split.

### 4.1 Tier assignment rules

**Superposition tier (medium quality, order-insensitive).**
Prose web, encyclopedic text, science prose, multilingual. Tolerates medium quality because the
objective only extracts coarse distributional structure per document. This is where the tokens are
abundant, and it is the safest place in the entire pipeline to repeat data (§7).

**Recovery tier (premium, order-sensitive) — exclusive owner of:**

- **all code** — `def f(x): return x**2 + 3*x` as an unordered bag of 16 tokens is nearly
  information-free for syntax, and MCE actively teaches that those tokens are exchangeable;
- **all mathematics and formal material** — derivations, proofs, LaTeX equations, Lean/Coq/Isabelle;
- **all reasoning and CoT traces**;
- **all synthetic pedagogical data** — its value is in worked sequential explanation;
- **the premium cooldown** (the 1.6 Phase C analogue).

**This exclusion is a design decision taken on mechanistic grounds, not an empirical result.** The
paper's 10B A1B run used a 50/50 FineWeb-Edu/DCLM mixture — pure prose — in *both* phases. **There
is zero published evidence about TST on code or mathematics in either direction.** The exclusion is
the low-risk reading, and it has a convenient side effect: the scarcest data in the corpus is
preserved entirely for real next-token training.

The tiered split itself is also an extrapolation. The paper used the same mixture in both phases.
See the canary gate in §8.

### 4.2 Multilingual reallocation

Metis-1.6 gives multilingual 10B tokens (1.0%) and will produce an effectively monolingual model.
The superposition phase changes the economics: 3–5T of native multilingual text in the
superposition tier costs **190–310B tokens of compute** at `s=16` (÷16), and bag-of-words is
well-matched to what that data primarily teaches at first exposure — vocabulary and distributional
co-occurrence within a language.

Recommended: **3–5T multilingual in the superposition tier, plus 200–400B premium multilingual in
the recovery tier.** This is a materially better multilingual model for a rounding error in compute,
and it is a capability that only becomes affordable because superposition is cheap.

### 4.3 Candidate source pools

Sizes are approximate published or planning figures, before global cross-source deduplication.
Overlap between the web reservoirs is heavy; expect substantial shrinkage.

**Premium recovery tier — target 6.0T unique (§3.3 selected budget)**

| Category | Sources | Rough unique target |
|---|---|---:|
| Web HQ | Nemotron-CC v2/v2.1 high-quality tiers, FineWeb-Edu, DCLM-baseline | 3.5–4.5T |
| Code | Stack v2 permissive dedup, Nemotron repository code v3, Stack-Edu, fresh GitHub | 0.8–1.0T |
| Mathematics | MegaMath, Nemotron-CC-Math score 3–4, FineMath, Proof-Pile-2, OpenWebMath | 0.4–0.5T |
| Science | FinePDFs English edu tier, peS2o, PMC Open Access | 0.6–1.0T |
| Synthetic pedagogy | Cosmopedia v2, Nemotron synthetic QA/reasoning/textbooks, own generation | 0.5–1.0T |
| Reference / books / legal | as 1.6, scaled | 0.1–0.2T |
| Multilingual (premium) | FineWeb2 high-quality tiers, verified translation | 0.2–0.4T |

Synthetic pedagogy is the category to scale hardest. Metis-1.6 caps generated material at 8.6% of
exposures, which is conservative relative to 2026 practice; Nemotron-CC's own result is that
synthetic rewrites of web text outperform the source web text downstream. The 1.6 verification
gates (source genealogy, grounding record, generator identity/license, programmatic or execution
verification) should carry forward unchanged — the recommendation is more volume through the same
gates, not looser gates.

**Superposition tier — target 15–24T**

| Source | Rough available |
|---|---:|
| FineWeb (full, minus the Edu slice promoted to premium) | ~15T |
| Essential-Web v1, lower quality tiers | ~24T total |
| Nemotron-CC v2 mid tiers | part of ~6.6T |
| TxT360 | ~5T |
| Zyda-2 | ~5T |
| DCLM pool below the baseline filter | very large |
| FineWeb2 native multilingual | ~20T across languages |

The superposition tier is comfortably oversubscribed. This is the one part of the 1.7 data problem
that is not hard.

## 5. Architecture integration contract

TST's low risk in the paper rests on "no architecture change." Metis-1.7 Logos is a hybrid
Mamba-2/attention recursive MoE with four-stream mHC, typed depth memory, and N-gram conditional
memory — none of which was tested with TST. Every item below is a design proposal requiring canary
validation, not an established result. §2.2 is the governing constraint throughout.

### 5.1 N-gram conditional memory — key on the trailing real tokens of each bag

The N-gram tables are keyed on canonicalized suffix 2-gram/3-gram token IDs. During superposition
there are no per-position token IDs; the input is a mean of `s` embeddings.

- **Do not disable the pathway during superposition.** Switching a whole input pathway on at the
  boundary is the same class of perturbation as Table 2's re-initialization. The 1.6 plan's
  near-zero gate initialization makes it gradual rather than abrupt, which probably survives, but
  it also means the tables receive no training during the phase consuming ~85% of all tokens.
- **Do not re-key on bag-level hashes.** Different keying semantics across phases is a worse
  mismatch than either alternative.
- **Proposal: key on the trailing real token IDs of each bag.** Bags are contiguous and
  non-overlapping, so the token IDs still exist — only their embeddings were averaged. At s-token
  position `j`, keying on `(t_{js+s−2}, t_{js+s−1})` to help predict `t_{js+s}` is the **identical
  relationship** as in standard next-token training. The MCE loss weights that target at `1/s`, so
  the gradient is correctly aligned but attenuated.
- **Use power-law `1/i` bag weighting** (the paper's recommendation for `s ≥ 8` anyway). It puts
  the highest weight on the first target in the bag — exactly the token the N-gram path predicts —
  which partially recovers the `1/s` attenuation. This is a genuine synergy between the two
  mechanisms and a reason to prefer power-law weighting on independent grounds.

### 5.2 MoRE routing — off during superposition, ramped after the boundary

During superposition a "token" is a bag of `s`. The continuation router's difficulty distribution,
mean-depth budget loss, adaptive-`k` target, and halt calibration would all be fit to a
distribution that does not exist at inference.

- Run the superposition phase at **depth 1, fixed `k`** — which is exactly the dense warm-start the
  1.6 plan already specifies (§4 curriculum), extended to cover the whole phase.
- **Ramp MoRE approximately 5% of recovery steps *after* the boundary, not at it.** Recovery must
  be allowed to complete before a routing curriculum is layered on, or the two are confounded and
  an ambiguous result cannot be attributed.
- All routers still initialize at step 0 so checkpoints stay stage-flip compatible (1.6 lesson).
- Consequence: the routers see 0% of the superposition tokens and 100% of the recovery tokens. At
  the selected 6.0T recovery budget that is still 6× the entire 1.6 token budget. Acceptable.

### 5.3 Per-step FLOPs are not equal across phases

Depth 1 in superposition versus mean depth ~2 in recovery means superposition steps are materially
cheaper than recovery steps, violating the paper's equal-FLOPs-per-step premise in our favour. The
same compute buys more superposition steps than §3 assumes.

**Do not exploit this by inflating `r` beyond the validated band.** Keep `r` defined as a step ratio
matching the paper and bank the surplus as compute headroom. The multipliers in §3.1 then become
conservative rather than optimistic, which is the correct direction given §2.1.

Note also that depth 1 → adaptive depth is itself a discontinuity at the boundary. It is not the
input/output-embedding class that Table 2 shows is fatal, and the 1.6 plan already contains the same
discontinuity via its dense warm-start, so this is a monitoring item rather than a blocker.

### 5.4 Bag boundaries must respect document boundaries

The 1.6 packer uses EOS separators, document-aware masking, and Mamba-2 state reset at document
boundaries. A bag spanning two documents averages embeddings from unrelated text.

**Required packer change:** pad each document to a multiple of `s` during the superposition phase,
or drop the straddling bag. The current packer does not handle this and it is a silent corruption
if missed.

### 5.5 Effective context — the one place our architecture beats theirs

Equal-FLOPs is maintained by lengthening the data sequence `s×`. At `s=16` on a 4096 base context,
the superposition phase covers **65,536 raw tokens** of effective context at 4096 processed
positions. The paper flags this as unevaluated upside: *"this could likely have positive effects on
long context performance."*

Their all-attention Qwen3 model gained little from it. Logos is O(n) Mamba-2 with NoPE and a 131k
context target reached via a single direct jump (no RoPE OOD problem, no staged ladder). The
superposition phase is therefore effectively free long-context pre-conditioning at half the final
target length, plus reduced truncation of native long documents.

This argues for **larger** `s`, in direct tension with §3's data-efficiency argument for smaller
`s`. Resolve empirically in the canary sweep; do not assume either direction.

### 5.6 Vocabulary interaction — the 131k expansion helps here

Optimal `s` scales roughly inversely with tokenizer fertility: a larger vocabulary means each token
carries more text, so a fixed-size bag spans more characters and destroys more ordering information.

This is favourable for the selected configuration. The 10B A1B run validated `s=16` at Qwen3's
~151k vocabulary. Metis-1.6's 65,536 vocabulary has materially higher fertility, so `s=16` there
would have covered less text per bag than the validated point — meaning the transfer was uncertain
and arguably pointed above 16. **1.7's expansion to ~131,072 puts Metis in the fertility regime
where `s=16` was actually measured**, so the validated point transfers far more cleanly. This is an
independent argument for the vocabulary expansion beyond its own merits.

**SuperBPE must be decided before the `s` sweep, not after.** It reduces token count by roughly a
third, which lowers fertility further and pulls optimal `s` back down. Sweeping `s` against a
tokenizer that is then replaced wastes the sweep. Sequence: fix the tokenizer (vocabulary size and
SuperBPE yes/no) → then sweep `s ∈ {8, 12, 16, 20}` against it → then lock.

Do not assume `s=16` transfers on the strength of the fertility argument alone; the sweep is still
required. The argument only says the prior should be centred on 16 rather than above it.

### 5.7 Learning-rate schedule and boundary hygiene

- **One continuous WSD schedule across both phases.** Warmup and stable through superposition,
  stable through recovery, decay in the premium cooldown. No restart, no re-warmup at the boundary.
- At the boundary: same weights, same optimizer state, same schedule position. **Only the input
  bagging and the loss change.** Nothing else.
- The paper resumes with "the TST code fully removed" specifically to prevent contamination. Mirror
  that — the recovery phase should not be able to reach the bagging path at all.

### 5.8 mHC and precision

Four-stream recursion-aware mHC is internal to the block and agnostic to input granularity. Keep it
on unchanged through both phases so the residual topology is not perturbed at the boundary.

Precision: averaging `s` BF16 embeddings is numerically unremarkable. The MCE loss evaluates `s` CE
terms per position over a ~131,072-wide vocabulary, but there are `s×` fewer positions, so total loss
FLOPs and memory are unchanged — the existing FP32 chunked-logit accumulation policy carries over
without modification.

## 6. The data-neutral variant worth testing first

The paper's Figure 6 ablates input-only, output-only, and full superposition at `s=4`, `r=0.5`. All
three beat baseline; full superposition is best. But **output-only superposition** (ordinary token
inputs, bag-of-tokens *target*) increases no data consumption at all, and the paper's limitations
section names it explicitly for our situation:

> "In this alternative view, output-only superposition offers a significant advantage, as it
> outperforms the baseline pretraining regime without increasing data consumption."

For a project whose binding constraint is premium-tier data volume (§3.3), this may be the better
fit than full TST. It is only tested at small scale and `s=4`, so it is not a drop-in — but because
input granularity is unchanged it **sidesteps every problem in §5.1 through §5.6 simultaneously**:
N-gram keying works normally, routers see real tokens, bags never straddle documents, no packer
change, no vocabulary interaction.

**It must be an arm in the canary.** If output-only captures most of the gain, 1.7 gets a quality
improvement for zero additional data and near-zero integration risk.

## 7. Replay policy

The paper does not test data repetition in either phase.

**Position: the superposition phase is the safest place to repeat data anywhere in the pipeline.**
MCE over a bag destroys within-bag ordering, so verbatim sequence memorization is close to
mechanically impossible — and memorization is the mechanism behind the usual multi-epoch penalty.
The epoching penalty should therefore be attenuated in this phase specifically.

This is reasoning, not evidence. Cheap canary: superposition phase at 2 epochs of half the pool
versus 1 epoch of the full pool, matched total tokens.

**Practically, replay should not be needed.** The superposition tier is oversubscribed by 30T+
(§4.3). Treat replay as insurance against acquisition shortfall, not as a planned mechanism.

Recovery-tier replay follows ordinary rules — the 1.6 policy (unique through the capability phase,
bounded premium replay in the cooldown, no generated material in the cooldown) carries forward.

## 8. Gates before committing to TST for 1.7

Ordered. Each gate is cheap relative to what it protects.

1. **Measure Portage throughput on the fused Logos architecture first.** The §3.3 configuration is
   specified in data, so compute is an output rather than a constraint — but 7.5T
   baseline-token-equivalents is still ~7.5× the 1.6 run, and nothing in the repository yet
   establishes that it fits the allocation in usable calendar time. The 1.6 plan has exact token
   accounting and no measured tokens/sec; routed-expert execution is still a per-expert Python loop
   rather than a fused grouped GEMM. This gate decides whether 30T is a plan or a wish, and it is
   more valuable than anything else in this document.
2. **Run the TST canary at Praxis scale (A0.46B, 50–100B tokens)** inside the existing eight-run
   ablation wave. The harness exists; the marginal cost is small. Required arms:
   - full TST versus pure-NTP baseline at matched FLOPs;
   - **output-only superposition** (§6);
   - tiered medium-superposition/premium-recovery split versus the paper's same-mixture control;
   - code and mathematics included in versus excluded from the superposition phase (§4.1);
   - `s ∈ {8, 12, 16, 20}` sweep **against the final 1.7 tokenizer**, not 1.6's (§5.6);
   - N-gram trailing-token keying versus gated-off during superposition (§5.1);
   - superposition-phase replay, 2×half versus 1×full at matched tokens (§7).
3. **Verify boundary hygiene explicitly.** Reproduce the paper's Table 2 randomization control on
   our architecture to confirm the mechanism is present and that our boundary does not accidentally
   trip it.
4. **Confirm the packer handles document-aligned bagging** before any long run (§5.4).

## 9. What this plan does not claim, and the fallback

Not claimed:

- That the 2.28 exchange rate survives to a ~16.5T-equivalent target. §2.1 is direct evidence it decays
  with overtraining ratio, and the paper states no scaling law exists.
- That TST is safe on code, mathematics, or formal material. No evidence exists in either
  direction; §4.1 excludes them as the low-risk reading.
- That quality tiering across phases works. The paper used one mixture in both phases.
- That TST survives contact with recursion, mHC, typed depth memory, or hashed N-gram memory. All
  of §5 is proposal.

**Falsification criteria.** Abandon TST for 1.7 if the canary shows: recovery failing to reach and
pass the matched-FLOPs baseline; the N-gram tables measurably degraded at the boundary; loss of
router calibration that the post-boundary ramp cannot recover; or a tiered split materially worse
than the same-mixture control.

**Fallback, and it is a good one.** If TST fails every gate, 1.7 Logos runs **pure next-token
prediction on the 6.0T premium tier**, optionally topped up from the upper end of the medium tier.
That is already 6× the Metis-1.6 budget and is the single largest quality lever available to this
generation. Per §3.5 this is not a separate plan to switch to — the 6T premium NTP run is a subset
of the selected configuration, so the fallback is reached by continuing, not by restarting.

**The 1.7 token budget increase does not depend on TST and should be committed independently.**

## 10. Open items

- Measured Logos tokens/sec on Portage, and whether 7.5T-equivalent fits the allocation in usable
  calendar time (gate 1).
- **Tokenizer, which gates the `s` sweep.** Vocabulary is ~131,072 (up from 65,536), which forces
  uint32 shard packing and doubles token storage. SuperBPE and unigram/EM training are both open;
  both change fertility and therefore optimal `s`, so they must be settled before the sweep (§5.6).
  Any reserved control or delimiter tokens for downstream deployment must also be allocated here —
  the tokenizer is frozen at PT time and cannot be revised in post-training.
- Confirmation of `s=16` from the canary sweep. §5.6's fertility argument centres the prior on 16
  but does not replace the measurement. If the sweep prefers a different `s`, `r` must be
  re-derived from `M/P = r·s/(1−r)` and re-checked against the validated band (§3.3).
- Multimodality. 1.7 adds a ~200M ViT alongside the LM. TST is defined over token embeddings; how
  image tokens participate in — or are excluded from — the superposition phase is undesigned. The
  safe default is to exclude vision entirely from superposition and introduce it at or after the
  recovery boundary, but that interacts with §2.2's representation-continuity constraint and needs
  its own analysis.
- Whether Praxis-1.7 also gets TST. The paper's optimal `s` scales with model size (~8 at the
  Praxis class versus 16 at the Logos class); the canary runs at Praxis scale regardless, so this
  resolves itself as a by-product.
- Decontamination scope for the superposition tier. Running the full 1.6 pipeline (63 registry
  entries, five matching schemes) over 20T+ instead of 1T is a >20× Rhea CPU scale-up. Exact
  SHA-256 matching must be retained — bag-of-words can still surface an MCQA answer token — but the
  fuzzy 13-word, 8-word, and code-skeleton overlap passes are largely defusable for that tier since
  contiguous spans cannot be reconstructed from bags. Needs a decision with a written rationale, not
  a silent relaxation.
- Global deduplication scope and cost at 30T+ candidates. The governing constraint is that dedup is
  the one stage that cannot stream: retain MinHash bucket keys and the exact-dedup filter
  permanently, discard the text.
- Whether the superposition tier needs its own quality profile in
  `configs/metis17/quality-profiles.yaml` or can reuse a relaxed 1.6 diversity-tail profile.
- **Premium-tier reachability — the single largest risk in this plan.** 6.0T unique premium after
  global dedup is at or slightly beyond what the 2026 public reservoirs comfortably support, and
  unlike the medium tier there is no headroom to fall back on. Needs a source-by-source study of the
  kind [`metis16_replacement_data_research.md`](metis16_replacement_data_research.md) did for 1.6,
  including an explicit answer for what happens at 5T or 4.5T rather than 6T (§3.3: `r` leaves the
  validated band and the configuration must be re-derived).
- Education/tutoring data tier and the pretraining-versus-post-training split for deployment-specific
  behaviour and schemas. Scoped in conversation 2026-07-25 but deliberately not written here pending
  a decision on how far the therapy-adjacent surface should extend.

## 11. Explicitly out of scope

- **Metis-1.6 is untouched.** No source, phase total, quota, gate, or release contract in
  `manifests/metis-1.6.yaml` changes as a result of this document. The 1.6 acquisition proceeds as
  locked.
- No 1.7 architecture change is proposed here beyond the integration contract in §5. MoRE-Core,
  mHC, depth memory, N-gram memory, and the hybrid backbone are inherited as-is.
- Post-training is unchanged. TST is a pretraining-phase method that leaves the inference-time
  architecture identical, so the 1.6 post-training sequence
  ([`metis16_posttraining_contract.md`](metis16_posttraining_contract.md)) carries forward.
- Context extension is unchanged as a plan, though §5.5 suggests the superposition phase may make
  the single 4096→131k jump easier. Do not resize the extension budget on that speculation.
