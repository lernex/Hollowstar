# Metis-1.6 post-training: frontier research and recommended plan

Research pass 2026-07-25. All primary sources fetched and read directly (full text or HTML, not
summaries) except where marked *abstract only*. This document is advisory: it proposes changes to
`configs/metis16/posttraining.yaml` and §5 of `METIS_1.6_PLAN.md`, and states which of its own
claims are evidenced versus synthesized.

---

## 0. Executive summary

> **Revised the same day.** The first draft of this section was written before I had read the two
> closest architectural analogs — **Loopie** (arXiv 2607.16051, looped **MoE**, including a
> 6B-A0.6B in Praxis's exact class) and **Nanbeige4.2-3B** (a shipped Looped Transformer whose
> modeling code already contains LoopSplit, mHC with depth attention, and concatenated n-gram
> embeddings). Reading them changed three of the six recommendations. §6 contains the corrected
> analysis and supersedes items 2–4 below wherever they conflict. Items 1, 5 and 6 survive intact.

1. **The specialist → on-policy-distillation shape of the current pipeline is right, and is now the
   frontier default** (GLM-5.2, DeepSeek-V4, MiniCPM5-1B, MOPD all converged on it independently).
   Keep it. Do not switch to model merging — merging is measurably worse for multi-domain
   consolidation.
2. **~~Rollout fidelity is the make-or-break risk.~~** *Superseded by §6.2.* The risk is real but the
   correct response is not a heroic rollout engine — it is **running post-training at fixed depth**,
   which is what every team that has successfully post-trained a looped model actually did.
3. **~~Adopt RLTT per-pass credit as the highest-value addition.~~** *Superseded by §6.3.* Demoted to
   a proxy-scale A/B. Loopie post-trained a looped 128-expert MoE with plain GSPO, no per-loop credit
   and no reported instability, reaching AIME24 80.42 at A0.6B.
4. **~~Make depth a first-class action; route replay is mandatory.~~** *Superseded by §6.2 and §6.4.*
   Fix depth during RL instead; measure the routing mismatch before adopting R3 rather than assuming
   it.
5. **Two evidence-backed free wins, both cheap:** FP32 LM-head logits during RL (moved ScaleRL's
   *asymptote* from 0.52 → 0.61 — one of only two interventions in a 400k-GPU-hour study that moved
   the ceiling at all) and optimizer epsilon at 1e-15 (RL gradient magnitudes run 1e-18–1e-5, mostly
   below 1e-14).
6. **Retarget one specialist.** Replace closed-book `specialist_knowledge` with a
   grounding/abstention specialist using Abstain-R1's reward structure — verifiable, and it attacks
   Metis-1.5's #1 documented failure. **Rationale corrected 2026-07-25:** this originally rested
   partly on the plan's retrieval-substitution product thesis, which has since been retired
   (`METIS_1.6_PLAN.md` §1.5). The recommendation stands on the *verifiability* leg alone, which was
   always the stronger one — but see row 8 for the question that reopens.

**And the largest gap the first draft missed entirely (§6.5): SFT scale.** The two most
token-efficient small models both run instruction training at *pretraining* scale — Loopie's
"Supervised Pre-Training" is 2T tokens with SFT-style masking at global batch ≥1024, and MiniCPM5-1B
uses 200B tokens of deep-thinking SFT plus 200B of hybrid-thinking SFT for a 1B model. Metis plans
~30M instructions. That is the biggest single divergence between this plan and what actually works
at small scale.

---

## 1. What the frontier actually does now

### 1.1 Consolidation: specialists then on-policy distillation (converged)

Four independent 2026 pipelines use the same shape — train domain specialists in isolation so they
never fight over gradients, then distil them back into one unified student on-policy with reverse KL:

| System | Pattern |
|---|---|
| **MiniCPM5-1B** | RL teachers for math, code, closed-book QA, writing → OPD into the release model. RL+OPD raised the average by **16 points** and cut max-token-budget-hitting responses by **29 percentage points**. Reverse KL used *as the advantage estimate* inside the RL framework, top-k from both student and teacher, computed on the **union** of the two token sets. Prompts reused from each RL teacher's own training set — no extra data curation. |
| **DeepSeek-V4** | Per-domain expert (SFT then GRPO), then one unified model via reverse-KL on-policy distillation against the specialist teachers. |
| **GLM-5.2** | Specialists each consume their full compute budget on a single capability space, "entirely avoiding cross-domain gradient interference," then OPD merges them. |
| **MOPD** (2606.30406) | Formalizes it: per-prompt teacher *routing* (one teacher chosen by domain, not logit averaging), beats single-teacher OPD, weight merging (TIES/DARE/AdaMerging), and multi-domain RL on one model. |

Merging is the losing branch: "directly merging independently trained domain experts often degrades
performance because their raw task deltas contain incompatible residual directions."

**Read for Metis:** `opd_consolidation` is correctly designed and correctly placed. The `top_k_per_model: 32` + `union_student_and_teacher_top_k: true` + `sealed_prompt_domain` routing choices all match the frontier.

### 1.2 OPD is now mechanistically understood — and it has hard constraints

*Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe* (2604.13016) is the
most decision-relevant paper in this pass.

**Two success conditions, both necessary:**
- **Thinking-pattern compatibility**, measured as top-k overlap ratio between student and teacher.
  Successful runs rise ~72% → 91%; failed runs are flat from initialization. "Early-stage
  thinking-pattern mismatch causes a loss of distillation benefit that cannot be recovered later."
- **Genuinely novel teacher capability.** Same-pipeline teachers give limited improvement;
  RL-post-trained teachers give substantially stronger gains.

**Teacher benchmark score does not predict OPD outcome.** In their reverse-distillation experiment,
distilling the RL-trained JustRL-1.5B toward its own pre-RL checkpoint regressed it *almost exactly*
to pre-RL performance — and swapping in the larger, higher-scoring R1-Distill-7B produced a
"nearly indistinguishable" trajectory to the same regressed level. OPD acquires the teacher's
thinking patterns and overwrites the student's.

**Mechanism:** overlapping tokens carry **97–99% of probability mass** for both models throughout
training. Optimizing only the overlap region recovers nearly the full benefit; the non-overlap region
contributes little. Overlap optimization is self-reinforcing.

**The constraint that bites Metis:** teacher supervision quality degrades with prefix depth. The
teacher's accuracy advantage when continuing from a student prefix falls from **+0.37 at 1K tokens to
+0.02 at 16K**. Response-length sweet spot is **3K–7K**; beyond ~10K they observe late-stage overlap
collapse, entropy spikes and gradient-norm instability. They state it does "not extend cleanly to
longer-horizon settings."

**Recipe:** reverse KL on student rollouts; sampled-token or top-k with k=4–16 (top-1 fails; sampled
token ≈ top-k); temperature 1.0; LR 1e-6; **1 epoch**; 4 rollouts/prompt; teacher-aligned prompt
templates help but lower entropy, so mix in out-of-distribution prompts; if initial overlap is low,
do an off-policy cold start (SFT on ~200K teacher rollouts) first — the two-stage version
"substantially outperforms pure OPD."

**Diagnostics to log:** overlap ratio, overlap-token advantage, entropy gap, gradient norm,
per-position entropy.

Related: *Rethinking On-Policy Self-Distillation for Thinking Models* (2607.05184) finds
privileged-context self-distillation *hurts* thinking models (up to −17% relative avg@16), worst at
long reasoning budgets, by suppressing high-entropy **forking positions** and reducing verification/
backtracking/hedging markers. Plain OPD still helps. Do not add a privileged-context self-distil
stage.

### 1.3 RL algorithms: what is settled

**ScaleRL** (2510.13786, >400k GPU-hours) fits RL compute-performance to a sigmoid
`R_C − R_0 = (A − R_0) / (1 + (C_mid/C)^B)` and separates interventions that move the **asymptote A**
from those that only move **efficiency B**:

- Asymptote movers: **loss type** (CISPO and GSPO both substantially beat DAPO) and the
  **FP32 LM-head logits fix** (A: 0.52 → 0.61).
- Efficiency-only: loss aggregation, advantage normalization, curriculum, zero-variance filtering,
  off-policy pipelining.
- Final recipe: CISPO; batch-level advantage normalization; prompt-level loss aggregation; FP32
  logits in *both* generator and trainer; no-positive-resampling (drop prompts with ≥0.9 historical
  pass rate); PipelineRL-8; **forced interruption via an end-of-thinking phrase rather than length
  penalties**; zero-variance filtering.
- Fit after ~1.5k GPU-hours; extrapolating 8k → 16k GPU-hours held; asymptote reproducible to ±0.02.
- **A 17B×16 MoE scaled predictably and reached a much higher asymptote than the 8B dense model
  using 1/6 of its RL compute.** Encouraging for Logos.

**CISPO** (MiniMax-M1, 2506.13585) clips the *importance weight* with a stop-gradient rather than
clipping/dropping token updates:
`J = E[ (1/Σ|o_i|) Σ sg(clip(r_i,t, 1−ε_lo, 1+ε_hi)) · Â_i,t · log π_θ(o_i,t) ]`, with ε_lo
effectively disabled and only ε_hi tuned. The motivation is directly relevant to a recursive
reasoner: rare reflective tokens ("However", "Recheck", "Wait", "Aha") carry high IS ratios, get
clipped out after the first on-policy update under PPO/GRPO, and then never contribute again —
yet they are "crucial for stabilizing entropy." CISPO matched DAPO in 50% of the steps (2× speedup).
MiniMax-M1 is a hybrid-attention MoE, the closest published architecture to Metis.

Their other RL-engineering findings, all portable:
- Train/inference probability correlation was ~0.90; root-caused to high-magnitude activations in the
  LM head; fixed to 0.99 by making the **LM output head FP32**.
- AdamW **β1=0.9, β2=0.95, eps=1e-15** because gradients spanned 1e-18 to 1e-5 with the majority
  below 1e-14.
- Pathological repetition early-stop: halt generation if 3,000 consecutive tokens each have p > 0.99.
- Under length scaling, negative samples hit the context limit faster than positives, concentrating
  large negative gradients in late segments → fixed with combined sample-level and token-level loss
  normalization.

**Entropy control:** entropy collapses early and coincides with plateaus. Clip-Cov / KL-Cov target
the high-covariance tokens that drive collapse. A small set of high-entropy "forking" tokens carries
most of the credit — updating only ~20% of tokens suffices and low-entropy tokens can hurt.
Separately, **clip-low raises entropy and clip-high lowers it** (2509.26114), which makes asymmetric
clip bounds an explicit entropy dial rather than an arbitrary pair of constants.

**Curriculum / replay:** ExGRPO buckets trajectories by correctness and prioritizes
intermediate-difficulty prompts with low-entropy trajectories; Prompt Replay prioritizes prompts near
pass-rate 0.5 while preserving on-policy optimization. Consistent with the 10–90% avg@16 band already
in the Metis contract.

**Forgetting:** *RL's Razor* (2509.04259) — forgetting is governed by forward KL between tuned and
base policy *on the new task*, not by the algorithm; on-policy RL is biased toward KL-minimal
solutions and therefore forgets less than SFT at matched new-task accuracy. In a 10-stage chain, the
SFT stages are the dangerous ones.

**Verifier integrity:** imperfect verifiers admit false positives, which is the direct route to
reward hacking. Two concrete hardening methods: **verifier fuzzing** (2606.01066 — adversarial
completions, buggy-vs-strict reference comparison, reports false-positive/negative/exploit rates) and
**isomorphic perturbation testing** (2604.15149 — enforce invariance under logically isomorphic task
variants). Mitigations reduced reward hacking by up to 54.6%.

### 1.4 MoE-specific RL: routing alignment is a separate axis from the loss

**R3 — Rollout Routing Replay** (2510.11370), the most important MoE-RL result:

- Train/inference token-probability KL: **1.535e-3 for MoE (Qwen3-30B-A3B) vs 6.4e-4 dense
  (Qwen3-8B)**. With R3: **7.5e-4**, i.e. essentially dense-like.
- **~10% of routers select different experts** between training and inference; **94% of tokens select
  a different expert in at least one layer**; ~6 routers differ per token. Extreme-discrepancy tokens
  are an order of magnitude more frequent than in dense models, and R3 cuts that frequency by an
  order of magnitude.
- Mechanism: cache only the **binary top-K expert mask** from the inference engine and substitute it
  into the training gate,
  `g_replay,i = I_infer,i · exp(s_train,i) / Σ_j I_infer,j · exp(s_train,j)`, for all MoE layers.
  Gradients still flow to the router logits. No auxiliary balancing loss added.
- Results (AIME24/25, AMC23, MATH500-L5 average, with crash step):

  | | Avg | Crash |
  |---|---|---|
  | GRPO | 48.84 | step 120 |
  | GSPO | 66.76 | — |
  | GRPO+R3 | 68.05 | — |
  | **GSPO+R3** | **69.00** | — |

  Single-step config: GRPO 62.23 (crash 60), +TIS 66.24 (crash 105), **+R3 71.83 (no crash)**.
- Caveat: R3 does not remove intra-framework nondeterminism (8.4e-4 KL between two identical
  Megatron forward passes).

**PR² — Predictive Routing Replay** (2606.00395) addresses R3's *staleness*: frozen replay is only
valid while cached and current routes stay close. A zero-initialized per-layer linear predictor adds
a logit bias trained with a stop-gradient KL to the current router. Zero-deviation ratio under off-2
training: GRPO 76.9%, routing replay 88.8%, **PR² 91.5%**. On Qwen3-30B-A3B, GRPO+PR² averaged
46.80 vs GRPO+R2 37.55 and GSPO 35.76 (off-2), holding up at off-4 and off-8. **But on OLMoE-1B-7B
the gains shrink to ~0.5–1.4%** — the benefit scales with how off-policy the pipeline is, and PR²
only applies to disaggregated rollout/training pipelines.

**RSPO** (2510.23027, ACL 2026) names the same phenomenon "router shift" and instead rescales
importance-sampling weights using router logits; 77.1 average Pass@1 across five reasoning
benchmarks. *Abstract-level detail only.*

**Read for Metis:** the loss choice (GSPO/CISPO) and the routing-alignment choice (R3/PR²/RSPO) are
orthogonal, and **GSPO+R3 beat both alone**. Metis should do both.

### 1.5 Looped / recursive / latent-depth models under RL — the thin, decisive literature

This is where Metis has almost no cover, so I read everything I could find.

**Ouro's own RLVR failed.** Their exploratory RLVR alignment "did not yield significant performance
gains over the final SFT checkpoint," and the stated cause is infrastructure: their vLLM-based system
"could not efficiently perform rollouts with dynamic early exits and subsequently use that
variable-depth information for the update step." Treat this as the single most important prior for
Metis's post-training.

**RLTT — per-loop credit assignment** (2602.10520) is the fix that worked. Outcome-only GRPO credits
only the *final* latent state before token emission, which mismatches a model that performs T_max
internal refinements per token. RLTT replaces
`∇ log P^(T_max)(y|·) · Â` with `Σ_t ω_t ∇ log P^(t)(y|·) · Â`, `Σω_t = 1`, with three weighting
options: uniform, progressive (`t^α`), or **exit-probability** (using the model's own learned halting
signal).

| Benchmark mean | Ouro-1.4B GRPO | +RLTT | Ouro-2.6B GRPO | +RLTT |
|---|---|---|---|---|
| Math (MATH500/AIME24/AIME26/BeyondAIME/GSM8K) | 41.7 | **46.0** | 42.5 | **51.2** |
| Non-math (ARC-C/MMLU-ST/GPQA/MBPP) | 59.5 | **64.8** | 65.2 | **71.8** |

GPQA 19.7 → 38.4 and AIME24 16.7 → 33.3 at 2.6B. Paired t-tests significant on 8/9 benchmarks at
2.6B. Secondary effects: **responses got shorter with no brevity incentive**; 10% less wall-clock than
GRPO; at *constrained* loop budgets the advantage is largest (GSM8K at 1 loop: 33.2 → 59.4;
MATH500 at 2 loops: 66.2 → 81.2). Costs: per-loop log-probs must be retained, which forced them to
halve token packing per GPU (16,384 → 8,192) and add gradient accumulation.

**The critical limitation:** "our experiments use a fixed loop depth during training and inference,
which sacrifices Ouro's native ability to adaptively choose when to early-exit." They ran T_max = 4
fixed. **So no published work has done RL with adaptive per-token depth.** The credit-assignment
survey (2604.09459) confirms the gap explicitly: it is organized entirely around observable action
sequences, and "there is no discussion of latent thinking, recursive depth, or intermediate
computations within a single token generation."

**SLPO** (2607.19691, July 2026) supplies the missing formalism for making internal decisions
RL-trainable. For continuous latent states it estimates a surrogate Gaussian likelihood from K=4
dropout forward passes, treats realized states as stop-gradient targets, and — the part that matters
for Metis — writes the **trajectory likelihood as the sum of three terms**:

```
log π(ξ|x) = Σ_t log π(h_t | x, h_<t)        # latent steps (surrogate)
           + Σ_s log π(a_s | x, h_1:τ, a_<s)  # answer tokens
           + log P(τ)                          # the stopping time itself
```

with RLOO advantages over G=8 rollouts, LR 1e-6. Stage 1 is a **stopping-gate cold start**: sample
trajectories, enumerate stopping lengths t ∈ [T_min, T_max], decode at each, build the *valid-stop
set* (lengths that yield a correct answer), and train the gate to place probability mass there:
`L_stop = −(1/N) Σ_n log Σ_{t∈V} P(τ=t)`, `P(τ=t) = ρ_t Π_{k<t}(1−ρ_k)`. The learned gate then
allocates more depth to harder problems (r = 0.30). Gains are modest in absolute terms (GSM8K
55.22 → 55.27 Acc but Pass@16 67.48 → 70.28 on CODI/Llama-1B) and it costs ~32× forward evaluations
per step; the *formalism* is the contribution, not the numbers.

**Metis is strictly easier than SLPO's setting**: its continuation controller already emits an
explicit per-token Bernoulli `p(continue)`, so `log P(τ)` is directly computable — no surrogate
Gaussian needed. The same is true of the adaptive-k choice.

Supporting work:
- **LSRL** (Findings EMNLP 2025): process-supervised GRPO on latent recurrent states — decode *every*
  recurrent depth and grade the partial solution. +4.27 GSM8K, +2.06 MathQA over a depth-8 baseline
  on Huginn. Real but expensive (an external grader per depth).
- **Learned stochastic stopping** (2606.29983): a stopping head parameterizing a categorical
  distribution over depths, trained by REINFORCE on negative sampled-depth loss with entropy
  regularization; stabilizes extrapolation *beyond* trained depth and allocates depth to difficulty
  without supervision. Smoother than a deterministic threshold gate.
- **LoopFormer** (2602.11451): conditions each loop step on internal time `t` and step size `Δt` and
  trains a **shortcut-consistency** objective aligning short trajectories to the full trajectory's
  final representation. Result: choose any budget M ≤ L at inference, no retraining. This is the
  published mechanism for the plan's "caps 1–5 must be supported operating points" requirement.
- **S-GRPO**: decaying reward by exit position — rewards *earlier* correct exits.
- **Switchable latent reasoning** (2606.13106): on-policy RL where recurrence continuation is an
  action and rollouts are variable-length; helps on multi-step problems, hurts on simple ones.

### 1.6 Grounding, abstention and non-verifiable domains

**Abstain-R1** (2604.17073) gives a directly implementable reward for the plan's product thesis:

- format reward (tags + `\boxed{}`): 1/0
- answerable: **+1** correct, **−1** if it emits boxed "I don't know", 0 otherwise
- unanswerable: **1.0** if it abstains *and* the clarification (what's missing) is verified correct,
  **0.3** if it abstains without a correct clarification, 0 otherwise
- data: 30% unanswerable / 70% answerable; GRPO, LR 1e-6, β=0.001, clip 0.2, 5 rollouts, 4096-token
  responses, 100 steps ≈ 20 GPU-hours on 4×A100

Results on a 3B backbone: U-Ref (refusal on unanswerable) 9.4% → **68.1%**, correct clarification
0.6% → **55.1%**, answerable accuracy 48.8% → **57.2%**, false-unknowns up only 1.6pp. It generalizes
to unseen abstention benchmarks and **a 3B model matches or exceeds 7B/32B baselines on abstention
metrics** — the signal matters more than scale. Open-ended/tool-augmented transfer is explicitly
future work.

Also: **TruthRL** (ternary reward under GRPO to encourage abstention under uncertainty),
**TIAR** (trajectory-informed advantage reweighting for abstention), **faithfulness-aware step-level
RL for small reasoning models** (2602.05897), and search-agent abstention (2607.10738).

For non-verifiable domains, rubrics have displaced plain judges: **Rubrics as Rewards** (2507.17746,
+31% HealthBench / +7% GPQA-Diamond over LLM-judge baselines), **Open Rubric System** with pairwise
adaptive rubrics (pointwise scalarization has a discriminability ceiling and is gameable),
programmatic **Judge Code** rubrics, and pairwise GenRM + BRPO for hacking resistance.

### 1.7 Small active-parameter models

Consistent finding: at 0.5–1B, RL alone suffers sparse rewards and instability; distillation first,
then RL, is the working order. Data quality matters more at small scale than large. This supports the
existing plan's sequencing (heavy SFT/distillation before RL) and argues for keeping the OPD
consolidation stage rather than trying to get everything from RL.

### 1.8 Infrastructure reality check

- **verl runs on ROCm** (MI300X/MI325X/MI355X) with FSDP/FSDP2/Megatron backends and vLLM as the
  validated inference engine (SGLang in progress). **slime** has an AMD MI300X tutorial. LMSYS
  published ROCm support for large-scale RL post-training on AMD Instinct (March 2026). So the
  generic stack exists on Metis's hardware.
- **Hybrid SSM models are now first-class in vLLM**, including disaggregated prefill/decode with
  Mamba conv-state transfer (April 2026), plus ReplaySSM-style state recompute. The Mamba half of
  Metis is no longer exotic; the recursion + mHC + depth-memory half still is.
- **FP8 RL is a known hazard.** "BF16-train + FP8-rollout suffers severe training instability and
  catastrophic accuracy collapse under long-rollout generation." Two published fixes: FP8-RL
  (2601.18150 — blockwise FP8 W8A8 + FP8 KV with per-step QKV scale recalibration + TIS/MIS
  correction, +44% rollout throughput) and **Jet-RL** (2601.14243 — *identical* quantization flow for
  training and inference, eliminating the mismatch and removing calibration entirely). Metis is
  FP8-first, so this is directly on the critical path.
- Rollout efficiency for small-active models: adaptive rollout allocation (2602.01601), variance-
  informed budget allocation, quantized rollout (2602.13953), APRIL partial rollouts (works on both
  H100 and MI300).

---

## 2. Why the standard recipe does not transfer to Metis unmodified

Metis-1.6 simultaneously has five properties that published RL recipes do not assume:

| Property | Consequence for RL |
|---|---|
| Per-token learned depth, 1–5 passes, monotonic packed active set | The halting decision is an action with no published RL recipe. If it is excluded from the policy, RL shifts the hidden states that drive halting, so the depth distribution silently drifts between rollout and update — Ouro's exact failure. |
| Expert routing re-decided *every pass* | 12×5 = 60 routing sites/token (Praxis), 20×5 = 100 (Logos), vs 48 in the model where R3 measured 94% per-token divergence. A pass-1 flip changes the state entering pass 2, so errors compound rather than average. |
| 4-stream mHC + typed depth memory + cached N-gram rows | A separate rollout engine must reproduce all of this bit-similarly, or the logprobs being differentiated are not the ones that generated the data. |
| ~0.47B active (Praxis) / ~1.19B (Logos) | Sparse-reward regime; distillation-first is required; but rollout is cheap, which makes a single-engine design affordable. |
| FP8-first on MI300A | Compounds the train/infer mismatch that ScaleRL, MiniMax-M1 and the FP8-RL papers all independently identify as the dominant failure. |

---

## 3. Recommended plan

### 3.1 Four cross-cutting mechanisms (build these before any stage runs)

**M1 — Rollout fidelity contract (highest priority; this is the Ouro lesson).**

Adopt a **single-engine rollout** for v1: generate with the training forward path itself, not a
separate inference stack. Reasons: it structurally eliminates the routing mismatch, the depth
mismatch, the mHC/memory/N-gram reimplementation risk, and the FP8 rollout mismatch in one move —
four of the five documented failure classes. At 0.47B active on 128 MI300A APUs this is affordable;
revisit only if measured rollout throughput blocks the campaign. Required regardless:

- **FP32 LM head during all RL and OPD stages.** Move `lm_head` out of `fp8_roles` for post-training.
  Evidenced twice independently (ScaleRL asymptote 0.52→0.61; MiniMax-M1 correlation 0.90→0.99).
- **Unified precision flow** (Jet-RL principle): whatever precision generates must be the precision
  that scores. If a separate engine is later introduced, add TIS/MIS correction.
- **Route replay (R3)**: cache the binary top-k expert mask per (layer, pass, token) at rollout and
  substitute it into the training gate. Budget: k≤8 × 60 sites × response tokens; cache response
  tokens only, never the prompt (at 131k agentic contexts, prompt-inclusive caching is ~128MB/seq).
- **Continuation replay (new; the depth analogue of R3)**: cache the realized per-token halt/continue
  decisions and force the training forward to reproduce them via the existing `force_depth` tensor
  path in `_continuation_decision`. Without this, the training pass re-derives depths from updated
  weights and computes gradients for a *different* computation than the one that earned the reward.
  This is my synthesis, not a published method — flag as such, and validate it with a parity test
  (rollout logprobs vs recomputed logprobs, target correlation ≥0.99, the MiniMax-M1 diagnostic).
- **Numerics**: optimizer eps 1e-15 for RL stages, β2 0.95; repetition early-stop at 3,000
  consecutive tokens with p>0.99.

**M2 — Depth-aware policy and per-pass credit.**

- Include the depth trajectory in the sequence likelihood, SLPO-style:
  `log π(seq) = Σ_tokens log π(token) + Σ_tokens log P(τ_token)`, where
  `P(τ=r) = p_r Π_{j<r}(1 − p_j)` from the continuation controller's own Bernoulli outputs. Metis can
  compute this exactly — no surrogate needed.
- Use a **sequence-level ratio** (GSPO) rather than token-level, because per-token depth ratios would
  be even higher-variance than the per-token expert ratios GSPO already fixes; and apply
  **CISPO-style weight clipping with stop-gradient** so a clipped sequence still contributes gradient
  instead of being dropped — preserving the rare reflective tokens that CISPO was built to protect,
  which is exactly the behaviour a recursive verifier-style reasoner needs.
- Add **RLTT per-pass credit**: `Σ_r ω_r ∇ log P^(r)(y|·) · Â` with `ω_r` = the model's own survival
  gates (RLTT's best-performing "exit PDF" weighting *is* Metis's `pass_survival_gate`). Implement by
  substituting advantage for cross-entropy in `_chunked_weighted_causal_loss_sum`. Expect the memory
  cost RLTT reported — plan to halve microbatch token packing and compensate with grad accumulation.
- Keep asymmetric clip bounds as a deliberate **entropy dial** (clip-low raises entropy, clip-high
  lowers it), not as fixed constants. Current config's 3e-4/4e-4 is a starting point to sweep.

**M3 — Depth calibration before RL (cheap, high-value, ships this week).**

Port SLPO's stopping-gate cold start using Metis's existing `force_depth`: for each prompt in a
calibration set, run forced depths 1..5, record which depths produce a correct answer (the valid-stop
set), and supervise the continuation controller with
`L_stop = −log Σ_{r ∈ valid} P(τ=r)`, biased toward the *smallest* correct depth. Cost: 5 forward
passes per prompt, no RL machinery. This directly targets the halt calibration that the training
health gates already watch (`halt_collapse_fraction`) and the eval gate already scores
(`recursion_depth_distribution`, `underthinking_rate`, `overthinking_rate`).

**M4 — Two-axis compute control (tokens *and* depth).**

Metis is the only model here with two compute axes, and they must be shaped separately:

- **Tokens**: keep the planned DAST-style pass-rate-conditioned budget (λ=0.05), but switch the
  mechanism to ScaleRL's **forced interruption via an end-of-thinking phrase** rather than a length
  penalty — the asymptote evidence favours interruption, and it avoids the documented
  "negative samples hit the context limit first" gradient pathology.
- **Depth**: add an S-GRPO-style **decaying-reward-by-exit-pass** term, gated on the same avg@16
  pass-rate so it only applies where the model already succeeds. λ_depth separate from and smaller
  than λ_length; correctness must always dominate.
- **Elastic depth**: train with stochastic depth caps plus LoopFormer-style shortcut consistency
  (align the cap-M representation to the cap-5 representation) so caps 1–5 are all shippable
  operating points. This yields a product feature no competitor has: an **effort dial that costs no
  extra tokens**. Note RLTT's finding that per-pass credit gives its *largest* gains precisely at
  constrained loop budgets — the two mechanisms reinforce each other.

### 3.2 Revised stage sequence

Changes from `configs/metis16/posttraining.yaml` are marked **[NEW]**, **[CHANGE]**, **[CUT-RISK]**.

| # | Stage | Recommendation |
|---|---|---|
| 1 | `context_extension` | Keep as specified (single jump 4096→163,840, 18B tokens, gates 6/12/18B). **[NEW]** add depth-cap conditioning + shortcut consistency here, while it is still a cheap CPT-style objective. |
| 2 | `cold_start_sft` | Keep. **[CHANGE]** align response templates to the DeepSeek teacher's format — template mismatch alone measurably degrades later OPD overlap. |
| 3 | `overall_sft` | Keep (this is also the off-policy cold start that 2604.13016 shows is needed to raise initial overlap ratio to ~80%+ before any distillation). |
| 4 | **[NEW]** `depth_calibration` | M3 above. Insert between `overall_sft` and DPD. Non-RL, ~5 forwards/prompt. |
| 5 | `deepseek_dpd_pilot` → `deepseek_dpd` | **[CUT-RISK]** Keep the pilot and its gate, but be prepared to cut the full stage. With token-level distillation already disabled (`token_distillation_weight: 0.0`), what remains is a sequence-level DPO margin, and the strongest 2026 evidence for external-teacher bootstrapping is plain off-policy SFT on teacher rollouts — which stage 3 already does. If the pilot gate is marginal, spend those APU-hours on specialist RL instead. **[NEW]** log the overlap-ratio diagnostic here; it is the cheapest early predictor of whether the later OPD will work at all. |
| 6 | `specialist_reasoning` | Keep domain. **[CHANGE]** algorithm → GSPO sequence ratio + CISPO weight clipping + **R3 route replay + continuation replay** + RLTT per-pass credit. Add ScaleRL's batch-level advantage norm, prompt-level aggregation, zero-variance filtering, no-positive-resampling (≥0.9 pass rate). Keep 10–90% avg@16 filtering. |
| 7 | `specialist_code` | Same algorithm changes. **[NEW]** harden verifiers with fuzzing + isomorphic perturbation testing before the stage runs — the config already demands `adversarially_validated: true`; these give it a named method. |
| 8 | `specialist_knowledge` | **[CHANGE] retarget to `specialist_grounding`**: retrieval-grounded QA + calibrated abstention + citation faithfulness, using Abstain-R1's exact reward table and 30/70 unanswerable/answerable mix. Verifiable (does the claim appear in the retrieved span?), so it stays inside RLVR. **[RATIONALE CORRECTED 2026-07-25]** The original justification also cited the plan's retrieval-substitution product thesis, which is now retired (`METIS_1.6_PLAN.md` §1.5) — closed-book knowledge is no longer a battle Metis declined to fight. The retarget still holds, but on verifiability alone: closed-book knowledge has no cheap verifiable reward, so RLVR is the wrong instrument for it, not an instrument we chose not to use. **Reopened question:** with the thesis gone, closed-book knowledge needs an owner somewhere in the pipeline. The natural home is SFT/DPD (stages 2–5), where verifiability is not required, rather than a specialist RL branch. Decide explicitly; do not let it fall through the gap this retarget creates. |
| 9 | `specialist_writing` | **[CHANGE]** reward → rubric-based (RaR / pairwise adaptive rubrics) rather than a pointwise score, which has a discriminability ceiling and is hackable. |
| 10 | `specialist_agentic` | Keep, keep it "lightweight". **[CHANGE]** add turn-level credit assignment (turn-level advantages are already in the config; the survey's best-supported additions are hindsight/counterfactual credit — HCAPO, C3 — and information-gain credit). **[NEW]** entropy modulation for long horizons. |
| 11 | `opd_consolidation` | Keep the design — it matches the frontier. **[CHANGE] cap OPD response length at ≤8K**, not 131,072: teacher continuation advantage collapses from +0.37@1K to +0.02@16K and >10K responses produced overlap collapse and gradient instability. For the agentic teacher, distil **per turn** (each turn ≤8K) rather than one 131k trajectory. **[CHANGE]** top-k 32 → 16 and weight the *overlap* region (intersection carries 97–99% of mass; the union's extra tokens contribute little at real memory cost over a 65,536 vocab). **[NEW]** mandatory diagnostics: overlap ratio (must rise, target >90%), entropy gap, per-position entropy, gradient norm — abort if overlap is flat from initialization. Keep 1 epoch, LR 1e-6, temp 1.0, single-use rollouts. The specialists satisfy the "novel capability" condition precisely because they are RL-post-trained, which the paper shows matters far more than teacher size. |
| 12 | `pairwise_reward_model` → `preference_alignment` | Keep last (RL's Razor supports ending on on-policy stages). **[CHANGE]** rubric-grounded pairwise RM rather than a plain Bradley-Terry pointwise head. |
| 13 | `evaluation` → `publish_gate` | Keep. **[NEW]** add per-cap evaluation (caps 1–5) so the effort dial is a measured, published operating curve, and add ScaleRL sigmoid fits as a reported artifact. |

### 3.3 What to freeze during RL

Not covered by any literature I found; these are reasoned judgments, flagged as such:

- **Freeze the N-gram tables; keep their gates trainable.** RL touches a vanishing fraction of rows
  relative to 1T pretraining exposures, so sparse rows would receive high-variance updates from tiny
  batches; the tables are static local memory by design; and freezing removes sparse-gradient
  synchronization from the RL loop entirely. RL's Razor also argues for minimizing this distribution
  shift.
- **Keep the mHC controller trainable** (it is tiny and pass-conditioned), but monitor
  `mhc_stream_diversity` — the existing health gate is only an exact-collapse tripwire, and RL is
  exactly the regime where stream collapse would be plausible.
- **Keep the router trainable but rely on the aux-loss-free bias** as in pretraining; do not add a
  balancing aux loss during RL (R3 explicitly reports none is needed).

### 3.4 De-risking order

1. **Parity harness first.** Rollout logprobs vs recomputed training logprobs, target correlation
   ≥0.99 (MiniMax-M1's diagnostic). This single number tells you whether any RL stage can work. Run
   it with and without route replay and with and without continuation replay so you can attribute the
   gap.
2. **Depth calibration (M3)** on the proxy model — cheap, no RL, and it de-risks the halting
   behaviour that killed Ouro's RL.
3. **RLTT on the ~1.5B proxy** with fixed depth first (reproducing RLTT's setting exactly), then with
   adaptive depth + continuation replay. This is the novel step; do it small.
4. **One specialist end-to-end** (reasoning) on Praxis, with ScaleRL sigmoid fits from ~1.5k
   APU-hours to predict whether the full five-specialist campaign is affordable — and whether Logos
   RL is worth its allocation (ScaleRL's MoE result suggests it will be).
5. Then the full campaign.

---

## 3.5 (§6) The two closest analogs, and what they change

Added after reading Loopie and Nanbeige4.2 as primary sources. These are the most relevant references
in this document and they supersede parts of §1.5 and §3.

### 6.1 What Loopie and Nanbeige4.2 actually are

**Loopie** (*Loop the Loopies!*, arXiv 2607.16051, July 2026) — a **looped MoE** series, the only
published architecture that shares both of Metis's defining features:

| | Loopie-6B-A0.6B | Loopie-20B-A2B | Metis-1.6 Praxis |
|---|---|---|---|
| Stored layers | 18 | 27 | 12 |
| Recurrence | **R=2, fixed** | **R=2, fixed** | 1–5 adaptive, mean 2 |
| Routed experts / top-k | 128 / 8 | 128 / 8 | 128 / dynamic 1–8 |
| Expert hidden | 576 | 832 | 512 |
| Halting | **none** | **none** | per-token continuation router |

- **Loop pattern is "layer-loop", not "model-loop":** each layer is applied recurrently before moving
  to the next (1→1→2→2→3→3), not the whole stack (1→2→3→1→2→3). They cite execution locality and
  better empirical scaling; their ablation shows layer-loop substantially beats the same model with no
  recurrence at matched compute and stored params.
- **Why R=2:** they have a dedicated "Why Only Two Loop Steps?" section — marginal benefit of
  recurrence is largest at R=2, and N=4/N=8 diminish. Same shape as Ouro's curve and Metis's mean-2
  target, but they made it *fixed* rather than adaptive.
- **Token efficiency 7×:** Loopie-20B-A2B at 3.5T pretraining tokens matches or exceeds
  Nemotron-30B-A3B at 25T. Mechanism is mundane and worth copying: halving stored layers frees
  activation memory → double per-device microbatch → reinvest the measured gain into width
  (D 2048→2304) at matched optimizer-step time.
- **Results:** Loopie-6B-A0.6B AIME24 **80.42**, MMLU 82.70 — versus **Ouro-2.6B at 62.50** on AIME24.
  Loopie-20B-A2B: AIME24 92.09, gold-medal thresholds on IMO 2025 and IPhO 2025.
- They shipped **`vllm-loopie` and `megatron-loopie`** forks — i.e. the team that succeeded did build
  custom engine support for the looped model.

**Nanbeige4.2-3B** — a shipped Looped Transformer, 4B total / 3B non-embedding, 256K context. From
its own `config.json`: `num_loops: 2`, 22 layers, hidden 3072, 48Q/8KV × 128, vocab 166,144,
rope_theta 7e7, plus `loop_loss_weights: []` and `skip_loop_final_norm: false`. So: **fixed 2 loops,
no halting mechanism, RoPE not NoPE.** Base model trained on 28T tokens (up from 23T).

Its `modeling_nanbeige.py` contains, all config-gated, the features Nanbeige says are for 4.5:
- **LoopSplit** (`enable_double_loop_split`): unlooped edge layers plus a looped middle block,
  `first_unlooped_layers = (num_hidden_layers − loop_middle_layers) // 2`.
- **mHC** (`NanbeigeHyperConnectionModule`): `SinkhornKnopp.apply(h_res_logits, sinkhorn_iterations)`
  — doubly stochastic, same construction as Metis. With `mhc_diff_for_loop=True` it instantiates
  **separate HC modules per loop**; `mhc_double_stream_position_for_loop` can double stream count in
  the middle layers.
- **Depth attention** (`enable_depth_attention`): attends over cached K/V from *earlier layers*,
  stored every `depth_attention_stride` layers. **This is not Metis's mechanism** — Metis's depth
  memory attends over its own prior *passes* via typed anchors. Same name, different thing.
- **Concatenated n-gram embeddings**: polynomial rolling hash, orders 2..n, k tables per order,
  prime-sized vocabs, and `ngram_fused_mode` switching between **averaging and concatenation**.
  Injected by blending into the word embeddings — `x = (x + ngram·k(n−1)) / (1 + k(n−1))` — plus a
  sigmoid-gated layer fusion. Metis instead injects at two specific layers with pass-aware gates.

**Read:** Nanbeige is building substantially Metis's feature set on a larger budget, and their
*shipped* looped model uses fixed depth 2 with no halting.

### 6.2 Correction: post-train at fixed depth

There are now four independent data points and they all point the same way:

| Team | Architecture | Post-training depth | Outcome |
|---|---|---|---|
| Ouro | model-loop, learned early exit | attempted adaptive | **RLVR failed** — no significant gain over SFT; variable-depth rollouts were the blocker |
| RLTT | Ouro backbone | **fixed T_max=4**, adaptivity explicitly sacrificed | large gains over GRPO |
| Loopie | layer-loop MoE | **fixed R=2** | plain GSPO, no instabilities, frontier results |
| Nanbeige4.2 | looped, no halting | **fixed 2 loops** | shipped, beats Qwen3.5-9B |

**Nobody has successfully post-trained an adaptive-depth model.** The first draft of this document
recommended attempting it with a novel "continuation replay" mechanism. That inverts the risk posture
it should have. Revised recommendation:

**Correction to this correction.** An earlier version of §6.2 recommended freezing depth uniformly and
freezing the continuation controller. That conflates two different things, and the conflation is
wrong:

- **(a) Fixed uniform depth** — every token gets exactly 2 passes during RL. This is what Loopie,
  Nanbeige4.2 and RLTT did. Safe, but it removes the depth axis from RL entirely, so the allocation
  policy is only ever as good as pretraining plus calibration made it. For Metis this also means the
  central architectural claim goes unvalidated on the shipped model, which is a paper-level problem,
  not only a product one.
- **(b) Adaptive depth, sampled on-policy at rollout, replayed in the training forward, with
  `log P(τ)` in the sequence likelihood.** Depth stays fully dynamic and fully trainable; only the
  rollout-versus-update *mismatch* is removed. Gradients still flow into the continuation controller's
  logits — exactly as R3 replays the expert mask while leaving the router trainable.
- **(c) Naive adaptive depth with no replay** — what Ouro attempted, and it failed.

The four-team table shows nobody has *succeeded* at adaptive-depth post-training. The right response
is to remove the failure mechanism, not to abandon the capability. **Target (b).** Use (a) only as
the parity-harness control (where mismatch is zero by construction, so it validates the harness) and
as the fallback if (b) cannot reach the parity threshold. Order of work:

1. Build the parity harness; verify at fixed uniform depth (a) that rollout and recomputed logprobs
   correlate ≥0.99. This is the control, not the plan.
2. Turn on adaptive depth with continuation replay (b); re-measure parity. The gap between step 1 and
   step 2 is the quantity that decides whether this works.
3. Add RLTT per-pass credit as an A/B on top of (b).
4. Re-fit halting after RL with the §3.1 M3 depth-calibration stage regardless of path, since RL moves
   the hidden states the controller reads.

### 6.3 Correction: RLTT and R3 are conditional, not mandatory

Loopie post-trained a **128-expert, top-8, looped MoE** with plain **GSPO + asymmetric clipping +
DAPO dynamic sampling**, no per-loop credit assignment, no routing replay, no intermediate
supervision at loop boundaries — and reports no RL instability, while monitoring exactly the right
things (validation accuracy, mean response length, generation entropy, truncation rate).

So:
- **RLTT** (§1.5) measured its gains against GRPO on Ouro, an adaptive-exit model-loop architecture.
  At fixed depth with only the final pass emitting the answer, its motivating mismatch is much
  weaker. Demote to a proxy A/B against GSPO. My earlier "largest single expected gain" framing was
  over-claimed.
- **R3** remains the best-evidenced MoE-RL stabilizer, but its measurements come from a separate
  inference engine with multi-step off-policy updates. **Measure first**: instrument R3's own
  diagnostics (train-vs-rollout token-probability KL, per-layer expert-flip rate, extreme-token
  fraction) and adopt route replay only if the numbers justify it. Loopie is direct evidence that a
  looped 128-expert MoE can be RL-trained without it.
- **CISPO should not have been the primary recommendation.** It came from MiniMax-M1, a much larger
  and older model, and the closest analog used GSPO. Metis's config already specifies GSPO with
  asymmetric clipping — that choice is **vindicated by Loopie and needs no change**. Keep CISPO's
  clip-the-weight-don't-drop-the-sample idea as a fallback if entropy collapses.
- The FP32-LM-head recommendation stands on ScaleRL independently of MiniMax-M1.

Loopie's concrete RL settings, worth copying directly: sequence-level ratio with length
normalization `s_i(θ) = exp[(1/|o_i|) Σ log(π_θ/π_old)]`; asymmetric clip `ε_high > ε_low` on the
whole response; retain only prompt groups containing **both** successes and failures; response length
32K in stage 1, raised to 64K once rollout truncation exceeds 10%; math first, then code; stop before
sustained validation degradation.

### 6.4 Correction: token efficiency comes from preference alignment, not length penalties

Neither of the efficient small models uses a length penalty:

- **Nanbeige4.1**: point-wise RL with a general reward model cut LiveCodeBench-v6 overlong
  truncations from **5.27% → 0.38%**, then pair-wise RL added **+7.2 Arena-Hard-V2** (66.6 → 73.8) and
  +7.4 Multi-Challenge. The report attributes verbosity control to preference alignment, with no
  explicit output-length penalty. Their difficulty filter is **k ∈ [1,5] successes out of 8**
  (12.5–62.5%), not the 10–90% band from Nanbeige4-3B that Metis's config currently pins.
- **MiniCPM5-1B**: RL+OPD raised the average **16 points** and cut the overlong-response rate by
  **29 percentage points**. Length control came from a two-stage length *schedule* on math RL
  (DAPO-Math-17k, JustRL-style minimalist recipe) plus OPD — not a reward penalty.
- **RLTT** independently found responses got shorter with *no* brevity incentive.

Revised recommendation: **demote the DAST λ-length machinery.** Keep it as an optional knob, but the
primary mechanism should be (a) a two-stage response-length schedule, (b) ScaleRL's forced
interruption, and (c) the OPD consolidation stage, which is independently documented to cut overlong
responses. This is a simplification of the current config, not an addition.

Also note Nanbeige4.1's **code RL stage 2** is worth stealing outright: a gated time-complexity
reward, `R = R_format + R_correctness` when PassRate < 1, and `R_format + R_correctness + R_time`
only when PassRate = 1. Efficiency is rewarded strictly after correctness is saturated, which is the
clean way to add a second objective without letting it trade against the first.

### 6.5 The real gap: instruction training at pretraining scale

Both efficient small models do something Metis's plan does not:

- **Loopie's "Supervised Pre-Training" (SPT)**, presented as a novel contribution: **2T tokens** of
  instruction data (math, code, reasoning, tool-use) with an SFT-style loss mask (targets only) but
  at pretraining scale — **global batch ≥1024, sequence length ≥128K**, versus conventional SFT batch
  sizes of 32–128. Claimed effects: no catastrophic forgetting, no loss cliff at epoch boundaries, and
  simultaneous improvement of pretraining *and* reasoning metrics.
- **MiniCPM5-1B**: **200B tokens of deep-thinking SFT + 200B tokens of hybrid-thinking SFT** for a
  1.08B-parameter model (679M non-embedding, 24 layers, 16Q/2KV, 131K context).

Metis's `cold_start_sft` specifies ~30M source instructions. Even at a generous 2k tokens each that
is ~60B tokens, and it is run at SFT batch scale.

Recommendation: **restructure the Phase-C-to-SFT boundary.** Metis's 50B premium non-generated
cooldown and its cold-start SFT are currently separate stages with different batch regimes. Merge the
intent: run the instruction/reasoning corpus as a large-batch, long-sequence, target-masked stage
continuous with pretraining, rather than as a small-batch SFT. This costs no new data — the SFT mix
is already being collected — and it is the one change in this document that both efficient small
models independently support. It also fits RL's Razor: fewer, larger, gentler supervised updates
mean less forgetting before RL begins.

### 6.6 Architecture cross-checks (outside post-training scope, flagged because they are locked)

Not recommendations — but four Metis decisions now have contrary evidence from the closest analogs:

| Metis decision | Contrary evidence |
|---|---|
| Full-stack (model) loop, "Ouro-validated" | Loopie uses **layer-loop** and argues execution locality + better empirical scaling; their ablation is layer-loop vs no-loop, so this is suggestive rather than a head-to-head refutation |
| Middle-block looping rejected | Nanbeige's **LoopSplit** and MIT's **Hyperloop Transformers** (2604.21254) both loop only a middle block with unlooped edges; Hyperloop reports beating depth-matched baselines at ~half the parameters |
| Depth 1–5 adaptive, mean 2 | Loopie and Nanbeige both ship **fixed R=2**; Loopie's ablation says marginal benefit is largest at R=2 and diminishes at 4 and 8 |
| One mHC controller conditioned on a pass embedding | Nanbeige instantiates **separate HC modules per loop** (`mhc_diff_for_loop`) and can double stream count mid-stack |

Also relevant to the mHC lineage: **JPmHC — Dynamical Isometry via Orthogonal Hyper-Connections**
(2602.18308) and the original **Hyper-Connections** (2409.19606).

## 4. Honest uncertainty

- **Superseded by §6.2:** the first draft's central mechanism (GSPO + CISPO clipping + route replay +
  continuation replay + RLTT credit under adaptive depth) is a synthesis nobody has run. It is now the
  research lane, not the plan. The plan is fixed-depth RL, which is what four out of four teams did.
- **Loopie's post-training section is thin on the things Metis most needs to know.** It does not name
  its RL rollout engine, does not describe how recurrent state is handled during rollout sampling, and
  does not discuss MoE routing stability under RL at all. Absence of reported instability is not the
  same as absence of instability, and their fixed R=2 removes the depth half of the problem entirely.
- **The Nanbeige4.2 technical report PDF would not parse** (FlateDecode streams). Everything above
  about 4.2 comes from its model cards, its `config.json`, and its `modeling_nanbeige.py` — primary
  sources, but not the report's prose. The RL detail there is one sentence: outcome and process rewards
  combined "to improve training stability for the compact model." Worth retrieving properly.
- **`modeling_nanbeige.py` features are config-gated and mostly dormant in the 4.2 checkpoint.**
  LoopSplit, depth attention and the n-gram path are present but inactive unless their flags are set;
  4.2 ships with `num_loops: 2` and an empty `loop_loss_weights`. Do not read the released weights as
  evidence that those components are validated at scale.
- **PR² over R3 is conditional.** PR²'s large gains are in off-policy (off-2/4/8) regimes, and on the
  small OLMoE-1B-7B its advantage shrank to ~0.5–1.4%. With a single-engine on-policy rollout, plain
  R3 is likely sufficient; adopt PR² only if the pipeline goes asynchronous.
- **Depth inflation is the obvious reward-hacking failure** once depth becomes an action: the policy
  can buy accuracy with compute. The depth-budget loss, the S-GRPO decay term and the
  `recursion_depth_distribution` eval gate are the three defences; none is proven at this scale.
- **Abstain-R1 has not been validated in tool-augmented settings** — the paper says so. Metis would be
  extending it to retrieval, which is the natural direction but is not evidence.
- **MiniCPM5's exact OPD numbers could not be verified from primary sources.** The +16 average / −29pp
  truncation figures come from search-result summaries; the DeepWiki page I fetched does not contain
  them. Treat as indicative, not exact.
- RSPO, "Mock Worlds Real Skills", and the learned-stochastic-stopping paper were read at abstract or
  partial-PDF level only.

---

## 5. Primary sources

**On-policy distillation & consolidation**
- Rethinking On-Policy Distillation of LLMs: Phenomenology, Mechanism, and Recipe — arXiv 2604.13016
- Rethinking On-Policy Self-Distillation for Thinking Models — arXiv 2607.05184
- MOPD: Multi-Teacher On-Policy Distillation for Capability Integration — arXiv 2606.30406
- A Survey of On-Policy Distillation for LLMs — arXiv 2604.00626
- On-Policy Distillation — Thinking Machines Lab

**RL algorithms & scaling**
- The Art of Scaling Reinforcement Learning Compute for LLMs (ScaleRL) — arXiv 2510.13786
- MiniMax-M1 (CISPO) — arXiv 2506.13585
- Group Sequence Policy Optimization (GSPO) — arXiv 2507.18071
- DAPO — arXiv 2503.14476
- Clip-Low Increases Entropy and Clip-High Decreases Entropy — arXiv 2509.26114
- RL's Razor: Why Online RL Forgets Less — arXiv 2509.04259
- ExGRPO: Learning to Reason from Experience — arXiv 2510.02245
- OctoThinker: Mid-training Incentivizes RL Scaling — arXiv 2506.20512

**MoE-specific RL**
- Stabilizing MoE RL by Aligning Training and Inference Routers (R3) — arXiv 2510.11370
- PR²: Predictive Routing Replay for MoE-Based LLM RL — arXiv 2606.00395
- Towards Stable and Effective RL for Mixture-of-Experts (RSPO) — arXiv 2510.23027

**Closest analogs (read as primary sources; see §6)**
- Loop the Loopies! (Loopie 20B-A2B / 6B-A0.6B looped MoE) — arXiv 2607.16051
- Nanbeige4.2-3B and Nanbeige4.2-3B-Base model cards, `config.json`, `modeling_nanbeige.py` — HF
- MiniCPM5-1B model card — HF `openbmb/MiniCPM5-1B`
- Nanbeige4.1-3B: A Small General Model that Reasons, Aligns, and Acts — arXiv 2602.13367
- Hyperloop Transformers (MIT) — arXiv 2604.21254
- JPmHC: Dynamical Isometry via Orthogonal Hyper-Connections — arXiv 2602.18308
- Hyper-Connections — arXiv 2409.19606
- DeepLoop: Depth Scaling for Looped Transformers — arXiv 2607.13491

**Looped / recursive / latent-depth**
- Scaling Latent Reasoning via Looped Language Models (Ouro) — arXiv 2510.25741
- Prioritize the Process, Not Just the Outcome (RLTT) — arXiv 2602.10520
- SLPO: Scaling Latent Reasoning via a Surrogate Policy — arXiv 2607.19691
- LSRL: Process-Supervised GRPO on Latent Recurrent States — Findings EMNLP 2025
- Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping — arXiv 2606.29983
- LoopFormer: Elastic-Depth Looped Transformers via Shortcut Modulation — arXiv 2602.11451
- Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy RL — arXiv 2606.13106
- Universal Transformers Need Memory: Depth-State Trade-offs — arXiv 2604.21999

**Agentic RL & credit assignment**
- From Reasoning to Agentic: Credit Assignment in RL for LLMs — arXiv 2604.09459
- Nanbeige4.1-3B — arXiv 2602.13367
- Nanbeige4-3B Technical Report — arXiv 2512.06266
- AEM: Adaptive Entropy Modulation for Multi-Turn Agentic RL — arXiv 2605.00425
- CM2: RL with Checklist Rewards for Multi-Turn Agentic Tool Use — arXiv 2602.12268

**Grounding, abstention, rubrics, verifiers**
- Abstain-R1: Calibrated Abstention via Verifiable RL — arXiv 2604.17073
- TruthRL: Incentivizing Truthful LLMs via RL — arXiv 2509.25760
- Rubrics as Rewards — arXiv 2507.17746
- Open Rubric System: Pairwise Adaptive Rubric — arXiv 2602.14069
- LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking — arXiv 2604.15149
- Before the Model Learns the Bug: Fuzzing RLVR Verifiers — arXiv 2606.01066
- Faithfulness-Aware Step-Level RL for Small Reasoning Models — arXiv 2602.05897

**Precision & infrastructure**
- FP8-RL: A Practical and Stable Low-Precision Stack for LLM RL — arXiv 2601.18150
- Jet-RL: On-Policy FP8 RL with Unified Training and Rollout Precision Flow — arXiv 2601.14243
- verl on ROCm compatibility; slime on AMD MI300X; LMSYS ROCm RL post-training (2026-03)
- Hybrid Models as First-Class Citizens in vLLM; Disaggregated Serving for Hybrid SSM Models (2026-04)
