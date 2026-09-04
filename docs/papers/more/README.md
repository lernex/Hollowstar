# MoRE architecture paper

Working draft of the MoRE paper. Separate from `docs/papers/metis_technical_report/`,
which documents the Metis model line as a whole; this one makes an architecture
claim and has to survive adversarial review.

## Files

- `main.tex` — paper draft. Prose is real where the framing is decided; `\TODO{}`
  marks what must be written or measured, `\CITE{}` marks unresolved citations.
- `references.bib` — references, each flagged `[VERIFIED]` (arXiv id confirmed by
  search on 2026-07-25) or `[CHECK]` (from memory or a secondary source — confirm
  before submission).
- `ablation_campaign.md` — model sizing, token budget, FLOP math, APU allocation,
  parallelism strategy, and the code prerequisites.

## Build

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Naming — the acronym is not free

A search on 2026-07-25 found MoRE already in use at least three times:

| Paper | Meaning | Overlap |
|---|---|---|
| arXiv 2305.14628, *Getting MoRE out of Mixture of Language Model Reasoning Experts* | Mixture of **Re**asoning **E**xperts | Prompt ensembling for QA. Not an architecture. |
| arXiv 2505.22694, *MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning* | Mixture of Low-**R**ank **E**xperts | PEFT / multi-task. Not pretraining. |
| arXiv 2504.06426, *S'MoRE: Structural Mixture of Residual Experts* | near-collision | PEFT. |

None is a pretraining architecture and none has strong claim to the name in this
subfield, so the name is contestable rather than taken. But "MoRE is unused" is
not accurate and should not appear in the paper or in any outreach. Options:

1. **Keep MoRE**, define it in the abstract's first sentence, and cite the
   collisions nowhere (they are not related work). Acronym reuse across subfields
   is normal. Accept that search engines will mix the three.
2. **Disambiguate in the expansion** — keep the letters, choose an expansion that
   is clearly ours and state it once. Low cost, removes most of the ambiguity.
3. Rename. Only worth it if a *pretraining architecture* paper claims MoRE first.

Recommendation: option 1 or 2, decided before the first public artifact (arXiv,
model card, or repo rename) — whichever comes first fixes the name in practice.

## Competitive position — this got urgent in the last two months

Recursion combined with sparse experts went from unexplored to active:

| Paper | What it is | Why it matters |
|---|---|---|
| arXiv 2606.04438, **LoopMoE** | Block-recurrent MoE at 3B/9B. `IterAdaLN` conditions on iteration index + token state. Compared to vanilla MoE at identical total params and per-token FLOPs. | The nearest neighbour. It is our ladder row 5 (Fixed LoopMoE), published. Their `IterAdaLN` is convergent with our pass-conditioned mHC controller. |
| arXiv 2607.16051, **Loop the Loopies!** | Loopie series, 20B/A2B and 6B/A0.6B looped MoE + reasoning post-training, claims frontier-level reasoning. | Direct competitor at Logos scale, published this month. |
| arXiv 2606.16825, **Tying the Loop** | Compares per-iteration adaptation: depth-wise LoRA, low-rank modulation, **MeSH** (explicit memory buffer with step-wise routers). | MeSH is the nearest neighbour to MoRE-RM. Read it before claiming the memory. |

What remains unclaimed, and therefore what the paper must lead with:

1. **Depth is per-token adaptive** — these models fix the loop count.
2. **Width is per-token adaptive inside a recursive model** — adaptive-`k` exists
   in flat MoE (AdaMoE, DynaMoE, ProbMoE, "Harder Tasks Need More Experts") but
   not, so far, inside a loop.
3. **The two axes share one budget**, so the model trades depth against width per
   token rather than satisfying two independent schedules.
4. **Route-typed recurrent memory** (MoRE-RM), pending the MeSH comparison.

Adaptive depth alone is Mixture-of-Recursions (2507.10524). Adaptive `k` alone is
four papers deep. **Neither is the claim.** The claim is joint allocation, and the
paper is only as strong as the ablations that isolate it — which is why rows 6, 12,
and 13 in the ladder are not optional.

## Title

Chosen: **Three Dials, One Budget: Per-Token Compute Allocation with MoRE**

It states the contribution, implicitly indicts the one-dial prior work, names the
acronym, and has a rhythm you can say out loud. Alternates are listed in the
comment block at the top of `main.tex` and are cheap to swap until the first
public artifact fixes the name:

- *MoRE Is More: Joint Per-Token Routing of Depth, Width, and Pathway* — the pun is
  right there, teaches the acronym, slightly more casual than the venue may like
- *How Much Compute Does a Token Deserve?* — question titles travel well on social
  media, weaker in a proceedings index
- *Goldilocks Routing: Never Too Much, Never Too Little, Per Token* — the friendliest
  framing, and the one furthest from what the abstract actually proves

## Tone

The draft has a deliberate voice and `main.tex` carries the policy in a comment
so it stays consistent. Humour lives in the abstract (one dry line), the
introduction, figure captions, and limitations. Methods, results, and related
work stay sober — a joke in a results table reads as a hedge.

One hard rule: **never joke at another paper's expense.** Joke at the problem, at
uniform compute, and at ourselves. The authors of LoopMoE, MoR, and MeSH are
plausibly the reviewers, and self-deprecating honesty in a limitations section
buys more credibility than a clever line at someone else's cost ever will.

## MeSH vs MoRE-RM — resolved

[arXiv 2510.07739](https://arxiv.org/abs/2510.07739) turned out to be a different
animal than its one-line description suggests. MeSH keeps `B` state slots
`m_b ∈ R^(L×D)` (slot 0 initialized to embeddings, `B = N_loop + 3`), with Write
and Read routers that have **unique parameters per iteration**, condition on
`h^(t)` alone, softmax over slots, additive write, weighted-sum read.

The consequence is that **MeSH is convergent with our mHC path, not with
MoRE-RM.** Parallel per-position state slots mixed by iteration-conditioned
learned weights is a multi-stream residual with pass-dependent mixing. Meanwhile
MeSH's buffer stores no route information (no coalition, no `k`, no entropy, no
halting confidence — none of which exist in a fixed-loop model), its read feeds
only the representation path, and its recursion count is fixed with no halting.

Written up as `main.tex` §4.2. It also generated a new falsifier now in the
claims table: strip the route typing from MoRE-RM entries and re-run. If typed
and untyped perform identically, we built an expensive MeSH and should say so.

## Status

- [x] Naming boundary (contribution vs integration) — criterion fixed in `main.tex` §5
- [x] Ablation ladder — 11 rows + 2 random-policy controls
- [x] Compute plan — sizing, 50B budget, measured full-machine two-batch schedule, DP strategy
- [x] MeSH read; MoRE-RM distinction written (`main.tex` §4.2)
- [x] Title chosen
- [x] `ablation` model family, dense-FFN path, pathway-frozen mode, random-policy controls
- [x] Grouped expert GEMM (numerically identical to the loop; test asserts it)
- [x] `metis_ablation` package: specs, strided sampler, DP trainer, routing analyzer, campaign planner
- [x] Slurm launchers for the 13-row Wave 1, split into measured makespan-first 1a/1b batches (`slurm/ablation/`)
- [ ] Resolve `[CHECK]` citations in `references.bib`
- [x] Wave 2 (scaling ladder, 8 rows) and wave 3 (paired seeds, 5 rows)
- [x] Archetype learning-rate sweep generator
- [x] Checkpoint resume with atomic writes and a schedule-change guard
- [x] FP8-vs-BF16 parity check per run
- [ ] Portage canary: measure MFU, confirm FP8 parity, race grouped vs loop
- [ ] Run the campaign; fill every `\TODO{}`

## Code

| Path | What it is |
|---|---|
| `src/metis_ablation/specs.py` | The 13 rows as executable specs. Enforces the identical-global-batch invariant and solves the dense controls against the audited FLOP/parameter model. |
| `src/metis_ablation/sampler.py` | Phase-proportional strided sampler over the 1T release. Guarantees every row reads the same token window per step regardless of world size. |
| `src/metis_ablation/train.py` | Data-parallel trainer with replicated experts. Separate from `metis_training.train` so the 1T runs are not destabilized. |
| `src/metis_ablation/analysis.py` | Hook-based routing analyzer: depth/width distributions, depth–width correlation, expert transition matrices, halt calibration. |
| `src/metis_ablation/campaign.py` | `plan` prints the wave and cost model; `slurm` emits the launchers. |
| `tests/test_more_ablations.py` | Tests for production immutability, iso-FLOP model work, measured launch geometry, safe execution batches, and end-to-end trainer runs for all 26 rows. |
