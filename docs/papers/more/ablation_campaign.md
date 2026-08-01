# MoRE ablation campaign — sizing, compute, allocation, and execution

Status: planning draft, 2026-07-25. Supersedes the "Fast ablation allocation"
sizing in `METIS_1.6_PLAN.md:1038` for the post-PT window. Every number here is a
planning estimate; replace with measured tokens/second from the canary before
committing a calendar date.

## 0. Window and available hardware

The campaign runs **after Metis-1.6 pretraining completes**, during Praxis/Logos
continued pretraining (context extension) and post-training. At that point the
family releases most of Portage:

| Consumer | APUs | Notes |
|---|---:|---|
| Metis-1.6 Logos CPT / post-training | 64 | |
| Metis-1.6 Praxis CPT / post-training | 64 | |
| **MoRE ablation campaign** | **384** | 96 nodes at 4 APUs/node |

MI300A peak: **1961 TFLOP/s FP8 dense** per APU → 384 APUs = **753 PFLOP/s peak**.
Planning MFU band: **5% / 10% / 15%** (conservative / planning / good). The 10%
column is the one to plan against; treat 15% as reachable only after the grouped
expert GEMM in §5 lands.

## 1. Model size

**Half-Praxis, as proposed.** Confirmed as the right size — it is the plan's
existing proxy manifest and it fits on a single 128 GB APU with FP32 masters and
optimizer state, which is what makes §4's parallelism strategy possible.

| | Praxis (production) | **Proxy S (ablation primary)** |
|---|---:|---:|
| `d_model` | 2048 | **1792** |
| Physical layers | 12 (10 Mamba + 2 attn) | **10 (8 Mamba + 2 attn)** |
| `latent_dim` | 1024 | **896** |
| `expert_intermediate_dim` | 512 | **448** |
| Routed experts | 128 | **96** |
| `k` range / target | 1–8 / 4 | **1–8 / 4** |
| `max_passes` / target | 5 / 2 | **5 / 2** |
| Stored core | ~2.9B | **~1.5B** |
| Core active / pass @ k=4 | ~0.47B | **~0.29B** |

Derivation of the 0.29B (matches the plan's figure, useful for the FLOP math):

- Mamba-2 mixer ≈ 23.1M/layer × 8 = 185M
- Attention ≈ 8.0M/layer × 2 = 16M
- Latent down+up = 3.21M/layer × 10 = 32M
- Experts (k+1) × 1.204M/layer × 10 → k=4: 60M, k=8: 108M
- **Core active/pass @ k=4 = 293M**, @ k=8 = 342M
- Tied embedding / LM head = 117M, applied **once per token**, not per pass

## 2. FLOPs per token

**Use `python -m metis_ablation.campaign plan`, not a hand formula.** The
authority is `metis_training.metrics.estimate_hardware_flops`, which the trainer
already reports against every telemetry step. It counts three things a bare
`6 x active_parameters` estimate misses, all of which the accelerators really
execute:

- the **pass-level recompute replay** (`activation_recompute_policy: pass`
  performs two forwards per backward, so executed work is 8F/6F of model FLOPs);
- the **depth-memory projections**, which run once per layer per pass and once
  per mHC stream on top of that;
- **parameter-free work** -- attention's quadratic score/value products and the
  Mamba-2 chunked-scan einsums.

For MoRE-Core at the proxy geometry that is **7.30 GFLOP/token executed**, not
the 4.22 a naive `6N` estimate gives and not the 3.5 quoted core-only in
`METIS_1.6_PLAN.md:1052`. Scheduling against the naive number would under-budget
the campaign by 1.7x. The per-row figures are in the master table in section 4.

## 3. Token budget — 50B, not 10B

**Recommendation: 50B, and it is not a close call.**

| Budget | Campaign executed FLOPs (13 rows) | Wall clock @5% | @10% | @15% |
|---|---:|---:|---:|---:|
| 10B | 945 EFLOP | 8.0 h | 4.0 h | 2.7 h |
| **50B** | **4,726 EFLOP** | **39.8 h** | **19.9 h** | **13.3 h** |
| 100B | 9,453 EFLOP | 80 h | 40 h | 27 h |

Wall clock is the longest single row with all thirteen running concurrently, not
the serialized total.

Three reasons 10B is the wrong choice:

1. **Fairness of the dense control.** At 10B, the 1.5B parameter-matched dense
   model sees 6.7 tokens/parameter — far under Chinchilla-optimal (~20). It is
   badly undertrained, so beating it proves little, and the bias runs *in
   MoRE's favour*, which is exactly the direction a reviewer attacks. At 50B it
   sees 33 tokens/parameter, past Chinchilla. MoRE-Core sees ~170
   tokens/active-parameter, a normal modern overtrained regime.
2. **Routing policies need time to stabilize.** The depth and width policies
   train through Gumbel straight-through under two quadratic budget penalties.
   Early training measures the warmup transient, not the converged policy. A
   conditional-compute advantage that appears at 10B and reverses at 50B would
   be a very bad thing to discover after publication.
3. **Cost is not the constraint.** 50B costs roughly half a day of the pool.

Sample the 50B **proportionally from the 1T mixture including its phase
structure** (`METIS_1.6_PLAN.md:204`), not the first contiguous 50B. Use the
**identical token stream in identical order for every row** — that removes data
order as a confound for free, and it is worth a sentence in the paper.

Keep headroom to extend rows 2, 10, and 11 to 100B if their curves have not
separated.

## 4. Allocation — one concurrent wave, APUs proportional to FLOPs

Allocating APUs in proportion to each model's total FLOPs makes every model
finish at the same time, which keeps the wave operationally simple and gives
every row identical fabric conditions (a comparability plus, since all 13 share
the machine).

384 APUs / 57.3 GFLOP-token = **6.7 APUs per GFLOP/token**.

### Master table — wave 1, all 13 runs, 50B tokens each

| # | Model | Isolates | Stored / Active per pass | GFLOP/tok | APUs | Nodes | Hrs @10% |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | Dense, FLOP-matched | dense reference at MoRE's executed compute | 1.12B / 0.70B | 7.29 | 28 | 7 | 18.4 |
| 2 | Dense, parameter-matched | dense reference at MoRE's stored parameters | 1.82B / 1.41B | 12.96 | 56 | 14 | 16.4 |
| 3 | MoE k=4 | sparse routing without recursion | 1.83B / 0.30B | 4.10 | 16 | 4 | 18.2 |
| 4 | MoE k=8 | wider single-pass MoE reference | 1.83B / 0.30B | 4.49 | 16 | 4 | 19.9 |
| 5 | Fixed LoopMoE | recursion at fixed depth and fixed k | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| 6 | Loop, pathway frozen | PATHWAY: identical to row 5 except experts are chosen once | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| 7 | MoR + dense FFN | adaptive depth without sparse experts | 0.72B / 0.30B | 7.28 | 28 | 7 | 18.4 |
| 8 | MoR + fixed-k MoE | DEPTH: adaptive depth against row 5's fixed depth | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| 9 | Fixed depth, adaptive k | WIDTH: adaptive k against row 5's fixed k | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| 10 | MoRE-Core | all three axes together | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| 11 | MoRE-RM | route-typed recurrent depth memory | 1.83B / 0.30B | 7.30 | 32 | 8 | 16.2 |
| 12 | Random-k control | is the LEARNED width policy doing anything? | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| 13 | Random-depth control | is the LEARNED depth policy doing anything? | 1.83B / 0.30B | 7.30 | 28 | 7 | 18.5 |
| | **Total** | | | **94.5** | **372** | **93** | **19.9** |
| | Spare (eval, canaries, restarts) | | | | 12 | 3 | |

Regenerate this table with `python -m metis_ablation.campaign plan`; it is
computed from `metis_training.metrics.estimate_hardware_flops`, the same
accounting the trainer reports against every telemetry step, rather than from a
parallel hand-derivation that can silently drift.

Every row is the **Proxy S geometry** from §1 (half-Praxis: 10 layers, `d_model`
1792, latent 896, 96+1 experts, `R_max`=5) except where the architecture forbids
it: rows 1 and 7 are dense, so stored equals active and a dense-FFN recursive
model simply cannot hold 1.5B stored at 0.29B active. Report both numbers; the
asymmetry is what the sparse axis buys.

**One wave, 13–40 h wall clock depending on MFU; plan against ~20 h.**

### Wave 2 — scaling ladder

One size is a data point; three sizes are a trend. Four archetypes at two
smaller geometries turn "MoRE beats its baselines at 1.8B" into a slope, with
Praxis and Logos as the fourth and fifth points on the same curve.

| # | Model | Isolates | Stored / Active | GF/tok | APUs | Nodes | Hrs @10% |
|---:|---|---|---|---:|---:|---:|---:|
| 20 | Dense, parameter-matched (XS) | scaling point: dense reference at MoRE's stored parameters | 0.59B / 0.42B | 4.48 | 64 | 16 | 5.0 |
| 21 | MoE k=4 (XS) | scaling point: sparse routing without recursion | 0.59B / 0.13B | 2.13 | 32 | 8 | 4.7 |
| 22 | MoRE-Core (XS) | scaling point: all three axes together | 0.59B / 0.13B | 3.61 | 56 | 14 | 4.6 |
| 23 | MoRE-RM (XS) | scaling point: route-typed recurrent depth memory | 0.59B / 0.13B | 3.61 | 56 | 14 | 4.6 |
| 24 | Dense, parameter-matched (XXS) | scaling point: dense reference at MoRE's stored parameters | 0.21B / 0.13B | 1.73 | 28 | 7 | 4.4 |
| 25 | MoE k=4 (XXS) | scaling point: sparse routing without recursion | 0.21B / 0.05B | 1.10 | 16 | 4 | 4.8 |
| 26 | MoRE-Core (XXS) | scaling point: all three axes together | 0.21B / 0.05B | 1.74 | 28 | 7 | 4.4 |
| 27 | MoRE-RM (XXS) | scaling point: route-typed recurrent depth memory | 0.21B / 0.05B | 1.74 | 28 | 7 | 4.4 |
| | **Total** | | | | **308** | **77** | **5.0** |

The geometries scale `d_model`, latent width, layer count, and expert count
together so the model's aspect ratio is roughly constant — scaling one dimension
alone would confound size with shape. **The N-gram table scales too**: held at
0.30B it would be over half of the smallest model, and the "scaling" curve would
mostly be measuring a constant lookup table. Slot counts are chosen so the table
stays a roughly constant fraction of routed-expert capacity
(`test_scaling_ladder_keeps_the_ngram_table_proportional`).

Stored parameters across the ladder: **0.21B / 0.59B / 1.83B**, with the dense
control re-solved to match at each point.

### Wave 3 — paired seeds

A second seed is insurance against one lucky headline result, not a different
model design. Only the three rows the abstract quotes need it. Same data order,
same schedule, different initialization — the launchers pass
`--seed 27182818` automatically, and
`test_seed_wave_mirrors_its_parents_and_changes_only_the_seed` asserts that
nothing else moved.

| # | Model | Isolates | Stored / Active | GF/tok | APUs | Nodes | Hrs @10% |
|---:|---|---|---|---:|---:|---:|---:|
| 40 | Dense, parameter-matched (seed 2) | paired repeat of dense-param-matched | 1.82B / 1.41B | 12.96 | 112 | 28 | 8.2 |
| 41 | MoRE-Core (seed 2) | paired repeat of more-core | 1.83B / 0.30B | 7.30 | 112 | 28 | 4.6 |
| 42 | MoRE-RM (seed 2) | paired repeat of more-rm | 1.83B / 0.30B | 7.30 | 112 | 28 | 4.6 |
| | **Total** | | | | **336** | **84** | **8.2** |

### Whole campaign

| Wave | Rows | Executed FLOPs | Wall clock @10% MFU |
|---|---:|---:|---:|
| 1 — the ladder | 13 | 4,726 EFLOP | 19.9 h |
| 2 — scaling | 8 | 1,006 EFLOP | 5.0 h |
| 3 — seeds | 3 | 1,378 EFLOP | 8.2 h |
| **Total** | **24** | **7,110 EFLOP** | **~33 h sequential** |

Waves run one after another, so the 384-APU constraint applies per wave, not to
the sum. Add the archetype learning-rate sweep (12 short runs at 1B tokens,
about an hour) ahead of wave 1.

## 5. Parallelism — pure data parallelism, replicated experts, no EP

**Recommendation: DP only. No expert parallelism, no tensor parallelism, no
pipeline parallelism.**

Memory per APU for the 1.5B proxy: BF16 params 3 GB + FP32 masters 6 GB + FP32
Adam moments 12 GB ≈ **21 GB** of 128 GB, before activations, which are bounded
by pass-level recompute (`activation_recompute_policy: pass`). All 96 routed
experts replicated on every rank costs 116M params — trivial.

Consequences, all good:

- **Zero all-to-all.** Routing becomes a local gather/scatter. No dispatch/combine
  collectives, no dropless overflow path, no expert-load stragglers.
- **No systems confound in the science campaign.** Every row's throughput depends
  only on its own arithmetic. This matters: an EP campaign would make row-to-row
  wall-clock differences partly an artifact of routing skew, and wall-clock is
  one of the reported axes.
- The plan's canary race (`METIS_1.6_PLAN.md:1041`) should still be run, but the
  expected winner is replicated-experts + DP by a wide margin.

### Gradient synchronization is the real bottleneck, not compute

Ring all-reduce moves ~2× model size per rank per step regardless of DP width:
**~6 GB per rank per step** for a 1.5B model in BF16. At 28 APUs the step compute
is only ~0.77 s (global batch 1M tokens), so an unhidden all-reduce would
dominate.

Mitigations, in order of preference:

1. **Bucketed all-reduce overlapped with the backward pass** (standard DDP). Non-optional.
2. **Gradient accumulation ×2 → global batch 2M tokens, ~25k steps, ~1.54 s/step.**
   Comfortably hides the collective. Use the *same* global batch for all 13 rows —
   comparability matters more than per-row optimality here.
3. ZeRO-1 optimizer sharding if memory headroom is wanted for larger micro-batches.
   Not needed for capacity; it does not reduce collective volume.
4. FP8 gradient reduction halves the wire cost but adds a numerical risk the
   science campaign does not need. Skip unless MFU measurement demands it.

### Grouped expert GEMM — landed

`AdaptiveDroplessMoE._execute_local` issued one `torch.nonzero` per expert,
which forces a device-to-host synchronization each time. Under production expert
parallelism a rank owns 1–6 experts and the cost is invisible; under
replicated-expert DP a rank owns all 96, giving thousands of stalls per forward.

`expert_execution: grouped` sorts assignments once and derives segment
boundaries from a single `bincount`, cutting synchronizations from 96 per layer
to one. Numerics are unchanged — `test_grouped_expert_execution_matches_the_loop_exactly`
asserts bitwise-close losses between the two paths. Praxis and Logos keep
`expert_execution: loop`, so production is byte-identical to before.

## 6. Learning rate and hyperparameters

One learning rate across architectures of different active widths is not fair;
tuning all thirteen is not worth the compute. The compromise is a short sweep
over the four **archetypes** — dense, single-pass sparse, fixed loop, adaptive
MoRE — at 1B tokens each, three rates apiece, with every row inheriting its
archetype's winner:

```bash
python -m metis_ablation.campaign sweep --destination slurm/ablation --output-root "$METIS_SCRATCH/more-ablations" --release-root "$METIS_RELEASE_ROOT"
```

Twelve runs, roughly an hour. Report the sweep and the assignment in the paper:
a reviewer who sees the compromise stated is satisfied, one who suspects it is
not.

Everything else is identical across all rows and enforced in code — global batch
(`AblationSpec.__post_init__`), schedule shape
(`test_schedule_is_identical_across_rows`), warmup, sequence length, data order,
and the seed within a wave.

## 7. Running it

```bash
python -m metis_ablation.campaign plan --wave 1
```

```bash
python -m metis_ablation.campaign slurm --wave 1 --destination slurm/ablation --output-root "$METIS_SCRATCH/more-ablations" --release-root "$METIS_RELEASE_ROOT"
```

```bash
bash slurm/ablation/wave1/launch-wave.sh
```

Repeat with `--wave 2` and `--wave 3`. The committed launchers reference
`$METIS_REPO`, `$METIS_SCRATCH`, and `$METIS_RELEASE_ROOT`; regenerate them on
Portage rather than editing them by hand.

Single-row smoke test, no release needed:

```bash
python -m metis_ablation.train --row more-core --output /tmp/more --synthetic --max-steps 4 --device cpu
```

### Restarts

Every row resumes from its newest checkpoint by default, restoring model,
optimizer, and RNG state, so a preempted job continues rather than restarting.
Two guards make that safe rather than merely convenient: checkpoint writes are
staged and renamed, so a job killed mid-write leaves the previous checkpoint
intact; and a resume whose step count or learning rate differs from the
checkpoint is **rejected**, because resuming a cosine decay against a different
horizon would silently train a different model. Pass `--no-resume` to start over
deliberately.

### What each run writes

| Path | Contents |
|---|---|
| `<row>/run.json` | Full manifest: spec, model config, optimizer, parameter audit, schedule, curriculum, sampler plan |
| `<row>/telemetry/rank-*.jsonl` | Per-step loss, LR, grad norm, tokens/s, executed FLOPs, MFU, depth histogram, peak HBM, and the model's own routing telemetry |
| `<row>/analysis/routing-step-*.json` | Depth and width distributions, joint depth×width histogram, depth–width correlation, expert transition matrices, per-pass active ratios, halt calibration |
| `<row>/checkpoints/step-*/state.pt` | Model + optimizer state |
| `<row>/summary.json` | Final loss, steps, tokens, wall clock |

## 8. Correctness notes

Three bugs were found and fixed during implementation that would each have
produced plausible-looking but wrong results. They are recorded here because
each is the kind of thing a loss curve cannot reveal.

- **The depth-memory gate only closed half way.** `memory_gate_scale=0` silenced
  the representation path but left the retrieved summary flowing into the
  continuation, width, and pathway heads. MoRE-Core would still have *routed* on
  retrieved memory, so the MoRE-Core / MoRE-RM pair would have measured
  something narrower than it claims. The gate now applies to the summary as
  well; at the production scale of 1.0 the change is exactly a no-op.
  (`test_disabling_depth_memory_also_silences_the_routing_path`)
- **Expert transitions were compared positionally across packed passes.** A
  packed pass is a *subsequence* of its predecessor, not a prefix, so pairing by
  position matched unrelated tokens and manufactured off-diagonal mass out of
  nothing — directly inflating the evidence for the pathway axis. Records are
  now scattered back through each pass's own active mask before comparison.
  (`test_transitions_are_aligned_when_later_passes_are_packed`)
- **The storage policy was never applied.** `model.to(device)` preserves dtype,
  so every router stayed in BF16 instead of the FP32 the model tags them for. A
  BF16 router does not crash anything; it just makes the discrete decisions
  noisier, which is the hardest class of bug to catch. The trainer now calls
  `apply_parameter_storage_policy` and asserts the result in both directions.
  (`test_storage_policy_is_applied_to_routers`)

Two further gaps closed for robustness rather than correctness: replicated
N-gram tables produce sparse gradients that the dense reducer refuses, so
`enable_managed_sparse_gradient_sync` is now called and a multi-rank launch no
longer raises on its first step; and FP8 numerics are checked once per run
against a BF16 reference forward, aborting the row rather than letting it be
compared against twelve rows that stayed in parity.

### Standing caveats

- **Health gates are deliberately loosened.** The ladder contains rows whose
  routing is degenerate *by construction* — fixed k, random k, depth one — so
  the entropy and halt-collapse gates that protect a production run would abort
  the experiment they exist to measure. Non-finite loss and token drops still
  abort. The pre-clip gradient-norm gate is suspended for the warmup window.
- **Rows trained with `routed_k_mode=fixed` still carry unused `k_router`
  parameters.** Negligible in size; state it in the methods section.
- **Wall clock assumes 10% MFU**, which is unmeasured on Portage. Replace every
  estimate with measured tokens/second from the canary before promising a date.

## 9. Pre-flight checklist

1. `python -m metis_ablation.campaign plan --wave all` — allocation and cost.
2. Single-node canary at 16 APUs: confirm FP8 parity passes, record MFU, and
   compare `expert_execution: grouped` against `loop` for throughput.
3. Sweep (12 runs, ~1 h), pick per-archetype learning rates.
4. Wave 1 (13 rows, ~20 h). 5. Wave 2 (8 rows, ~5 h). 6. Wave 3 (3 rows, ~8 h).
7. Analysis pass over checkpoints; fill every `\TODO{}` in `main.tex`.
