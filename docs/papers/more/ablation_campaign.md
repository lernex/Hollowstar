# MoRE ablation campaign — sizing, compute, allocation, and execution

Status: measured Portage canaries, 2026-08-18. Supersedes the "Fast ablation
allocation" sizing in `METIS_1.6_PLAN.md:1038` for the post-PT window. Planning
tables remain for compute accounting; the accepted throughput lanes are
recorded in §5.

## 0. Window and available hardware

Wave 1 runs **after the Praxis and Logos jobs release Portage**, so it may use
the full 512-APU machine:

| Consumer | APUs | Notes |
|---|---:|---|
| **MoRE ablation campaign** | **up to 512** | 128 nodes at 4 APUs/node |

MI300A peak: **1961 TFLOP/s FP8 dense** per APU → 512 APUs = **1.004 EFLOP/s peak**.
Planning MFU band: **5% / 10% / 15%** (conservative / planning / good). The 10%
column is the one to plan against; treat 15% as reachable only after the grouped
expert GEMM in §5 lands.

## 1. Model size

**Parameter-matched shallow recurrent block.** The proxy keeps the original
1.8B stored-parameter and 7.3 GFLOP/token envelope, but concentrates it into one
Mamba-2 layer and one attention layer that are reused recurrently. This is both
closer to the object under study -- the repeated block -- and substantially
better matched to MI300A matrix dimensions than ten launch-bound narrow layers.

| | Praxis (production) | **Proxy S (ablation primary)** |
|---|---:|---:|
| `d_model` | 2048 | **4096** |
| Physical layers | 12 (10 Mamba + 2 attn) | **2 (1 Mamba + 1 attn)** |
| `latent_dim` | 1024 | **2048** |
| `expert_intermediate_dim` | 512 | **1152** |
| Routed experts | 128 | **72** |
| `k` range / target | 1–8 / 4 | **1–8 / 4** |
| `max_passes` / target | 5 / 2 | **5 / 2** |
| Stored total | ~3.5B | **1.81B** |
| Active / pass @ k=4 | ~0.47B | **0.279B** |

Audited primary-proxy parameter categories:

- Routed experts: 1.019B stored.
- Embedding / tied head: 0.268B stored; embedding runs once per token and the
  tied head runs once per active token-pass.
- Mamba + attention mixers: 0.160B stored.
- N-gram tables: 0.300B stored.
- **Active/pass @ k=4 = 0.279B**, @ k=8 = 0.336B.

## 2. FLOPs per token

**Use `python -m metis_ablation.campaign plan`, not a hand formula.** The
authority is `metis_training.metrics.estimate_train_flops` for the scientific
compute comparison and `estimate_hardware_flops` for the work the accelerator
actually executes. The latter counts three things a bare
`6 x active_parameters` estimate misses:

- the **pass-level recompute replay** (`activation_recompute_policy: pass`
  performs two forwards per backward, so executed work is 8F/6F of model FLOPs);
- the **depth-memory projections**, which run once per layer per pass and once
  per mHC stream on top of that;
- **parameter-free work** -- attention's quadratic score/value products and the
  Mamba-2 chunked-scan einsums.

For MoRE-Core at the proxy geometry the useful model work is **7.09
GFLOP/token**. The LM head runs and is checkpoint-replayed on every active
token-pass, so the accepted 80-APU no-layer-replay lane issues **8.16
GFLOP/token**. Layer checkpointing raises that to **9.45 GFLOP/token**.
Scheduling and MFU use the hardware number.
The iso-FLOP scientific claim uses the model number, so a memory-saving
implementation choice cannot silently redefine the comparison.

## 3. Token budget — 50B, not 10B

**Recommendation: 50B, and it is not a close call.**

| Budget | Campaign executed FLOPs (13 rows) | Two-batch wall @5% | @10% | @15% |
|---|---:|---:|---:|---:|
| 10B | 1,033 EFLOP | 13.0 h | 6.5 h | 4.3 h |
| **50B** | **5,166 EFLOP** | **64.8 h** | **32.4 h** | **21.6 h** |
| 100B | 10,331 EFLOP | 129.6 h | 64.8 h | 43.2 h |

Wall clock is the sum of the longest row in each of the two executable Wave-1
batches. The shared-seed 50-step real-data batches predict **48.8 hours**
before queue and restart overhead.

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
3. **Cost is not the constraint.** 50B costs roughly two days across the two
   measured full-machine batches.

Sample the 50B **proportionally from the 1T mixture including its phase
structure** (`METIS_1.6_PLAN.md:204`), not the first contiguous 50B. Use the
**identical token stream in identical order for every row** — that removes data
order as a confound for free, and it is worth a sentence in the paper.

Keep headroom to extend rows 2, 10, and 11 to 100B if their curves have not
separated.

## 4. Allocation — measured lanes in two execution batches

The accepted lanes require 800 APUs in aggregate, so the thirteen scientific
rows cannot all run simultaneously. The measured makespan-first schedule uses
two batches: 1a combines the dense, single-pass, learned, and random controls at
440 APUs; 1b keeps the fixed-loop baseline with the depth, width, and pathway
isolations at 360 APUs. Both batches kept every row above 500k under the shared
initialization policy.

### Master table — Wave 1, all 13 runs, 50B tokens each

| # | Model | Batch | Stored / active per pass | Model / executed GFLOP/tok | APUs | Nodes | Measured hours |
|---:|---|:---:|---|---:|---:|---:|---:|
| 1 | Dense, FLOP-matched | 1a | 1.44B / 0.87B | 7.09 / 7.62 | 80 | 20 | 6.2 |
| 2 | Dense, parameter-matched | 1b | 1.81B / 1.24B | 9.31 / 9.85 | 80 | 20 | 8.4 |
| 3 | MoE k=4 | 1a | 1.81B / 0.279B | 3.54 / 4.08 | 20 | 5 | 17.0 |
| 4 | MoE k=8 | 1a | 1.81B / 0.336B | 3.88 / 4.42 | 20 | 5 | 17.9 |
| 5 | Fixed LoopMoE | 1b | 1.81B / 0.279B | 7.09 / 9.45 | 40 | 10 | 20.5 |
| 6 | Loop, pathway frozen | 1b | 1.81B / 0.279B | 7.09 / 9.45 | 40 | 10 | 20.1 |
| 7 | MoR + dense FFN | 1b | 0.85B / 0.278B | 7.08 / 8.15 | 80 | 20 | 22.1 |
| 8 | MoR + fixed-k MoE | 1b | 1.81B / 0.279B | 7.09 / 8.16 | 80 | 20 | 24.4 |
| 9 | Fixed depth, adaptive k | 1b | 1.81B / 0.279B | 7.09 / 9.45 | 40 | 10 | 20.5 |
| 10 | MoRE-Core | 1a | 1.81B / 0.279B | 7.09 / 8.16 | 80 | 20 | 23.9 |
| 11 | MoRE-RM | 1a | 1.81B / 0.279B | 7.09 / 8.16 | 80 | 20 | 24.4 |
| 12 | Random-k control | 1a | 1.81B / 0.279B | 7.09 / 8.16 | 80 | 20 | 24.4 |
| 13 | Random-depth control | 1a | 1.81B / 0.279B | 7.09 / 8.16 | 80 | 20 | 11.5 |
| | **Batch 1a total** | | | | **440** | **110** | **24.4** |
| | **Batch 1b total** | | | | **360** | **90** | **24.4** |

Regenerate this table with `python -m metis_ablation.campaign plan`; it is
computed from `metis_training.metrics.estimate_hardware_flops`, the same
accounting the trainer reports against every telemetry step, rather than from a
parallel hand-derivation that can silently drift.

Every row is the **Proxy S geometry** from §1 (two physical layers, `d_model`
4096, latent 2048, 72+1 experts, `R_max`=5) except where the architecture forbids
it: rows 1 and 7 are dense, so stored equals active and a dense-FFN recursive
model simply cannot hold 1.8B stored at 0.279B active. Report both numbers; the
asymmetry is what the sparse axis buys.

**Two execution batches, 48.8 measured hours before queue/restart overhead.**

### Wave 2 — scaling ladder

One size is a data point; three sizes are a trend. Four archetypes at two
smaller geometries turn "MoRE beats its baselines at 1.8B" into a slope, with
Praxis and Logos as the fourth and fifth points on the same curve.

| # | Model | Isolates | Stored / Active | GF/tok | APUs | Nodes | Hrs @10% |
|---:|---|---|---|---:|---:|---:|---:|
| 20 | Dense, parameter-matched (XS) | scaling point: dense reference at MoRE's stored parameters | 0.59B / 0.42B | 3.53 | 60 | 15 | 4.2 |
| 21 | MoE k=4 (XS) | scaling point: sparse routing without recursion | 0.59B / 0.13B | 1.76 | 40 | 10 | 3.1 |
| 22 | MoRE-Core (XS) | scaling point: all three axes together | 0.59B / 0.13B | 3.54 | 60 | 15 | 4.2 |
| 23 | MoRE-RM (XS) | scaling point: route-typed recurrent depth memory | 0.59B / 0.13B | 3.54 | 60 | 15 | 4.2 |
| 24 | Dense, parameter-matched (XXS) | scaling point: dense reference at MoRE's stored parameters | 0.21B / 0.13B | 1.41 | 24 | 6 | 4.2 |
| 25 | MoE k=4 (XXS) | scaling point: sparse routing without recursion | 0.21B / 0.05B | 0.94 | 16 | 4 | 4.2 |
| 26 | MoRE-Core (XXS) | scaling point: all three axes together | 0.21B / 0.05B | 1.89 | 24 | 6 | 5.6 |
| 27 | MoRE-RM (XXS) | scaling point: route-typed recurrent depth memory | 0.21B / 0.05B | 1.89 | 24 | 6 | 5.6 |
| | **Total** | | | | **308** | **77** | **5.6** |

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
model design. Repeat the three rows the abstract quotes plus the fixed/frozen
pathway pair, whose expected effect may be smaller. Same data order, same
schedule, different initialization — the launchers pass
`--seed 27182818` automatically, and
`test_seed_wave_mirrors_its_parents_and_changes_only_the_seed` asserts that
nothing else moved.

| # | Model | Isolates | Stored / Active | GF/tok | APUs | Nodes | Hrs @10% |
|---:|---|---|---|---:|---:|---:|---:|
| 40 | Dense, parameter-matched (seed 2) | paired repeat of dense-param-matched | 1.81B / 1.24B | 9.85 | 80 | 20 | 8.7 |
| 41 | MoRE-Core (seed 2) | paired repeat of more-core | 1.81B / 0.279B | 8.16 | 80 | 20 | 7.2 |
| 42 | MoRE-RM (seed 2) | paired repeat of more-rm | 1.81B / 0.279B | 8.16 | 80 | 20 | 7.2 |
| 43 | Fixed LoopMoE (seed 2) | paired repeat of loop-fixed | 1.81B / 0.279B | 9.45 | 40 | 10 | 16.7 |
| 44 | Loop, pathway frozen (seed 2) | paired repeat of loop-pathway-frozen | 1.81B / 0.279B | 9.45 | 40 | 10 | 16.7 |
| | **Total** | | | | **320** | **80** | **16.7** |

### Whole campaign

| Wave | Rows | Executed FLOPs | Wall clock @10% MFU |
|---|---:|---:|---:|
| 1 — the ladder, batches 1a + 1b | 13 | 5,166 EFLOP | 32.4 h at 10% MFU / **48.8 h measured** |
| 2 — scaling | 8 | 926 EFLOP | 5.6 h |
| 3 — seeds | 5 | 2,254 EFLOP | 16.7 h |
| **Total** | **26** | **8,346 EFLOP** | **~54.7 h at 10% MFU; ~71.1 h with measured Wave 1** |

Waves run one after another, so the 512-APU machine limit applies per
batch, not to the sum. Add the archetype learning-rate sweep (12 short runs at 1B tokens,
about an hour) ahead of wave 1.

## 5. Parallelism — pure data parallelism, replicated experts, no EP

**Recommendation: DP only. No expert parallelism, no tensor parallelism, no
pipeline parallelism.**

The measured lanes use memory in two ways. The 80-APU adaptive and dense rows
fit six sequences per APU with no layer replay; the 40-APU fixed-depth rows fit
twelve sequences per APU with layer replay. All 72 routed experts are
replicated on every rank, so the campaign still avoids an all-to-all systems
confound.

Consequences, all good:

- **Zero all-to-all.** Routing becomes a local gather/scatter. No dispatch/combine
  collectives, no dropless overflow path, no expert-load stragglers.
- **No systems confound in the science campaign.** Every row's throughput depends
  only on its own arithmetic. This matters: an EP campaign would make row-to-row
  wall-clock differences partly an artifact of routing skew, and wall-clock is
  one of the reported axes.
- The plan's canary race (`METIS_1.6_PLAN.md:1041`) selected
  replicated-experts + DP by a wide margin.

### Gradient synchronization is the real bottleneck, not compute

Ring all-reduce moves ~2× model size per rank per step regardless of DP width:
**~6 GB per rank per step** for a 1.5B model in BF16. Every accepted lane
consumes the same 480-sequence / 1,966,080-token optimizer step, so an unhidden
all-reduce would dominate the faster rows.

The original Portage runtime was silently using RCCL's socket fallback. A
two-node, eight-APU 1 GiB all-reduce measured **2.49 GB/s algorithm bandwidth**
over `NET/Socket`. The
[HPE Slingshot RCCL guidance](https://github.com/HewlettPackard/shs-ccl-docs/blob/main/rccl/rccl_tuning_guide.md)
(June 2026), its OFI plugin, and the prescribed CXI rendezvous settings raised
the same measurement to **52.3 GB/s** over
`NET/OFI/*/GDRDMA` on 2026-08-17, a 21× transport improvement. Multi-node
launches therefore request `--network=disable_rdzv_get` and fail closed unless
the pinned `librccl-net.so` is present. Single-node probes deliberately keep
RCCL's native transport because forcing OFI there allocates an unnecessary VNI.
Generated jobs also terminate a step after twenty minutes without rank-zero
telemetry progress. This catches the Portage failure mode where a compute node
reboots, its four ranks disappear, and the surviving ranks otherwise remain at
100% GPU in collectives indefinitely.

The plugin's hwloc dependency is built without ROCm device discovery. The
PyTorch wheel already carries ROCm-SMI, while a ROCm-enabled external hwloc
loads a second ROCm-SMI SONAME; both copies register the same process-exit
tables and the otherwise successful job aborts in a double free at teardown.
PCI topology discovery is sufficient for the four CXI devices and preserves
clean communicator shutdown.

Mitigations, in order of preference:

1. **Bucketed all-reduce overlapped with the backward pass** (standard DDP). Non-optional.
2. **Measured micro-batch/accumulation pairs that all multiply to 480
   sequences.** The retained lanes use 80×6×1, 40×12×1, 20×12×2, or
   24×10×2. Comparability matters more than a row-specific global batch.
3. ZeRO-1 optimizer sharding if memory headroom is wanted for larger micro-batches.
   The Wave trainer now supports deterministic full-world ownership and
   rank-local, hash-bound optimizer checkpoint shards. Parameters remain
   replicated and are broadcast in fixed owner order after each update, so
   model math is unchanged and a preempted run restores every owner's state
   without consolidating tens of gigabytes on rank zero.
4. FP8 gradient reduction halves the wire cost but adds a numerical risk the
   science campaign does not need. Skip unless MFU measurement demands it.

The campaign uses blockwise 8-bit Muon momentum while retaining FP32 AdamW
states and FP32 master weights. Gupta et al.,
[Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106v3)
(February 2026), found that 2,048-value linear-quantization blocks matched
full-precision Muon within 0.002 validation loss from 162M through 2.7B
parameters and within seed variation on six downstream tasks, while reducing
optimizer-state memory by up to 62%. Their fully quantized AdamW hybrid showed a
small but consistent quality loss, so this implementation deliberately
quantizes only Muon momentum. The retained Portage lanes include that policy and
keep AdamW state unquantized.

### Grouped expert GEMM — landed

`AdaptiveDroplessMoE._execute_local` issued one `torch.nonzero` per expert,
which forces a device-to-host synchronization each time. Under production expert
parallelism a rank owns 1–6 experts and the cost is invisible; under
replicated-expert DP a rank owns all 72, giving thousands of stalls per forward.

`expert_execution: grouped` sorts assignments once and derives segment
boundaries from a single `bincount`, cutting synchronizations from 96 per layer
to one. Numerics are unchanged — `test_grouped_expert_execution_matches_the_loop_exactly`
asserts bitwise-close losses between the two paths. Praxis and Logos keep
`expert_execution: loop`, so production is byte-identical to before.

### FP8 scaling on Portage

The ablation family uses Transformer Engine **blockwise FP8 scaling** while
Praxis and Logos retain delayed scaling. Transformer Engine 2.17 documents
blockwise FP8 as a native gfx942 path: each row/column block gets its own scale
rather than sharing one scale across a whole tensor. The former current-scaling
path measured **3.06% MFU** against **2.81%** for delayed scaling; the retained
Wave-1 lanes use blockwise scaling with per-run BF16 parity checks.

MoRE-Core's accepted lane runs **80 APUs × 6 sequences × 1 accumulation** with
no layer replay. The 480-sequence global batch and token order remain identical
to every other row.

The learned depth and width policies are **budgeted rankings**, not independent
Bernoulli/Gumbel draws whose means are merely penalized. At every batch, the
routers rank tokens while a maximum-entropy marginal supplies exact integer
counts at mean depth 2 and mean k 4. This makes collapse to all-depth-1,
all-depth-5, all-depth-2, or all-k-4 impossible without dictating *which* tokens
receive the compute. Random-policy controls use the same marginals with random
rankings, so learned-versus-random comparisons spend exactly the same compute.

The primary depth marginal uses the trained support **{1, 2, 3}**, uniformly at
mean 2: easy tokens receive one pass, medium tokens two, and hard tokens three.
This retains three genuine compute levels while avoiding depth-4/5 tail batches
that measured as almost pure launch overhead on MI300A. `max_passes=5` remains
the architectural cap for explicit controls and later budget studies, but the
primary 50B-token policy does not train on those two tail levels.

### Measured Wave-1 throughput lanes (updated 2026-09-03)

The target is 500k real release-data tokens/s for every row. The table reports
the warmed median from the retained canary window; every run used a 480-sequence
global batch, 4,096-token sequences, NS=5 Muon, blockwise int8 Muon momentum,
full-world optimizer ownership, four expert-weight chunks, an 8,192-token LM
head chunk, and the OFI/CXI runtime. FP8 loss parity stayed below 0.001 in every
retained row.

| Row | APUs | Micro × accum | Recompute | Precision exception | Median tok/s |
|---|---:|---:|---|---|---:|
| dense-flop-matched | 80 | 6 × 1 | none | dense FFN BF16 | **2,227,718** |
| dense-param-matched | 80 | 6 × 1 | none | dense FFN BF16 | **1,654,017** |
| moe-k4 | 20 | 12 × 2 | none | — | **816,419** |
| moe-k8 | 20 | 12 × 2 | none | — | **776,634** |
| loop-fixed | 40 | 12 × 1 | layer | — | **677,463** |
| loop-pathway-frozen | 40 | 12 × 1 | layer | — | **690,935** |
| mor-dense-ffn | 80 | 6 × 1 | none | — | **627,100** |
| mor-fixed-k | 80 | 6 × 1 | none | — | **570,256** |
| fixed-depth-adaptive-k | 40 | 12 × 1 | layer | — | **678,299** |
| more-core | 80 | 6 × 1 | none | — | **582,315** |
| more-rm | 80 | 6 × 1 | none | — | **568,392** |
| random-k | 80 | 6 × 1 | none | — | **569,222** |
| random-depth | 80 | 6 × 1 | none | — | **1,212,482** |

In the final shared-initialization 440-APU batch, Core and RM differ by 2.4% in
throughput.
The earlier 512k-versus-695k comparison mixed Core's long window with RM's
first nine warmed steps. RM does not have a magically faster kernel: Core keeps
the recurrent-memory module parameter/compute matched and gates its retrieved
summary to zero, while RM lets that same path affect representation and routing.
Both kept exact realized depth 2/k 4 with no dropped assignments; RM's
`depth_memory_last_norm` remained nonzero and reached 12.80, confirming that
the memory path is active rather than decorative.

Rank-zero training-batch routing captures at step five show that the exact
means are not hiding constant decisions. Both learned rows populated depths
1/2/3 in exact thirds and populated the complete k=1..8 support. Depth and
width remained distinct axes (Pearson r = -0.055 for Core and -0.018 for RM).
The full top-k coalition also changed substantially. Core's mean Jaccard
overlap was 0.155 from pass 1→2 and 0.422 from pass 2→3, with exact-set matches
of only 1.6% and 7.1%; RM measured 0.158/0.419 Jaccard and 1.6%/7.2% exact
matches. The corresponding top-1 winner off-diagonal masses were 0.837/0.578
for Core and 0.838/0.568 for RM. Active-token ratios were exactly 1, 2/3, and
1/3 across the three trained passes. Thus the policies are neither
all-depth-2, fixed-k-4, nor a replay of the same expert coalition.

These are the rates measured in the final two execution batches, not a sum of
isolated canaries. The slowest row is RM at 568,392 tokens/s.

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
export METIS_REPO=/home/users/vollmerc/Metis
export METIS_SCRATCH=/lus/lustre1/vollmerc
export METIS_RELEASE_ROOT=/lus/lustre1/vollmerc/metis-1.6/releases/metis-1.6-data-r2
export METIS_ABLATION_RUNTIME="$METIS_REPO/ops/more-ablation-runtime.sh"
export METIS_ABLATION_EXCLUDE_NODES=parrypeak026

cd "$METIS_REPO"
PYTHONPATH=src python -m metis_ablation.campaign plan --wave all
```

```bash
PYTHONPATH=src python -m metis_ablation.campaign sweep \
  --destination slurm/ablation \
  --output-root "$METIS_SCRATCH/more-ablations" \
  --release-root "$METIS_RELEASE_ROOT"
bash slurm/ablation/sweep/launch-sweep-1a.sh
# Launch only after sweep batch 1a has completed:
bash slurm/ablation/sweep/launch-sweep-1b.sh
```

After selecting the four archetype winners, write:

```bash
cat > "$METIS_SCRATCH/more-ablations/learning-rates.json" <<'JSON'
{
  "dense-param-matched": 0.00018,
  "moe-k4": 0.00018,
  "loop-fixed": 0.00018,
  "more-core": 0.00018
}
JSON
```

Replace those values with the measured winners, then generate and launch the
two explicit Wave-1 batches:

```bash
PYTHONPATH=src python -m metis_ablation.campaign slurm \
  --wave 1 \
  --destination slurm/ablation \
  --output-root "$METIS_SCRATCH/more-ablations" \
  --release-root "$METIS_RELEASE_ROOT" \
  --learning-rates "$METIS_SCRATCH/more-ablations/learning-rates.json"
bash slurm/ablation/wave1/launch-wave-1a.sh
# Launch only after batch 1a has completed:
bash slurm/ablation/wave1/launch-wave-1b.sh
```

Repeat generation with `--wave 2` and `--wave 3`; both fit in one execution
batch. Regenerate on Portage rather than editing generated files by hand.
Committed untuned launchers reference required
`METIS_ABLATION_LR_<ARCHETYPE>` variables and fail closed; they cannot silently
fall back to the default learning rate.

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
- **The same gate accidentally disabled shared N-gram memory.** N-gram retrieval
  is common to every Wave-1 row; only route-typed recurrent depth memory differs
  between MoRE-Core and MoRE-RM. The curriculum now carries independent depth
  and N-gram gate scales, and the production token schedule keeps its previous
  joint annealing while both ablation rows keep N-gram injection at 1.0.
  (`test_more_core_keeps_shared_ngram_memory_enabled`)
- **Exact hard budgets bypassed soft-policy calibration.** The `{1,2,3}`
  assignment kept realized depth exactly 2.0, but its soft continuation
  probabilities climbed to 2.94 expected passes in four steps because the
  aggregate penalty was too weak. Reusing the adaptive path's integral
  controller was not the answer: delayed response made expected depth oscillate
  between 2.94 and 1.02 while realized compute stayed fixed. Budgeted depth and
  width now train directly against their own detached exact assignments:
  binary cross-entropy for continue/halt and categorical cross-entropy for k.
  The labels are the token rankings selected under the exact quota, so the
  calibration sharpens easy/hard separation without changing one executed
  token or carrying controller state across steps. On Portage, coefficients
  `depth=4.0` and `k=0.1` moved expected depth to 2.16 and expected k from 7.24
  to 4.12 by step eight; weighting k at 1.0 slowed the language-model objective
  because that loss is evaluated once per MoE layer and pass.
  (`test_exact_budget_calibration_teaches_the_selected_depth_and_width`)
- **The aux-loss-free expert bias was too slow for the routing transient.**
  At `1e-3`, expert-load CV still exceeded 2.4 after twenty steps; raising the
  non-gradient sign update to `0.05` reduced it to 1.64 at the same point while
  depth, k, and loss stayed on their calibrated trajectories. The bias changes
  expert identity only, never combination affinity or executed assignments.
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
- **Wave-2 and Wave-3 wall clocks remain model-based estimates.** Wave 1 uses
  measured Portage rates; the smaller scaling geometries and second-seed batch
  still need their launch-day warm-step telemetry checked before promising an
  end time.

## 9. Pre-flight checklist

1. Confirm the Portage checkout, runtime, release inventory, and idle-node count.
2. `python -m metis_ablation.campaign plan --wave all` — allocation and cost.
3. Sweep (12 runs in two capacity-safe batches), record the four archetype
   winners in `learning-rates.json`.
4. Wave 1a, then Wave 1b (13 rows, about 49 measured hours before queue/restart
   overhead).
5. Wave 2 (8 rows, estimated ~4 h at 10% MFU).
6. Wave 3 (5 rows, estimated ~13 h at 10% MFU).
7. Analysis pass over checkpoints; fill every `\TODO{}` in `main.tex`.
