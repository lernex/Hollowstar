# Wave-1 control recovery and recorded results, 5 September 2026

Scope: the non-Core/RM ablations. No Core/RM job, checkpoint, source checkout,
or recovery recipe was changed by this intervention.

## Completed original runs

Each completed row below reached **25,429 optimizer steps and
49,995,448,320 sampled tokens**. The difference from the requested 50B is the
original sampler's documented block alignment, not missing work.

| Original row | Job | Completed (UTC) | Final training loss | Median of last 100 logged losses | Training wall hours | Interpretation |
|---|---:|---|---:|---:|---:|---|
| Dense FLOP-matched | 495832 | 2026-09-05 05:17:10 | 1.535035 | 1.933596 | 12.374 | Not affected by the routed-expert cache defect |
| MoE k=4 | 495823 | 2026-09-05 13:14:49 | 1.845806 | 2.239998 | 20.658 | Completed, but not a valid full-expert-training baseline |
| MoE k=8 | 495824 | 2026-09-05 14:06:56 | 1.835471 | 2.225776 | 21.532 | Completed, but not a valid full-expert-training baseline |
| Random-depth | 495821 | 2026-09-05 11:24:29 | 1.873139 | 2.259818 | 18.813 | Completed, but not a valid full-expert-training baseline |

These are **training losses, not held-out quality measurements**. The last
batch is not representative of the full distribution; the tail medians and
matched 1,000-step windows are retained rather than treating a single final
loss as the architecture result. Wall hours come from each trainer summary;
scheduler elapsed times, including startup and shutdown, are recorded
separately.

The original source is
`ba9809284cf787d3e69df96620048dd96345d66f`, with campaign identity
`237c3878fa5c64691df626b14c3219dc989c5645e75978daa220fa23ce1e357a`.
Original artifacts remain under
`/lus/lustre1/vollmerc/more-paper-wave1-20260904/more-ablations`.

## The defect also affected sparse controls

Read-only inspection of the actual optimizer checkpoints found:

| Row | Checkpoint step | Routed chunks with nonzero momentum |
|---|---:|---:|
| MoE k=4 | 25,429, final | 0 / 16 |
| MoE k=8 | 25,429, final | 0 / 16 |
| Random-depth | 25,429, final | 0 / 16 |
| Random-k | 15,000 | 0 / 16 |
| Loop, pathway frozen | 15,000 | 16 / 16 |

Each chunk was matched to its recorded optimizer owner and unique
master-parameter shape. Every element of its momentum buffer was counted;
this was not a sample of a few coordinates. Checkpoint identities, optimizer
rank coverage, and recorded shard sizes were checked. This is not a
recomputation of every optimizer-shard checksum.

The affected rows use the original no-layer-recompute grouped-expert path.
The BF16 no-grad parity forward cached detached concatenations, so later
training did not deliver learned credit to those expert chunks. The
layer-recomputed frozen-loop control has nonzero momentum in all 16 chunks.
Weight decay can still move an expert's weights: zero learned momentum does
**not** mean the stored weights stayed bitwise unchanged.

Fix `227c62a7a3499b90bfef48706b1d78f358aa26ff` repairs this cache lifetime.
The user approved **fresh corrected sparse runs**, not resuming the invalid
optimizer lineage or silently replacing historical results.

## Original queue recovery

Dense FLOP-matched completion released MoR+dense-FFN job `495825`, and
random-depth completion released fixed-depth/adaptive-k job `495827`.
Together with MoE k=8 and the frozen-loop row, all four had advancing
telemetry, finite losses, and zero recorded dropped expert assignments.

Random-k was not merely waiting: job `495818` first had a node failure, then
its automatic restart failed again on `parrypeak063` with an RCCL system
error. Its last logged step was 15,510; its last committed checkpoint was
15,000. This failure cancelled the dependent MoR+fixed-k replacement
`495852`. Dense parameter-matched replacement `495853` had already been
cancelled through the failed RM dependency.

The unaffected dense parameter-matched row was restored as job **495942**.
The capacity-only dependency on fixed-loop job **495829** was removed after
confirming idle capacity. Both started at approximately **13:34 UTC** and
subsequently wrote advancing training steps. Their original source,
initialization, learning rates, data order, batch geometry, and output roots
were preserved.

New submissions exclude `parrypeak[007,012,020,026,056,063-064]`; this is a
user-job exclusion, not a cluster drain or a claimed repair of those nodes.
Already advancing jobs were not restarted.

The original MoE k=8 finished during this intervention; its final result and
terminal optimizer audit were added without interrupting it. Across the
13:39--14:08 UTC observation interval, all five remaining original jobs
advanced: dense parameter-matched by 1,070 steps, fixed loop by 550,
fixed-depth/adaptive-k by 480, MoR+dense by 1,040, and frozen loop by 470.
The failed original random-k is intentionally retained as historical
evidence, not treated as a live run.

## Corrected campaign protocol

The corrected controls use immutable source
`c7aea46eef2e407ae92ed9e9c0f874e8bf836237`, not the moving main branch.
Its model is identical to cache-fix commit `227c62a`; it also exposes the
explicit first-step expert-gradient requirement. The pinned history includes
the optimizer restore repair that preserves FP32 master weights and moments
and int8 momentum rather than converting their checkpoint dtypes.

All five rows retain the original serialized model configurations, row
specifications, release, seed, learning rates, 25,429-step schedule, and
20/80-rank allocations. The new optional routing features remain disabled:
compute allocation is explicitly `legacy`, and neither observed-depth credit
nor a causal/Core/RM experiment is enabled.

`run_corrected_control.sbatch` uses one allocation per row. It first executes
100 optimizer steps against the **full** training horizon, then requires
finite nonzero gradients in all 16 expert chunks on every rank and nonzero
saved momentum in all 16 optimizer-owned chunks. Only then may it resume the
same run identity to the full token budget. The regular checkpoint interval
is 500 steps. The early summary and per-rank/per-chunk evidence are retained
in the gate record before normal retention removes the early checkpoint.

The corrected output root is separate:
`/lus/lustre1/vollmerc/more-paper-controls-corrected-20260905/more-ablations`.
The wrapper rejects Core/RM rows, incorrect allocations, changed source,
and mismatched checkpoint identity. No old sparse checkpoint is imported.

### Live corrected rows at 14:19 UTC

| Corrected row | Job | Nodes | State | Latest logged step |
|---|---:|---:|---|---:|
| MoR + fixed-k MoE | 495953 | 20 | Full training, resumed after the expert gate | 360 |
| MoE k=4 | 495954 | 5 | Full training, resumed after the expert gate | 290 |
| MoE k=8 | 495955 | 5 | Full training, resumed after the expert gate | 150 |
| Random-k | 495956 | 20 | Pending resources | - |
| Random-depth | 495957 | 20 | Pending priority/capacity | - |

All three started rows saved nonzero momentum in **16/16 expert chunks** at
step 100. Initial and post-resume gradient evidence also covers **16/16
chunks on every rank**: 80 ranks for MoR+fixed-k and 20 each for the two
single-pass MoEs. The resumed run identities match their pilot identities.
These are learning/runtime qualification results, not completed 50B quality
results.

The two pending rows have **no Slurm dependencies** on failed original jobs
or on Core/RM. They are waiting for their 20-node allocations and will execute
the same gate before full training. All three running corrected rows had
telemetry less than 20 seconds old at this snapshot.

Use the corrected submission ledger and status snapshot for these new jobs;
the original campaign's hard-coded watcher still describes the original
output root and must not be mistaken for the corrected campaign.

## Evidence files

- `finished/<row>/run.json` and `summary.json`: exact copies of completed
  run manifests and summaries; no model weights or training samples.
- `results.json`: source hashes, completed results, original-run progress,
  sampler accounting, tail statistics, and matched-step loss windows.
- `completed-training-loss-windows.tsv`: complete 1,000-step windows with
  exactly 100 logged samples per cell for the completed original rows.
- `checkpoint-audit.json`: per-chunk optimizer evidence, checkpoint cursors,
  source revisions, and identity bindings.
- `moe-k8-final-checkpoint-audit.json`: the terminal audit added when k=8
  finished, superseding its earlier step-20,000 observation.
- `progress-confirmation.json`: paired observations covering every remaining
  active original control.
- `original-job-accounting.psv`: scheduler history, including failed,
  cancelled, completed, and restored jobs.
- `random-k-failure.txt`: the relevant failure-log lines and source hash.
- `checkpoint_gate.py`: fail-closed expert-learning gate for the separately
  recorded corrected campaign.
- `run_corrected_control.sbatch`: the two-stage, same-identity allocation
  wrapper, including a 20-minute no-progress watchdog.
- `corrected-submission.json`: the separate campaign's pinned source,
  operational-script hashes, complete submission arguments, and five job IDs.
- `corrected-status.json`: running and queued corrected rows, latest training
  steps, gate records, and all-rank post-resume gradient evidence.
- `gates/<row>.json`: exact copies of the passing 100-step expert-learning
  records, including each pilot's summary and checkpoint digest.
