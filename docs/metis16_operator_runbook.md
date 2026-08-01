# Metis-1.6 login2, Rhea, and Portage operator runbook

The three environments have deliberately separate responsibilities:

| Environment | Responsibility | Profile |
|---|---|---|
| `login2` | Authenticate, acquire 1T-base candidates, the 25.91B-token context reserve, and evaluation holdouts onto Lustre | `login2` |
| Rhea | Normalize, filter, deduplicate, decontaminate, train the tokenizer, tokenize, select, and shard both the 1T base and exact 18B context release | `rhea` |
| Portage | Consume the verified immutable base and post-training releases for simultaneous Praxis/Logos base training, context extension, and post-training | autonomous trainer configuration, not a data-prep profile |

## Information confirmed by the account owner

- The account is `vollmerc` in group `sumusa`. The acquisition root is
  `/lus/lustre1/vollmerc/metis-1.6`; the launcher creates it if needed. `/lus/lustre1` itself must
  never be used as the data root.
- `login2` is the permitted acquisition host.
- GNU Screen, `/usr/bin/python3.11`, Cray Python modules, Apptainer, and the Lustre tools are present.
- Hugging Face, GitHub, and Common Crawl HTTPS endpoints are reachable without a proxy.
- A `0/0` default quota report does not prove that usable capacity has been assigned. Capacity confirmation
  remains an operator prerequisite.
- Rhea's scheduler account, partition, QoS policy, array limit, node memory, wall-time limit, and
  view of the Lustre filesystem are not yet known. Its profile therefore fails closed.

Before the first login2 launch, confirm three remaining account-level prerequisites:

- the account owner can clone the private `lernex/Metis` repository;
- the user's Hugging Face account has accepted the gated HLE and GPQA dataset terms as well as the
  gated NVIDIA source terms (the preflight checks every one and stops before large downloads);
- a read-only GitHub token with repository Metadata access is available, or `gh auth` is already
  configured for the account, so FreshGitHub can reject forks and mirrors fail-closed;
- `pypi.org`/Python package-file HTTPS is reachable for the first hash-locked runtime install, and
  at least 3TB free for acquisition (5TB recommended) is genuinely available despite the ambiguous
  `0/0` report. Rhea later requires 8TB free (12TB recommended) while it builds the release.

Portage's `parry` partition, `MaxArraySize=1001`, 10,000-job limit, five-day wall-time limit, and
roughly 512,000MB nodes are Portage facts. They are not copied into the Rhea profile.

The later one-command Portage model-training handoff, autonomous probe ladder, dual-family
allocation, requeue behavior, and evidence gates are specified in
`docs/metis16_portage_training_runbook.md`.

Portage is not ready from tokenized shards alone. Before the Portage command,
Rhea must also build the sealed post-training umbrella containing static
SFT inputs, the DeepSeek capability, verifiers, preference/evaluation data,
and sealed generation adapters. The release builder installs the common
checkpoint-bound hook itself. Portage must also expose a compatible
site runtime or pinned offline bundle. Once those release gates are complete,
the only Portage launch/resume command is:

```bash
./ops/start-portage-training.sh
```

## Clone once, then one login2 command

From the account owner's home directory:

```bash
git clone git@github.com:lernex/Metis.git
cd Metis
```

Then the only acquisition command is:

```bash
./ops/start-acquisition.sh \
  --lustre-root /lus/lustre1/vollmerc/metis-1.6 \
  --quota-acknowledgement administrator-confirmed
```

Use `administrator-confirmed` only after the Lustre administrator has confirmed at least the
required usable capacity for this account. If the administrator explicitly confirms that the
project has no hard quota, use `--quota-acknowledgement unlimited` instead. The same value may be
provided through `METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT`; an ambiguous `0/0` report without either
explicit acknowledgement is a production preflight failure.

The launcher prompts invisibly for the user's Hugging Face read token only when no existing local
credential is available. It also reuses `GITHUB_TOKEN`, `GH_TOKEN`, or `gh auth`, and otherwise
prompts invisibly for a read-only GitHub token. Neither token is placed in the command line, Screen
session name, state files, or logs. A discovered Hugging Face token file must be owned by the current
account, must not be a symlink, and must have no group or world permissions (mode `0600` or stricter).
Never paste either credential into email, Git, YAML, or a shell command argument. GH Archive and
codeload provide events and source bytes; the authenticated metadata check is what makes the
fork/mirror exclusion fail closed. GitHub rate-limit reset headers are honored automatically.

The launcher creates the directory, starts one `metis16-acquisition` GNU Screen session, installs
the pinned Python runtime, runs the full acquisition doctor, resolves the immutable source lock,
and runs the restart-safe supervisor in the foreground inside Screen. The runtime is installed from
`requirements-metis16-data.lock`, which pins the complete transitive dependency graph and includes
SHA-256 hashes for every accepted distribution. Bootstrap uses `--require-hashes` and
`--only-binary=:all:`; it neither performs an unbounded pip upgrade nor compiles unreviewed source
packages on the login host. It returns immediately, so the SSH connection may close without
stopping acquisition.

The runtime contract supports CPython 3.11 and 3.12. Its Linux x86_64 wheels were resolved for both
ABIs; `login2` uses the confirmed `/usr/bin/python3.11`. If the direct input file, generated lock,
Python ABI range, or installed package set differs, bootstrap rebuilds the dedicated virtual
environment before doing any data work. The human-edited input file is not an installation surface:
it exists only to regenerate and review the transitive lock.

Acquisition advances in dependency-safe waves: packaged Hugging Face/index payloads first; Common
Crawl, pinned repository, canonical-source, and recent-GitHub materializers second; and engineering
discussion extraction only after the repository-license cache exists. A failed wave prevents every
dependent wave from starting. Network concurrency is bounded independently for Hugging Face,
Common Crawl, canonical sites, and public GitHub archives.

The same source requests include the continued-pretraining reserve: 43.91B
long-document candidates in total, 25.91B beyond the candidates already
needed by the 1T plan. The reserve reuses 31 audited base-source families and
their download fallbacks; it does not download the foreign-tokenized ProLong
release. See
[`metis16_context_extension_data.md`](metis16_context_extension_data.md).

```bash
screen -r metis16-acquisition
METIS_LUSTRE_ROOT=/lus/lustre1/vollmerc/metis-1.6 ./metisctl status --profile login2
tail -f /lus/lustre1/vollmerc/metis-1.6/logs/metis-1.6-data-r1/acquisition/screen.log
```

Rerunning the launcher is the resume operation. Completed content-addressed tasks are skipped, and
the singleton lock prevents two supervisors from writing the same acquisition concurrently.

### Common Crawl acquisition is bounded by the URL-index scan, not by the network

Each Common Crawl release publishes 300 Parquet URL-index partitions in `subset=warc`, roughly
570 MB and seven million rows each — about 172 GB and 2.1 billion rows per crawl, and each of the
four fresh routes scans all five crawls. The download is a rounding error next to the scan: the
per-row metadata gates are pure Python, so a route that appears "stuck downloading" is almost
always scanning. Judge progress by partitions committed, not by bytes moved:

```bash
sqlite3 "$(ls -d /lus/lustre1/vollmerc/metis-1.6/raw/*/freshweb/runs/*/state.sqlite3 | head -1)" \
  "SELECT COUNT(*), SUM(scanned), SUM(selected) FROM partitions"
```

`index_scan_workers` in the `login2` profile is the throughput lever. Each worker is a process
that scans one partition end to end, and it also fetches that partition, so the setting bounds
concurrent index requests as well. Raise it toward the cores login2 can spare (16 is the ceiling)
or override it per run:

```bash
METIS_CC_INDEX_SCAN_WORKERS=12 ./ops/start-acquisition.sh --lustre-root /lus/lustre1/vollmerc/metis-1.6
```

Throughput, retry, and cache settings are deliberately excluded from the run fingerprint, so
retuning them **resumes** an in-flight acquisition. Partitions already committed to the ledger are
never rescanned. `range_workers`, `shard_count`, `keep_index_files`, and
`final_opt_out_reserve_multiplier` are *not* excluded: changing any of those mints a new fingerprint
and abandons every partition already scanned. Leave them alone mid-run.

`keep_index_files: false` makes each route re-fetch the index rather than hold ~860 GB of Parquet on
Lustre. That trade is now cheap — fetching a partition costs a few percent of scanning it — but it
is fingerprinted, so it can only be reconsidered for a new generation, not for a run in flight.

The acquisition ledger is a SQLite database, which is not designed for a network filesystem. The
URL-index phase commits once per partition and does not care, but the WARC phase commits once per
fetched group. If login2 exposes node-local scratch, point the ledger at it; the working copy is
published back to Lustre every `state_checkpoint_seconds`, and a crash rewinds to the last
checkpoint rather than to the start, because shard recovery truncates each shard back to the
offset the ledger committed:

```bash
METIS_CC_STATE_SCRATCH=/local/scratch/$USER ./ops/start-acquisition.sh --lustre-root /lus/lustre1/vollmerc/metis-1.6
```

### The repository index is a random-write database, and Lustre is the wrong disk for one

The Nemotron repository builder groups every metadata row on disk before it
fetches a single archive, and both of its hot tables are `WITHOUT ROWID` B-trees
keyed on `sha256(repo, commit)`. That key is uniformly random, so each of the
tens of millions of inserts lands at a random position in a multi-gigabyte tree.
Once the tree outgrows the page cache, every insert becomes a page read — and on
Lustre a page read is a network round trip. That is what a report like

> 20,468,796 repository commits indexed. 45,840,280 requested paths indexed.
> 0 repositories processed. 0 output batches produced.

meant: the builder was healthy and grouping, just at storage speed rather than
CPU speed. Measured against the real metadata, that was 191 of 2,748 row groups
in sixty hours — 7%, on a curve that gets *slower* as the tree grows, because a
bigger tree means a lower cache hit rate. No page cache fixes that; the finished
tree is far larger than any cache worth giving it.

The grouping phase is therefore an external sort. Every row is appended to one
of `repository_index_sort_buckets` spool files chosen by the leading byte of its
repo key — a sequential write, which is what a parallel filesystem is good at —
and the buckets are then loaded in ascending order, each sorted before insert.
Because bucket order is key order, every row sorts after everything already in
the tree, so the load appends to the rightmost leaf instead of seeking to a
random one. Watch it with:

```bash
"$HOME/.cache/metis/runtime-login2/bin/python" -c "import sqlite3,sys;c=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True,timeout=120);print({k:c.execute('SELECT COUNT(*) FROM '+k).fetchone()[0] for k in ('metadata_units','sort_buckets','repo_commits','requested_paths','repository_state','output_batches')})" /lus/lustre1/vollmerc/metis-1.6/raw/nemotron_repository_code_v123/repository-index-cache/requests.sqlite3
```

Read it in two phases. `metadata_units` climbing toward the total row-group count
is the spool phase; `sort_buckets` climbing toward `repository_index_sort_buckets`
is the load phase; `repository_state` and `output_batches` moving off zero means
archives are finally being fetched. `database is locked` here is not a fault —
the index uses a rollback journal on Lustre, so readers wait for the writer. Pass
a `timeout` as above rather than treating it as a hang.

`metadata_units` is the resume unit — one Parquet row group. Units are recorded
complete only once their rows sit in a closed spool part, so a restart replays at
most one roll (`repository_index_spool_rows`) and never starts over. A crash
during the load resumes at the first bucket `sort_buckets` does not name.

Two consequences worth knowing. The spool costs transient disk roughly the size
of the index itself, under `repository-index-cache/index-sort`, and is deleted
once the load completes. And the load cannot absorb rows that arrive after it has
started: if new metadata is downloaded mid-build the next run fails closed and
asks you to rebuild, rather than silently dropping those rows into buckets that
were already drained.

If login2 exposes node-local disk, point the index at it — the sort makes this
much less important than it was, but it still helps:

```bash
METIS_REPO_INDEX_SCRATCH=/local/scratch/$USER ./ops/start-acquisition.sh --lustre-root /lus/lustre1/vollmerc/metis-1.6
```

The working copy is republished to Lustre between metadata units and after every
committed output batch, so a lost node rewinds to the last checkpoint and never
past published output.

`repository_index` now has its own scheduling lane. Its grouping phase does no
network work at all, and while it held the single GitHub slot the repository and
discussion builders were blocked behind it for days with idle workers. The two
lanes together put at most two consumers on public GitHub endpoints.

The command remains fail-closed if the root is unsafe, capacity is insufficient, a credential or
gated source is unavailable, a materializer has not passed its fixture, a source remains a remote
plan, the repository is dirty, holdouts are incomplete, or an artifact hash/size no longer matches.

## Immutable login2-to-Rhea handoff

Successful acquisition emits `state/metis-1.6-data-r1/ACQUISITION_READY.json`. It binds:

- the fully expanded data manifest;
- the immutable source lock;
- the hash-locked Python dependency contract and acquisition interpreter identity;
- every download completion marker;
- materialized artifact paths, byte sizes, and SHA-256 hashes;
- the evaluation holdout bundle;
- the clean repository commit used for acquisition.

It also reports measured per-source candidate counts and the deterministic replacement allocation.
If one source is short, compatible donor surplus is assigned automatically without changing the
source category, phase, or freshness target. The handoff stops if those reserves are insufficient.
For the three singleton fresh Common Crawl routes, acquisition first widens the five preferred 2026
crawls and then automatically activates `CC-MAIN-2026-04`; it never substitutes historical generic
web. No operator flag is needed for either path.

When a materializer represents its output as a directory, the directory inode is never treated as
an artifact. The handoff records and verifies its acquisition receipt plus every shard named by the
receipt.

Rhea verifies that handoff before submitting any CPU work. A changed manifest, dependency lock,
supported Python ABI contract, source lock, completion marker, holdout bundle, artifact inventory,
or repository commit is fatal. Rhea may use either CPython 3.11 or 3.12, but must install the exact
same dependency lock. The artifacts are bound by paths relative to the acquisition root, so Rhea
may expose the same Lustre content at a different absolute mount prefix.

The submission command performs only fast structural and size checks. It then places a restartable
`handoff_signature` Slurm array ahead of normalization: one task hashes one frozen acquisition
artifact, evaluation holdout artifact, or Common Crawl opt-out artifact. Each successful task writes
an immutable marker bound to the handoff hash. A reducer validates every marker and current file
stat, then writes `HANDOFF_VERIFIED.json`; normalization refuses to run without that exact marker.
Resubmission schedules only missing hash tasks. This avoids making the operator synchronously reread
the entire acquisition before Slurm can accept the build.

The explicit command below remains available as an optional synchronous diagnostic, but it is not
required before `submit build`:

```bash
METIS_LUSTRE_ROOT=/path-visible-on-rhea ./metisctl verify-handoff --profile rhea --deep
```

## The CPU build moved from Rhea to Portage

Rhea does not share a filesystem with `login2`, so the CPU build runs on Portage instead, under the
separate `portage-cpu` profile. Portage still never authors data with its training profile, and
`portage-cpu` never submits GPU work.

Portage has no CPU-only partition, so every allocation is a whole `parry` node with its four MI300A
accelerators idle. Because each stage task is a single-threaded process, `portage-cpu` sets
`scheduler.exclusive_nodes: true` and each array entry runs `tasks_per_job` task indices
concurrently inside its node. `max_concurrent` therefore throttles *nodes*, not work units, and
`METIS_TASK_LIMIT` stops a trailing partial group from entering the range owned by another
submission.

32 nodes is the standing recommendation. The wide stages finish in hours there; beyond it Lustre
metadata becomes the limit and the single-node reducers do not move at all.

```bash
export METIS_LUSTRE_ROOT=/lus/lustre1/vollmerc/metis-1.6
export METIS_SLURM_ACCOUNT=CONFIRMED
export METIS_SLURM_PARTITION=parry
./ops/collect-site-info.sh --lustre-root "$METIS_LUSTRE_ROOT" --role compute --partition parry
./ops/bootstrap.sh --profile portage-cpu --role compute --lustre-root "$METIS_LUSTRE_ROOT"
./metisctl doctor --profile portage-cpu --role compute
./metisctl verify-handoff --profile portage-cpu
./metisctl submit build --profile portage-cpu
```

`scheduler.site_values_confirmed` stays `false` until `collect-site-info.sh --role compute` has
confirmed cores per node, `RealMemory`, `SelectTypeParameters`, and `MaxArraySize`. If a `parry`
node reports 24 CPUs rather than the 96 expected from four 24-core MI300A packages, every
`tasks_per_job` in the profile must be divided by four.

### The tokenizer sample is built in parallel

`tokenizer_sample` is three stages rather than one pass over the whole eligible corpus:

1. `tokenizer_sample_scan` counts available bytes per source in each shard;
2. `tokenizer_sample_plan` fills the largest holders of each source first until its stratified byte
   target is met, and fails closed if the corpus cannot supply it;
3. `tokenizer_sample` writes only its planned slice to `tokenizer/sample-parts/`.

`tokenizer_train` reads those parts directly, so the sample is never concatenated back into one
file. Filling largest-holders-first rather than spreading each source proportionally is what keeps
the per-source overshoot to less than one document in total, matching the serial sampler, instead of
one document per shard.

### Verification is sharded

`verify_shard` re-hashes and re-audits one packed shard per task. `verify` then aggregates the
receipts, but only accepts a receipt whose `binding_sha256` matches the current selection hash,
token-count contract, tokenizer contract, and maximum document exposures. Anything stale or missing
is re-verified inline, so sharding cannot weaken the gate.

## Rhea remains intentionally sealed

Once Rhea exists, fill its confirmed Slurm values and verify that it sees the same Lustre directory.
Then bootstrap its own Python environment and submit the build:

```bash
export METIS_LUSTRE_ROOT=/exact/path-visible-on-rhea
export METIS_SLURM_ACCOUNT=CONFIRMED
export METIS_SLURM_PARTITION=CONFIRMED
export METIS_SLURM_MAX_ARRAY_SIZE=CONFIRMED
./ops/bootstrap.sh --profile rhea --role compute --lustre-root "$METIS_LUSTRE_ROOT"
./metisctl doctor --profile rhea --role compute
./metisctl verify-handoff --profile rhea
./metisctl submit build --profile rhea
```

After ordinary base selection, the dependency graph runs `context_select`,
`context_prepare`, 96 restartable `context_pack` tasks, and
`context_verify`. The result contains exactly 18B active Metis-tokenizer
tokens, compact 163,840-token rows, exact 6B/12B/18B gate tranches, and 384
training-disjoint 131K calibration records. A short source, cross-domain
fallback, token-count mismatch, or incomplete calibration set stops the
release.

Do not set `scheduler.site_values_confirmed: true` until those values are measured on Rhea. If Rhea
cannot mount the same acquisition content at any path, stop: a separately verified transfer/staging
backend is required before CPU preparation. Do not substitute Portage scheduler values.

## Context extension does not fit without both memory knobs

Context extension is the one Metis-1.6 stage where a step does not fit on an
MI300A. A physical pass at 163,840 tokens keeps four mHC streams alive per
layer, and pass-level activation recompute means every block's activations are
live simultaneously. **The figure is per rank**, which is the part that catches
people out: adding APUs replicates the problem instead of dividing it, so no
amount of data parallelism fixes it.

Two mechanisms do fix it, and the `context_extension.memory_strategy` block in
`configs/metis16/posttraining.yaml` is the source of truth for both.

**`activation_recompute_policy: layer`** replays one `Metis16Block` at a time
instead of one whole pass. The arithmetic is unchanged — every block is still
replayed exactly once, so the 8/6 replay factor in `estimate_hardware_flops`
covers it as-is and the throughput model needs no new term. What changes is
that one block's activations are live rather than all of them. Praxis fits on
this alone.

**`context_parallel_size: 4`** shards the sequence so each rank owns 40,960
contiguous tokens. Four matches the four APUs on a Cray EX255a node, and that
is not cosmetic: the group exchanges SSM state and gathered keys at every layer
of every pass, and the traffic has to stay on Infinity Fabric rather than
crossing Slingshot. Logos needs this; Praxis inherits it because the two
families share a launcher.

### What context parallelism actually costs

- **Attention is imbalanced.** Shards are contiguous because Mamba-2's SSD
  recurrence carries state strictly left to right and a rank needs an interval
  to have a well-defined incoming state. The consequence is that rank `r`
  attends over roughly `(r+1)/CP` of the sequence, so the last rank does
  `2·CP/(CP+1)` = 1.6x the mean work at `CP=4`. With 3 of 20 Logos layers
  attentional that is a step-time penalty in the high teens. Striping the
  attention layers alone, with a permutation all-to-all on either side, would
  recover it; that is a 1.7 item, not a 1.6 one.
- **Micro-batch is pinned to 1.** Continuation packing flattens
  `[batch, sequence]` into one row ordered by batch and then position, which
  interleaves batch rows across shards. The model raises rather than let rank
  `r+1`'s row-0 tokens inherit rank `r`'s row-1 SSM state. At 163,840 tokens
  one sequence already is the micro-batch, so nothing is lost.
- **Gradients are exact, not truncated.** The cheap implementation detaches the
  incoming SSM state and truncates backpropagation at every shard edge. For a
  run whose entire purpose is long-range dependency that is the wrong corner to
  cut, so every cross-rank tensor moves through a differentiable all-gather
  whose backward is a reduce-scatter.

### Launching

The launcher must place context-parallel ranks **inside** a node. The rank
layout is `replica -> expert -> context` with context varying fastest, so a CP
group is always a contiguous block of `context_parallel_size` ranks:

```bash
srun --ntasks-per-node=4 --gpus-per-task=1 ...
```

`build_parallel_topology` takes `context_parallel_size` and, for context
extension, an explicit `expert_parallel_size` — the stage runs on a different
rank budget than pretraining, so its EP/CP split comes from the manifest rather
than from the locked pretraining shape. World size must equal
`expert_parallel_size * replicas * context_parallel_size`.

Note that expert-shard gradients are averaged over **both** the replica and the
context axis. A CP rank holds a different slice of the sequence but the same
expert weights, so its gradient is as real as a replica's; `expert_data_size`
is the divisor, not `expert_replica_count`.

### Before trusting the run

1. **Measure peak HBM at the first gate.** The analytic estimate is
   activation-only and omits optimizer state and kernel workspace. Layer
   recompute plus `CP=4` should leave large headroom; confirm it rather than
   assume it.
2. **Run the fused Mamba parity check.** Context parallelism cannot use
   `Mamba2.forward` (it owns its zero initial state and offers no way to seed
   it), so `FusedMamba2._context_parallel_forward` drives
   `mamba_chunk_scan_combined` directly and replays the projection split,
   causal convolution and gated norm itself. That is exactly the code that rots
   across a `mamba_ssm` version bump, so
   `FusedMamba2.assert_context_parallel_parity()` runs both paths at `CP=1` and
   refuses the job if they disagree. **This has only been validated against the
   reference implementation on CPU — the fused path needs a Portage canary
   before the real run.**
3. **Confirm FlashAttention aligns causal masking bottom-right.** The CP key
   layout truncates each document's keys at this rank's last local query
   precisely so that bottom-right alignment coincides with the true causal
   mask. A build that top-left-aligns when `seqlen_q != seqlen_k` would let the
   first shard's queries read the future, silently.

Attention stays in BF16 throughout. FP8 attention is worth roughly 22% of CPT
wall clock, but its accuracy collapses at long contraction dimensions — and the
CPT promotion gate is itself a needle test over sealed 131,072-token records,
so the failure mode and the acceptance criterion are the same measurement. FP8
remains on for the QKV and output projections and every other role already in
`fp8_roles`.
