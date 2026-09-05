# Metis-1.7 data preparation: verified objects, parallel download and prep

Date: **2026-09-05**

Status: **separate 1.7 implementation with overlapping acquisition and
preparation**. The new `metis_data17` path does not modify the 1.6 build
graph. The target is approximately **200 TB of actual compressed payload**,
including **25.7 TB fresh WET**, not a reserve-inclusive budget. Expanded
text, intermediates and uint32 IDs have a separate working-storage limit.
Candidate ceilings exceed 200 TB so blocked sources and whole-object slack
do not silently shrink that target; this is not a completed-download claim.

The [acquisition plan](metis17_200tb_pretraining_corpus_research.md) and
[200 TB ledger](metis17_200tb_acquisition_ledger.csv) define the candidate
data. This document defines how to prepare it without repeating the 1.6
critical path. Training remains **30T source-token exposures with TST bags
of 16, then 5T ordinary NTP exposures**.

## 1. Decisions that are no longer optional

1. **Acquire the complete content-bearing Nemotron-CC v2, v2.1 and CC-Math
   releases** at their pinned revisions, after access/terms and schema
   admission. Organic, synthetic and translated rows are accounting
   partitions, not instructions to omit part of those releases.
2. **No bulk Common Crawl URL indexes, CDX scans, per-page WARC lookup or
   indexed repair in the default acquisition.** Download complete WET
   objects and the selected complete CC-NEWS WARC objects.
3. **No GitHub/Software Heritage reconstruction.** The training payload must
   already be in the acquired object. Do not fetch repositories, missing
   source files, dependencies or replay images to make a record usable.
4. **Prepare each verified object while other objects are downloading.**
   No whole-corpus acquisition barrier before independent preparation.
5. **Do not confuse preparation progress with a releasable corpus.**
   Global selection, comparison-dependent decisions and final release still
   require a sealed inventory and complete evidence.
6. **Do not change the live 1.6 profiles, locks, artifacts or jobs.** All new
   behavior belongs to isolated 1.7 inputs and an explicitly implemented path.

### Full Nemotron coverage

| Dataset | Complete pinned payload TB | Allocated TB | Acquisition |
|---|---:|---:|---|
| Nemotron-CC v2 | **10.333000437893** | **10.40** | All organic quality tiers, synthetic, DQA and translated-DQA Parquet |
| Nemotron-CC v2.1 | **4.590657078792** | **4.63** | All organic, rephrased, translated and DQA Parquet |
| Nemotron-CC-Math v1 | **0.259937827291** | **0.26** | All `3`, `4plus` and `4plus_MIND` content |

These are strong curated candidate sources. A synthetic label is not a
low-quality verdict. Preserve it to distinguish provenance, repetition,
date and phase use. Full acquisition does not bypass licensing, benchmark
decontamination or final mixture decisions.

Enumerate each `(repository, revision)` once and reuse that catalogue for
its provenance partitions. Their selected-object union must equal the
complete intended payload with multiplicity one. Match exact organic
directory names: a prefix check must not accidentally exclude
`High-Quality-Synthetic` together with `High-Quality`.

For CC-Math, `3plus` is the union of `3` and `4plus`; its 133B reported
organic tokens already include the approximately 52B `4plus`. MIND adds
approximately 73B derivative tokens, not an additional organic crawl.
The actual text is supplied; no Common Crawl reconstruction is needed.

## 2. Apply the measured lessons, not every historical instruction

Read together:

- [May 2026 throughput design](metis16_data_prep_throughput_plan.md):
  independently scheduled source/shard jobs, a local cache of downloaded
  objects, idempotent outputs and manifest aggregation.
- [July 2026 data plan](metis16_pretraining_data_plan.md):
  provenance, source-specific quality, contamination controls and exact
  final-tokenizer accounting.
- [August acquisition lessons](metis17_data_prep_lessons.md):
  pointer/index bottlenecks, unsatisfiable license declarations, source
  fingerprint traps and operational failures.
- [August pipeline lessons](metis17_data_pipeline_lessons.md), especially
  sections 0, 0b, 10, 14, 16 and 17:
  restart safety, inert settings, length bias, serialized passes, redundant
  tokenization and repeated whole-corpus verification.

The May design's English-only filter, uint16 assumptions, pod topology and
speculative speedups are **not** the 1.7 specification. Its useful cache
means downloading an actual content shard once, not reconstructing a
pointer corpus. Likewise, the older instruction to start build-shaped
sources first is superseded here: **exclude that acquisition shape**.
Some August failure modes were subsequently fixed; reuse those fixes rather
than describing every historical defect as still live.

## 3. Keep the network path simple

### 3.1 Allowed versus excluded acquisition

| Allowed | Excluded from this plan |
|---|---|
| HF Parquet/JSONL with actual text, code, proofs or complete usable transcripts | Repo/commit/path/SWHID-only corpora and metadata presented as source code |
| Materialized code inside Dolma, or nested `files[].content` in Stack | Resolving missing Stack/Nemotron/Software Heritage file contents |
| Complete selected WET and CC-NEWS objects | CC columnar URL indexes, CDX scans and WET-to-WARC range repair |
| Small object path lists, pinned file manifests, schema and license metadata | A corpus-global URL-keyed SQLite ledger on Lustre |
| Shipped code/tool transcripts used as supplied | Re-cloning task repositories or fetching Docker images to replay them |
| Local canonical-content, token-offset and contamination indexes | Interpreting the no-CC-index rule as permission to remove decontamination |

Repository URLs and commit IDs may remain **provenance metadata** when the
content itself is inline. They must not trigger network requests. An
ordinary import or reference in a source file is not itself a pointer-only
corpus; the question is whether external retrieval is needed to construct
the intended training example.

Recorded tool commands inside a transcript are data, not instructions for
the prep worker to execute. Source parsers/renderers must not fetch external
resources implicitly. This corpus path needs no GitHub token merely to
rehydrate code; deploying the Metis software itself is a separate activity.

The earlier six-project GitHub archive lane and SWE-rebench/V2-PR
acquisitions are removed from the funded plan. SWE-rebench contains real
patches, not merely pointers, but its complete repository context and replay
environment are external. This is a conservative deadline decision, not a
claim that its patch files contain no code.

Open-SWE, CUDA and Lean candidates remain only as **shipped-content views**.
Reject/quarantine examples that need missing external context; do not fix
them with a repository fetch. Unknown execution status stays unknown.
Any optional local proof/static validation must use already available,
approved offline tooling; it cannot create an acquisition dependency.

### 3.2 Small manifests are not the old Common Crawl indexes

The four selected WET path lists total **803,340 compressed bytes**, about
0.8 MB. The ledger allows **at most 1 GB** for path lists and bounded CC
catalogue metadata, not a 1 GB download requirement.

Do not perform a 400,000-object HEAD sweep before starting WET transfers.
Freeze the path lists, use their published approximate archive sizes for
planning, and record each object's actual `Content-Length`/bytes/checksum
as its GET completes. Sample-probe when needed; do not turn catalogue
resolution back into a page/request-count project.

HF inventory still needs complete pagination, `expand=False`, and
independent size/count reconciliation. Cache that inventory once. Prep
workers must not each resolve the Hub or fetch the same object again.

FinePDFs premium views use the quality metadata in the downloaded parent.
The optional remote Edu-membership projection is removed as another
unnecessary dependency; these local views are not claimed to be the exact
published Edu subset.

WET fidelity is an explicit compromise. Reject or downgrade damaged
code/math/table text instead of following it back to WARC. Prefer the
already acquired Nemotron math/code and PDF/scientific sources for
high-fidelity premium material. CC-NEWS HTML is extracted locally from the
complete WARC object already downloaded; no external index is required.

## 4. The producer/consumer path

```text
frozen acquisition batch + source/format admission
                         |
              two approved download hosts
                         |
             temporary object -> integrity check
                         |
               immutable RAW_READY receipt
                         |
                bounded prep-work catalogue
                         |
        CPU workers: extract / normalize / metadata / features
                         |
             quality + fixed-policy decontamination
                         |
          candidate text + signatures + per-object counters
                         |
         text-stable records + frozen tokenizer -> token IDs
                         |
           sealed comparison inventory / final eligibility
                         |
              select indices -> pack views -> release

holdout/opt-out preparation -----^
stratified tokenizer sampling --------------------^
```

The first finished and verified object can enter prep while the remaining
source, other datasets and later download batches remain in flight.
**Never normalize a growing `.part` file.**

### 4.1 Publish readiness per object, not per source

A `RAW_READY` receipt must bind at least:

```text
acquisition_batch_id
source_id + immutable upstream revision
object_id + original object key
relative durable path
actual bytes + local SHA-256
upstream checksum/identity and validation result, when available
source format / adapter version
completion identity
```

The object is durable before its receipt is atomically published. A
downloaded filename alone is not readiness. A byte-size-only marker is not
equivalent to a deep integrity receipt.

Use the existing verified-download and atomic-publication mechanisms where
they fit. Compute hashes during acquisition where supported, or reuse the
existing verified hash. Do not add a full-corpus serial hash scan between
download and prep. Where an independent deep read is required, scope it to
the object before its consumer; one shard task must never verify the whole
dataset.

A downloader restart reuses correctly verified completed objects. A
normalizer restart reuses valid output receipts. Duplicate delivery or
execution is tolerable; duplicate **publication or token accounting** is
not. Conflicting output hashes for the same content/policy identity are an
error, not a last-writer-wins update.

### 4.2 Batch the scheduling, not the correctness

Do not submit one Slurm job per WET file. Use rolling byte/row-balanced
microbatches or an approved persistent CPU-worker allocation.

Each work manifest is immutable and names its exact objects/ranges. Receipt
identity must depend on those inputs, not only a rank such as `task-000000`.
Changing worker count, array throttle or polling interval must not move
already published data to a different logical task.

Amortize discovery using per-producer sealed ready-list segments and a
checkpointed cursor. Do not recursively rescan hundreds of thousands of
files on every scheduler tick. The catalogue is object-scale bookkeeping,
not a URL/document-level random-write database. Durable object receipts and
sealed work manifests remain authoritative; a lost scheduler cache must
be reconstructible.

Keep any mutable high-IOPS work state on approved local scratch. Do not
assume either login node has NVMe. Sequential shared-filesystem object
reads/writes are permitted; a random-I/O SQLite fallback on Lustre is not.
If an algorithm requires scratch that is unavailable, stop that stage with
a capacity error instead of silently relocating its database.

### 4.3 Split once, without repeatedly decoding the same archive

- Use Parquet row groups or explicit record ranges, preserving whole logical
  records, to expose real parallel work.
- Decode non-seekable gzip/Zstandard objects once into bounded normalized
  chunks. Do not use repeated full-container scans as the 200 TB default.
- Keep unusually large indivisible documents in an explicit jumbo lane with
  measured memory/walltime requirements; do not truncate them silently.
- Batch small inputs by work. A file count is not a byte/row/CPU-cost model.

The current `_split_oversized` modulo scheme is coverage-safe but has each
part read and decompress the full container. It was a sensible 1.6 tail
mitigation; it is not the preferred 1.7 bulk path.

Document completeness can impose a source-local barrier. For French
Science Commons, parsing pages can overlap download, but a paper cannot be
published as complete until its page set is closed and ordered. Stack's
publisher-provided repository rows are used as supplied; upstream
deduplication holes are recorded, not filled from GitHub.

## 5. What can actually overlap

| Work | During download? | Required boundary |
|---|---|---|
| Integrity/provenance for an object | Yes | Its complete transfer and applicable upstream integrity checks |
| Extraction, format normalization and metadata repair | Yes | Verified object; admitted source adapter; no external reconstruction |
| Privacy/license/hygiene and quality features | Yes | Frozen applicable policy; source-specific metadata semantics |
| Holdout preparation and contamination-index build | Yes, independently | The small evaluation-only bundle, not the training corpus |
| Record-level decontamination | Yes | Correct frozen holdout index and matching policy; changed text must be checked again |
| Exact/near/code/span signatures and sorted intermediate runs | Yes | The text version each signature describes |
| Final duplicate winners, span frequencies and comparison-dependent retention | Only for a closed comparison scope | Every selected input in that scope; never first-arrival-wins |
| Tokenizer sample collection | Yes | Representative, predetermined source/domain/language coverage |
| Tokenizer training/freezing | Potentially early | All required sample strata and their eligibility/provenance gates |
| Bulk token-ID caching | Yes, after tokenizer freeze | Text is stable, or explicitly provisional with later invalidation |
| Final mixture, exact exposure selection and packing | After required inventory closure | Final eligibility and tokenizer counts; no silent source shortfall |
| Final release | No partial release claim | Complete manifests, integrity, policy, coverage and 30T/5T accounting |

An unfinished global dedup pass does not prevent all useful preparation.
Produce immutable features/signatures and reusable text/ID artifacts early;
delay the decisions that depend on seeing later data.

Conversely, independent preparation is not proof of global uniqueness.
Keep provenance and occurrences when canonicalizing exact storage copies.
A later higher-fidelity or license-preferred occurrence must not lose just
because it downloaded second. Cross-snapshot near-duplicate retention follows
the acquisition plan's explicit temporal/scope policy, not an accidental
microbatch boundary.

### 5.1 Tokenizer timing and the meaning of "tokenize once"

Choose representative sampling strata and a deterministic sample-object
schedule before download order biases the sample. Fetch the needed sample
objects early; require code, mathematics, long documents and intended
languages to be represented. Do not freeze a tokenizer trained only on
the first fast English dataset.

No bulk IDs are produced with a disposable provisional tokenizer. While
waiting for the final artifact, continue normalization, scoring, signatures
and eligible sample collection.

The production artifact must contain **131,072 vocabulary entries, including
the seven configured special tokens**, with individual digit splitting
serialized before byte-level pretokenization. A YAML flag alone is not proof
of that behavior. Small test tokenizers are explicitly non-production; do
not pad an undersized learned vocabulary with dummy entries.

Once frozen, cache IDs by **final text hash + tokenizer hash + tokenization
policy**. Use little-endian **uint32**, not the 1.6 uint16 representation.
Record document offsets and exact token counts alongside IDs. Selection,
replay and TST/NTP views reference these IDs rather than re-encoding text.
Offsets count token elements, not bytes. Mutable cache/sampling indexes use
bounded, verified node-local scratch; the durable IDs and receipts stay on
shared storage. Cache reuse is partition/session scoped. Separate workers
may encode the same content independently; no global cross-worker
tokenize-once guarantee is implied.

The production training sample is **150 GB of usable UTF-8 text**, with
**30 GB in each required web, code, math, science and multilingual category**.
Whole documents may produce a bounded, reported overshoot; holdout bytes
cannot satisfy the training requirement. The explicit recipe also requires
six named curated sources and native Arabic, Simplified Chinese, German,
French, Japanese, Russian and Spanish coverage. These are sampling gates,
not claims that every incoming document is representative.

The live acquisition `RUN.json` was frozen with the old 160 GB setting.
Do not edit it. Seal `configs/metis17/tokenizer-recipe.yaml` separately
against that exact RUN and eligibility generation. Sampling, training and
later ID-cache admission must use the same recipe identity. Freeze the
actual admitted sample input set and provenance; hashing only that set is
not a guarantee of identical samples under different arrival schedules.

Document deletion, changed sampling weights or a different dedup winner can
often be expressed as metadata masks over unchanged text/IDs. But repeated-
span removal, repaired extraction or other text edits invalidate affected
signatures and IDs. Do not call stale pre-cleaning IDs final. Either delay
encoding those records until their text-changing stages finish, or report
and redo only the affected cache entries.

Late rights/opt-out changes also require correct downstream invalidation.
If a prohibited record was used to train the tokenizer and the applicable
policy forbids that derived use, a row-removal mask alone is insufficient.
Do not promise every late policy change is free.

### 5.2 Do not append to frozen 1.6-style build inputs

Acquire through separately pinned batch identities and object receipts.
The final selected union is sealed only after the chosen batches are
accounted for. P2/P3 budget reservations are not automatically required
inputs: activate a new batch explicitly when more coverage is needed.

Never append to an in-flight `build.inputs.json`, retarget an existing
fingerprinted source, or forge `ACQUISITION_READY.json` while downloads
remain incomplete. The final release barrier must cover every selected
object exactly once and explicitly account for legitimate policy rejection;
an unprocessed or failed download is not a rejected document.

## 6. Decontamination: carry forward the actual lesson

The benchmark inventory remains evaluation-only in **both** phases, with
configured splits plus split-agnostic declared genealogy. TST is not a
reason to weaken those protections.

The August pipeline lessons, section 14, recommend disabling short 8-gram
and code-skeleton matching because they disproportionately removed long
documents. The checked-in
[1.6 CPU profile](../configs/metis16/portage-cpu.yaml) explicitly says
**NOT YET APPLIED** and retains `4` / `32` to describe the r2 run honestly.
Do not edit that history or claim those settings were already off.

For the new 1.7 profile, the starting policy is:

```yaml
# Implemented 1.7 starting policy, not a modification to the 1.6 profile.
decontamination:
  minimum_matching_ngrams: 2
  minimum_short_matching_ngrams: 0
  minimum_code_matching_ngrams: 16
  minimum_code_skeleton_matching_ngrams: 0
  match_fraction: 0.002
  contiguous_run_minimum: 8
```

Retain exact/normalized matching, the 13-gram core, contiguous-run evidence,
12-gram code overlap and explicit genealogy. The zero values are
based on measured false positives, **not** a claim that bagging prevents
contamination. Re-measure known-copy detection and removal rates by length,
language and domain on real 1.7 samples before freezing this policy.

Separate the immutable benchmark inventory from tunable detection policy.
The loader must report and round-trip the actual float thresholds and
disabled families. An index built under one policy cannot silently be read
under defaults. Changing matching policy invalidates the relevant index/
eligibility artifacts, not raw downloads or unrelated extraction.

The local contamination index is still required. Removing network CC URL
indexes must not remove this independent integrity/control stage.

## 7. Lesson-to-contract checklist

| 1.6 failure / evidence | Required 1.7 behavior |
|---|---|
| Pointer corpora and crawl builders consumed days while packaged data moved easily; acquisition lessons 1-2 | No reconstruction driver in the admitted source plan; prove real payload fields before scaling |
| Unsatisfiable per-record license declarations; acquisition lessons 4b | Test the actual row/collection evidence against the declared policy; no invented licenses |
| A successful file list omitted unenumerated data; new source audit | Complete paginated manifests and independent counts/sizes; test missing final page and duplicate assignments |
| `cleanup_raw` destroyed resume prerequisites; pipeline lesson 0 | Retain raw inputs by default; no automatic cleanup that invalidates a future resume; supported retirement receipts only |
| Tuning changed fingerprints and orphaned progress; acquisition lesson 3 | Worker counts and throughput controls do not alter data identity; no in-place candidate retargeting |
| Unrelated code/scheduler changes invalidated completed work; pipeline 0a/6/13 | Output-sensitive stage contracts; explicit total stage-code mapping; preserve valid object caches |
| YAML/character loops dominated per-record time; pipeline 1/1b | Cache policy/model loading per worker; profile the real hot path before replacing algorithms |
| Large files and 80.9x job-size variation caused tails/timeouts; acquisition 6 and pipeline 12/17h | Row/byte-balanced work, one-pass reblocking, measured jumbo lane and RAM-aware tasks per job |
| JSON decode and text rewriting serialized selection; pipeline 16/17 | Columnar metadata projection, indexed corpus views, parallel reducers over aggregates rather than all document text |
| Counting discarded IDs and packing encoded again; pipeline 17c | Durable tokenizer-keyed ID cache; exact counts and packing from the same artifact |
| Every verify task re-read every shard; pipeline 17d-e | Each worker verifies its own inputs/outputs; a separate aggregate verifies the set once, in parallel where needed |
| Settings survived config files but never reached readers; pipeline 0b | Round-trip source fields, thresholds, dtype and tokenizer policies; fail on unimplemented declared behavior |
| Long valid material was over-filtered; pipeline 14 | Proposed short-ngram/code-skeleton off for 1.7; known-copy controls and length/domain/language retention reports |
| A legitimate empty file stopped the whole graph; pipeline 10b and current normalize code | Explicit valid-empty receipts; unexpected source/config-wide zero yield pauses that lane and cannot satisfy final coverage |
| Failures hid tracebacks or appeared under stale logs; acquisition 5 and pipeline 0b | Object/config/row-aware failures, live run-bound progress, nonzero failing cases before trusting zero errors |
| Slurm moved wrappers and interpreters; acquisition 6 | Reuse exported `METIS_ROOT`/`METIS_PYTHON`; test the spooled-script path; requeue safe failures without whole-stage repetition |
| Fixed-width/large-ID and inert whitespace declarations; pipeline 17i | Frozen supported 1.7 dtype/tokenizer; artifact-level code/math round trips, not unimplemented YAML promises |
| Shards/s suggested a healthy job was slowing; pipeline 17f | Progress and ETA in actual bytes/rows/tokens and oldest queued work, not raw task count |
| Weak fixtures missed replay/fallback bugs; pipeline 17g | Differential tests with multiple shards, replay, deficits and aliases; mutate each relevant rule and require failure |

Counters must distinguish a completed empty shard, a missing field, access
denial, corrupt input, an unimplemented adapter and a source genuinely
exhausted after filtering. Do not wait until hundreds of TB have arrived to
notice a systematically empty source.

## 8. Capacity, backpressure and the calendar

Downloads run on the two approved egress paths. CPU preparation runs in a
separate approved allocation with the correct shared-storage view, not on
busy login nodes or by borrowing resources from live MI300A training jobs.
Confirm actual memory/scratch/array limits; historical Rhea and Portage
settings are not interchangeable.

Processor runtimes and any scoring-model artifacts must already be
provisioned under the approved pinned runtime contract. Per-record workers
do not install packages, pull images or download model weights on demand.

Bound ready-input bytes, active decoded bytes, output staging and per-node
RSS. Compute worker count from both CPU and memory budgets, including
native-library/compression threads. Backpressure slows intake or dispatch
when storage/processing falls behind; it never deletes unretired inputs.

Track:

```text
payload bytes received / retries / remaining budget
verified objects and pending/failed object IDs
ready raw bytes / oldest ready age
source-input-byte-equivalent prep rate
normalized rows and text bytes / rejection reasons by source and size
signature backlog / comparison scopes open or closed
tokenizer sample-stratum completion / frozen tokenizer hash
token IDs produced / cache hits / invalidated text versions
remaining global merge and final release work
```

Compare compressed source bytes processed with compressed source bytes
downloaded when asking whether prep keeps up. Decoded MB/s and network
MB/s do not have the same denominator.

The intended critical path is approximately:

```text
max(download time, overlapping per-object prep path)
    + unresolved global decisions
    + remaining tokenization/selection/packing/verification
```

This includes pipeline warm-up/drain, tokenizer readiness, shared CPU/I/O
contention and source-local assembly barriers. It is not a promise that all
prep disappears into the download window. For illustration only, 12 days
download + 7 days independent prep + 2 days final work is roughly 21 days
serially; sufficient overlapping resources can approach 14 days plus
warm-up/drain instead. Measure the actual rates before using that as an ETA.

## 9. The separate implementation and its remaining boundaries

| Surface | New implementation and contract |
|---|---|
| [`common.py`](../src/metis_data17/common.py), [`acquisition.py`](../src/metis_data17/acquisition.py) | Stable content-object identity, HTTP range resume, integrity-bound RAW_READY receipts, per-source/global intake reservations; retained raw files |
| [`catalogue.py`](../src/metis_data17/catalogue.py) | Complete paginated HF inventories, HPLT JSONL/MD5 manifests and small CC object-path lists; independently sealed source counts/bytes |
| [`prep.py`](../src/metis_data17/prep.py), [`prep_readers.py`](../src/metis_data17/prep_readers.py) | One-pass streaming reblocking, inline-content adapters, immutable READY chunks and exact document coverage at EOF |
| [`policy.py`](../src/metis_data17/policy.py), [`prep_policy.py`](../src/metis_data17/prep_policy.py) | Separately frozen benchmark/opt-out inputs and 1.7 matching thresholds; cached policies and verified memory maps loaded once before each node forks |
| [`worker.py`](../src/metis_data17/worker.py) | Incremental per-producer journals, process-safe object claims, small raw-reader pool plus independent chunk work, explicit failures and measured source-canary admission |
| [`dedup.py`](../src/metis_data17/dedup.py) | Metadata-only exact occurrences/winners and geometric sorted-run compaction; higher configured quality priority wins independently of arrival order |
| [`dedup_signatures.py`](../src/metis_data17/dedup_signatures.py) | Actual scoped MinHash/span/code signatures; signature production is not completed near-duplicate deletion |
| [`tokenizer.py`](../src/metis_data17/tokenizer.py) | Artifact-validated 131,072-entry digit-split tokenizer, stratified sampling and bounded local-scratch uint32 ID caches |
| [`tokenizer_pipeline.py`](../src/metis_data17/tokenizer_pipeline.py), [`tokenizer_service.py`](../src/metis_data17/tokenizer_service.py) | Sealed recipe, eligible-only sampling, one training owner, generation-bound readiness and replay-safe partition caching |
| [`runtime.py`](../src/metis_data17/runtime.py), [`prepare.sbatch`](../slurm/metis17/prepare.sbatch) | Commit-pinned CPU workers on idle Slurm nodes, explicit interpreter/environment, low-priority supervision and bounded restart behavior |

The streaming producer publishes `part-*.READY.json` while scanning a raw
object. Workers can filter those chunks immediately, but the result remains
`ELIGIBLE_PENDING_OBJECT_COMPLETION` until the producer seals complete
coverage at EOF. Promotion reuses the screened data. Only immutable
`ELIGIBLE.json` receipts with `eligible=true`, `training_ready=true` and
complete object evidence may enter deduplication or tokenizer sampling.
`FILTERED.json`, normalized chunks and mutable current-state aliases are
not equivalent eligibility evidence.

Receipt hash domains are explicit: `stage_receipt_sha256` means the
canonical payload seal, `digest_json(read_receipt(path))`. A full JSON-file
checksum has a separately named field. Parquet `sha256` is always the
full-file checksum. These domains must never be accepted interchangeably.

Eligibility and reducer generations are separate. A reducer-only change
rebuilds its metadata/signatures without changing the saved eligibility
generation or tokenizer input identity. Failed work records the worker
implementation and capacity identity, so a corrected worker or explicitly
expanded capacity can retry without deleting valid inputs or completions.

Quality-prioritized acquisition does not itself implement quality-aware
deduplication. The exact index retains all occurrence provenance and ranks
winners by configured source/partition priority, known record quality and
stable tie breakers. A later better occurrence can become the winner.
Unknown quality stays unknown; this comparator is not an undisclosed
learned quality model.

**September 5 compaction repair:** profiling the actual old workers found
`ingest_eligible -> compact_dedup -> metadata_lock`, with active-object slots
blocked and most CPUs idle. There were three synchronous maintenance sites,
including a hidden WorkingBudget pre-publication guard. Opt-in
`--defer-compaction` bypasses all three while retaining bounded within-batch
sorting, quota enforcement, occurrence leaves and atomic publication.
The first pilot exposed a second bottleneck: publishers still held every
affected bucket lock, while one maintenance lane could not keep up. The
real index reached **217-229 active runs per bucket**, and fresh stack
traces again showed `_publish_batch -> metadata_lock`.

Deferred publishers now append under the short publication transaction
without waiting for the long bucket merge. A compactor re-reads that
transaction's current run set and replaces only its pinned inputs, retaining
concurrent arrivals. Per-bucket leases let independent nodes compact
**disjoint** buckets; a durable round-robin cursor covers the complete
partition. Each node reserves one maintenance process slot, so backpressure
cannot occupy every process needed to relieve it. Maintenance normally
amortizes at least eight equal-tier runs and performs one merge per lease.
Publication waits explicitly at **256 active runs per bucket** rather than
letting fan-in and control-file size grow without bound. Maintenance
continues while a stopping worker drains admitted work.

This does not dedicate extra nodes or make first arrival the winner. Only
the exact-index generation changes; normalization and eligibility artifacts
remain reusable.

A subsequent profile of the **actual publication-lock owner** found the
same batch manifest being JSON-encoded and hashed again for each of its
64 buckets. Verification now caches the digest and run lookup **only
within one locked publication transaction**, while checking every bucket's
individual pin. On the same locked **21,865-run live index**, the old
view took **16.46 seconds**, versus **8.44 seconds** for the cached view
run first; the complete returned run sets were identical. This is not a
persistent cache that could hide later receipt changes.

The fleet startup also exposed eager per-node receipt discovery: every
coordinator reread every `RAW_READY` receipt before assigning any work.
With almost 29,000 objects, all worker pools sat idle for several minutes.
Discovery now decodes the already-sealed event journal; the claimed reblock
task verifies the actual immutable receipt **before reading source text**.
Negative source-admission lookups are shared within each scheduling tick.
This scheduling-only change does not change either data generation and is
for subsequent launches; already-working processes need not be restarted.

Publication admission uses a separate kernel-queued lock. Only one ingester
contends with the independent compactors for the shared transaction; the
rest do not create a distributed-lock polling storm. Existing immutable
index-layout validation also avoids that write-lock queue. Acquiring it
again for every batch would defeat the admission separation.

All six declared decontamination thresholds were compared with the actual
sealed live index on September 5 and matched. Startup now rejects a
disagreement or an unknown declared knob instead of accepting an inert
configuration. The shared source index and 1.6 jobs are not changed.

The existing **10% canary admission floor is only a gross-failure guard**,
not proof that retention is correct. Source/language/domain/length retention,
long-document behavior and raw-WET learned-quality selection still need
their explicit quality evidence. In particular, a WET compliance pass
remains quality-selection-pending, not training eligible.

Index metadata is not another copy of the corpus. Real Dolma PDF metadata
includes full alternate document text in `pre_sa_key` and `no_references`;
one observed 407k-character paper carried more than 1 MB of repeated
metadata. Index records inline at most 64 KiB and otherwise retain a sealed
prepared-row metadata reference. `dedup.read_reference_metadata` resolves it
with checksum/row validation and reads only the metadata column. The source
text, eligibility decisions, winner comparator and original metadata remain
unchanged; long papers are not dropped to satisfy an index-size heuristic.

**Still separate from this launch:** unreviewed small candidates in the
larger ledger, learned/raw-web quality selection, closing near/span comparison
scopes, final mixture/replay feasibility, and wiring final selection and
packing to the new ID references. No partial index is described as a final
35T training release. Unknown crawl scope does not authorize global
cross-snapshot English near-deletion.

## 10. Evidence required before the 200 TB run

- Full Nemotron family partitions cover every intended object exactly once,
  including transformed directories with names similar to organic ones.
- A code record with inline content and a repository URL remains eligible
  for source admission; the same record with only pointers fails without
  attempting a fetch.
- A tiny multi-source run in which prep starts after the first verified
  object **and before the last download finishes**.
- No prep read of a partial/unverified object and no origin/GitHub/SWH
  network calls from prep workers.
- Retry, duplicate delivery and coordinator/worker restarts preserve
  exactly-once published coverage and reuse completed downloads.
- Finalization refuses any missing selected object, failed comparison
  scope or stale policy/tokenizer artifact; legitimate empties are explicit.
- A late higher-priority duplicate produces the same final selection
  independent of arrival/worker order.
- Page-grouped sources cannot become incomplete documents silently.
- The tokenizer sample meets its frozen strata; a changed text/tokenizer
  invalidates the right ID cache and replay does not re-encode unchanged text.
- Effective decontamination values survive index save/load; disabled
  families stay disabled and known prohibited copies still fail.
- Per-worker verification is scoped, and bounded queues/RSS remain within
  the approved storage and compute allocation on real representative objects.

These are acceptance criteria, not results obtained merely by writing this
document. Local tests and live startup evidence have different scopes; neither
establishes the final corpus's quality or a 10-11-day completion guarantee.
The live 1.6 pipeline remains unchanged.

## 11. Bounded startup and operational interpretation

The initial activation in
[`configs/metis17/pipeline.yaml`](../configs/metis17/pipeline.yaml) contains
eight source families: complete CC-Math, CC-Code, the permitted Dolma science
view, complete CC-v2 and CC-v2.1, HPLT English WDS10, HPLT non-English, and
August 2026 WET. Complete Nemotron catalogues are not permission to exceed
the current intake ceiling. English WDS9, the other WET months, NEWS and
the remaining ledger selections are not silently added to frozen `RUN.json`.

The initial limits are **400 GB raw and 2 TB total working storage**, with
`capacity_confirmation: pending`. Lustre reporting petabytes free does not
prove the user's inherited/default quota. These are a bounded start while
the full-run capacity is unresolved, not a reduction of the 200 TB plan.
No administrator confirmation is inferred from an old 1.6 log.

Acquisition uses the independent `ens2f3` 1 Gb/s routes on login1 and login2:
HF on the first, independent HPLT/CC origins on the second. In-flight
partials precede new work; new work is quality ordered, with independent
origins allowed to fill otherwise idle bandwidth. A source/object that
cannot fit a reservation does not stall smaller admissible objects from
other sources. Capacity limits are not raised automatically.
Known objects that cannot fit are postponed before making a network
request. Unknown-size objects need conservative headroom before another
probe; in-flight reservations remain resumable. A full intake budget must
not turn into thousands of pointless HF/CC requests that resemble active
payload downloading.

Two canary objects per admission group are initially permitted. Successful
whole-object screening and at least the configured 10% acceptance are
needed before opening the group's bulk lane. Zero yield, missing source
URLs needed for opt-out enforcement, or pending quality selection remain
visible reasons for review, not reasons to fabricate an admission marker.
An admitted source still undergoes document-level screening on later data.

The supported entry point is `python -m metis_data17.cli`. Its `init`,
`resolve`, `download`, `import-policy`, `prep`, `supervise-prep` and `status`
commands belong to this separate path. Slurm receives the immutable
`METIS17_CODE`, `METIS17_ROOT`, `METIS17_PYTHON` and `METIS17_WORKERS`
values explicitly; it does not inherit an HF credential via `--export=ALL`.
Only owned 1.7 services may be restarted. Do not pull into or modify a live
1.6/MoRE checkout.

Two real startup failures are now covered by the launcher:

- Run Python with `-B` and `PYTHONDONTWRITEBYTECODE=1`. On September 5,
  `py-spy` found the first canary blocked in importlib `_write_atomic` on
  a Lustre lock, before processing any corpus data. Disabling bytecode
  writes allowed the same committed canary to finish.
- Preserve the virtualenv launcher's absolute path, **not its resolved
  symlink target**. Resolving `runtime-login2/bin/python` selected the system
  interpreter and lost the qualified packages in the first Slurm rollout.
  Runtime paths are now recorded with each submitted job; corrected
  code/runtime pairs have independent restart-storm guards.

CPU workers use renewable 12-hour allocations, stop accepting new objects
before the walltime boundary, and relinquish idle allocations rather than
holding otherwise usable nodes indefinitely. Slurm backfill can take
longer than 20 seconds even when `sinfo` reports idle nodes; a short
`srun --immediate` failure is not proof that compute is unavailable.

Use `status/download-*.json`, `status/prep-*.json`, sealed event journals,
Slurm job logs and actual interface deltas together. A live Screen session,
an old progress file or a completed source catalogue alone does not prove
that bytes are moving or preparation is keeping up.

### September 5 startup evidence

The isolated release root is `/lus/lustre1/vollmerc/metis-1.7`. Code is
deployed into commit-addressed checkouts beneath `code/`; the existing
1.6/MoRE checkout is not a deployment target.

Slurm canary job **495905**, using `8d539a4` with bytecode writes disabled,
accepted **31 of 36** real documents: science `5/5` and `24/29`, and two
HPLT language canaries `1/1` each. Its source admissions opened the science
download lane automatically. This established the real policy/format path,
not corpus-scale throughput.

The corrected continuous deployment is **`51c4b36`**. Jobs **495920,
495922, 495923 and 495924** were started on four available nodes, each with
32 chunk workers and two raw-reader slots. At **07:25:44 UTC**, the first
worker had completed 109 objects and indexed 264 documents while
acquisition remained in progress. These were initially tiny canary objects;
their objects/second must not be extrapolated to large HPLT archives.

The 400 GB raw / 2 TB working ceiling is still in force. Both user and
group quota reports explicitly inherit defaults, whose inspection requires
administrator permission. The machine's free-space report is not an
approval to expand this bounded start to the complete 200 TB plan.
