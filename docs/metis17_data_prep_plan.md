# Metis-1.7 data preparation: verified objects, parallel download and prep

Date: **2026-09-04**

Status: **implementation contract for a new 1.7 path, not a running pipeline
or a capability already provided by the 1.6 CLI**.

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

Once frozen, cache IDs by **final text hash + tokenizer hash + tokenization
policy**. At a vocabulary above 65,536, budget uint32 and test high token IDs.
Record document offsets and exact token counts alongside IDs. Selection,
replay and TST/NTP views reference these IDs rather than re-encoding text.

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
# Proposed 1.7 policy, not a modification to the 1.6 profile.
decontamination:
  minimum_matching_ngrams: 2
  minimum_short_matching_ngrams: 0
  minimum_code_matching_ngrams: 16
  minimum_code_skeleton_matching_ngrams: 0
  match_fraction: 0.002
  contiguous_run_minimum: 8
```

Retain exact/normalized matching, the 13-gram core, contiguous-run evidence,
12-gram code overlap and explicit genealogy. The proposed zero values are
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

## 9. Implementation gap: why this is not one existing flag

| Current code | Required new 1.7 work |
|---|---|
| [`download._download_hf_file`](../src/metis_data/download.py) returns verified file size/SHA; `run_download_task` publishes its completion after all task items | Reuse verification and emit object readiness immediately in the new path; keep 1.6 task semantics intact |
| [`state.atomic_json` and task locks](../src/metis_data/state.py) provide publication/coordination primitives | Scope receipts and locks to immutable object/batch identities; add efficient ready-list discovery |
| [`cli.cmd_submit` / resume](../src/metis_data/cli.py) require `download.build_ready` even when the handoff flag is disabled | Add a separate verified-object ingestion/dispatch path; do not bypass the final build gate |
| [`prepare_build_inputs`](../src/metis_data/build_inputs.py) requires every download task complete and rejects changed frozen inputs | Use immutable batch/object work manifests, then an explicit final sealed union |
| [`_normalize_task`](../src/metis_data/stage_runner.py) reads the global build input list/contract and final opt-out policy through handoff | Reuse/extract normalization logic behind per-object inputs and independently frozen policy artifacts; never fabricate the global handoff |
| [`BUILD_GRAPH`](../src/metis_data/slurm.py) chains whole stages; decontamination index is late | Separate ready-object work from comparison/release barriers; build holdouts/index independently and early |
| Counting/packing and selection still embody 1.6 representations | Implement reusable IDs, projected metadata and indexed views under new generation contracts |

This also needs an actual 1.7 manifest/profile and supported vocabulary/dtype
contract. The static-content path must not require enabling the legacy
`dynamic_materializers_enabled` gate merely to prepare packaged objects;
use an explicit allowed-driver contract instead. No guessed launcher
command is supplied here: a command is only
documented as runnable after these surfaces are wired and demonstrated.

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

These are implementation acceptance criteria, not results already obtained
by writing this document. The present change updates the acquisition and
prep plans only; the live 1.6 pipeline remains unchanged.
