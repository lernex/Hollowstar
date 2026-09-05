# Metis-1.7: a deadline-aware 200 TB pretraining acquisition plan

Status: **research recommendation; not an executable acquisition lock**

Research cutoff and revision: **2026-09-04**

Training brief: **30T source-token exposures with TST bags of 16, then 5T ordinary NTP exposures**

Protected acquisition requirement: **approximately 25 TB compressed of recent 2026 Common Crawl**

This revises the acquisition research introduced in commit
`1d275538516d4790833b0a6e6d66c8476bdb9962`. The
[CSV ledger](metis17_200tb_acquisition_ledger.csv) records acquisition
allocations, payload selectors and evidence status. The
[dated source audit](metis17_data_frontier_source_audit.md) records primary
sources, alternatives and limitations. The
[1.7 preparation contract](metis17_data_prep_plan.md) applies the measured
1.6 lessons and defines verified-object prep concurrent with downloads.
The
[July TST note](metis17_tst_pretraining_data_plan.md) is historical research,
not the current token budget or source inventory.

## 1. Decision

**Use approximately 200 decimal TB as a staged acquisition envelope, not an
instruction to mirror every large dataset. Put 2026 freshness inside that
envelope, acquire scarce premium material early, and release the overlapping
bulk-web tail only when measured coverage needs it.**

The old basket had many good sources. Its largest weaknesses were allocation
and unsupported assumptions, not a lack of dataset names:

| Old plan | Revised decision |
|---|---|
| 53.418 TB full FineWeb plus 56.527 TB almost-full HPLT, before funding freshness | Current, metadata-rich web and language/quality-selected coverage take priority; overlapping tails are conditional |
| Approximately 25.6 TB of 2026 Common Crawl outside the core basket | A protected freshness allocation inside the budget |
| Every acquired source followed by blanket global near-deduplication | Separate physical copies, publisher overlap, within-snapshot duplicates, temporal versions and synthetic parents |
| A hard 42-45T globally unique-token completion gate | Prove the 30T and 5T exposure streams are feasible under explicit reuse rules; extra unique inventory is headroom, not an arbitrary launch blocker |
| New release dates treated as evidence of current knowledge | Track capture date, document date and novelty separately |
| Whole Stack v3 token count treated as the useful code supply | Separate 3.611T whole-corpus estimates from **159.320B publisher-classified permissive candidates** under the current quarantine policy |
| Derived KaTeX/TOON material presented prominently in acquisition planning | Preserve source structure; optional derived views do not consume a fictitious new-data budget or block acquisition |
| Ten days inferred from two 1GbE paths | Ten days requires 231.5 MB/s sustained payload and a compatible origin mix; plan against measurements |

**A material inventory error changes the budget:** the old "exact 200.022 TB"
ledger used truncated Hugging Face `siblings` listings for at least
Essential-Web and Dolma 3.5. At the same pinned revisions, complete
publisher-tree accounting gives **75.323 TB**, not 26.146 TB, for
Essential-Web, and **23.397 TB**, not 7.869 TB, for the intended complete
Dolma 3.5 pool. Dolma also contains Zstandard files that the old gzip-only
selector would omit. Correcting Essential and restoring the intended
complete Dolma pool raises the old basket to **264.727 TB before
fresh 2026 Common Crawl**; that is not the size of the unchanged, incomplete
gzip-only selector. Appending the four WET snapshots would bring it to
approximately **290.312 TB**, not 200 TB. The acquisition basket must be
rebuilt with bounded selectors; more download workers cannot fix this.

**No training mixture percentages are locked here.** Acquisition-byte shares
are not training-token shares. The scarce quantity is useful, permitted,
decontaminated data in the required domains, languages and quality bands.

**Operator constraints, clarified September 4:** full Nemotron-CC v2,
v2.1 and CC-Math acquisition; no bulk CC URL indexes or indexed WARC
repair; no GitHub/Software Heritage reconstruction; and preparation must
start from verified completed objects while other downloads continue.
These supersede the earlier optional repair/repository lanes.

## 2. The 35T contract and the five different quantities

Keep separate ledgers for:

1. **Transferred bytes:** actual compressed payload, metadata, retries and any
   materialization traffic. Deduplicate receipts so resumed files are not
   counted as newly acquired corpus.
2. **Logical text bytes:** decoded content after declared normalization, with
   metadata overhead reported separately.
3. **Accepted unique inventory:** distinct canonical text under the frozen
   Metis tokenizer, plus a separate estimate of near/parent-cluster diversity.
4. **Training exposures:** every selected occurrence, including deliberate
   replay, with source, phase and multiplicity.
5. **Processed model positions and measured compute:** TST bag positions are
   not ordinary source tokens and are not a FLOP measurement.

Preserve a canonical record envelope across those ledgers:
`source_id`, immutable revision/batch, object and row/record identity,
original and normalized hashes, canonical-content and occurrence IDs,
language/script, domain, capture/document dates with evidence, structural
and quality signals, organic/synthetic/translated status, parent and
generator lineage, license evidence, and document/repository grouping.
Do not throw those fields away at normalization and then attempt to infer
them again when choosing the curriculum.

### 2.1 TST arithmetic

The following is planning arithmetic, September 2026:

```text
TST source exposures              = 30.000T
bag size                          = 16
nominal TST positions              =  1.875T
ordinary NTP exposures/positions   =  5.000T
total source exposures            = 35.000T
nominal processed positions       =  6.875T
TST step fraction, equal batches   = 1.875 / 6.875 = 3/11 = 27.27%
```

Changing batch size, context, routing, loss implementation or padding changes
the compute interpretation. Do not call 6.875T a measured NTP-equivalent
quality or wall-clock budget.

The May 2026 [Nous TST experiment](https://arxiv.org/abs/2605.06546) reached
2T total source tokens at 10B-A1B scale, using a 50/50 DCLM/FineWeb-Edu mixture
in both phases. It does not establish a 35T scaling law or the outcome of a
broad-to-premium transition on Metis's architecture. The 5T ordinary phase
accounts for roughly 73% of nominal processed positions: **it is the main
recovery and capability-learning phase, not a small cleanup after training.**

### 2.2 Unique inventory is not exposure count

For a first-pass design within each phase:

```text
U_union = U_TST + U_NTP - U_shared_between_phases
```

If both phases use distinct documents internally, 30T plus 5T can require
between 30T and 35T in the union, depending on cross-phase reuse. Repetition
within a phase is another decision and must not be hidden in that identity.
Neither arithmetic nor the TST paper requires 42T or 45T unique tokens.

Preferred planning posture: first seek enough broad material for a largely
first-exposure 30T stream and enough premium material for a largely
first-exposure 5T stream. Keep measurable headroom where it is inexpensive.
If a scarce premium stratum needs replay, record the proposed multiplicities
and obtain capability evidence before freezing them. Never pad a shortage
with unannounced epochs, lower-quality text, benchmark-derived answers or
unbudgeted synthetic generation.

For every stratum, feasibility means
`planned_exposures <= accepted_unique_tokens * approved_max_multiplicity`,
with the actual sampler honoring the multiplicity distribution, not only
its average. Also report cross-phase reuse and synthetic-parent
concentration. A global total can pass while a small source is replayed far
more often than intended.

Recent repetition evidence is encouraging but conditional. The May 2026
[mixture-pretraining study](https://arxiv.org/html/2605.12715v2), also
highlighted by Apple in August 2026, finds that scarce domains can tolerate
more replay while generic data stays fresh. Its models were 101M-805M dense
Transformers, not a 35T TST MoE. Its reported 15-20 target repetitions are
**not** a proposed Metis cap. The August 2026
[domain-repetition study](https://arxiv.org/abs/2608.14071) likewise supports
domain-specific measurement, not a universal multiplier.

Count after eligibility, tokenizer choice and packing. Record input tokens,
loss-bearing targets, special tokens, padding and dropped tails separately.
The 16-fold bagging factor never divides acquisition bytes or raw token-ID
storage by sixteen.

## 3. Acquisition basket and release of the budget

The companion CSV distinguishes an **allocation ceiling** from an
**enumerated publisher inventory**. A subset budget is not a claim that an
exact file list has already been frozen. Rows requiring license approval,
schema checks or a bounded selector are not ready-to-run downloads.

Acquire in waves. Current curated anchors, scarce premium sources and the
freshness pilot go first. Conditional parent corpora and duplicate-heavy
tails go last. The September 5 clarification replaces the old unspent
reserve with actual payload: acquire approximately 200 TB while keeping
the highest-quality approved sources first. Final eligible yield remains
a separate measured condition.

### Full Nemotron means the entire content-bearing release

| Dataset | Complete pinned payload TB | Allocated TB | Included |
|---|---:|---:|---|
| **Nemotron-CC v2** | **10.333000437893** | **10.40** | All organic quality tiers, synthetic text, DQA and translated-DQA Parquet |
| **Nemotron-CC v2.1** | **4.590657078792** | **4.63** | All organic, rephrased, translated and DQA Parquet |
| **Nemotron-CC-Math v1** | **0.259937827291** | **0.26** | All `3`, `4plus` and `4plus_MIND` content |

All of these acquisition rows are **P1**, including the transformed and
medium-quality CC portions. The split rows sum to each whole repository's
payload; they are not instructions to stop after the organic/HQ portion.
These are core curated candidates, and synthetic content is not being
rejected merely for being synthetic.

Full acquisition still requires accepted terms and a real schema preflight.
Final licensing, contamination, quality masks and training weights are
separate decisions; acquiring every shard does not promise every token will
be used at equal weight.

### Revised September 5 contract: approximately 200 TB of actual compressed payload

The user explicitly requires **200,000,000,000,000 bytes of useful downloaded
payload**, including **25.7 TB of fresh WET**, not a 200 TB budget padded
with unspent reserve. Metadata, failed transfers and retry traffic are not
corpus bytes. Plain-JSONL sources count their actual wire payload; they are
not mislabeled compressed. This is a transfer-time target, **not** a 200 TB
limit on expanded text, intermediate artifacts or token IDs.

The previous plan had only **156.4681 TB of named primary/complementary
ceilings**, **34 TB conditional**, and **9.5319 TB unspent**. Calling that
200 TB of datasets was incorrect. The current executable source batches are:

| Activation batch | Candidate ceiling, decimal TB |
|---|---:|
| Seed `pipeline.yaml` | 46.7700 |
| `expansion-web.yaml` | 52.2400 |
| `expansion-complements.yaml` | 42.3940 |
| `expansion-target.yaml` | 66.4619 |
| All configured candidates | **207.8659** |
| Excluding blocked Darwin | **206.0659** |
| Release-wide actual payload target and cap | **200.0000** |

Resolve manual gates and new adapters on small objects, then let admitted
lanes move and prepare verified objects while the rest are downloading.

**CSV contract:** `inventory_bytes` is populated only for an exact inspected
inventory; `inventory_scope` says whether that is the full parent, a specific
subset or merely an observed growing prefix. Blank means unmeasured or
publisher-rounded, not zero. `budget_bytes`/`budget_tb` are allocations,
not observed downloads. Selectors are descriptive sets or semicolon-separated
patterns, not ready-made CLI arguments. The eventual lock must enumerate
their exact physical objects.

The current CSV removes the reserve, increases English HPLT from 7 TB to
**16.5319 TB of WDS10/9/8 ceilings**, and allows up to **32 TB** of the
reviewed FineWeb parent rather than 16 TB. Consequently its candidate
ceilings sum to **216 TB**, not 200 TB. They must not all be downloaded.
The live release-wide payload limit remains **200 TB**; high-priority
admitted sources precede the historical target-filling tails.

The seed, web, complements and target activation files together declare
**207.8659 TB of source ceilings**. Excluding Darwin's **1.8 TB** blocked
masked-checksum source leaves **206.0659 TB** before exact-object slack
and admission failures. This deliberate headroom replaces the unfunded
assumption that every old ledger candidate would automatically become
usable. It is not proof that 206 TB has passed preparation. Small sources
still needing adapter/rights review do not count toward this executable
pool. If another source fails, its bytes must be replaced explicitly
within the same 200 TB payload target, not reported as a completed corpus.

Bounded HF sources seal both selected and omitted object IDs. WET/HPLT
object sizes are learned on GET, without a full-object HEAD sweep; the
final acquisition closure must account for their completed whole objects
and omitted remainder. Global capacity can stop inside a source ceiling.
No incomplete file counts toward the target.

Small plain-JSONL candidates are labeled **raw**, not falsely described as
compressed; their full wire bytes count against the same envelope. Source
allowances cap selected payload. Retry/metadata traffic is reported separately
as elapsed-time/network overhead; it cannot fill a missing payload allocation.
It must not silently change the frozen selected-object list.

### 3.1 A byte cap must resolve to physical objects

For a capped source, pin the upstream revision, enumerate the complete tree
with pagination, stratify by publisher-supported snapshot/language/component,
and select whole objects using a deterministic seed and explicit priorities.
Record the selected IDs, sizes, skipped remainder and coverage by stratum.
Stop below the cap if the next object would exceed it; do not pretend a
partially downloaded Parquet file is a useful corpus shard.

**A document-quality filter does not automatically save network bytes.**
When quality labels are inside mixed Parquet shards, all necessary shard or
column bytes still cross the link. Only publisher-partitioned score bins,
available membership/annotation sidecars, or a measured projected/range-read
strategy can change that acquisition cost. Keep post-download filtering
separate from the physical selector.

Document-level pilots can inform the next object tranche, but must not be
used to claim a byte-perfect high-quality-only subset that the publisher
does not ship. Reconcile every selected physical tree against an independent
size/count source where possible; never use a truncated `siblings` response
as a complete inventory.

Subtract already cached objects from **remaining network bytes** only after
their identity, integrity, license and current removal status match the new
lock. Reuse a supported immutable import receipt; do not relabel a 1.6
artifact as 1.7 or reuse old token IDs with a different tokenizer. Report
gross selected payload, cache hits and actual transfer separately.

**Waves must not mean appending to frozen build inputs.** Keep exploratory
pilots and acquisition tranches under separate immutable batch identities,
with composable object receipts. Freeze the selected union when creating
the production data release. If the current tooling cannot import such
batches without rebinding existing work, implement that contract first or
use a new release identity for the next tranche. Never change
`build.inputs.json`, task enumeration or candidate-token fingerprints under
an in-flight release to make a late source fit.

### 3.2 Concrete web replacements and additions

These are acquisition choices, not assertions of equal information coverage:

| Change | Physical effect | Why |
|---|---|---|
| Buy the **new Ultra-FineWeb late-2025 HQ/L2 branch** before full L1 | **0.478 TB instead of 2.832 TB**; avoids 2.354 TB of parent transfer initially | August 2026 selected release through `2025-51`, with a small-scale matched-crawl result supporting a pilot |
| Pilot **Ultra-FineWeb `en_v1_4`** instead of making full FineWeb mandatory | **6.747 TB candidate instead of 53.418 TB parent** | Real December 2025 uploaded selection reaching `2025-26`; not present in the default loader config |
| Replace full Essential-Web mirroring with bounded snapshots or packaged specialists | **10.125 TB for 2023+2024**, or **2.446 TB packaged STEM**, versus 75.323 TB full pool | Physical selection saves traffic; filtering after downloading the entire pool does not |
| Acquire **all Nemotron CC v2 and v2.1** | **14.924 TB** complete payload within **15.03 TB** allocation | Organic/transformed partitions remain separate for provenance and later mixing, not acquisition exclusion |
| Use **Dolma's topic/quality partitions** rather than a guessed whole-pool size | Select real `cc_all_dressed/<topic>/vigintile_*` objects | Current physical partitions permit quality-aware transfer selection; mixture weights still require calibration |
| Add **CC-NEWS and the September Wikipedia export** | About **0.578 TB NEWS + 0.046 TB reference** | Later captures/canonical revisions at small cost, rather than relying on 2026 labels on old corpora |

The new Ultra-FineWeb HQ and v1.4 paths are in
`openbmb/Ultra-FineWeb` at
`02c85641e3d19a854be2e09139c25adaa9518063`. Their payload field is
**`content`**, not `text`; the new HQ branch's exact token count is not
published. Review the cards' redistribution restriction despite the Apache
license tag. The [source audit](metis17_data_frontier_source_audit.md)
contains exact bytes, selectors, pins, evidence and remaining gates.

FineWeb-Edu and DCLM remain useful controlled baselines, with explicit
May/August 2026 and March/May 2026 supporting experiments. They are not
mandatory full historical parents. A selected educational corpus must still
be paired with broad knowledge/language coverage.

**Correct the Nemotron token story as well as the bytes.** The v2 CC
repository's category rows total 5.8682T reported tokens, not the broader
release's 6.5858T including separately hosted categories. In v2.1, only
96.4B of 2.5448T reported tokens are organic English, from H1 2025; most
of the extension is synthetic rephrasing, including v2 parents. Neither
repository update dates nor new synthetic sequences establish 2026 facts.
That is an accounting distinction, not a reason to omit useful generated
data: the full v2/v2.1 payloads remain primary acquisitions.

### 3.3 Multilingual, PDF and specialist improvements

**HPLT can fund most of the fresh-crawl allocation without dropping a
language.** Its complete non-English payload is **24.293 TB**. Add a
**7 TB English ceiling drawn initially from WDS 9-10**, for at most
**31.293 TB**, rather than the old 56.527 TB selection. That frees about
**25.233 TB**. Keep language weights separate: this is warehouse optionality,
not a recommendation to train in those byte proportions.

Do not apply the same high-WDS cutoff everywhere. The April 2026 HPLT
report and audited inventories show sharp language differences; a global
WDS >=9 rule would retain only about 0.1% of Traditional Chinese bytes.
The highest scores are also not proven universally better than a diverse
sample. The English cap exploits redundancy elsewhere in this basket;
language-specific admission still follows measurement.

HPLT's manifest and current English object lengths disagree by **2.280 GB**.
Reconcile selected lengths and checksums before production; do not hide
the mismatch inside a rounded TB total.

| Improvement | Acquisition decision |
|---|---|
| **License-partitioned Dolma S2ORC science** | Acquire the exact CC-BY/CC0/MIT/public-domain suffixes: **88.887 GB**. The full 647.118 GB branch contains **502.205 GB of `null`-license files** |
| **Dolma non-web package** | Approximately **4.354 TB** of materialized code, Olmo-crawled PDFs, approved S2ORC, textbooks/wiki and math components; omit its HPLT and FinePDF-derived copies |
| **FinePDFs versus Edu** | Parent **5.374 TB** once; use its downloaded quality metadata locally and retain advanced technical material. No second Edu text or remote membership-ID acquisition |
| **Common Corpus** | Keep **4.489 TB** once with row-level provenance/license handling; OpenCulture is already inside it and much content is historical |
| **UltraData/Nemotron math** | Acquire all CC-Math `3`, `4plus`, `4plus_MIND` text plus the selected UltraData tiers; retain parent relationships without dropping MIND |
| **New small science/formal candidates** | French Science Commons, Math-v4, Science-v2 and actual Lean data are useful bounded additions, not missing trillions of premium text |
| **Published Dolmino/Longmino mixtures** | Optional small curriculum references; do not duplicate their ingredients or repeat a 100B recipe fifty times to manufacture 5T |

Dolma needs `.jsonl.gz`, `.jsonl.zst` **and** `.json.zst` component manifests.
A gzip-only snapshot omits the new PDF/textbook material. The exact
S2ORC selector uses suffixes such as `_cc_by.jsonl.zst`, **not `*cc_by*`**,
which would also match NC/ND licenses.

There are several source-adapter traps the previous ledger would not catch:

| Source | Actual payload / issue |
|---|---|
| New Ultra-FineWeb branches and L3 | **`content`**, not `text`; selected branches absent from the default loader |
| UltraData-Math | **`content`**; L2/L3 do not retain a complete per-row parent map |
| scholarweave/arxiv-latex | **`latex`**, not `latex_source`; selected/lossily decoded source files, not complete byte-exact archives |
| FinePhrase | **`rollout_results[0].text`** is the rewrite; `text` is the repeated original |
| French Science Commons | Rows are pages: group by **`id`**, order by **`page`** |
| Nemotron Legal-v1 | Apparent **CaseHOLD/summary content-label reversal** plus benchmark and placeholder issues; **defer the family**, not a name-based cleanup |

Specialized-v1 also has CC-BY-SA/GFDL exceptions, while some v1.1/v1.2
configurations derive from benchmark training tasks. These are per-source
admission gates, not paperwork to waive to meet the 5T target. Full evidence
and exact pins are in the [source audit](metis17_data_frontier_source_audit.md).

### 3.4 Code needs a licensed-content inventory, not another large name

The four existing code payloads really total **5.217 TB**; their byte
inventories reconcile. But Stack v3's **3.611T reported tokens are not
3.611T eligible code**. Its publisher statistics classify only **159.320B
tokens / 506.873 GB decoded text** as permissive; the remaining 95.59% of
decoded bytes are `no_license`. Under the existing quarantine policy,
start from that smaller candidate inventory and apply further gates.
Do not assume unreviewed `no_license` files are allowed, or infer their
compressed subset cost by multiplying a decoded-byte percentage.

Keep Stack's **3.55 TB ceiling conditional on a license-aware pilot**.
Prioritize materialized Dolma code/prose, Common Corpus code and admitted
Nemotron code pages as complementary sources, while measuring their own
licenses and overlap. A repository-grouped Stack row can have deduplication
holes and is not a complete executable checkout. Its September revision
still ends at **August 7, 2025 GitHub content**.

Useful absent additions include **Open-SWE-Traces** (42.600 GB compressed)
and **Nemotron-SFT-CUDA-v1** (82.8 MB raw JSONL), used only as shipped
code/tool transcripts. They do not authorize fetching task repositories,
missing files or execution images. Incomplete examples are rejected or
quarantined, not reconstructed.

SWE-rebench-V2 and its PR extension are removed from the funded plan.
Their patch/test text is real, but complete repository context and replay
depend on external checkouts. This conservative exclusion avoids another
small dataset turning into a large acquisition project.

In particular, all **107,267 Open-SWE v1.2 records have `resolved=-1`**:
unknown, not success. Its splits are harness names, not a generic `train`.
Canonicalize actual parent fields such as `hf_dataset_name` into recognized
genealogy metadata; preserve originals. Do not let a bool cast turn -1 into
"verified", or treat a zero-result success filter as proof the source lacks
content. Unknown status does not trigger a repository/image download.

The earlier 0.05 TB official GitHub archive lane is also removed. Current
software sources remain recorded as research in the audit, not active
download targets. Do not claim equivalent 2026 software coverage from an
older packaged dataset's newer upload date.

**No-reconstruction rule:** repo URLs and commit IDs may be provenance
metadata when actual code is inline. Pointer-only Nemotron metadata,
Stack-Edu/SWHID collections, missing Stack files, repository archives and
dependency/image fetches are not a fallback. Materialized code inside Dolma
and Stack's `files[].content` can be used as supplied after their other
gates. Prep workers do not fetch origin content.

## 4. The freshness layer: extracted text versus original web records

The relevant Common Crawl formats are **WET** (extracted text), **WARC**
(original capture records, including HTML responses), and **WAT** (metadata).
The requested "WAL" layer is treated here as recent Common Crawl content, not
SQLite write-ahead logs.

The bulk freshness lane uses complete selected WET objects for bandwidth
efficiency. This is a deadline-driven compromise, **not a claim that WET
matches frontier HTML extraction quality**. WET can lose tables, links,
code layout and mathematical structure. Scientific/coding premium content
therefore relies on faithful packaged sources. **There is no WET-to-WARC
repair lane in the default plan.** CC-NEWS uses whole already-downloaded
WARC objects, not per-page retrieval.

### Published snapshots and a genuinely newer addition

The four official release tables, inspected September 4, 2026, support:

| Official release | Crawl ID | WET TiB, publisher-rounded | Approximate decimal TB | WET objects |
|---|---|---:|---:|---:|
| [May 2026](https://commoncrawl.org/blog/may-2026-crawl-archive-now-available) | `CC-MAIN-2026-21` | 5.85 | 6.432 | 100,000 |
| [June 2026](https://commoncrawl.org/blog/june-2026-crawl-archive-now-available) | `CC-MAIN-2026-25` | 5.69 | 6.256 | 100,000 |
| [July 2026](https://commoncrawl.org/blog/july-2026-crawl-archive-now-available) | `CC-MAIN-2026-30` | 5.89 | 6.476 | 100,000 |
| [August 2026](https://commoncrawl.org/blog/august-2026-crawl-archive-now-available) | `CC-MAIN-2026-34` | 5.84 | 6.421 | 100,000 |
| **Total** | Four complete WET manifests | **23.27** | **25.586** | **400,000** |

Each manifest is
`https://data.commoncrawl.org/crawl-data/<crawl-id>/wet.paths.gz`.
Its entries are bucket-relative object keys. Complete manifest enumeration
confirmed 100 segments and 1,000 objects per segment for each snapshot.
**25.586 TB is converted from rounded release statistics, not an exact sum
of 400,000 HEAD responses.** A 25.7 TB allocation allows modest inventory
headroom. Freeze the selected keys before launch, then record actual object
bytes and checksums during GET completion; do not require a full HEAD sweep
before any useful transfer starts.

The latest published CC-MAIN collection at the cutoff is August, not
September. June/July announcement date ranges disagree with the
[collection catalog](https://index.commoncrawl.org/collinfo.json);
individual capture timestamps remain the authority for temporal selection.
Retain the discrepancy in provenance rather than inventing one date range.

**Add a 0.60 TB CC-NEWS lane.** The
[August 2026 archive](https://data.commoncrawl.org/crawl-data/CC-NEWS/2026/index.html)
reports 490 WARC objects, approximately 0.524467 TB. The
[September manifest](https://data.commoncrawl.org/crawl-data/CC-NEWS/2026/09/warc.paths.gz)
already contained 50 objects on September 4; HEAD enumeration summed them
to **53,152,407,851 bytes**. Together they are approximately **0.578 TB**.
This buys news captures into September, materially later than August
CC-MAIN, for little additional transfer. It is not comprehensive September
web coverage or proof every article was newly published.

Freeze the observed September list and checksum rather than following a
mutable monthly manifest during a release. Later additions require a new
explicit lock revision. The [source audit](metis17_data_frontier_source_audit.md)
records observed manifest digests.

**Bulk CC URL-index allowance: zero. Indexed WARC repair allowance: zero.**
The former 0.45/0.50 TB lanes are removed, not quietly renamed. The four WET
path lists total only **803,340 bytes**, about 0.8 MB; the revised ledger
allows **at most 1 GB** for these and bounded CC catalogue metadata.
This is an upper limit, not an expected gigabyte download.

The major 1.6 cost was also URL scanning, random-write ledgers and many
small requests. Removing that work is more important than the byte saving
alone. Prioritize NEWS, August, July, June, then May, and use object-level
receipts rather than the old URL-based freshweb builder.

### 4.1 Acquisition shape

- Resolve and hash each selected crawl's object path list once. Assign whole
  objects deterministically across approved acquisition hosts; record an
  exactly-once coverage report.
- Download sequential objects with resumable object-level receipts. Do not
  reconstruct billions of pages through a global URL-keyed SQLite database.
- Use bounded worker-local spools and approved local scratch where required.
  Sequential writes of verified objects to shared storage are different from
  random database I/O on Lustre.
- Keep capture IDs, record IDs, URLs, original hashes and extraction versions.
  Preserve **`WARC-Refers-To`** as parent provenance where available, not as
  an instruction to retrieve the parent.
- Do not fetch WAT, columnar/CDX URL indexes or source-page ranges to enrich
  records. Missing source content remains missing.
- Publish a verified completed-object receipt and enqueue prep immediately;
  do not wait for the whole crawl or whole source to finish.

The August 2026 [Datatrove reader](https://github.com/huggingface/datatrove/blob/a649de79c14a550dc90f48a15c025f2dd3fd3b57/src/datatrove/pipeline/readers/warc.py#L87-L140)
does not retain that parent header in its ordinary emitted record. A
downstream metadata adapter cannot recover an already discarded value;
the reader-level contract needs explicit work before the pilot.

Local normalization, quality assessment and decontamination run on approved
CPU workers while the next objects download. This is sequential content
acquisition feeding a prep queue, not a URL-reconstruction pipeline.

### 4.2 Quality and recency gates

Use a representative pilot across crawl segments, languages, document
lengths and domains, not the first few paths in a manifest. Assess the
actual WET text and already-downloaded NEWS HTML for corruption, missing
structure, quality, temporal novelty and preparation cost. Do not claim
source-HTML fidelity that WET no longer provides, or reintroduce WARC
lookup as a prerequisite for this pilot.

Use complementary education, knowledge, spam/template and structural signals.
News, current technical documentation and valuable reference pages need not
look like school textbooks. Do not let an education-only gate remove exactly
the fresh material this allocation was created to acquire.

Track at least:

```text
capture_timestamp
document_publication_timestamp
document_modified_timestamp
date_evidence_and_confidence
previous_capture_or_content_identity
is_new_to_selected_inventory
language_and_domain
extraction_method_and_quality
```

Report distinct token shares for **captured in 2026**, **reliably published or
updated in 2026**, and **new to the selected corpus**. They are not
interchangeable. Missing publication dates remain unknown; they are not
silently replaced with the crawl date.

For synthetic and translated documents, keep generation time separate from
the parent document's date. Rewriting a 2024 page in 2026 does not update its
facts. News-wire syndication, translations and lightly revised recurring
pages also need parent/cluster exposure accounting.

Protect a measured high-quality fresh stratum in the ordinary NTP candidate
view as well as in TST. Otherwise a fresh broad phase followed by a much older
premium phase can undermine the intended recency. The final amount follows
retained-token counts and temporal capability probes, not a guessed byte-to-
token conversion.

The May 2026 [Kyutai temporality study](https://arxiv.org/html/2605.22769v2)
makes this a concrete canary: compare a shuffled mix against a recency-aware
ordinary-NTP tail while preserving broad capability. Its sequential and
shuffled training ranges differ, so it does not prove that chronological
ordering alone is optimal. Its KairosQA release is an evaluation candidate
through 2025, not training data or sufficient coverage of 2026 events.

If the WET pilot disappoints, escalate the measured quality/coverage gap or
choose a justified ready-made fresh payload. Do not silently spend its
protected budget on old HPLT/FineWeb material, and do not fall back to
URL/GitHub reconstruction.

One useful initial design is one deterministically chosen WET object from
each of the 100 segments in each snapshot: **400 objects, approximately
25.6 GB at the published average size**, not 25.6 TB. Treat the object as
the sampling cluster. This is a proposed authorized pilot size, not data
downloaded by this research task or a sufficient sample for every rare
language. Start with a smaller format/receipt canary before expanding to
this representative pilot; NEWS samples use the complete objects already
selected, without index lookups.

### 4.3 A cheap reference refresh alongside the crawl

Add the completed
[September 2026 English Wikipedia current-content export](https://dumps.wikimedia.org/other/mediawiki_content_current/enwiki/2026-09-01/xml/bzip2/):
**19 objects, 46,445,603,916 compressed bytes**, fitting a **0.05 TB**
allocation. The publisher's `SHA256SUMS` was available and matched the
complete object list. This brings a current canonical reference source into
the ordinary-NTP candidates at very small transfer cost.

Use the new current-content export, not a blind `latest/` mirror, metadata
stubs or all-history files. Preserve revision/date/URL attribution and
faithfully handle wikitext, tables, math and template dependencies; exclude
discussion/user namespaces from the training view. The
[official licensing terms](https://dumps.wikimedia.org/legal.html) require
appropriate attribution and policy review. A current snapshot is not proof
every article is new, correct or independent of older web/PDF copies.

## 5. What the 5T ordinary phase needs

Build independently queryable premium pools, rather than labeling the last
5T of a broad stream "premium":

| Pool | Required distinction |
|---|---|
| High-quality general and fresh prose | Knowledge-bearing/educational/reference/news are separate labels; preserve broad language competence |
| Repository-aware code and technical documentation | Organic code, tests, documentation, issue/patch context and generated code are distinct; keep causal order and provenance |
| Mathematical and scientific documents | Source LaTeX, faithful extracted equations, explanatory prose, formal proofs and synthetic solutions are not interchangeable |
| Native long-form material | Real long documents and coherent repositories, not arbitrary concatenation presented as natural long context |
| Multilingual premium text | Measure language/script-specific quality and available tokens; an English classifier score is not a universal quality scale |
| Synthetic pedagogy and specialist tasks | Preserve parent, generator, prompt family, verification and benchmark genealogy; small SFT collections are not trillions of premium pretraining tokens |

The February 2026 [GLM-5 report](https://arxiv.org/html/2602.15763v2) and
February/August 2026 [Kimi K2.5 report](https://arxiv.org/html/2602.02276v2)
specifically motivate preserving repository context, issue/review/commit
relationships and scientific documents. Their proprietary corpora are
evidence for these capabilities, not available download capacity.

Within the fixed 5T total, distinguish **ordinary-NTP recovery**, **capability
and long-context consolidation**, and **final annealing**. Their boundaries
and domain weights remain experimental. Do not start learning-rate decay
merely because superposition ended; do not spend the entire recovery phase
on narrow synthetic QA; and do not count SFT/RL datasets as a substitute for
the full ordinary causal corpus.

Treat acquired code as untrusted. Parsing is not execution, and a
publisher's "verified" label is not a Metis execution receipt. Optional
local compile/test/proof work uses already available approved offline
tooling in isolation with resource limits and no production credentials.
It must not download repositories, dependencies or replay images.
Unavailable verification stays unknown or quarantined; it does not trigger
reconstruction or run inside a training/login host's working environment.

The May/July 2026 [MiniMax-M2 report](https://arxiv.org/html/2605.26494v2)
describes 19.9T constant-phase and 9.3T decay-phase tokens, including natural
long PDFs and code. It supports staged preparation, not copying its ratio
into Metis.

Keep JSON and original LaTeX/code. KaTeX-compatible or TOON views can be
derived later with parent links and round-trip/semantic checks. They create
additional training representations, not new independent facts or new
acquisition capacity. They are not on the download critical path.

## 6. Overlap policy: do not replace one silent failure with another

**The previous blanket global near-dedup instruction is withdrawn.**
Cross-publisher overlap is real, but it does not make every cross-snapshot
document version disposable. The policy is scoped:

| Relationship | Treatment |
|---|---|
| Duplicate physical objects, samples, repacks, removed/history trees | Avoid administrative duplication at acquisition; exclude publisher-removed material and evaluation views |
| Identical normalized content across sources | Store a canonical content identity with all provenance/occurrences; prevent accidental double allocation |
| Near-duplicate English pages within a crawl/selection cohort | Use a measured within-cohort policy with retention audits |
| Similar English pages across crawl dates | Track temporal clusters and meaningful changes; do not globally delete them by default |
| Multiple languages or translations | Preserve language/script and translation parent; do not collapse different language realizations into an English-only survivor |
| Code copies, forks, formatting and repository context | Use code-aware identities and license-aware handling without destroying a coherent repository example |
| Source LaTeX, rendered/OCR paper and revised paper versions | Prefer faithful permitted representations; preserve document/version relationships rather than treating every rendering as new knowledge |
| Synthetic rewrites or generated solutions from one seed | Exact text identity plus parent-cluster accounting; explicit caps/replay, not fictitious independent coverage |

Physical canonicalization does not dictate the training exposure distribution.
A document may be stored once and deliberately revisited. Its occurrence and
phase counters must make that choice visible.

Quality-aware winners must be deterministic and independent of download
order. Preserve usable license evidence, source fidelity, document
completeness, quality signals and date evidence; resolve ties with stable
IDs. Never let the fastest download host choose the corpus.

No single education classifier, semantic embedding threshold or prose
MinHash setting is appropriate for every language, formula and code file.
Evaluate removals by language, domain, length, date and synthetic parent,
including documents that should survive. Report both unique storage and
planned exposure effects.

Do not send every document to a frontier model. Reuse trustworthy publisher
scores where their semantics fit; calibrate lightweight selectors on a
stratified labeled sample and reserve expensive adjudication for ambiguous
or specialist cases. Measure selector throughput and the processing
allocation before committing to rescore tens of billions of documents.
An ensemble that cannot finish before the release is not an operational
quality gate.

## 7. Decontamination and licensing are admission gates

Keep all configured evaluation-only material out of **both** training phases,
including reformulations, answer repositories and generated descendants.
TST is not a defense against benchmark leakage, privacy leakage or
memorization.

The current [holdout registry](../manifests/contamination/eval-holdouts.yaml)
is split-specific: some evaluation datasets publish their evaluation records
under a split named `train`. In addition,
[benchmark genealogy matching](../src/metis_data/decontaminate.py) builds
dataset aliases without a split exemption. Thus an explicitly benchmark-
derived synthetic source can be rejected even when it advertises training
seeds. Do not confuse either rule with automatically indexing every split of
every benchmark.

Freeze a 1.7 holdout inventory before large-scale normalization. Record
source/seed exclusions before generating derivatives. Preserve genealogical
metadata through every stage; exact text matching will not recover ancestry
once a question has been rewritten.

Apply the measured 1.6 length-bias lesson before the first 1.7 submission:
the proposed 1.7 policy sets short-ngram and code-skeleton matching to
**zero**, retains exact/normalized, 13-gram, contiguous-run, code-overlap and
genealogy protections, and re-measures known-copy detection and retention
by length/language/domain. This is not a TST-specific relaxation. The
checked-in 1.6 profile still records `4`/`32` with a **NOT YET APPLIED**
warning; those historical values are not changed. See the
[prep policy and round-trip contract](metis17_data_prep_plan.md#6-decontamination-carry-forward-the-actual-lesson).

Dataset-level licenses describe the compilation or release; they do not
automatically clear underlying web pages, repositories, papers or generated
outputs. Before admitting a source:

- archive the exact terms and source revision;
- distinguish compilation license, document/repository license and generator
  obligations;
- verify the row or collection actually carries the evidence the configured
  policy demands;
- quarantine unresolved/no-license material under the approved policy;
- apply publisher opt-outs, removals and takedown updates; and
- distinguish permission to train/release a model from permission to
  redistribute the corpus.

Apply privacy, secret/credential and disallowed-content screening as well;
public availability and an acceptable compilation license do not replace
those gates. Publish aggregates and provenance, not raw source samples or
unnecessary personal metadata.

Where a web derivative supplies only compilation-level terms, the admitted
policy must explicitly say whether that evidence is acceptable and record
the remaining underlying-rights uncertainty. Do not declare
`per_record_required` for a corpus that cannot supply those fields, and do
not "repair" the resulting zero yield by inventing document licenses.
Repository/paper collections with genuine per-item evidence remain subject
to their stricter applicable policy.

Do not automatically accept a gated agreement, reclassify an absent license,
or weaken the inherited policy to make a token target pass.

## 8. The measurement that decides whether another terabyte is worth buying

Run a bounded, stratified pilot before releasing a large optional tranche.
Use immutable object IDs and a fixed seed; include high/medium/low score
bins, all relevant dates/languages, and both short and long documents.
Small schema probes and representative yield measurements are different
steps; ten convenient rows cannot forecast a 50 TB corpus.

For each stratum, record:

```text
physical objects expected / received / verified
transferred payload / metadata / retry bytes
logical text bytes and source rows
payload, license and parse acceptance
quality rejection reasons
exact-content and near/parent-cluster overlap
decontamination removals and genealogy exclusions
retained frozen-tokenizer tokens
premium-eligible and TST-eligible tokens
2026 capture / verified document-date / novel-content tokens
length distributions and code/math structural validity
normalization, scoring, dedup and tokenization throughput
source/phase proposed exposure multiplicities
```

Use byte-/population-weighted estimates and uncertainty bands, not an
unweighted average of differently sized shards. Measure marginal yield
against the **already selected** corpus. A high standalone-quality score can
coexist with almost no additional useful coverage.

An acquisition tranche earns expansion by adding needed, permitted,
decontaminated tokens or a valuable missing stratum per unit of transfer and
preparation time. There is no scientifically justified universal acceptance
percentage; establish expectations from real samples and surface deviations.
Zero output must distinguish legitimate rejection from a broken schema,
unimplemented setting or missing measurement.

Freeze the tokenizer before authoritative counts. Use artifact-level
behavioral evidence for code whitespace, mathematical notation, multilingual
fertility and digit handling; a YAML flag alone is not proof.

## 9. Download calendar: ten days is a best case, not a commitment

Planning arithmetic, September 2026, with decimal TB and MB:

| Sustained aggregate payload | Time for 200 TB |
|---:|---:|
| 250 MB/s: physical 2Gb/s ceiling, no overhead | 9.26 days |
| 231 MB/s | 10.02 days |
| 200 MB/s | 11.57 days |
| 180 MB/s | 12.86 days |
| 167 MB/s | 13.86 days |
| 125 MB/s | 18.52 days |
| 110 MB/s | 21.04 days |

The August 2026 [local pipeline measurements](metis17_data_pipeline_lessons.md)
recorded an earlier 231 MB/s dual-login run. The
[acquisition incident record](metis17_data_prep_lessons.md) later observed
approximately 110 MB/s aggregate for Hugging Face and 167 MB/s across two
paths against another origin. These are different observations, not a
guaranteed rate or proof that one diagnosis applies to every publisher.

At 110 MB/s, ten days can transfer only **95.04 TB from HF**. The old basket's
erroneous 143.495 TB HF subtotal would alone take **15.10 days**, even if
HPLT downloaded in parallel. Correcting Essential-Web and restoring the
intended complete Dolma pool increases that old HF inventory to
**208.200 TB**, before adding freshness. The revised plan must track publisher volumes and
measured rates, not just a global 200 TB sum.

For the superseded reserve-inclusive CSV, named allocations contained **132.8171 TB of HF
payload/support**, **57.651 TB from non-HF origins**, and **9.5319 TB of
unassigned reserve**. At the historical HF ceiling, the named HF portion
alone needs **13.97 days**. If the entire reserve also becomes HF transfer,
that bound rises to **14.98 days**, before downtime and preparation.
Excluding P3's historical tail reduces the HF allocation to **98.8171 TB**,
or **10.40 days** at that ceiling. These are conditional transfer bounds,
not measured current performance.

On **September 5, 2026**, a concurrent 120-second measurement after fixing
origin scheduling observed **121.186 MB/s RX on login1** and **118.826 MB/s
RX on login2**, with **229.979 MB/s useful compressed-payload progress**.
This is practical saturation of both 1 Gb/s external links, not a promise
of 250 MB/s application payload. At that sustained payload rate, 200 TB
takes approximately **10.1 days of transfer**, excluding source-specific
slowdowns, interruptions and remaining preparation. Sharing HF on both
hosts is necessary to avoid stranding an uplink in a long HF-only tail.

After sharing **HF, CC and HPLT on both hosts**, the later September 5
120-second sample measured **119.875 MB/s** and **121.350 MB/s RX**,
respectively, with **zero RX drops or errors**. Both downloaders had
96 active transfer slots. The fully sealed payload counter excludes
still-growing partial objects; it must not be used alone as an instantaneous
download-rate measurement.

The P0+P1 priority tranche has **63.23 TB of HF allocation**; the other
57.651 TB uses independent origins. Full Nemotron inclusion is now explicit
in this priority tranche. It can make useful progress substantially
earlier than a blind all-family mirror, but whether it supplies enough
accepted tokens remains a measurement, not a promise.

Use:

```text
calendar lower bound >= max(
    all transferred bytes / measured aggregate payload rate,
    each origin's bytes / that origin's sustained rate
)
```

Two NICs do not automatically aggregate for one process or flow. Confirm
default routes and approved independent acquisition hosts. Use official
alternate endpoints only when they identify the same permitted immutable
objects; do not rely on unverified mirrors or bypass publisher controls.
Track failures, retries and object tails as additional time.

Use bounded retries with publisher `Retry-After` handling and persist
partial/completed-object state. If one origin slows or access is denied,
pause that lane and continue independently admitted work; do not silently
substitute a different dataset revision, mirror, or lower-quality source.

Before accepting the login links as a hard limit, check whether the site
offers an approved data-transfer node or a scheduled higher-bandwidth path.
That is a potentially larger calendar improvement than any change to
download concurrency. Availability, quota and origin throughput must be
confirmed; this plan assumes no new machine, paid cloud budget or site
network change.

Overlap independent origins and pipeline stages. Do not overlap conflicting
writers to the same object, saturate login-node CPUs with scoring, or consume
training allocations without approval. A smaller high-value first wave can
be ready before the full conditional 200 TB envelope is spent.

## 10. Prepare verified objects while downloads continue

**Download/prep overlap is a 1.7 requirement, not an optional optimization.**
The [detailed preparation contract](metis17_data_prep_plan.md) maps the
1.6 throughput design and measured lessons to per-object receipts,
rolling CPU work, reusable artifacts and explicit final barriers.

```text
download -> verify one object -> publish RAW_READY -> enqueue CPU prep
              next objects keep downloading          |
                                       normalize / score / decontam /
                                       signatures / stable-text IDs
                                                     |
                                 sealed global decisions -> final pack
```

| Can overlap download | Must wait for its actual prerequisite |
|---|---|
| Per-object extraction, normalization, metadata and quality work | That object's integrity receipt and applicable policy |
| Holdout/index preparation and record-level decontamination | The fixed evaluation bundle/index, not all training downloads |
| Signatures and sorted runs | Final winners/frequencies wait for a closed comparison scope |
| Representative tokenizer sample collection | Tokenizer freeze waits for all required sample strata |
| Cached uint32 IDs for stable text | Frozen tokenizer; redo only text versions changed by later cleaning |

Final source/phase selection, packing and release require sealed eligible
inventory and exact accounting. Do not make arrival order choose dedup
winners or use partial global results as final counts.

The [August 2026 preparation lessons](metis17_data_pipeline_lessons.md)
identify serial whole-corpus passes, repeated JSON decoding/tokenization,
selection that rewrites text, and incompatible cleanup/resume requirements.
Buying faster networking does not repair those failure modes.

Required 1.7 design:

- Columnar metadata with projection; shard-level aggregates for reducers.
- One frozen-tokenizer pass, durable token IDs and document offsets.
- Selection and curriculum views as indices, not copies of all selected text.
- Bounded partition-local work and scratch; no corpus-global random-I/O
  SQLite ledger on Lustre.
- Explicit byte/row-balanced tasks so a giant shard cannot determine the
  entire stage tail.
- Data/policy identity separated from throughput knobs; immutable release
  inputs and recorded stage-code provenance.
- Retain source inputs until supported verified retirement can be resumed.
  A cleanup stage must not delete what a future submission requires.
- Independent imported-object receipts for data prepared elsewhere.

At the proposed vocabulary scale above 65,536 IDs, budget for uint32:
**35T token IDs alone occupy 140 decimal TB**. TST still needs the constituent
IDs; embeddings change during training, so pre-averaged vectors are not a
substitute. Reuse indexed views rather than duplicating IDs for every phase.

Do not reserve only 200 TB of disk for 200 TB of downloads. Measure:

```text
peak storage = retained raw
             + live normalized representations
             + token-ID store and offsets
             + active dedup/sort/index scratch
             + checkpoints/receipts/retry headroom
```

Illustrative, not measured: retaining 200 TB raw, 400-800 TB of expanded
staging, 140 TB of IDs and 100 TB scratch would require 840-1,240 TB.
Keep normalized data compressed and process in waves where that preserves
resumability. Shared-filesystem free space is not a confirmed quota or an
IOPS guarantee.

Before any deployment or acquisition run, inspect `squeue`, `sacct`, current
logs, scratch availability and the source lock. This revision changes no
runtime defaults, manifests, active jobs or frozen Metis-1.6 inputs.

### 10.1 Minimum engineering before this becomes executable

This document does not create a working `metis-1.7` manifest, downloader,
normalizer or TST packer. The existing 1.6 validator still enforces a 65,536-
entry tokenizer, among other generation-specific contracts.

The current CLI requires `download.build_ready` before build submission
even with the handoff flag disabled. `prepare_build_inputs` also requires
every download task complete, and normalization reads that frozen global
input list and handoff-bound opt-out policy. A separate 1.7 ready-object
dispatcher and per-object policy/input contract are required. Do not
simulate overlap by disabling guards, appending to `build.inputs.json` or
writing a false `ACQUISITION_READY.json`.

Reuse the existing
[`source_lock._iter_repo_files`](../src/metis_data/source_lock.py) for
paginated, revision-pinned tree enumeration and allow/deny selection; it
already avoids `siblings` for corpus inventory and uses `expand=False`.
The undercount diagnosed here was in the research ledger, not proof that
the current resolver uses that same broken enumeration.

There is a separate budget-contract gap:
`source_lock._select_files` currently treats `target_bytes` as a **minimum
to reach**, includes the object that crosses it, and lets `take_all=True`
select the whole matching family. These CSV allocations are **ceilings**.
Do not feed them into that interface and claim a hard cap. A 1.7 selector
must add explicit maximum-transfer semantics and independently reconcile
the full physical object list, without changing in-flight 1.6 behavior.

| Before | Required work |
|---|---|
| Bulk transfer | Isolated 1.7 release/root, admitted terms, complete content-object locks, resumable transfer and per-object readiness receipts |
| Concurrent prep | New ready-object dispatcher, independently frozen policies and immutable work manifests; no global-acquisition bypass |
| Expanding a new source | Correct adapter for shipped row content, no external reconstruction, explicit empty/error handling and a representative end-to-end pilot |
| Corpus-scale preparation | Scalable metadata/index views, scoped dedup, round-tripped quality/genealogy fields, explicit worker/scratch capacity |
| Authoritative token inventory | Frozen 1.7 tokenizer and supported token-ID dtype; no reuse of incompatible 1.6 token IDs |
| Training emission | Separate 30T bagged and 5T ordinary streams, exact counters, explicit repeat policy and boundary/masking safeguards |

Content acquisition can precede final mixture weights and tokenizer training,
provided its source identity and receipts are sound. That is how to get ahead
without mutating a live release or waiting for every downstream feature.

## 11. Release gates and the order of work

| Gate | Evidence required before advancing |
|---|---|
| Source admission | Immutable revision, exact physical selector, actual inline content, satisfiable license policy, approved access |
| Pilot | Representative quality/yield/overlap/date measurements; prep starts before the final download completes; no URL/GitHub hydration |
| Acquisition lock | Frozen object membership and available upstream sizes/checksums; explicit bounded unknowns, actual GET receipts and exactly-once assignment before final handoff |
| Preparation | Reason-counted outputs, scoped dedup, full decontamination, final-tokenizer counts and metadata round trips |
| Curriculum feasibility | Emittable 30T TST and 5T NTP streams; explicit overlap/repetition; premium/domain/language/freshness coverage |
| TST readiness | Matched-regime canary, phase-boundary continuity, document-aligned packing, masked padding/tails and target-shift correctness |
| Final release | Replayable manifests, valid receipts, no unsupported cleanup dependency, exact exposure accounting and disclosed provenance limitations |

An ETag is not universally a cryptographic checksum, particularly for
multipart objects. Record what the publisher actually provides and distinguish
identity/version evidence from integrity evidence.

Every partition must cover each selected unit exactly once. Tests for
document-aligned bags must catch a bag or target crossing unrelated document
boundaries, incorrect duplicate-target weighting and loss on padding. Fast
implementations must be pinned to readable references with differential
tests, not merely exercised.

The source enumerator also needs a regression fixture larger than an API
page: intentionally truncate the final page or omit a directory and require
inventory reconciliation to fail. The Essential-Web/Dolma undercount is
exactly the repository's "every processed shard verifies, but some were
never assigned" failure mode at the acquisition boundary.

The immediate sequence is: admit and pin sources; start protected freshness
and premium pilots; acquire current core payloads while measuring
preparation; freeze the tokenizer and count; expand only the strata still
missing. Do not wait for an exhaustive speculative mixture search before
acquiring already justified content, and do not launch all 200 TB before
testing the paths that failed in 1.6.

**Stop expanding the optional bulk tail when the exposure streams and
protected freshness/premium coverage are supported.** If they remain short,
escalate that measured shortage; do not rename downloaded bytes "tokens" or
silently turn a one-pass plan into repeated training.
