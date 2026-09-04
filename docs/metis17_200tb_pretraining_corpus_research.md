# Metis-1.7 pretraining corpus: a quality-first 200 TB acquisition plan

Status: **research recommendation; acquisition and training mixture are not locked**  
Research cutoff: **2026-09-04**  
Current training brief: **30T source-token exposures in TST at bag size 16, then 5T ordinary NTP cooldown**

This document uses the current 30T + 5T brief. It does not silently carry forward the older
24–25T TST + 6T recovery sizing in `metis17_tst_pretraining_data_plan.md`.

## Executive decision

Use the 25 source families in the companion
[`metis17_200tb_acquisition_ledger.csv`](metis17_200tb_acquisition_ledger.csv). At the pinned
revisions and file selectors, they comprise:

- **200,021,721,850,099 compressed transfer bytes**;
- **200.021721850 decimal TB**, or **181.918696262 TiB**;
- **309,868 content objects** across thousands of configurations, snapshots, and languages; and
- substantially more than 35T publisher-reported tokens before cross-family deduplication, although
  those publisher totals use incompatible tokenizers and must not be summed as an exact count.

The right answer is **25 canonical families, not hundreds of top-level dataset names**. FineWeb 2
alone has 1,868 language-script configurations; HPLT 3 has 198 languages; The Stack v3 has 713
programming languages; and the selected manifests contain more than 300,000 payload objects. Turning
these into hundreds of aliases would make provenance, licensing, resumption, and global deduplication
harder without creating a single new document.

The basket is deliberately broad, but “200 TB of the highest-quality data” needs one hard
qualification: **there is not 200 TB of mutually unique, uniformly premium public text**. The tail is
necessarily broad web and multilingual material, and the named frontier corpora overlap heavily.
The recommendation maximizes useful optionality under a 200 TB *gross transfer* envelope; it is not a
claim that every acquired byte should enter training.

Most importantly, **do not treat the 200 TB number as the completion condition**. The completion
condition is a measured inventory of accepted, globally deduplicated, decontaminated text tokenized
with the frozen Metis-1.7 tokenizer:

- minimum gate: **42T unique accepted tokens** (20% above the 35T exposure plan);
- preferred gate: **45T unique accepted tokens**; and
- separate premium gate: enough high-quality material to supply the **5T NTP cooldown** without an
  accidental amount of repetition.

Acquiring first and choosing ratios after global preparation is the correct strategy. It only works
if every source's quality, language, date, modality, license, organic/synthetic, and derivation labels
survive preparation.

## 1. The byte arithmetic: the proposed 200 TB safety margin is not actually safe

The 4.5 bytes/token assumption is about **decoded UTF-8 text**, not Parquet, gzip, or Zstandard
transfer size. The three quantities must never share one column:

1. `transfer_bytes`: compressed bytes downloaded from the publisher;
2. `logical_text_bytes`: normalized text bytes before tokenization; and
3. `accepted_unique_tokens`: tokens remaining after every rejection and deduplication stage.

The exact 200.022 TB ledger is quantity 1. The 157.5 TB calculation is quantity 2. Because large
Parquet/Zstandard corpora commonly expand materially, downloading the full ledger will require **far
more than 200 TB of preparation storage**. FineWeb's pinned 53.418 TB Parquet payload alone is
reported as more than 18.5T GPT-2 tokens; HPLT's 56.876 TB full compressed release reports 29.831T
tokens; Essential-Web's 26.146 TB Parquet payload reports 24T tokens. These counts overlap and use
different tokenizers, but they prove the unit mismatch.

Even if 200 TB meant decoded text, the safety margin is only 21.25%:

| Assumption | Result |
|---|---:|
| 35T tokens × 4.5 bytes/token | 157.5 TB decoded text |
| 200 TB ÷ 4.5 bytes/token | 44.444T pre-filter tokens |
| maximum loss while retaining 35T | **21.25%** |
| decoded bytes needed at 30% loss | 225.0 TB |
| decoded bytes needed at 35% loss | 242.308 TB |
| decoded bytes needed at 40% loss | 262.5 TB |
| decoded bytes needed at 50% loss | 315.0 TB |

Cross-publisher Common Crawl overlap alone can make a 21.25% loss allowance unrealistic. The safe
policy is therefore **token-gated staged acquisition**, not “download 200 TB and assume it is enough.”

TST at bag size 16 does not reduce source-token demand by 16×. It reduces processed model positions
during the superposition phase while consuming the stated 30T source tokens. The corpus still needs
to provide 35T source-token exposures in total.

## 2. The exact acquisition basket

All sizes below are compressed transfer payload at the pinned revision. “Reported scale” is useful
orientation, not a common-tokenizer measurement. Test splits, removed data, sample duplicates, and
known pointer-only partitions are excluded.

| # | Source family | Transfer TB | Reported scale / role | Why it is in the basket |
|---:|---|---:|---|---|
| 1 | FineWeb | 53.417984 | 18.5T+ GPT-2 tokens | Full cleaned English web baseline explicitly requested; physical `data/` only. |
| 2 | FineWeb 2 filtered train | 9.443458 | 5B documents; 1,868 language-script configs | Broad multilingual web; `_removed` and test data excluded. |
| 3 | Nemotron-CC v2 | 10.333000 | 6.586T tokens | Organic, synthetic, DQA, translated, math, code, and specialist slices with labels. |
| 4 | Nemotron-CC v2.1 | 4.590657 | 2.5448T new tokens | New organic/synthetic/translated/DQA content intended to complement v2. |
| 5 | Essential-Web v1.0 | 26.145842 | 24T tokens; 23.6B documents | Best large metadata-rich web anchor for later quality slicing. |
| 6 | DCLM baseline 1.0 Parquet | 7.419668 | 3.88T tokens | Strong reproducible baseline; explicitly requested; raw 240T pool rejected. |
| 7 | Dolma 3.5 pool | 7.869382 | nearly 10T tokens | Current mixed web, academic, PDF, discussion, and 1.351T+ code-token pool. |
| 8 | The Stack v3.1 train | 3.546040 | 11.5 TB decoded; ~3.6T tokens | Freshest large downloadable GitHub code corpus with inline contents. |
| 9 | Common Corpus | 4.489487 | 2.267T tokens | Open-licensed culture, government, science, code, and web with provenance. |
| 10 | FinePDFs train | 5.374387 | ~3T tokens; 476M documents | Multilingual PDF/OCR layer; test splits excluded. |
| 11 | UltraData-Math | 0.552413 | 290B+ tokens | Current web and synthetic math; normalizes MathML/KaTeX/AsciiMath to LaTeX. |
| 12 | Nemotron-CC-Code v1 | 0.563192 | 427.9B tokens | Inline Common Crawl code/tutorial text with code and equation preservation. |
| 13 | Nemotron-CC-Math v1 | 0.259938 | 133B 3plus; 52B 4plus; 73B MIND | High-fidelity LaTeX-normalized math with explicit quality tiers. |
| 14 | Nemotron Specialized v1 | 0.350923 | specialist synthetic family | STEM, textbooks, scientific coding, rewrites, RQA, and other labeled slices. |
| 15 | Nemotron Specialized v1.1 | 0.034883 | specialist extension | Formal logic, economics, algorithmic, code-concept, and MC families. |
| 16 | Nemotron Specialized v1.2 | 0.053621 | specialist extension | New fact-seeking, moral-scenario, generative, and MC material. |
| 17 | Nemotron Legal v1 | 0.006991 | legal specialist family | Jurisdiction/task-labeled legal text, subject to placeholder validation. |
| 18 | Nemotron Code v1 synthetic only | 0.221904 | inline synthetic code | Content-bearing synthetic partition; repository metadata pointers excluded. |
| 19 | Nemotron Code v2 synthetic only | 0.886205 | inline QA/review/rewrite/transpile code | Content-bearing update; 20.919 GB metadata pointer partition excluded. |
| 20 | MegaMath content partitions | 1.116312 | web, QA, translated code, and code blocks | Broad math specialist layer; `megamath-code` pointers excluded. |
| 21 | Darwin-CC | 1.798020 | 504B tokens; 1.02B documents | Current evolved-cleaner derivative; parent relation retained for dedup. |
| 22 | Ultra-FineWeb L1 | 2.832186 | 1T+ tokens from six 2025 snapshots | Fresh organic crawl layer through CC-MAIN-2025-51. |
| 23 | Ultra-FineWeb L3 | 1.899216 | 400B+ English + 200B+ Chinese | Current labeled synthetic QA and multi-style web expansions. |
| 24 | scholarweave/arxiv-latex | 0.289210 | 3.12M source bundles | Full LaTeX, bibliography, style, date, and per-paper license fields. |
| 25 | HPLT 3.0 selected payload | 56.526802 | 29.831T tokens full release; 198 languages | Massive multilingual anchor; 14 lowest-WDS English shards omitted to trim total. |
|  | **Total** | **200.021722** |  | **Pinned gross transfer basket** |

By compressed bytes, this is 174.407 TB bulk web, 17.733 TB mixed/open/PDF, 5.217 TB explicitly
code-focused, 2.218 TB explicitly math/LaTeX-focused, and 0.446 TB specialist. Those labels understate
code, math, and science because Dolma 3.5, Common Corpus, FinePDFs, Essential-Web, and HPLT also contain
them.

The exact revisions, byte counts, selectors, content fields, license gates, and overlap families are
in the CSV. Freeze that file into the eventual source lock; never resolve `main` again during a
release.

## 3. What is genuinely current and what is an exception

Nearly every selected family is late 2025 or 2026. Three older families are retained deliberately,
not through inertia:

- **FineWeb (July 2025 update)** is explicitly requested and remains relevant through the
  FineWeb-Edu half of the [May 2026 TST reference](https://arxiv.org/abs/2605.06546). Its role is a
  stable full-English baseline, not freshness.
- **DCLM baseline (July 2024 release)** is explicitly requested and is the other half of that May
  2026 TST reference corpus. The high-quality 3.88T baseline is retained; the raw 240T pool is not.
- **MegaMath (April 2025 release)** remains a named source/baseline in the 2026
  [UltraData-Math release](https://huggingface.co/datasets/openbmb/UltraData-Math). It is included as
  specialist content, with its pointer-only code partition removed.

Current publisher evidence matters more than release badges. NVIDIA's
[Nemotron 3.5 Lightning pretraining documentation](https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/pretrain.html)
(2026) still uses the Nemotron CC, CC Math, CC Code, code, and specialized families. The
[HPLT 3.0 paper](https://arxiv.org/abs/2511.01066) (November 2025) documents the multilingual release.
[Dolma 3.5](https://huggingface.co/datasets/allenai/dolma3.5_pool) (July 2026),
[The Stack v3.1](https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train) (September 2026),
[Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) (May 2026), and
[FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) (April 2026) are the most important
current broad additions.

There is still a freshness ceiling. The best large curated public corpora released in 2026 mostly end
in 2024 or 2025. Ultra-FineWeb L1 reaches late 2025; Dolma 3.5 reports PDFs through November 2025; The
Stack v3 cutoff is 2025-08-07. A model trained only on the core basket will not have comprehensive 2026
world knowledge.

### Optional 2026 freshness swap

Common Crawl's official releases report compressed WET sizes of 5.85 TiB for
[May 2026](https://commoncrawl.org/blog/may-2026-crawl-archive-now-available), 5.69 TiB for
[June 2026](https://commoncrawl.org/blog/june-2026-crawl-archive-now-available), 5.89 TiB for
[July 2026](https://commoncrawl.org/blog/july-2026-crawl-archive-now-available), and 5.84 TiB for
[August 2026](https://commoncrawl.org/blog/august-2026-crawl-archive-now-available): **23.27 TiB, or
about 25.586 decimal TB**, across four WET snapshots.

This is not silently included in the exact basket because WET is raw extracted crawl text, not a
frontier-quality finished dataset. If 2026 factual freshness is a first-class objective, the best
swap is:

1. replace up to 25.6 TB of the lowest-score HPLT English tail, not specialist data;
2. download WET objects directly, avoiding URL/repository pointer reconstruction;
3. run a current classifier/selector ensemble rather than legacy C4/Gopher/KenLM gates;
4. deduplicate against every already-acquired Common Crawl family; and
5. measure its accepted unique-token yield before keeping the full tranche.

This needs a machine with direct egress and node-local NVMe. Metis-1.6 already demonstrated that
random-I/O crawl reconstruction on a Lustre-backed login node is the wrong execution shape.

## 4. Source-family findings

### 4.1 Bulk web and multilingual

**Essential-Web v1.0 (October 2025)** is the best *organizing* corpus in the basket. Its 24T-token
release carries subject, page type, complexity, quality, snapshot, and other document-level signals.
That makes it possible to decide the final Metis mixture after preparation without treating the web
as one opaque bucket. It is still Common Crawl-derived and therefore overlaps FineWeb, DCLM,
Nemotron, Darwin, and HPLT.

**FineWeb v1.4 (July 2025)** is valuable as a reproducible English baseline and because the user asked
for the whole family. Acquire the physical `data/` payload once. The named `sample-10BT`,
`sample-100BT`, and `sample-350BT` views are subsets, not new information, and should not consume
transfer or dedup time.

**FineWeb 2 v2.1.1 (October 2025)** should be acquired only as filtered train. The repository also
contains **10.703 TB of `_removed` data** and **4.042 GB of test data**. “Entire FineWeb 2” must not
mean training on publisher-designated removed documents or its evaluation split.

**Nemotron-CC v2 + v2.1 (December 2025–July 2026)** are unusually useful because their categories
survive into the payload. NVIDIA explicitly says v2.1's 2.5448T tokens are new and meant to complement
v2's 6.6T. Preserve organic, quality level, translated source language, DQA, model used, rephrase
prompt, and synthetic/translated status. “All Nemotron” is acceptable for acquisition; it is not a
license to flatten 2.1T tokens of synthetic rewrites into the same bucket as fresh organic pages.

**HPLT 3.0 (November 2025 paper)** supplies scale and language coverage no other selected family
matches. It has publisher WDS score bins. The exact basket removes fourteen named shards from the
lowest English score bucket and otherwise retains the release. In final selection, WDS 10–8 should
not receive the same prior as WDS 5; preserve the bin.

**Ultra-FineWeb L1/L3 (August 2026)** provides the most current packaged web layer found. L1 is the
organic 2025 crawl; L3 is synthetic English/Chinese QA and multi-style data. The broad
`openbmb/Ultra-FineWeb` repository also contains historical/repacked trees; do not snapshot it
blindly. The two canonical L1/L3 repositories in the ledger avoid that version duplication.

**Darwin-CC (March 2026)** is an evolved-cleaner derivative of Nemotron-CC, not an independent crawl.
Its paper reports gains against raw and several filtered baselines, making it worth acquiring; its
parent relationship makes it a high-risk overlap group.

### 4.2 PDFs, science, books, government, and licensed material

**FinePDFs (April 2026)** adds 3T reported tokens across 1,733 language-script pairs. Its OCR/layout
path is materially different from WET extraction and useful for math/science. Every small test split
is excluded. FinePDFs-Edu is a quality subset of this family, so separately downloading it would count
the same documents twice; preserve or rederive its membership instead.

**Common Corpus (May 2026 / ICLR 2026)** is the cleanest provenance-oriented anchor: 2.267T tokens
across open culture, government, science, source code, and web, with granular license/provenance.
Dataset-level openness is not enough for Metis's release policy—validate the allowed license at the
record or collection level and preserve the evidence.

**Dolma 3.5 (July 2026)** supplies a current mixed pool: academic publications, OCR'd PDFs through
November 2025, web and discussion data, plus a code pool expanded to more than 1.351T tokens. Several
components originate in families represented elsewhere, so component provenance is mandatory.

### 4.3 Code: actual bytes, not repository coordinates

The code rule is absolute: a selected partition must expose source or generated code in the row. A
dataset named “code” does not pass this gate.

| Source | Inline payload | Keep | Reject |
|---|---|---|---|
| The Stack v3.1 train | `files[].content` | all permitted-license train content | quarantine `license_type=no_license` unless policy approves it |
| Dolma 3.5 code components | `text` | content-bearing components | any component found to be a locator after row sampling |
| Common Corpus OpenSource | `text` | licensed code with provenance | rows failing the allowed-license policy |
| Nemotron-CC-Code v1 | `text` | processed code pages and tutorials | none by shape; still sample-validate |
| Nemotron Code v1 | `content` | `Synthetic-Code/**` | `Nemotron-Code-Metadata/**` (24.283 GB of pointers) |
| Nemotron Code v2 | `content` | `Synthetic-Code/**` | `Nemotron-Code-Metadata/**` (20.919 GB of pointers) |
| MegaMath | inline text/code in five selected families | QA, code-block, translated-code, web, web-pro | `megamath-code/**` (1.846 GB of pointers) |
| arXiv/TeX | `latex_source`; TeX files in Stack v3 | actual source | metadata-only records |

**The Stack v3.1 (September 2026)** is the main raw-code anchor. It stores full file contents grouped by
repository, covers 713 detected languages, and has a 2025-08-07 cutoff. Version 3.1 matters: the
publisher fixed a partition error that allowed duplicates to leak, reducing the train count from 4.9T
to 3.6T tokens. The file manifest is 3.546 TB compressed; the card's internally consistent train table
reports 11.5 TB decoded. A separate 15.9 TB sentence on the same card appears stale, so it is not used
for capacity arithmetic.

**Do not acquire Nemotron Code v3 as code.** Its released rows contain `repo`, `rel_path`, and
`commit_id`; that is the exact pointer shape that made Metis-1.6 data preparation painfully slow.
Likewise, The Stack v2/Software Heritage reconstruction paths remain outside the core basket.

Before admitting any code partition, the preflight must read real rows and assert:

- at least one nonempty content field is present;
- content length and code-token fraction are plausible;
- path/repository identifiers are metadata, not the only payload;
- language, license, timestamp/commit cutoff, source, and synthetic method survive normalization; and
- a small language-stratified sample parses or compiles at an expected rate, with generated-code
  failures reported separately from organic code.

### 4.4 Math, LaTeX, and KaTeX

The strongest LaTeX plan is a **multi-source labeled stack**, not one dataset:

1. **scholarweave/arxiv-latex (August 2026):** 3.12M full source bundles, bibliography/style fields,
   paper dates, and per-paper license. Filter licenses; do not strip macro definitions before formula
   recovery.
2. **UltraData-Math (February–April 2026):** more than 290B tokens across L1 web, L2 preview, and L3
   synthetic forms. It explicitly converts MathML, KaTeX, and AsciiMath into LaTeX.
3. **Nemotron-CC-Math v1 (August 2025 release, current 2026 paper cycle):** Lynx rendering and LLM
   cleanup preserve equations and normalize math. Quality-3 and quality-4/5 raw partitions are
   disjoint; MIND is a transformed derivative of 4plus.
4. **Nemotron-CC-Code v1 (December 2025):** code tutorials and documentation with equation retention
   and LaTeX normalization.
5. **FinePDFs, Common Corpus, and Dolma 3.5 (2026):** long-form scientific and technical material,
   retaining their OCR/source provenance.
6. **The Stack v3.1 (September 2026):** actual TeX/LaTeX project files, including practical macro and
   package usage.

KaTeX is a renderer supporting a defined subset/ecosystem around TeX syntax, not a separate natural
language. Build a deterministic specialist derivative after source normalization:

- extract display and inline math without losing surrounding explanatory text;
- retain original source, macro preamble, normalized LaTeX, document ID, source family, and license;
- render every candidate against a **pinned KaTeX version** and record `render_ok`, error class,
  unsupported command, macro dependency, and normalization transform;
- create paired original ↔ KaTeX-compatible variants only when the transformation round-trips or has
  an auditable semantic check;
- keep invalid examples in a diagnostic quarantine, not the training stream; and
- deduplicate formulas with their explanatory context, not by formula string alone. Common formulas
  such as `x^2` are not independent documents.

Do not flood the corpus with procedurally generated formula strings merely to increase “LaTeX
tokens.” Fidelity, context, and variation in real mathematical exposition matter more than byte
count.

## 5. TOON: build it as a paired derived language, not a fake bulk source

The official [TOON specification](https://github.com/toon-format/spec/blob/main/SPEC.md) is a v4.1
Working Draft dated July 2026. It encodes the JSON data model using indentation and compact tabular
forms, especially for uniform arrays of objects. It is promising where field names repeat many
times; it is not automatically better for short, deeply heterogeneous, or schema-heavy structures.

No credible, large, authoritative TOON-native pretraining corpus was found as of 2026-09-04.
Downloading small community conversions to pretend there is one would optimize the dataset list, not
the model.

The correct corpus is a **deterministic paired conversion** from structured records already acquired:

- Essential-Web taxonomy/quality records;
- FineWeb and FineWeb 2 document metadata;
- Stack v3 repository objects with uniform file arrays;
- Common Corpus provenance/license structures;
- arXiv metadata and references;
- Dolma component metadata; and
- additional current permissively licensed structured snapshots admitted under the same source-lock
  process.

For each normalized JSON value, store:

```text
record_id
source_id
source_revision
toon_spec_version = 4.1
toon_spec_commit
canonical_json
toon
shape_class
json_tokens_metis
toon_tokens_metis
roundtrip_ok
```

The acceptance invariant is strict:

```text
decode_toon(encode_toon(canonical_json)) == canonical_json
```

Test Unicode, escapes, empty collections, null/boolean/number distinctions, delimiter collisions,
arrays of uniform objects, nested arrays, heterogeneous arrays, and deep objects. Generate at least
four labeled shape classes: uniform tabular, nested uniform, heterogeneous/deep, and adversarial
conformance cases.

Keep **both JSON and TOON**. A February 2026 independent evaluation
([arXiv:2603.03306](https://arxiv.org/abs/2603.03306)) reported plain JSON as strongest overall in its
tasks and found that TOON's schema/prompt tax can erase savings on small structures. That result does
not disqualify TOON; it rules out replacing JSON by ideology. Later Metis ablations should decide
where TOON improves accepted-token efficiency and accuracy.

Derived TOON and JSON/TOON pairs do **not** count as new acquisition bytes or unique facts. Count them
only in the eventual training-mixture ledger.

## 6. The overlap map

The selected names are not independent universes:

| Overlap group | Main members | Risk | Required treatment |
|---|---|---|---|
| Common Crawl English | FineWeb, DCLM, Essential-Web, Nemotron v2/v2.1, Darwin, HPLT English, Ultra-FineWeb, Dolma web | very high exact and near overlap | global URL/content/near dedup; choose survivor by quality and fidelity |
| Common Crawl multilingual | FineWeb 2, HPLT, FinePDFs, Nemotron translations | high plus cross-language translations | language-aware dedup; retain translation parent IDs |
| PDF/science | FinePDFs, Dolma 3.5, Common Corpus, arXiv source | exact paper and OCR/source variants | prefer source LaTeX for formulas; retain high-quality OCR only when it adds layout/text |
| GitHub/source code | Stack v3, Dolma code, Common Corpus OpenSource, MegaMath code blocks | high repository/file overlap | normalized file hash plus repository identity; license-aware survivor selection |
| Nemotron derivatives | CC v2/v2.1, Darwin, CC Math, CC Code, specialized, Ultra-FineWeb synthetic | parent/rewrites rather than only exact copies | preserve generator/prompt/parent labels; cap semantic clusters later |
| Math web | UltraData-Math, Nemotron-CC-Math, MegaMath, FinePDFs, Dolma | high source-domain and synthetic-seed overlap | formula/context hashes plus document-level near dedup |
| Licensed/public documents | Common Corpus, Dolma, FinePDFs, HPLT, arXiv | recurring public-domain and government copies | provenance graph; prefer the clearest license and highest-fidelity representation |

Deduplication must be global and quality-aware. “Whichever source arrived first wins” would make the
final corpus depend on download order. Assign survivor precedence from explicit features:

1. usable license evidence and opt-out compliance;
2. source fidelity (source LaTeX/code over lossy rendering when semantically equivalent);
3. document completeness and parse/render validity;
4. publisher quality score and Metis classifier ensemble;
5. recency when quality and fidelity tie;
6. provenance richness; and
7. deterministic source/document ID as the final tie-breaker.

Use separate fingerprints for prose, code, and math. A prose MinHash configuration is not a safe
deduplicator for source code; line reformatting and boilerplate licenses dominate. A code fingerprint
should normalize comments/whitespace only as a secondary view while retaining the original. For
math, deduplicate the document/context and also cluster formulas; never discard all explanations of a
common expression.

Synthetic rewrites deserve two identities: exact text identity and semantic parent cluster. Exact
dedup removes copies; the parent cluster enables later caps without pretending paraphrases are
unrelated knowledge.

## 7. Acquisition and preparation contract

### 7.1 Manifest preflight before a byte-scale run

For every selected configuration:

1. pin immutable revision and enumerate every object;
2. assert coverage: every object selected exactly once, no gaps, no duplicates;
3. record expected transfer bytes, checksum/ETag, format, compression, split, and row schema;
4. read real rows and prove the declared content field contains text/code rather than a pointer;
5. validate that the declared license policy can actually be satisfied by the row or collection;
6. reject tests, benchmark data, removed partitions, history trees, and convenience samples;
7. freeze the decontamination registry before normalization; and
8. generate a source receipt that remains valid through handoff.

This is where the Metis-1.6 failure mode should be killed. A repository with `repo`, `commit`, and
`path` but no content must fail immediately, not normalize to zero a week later.

### 7.2 Canonical record envelope

Every normalized record should carry at least:

```text
document_id
source_id
source_revision
source_config
source_object
source_row_id
url_or_repo_identity
snapshot_or_document_date
language_script
modality
domain_taxonomy
publisher_quality
metis_quality_scores
organic_synthetic_translated
generator_and_prompt_family
derivation_parent
license_id_and_evidence
original_content_hash
normalized_content_hash
text
```

Do not throw metadata away to save a few percent of preparation storage. The whole reason to choose
ratios after preparation is to query these dimensions later.

### 7.3 Stage order

1. **Acquire and verify.** Content-address objects and compare actual bytes/checksums to the frozen
   source lock.
2. **Normalize losslessly first.** Repair encodings, preserve code/math structure, and emit explicit
   failures. Keep original hashes and transformations.
3. **Apply license/opt-out and safety gates.** Fail closed, but distinguish a rejected row from a
   broken source.
4. **Decontaminate.** Match exact and normalized benchmark forms, including code and math variants,
   before any benchmark can be folded into synthetic data.
5. **Global exact dedup.** Include cross-source, cross-snapshot, and component relationships.
6. **Modality-aware near dedup.** Prose, code, math, and translations need different rules.
7. **Quality scoring.** Retain raw features and ensemble scores; do not prematurely hard-filter every
   family to one threshold.
8. **Tokenize with the frozen Metis-1.7 tokenizer.** Count accepted tokens per every preserved
   stratum.
9. **Build deterministic corpus views.** Broad TST candidates, premium NTP candidates, specialist
   candidates, and diagnostic quarantines—without yet choosing mixture weights.
10. **Generate derived representations.** TOON pairs and KaTeX-compatible math only after source
    records are stable; link every derivative to its parent.

Every partitioner must prove exactly-once coverage. Every fast normalization or scoring path must be
differentially tested against its readable reference. These are correctness gates, not performance
tests.

### 7.4 TST-specific packing constraints

Bag size 16 makes boundary handling more important, not less:

- never let a 16-token bag cross document, source, language, license, or synthetic/organic boundaries;
- never pad a bag with tokens whose loss mask is accidentally live;
- record dropped tail tokens per document and prove the aggregate matches expectation;
- preserve within-document contiguity; random token mixtures would destroy what TST is meant to learn;
- keep dedup and decontamination before TST packing; and
- build the NTP cooldown view independently from the premium accepted pool rather than treating the
  last 5T tokens of the TST stream as “cooldown.”

## 8. Gates before deciding ratios or stages

The user is right to postpone ratios. The following measured table must exist first, by source,
language, domain, quality bin, date, modality, license, and organic/synthetic class:

- compressed transfer bytes;
- logical normalized text bytes;
- rows accepted/rejected with reason counts;
- exact-dedup survival;
- near-dedup survival;
- decontamination removals;
- unique tokens under the frozen Metis tokenizer;
- bytes/token and tokens/document distributions;
- source and domain concentration;
- 2025/2026 freshness fraction;
- code parse/compile rates by language;
- LaTeX and KaTeX render rates with error taxonomy;
- synthetic-parent cluster sizes;
- benchmark contamination hit rates; and
- premium-candidate tokens available for the 5T cooldown.

Only then decide:

- which quality strata are broad enough for 30T TST exposure;
- which strata deserve the 5T NTP cooldown;
- whether any source needs repetition and how much;
- language and domain floors/caps;
- synthetic and translated caps;
- source-temperature or quality-temperature schedules; and
- annealing order.

No mixture percentages in this report are disguised guesses. The compressed-byte composition is an
acquisition fact, not a training recommendation.

## 9. Reserve ladder and explicit exclusions

### Reserve ladder, in order

1. **2026 Common Crawl WET freshness overlay:** direct content, but only after current quality filters
   and global dedup; swap against low-score HPLT/FineWeb tail.
2. **Additional HPLT English low-WDS objects:** the exact basket omits only 0.349 TB; restore if the
   gross target is allowed to drift upward.
3. **GneissWeb-style current selector pass over already acquired FineWeb:** classifier/selection
   sidecar, not new text bytes.
4. **Propella annotations:** metadata sidecar for later scoring, not text capacity.
5. **A newly verified CuraWeb public release:** watchlist only; no payload was verified by the cutoff.

### Do not count these as new corpus bytes

- FineWeb sample configurations;
- FineWeb-Edu and FinePDFs-Edu when their parent payload is already acquired;
- old or historical Ultra-FineWeb version trees;
- synthetic rewrites as independent factual coverage;
- TOON conversions of already acquired JSON;
- KaTeX-normalized variants of already acquired LaTeX;
- classifier annotations; or
- repository/file metadata without source content.

### Excluded from the core plan

- **Nemotron Code v3:** pointer-only code metadata.
- **The Stack v2 / Software Heritage reconstruction:** content retrieval indirection.
- **Raw DCLM 240T pool:** too noisy and too large relative to the curated baseline.
- **FineWeb 2 `_removed`:** publisher-removed material.
- **Every test split:** held out from training.
- **TxT360 as a blind snapshot:** repository contains version/repack/augmentation duplication and its
  content cutoff is older than the selected current sources.
- **C4, SlimPajama, RedPajama, RefinedWeb:** superseded bulk-web choices for a 2026 frontier plan.
- **OpenWebMath and Proof-Pile-2 as bulk math:** superseded by 2025–2026 math/PDF/LaTeX sources; useful
  only as historical baselines or niche formal-language recovery.
- **KenLM perplexity or GPT-3/Gopher heuristics as the primary quality gate:** outdated. A modern
  ensemble can retain interpretable heuristics as features, not as the definition of quality.

## 10. Licensing and undisclosed facts

Dataset-card licenses do not automatically clear every underlying document. This report is a data
engineering recommendation, not a legal opinion. Before acquisition/use:

- accept and archive the exact NVIDIA/Darwin gated terms;
- review upstream model-license obligations attached to generated Nemotron subsets;
- enforce per-record/per-collection license rules for Common Corpus, Stack v3, and arXiv;
- quarantine Stack v3 `no_license` repositories unless the release policy explicitly permits them;
- preserve takedown/opt-out lists and bind them to a source revision; and
- retain evidence, not merely a normalized license string.

Important undisclosed or unresolved facts as of 2026-09-04:

- exact cross-publisher overlap among the giant Common Crawl derivatives;
- exact accepted Metis-tokenizer yield;
- exact logical disk footprint after Metis normalization;
- whether every gated payload still matches its public manifest after access is granted;
- a public high-quality corpus extending comprehensively through September 2026;
- a large authoritative TOON-native corpus; and
- final quality/utility of TOON for Metis without controlled JSON-vs-TOON ablations.

## Bottom line

The companion ledger is a defensible **200.022 TB gross, content-bearing acquisition universe**. It
contains the entire useful training payload of FineWeb, FineWeb 2 filtered train, DCLM baseline,
Nemotron CC v2/v2.1 and the current high-value open families, while explicitly avoiding the pointer
traps that hurt Metis-1.6. It also creates first-class LaTeX/KaTeX and TOON paths without pretending
derived formats are new knowledge.

But the most important planning change is this: **freeze datasets now, freeze ratios later, and gate
completion on unique accepted Metis tokens—not on downloaded bytes.** If the globally prepared corpus
does not reach 42–45T accepted unique tokens, add the 2026 freshness reserve or more decoded capacity.
If it does, stop acquiring even if a nominal byte target remains. Quality is the constraint; disk
occupancy is not the objective.
