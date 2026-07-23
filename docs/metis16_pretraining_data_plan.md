# Metis-1.6 Pretraining Data Plan

Status: Data Manifest v1.0  
Release name: `metis-1.6-data-r1`  
Effective date: 2026-07-21

## Release contract

Metis-1.6 trains on exactly **1,000,000,000,000 token exposures**, counted only after the final
Metis tokenizer has been accepted. Upstream token estimates, downloaded bytes, document counts,
and counts from another tokenizer never satisfy the release target.

The machine-readable source of truth is [`manifests/metis-1.6.yaml`](../manifests/metis-1.6.yaml).
Every individual source, pinned Hugging Face revision, target, provenance flag, license disposition,
quality profile, and deduplication namespace is declared in [`manifests/sources/`](../manifests/sources/).
The CLI refuses a manifest whose source, category, phase, freshness, replay, tokenizer, or exclusion
totals do not reconcile exactly.

| Category | Phase A | Phase B | Phase C | Total | Share |
|---|---:|---:|---:|---:|---:|
| General and educational web | 420B | 85B | 20B | 525B | 52.5% |
| Code and software engineering | 90B | 55B | 15B | 160B | 16.0% |
| Mathematics and formal reasoning | 40B | 35B | 10B | 85B | 8.5% |
| Science, papers, PDFs, and technical text | 90B | 30B | 5B | 125B | 12.5% |
| Synthetic pedagogical and factual data | 25B | 45B | 0B | 70B | 7.0% |
| Books, encyclopedic, legal, and reference | 25B | 0B | 0B | 25B | 2.5% |
| Translated and native multilingual | 10B | 0B | 0B | 10B | 1.0% |
| **Total** | **700B** | **250B** | **50B** | **1,000B** | **100%** |

The exact source tables are:

- [`web.yaml`](../manifests/sources/web.yaml): 9 sources, exactly 525B.
- [`code.yaml`](../manifests/sources/code.yaml): 9 sources, exactly 160B.
- [`math.yaml`](../manifests/sources/math.yaml): 8 sources, exactly 85B.
- [`science.yaml`](../manifests/sources/science.yaml): 8 sources, exactly 125B.
- [`synthetic.yaml`](../manifests/sources/synthetic.yaml): 7 sources, exactly 70B.
- [`reference.yaml`](../manifests/sources/reference.yaml): 13 sources, exactly 25B.
- [`multilingual.yaml`](../manifests/sources/multilingual.yaml): 2 sources, exactly 10B.

### Exact source allocations

These are final-tokenizer-measured training exposures, not raw reservoir sizes. “Generated or
transformed” is provenance, not the capability category; for example, translated text remains in
the multilingual category and verified synthetic code remains in code.

| Category | Source | Phase A | Phase B | Phase C | Total | Fresh | Generated or transformed |
|---|---|---:|---:|---:|---:|:---:|:---:|
| web | `nemotron_cc_v2_organic` | 150B | 25B | 5B | 180B | no | no |
| web | `nemotron_cc_v21_organic` | 60B | 15B | 5B | 80B | no | no |
| web | `fineweb_edu` | 60B | 15B | 5B | 80B | no | no |
| web | `dclm_baseline` | 45B | 5B | 0 | 50B | no | no |
| web | `essential_web` | 25B | 5B | 0 | 30B | no | no |
| web | `txt360` | 18B | 2B | 0 | 20B | no | no |
| web | `zyda2_unique` | 14B | 1B | 0 | 15B | no | no |
| web | `metis_freshweb_2026` | 25B | 7B | 3B | 35B | yes | no |
| web | `fineweb_diversity_tail` | 23B | 10B | 2B | 35B | no | no |
| code | `nemotron_repository_code_v123` | 37.91B | 22.928B | 5.982B | 66.82B | no | no |
| code | `nemotron_cc_code_v1` | 18B | 10B | 3B | 31B | no | no |
| code | `stack_edu` | 9B | 6B | 2B | 17B | no | no |
| code | `stack_v2_unique` | 4B | 1B | 0 | 5B | no | no |
| code | `metis_freshgithub_2025_26` | 10B | 6B | 2B | 18B | yes | no |
| code | `metis_freshsoftwaredocs_2025_26` | 6B | 3B | 1B | 10B | yes | no |
| code | `metis_freshengineering_discussions_2025_26` | 3B | 3B | 1B | 7B | yes | no |
| code | `metis_systems_formal` | 0.09B | 0.072B | 0.018B | 0.18B | no | no |
| code | `verified_synthetic_code` | 2B | 3B | 0 | 5B | no | yes |
| math | `nemotron_cc_math_4plus` | 20B | 15B | 5B | 40B | no | no |
| math | `nemotron_cc_math_unique_3` | 8B | 6B | 1B | 15B | no | no |
| math | `nemotron_math_proofs` | 3.6B | 5.4B | 0 | 9B | no | yes |
| math | `proof_pile2_math` | 3.34B | 3.51B | 2B | 8.85B | no | no |
| math | `finemath_unique` | 2B | 2B | 1B | 5B | no | no |
| math | `megamath_unique` | 2B | 2B | 1B | 5B | no | no |
| math | `openwebmath_unique` | 1B | 1B | 0 | 2B | no | no |
| math | `formal_theorem_corpora` | 0.06B | 0.09B | 0 | 0.15B | no | no |
| science | `finepdfs_edu_english` | 43.97075B | 12.99025B | 3B | 59.961B | no | no |
| science | `pes2o` | 19B | 5B | 0 | 24B | no | no |
| science | `pmc_open_access` | 11B | 3B | 0 | 14B | no | no |
| science | `proof_pile2_science` | 4B | 1B | 0 | 5B | no | no |
| science | `openstax` | 0.02925B | 0.00975B | 0 | 0.039B | no | no |
| science | `metis_freshscience_2025_26` | 6B | 3B | 1B | 10B | yes | no |
| science | `metis_freshdocs_2025_26` | 5B | 4B | 1B | 10B | yes | no |
| science | `patents_engineering_reports` | 1B | 1B | 0 | 2B | no | no |
| synthetic | `nemotron_specialized_fact_seeking` | 5B | 10B | 0 | 15B | no | yes |
| synthetic | `nemotron_v21_dqa` | 4B | 4B | 0 | 8B | no | yes |
| synthetic | `nemotron_specialized_reasoning_scicode` | 4B | 8B | 0 | 12B | no | yes |
| synthetic | `nemotron_wiki_rewrite` | 3B | 3B | 0 | 6B | no | yes |
| synthetic | `cosmopedia_v2` | 4B | 4B | 0 | 8B | no | yes |
| synthetic | `nemotron_math_textbooks` | 3B | 9B | 0 | 12B | no | yes |
| synthetic | `nemotron_rqa_verified_reasoning` | 2B | 7B | 0 | 9B | no | yes |
| reference | `finewiki` | 5B | 0 | 0 | 5B | no | no |
| reference | `wikimedia_reference` | 3B | 0 | 0 | 3B | no | no |
| reference | `public_domain_books_gutenberg` | 1.5B | 0 | 0 | 1.5B | no | no |
| reference | `public_domain_books_pre1929` | 2.5B | 0 | 0 | 2.5B | no | no |
| reference | `common_pile_loc_reference` | 1.5B | 0 | 0 | 1.5B | no | no |
| reference | `common_pile_doab_reference` | 1B | 0 | 0 | 1B | no | no |
| reference | `common_pile_libretexts_reference` | 0.5B | 0 | 0 | 0.5B | no | no |
| reference | `roots_stackexchange` | 3B | 0 | 0 | 3B | no | no |
| reference | `open_law_caselaw` | 2.3B | 0 | 0 | 2.3B | no | no |
| reference | `open_law_usgpo` | 0.7B | 0 | 0 | 0.7B | no | no |
| reference | `nemotron_legal_v1` | 2B | 0 | 0 | 2B | no | yes |
| reference | `metis_govreference_uk_hansard` | 1.5B | 0 | 0 | 1.5B | no | no |
| reference | `metis_govreference_regulations` | 0.5B | 0 | 0 | 0.5B | no | no |
| multilingual | `nemotron_v21_translated_hq` | 7B | 0 | 0 | 7B | no | yes |
| multilingual | `fineweb2_native_multilingual` | 3B | 0 | 0 | 3B | no | no |

One deliberate correction was made to the proposed table: the proposed 1B of generated Nemotron
Math Proofs in Phase C was moved to the non-generated Proof-Pile-2 math allocation. “No generated
data in Phase C” is therefore a tested invariant rather than an aspiration.

Three acquisition corrections keep the recipe legally and operationally defensible without changing
any category or phase total. The unverified third-party S2ORC snapshot was removed because its
database-level ODC-By label does not establish the rights for every underlying paper, and it overlaps
the stronger paper reservoirs. Its 15B allocation stays in the same phases: 7B moves to FinePDFs
(6B Phase A / 1B Phase B), 4B to peS2o (3B / 1B), and 4B to the licensed Common Pile PubMed/PMC
snapshot (3B / 1B). The pinned historical CC-BY OpenStax snapshot is capped at its approximately
39M-token inventory; its final 1M shortfall and the earlier displaced allocation move to FinePDFs in
the same phases. The pinned canonical formal repositories are also bounded reservoirs rather than
multi-billion-token sources. A production-filter canary over the pinned commits measured about
186.5M formal-math and 231.4M systems-code byte-estimated tokens, so their exposure caps are 150M
and 180M respectively, with the original remainder moved to Proof-Pile-2 math and NVIDIA repository
code in the same phases. Wholesale
Pile of Law is replaced by pinned Common Pile Caselaw Access Project and USGPO primary-law snapshots,
with per-record license enforcement retained.

Because categories describe capability rather than provenance, generated code, math proofs, and
legal material sit outside the dedicated 70B synthetic-pedagogy category. The complete manifest has
**86B explicitly generated exposures (8.6%)** and **93B generated-or-transformed exposures (9.3%)**,
both below the 15% hard cap. The executable cap uses the stricter 93B definition.

## The three phases

### Phase A — 0 to 700B: broad foundation

Phase A establishes language, world knowledge, scientific and software foundations, informal
registers, long-form material, reference knowledge, and limited multilingual coverage. It is all
unique after the release's global deduplication pass. Medium-quality material is admitted only when
it passes the diversity-tail profile and contributes coverage missing from the premium sources.

### Phase B — 700B to 950B: capability intensification

Phase B shifts toward code, math, science, worked explanations, grounded synthetic material, and
structured reasoning. It contains **175B unique tokens plus 75B controlled replay**. Replay is
Hamilton-apportioned across the declared Phase B source targets, uses only already accepted records,
and never takes a record beyond four total exposures.

### Phase C — 950B to 1T: premium cooldown

Phase C is **50B controlled replay** from the highest-priority non-generated web, code, math,
science, and technical records during final learning-rate decay. No ordinary synthetic data,
medium-quality filler, or new unvetted source enters this phase. The packer and verifier both reject
any Phase C shard containing a generated record.

## Freshness layer

The freshness layer is exactly **90B inside the trillion**, not on top of it:

| Freshness bucket | Exposure target |
|---|---:|
| General web through `CC-MAIN-2026-25` | 35B |
| Software source, documentation, and engineering discussion | 35B |
| Open science published in 2025–2026 | 10B |
| Current official documentation, specifications, and manuals | 10B |

Fresh records carry the strongest applicable capture, publication, commit, or retrieval date; canonical source;
version/current/deprecated status where applicable; license; and content hash. Freshness never
overrides the quality, privacy, license, or contamination gates.

### Fresh Common Crawl construction

The four WARC-backed freshness rows total **65B final exposures**: 35B general web, 10B current
software documentation, 10B recent science, and 10B current official documentation. Every row is
selected across exactly `CC-MAIN-2026-08`, `-12`, `-17`, `-21`, and `-25`. The materializer reads
all 1,500 WARC URL-index Parquet partitions through the bulk columnar index, deterministically
selects bounded candidates, and retrieves validated record byte ranges from WARC files; it never
mirrors a complete crawl or substitutes structure-poor WET text.

Selection favors official documentation and standards, universities and educational sites,
technical blogs, product/API documentation, scientific organizations, institutional and government
pages, substantive reporting, software discussions, and release material. The four routes have
separate category contracts. Every retained page must pass HTTP/WARC/digest integrity, English
evidence, publisher `noai`/`noml`/`notrain`/`noarchive`, structural-text quality, secret/PII, and
canonical-URL checks and must expose explicit reusable open-license evidence. Recent reporting is
eligible only when that evidence is present. Science additionally requires a publication date;
software and official-documentation pages require explicit version evidence.

For the general-web route, “fresh” means captured in one of the five 2026 snapshots; it does not
claim that every retained page was first published in 2026 when the page exposes no trustworthy
publication date. Science remains publication-fresh, and software repositories remain commit-fresh.

Canonical URL and payload-digest winners are selected across all five snapshots before downstream
global exact and MinHash deduplication. Semantic deduplication remains disabled. The official
Common Crawl publisher opt-out registry is applied before WARC retrieval and snapshotted again at
the end of acquisition; Rhea reapplies that final hash-bound snapshot during normalization so an
opt-out received while the multi-day acquisition was running is still honored.

Acquisition materializes a 5% FreshWeb reserve beyond the immutable source-lock minimum. This is
only protection against opt-outs arriving during the run, not extra training exposure: final
selection remains exactly 1T tokens, and the handoff recount fails if the final opt-out snapshot
leaves any locked source below target.

These gates intentionally prefer a reported shortfall over unverifiable freshness. Candidate
selection is oversampled by route and the receipts report every rejection reason. If a route is
short, expand its bounded candidate scan or add a separately licensed canonical source in a new
manifest revision; do not weaken the evidence rules or silently borrow tokens from another bucket.

## Candidate acquisition budget

The manifest intentionally requests more candidate material than the final exposure target. Current
source-lock headroom totals are **1.586231T candidate-token estimates** and approximately **1.264871TB
of planned compressed candidate payload**, before the 5% FreshWeb opt-out reserve. The byte estimate
is a planning heuristic, not a quota:
it excludes the amplification from Git repository objects, WARC/HTML retrieval, extraction,
Parquet metadata, deduplication indices, sort spill, and temporary copies.

Plan at least **25TB free** before acquisition and prefer **40TB or more** at peak. The final token
IDs alone occupy about 1.82TiB as uint16. Before every large stage the operator profile reserves a
5TB safety floor. HPE administrators must provide the correct Lustre path, quota, inode policy, and
striping recommendation for the actual allocation.

Candidate reservoirs are selected by pinned file manifests and deterministic source-specific
headroom. The pipeline does not mirror all 6.6T Nemotron-CC-v2 tokens, all 24T Essential-Web tokens,
or complete Common Crawl snapshots. Rejected raw shards may be removed only after provenance,
hashes, rejection statistics, and the final release have been verified.

Small, bounded Common Pile snapshots marked `take_all` are the exception: their candidate
multipliers represent the complete published UTF-8 inventory at the planning assumption of four
bytes per token, and their compressed-byte factors are calibrated to the pinned LFS payload sizes.
This makes the plan reserve the bytes it will actually download instead of pretending a full
snapshot is only the size of its final selected exposure. `take_all` never permits underfill: source
resolution aborts before downloading if the pinned matching-file inventory is below its target.

## Acquisition truth and launch gate

There are three acquisition classes:

1. **Packaged Hugging Face data.** The resolver checks the pinned 40-character commit, lists matching
   files, deterministically chooses enough candidate bytes, and writes an immutable source lock.
   Download tasks verify size and available LFS SHA-256 evidence before completion.
2. **Repository metadata.** Nemotron Code v3 is an 8.2GB metadata index, not 173B source-code tokens.
   Its `repo`, seven-character commit ID, and relative path must be resolved into actual licensed
   files at the pinned commit. Metadata bytes never count as training material.
3. **Dynamic sources.** Common Crawl, recent GitHub, official sites, and bulk public registries first
   produce immutable selection plans. Their materializers must then fetch WARC byte ranges, pinned
   repository objects, canonical documents, or bulk archives onto Lustre.

Dynamic selection locks are resolved by production materializers before the acquisition handoff:
Common Crawl URL-index partitions become validated WARC byte-range payloads; NVIDIA repository
metadata becomes pinned repository-file payloads; GH Archive activity becomes pinned public
repository snapshots and license-filtered engineering discussions; and canonical registries become
checksummed local documents. A URL, repository index, or selection plan is never accepted as
training material. Each materializer has a bounded fixture and the acquisition handoff requires
zero unresolved remote plans.

The login2 launch is operationally gated by two live facts the repository cannot manufacture: the
account owner must have at least 25TB of genuinely available Lustre capacity (40TB preferred), and
the exact clean repository commit used for acquisition must be available to clone. Rhea remains a
separate later gate until its Lustre mount path and Slurm settings are confirmed. Those limitations
do not require Portage compute nodes or an S3 gateway for acquisition; the confirmed login2 host has
direct HTTPS access to Hugging Face, Common Crawl, and public GitHub archives.

The final operator release must turn every source lock into local materialized files with byte
counts and hashes. `metisctl report` must show zero unresolved remote plans before the CPU build is
submitted.

The production profile also keeps `license_review_complete: false` until the source terms,
individual gated-data acceptance, per-record license rules, generator-license implications, and HPE
research policy have actually been reviewed. The verifier refuses to create a release while that
gate is false; changing the flag is an attestation, not a way to skip the review.

## Canonical record

Every normalized document uses the same logical schema:

```json
{
  "id": "stable source-scoped document id",
  "text": "normalized training text",
  "metadata": {
    "source_id": "manifest source id",
    "category": "web|code|math|science|synthetic|reference|multilingual",
    "source_revision": "pinned source revision",
    "source_file": "upstream shard or repository path",
    "license": "record or reviewed source license",
    "license_status": "manifest disposition",
    "generated": false,
    "fresh": false,
    "priority": 1,
    "quality_features": {}
  }
}
```

Nested upstream metadata is retained and its scalar fields are mapped to canonical evidence fields.
No missing quality score, language score, license, date, version, verification result, or structural
check is silently invented. Source-level reviewed licenses are valid for every record; sources marked
per-record still require per-record evidence.

## Filtering and quality gates

The quality profiles are declared in
[`configs/metis16/quality-profiles.yaml`](../configs/metis16/quality-profiles.yaml). They combine
universal and domain-specific gates.

Universal gates:

- valid UTF-8 and non-empty extractable text;
- minimum/maximum length, alphabetic fraction, symbol density, repeated-line fraction, and URL density;
- English confidence where the source is not explicitly multilingual;
- private-key, cloud-key, GitHub-token, and model-key scans;
- conservative phone, email, and US SSN pattern rejection;
- evaluation contamination rejection;
- fail-closed license evidence;
- no known low-value model boilerplate where the source profile forbids generated-looking prose.

Domain gates:

- **Web:** upstream/computed quality score, educational score where requested, capture date for fresh
  material, extraction completeness, and model-generated/SEO-spam rejection.
- **Repository code:** allowlisted license; fork/mirror, vendor, generated, minified, lockfile, data
  blob, tutorial clone, benchmark-solution, secret, and near-duplicate removal; recent commit and
  active-repository evidence for the fresh slice; parser/static checks where declared.
- **Code-text interleaving:** minimum code/text ratio and coherent tutorial/documentation context.
- **Math/formal:** score-3 or score-4 partition evidence, equation integrity, complete statement and
  argument, and parser/compiler evidence for Lean/Coq/Isabelle/Metamath material.
- **PDF/science:** OCR confidence, reading order, header/footer repetition, bibliography/body ratio,
  title or abstract, equation/table preservation, open-access status, and structural completeness.
- **Official docs:** canonical origin, exact version or commit, publication/retrieval date, current or
  deprecated status, and canonical license.
- **Synthetic:** source genealogy, grounding record, generator identity/license, and programmatic,
  execution, static, or source verification. Generic unverified rewrites are rejected.
- **Books/legal/reference:** chapter integrity, primary-source and jurisdiction evidence, and explicit
  public-domain/open-license status.
- **Translated/native multilingual:** translation-quality or language-allowlist evidence rather than
  treating all non-English material as one bucket.

Rejection reports are written per task and include reason counts. A task accepting zero records is a
failure. Source deficits are fatal: no silent global fallback is allowed.

## Global deduplication and winner policy

Deduplication is global across overlapping web descendants, not performed independently by dataset
brand. It runs in this order:

1. canonicalize text for identity comparison while preserving the original selected extraction;
2. full canonical SHA-256 signatures and externally sorted distributed duplicate finding;
3. exact duplicate removal using deterministic source/document priority;
4. a two-pass exact normalized three-sentence repeated-span/template scan (minimum 24 words): a
   compact 128-bit-prefix prefilter first proves that a span occurs across distinct documents, then
   only those candidates receive full SHA-256 signatures; both global passes use bounded external
   sorts across 64 finder buckets, keep the highest-priority occurrence, strip losing boilerplate,
   quarantine the original, and drop remnants below 50 words or three sentences;
5. MinHash signatures over 5-grams, 20 buckets, 10 hashes per bucket, fixed seed;
6. probabilistic near-duplicate candidate clustering tuned for the 0.82 operating region (the LSH
   collision curve is not falsely described as a hard Jaccard cutoff), followed by a disk-backed,
   partitioned priority-winner pipeline: streamed duplicate pairs, component construction, per-rank
   candidate hydration, 256 independent component-bucket resolvers, and per-rank removal finalizers;
7. reject repository forks/mirrors, vendor/build trees, generated/minified files, lockfiles, encoded
   blobs, and known benchmark-solution repositories when the source metadata or path identifies them;
8. code-specific normalized-file and exact function/96-token-block fingerprints, removing a lower-
   priority file when copied structural units account for at least 80% of its fingerprinted content;
9. benchmark decontamination and quarantine;
10. a final canonical SHA-256 duplicate audit before tokenizer sampling and schedule selection.

Every exact/span/code/MinHash substage publishes counts, sizes, hashes, and completeness manifests,
including explicit empty outputs. Missing ranks or buckets fail closed. The code structural pass uses
128-bit BLAKE2b structural fingerprints, while the initial and final whole-document audits use full
SHA-256. They write removal maps and quarantine records instead of silently mutating provenance.
**Semantic/embedding deduplication is intentionally disabled**: it risks deleting independently
written explanations that express the same useful idea.

When content overlaps, prefer: cleared license, primary/canonical source, human original over a
synthetic rewrite, structurally complete extraction, current version for time-sensitive material,
higher quality score, then deterministic hash. This ordering is encoded as record priority so a
restart reaches the same winner.

## Benchmark decontamination

Evaluation data is declared separately in
[`manifests/contamination/eval-holdouts.yaml`](../manifests/contamination/eval-holdouts.yaml). It is
never a training source. Registry v2 contains **63 benchmark registry entries across 36 family
labels and 203 pinned configuration/split/file jobs** spanning frontier knowledge, general reasoning, instruction following,
math/competition math, code/software engineering, retrieval and long context, science/medicine,
safety and hazardous knowledge, bias, legal/finance, and multilingual transfer. This includes HLE,
all GPQA variants including Diamond, MMLU/MMLU-Pro, MMMU, BBH, ARC, IFEval,
GSM8K/MATH/MATH-500/AIME/OlympiadBench, HumanEval/MBPP/EvalPlus/APPS/DS-1000/SciCode/
SWE-bench/LiveCodeBench, DROP/HotpotQA/Natural Questions/SimpleQA/LongBench,
WMDP/HealthBench/LegalBench, MGSM/XCOPA/XNLI, and the other entries in the manifest.

Each benchmark row is decomposed into separately indexed prompts, contexts/passages, choices and
distractors, answers/rationales, code, and test cases. The build uses canonical exact SHA-256,
normalized 13-word overlap (two matches), normalized 8-word short-fragment overlap (two matches),
syntax-gated 12-code-token overlap (two matches), and identifier/literal-normalized code-skeleton
overlap. Every threshold must be satisfied against one benchmark row; matches from unrelated
examples are never combined, and shingles appearing across more than 32 benchmark rows are
suppressed as corpus-generic. A hit removes the entire training document and writes it to the
contamination quarantine with its reason. Benchmark-answer repositories, mirrors, and solution collections are also excluded
during code hygiene. Explicit upstream `benchmark`, evaluation-dataset, or seed/source-dataset
genealogy is matched against the same pinned registry, with conservative handling for ambiguous
names such as `math` or `apps`. The build fails if any configured
holdout job, the complete holdout bundle, or the contamination index is unavailable.
The row-aware shingle postings are persisted as sorted binary arrays and memory-mapped read-only by
filter workers, rather than replicated as enormous JSON/Python mappings in every process.

## Tokenizer

The tokenizer is a new **65,536-entry byte-level BPE**, including all special tokens. Because the
largest valid ID is 65,535, final shards are stored as little-endian uint16. The release gate rejects
any larger ID or any production vocabulary that is not exactly 65,536 entries.

The 160GB tokenizer sample is stratified from the final post-filter, post-dedup, post-decontamination
mixture. It preserves ordinary English, code/identifiers/whitespace, LaTeX and scientific notation,
tables and Markdown, URLs, Unicode and names, multilingual text, and the FIM/tool serialization
tokens. The report records round-trip failures, characters per token, tokens per document, and
per-domain fertility. The trillion-token count runs only after this report is accepted.

## Exact selection, packing, and release

Final-token counts are calculated per document with one end-of-document token. Deterministic
selection fills each source/phase unique quota, builds bounded replay pools, and writes an exact
phase schedule. No source is allowed to be short; same-category substitution requires a new manifest
revision rather than an implicit runtime choice.

The schedule is packed into **1,000 shards of exactly 1B token IDs** with a sidecar index recording
document offsets, source IDs, replay status, and content hashes. The verifier rehashes every shard,
checks uint16 byte size, recomputes every source/phase total, checks Phase C provenance, emits a
license ledger, and only then writes `RELEASE.json`.

The intended immutable release layout is:

```text
metis-1.6-data-r1/
├── phase-a/
├── phase-b/
├── phase-c/
├── tokenizer/
├── manifests/
├── provenance/
├── reports/
├── RELEASE.json
└── …
```

The tokenizer and source/selection manifests are referenced by hash from the release descriptor.
The model-training job must require a verified `RELEASE.json` and must not read mutable normalized
or raw directories directly.

[`configs/metis16/pretraining.yaml`](../configs/metis16/pretraining.yaml) fixes the trainer-facing
token cursor, 4096-token base context, phase boundaries, within-phase deterministic shard ordering,
and resume state. It intentionally does not invent an unmeasured Portage learning rate or global
batch. Before a trainer launch, validate the complete on-disk release with:

```bash
./metisctl training-contract --release /lus/lustre1/vollmerc/metis-1.6/releases/metis-1.6-data-r1
```

## Split login2, Rhea, and Portage operator flow

Acquisition runs in GNU Screen on `login2`, the server that hosts/mounts Lustre; it is **not a
Slurm job**. The later normalization, quality, deduplication, decontamination, tokenizer, token
counting, selection, and packing stages run as Slurm jobs on Rhea against the same immutable
acquisition. Portage is used later for model training. Secrets stay in the user's shell/credential
stores, never in Git. Before handoff, commit and push this implementation to the documented release
branch. The source lock records the exact clean commit, so later branch movement cannot change an
acquisition already in progress.

```bash
git clone --branch codex/metis16-data-acquisition git@github.com:lernex/Metis.git
cd Metis
./ops/start-acquisition.sh \
  --lustre-root /lus/lustre1/vollmerc/metis-1.6 \
  --quota-acknowledgement administrator-confirmed
```

The launcher securely prompts for missing credentials, bootstraps, checks every production gate,
and starts one restart-safe foreground supervisor inside Screen. It returns immediately, so the SSH
session can close. Monitoring does not require attaching to Screen:

Bootstrap installs the complete transitive runtime from `requirements-metis16-data.lock` with
package hashes required and binary wheels only. The lock embeds the digest of its reviewed direct
input file and supports CPython 3.11/3.12; both Linux x86_64 ABI resolutions were checked. There is
no floating pip upgrade, editable project install, or source-package compilation in the production
path. The runtime-lock digest and Python compatibility contract are frozen into both
`sources.lock.json` and `ACQUISITION_READY.json`, so login2 and Rhea cannot silently process the
same release with different dependency graphs.

```bash
export METIS_LUSTRE_ROOT=/lus/lustre1/vollmerc/metis-1.6
./metisctl status --profile login2
./metisctl report --profile login2
```

The supervisor also downloads and pins the evaluation-only holdout bundle, then emits the hashed
`ACQUISITION_READY.json` contract. After `status` reports `build_ready: true`, enter Rhea, supply its
confirmed scheduler values, verify the immutable handoff, and submit CPU preparation:

```bash
./ops/bootstrap.sh --profile rhea --role compute --lustre-root "$METIS_LUSTRE_ROOT"
./metisctl doctor --profile rhea --role compute
./metisctl verify-handoff --profile rhea
./metisctl submit build --profile rhea
```

Safely resume unfinished acquisition by rerunning the same launcher command. Completion markers
skip finished tasks, while the Screen session check and filesystem singleton lock prevent duplicate
supervisors.

```bash
./ops/start-acquisition.sh \
  --lustre-root /lus/lustre1/vollmerc/metis-1.6 \
  --quota-acknowledgement administrator-confirmed
```

If an interrupted login2 process left a lock behind, first confirm that no acquisition Screen or
supervisor is still running on the same host, then ask the tool to remove only locks older than the
24-hour safety window:

```bash
./ops/clean-failed.sh
```

`submit pipeline` is deliberately rejected for this split-host design. The Screen acquisition
supervisor and the Slurm build both use deterministic completion markers, so each can be resumed
without redownloading or rebuilding completed tasks.

The login2 host, user path, Python runtime, detached Screen support, and primary network endpoints are
now confirmed. The remaining questions are only for Rhea: its Slurm account, partition, QoS,
maximum array size/concurrency, job time/memory limits, and the exact path by which its compute nodes
see this acquisition. The complete remaining list is in
[`docs/metis16_portage_site_intake.md`](metis16_portage_site_intake.md).

## Explicitly out of scope

- No 250M-model or reduced-model ablation is part of this release plan.
- HLE and GPQA are evaluation-only and receive zero base-pretraining tokens.
- UltraData-SFT-2605 is post-training data and receives zero base-pretraining tokens.
- This data factory creates the immutable pretraining release; it does not select Portage GPU
  topology or launch the model trainer until the release and cluster-specific trainer contract are
  separately verified.
