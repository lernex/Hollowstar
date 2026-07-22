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
- [`code.yaml`](../manifests/sources/code.yaml): 11 sources, exactly 160B.
- [`math.yaml`](../manifests/sources/math.yaml): 8 sources, exactly 85B.
- [`science.yaml`](../manifests/sources/science.yaml): 10 sources, exactly 125B.
- [`synthetic.yaml`](../manifests/sources/synthetic.yaml): 7 sources, exactly 70B.
- [`reference.yaml`](../manifests/sources/reference.yaml): 8 sources, exactly 25B.
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
| code | `nemotron_repository_code_v123` | 35B | 20B | 5B | 60B | no | no |
| code | `nemotron_cc_code_v1` | 15B | 8B | 2B | 25B | no | no |
| code | `stack_edu` | 7B | 2B | 1B | 10B | no | no |
| code | `stack_v2_unique` | 4B | 1B | 0 | 5B | no | no |
| code | `metis_freshgithub_2025_26` | 10B | 6B | 2B | 18B | yes | no |
| code | `metis_freshsoftwaredocs_2025_26` | 6B | 3B | 1B | 10B | yes | no |
| code | `metis_freshengineering_discussions_2025_26` | 3B | 3B | 1B | 7B | yes | no |
| code | `metis_swe_interleave` | 3B | 2B | 1B | 6B | no | yes |
| code | `metis_systems_formal` | 3B | 3B | 1B | 7B | no | no |
| code | `verified_synthetic_code` | 2B | 3B | 0 | 5B | no | yes |
| code | `accepted_notebooks_tests_examples` | 2B | 4B | 1B | 7B | no | no |
| math | `nemotron_cc_math_4plus` | 20B | 15B | 5B | 40B | no | no |
| math | `nemotron_cc_math_unique_3` | 8B | 6B | 1B | 15B | no | no |
| math | `nemotron_math_proofs` | 3B | 4B | 0 | 7B | no | yes |
| math | `proof_pile2_math` | 3B | 3B | 2B | 8B | no | no |
| math | `finemath_unique` | 2B | 2B | 1B | 5B | no | no |
| math | `megamath_unique` | 2B | 2B | 1B | 5B | no | no |
| math | `openwebmath_unique` | 1B | 1B | 0 | 2B | no | no |
| math | `formal_theorem_corpora` | 1B | 2B | 0 | 3B | no | no |
| science | `finepdfs_edu_english` | 32B | 10B | 3B | 45B | no | no |
| science | `pes2o` | 16B | 4B | 0 | 20B | no | no |
| science | `s2orc_arxiv_unique` | 12B | 3B | 0 | 15B | no | no |
| science | `pmc_open_access` | 8B | 2B | 0 | 10B | no | no |
| science | `proof_pile2_science` | 4B | 1B | 0 | 5B | no | no |
| science | `openstax` | 4B | 1B | 0 | 5B | no | no |
| science | `metis_freshscience_2025_26` | 6B | 3B | 1B | 10B | yes | no |
| science | `metis_freshdocs_2025_26` | 5B | 4B | 1B | 10B | yes | no |
| science | `stable_rfcs_specs` | 2B | 1B | 0 | 3B | no | no |
| science | `patents_engineering_reports` | 1B | 1B | 0 | 2B | no | no |
| synthetic | `nemotron_specialized_fact_seeking` | 5B | 10B | 0 | 15B | no | yes |
| synthetic | `nemotron_v21_dqa` | 4B | 4B | 0 | 8B | no | yes |
| synthetic | `nemotron_specialized_reasoning_scicode` | 4B | 8B | 0 | 12B | no | yes |
| synthetic | `nemotron_wiki_rewrite` | 3B | 3B | 0 | 6B | no | yes |
| synthetic | `cosmopedia_v2` | 4B | 4B | 0 | 8B | no | yes |
| synthetic | `metis_textbook_synthetic` | 3B | 9B | 0 | 12B | no | yes |
| synthetic | `metis_verified_reasoning` | 2B | 7B | 0 | 9B | no | yes |
| reference | `finewiki` | 5B | 0 | 0 | 5B | no | no |
| reference | `wikimedia_reference` | 3B | 0 | 0 | 3B | no | no |
| reference | `public_domain_books` | 4B | 0 | 0 | 4B | no | no |
| reference | `common_pile_reference` | 3B | 0 | 0 | 3B | no | no |
| reference | `roots_stackexchange` | 3B | 0 | 0 | 3B | no | no |
| reference | `open_law` | 3B | 0 | 0 | 3B | no | no |
| reference | `nemotron_legal_v1` | 2B | 0 | 0 | 2B | no | yes |
| reference | `metis_govreference` | 2B | 0 | 0 | 2B | no | no |
| multilingual | `nemotron_v21_translated_hq` | 7B | 0 | 0 | 7B | no | yes |
| multilingual | `fineweb2_native_multilingual` | 3B | 0 | 0 | 3B | no | no |

One deliberate correction was made to the proposed table: the proposed 1B of generated Nemotron
Math Proofs in Phase C was moved to the non-generated Proof-Pile-2 math allocation. “No generated
data in Phase C” is therefore a tested invariant rather than an aspiration.

Because categories describe capability rather than provenance, generated code, math proofs, and
legal material sit outside the dedicated 70B synthetic-pedagogy category. The complete manifest has
**84B explicitly generated exposures (8.4%)** and **97B generated-or-transformed exposures (9.7%)**,
both below the 15% hard cap. The executable cap uses the stricter 97B definition.

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

Fresh records carry their capture, publication, commit, or retrieval date; canonical source;
version/current/deprecated status where applicable; license; and content hash. Freshness never
overrides the quality, privacy, license, or contamination gates.

## Candidate acquisition budget

The manifest intentionally requests more candidate material than the final exposure target. Current
headroom totals are **1.49985T candidate-token estimates** and approximately **1.121572TB of
compressed packaged-source downloads**. The byte estimate is a planning heuristic, not a quota:
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

The current repository deliberately marks dynamic selection locks as `remote_source_plan` and the
normalizer rejects them. This prevents a dangerous false-success state, but it also means the
production run is **not operator-ready until the site-approved Common Crawl, GitHub/repository,
and canonical-registry materializers are connected and pass the tiny end-to-end fixture**. The
missing pieces depend on HPE's permitted egress path, approved credentials/service accounts, and
whether Portage exposes public S3 directly or through an HPE gateway. Do not ask the account owner
to run the full acquisition while this gate is red.

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
2. exact 64-bit signatures and distributed duplicate finding;
3. exact duplicate removal using deterministic source/document priority;
4. MinHash signatures over 5-grams, 20 buckets, 10 hashes per bucket, fixed seed;
5. probabilistic near-duplicate candidate clustering tuned for the 0.82 operating region (the LSH
   collision curve is not falsely described as a hard Jaccard cutoff), followed by priority-aware
   winner selection;
6. code-specific repository/file/function deduplication before packing structured code groups;
7. repeat the final content-hash check before schedule selection.

When content overlaps, prefer: cleared license, primary/canonical source, human original over a
synthetic rewrite, structurally complete extraction, current version for time-sensitive material,
higher quality score, then deterministic hash. This ordering is encoded as record priority so a
restart reaches the same winner.

## Benchmark decontamination

Evaluation data is declared separately in
[`manifests/contamination/eval-holdouts.yaml`](../manifests/contamination/eval-holdouts.yaml). It is
never a training source. The initial registry covers HLE, GPQA, MMLU/MMLU-Pro, GSM8K, MATH,
HumanEval, MBPP, SWE-bench, and LiveCodeBench.

The build creates an evaluation-only exact-hash and normalized 13-gram index. A training document is
removed on an exact match or at least two matching benchmark n-grams. Benchmark answer repositories,
mirrors, and solution collections are also excluded at repository selection. The build fails if the
holdout bundle or contamination index is missing.

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
./metisctl training-contract --release /hpe/assigned/path/metis-1.6/releases/metis-1.6-data-r1
```

## Portage operator flow

The repository should be cloned on the approved Portage login/download environment. Secrets stay in
the user's Hugging Face login/environment and HPE credential stores, never in Git. Before handoff,
commit this implementation, push it, and create the immutable `metis-1.6-data-r1` tag; the operator
must not build a trillion-token release from a moving branch.

```bash
git clone git@github.com:lernex/Metis.git
cd Metis
git checkout metis-1.6-data-r1
export METIS_LUSTRE_ROOT=/hpe/assigned/path/metis-1.6
./ops/bootstrap.sh --profile portage
```

Before production acquisition, run the full access and environment check:

```bash
./metisctl doctor --profile portage
```

Once every check and materializer gate passes, acquisition is independent of the later CPU build:

```bash
./metisctl submit download --profile portage
./metisctl status --profile portage
./metisctl report --profile portage
```

After the data is fully materialized, submit tokenizer training, filtering, exact counting, selection,
and packing:

```bash
./metisctl submit build --profile portage
```

Safely resubmit unfinished acquisition tasks:

```bash
./metisctl resume --profile portage
```

If an interrupted node left a lock behind, wait until the corresponding Slurm job is no longer
running, then remove only locks older than the 24-hour safety window:

```bash
./ops/clean-failed.sh
```

Slurm dependencies keep stages ordered and return control immediately, so the SSH session does not
need to remain open. Every array task is idempotent and has a deterministic completion marker.

## Explicitly out of scope

- No 250M-model or reduced-model ablation is part of this release plan.
- HLE and GPQA are evaluation-only and receive zero base-pretraining tokens.
- UltraData-SFT-2605 is post-training data and receives zero base-pretraining tokens.
- This data factory creates the immutable pretraining release; it does not select Portage GPU
  topology or launch the model trainer until the release and cluster-specific trainer contract are
  separately verified.
