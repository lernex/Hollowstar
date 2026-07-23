# Metis-1.6 Replacement Data Research and Policy

Status: implemented production policy
Policy: [`manifests/replacements.yaml`](../manifests/replacements.yaml)
Effective date: 2026-07-23

## Decision

Metis-1.6 does not have a generic “grab more web” fallback. Every one of the 56 planned sources is
assigned to an ordered replacement group. A source shortfall is handled in this order:

1. widen the original pinned source or dynamic route;
2. measure accepted tokens after the final tokenizer and all filters;
3. use unused, already-downloaded surplus from compatible donors in the declared order;
4. preserve the original quota source, category, phase, and freshness bucket in the release ledger;
5. stop the build if the ordered reserves cannot fill the exact quota.

The final schedule remains exactly 1T exposures. Replacement changes the physical source of a token,
not the target mixture. It may reduce generated data by replacing it with organic data; it cannot
increase generated data, move historical material into a fresh quota, move data between capability
categories, or place generated material in Phase C. The one explicit transformed-data exception is
inside the 10B multilingual category: high-quality translated NVIDIA text and native FineWeb2 text
may cover one another, while the category and phase remain fixed.

## Why these reserves

The ranking favors sources with direct quality evidence, strong metadata, enough unused capacity,
and licenses or access terms that the production pipeline can audit.

- NVIDIA reports 6.6T tokens in Nemotron-CC-v2 and 2.5T new tokens in v2.1. The v2.1 card
  distinguishes 26B high-quality organic, 16.9B medium-high organic, 53.5B medium organic,
  translated, synthetic, and DQA subsets. Metis treats those as separate provenance classes; the
  80B `nemotron_cc_v21_organic` row uses organic data only and does not silently draw from its
  synthetic partitions. [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)
  and [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1).
- FineWeb-Edu contains 1.3T score-3-or-higher educational tokens and its published ablations report
  gains over unfiltered FineWeb. Essential-Web provides a 24T taxonomy-annotated reservoir. These
  are strong web reserves, but Metis still reapplies its own global quality, license, deduplication,
  and contamination gates. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  and [Essential-Web](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0).
- Stack-Edu is a 125B educationally filtered code corpus. NVIDIA's CC-Code card reports 427.9B
  code-oriented web tokens with code/layout preservation and quality labels. They are preferable
  code reserves to unfiltered repository expansion. [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)
  and [Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1).
- Nemotron-CC-Math provides 52B 4plus tokens and 133B 3plus tokens. NVIDIA's 8B-model comparisons
  report stronger math, code, and reasoning results than FineMath, MegaMath, and OpenWebMath, so
  4plus leads the math reserve and unique score-3 material comes next.
  [NVIDIA's dataset report](https://huggingface.co/blog/nvidia/nemotron-cc-math).
- FinePDFs-Edu retains the top educational decile per language and reports stronger ablation
  performance than its unfiltered PDF parent. peS2o provides roughly 40M cleaned open-access papers
  with version and publication metadata. They lead the broad science reserve.
  [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) and
  [peS2o](https://huggingface.co/datasets/allenai/peS2o).
- FineWeb2 is filtered, deduplicated, PII-processed, ODC-By licensed, and evaluated across more
  than 1,000 languages. It is the native-language reserve rather than a generic multilingual crawl.
  [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).
- Common Pile primary-law, government, biomedical, patent, book, and reference subsets expose
  source-level metadata and license evidence. They remain subject to per-record checks because the
  project itself warns that upstream license metadata can be wrong.
  [Common Pile collection](https://huggingface.co/common-pile/datasets),
  [PubMed](https://huggingface.co/datasets/common-pile/pubmed),
  [USGPO](https://huggingface.co/datasets/common-pile/usgpo), and
  [USPTO](https://huggingface.co/datasets/common-pile/uspto).

## Exhaustive replacement matrix

The order shown is the order used by code. A source always consumes its own accepted material first,
so listing it first means “widen/reuse the same source before borrowing.” Compatibility rules may
skip a listed generated donor for an organic target.

| Replacement group | Planned sources covered | Ordered reserve |
|---|---|---|
| `web_organic` | `nemotron_cc_v2_organic`, `nemotron_cc_v21_organic`, `fineweb_edu`, `dclm_baseline`, `essential_web`, `txt360`, `zyda2_unique`, `fineweb_diversity_tail` | v2.1 organic → v2 organic → FineWeb-Edu → Essential-Web → DCLM → TxT360 → Zyda-2 → FineWeb diversity |
| `web_fresh_general` | `metis_freshweb_2026` | preferred 2026 crawls → January 2026 cold reserve → stop |
| `code_organic` | `nemotron_repository_code_v123`, `nemotron_cc_code_v1`, `stack_edu`, `stack_v2_unique`, `metis_systems_formal` | NVIDIA repository code → NVIDIA CC-Code → Stack-Edu → Stack v2 unique → formal/system registries |
| `code_fresh_software` | `metis_freshgithub_2025_26`, `metis_freshsoftwaredocs_2025_26`, `metis_freshengineering_discussions_2025_26` | recent repository code → current official software docs → licensed engineering discussions |
| `code_generated` | `verified_synthetic_code` | same verified partition → NVIDIA CC-Code → Stack-Edu → NVIDIA repository code |
| `math_organic` | `nemotron_cc_math_4plus`, `nemotron_cc_math_unique_3`, `proof_pile2_math`, `finemath_unique`, `megamath_unique`, `openwebmath_unique`, `formal_theorem_corpora` | Nemotron 4plus → unique score-3 → Proof-Pile-2 → FineMath → MegaMath → OpenWebMath → formal corpora |
| `math_generated_proofs` | `nemotron_math_proofs` | same proof set → Proof-Pile-2 → formal corpora → Nemotron 4plus → FineMath |
| `science_organic` | `finepdfs_edu_english`, `pes2o`, `pmc_open_access`, `proof_pile2_science`, `openstax`, `patents_engineering_reports` | FinePDFs-Edu → peS2o → licensed PMC → Proof-Pile-2 science → OpenStax → patents |
| `science_fresh_publications` | `metis_freshscience_2025_26` | preferred 2026 crawls with publication-date evidence → January 2026 cold reserve → stop |
| `science_fresh_official_docs` | `metis_freshdocs_2025_26` | preferred 2026 crawls with version evidence → January 2026 cold reserve → stop |
| `synthetic_grounded` | all seven dedicated synthetic-pedagogy sources | NVIDIA fact-seeking → NVIDIA DQA → verified RQA → math textbooks → reasoning/scicode → wiki rewrite → Cosmopedia |
| `reference_encyclopedic_qa` | `finewiki`, `wikimedia_reference`, `roots_stackexchange` | FineWiki → filtered Wikimedia → Stack Exchange |
| `reference_open_books` | both public-domain book rows plus LOC, DOAB, and LibreTexts | DOAB → LibreTexts → Gutenberg → Library of Congress → pre-1929 books |
| `reference_legal_government` | CAP law, USGPO, Nemotron Legal, UK Hansard, regulations | primary caselaw → USGPO → regulations → Hansard → Nemotron Legal |
| `multilingual` | `nemotron_v21_translated_hq`, `fineweb2_native_multilingual` | NVIDIA high-quality translation ↔ native FineWeb2, confined to the 10B multilingual quota |

This covers all 56 sources exactly once. The generated legal, code, proof, and pedagogy rows may be
replaced by organic material, but compatibility checks prevent the reverse direction from
increasing generated provenance.

The `nemotron_cc_v21_organic` quota deserves special clarity. Its manifest deliberately permits
only the published High-Quality and Medium-High-Quality organic partitions, which total 42.9B
upstream NVIDIA tokens; it excludes Medium-Quality and every synthetic subset. Its 80B table entry
is therefore an immutable mixture quota, not a claim that 80B unique accepted tokens exist in those
two partitions. The source resolver downloads their complete usable pinned inventory and the
final-tokenizer allocator fills the measured remainder from the ordered organic-web donors. The
release exposes the resulting physical-source totals instead of pretending the original reservoir
was larger.

## Fresh-data reserve

The preferred fresh selection still uses `CC-MAIN-2026-08`, `-12`, `-17`, `-21`, and `-25`.
`CC-MAIN-2026-25` is the June 2026 release: Common Crawl reports 2.10B pages, 479M previously unseen
URLs, and a 900-file columnar URL index. `CC-MAIN-2026-04` is a January cold reserve with 2.3B pages
and 616M previously unseen URLs. The implementation does not mirror either crawl. It queries the
columnar index and fetches selected WARC ranges.
[June release](https://commoncrawl.org/blog/june-2026-crawl-archive-now-available) and
[January reserve](https://commoncrawl.org/blog/january-2026-crawl-archive-now-available).

The cold reserve is not mixed in by default. Automatic acquisition first exhausts all permitted
selection rounds across the five preferred crawls. Only a measured shortfall activates January.
The same domain, structure, date/version, language, license, publisher opt-out, privacy, quality,
global exact, repeated-span, MinHash, and benchmark-decontamination rules apply. If the combined
route remains short, acquisition stops; old generic web cannot fill a 2026 freshness bucket.

## Candidate headroom and storage effect

The policy raises the planned acquisition envelope from 1.586231T to **1.7207509T estimated
candidate tokens**, and from 1.264871TB to **1.369685297TB planned compressed source payload**.
That is 134.5199B more candidate-token headroom and about 104.8GB more planned compressed payload,
not 134.5B extra training tokens. Actual WARC, repository-object, metadata, and scratch amplification
is still governed by the operator capacity gates.

At planning time, donor surplus can simulate the complete loss of 53 of 56 sources. The remaining
three are deliberately singleton fresh routes (`metis_freshweb_2026`, `metis_freshscience_2025_26`,
and `metis_freshdocs_2025_26`), because replacing them with historical or differently licensed
material would violate the freshness contract. Each has automatic cold-reserve widening instead.

This simulation is a reserve-planning test, not a promise that arbitrary future filtering losses
will always fit. The final-tokenizer allocator reruns on measured accepted counts and fails closed
if the real reserves are insufficient.

## Large datasets evaluated but not promoted to first-line reserves

- **Nemotron v2.1 synthetic rewrites:** large, but they would increase generator-style exposure and
  violate an organic source quota. They remain eligible only in explicitly synthetic rows.
- **Medium-quality v2.1 organic:** useful as emergency diversity material, but lower priority than
  the high and medium-high organic subsets, FineWeb-Edu, and Essential-Web. It is not silently mixed
  into the `nemotron_cc_v21_organic` row.
- **Dolma/Dolma 3 broad pools:** high-value open research assets, but broad pre-mixing makes them a
  less precise replacement than the source-native donors already in the recipe. Adding them would
  also duplicate large portions of the web, code, paper, and reference sources already selected.
  [Ai2 data documentation](https://docs.allenai.org/training_data/dolma).
- **UltraData-Math and other large synthetic math mixtures:** useful for post-training or a
  separately measured synthetic ablation, but not a safer replacement for organic equation-rich
  pretraining text. NVIDIA's published organic math comparisons give the current first-line donor
  stronger direct evidence.
- **Unfiltered S2ORC, broad Pile-of-Law mirrors, repacked token streams, and third-party
  derivatives:** rejected as replacement pools when their underlying per-document license,
  provenance, tokenizer, or contamination history cannot be reconstructed.

## Runtime guarantees

The source lock, acquisition handoff, exact selection, packed shard index, verification report, and
release manifest all retain both:

- `source_id`: the source that physically supplied the token;
- `quota_source_id`: the immutable source row and phase quota that the token filled.

Every transfer is emitted as a deterministic replacement receipt. Verification reconstructs the
allocation from accepted counts, checks the donor policy, recomputes exact source/category/phase and
freshness totals, prevents generated-data growth, and checks the global generated-or-transformed
cap. There is no semantic deduplication; exact, repeated-span, MinHash near-duplicate, and
code-structural passes remain the production deduplication system.
