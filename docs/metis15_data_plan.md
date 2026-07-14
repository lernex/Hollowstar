# Metis-1.5 Data Plan

Metis-1.5 uses a 60B-token base-plus-CPT plan:

- Base pretrain: 50B tokens.
- Continued pretrain / midtrain: 10B tokens.
- Post-training: 1.2M chat SFT examples, 600K reasoning SFT examples, 400K reward-model preference pairs, and 300K DPO pairs.

The release plan is only:

- `Lernex/Metis-1.5-base`
- `Lernex/Metis-1.5-think`

The chat SFT stage remains an internal bridge into think training, not a separate release target.

## Base Pretrain, 50B

| Bucket | Tokens | Share | Config |
| --- | ---: | ---: | --- |
| High-quality web: DCLM/DCLM-edu/DCLM-HQ | 12B | 24% | `high_quality_web_dclm` |
| High-quality web: FineWeb-Edu/FineWeb-HQ | 9B | 18% | `high_quality_web_fineweb` |
| Nemotron-CC-v2 / newer English web | 5B | 10% | `nemotron_quality_web` |
| Reference / encyclopedic | 3B | 6% | `reference_encyclopedic` |
| Academic STEM / science | 5B | 10% | `academic_stem_science` |
| Math / proof / symbolic pretrain | 6B | 12% | `math_proof_symbolic` |
| Open textbooks / educational reference | 3B | 6% | `open_textbooks_educational_reference` |
| Long-form books | 2B | 4% | `long_form_books` |
| Knowledge QA / explanations | 2B | 4% | `knowledge_qa_explanations` |
| Synthetic educational prose | 2B | 4% | `synthetic_educational_prose` |
| Reserve / balancing pool | 1B | 2% | `reserve_balancing_pool` |

Config: `configs/metis15_pretrain_mix.json`.

## CPT / Midtrain, 10B

| Bucket | Tokens |
| --- | ---: |
| High-score DCLM/FineWeb replay | 1.2B |
| Academic STEM | 2.0B |
| Math/proof documents | 2.0B |
| Verifiable math/problem-solution prose | 1.2B |
| Reference/wiki/StackExchange | 1.2B |
| FinePDFs OCR-quality English technical docs | 0.8B |
| Science instruction/literature tasks as text | 0.6B |
| Long-form/book replay | 0.5B |
| Hard general decontaminated examples | 0.5B |

Config: `configs/metis15_continued_pretrain_mix.json`.

## Post-Training

Chat SFT target: 1.2M examples from Tulu 3, NoRobots, SmolTalk2, filtered WildChat, SciRIFF, SciInstruct, OpenStax-derived science QA, OpenR1, NuminaMath, a tiny Metis-1.5 identity source, and a capped OpenHermes slice.

Reasoning SFT target: 600K examples from OpenR1-Math, NuminaMath, OpenThoughts3, OpenMathInstruct-2, SciRIFF, Bespoke-Stratos, s1K, and TemplateGSM capped at 25K.

Preference target:

- 328K bootstrap reward-preference pairs plus 72K on-policy pairs = 400K reward-model pairs.
- 246K bootstrap DPO pairs plus 54K on-policy pairs = 300K DPO pairs.
- HelpSteer3, Skywork v0.2, UltraFeedback cleaned, Tulu/OLMo preference mixes, math/science verifier pairs, and safety/helpfulness pairs are all explicit buckets.
- Nectar is capped at 10% final share.
- On-policy Metis pairs are generated after chat and think checkpoints exist.

## Required Hygiene

All Metis-1.5 configs are bucketed. Fallback must remain inside the same bucket. Every pretrain/CPT source carries English, dedup, code-ratio, and contamination metadata requirements. The local prep scripts enforce bucket-local fallback and a first-pass Latin-script guard; the full data build still needs the stronger fastText/CLD3-style LID, global MinHash dedup, and benchmark contamination pass before spending the serious GPU run.

Run:

```bash
make metis15-validate-data-plan
```
