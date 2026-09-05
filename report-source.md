# Research source record — Metis-1.7 200 TB pretraining corpus

> **Historical snapshot, superseded.** This preserves the research and arithmetic
> committed on September 4, 2026; it is not the current acquisition policy.
> The 25-family basket and 42T/45T unique-token gate below were replaced by the
> [current research and exposure plan](docs/metis17_200tb_pretraining_corpus_research.md)
> and [current acquisition ledger](docs/metis17_200tb_acquisition_ledger.csv).
> Historical byte totals refer to the
> [original ledger at `1d275538`](https://github.com/lernex/Hollowstar/blob/1d275538516d4790833b0a6e6d66c8476bdb9962/docs/metis17_200tb_acquisition_ledger.csv),
> not to that newer ledger.

Research cutoff: 2026-09-04  
Question: Which current, content-bearing corpora should Metis-1.7 acquire to maximize quality and breadth for 30T TST source-token exposures plus 5T NTP cooldown exposures, while making the gross acquisition approximately 200 TB and preserving freedom to choose mixture ratios only after preparation?

## Decision summary

- The recommended acquisition ledger contains 25 canonical source families, not hundreds of top-level repositories. Those families already expand to thousands of configurations, languages, snapshots, and specialist partitions. Splitting one publisher family into hundreds of names would add operational complexity rather than information diversity.
- The selected objects total exactly 200,021,721,850,099 decimal bytes (200.021721850 TB; 181.918696262 TiB) at the pinned revisions and 2026-09-04 external HPLT manifest snapshot.
- That number is compressed transfer payload, not decoded UTF-8. The user's 4.5 bytes/token arithmetic is a decoded-text calculation. The selected basket represents well over 200 TB after decoding and cannot be provisioned as a 200 TB working set.
- A byte target cannot prove that 35T usable Metis-tokenizer tokens survive cross-source deduplication, decontamination, license filtering, normalization, and quality gates. Acquisition should stop only after a frozen unique-token inventory reaches at least 42T and preferably 45T accepted tokens.
- Every selected code source contains inline code/text. Known pointer-only partitions are explicitly excluded.
- The LaTeX path is strong: arXiv source LaTeX, UltraData-Math, Nemotron-CC-Math, Nemotron-CC-Code, FinePDFs, Common Corpus, and TeX files inside The Stack v3.
- No credible large TOON-native corpus was found. TOON should be a deterministic, version-pinned derived paired representation built from acquired structured records, while retaining JSON.

## Arithmetic record

The historical per-source bytes are preserved in the
[original acquisition ledger](https://github.com/lernex/Hollowstar/blob/1d275538516d4790833b0a6e6d66c8476bdb9962/docs/metis17_200tb_acquisition_ledger.csv).

- Selected compressed transfer bytes: 200,021,721,850,099.
- Decimal TB: 200.021721850099.
- Binary TiB: 181.918696262164.
- 35T tokens at 4.5 decoded bytes/token: 157.5 TB.
- 200 TB decoded at 4.5 bytes/token: 44.444T tokens before rejection.
- Maximum tolerable rejection before falling below 35T: 21.25%.
- Decoded bytes required at 4.5 bytes/token with 30% rejection: 225.0 TB.
- Decoded bytes required at 4.5 bytes/token with 35% rejection: 242.308 TB.
- Decoded bytes required at 4.5 bytes/token with 40% rejection: 262.5 TB.
- Decoded bytes required at 4.5 bytes/token with 50% rejection: 315.0 TB.

## Evidence ledger

| Source | Date | Primary evidence used | Decision-relevant fact |
|---|---|---|---|
| FineWeb | July 2025 update | https://huggingface.co/datasets/HuggingFaceFW/fineweb | 18.5T+ GPT-2 tokens; default payload duplicates named sample/config views; acquire only physical `data/` payload. Foundational exception is supported by the May 2026 TST paper's continuing use of FineWeb-Edu. |
| FineWeb 2 | October 2025 v2.1.1 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | 1,868 language-script configurations from 96 snapshots; `_removed` and test configurations are intentionally not training payload. |
| Essential-Web | October 2025 | https://huggingface.co/datasets/EssentialAI/essential-web-v1.0 | 24T tokens and rich document-level taxonomy/quality metadata; strong base for post-prep mixture design. |
| DCLM baseline | July 2024 release; May 2026 current-use confirmation | https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet and https://arxiv.org/abs/2605.06546 | The 3.88T baseline remains relevant because the 2026 TST 10B-A1B reference used DCLM with FineWeb-Edu. Raw 240T DCLM is not selected. |
| Nemotron 3 pretraining family | December 2025–July 2026 | https://docs.nvidia.com/nemotron/nightly/nemotron/lightning35/pretrain.html | NVIDIA's current recipe confirms the CC v2/v2.1, CC Math, CC Code, code, and specialized families remain part of a frontier open recipe. |
| Nemotron CC v2.1 | December 2025 | https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1 | 2.5448T new tokens intended to complement v2; inline `text`; organic, translated, DQA, and synthetic categories. |
| Nemotron CC Math | August 2025 release; 2026 ICLR paper cycle | https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1 | Quality-3 and 4plus raw partitions are disjoint; MIND is derived; equations normalized to LaTeX; benchmark decontamination is reported. |
| Dolma 3.5 | July 2026 | https://huggingface.co/datasets/allenai/dolma3.5_pool | Nearly 10T tokens, PDFs through November 2025, and 1.351T+ code tokens with inline text. |
| The Stack v3.1 train | September 2026 | https://huggingface.co/datasets/HuggingFaceCode/stack-v3-train | `files[].content` is inline; 3.6T tokens after fixing a partition/dedup leak; cutoff 2025-08-07. |
| Common Corpus | May 2026 update / ICLR 2026 | https://huggingface.co/datasets/PleIAs/common_corpus | 2.267T tokens with document-level provenance/license across culture, government, science, code, and web. |
| FinePDFs | April 2026 | https://huggingface.co/datasets/HuggingFaceFW/finepdfs | About 3T tokens, 476M documents, 1,733 language-script pairs; test splits must not be trained. |
| UltraData-Math | February–April 2026 | https://huggingface.co/datasets/openbmb/UltraData-Math | 290B+ tokens; MathML, KaTeX, and AsciiMath normalized into LaTeX; distinct L1/L2/L3 strata. |
| MegaMath | April 2025 release; 2026 current-use confirmation | https://huggingface.co/datasets/IFM/MegaMath and https://huggingface.co/datasets/openbmb/UltraData-Math | Older exception because the 2026 UltraData work retains it as source/baseline; pointer-only code partition excluded. |
| Darwin-CC | March 2026 | https://arxiv.org/abs/2603.14420 and https://huggingface.co/datasets/GAIR/Darwin-CC | 504B-token Nemotron-CC derivative with evolved cleaners; parent relation must be tagged for cross-source dedup. |
| Ultra-FineWeb L1/L3 | August 2026 | https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1 and https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3 | Fresh 2025 organic crawl plus labeled English/Chinese synthetic expansions. |
| HPLT 3.0 | November 2025 paper | https://arxiv.org/abs/2511.01066 and https://data.hplt-project.org/three/sorted/manifest.json | 198 languages, 29.831T reported tokens, content-bearing compressed objects, and WDS quality bins. |
| arXiv LaTeX | August 2026 | https://huggingface.co/datasets/scholarweave/arxiv-latex | 3.12M rows with full source, bibliography/style material, dates, and per-paper licenses. |
| Common Crawl freshness reserve | May–August 2026 | https://commoncrawl.org/blog/may-2026-crawl-archive-now-available, https://commoncrawl.org/blog/june-2026-crawl-archive-now-available, https://commoncrawl.org/blog/july-2026-crawl-archive-now-available, and https://commoncrawl.org/blog/august-2026-crawl-archive-now-available | Four current WET releases total 23.27 TiB (about 25.586 decimal TB) compressed and are reserve-only because they are raw crawl text rather than finished frontier-quality data. |
| TOON specification | July 2026 v4.1 working draft | https://github.com/toon-format/spec/blob/main/SPEC.md | A versioned indentation/tabular encoding of the JSON data model; no large authoritative pretraining corpus is published. |
| Independent TOON evaluation | February 2026 | https://arxiv.org/abs/2603.03306 | Plain JSON remained strongest overall in the reported tests; TOON's prompt/schema overhead can erase token savings on small/simple structures. |

## Evidence gaps and limits

- Publisher token totals use different tokenizers and cannot be added as an exact Metis token count.
- Several Hugging Face viewer “disk size” badges are logical/decoded estimates; Git/LFS/Xet object sizes are transfer bytes. Both can drift when publishers repack shards without changing semantic content.
- Gated NVIDIA and Darwin payloads require license acceptance. Their object manifests were visible through the repository API, but row-level content was not downloaded during this research.
- Cross-family Common Crawl overlap is undisclosed. It must be measured on normalized records; publisher-local dedup does not settle cross-publisher duplication.
- Synthetic rewrites can be semantically duplicative without being MinHash duplicates. Keep generation-parent/category metadata so later sampling can cap clusters.
- The Stack v3 top-line card currently contains both a 15.9 TB statement and an 11.5 TB decoded train-table figure. The pinned Parquet transfer payload is 3.546 TB. The ledger uses the file manifest, and the report treats 11.5 TB / 3.6T tokens as the internally consistent decoded/train row.
- Essential-Web's Dataset Viewer size endpoint was unavailable during the audit; exact transfer bytes were summed from its pinned file manifest.
- No public CuraWeb payload was verified at the research cutoff.
- No large authoritative TOON corpus was verified at the research cutoff.

## Exclusion record

- `nvidia/Nemotron-Pretraining-Code-v3`: metadata pointers (`repo`, `rel_path`, `commit_id`), not inline code.
- `Nemotron-Code-Metadata` partitions in Code v1/v2: pointers, not code payload.
- `IFM/MegaMath/megamath-code`: pointers, not code payload.
- The Stack v2 and similar Software Heritage reconstruction paths: excluded from core acquisition because content retrieval would recreate the Metis-1.6 pointer bottleneck.
- FineWeb sample configurations: duplicate subsets of the selected default physical payload.
- FineWeb 2 `_removed` and test splits: removed-quality material or evaluation data.
- FinePDFs test splits: evaluation data.
- FinePDFs-Edu / FineWeb-Edu as separate downloads: subsets of acquired parent families, not independent bytes. Preserve or rederive membership labels instead.
- GneissWeb: current selector/classifier artifacts, not a released 10T-token text payload.
- Propella annotations: metadata sidecar, not primary text.
- Raw DCLM 240T pool, TxT360 version pile, raw Common Crawl WET: reserve-only because core curated content already fills the acquisition target; use only if the unique-token gate misses.
- C4, SlimPajama, RedPajama, RefinedWeb, OpenWebMath, Proof-Pile-2, and legacy global KenLM/Gopher-filtered recipes: superseded as primary bulk sources. They can remain provenance references, not core bytes.

## Reproducibility method

- Hugging Face revisions and file sizes were read from `https://huggingface.co/api/datasets/<id>?blobs=true` on 2026-09-04.
- Only content-bearing file selectors in the CSV were summed. README, `.gitattributes`, evaluation output, sample duplicates, removed data, test splits, and pointer-only partitions were excluded.
- HPLT totals came from the publisher NDJSON manifest. Fourteen named low-score English objects were subtracted by HTTP object `Content-Length`; the selected HPLT total is 56,526,801,884,263 bytes.
- All sizes are decimal bytes unless explicitly labeled TiB.
