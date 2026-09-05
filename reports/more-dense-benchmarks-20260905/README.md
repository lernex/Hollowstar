# Dense FLOP-matched benchmark results, 5 September 2026

All ten requested benchmarks completed in Slurm job `496031`, on one Portage
node with four independent MI300A workers. The allocation ran from 20:19:11 to
20:38:49 UTC and exited successfully. These are full-split scores, not the
two-example-per-task startup diagnostics.

| Benchmark | Shots | Examples | Accuracy (%) | Normalized accuracy (%) |
|---|---:|---:|---:|---:|
| MMLU | 5 | 14,042 | 24.60 | -- |
| MMLU-Pro (MC, non-CoT) | 5 | 12,032 | 11.65 | -- |
| ARC-Easy | 0 | 2,376 | 52.06 | 46.00 |
| ARC-Challenge | 0 | 1,172 | 21.50 | 25.26 |
| HellaSwag | 0 | 10,042 | 31.29 | 35.10 |
| Winogrande | 0 | 1,267 | 50.20 | -- |
| PIQA | 0 | 1,838 | 66.38 | 65.45 |
| BoolQ | 0 | 3,270 | 60.52 | -- |
| OpenBookQA | 0 | 500 | 19.40 | 30.20 |
| LAMBADA | 0 | 5,153 | 20.73 | -- |

LAMBADA final-word perplexity is **67.47754077945704**. Normalized accuracy is
the harness's `acc_norm` metric; raw accuracy is also retained rather than
silently substituting one for the other.

**The general benchmark results are weak despite the lower training loss.**
MMLU and Winogrande are near their uniform-choice baselines of 25% and 50%.
These measurements establish the dense reference, not a MoRE advantage or
a valid comparison against the original expert-cache-affected sparse runs.

## Model and protocol

The evaluated checkpoint is the completed dense FLOP-matched control at
step **25,429**, with **49,995,448,320 training tokens** and
**1,437,743,410 stored parameters**. Its run identity is
`736f74c8a6fe978d535cea0ce9f2905acaee43f957646abcada76c54512797e6`.
The exact checkpoint checksum is in `results.json` and every worker's metadata.

Native model execution uses the original training source
`ba9809284cf787d3e69df96620048dd96345d66f`, not the concurrently evolving model
implementation on `main`. The evaluator and launcher are pinned to
`37eed214c2398375a77149a1e6e196e494730d6b`; task preparation is pinned to
`515edaf47db8e7c13e24a1712c6a00d89ca47b4c`.
The new evaluation modules are loaded alongside, rather than in place of,
the original native packages.

The model uses its saved `depth_one` curriculum, recurrent-memory gate 0,
and N-gram gate 1. The released 65,536-token tokenizer and canonical-ID mapping
are restored, and canonicalization is recomputed against the sealed sidecar.
Execution is BF16 while retaining the model's mixed BF16/FP32 parameter-storage
policy. Optimizer shards are never opened.

The benchmark implementation is
[lm-evaluation-harness v0.4.13, released August 2026](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13).
Dataset commit pins and file checksums are recorded below. All **51,692**
evaluation examples and **244,417** likelihood requests are covered.
The maximum model input is **3,155 tokens**, below the unchanged **4,096-token**
limit. No evaluation examples or few-shot context were dropped or truncated.

MMLU uses five demonstrations. MMLU-Pro uses five category-matched
direct-answer demonstrations, valid answer-letter likelihoods, and the
harness's size-weighted group aggregation. This **non-CoT MC variant is not
directly comparable to the official MMLU-Pro CoT-generation scores**.
The other eight tasks use zero-shot likelihood protocols. There is no chat
template, and GSM8K is intentionally excluded.

Each APU owns whole tasks; there is no custom distributed metric averaging.
Identical contexts for single-token choices share an exact native forward.
Before any full scoring, the complete per-task request hashes are compared
with preparation. The result collector requires the expected checkpoint,
native source, dataset revisions, shots, full sample counts, and successful
request-identity gate.

## Records and regeneration

- `results.json`: exact aggregate scores, primary metric choices, standard errors,
  checkpoint identity, and training/evaluation accounting.
- `workers/worker-*/harness-results.json`: unmodified full harness aggregates,
  including all 57 MMLU subjects and 14 MMLU-Pro categories.
- `workers/worker-*/metadata.json`: completion status, input/source hashes,
  software versions, effective curriculum, precision, and scorer counters.
- `preparation.json`: full task and request census, fingerprints, context bounds,
  category coverage, and four-worker assignment.
- `download-provenance.json`: pinned dataset file names, byte counts, and SHA256s.
- `submission.json` and `accounting.tsv`: actual submission and final scheduler
  accounting; accounting timestamps are UTC.

No benchmark questions, answers, samples, or credentials are committed.
The initial dataset download hit a Hub throttle after cache isolation stopped
discovering the existing configured authentication. Pinned files were cached
with the installed Hub SDK 1.27.0 and its native rate-limit handling, without
changing the shared training environment or copying credentials.

Regenerate the machine-readable summary and paper table from the repository root:

```bash
python3 scripts/summarize_more_benchmarks.py \
  --suite configs/more_eval_suite.json \
  --report reports/more-dense-benchmarks-20260905 \
  --table docs/papers/more/dense_benchmark_results.tex
```

The native launcher is `slurm/ablation/dense-eval.sbatch`. It uses a separate
evaluation virtualenv with the read-only training-stack overlay in
`ops/more-eval-training-stack.pth` and the constraints in
`requirements-more-eval.txt`. Preserve an existing `HF_TOKEN_PATH` when
isolating `HF_HOME` for preparation; do not copy credentials into the cache.
