# Metis-1.6 Data Prep Throughput Plan

Date: 2026-05-14

## Context

Metis-1.5 CPU prep proved that the current pipeline is correctness-first and resumable, but not the fastest possible design for larger data volumes. The 1.5 run already pushed the single orchestrator hard: 48 normalize workers on a 32 CPU pod, 16-way source partitioning, high load, healthy memory, and strong network throughput. Past that point, raising worker count is likely to create contention rather than meaningful speedup.

Metis-1.6 should move from a single large normalization process with source partitions to a shard-level distributed prep architecture.

## Goal

Build a CPU prep path that can normalize, train tokenizer samples, and tokenize much larger Metis-1.6 mixtures without waiting hours on one pod-bound scheduler.

Target outcome:

- Use many independent source/shard jobs safely.
- Avoid repeated Hugging Face/S3 downloads for the same remote shards.
- Keep strict bucket/source accounting.
- Preserve English-only filtering, chunking, dedup hooks, and contamination metadata.
- Make all outputs idempotent and restart-safe.

## Proposed Architecture

### 1. Per-Source Job Split

Instead of one `prepare_normalized_shards.py` run managing all sources, launch each source as its own resumable job:

- One job per source for small sources.
- Multiple jobs per source for large sources.
- Each job writes to a unique source/partition output prefix.
- Parent orchestrator only schedules, tracks, and merges manifests.

This prevents slow sources from blocking fast ones and makes it easier to scale across multiple CPU pods.

### 2. Shard-Level Partitioning

Partition by upstream shard/file, not just source-level synthetic partitions.

For example:

```text
source = cpt_nemotron_cc_math_4plus
job 000 = parquet/zst shards 0000-0031
job 001 = parquet/zst shards 0032-0063
...
```

This gives real work separation and avoids multiple workers fighting over the same upstream shard iterator.

### 3. Local Hydration Cache

Before normalization, download remote dataset shards to local NVMe once:

```text
/workspace/metis16_cache/hf_shards/<dataset>/<revision>/<repo_path>
```

Then normalize from local files.

Benefits:

- Avoids repeated HF streaming overhead.
- Allows faster decompression and chunking.
- Reduces network variability.
- Makes restarts cheaper.
- Allows multiple normalization workers to read local data instead of hammering the same remote endpoints.

Cache should be size-capped and source-aware. Huge sources can hydrate only the shard range assigned to each job.

### 4. Idempotent Output Prefixes

Every job should write to a deterministic prefix:

```text
s3://.../metis16/normalized-shards/<stage>/<source>/part-000123/
```

Each job writes:

- `manifest.json`
- `shard-00000.jsonl.zst`
- `shard-00001.jsonl.zst`
- `samples.jsonl.zst` if needed for audit

A job is complete only if `manifest.json` exists and passes validation. Partial job prefixes can be safely deleted and retried without touching other jobs.

### 5. Manifest Merger

Add a separate merge step:

```text
normalized job manifests -> source manifest -> stage manifest
```

The merger should verify:

- planned docs/tokens by source
- actual docs emitted
- skipped rows
- exhausted sources
- fallback replacements
- bucket totals
- source caps
- English pass rate
- duplicate/contamination flags when available

No global fallback. Same-bucket fallback only.

### 6. Tokenization Jobs

Tokenization should also be shard-level:

```text
normalized shard -> tokenized shard
```

Instead of one process packing the whole stage, launch independent tokenization jobs that produce deterministic `.bin` or chunked token files, then merge/concatenate with a final index manifest.

This makes tokenization scale with CPU count and makes failed shard retries cheap.

### 7. Metrics And ETA

Every job should emit machine-readable progress:

```json
{
  "source": "cpt_arxiv_math_physics_cs",
  "job_id": "part-0007",
  "stage": "normalize",
  "docs_done": 42193,
  "docs_target": 60000,
  "bytes_read": 18374628192,
  "bytes_written": 912384512,
  "docs_per_sec": 843.2,
  "rss_gib": 1.4,
  "updated_at": "..."
}
```

The orchestrator should report ETA from recent deltas by active job, not from stale global averages.

## Expected Speedup

On one pod, shard-level partitioning should reduce source stalls and likely improve throughput by roughly 1.5-3x for mixed-source prep, depending on network and decompression bottlenecks.

Across multiple CPU pods, the speedup should be closer to horizontal scaling for large remote-shard sources, as long as S3/HF limits are respected. A 4-pod layout could plausibly turn multi-hour normalization into tens of minutes for comparable token counts.

The biggest gains should come from:

- local shard hydration
- independent per-source jobs
- tokenization parallelized by normalized shard
- cheap retries instead of whole-stage restarts

## Implementation Checklist

- Add `scripts/metis16_hydrate_shards.py`.
- Add `scripts/metis16_normalize_job.py`.
- Add `scripts/metis16_merge_normalized_manifests.py`.
- Add `scripts/metis16_tokenize_job.py`.
- Add `scripts/metis16_merge_tokenized_shards.py`.
- Add a lightweight orchestrator for local multi-process and multi-pod modes.
- Store job state under `state/metis16/jobs/*.json`.
- Use deterministic S3 prefixes for every job.
- Validate bucket/source totals before packing.
- Add a cleanup command that deletes only failed/incomplete job prefixes.
- Keep a compatibility path that can run on one CPU pod, but make multi-pod sharding the intended path.

## Non-Negotiables

- No duplicate S3 shards after restart.
- No cross-bucket fallback.
- No silent source exhaustion.
- No language drift: English-only filters remain document/file/row level.
- Long-document chunking happens before doc-count accounting.
- Token counts and source shares are verified after tokenization, not assumed from row counts.

