# Metis-1.6 data preparation: what went wrong and what 1.7 must do differently

Written 2026-08-02, at the close of Metis-1.6 acquisition.

Acquisition took **eleven days**. About one day of that was moving bytes. The
rest was four distinct classes of failure, none of which were about the data
being large. This document exists so 1.7 pays for each lesson once.

The single most useful sentence: **the failures were never where the volume
was.** 1.4 TB of packaged datasets moved without a single incident. Everything
that hurt was small, structural, or someone else's publishing decision.

---

## 1. Acquisition shape decides cost, not token count

The manifest listed 56 sources as if they were one kind of thing. They are
three, with cost models that differ by orders of magnitude.

| shape | what it is | cost driver | 1.6 outcome |
|---|---|---|---|
| `hf_snapshot` | dataset ships content | bytes | 54 sources, 1.4 TB, no incidents |
| pointer corpus | dataset ships `(repo, commit, path)` or blob IDs | row count | 3 sources, consumed a week |
| crawl build | corpus is reconstructed from Common Crawl | requests + random I/O | 4 sources, produced 184 KiB in 11 days |

**Do for 1.7:** declare `acquisition_shape` in the manifest as a first-class
field. Budget build-shaped sources by row count and request count, never by
bytes. Schedule them **first**, because they hold the longest critical path and
surface their surprises late.

**Audit columns, never names.** Three of nine code sources — Nemotron-Code
v1–v3, `stack_edu`, `stack_v2_unique`, 55% of the code budget — shipped indexes
rather than text and normalized to zero documents. `stack_edu` was missed on the
first pass because its name and reputation both say "code dataset." The check
that works is reading the actual payload columns of downloaded files against
`normalization_evidence._PAYLOAD_FIELDS`.

---

## 2. The acquisition host determines what is even possible

This is the expensive lesson. The entire 90B-token freshness layer was
withdrawn, and 1.6 ships **35B of freshness ending 2025-06** instead of 90B
ending 2026-06.

Not because Common Crawl is hard. Because of three properties of the host that
stack:

1. **Only login nodes have egress.** Compute nodes cannot reach
   `data.commoncrawl.org`. So acquisition is pinned to a login node.
2. **That node has no local scratch.** `sda` is unmounted, `/home` is NFS, `/`
   is an overlay. So the acquisition ledger lives on Lustre.
3. **The ledger is a `WITHOUT ROWID` B-tree keyed on canonical URL.** Inserts
   land at uniformly random positions. On Lustre a page read is a network round
   trip.

Result: a **663 GB** `state.sqlite3` with a **319 GB open rollback journal**,
1.5 TB of ledger across four sources, and 184 KiB of documents.

Tuning was tried and measured, so do not retry it: widening the Common Crawl
lane from 1 to 4 gave **2.2×** (0.96 → 2.10 cores). `index_scan_workers` is
already at its ceiling of 16. Neither is within orders of magnitude of enough.

**A supercomputer does not help here, and it is worth understanding why.** The
job does approximately zero FLOPs and consists of small random I/O plus many
small network requests. Top-500 standing comes from dense matrix math with a
fast interconnect. Lustre is built for a thousand ranks streaming one giant
checkpoint, which is the exact opposite access pattern. Frontier labs run this
work on cloud object storage across many cheap VMs that each hold their own
internet connection — different architecture, not more horsepower.

**Do for 1.7:** run crawl-shaped acquisition on a machine with (a) direct
egress, (b) node-local NVMe, (c) a filesystem that is not Lustre. Keep packaged
downloads and training on the HPC system. Then:

- **Move filtered output, not working sets.** Normalize and dedup at the source;
  ship only survivors. The difference is roughly 1 TB versus 10 TB.
- **Price object storage with zero egress** (e.g. R2) against GPU-pod volumes.
  The crawl is disk-and-egress heavy and needs no GPU.
- **Build the import path before either half runs.** The manifest, source lock,
  receipts, and `verify-handoff` currently assume one coherent state store on
  one machine. Splitting acquisition needs a first-class notion of *acquired
  elsewhere, imported with receipts* — a design task, not a config change.

**Scale warning:** at 30T tokens, packaged acquisition alone is ~150–200 TB and
on the order of two to three weeks of continuous transfer at the rate that
worked here.

---

## 3. Run fingerprints turn throughput knobs into destructive operations

`FreshWebOptions.max_workers` is a `FINGERPRINT_OPTION_FIELD`. `process_workers`
was computed as `min(10, 10 // lane_workers, range_workers)`. Therefore raising
`lane_max_workers.common_crawl` from 1 to 4 would have **halved it, changed
every in-flight run's fingerprint, and orphaned about a terabyte** of already
fetched partitions. Fixed in `fffef20`: lane width is scheduling, not identity.

`candidate_tokens` is also fingerprinted, so **retargeting a source discards its
work**. Raising `metis_freshsoftwaredocs` from 10B to 35B silently orphaned
136 GB.

**Do for 1.7:** every field that reaches a run fingerprint should be listed in
one place with a comment saying so, and any config key that feeds one should
name it. A knob that looks like tuning and behaves like `rm -rf` is a trap
regardless of how well the fingerprint mechanism itself works.

---

## 4. Benchmark repositories have no format contract

The decontamination registry pins **63 repos across 203 jobs**, and each is a
different research group's idea of how to publish. Four separate failures, each
one discovered only after everything before it succeeded:

| failure | cause | fix |
|---|---|---|
| `ImportError: Pillow` | MMMU/MathVista image columns decoded during streaming | cast to `decode=False`; the decoded object was discarded anyway |
| `trust_remote_code` prompt | `maveriq/bigbenchhard` ships a loading script | pin `lukaemon/bbh`, a data-only mirror |
| 1.8M single-word fragments | Natural Questions stores `{"text", "tokens"}`; the extractor recursed into token lists | skip tokenization keys |
| 7 GB registry at 424 MB/min | NQ attaches full Wikipedia **HTML** per row | skip markup keys; bound the row |
| `JSONDecodeError` at char 3414 | `str.splitlines()` splits on U+2028/U+2029/U+0085, all legal raw inside JSON strings; LongBench `triviaqa_e.jsonl` contains two | split on `\n` only |

**The general fix that actually converges** is not naming fields as you trip
over them. It is bounding the row by construction: skip content that cannot
match normalized text, drop any single fragment too large to be a benchmark
item, then cap total characters per row regardless of key names. Only that last
bound needs no knowledge of the schema — and it would have caught all of the
above without being told about any of them. Every bound now reports itself in
`HOLDOUTS.json`.

**Do for 1.7:** add a preflight that resolves every pinned benchmark's file list
and flags loading scripts *before* a multi-hour run starts. Sweeping all 63
repos via the HF API takes about a minute; discovering them one failure at a
time took a day.

---

## 4b. `per_record_required` was asserted for data that has no per-record license

Over half the corpus -- 1,016 of 1,849 files across nine sources -- normalized
to **zero accepted records**, every one rejected `missing_license`. Not a code
fault: the manifest declared `license.status: per_record_required`, which means
*each row carries its own license*, for sources where that is impossible.

The conflation is subtle and worth naming precisely. `per_record_required` and
"the upstream corpus aggregates differently-licensed material" sound like the
same statement and are not. The second is true of DCLM, Zyda-2, Essential-Web,
TxT360, OpenWebMath and MegaMath; the first is false of all of them, because a
Common Crawl derivative ships url, text and quality signals and has no
per-document license to ship. What those publishers actually grant is a
compilation licence -- CC-BY-4.0 or ODC-By-1.0 on the dataset card. The
manifest already recorded exactly that shape correctly for FineWeb, so the
correct value was one line away in the same file.

Two of the nine were different, and the difference only showed up by reading
real rows:

- **Proof-Pile-2 genuinely has per-record licences**, in `meta.
  max_stars_repo_licenses` (e.g. `["MIT"]`). `_find` descends into `metadata`
  but not `meta`, so a source that fully satisfied the policy failed it. Fixed
  by reading the field, not by relaxing anything.
- **OpenStax has a deliberate stricter rule**: accept CC BY 4.0 only when the
  statement appears in the book text, explicitly refusing the dataset-card
  label as per-book evidence. It works -- 69 of 76 books pass -- but the 7 that
  lack an attribution page yield zero records and therefore fail their whole
  task.

**Do for 1.7:** validate the licence policy against sampled rows at manifest
time, the same way sources are checked for text payload. One command per source
-- does a row satisfy the status this manifest asserts? -- would have caught all
nine before an eleven-day acquisition, let alone before a build. And note the
failure shape: a legitimately zero-yield file fails its entire array entry,
because "accepted zero records" is fail-closed. That guard is right and found
this, but a source with a small number of genuinely unlicensable documents
cannot currently express that.

---

## 5. Swallowed tracebacks were the most expensive bug of all

`cli.main` caught every exception and printed one line with no stack. Compare:

- **Without a traceback:** `FAIL JSONDecodeError: Unterminated string ... char 3414`
  cost most of a day across three restarts, two wrong hypotheses (truncated
  download, flaky proxy), and a cache scan that came back clean.
- **With a traceback:** the very next failure named `heegyu/bbq`, the file, and
  the line. Fixed in twenty minutes.

Fixed in `1fa8415`. `json.loads` failures now also name the file, line, byte
offset, and surrounding text.

**Do for 1.7:** no production tool hides a stack trace on failure. Any parse or
decode error names its source. This is worth more than any individual bug fix in
this document.

---

## 6. Operational gotchas specific to this stack

- **`ops/start-acquisition.sh` prints `FAIL: the Screen session exited during
  startup` on success.** It misled us at least four times. The real signal is
  the last log line and `"status": "complete"`.
- `screen -DmS` does not fork, so the launcher blocks for the run's lifetime.
  Silence is success.
- `journal_mode=TRUNCATE` on Lustre because WAL needs `mmap` for its `-shm`
  file. `database is locked` during writes is normal, not a fault.
- `metisctl status` reads the state DB and **blocks while acquisition writes**.
  Use filesystem checks for liveness instead.
- Every `git pull` invalidates the source lock (`repository_commit` binding) and
  forces a re-resolve. Batch code changes; do not pull mid-run.
- **That invalidation cascades.** A re-resolve writes a new lock file with a new
  digest, and `ACQUISITION_READY.json` records the *old* digest, so
  `submit build` dies with `The immutable source lock changed after
  acquisition` even though every acquired byte is untouched. Any code change
  between acquisition and submission triggers it. The fix is `metisctl
  rehandoff`, which re-runs the full attestation and refuses if anything
  describing the data moved; the trap is that without such a command the
  obvious workaround is deleting a self-hashed artifact by hand.
- **`resolve_sources` never rebuilds an existing lock — it only validates one**,
  and raises on a commit mismatch. Re-resolving therefore *requires* archiving
  the current lock out of the way first; calling the resolver and expecting a
  fresh lock silently gets you a validation failure instead. This is safe to
  automate only because `_validate_outputs` demands a completion marker whose
  `task_sha256` matches every task in whatever lock comes back, so a re-resolve
  that moved a task identity fails rather than rebinding to different work.
- Ctrl-C inside a C-level SQLite call is deferred and echoes `^C` only. Use
  Ctrl-Z then `kill -9 %N`.
- Deleting ~2 TB on Lustre takes many minutes and looks like a hang.
- **`sbatch` does not run the script you submitted.** It copies the contents to
  `/var/spool/slurmd/job<id>/slurm_script` and runs that, so `BASH_SOURCE`
  resolves under the spool directory. `stage.sbatch` derived the checkout that
  way and every array entry died in three seconds looking for
  `/var/spool/.metis-runtime/bin/python`. The submitter now exports
  `METIS_ROOT` and the script refuses to start without it. The two training
  scripts had the identical line.
  **This is a testing lesson more than a Slurm one:** two existing tests already
  executed the wrapper and asserted on its argv, but they ran it from its real
  path in the checkout -- the one condition that never holds in production. A
  wrapper test has to reproduce the staging, not just the environment.
- **The runtime is not inside the checkout.** `metisctl` falls back to
  `~/.cache/metis/runtime-<host>` when `$ROOT/.metis-runtime` is absent, and on
  this site that fallback is the real interpreter -- the one that downloaded all
  1.4 TB. `stage.sbatch` only knew the in-checkout default, so fixing
  `METIS_ROOT` alone would still have failed. The submitter now exports
  `METIS_PYTHON=sys.executable`, so the stages run under the exact interpreter
  the operator did. Two components deriving "the" runtime from independent
  defaults is a bug waiting for the day the defaults disagree.
- **Submit every stage `--requeue`.** Five of 58 `handoff_signature` tasks were
  killed at one second with `ExitCode 0:0` and no log file written at all --
  the scheduler ended them before they started. `afterok` then held all 51
  downstream jobs behind an unsatisfiable dependency, for five tasks worth ten
  seconds of work. Every stage is already idempotent through completion
  markers, so requeueing costs at most one repeated task and removes the entire
  class of "one evicted task stops the graph". A long single-node reducer is
  where this would really hurt.
- **Never size a walltime from the first samples to finish.** `normalize` was
  cut from 24h to 4h on the strength of the only two array entries that had
  completed, at 18:31 and 18:40. Five entries then hit the limit. Entry input
  bytes span **80.9x** -- 103.2 GB against 1.3 GB -- so the first finishers are
  by construction the smallest, and a uniform limit has to cover the largest.
  A TIMEOUT is not requeued, so each one stranded the whole downstream graph.
  The real fix is to pack array entries by input bytes rather than file count,
  so the limit tracks work instead of the worst case.
- **Group failures by node before blaming the code.** All eight of those
  cancellations, across two submissions a day apart, landed on `parrypeak073`
  or `parrypeak079`; nothing that ran elsewhere was cancelled. Neither node
  appears in `sinfo -R`, so Slurm keeps scheduling onto them and requeueing can
  land a task straight back on the same node. `sacct --format=JobID,State,
  NodeList` takes one command and would have pointed at the cluster
  immediately. Excluded via `scheduler.extra_sbatch_options`; re-check before
  the next release rather than carrying the exclusion forever.
- HF **API** runs ~45 KB/s here (EDR TLS inspection, latency-bound); bulk **CDN**
  transfer exceeds 100 MB/s. That asymmetry is why pointer corpora are
  catastrophic and content corpora are trivial.
- `expand=true` on the HF tree API caps a page at 50 entries instead of 1000 —
  a 20× round-trip penalty for fields that are returned either way.

---

## 7. Checklist for 1.7 acquisition, in order

1. Declare `acquisition_shape` per source; budget build-shaped sources by row
   count.
2. Decide the machine per shape. Crawl work needs egress + local NVMe. Write
   down where each source is acquired before writing the manifest.
3. Build the cross-machine import/receipt path first.
4. Preflight every benchmark repo for loading scripts and media columns.
5. Verify every source ships text, by reading columns of a sampled file.
6. Start build-shaped and crawl-shaped sources first.
7. Confirm no throughput knob reaches a run fingerprint.
8. Confirm stage scratch resolves to node-local storage on the machine that will
   actually run it — and that the fallback fails loudly rather than silently
   landing on a network filesystem.

---

## 8. What 1.6 actually shipped

- 72 download tasks, 50 sources, ~1.4 TB raw, 1T tokens planned
- Freshness: **35B, general web only, ending 2025-06** (planned: 90B across four
  buckets ending 2026-06)
- Code: StarCoderData 85.8B, github-code-clean 38B, Nemotron-CC-Code 31B, after
  three index-only sources were replaced
- Decontamination: 63 benchmarks, 203 jobs, 4.19M fragments, 1.97 GB
- 1.9 TB of dead Common Crawl ledger reclaimed

Scores from this corpus are **not comparable to published numbers**: it
decontaminates against benchmark *training* splits as well as test, which most
labs do not. That is deliberate. See `metis16_pretraining_data_plan.md`.
