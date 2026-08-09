# Metis-1.7 data pipeline: lessons from the 1.6 build

Written 2026-08-04, during the Metis-1.6 `metis-1.6-data-r1` build on Portage.
Everything here was measured against real data on Lustre, not reasoned about.

This file lived outside the repository for most of the 1.6 build, because
`require_clean_repository` treated an untracked file as dirty and committing it
moved HEAD, which un-pinned the source lock (§0). Writing down a lesson could
cost a rebuild, which is itself the lesson in §0. That is fixed: stage identity
is now a fingerprint of the modules a stage runs, so a file under `docs/`
changes nothing, and the document lives with the code it is about.

---

## 0. The worst defect in the pipeline: cleanup makes the build unrestartable

**This cost a 1.37 TB re-download and a full rebuild from normalize. It is the
most expensive single thing in this document, and it is not a performance bug.**

Three rules in the 1.6 pipeline are individually reasonable and jointly fatal.

1. `cleanup_raw` is a **stage inside the build graph**. After normalize is
   verified, it deletes `raw/`, `cache/huggingface`, `cache/common-crawl` and
   `cache/tmp/materializers` — the entire acquisition output.
2. `verify_acquisition_handoff` runs on **every `submit` and every `resume`**,
   and calls `_artifact_record` on every file the download tasks recorded. A
   missing file is a hard `RuntimeError: Acquisition output is missing`.
3. The source lock is bound to a repository commit, and
   `_validate_existing_lock` refuses to reuse it from any other commit:
   *"resume with the original commit or create a new data release."*

Read them together. The moment `cleanup_raw` completes, the build can only run
**forward**. It cannot be resubmitted, because rule 2 demands files rule 1 just
deleted. It cannot be fixed and resumed, because rule 3 pins it to the commit
whose inputs are gone. And it cannot be restarted at a new commit, because a new
commit means a new lock, a new execution contract, and therefore a re-run of
normalize — which needs the raw data that no longer exists.

A single interruption after that point — a cancelled job, a node failure, an
operator deciding to deploy a one-line speedup — costs the entire acquisition.

**How this actually played out.** Span dedup was running slowly. We cancelled
the array to deploy a file-pool fix (§1d). `cleanup_raw` had completed ~16 hours
earlier, silently, as a normal graph stage. Cancelling was therefore already
irreversible; we simply did not know it yet. Every recovery route was checked
and every one was closed:

| Route | Why it fails |
|---|---|
| `submit build` again | handoff verifier: 1,862 raw files missing |
| `resume build` | same verifier, same call site |
| restore the old lock + completions | works, but then the checkout must be the old commit, so the fix cannot ship |
| flip `require_acquisition_handoff` off | `gates` is *inside* the execution contract (§6a), so this invalidates normalize, which then needs raw |
| patch the verifier to honour cleanup receipts | a code change is a new commit is a new lock is a new contract — same wall |

The state *was* fully recoverable in the sense that matters for correctness: the
archived `sources.lock` / `build.inputs` / `ACQUISITION_READY` triple
reproduced the original `execution_contract_sha256` for `normalize`,
`exact_filter` and `span_prefilter_signature` exactly, and 923 GB of
exact-deduped corpus plus 1,202 span tasks were still on disk. It was recovery
into a **frozen** build: correct, resumable, and permanently unmodifiable.

There is a tempting fake fix here. `submit` calls the verifier with
`verify_artifact_hashes=False`, so it checks only `is_file()` and `st_size` —
1.37 TB of correctly-sized empty files would satisfy it. **Do not.** That
converts an integrity check into a lie, and nothing downstream would catch it.
If the only way to make a gate pass is to fabricate what it inspects, the gate
is telling you the truth about the state you are in.

### What 1.7 must do

- **Never let a graph stage delete a precondition that a later `submit`
  re-validates.** If cleanup must exist, the handoff verifier has to consult the
  cleanup receipt and accept documented, hash-recorded retirement. The receipt
  already contains everything needed — `verified-content/<stage>.jsonl` carries
  per-file size and SHA-256 for the whole tree.
- **Default to retaining.** The 1.6 fix was a `retain_stage_inputs` gate that
  makes verified cleanup skip its deletions. Portage reported **4.7 PB free**
  against a **1.37 TB** input set. The stage was solving a problem we did not
  have, at the cost of the ability to recover. At 30 T tokens the inputs get
  bigger, but so does the cost of re-fetching them — the ratio does not improve.
- **Separate "what the data is" from "what the code is" in the contract.**
  Binding stage completions to the repository commit means a typo fix in a
  comment invalidates a week of dedup. The contract should hash the manifest,
  the lock's *source content*, and the policy knobs that change output — not the
  commit SHA, and not the whole scheduler block (§6a). Code identity belongs in
  provenance, recorded and reported, not in the resume key.
- **Make the trap loud.** `cleanup_raw` should refuse to run, or at minimum log
  a one-line warning, when it is about to delete files that the active profile's
  `require_acquisition_handoff` gate will demand back.

### The general rule

An irreversible step and a revalidated precondition must never be the same
files. Before writing any stage that deletes, ask the question that would have
caught this in review: **"after this runs, what does `resume` check?"**

---

## 1. The single biggest performance defect: YAML parsed once per record

**Measured: 82% of normalize CPU time, 5.6x speedup available.**

```
as the build runs today (YAML per record):  0.27 MB/s
with quality profiles cached             :  1.52 MB/s
```

`evaluate_quality` opens with:

```python
profiles = profiles or load_quality_profiles()
```

`stage_runner.py:1348` calls it **without** `profiles=`, so every record in the
build re-reads and re-parses `configs/metis16/quality-profiles.yaml` off Lustre
to make one accept/reject decision against a ~100-line file that never changes.
A cProfile run over 300 records showed `yaml.compose_node` called 103,500 times
— the top eleven entries by `tottime` were all `yaml/`.

`profile_preflight.py` already does it correctly (`profiles=profiles`), which is
why the preflight sweeps 3,000 records in a minute while the build spent ~2
hours per input file.

**Fix:** `functools.lru_cache` on `load_quality_profiles`, or thread the cached
dict through from the caller. One line either way.

**Scope:** normalize only. `evaluate_quality` has exactly one call site in the
whole build. Dedup stages hash and compare; they never evaluate quality. Do not
expect this to speed up dedup.

**Confirmed in production, not just in the profiler.** After the fix the stage
ran at **77.4 markers/min against the previous run's 12.9/min peak — 6.0x**,
matching the predicted 5.6x. A 9-hour stage became roughly 30 minutes of work
plus a 4-hour single-file tail (see 10d). The cache itself: 2,000 calls in
0.089s, ~44us each, against ~10ms uncached.

**Related but minor:** `_manifest()` is uncached and `_stage_execution_contract`
calls it once per task — 0.109 s per call, about 14.6 CPU-minutes across the
whole graph. Negligible; note it only so nobody re-discovers it as a suspect.

**Lesson for 1.7:** profile the per-record path before assuming the bottleneck is
I/O or node count. The wall-clock story was never "two hours of work per file";
it was ~20 minutes of work wrapped in 100 minutes of re-parsing the same file.

---

## 1b. The next 2x is per-character Python loops — TOP PRIORITY AT 30T

With the YAML parse gone, the profile is dominated by walking each document
character by character in the interpreter. Measured on `pes2o`, 200 rows /
0.27 MB, post-cache: **1.32 MB/s, 1.0 ms per 1 KB document — about one
microsecond per character.**

| cost | calls for 200 rows |
|---|---|
| `sum(c.isalpha() for c in text)` — quality.py:150 | 269,535 |
| `sum(not c.isalnum() and not c.isspace() ...)` — quality.py:151 | 269,535 |
| `[c for c in text if c.isalpha()]` — evidence.py:400 | per char |
| `sum("LATIN" in unicodedata.name(c,"") for c in letters)` | **218,223 `unicodedata.name` calls** |
| `str.isalpha` total | 538,670 |
| `typing.__subclasscheck__` (isinstance overhead) | 36,400 |

Every document is traversed **four separate times**: alpha fraction, symbol
fraction, building a list of letters, then a Unicode-database lookup per letter
to decide whether it is Latin. `unicodedata.name()` per character is the worst
of them and is entirely avoidable.

**Fixes, cheapest first:**
- the Latin test needs no `unicodedata.name()`. An `str.isascii()` fast path
  settles the overwhelming majority; reserve the lookup for the remainder.
- `text_features` walks the string twice for alpha and symbol fractions. One
  pass, or `re.findall` / `str.translate` counting, gets both.
- `sum(map(str.isalpha, text))` is roughly 2x over the genexpr for free; a
  translate table or regex count is far more.
- `_computed_english_probability` truncates `words` to 250k characters but
  **not** `letters`, so a large document pays the full per-character cost on the
  single most expensive loop. Truncate both.

**Why this outranks everything else here at 30T.** Metis-1.6 normalized 1.37 TB
for a 1 T-token target; a 30 T-token corpus is roughly thirty times that input.
At today's rate normalize alone scales into days of pure CPU, and combined with
the file-size tail in 10d it becomes the dominant cost of the entire build. The
YAML fix gave a measured 6x. This is plausibly another 2x or better, putting
normalize **10x+** off where 1.6 began — the difference between a stage you plan
the week around and a stage you stop thinking about.

Do it before the first 30 T build, not during: once a build is in flight every
fix costs a full re-run (see 10e).

---

## 1c. Span dedup was NOT the sentence splitter -- it was a thrashing file pool

**This section originally recommended rewriting `sentence_spans` with a regex.
That recommendation was wrong, was tested, and is retracted. Read 1d first: the
real cause was a file-handle pool sized below its bucket count.**

The profile below is accurate as far as it goes, and it is a good example of how
a correct profile can still point at the wrong fix.

`span_prefilter_signature` ran at **7.53 MB/s** and took **11.5 hours to reach
1,190 of 1,862 tasks**, with node CPULoad between 18 and 62 out of 192 -- 10-32%
utilisation, the same "not saturated because it is stuck in the interpreter"
signature as the YAML bug.

Profiled on 300 real post-exact-dedup documents (9.13 MB):

| | time | share |
|---|---|---|
| `sentence_spans` (character-by-character Python loop) | **1.541s** | **68%** |
| `re.findall` in `_canonical_sentence` | 0.198s | 9% |
| `span_digest` | 0.066s | 3% |
| **SHA-256 itself** | **0.026s** | **1%** |

The cryptographic hashing everyone assumes is expensive is **1%**. The stage is
its sentence splitter, which walks every character of every document in Python:

```python
while cursor < length:
    character = text[cursor]
    if character == "\n": boundary = cursor
    elif character in ".!?": ...
```

The rule it implements -- a newline is a boundary, and a run of `.!?` followed by
whitespace or end-of-text is a boundary -- is one regex:
`re.finditer(r'[.!?]+(?=\s|$)|\n', text)`. That runs in C and should be 10-50x
faster, taking the stage to roughly 3x overall.

**This compounds across the span chain.** `sentence_spans` is called from
`iter_span_signatures` (used by `span_prefilter_signature` and
`span_signature`), and from `strip_duplicate_spans` and `build_span_dedup_filter`
(used by `span_filter`) -- three of the five span stages.

**Tested, and it does not work.** The regex rewrite is exactly equivalent --
5,823 documents, 2,800 of them real corpus rows, plus adversarial cases like
`"..!?.."`, `".\n."` and `"U.S.A."`, with zero mismatches -- and it is **not
faster**: 8.7 MB/s before, 8.3 MB/s after, i.e. 0.9x.

The profiler was right that `sentence_spans` holds 68% of the stage's `tottime`,
and the wrong conclusion was mine. That time is the **per-sentence** work inside
the loop -- `_canonical_sentence` doing NFKC normalisation and casefolding, the
`WORD_RE.findall` word count, the slicing, the `SentenceSpan` construction --
not the character scan that finds the boundaries. Replacing the scan removed the
part that was not costing anything.

**The lesson is about profiling, not about splitting.** `tottime` on a function
containing a loop attributes the loop body's inline work to the function, so a
hot function is not the same as a hot line. Before rewriting, measure the
candidate replacement in isolation -- here, timing `sentence_spans` with the
body stubbed out would have shown the scan was cheap in about a minute.

If this is revisited for 1.7, the target is the per-sentence work: memoise
`_canonical_sentence` (documents repeat sentences), or compute `words` from the
already-normalised string without a second regex pass. Do not touch the scan.

## 1d. What span dedup actually was: a file pool smaller than its bucket count

Three passes bucket signatures by `digest % finder_workers` and write each
bucket through an LRU pool of open file handles. All three sized that pool at
**32** while `finder_tasks` is **64**:

```
span_dedup._BoundedFilePool        64 buckets, 32 handles (config)
code_dedup.write_code_signatures   64 buckets, 32 handles (hardcoded)
final_dedup._FilePool              64 buckets, 32 handles (default argument)
```

The bucket for a signature is a hash, so the target is effectively random per
record. With half the buckets resident, roughly **every second write evicted a
handle and reopened another file**, and on Lustre an open and a close are
metadata round trips. This corpus emits about 6.4 billion span signatures.

Measured on the live run: `span_prefilter_signature` reached 1,190 of 1,862
tasks in **11 hours 31 minutes** at 18-62 CPULoad out of 192. The same work
profiled single-threaded runs at 7.45 MB/s, which across the ~640 resident
workers puts the compute cost of the whole 923GB corpus at **minutes**. Two
orders of magnitude apart, and the gap was entirely the pool.

**Confirmed by direct measurement on the same Lustre filesystem.** 64 buckets,
60,000 records, hash-random bucket order, one pool size against the other:

| pool | wall | throughput | opens |
|---|---|---|---|
| 32 handles | 35.17 s | 1,706 rec/s | 29,870 (**49.8%** of writes) |
| 96 handles | 0.13 s | 479,779 rec/s | 64 (one per bucket) |

**281x on the writer path.** The 49.8% miss rate is exactly the K/N = 32/64
predicted for an LRU under uniform random access, and the difference works out
to **1.18 ms per open-and-close** — one Lustre metadata round trip, paid on
every second record, 6.4 billion times.

Note what this measurement does and does not say. It isolates the writer. The
stage also reads inputs and computes signatures, which is CPU work the pool size
cannot touch, so the end-to-end stage speedup is bounded by the writer's share
of stage time and will be smaller than 281x. The mechanism and its magnitude are
settled; the stage-level multiplier still has to be measured in production.

**How to spot this class:** aggregate throughput far below single-threaded
throughput times worker count, with CPU unsaturated. That combination means the
workers are blocked on something, and on a shared filesystem the first suspect
is metadata operations, not bandwidth.

**For 1.7:** assert `maximum_open_files >= finder_tasks` at profile-validation
time. It is a one-line invariant and it silently cost more than every other
defect in this document combined.

---

## 2. Do not confuse node count with parallelism

I initially reported "we're using 3% of the cluster, 3–6x available." **That was
wrong** and acting on it would have destroyed ~1,300 completed tasks.

We occupied 17–32 nodes of 124, but each node ran ~48 worker processes. Tasks in
flight were `32 x 48 = 1,536` against a total of `1,862` tasks. Tasks are input
files; there are no more to hand out. **The real ceiling was 1.21x, not 3-6x.**

Nothing was saturated because nothing needed to be: CPULoad ~36 of 192, 497 GB
free of 512 GB RAM, ~74 MB/s writes.

The `*_find` stages look worse (6 elements for `span_find`, 4 for `code_find`)
but are the same story — their task counts come from bucket counts and
`tasks_per_job` already puts every task in flight.

**Lesson for 1.7:** compute `total_tasks` vs `tasks_in_flight` before proposing a
parallelism change. If they are close, the only lever left is making each task
faster (see §1).

**What was genuinely recoverable:** arrays whose element count exceeded their
`%N` throttle. Raised live with `scontrol update JobId=<id> ArrayTaskThrottle=<n>`
— no profile edit, no commit, no contract change:

| stage | was | now |
|---|---|---|
| `decontam_filter` | 32 | 47 |
| `verify_shard`, `pack` | 32 | 42 |
| `token_count`, `tokenizer_sample` | 32 | 39 |
| span/minhash/code signature+filter | 32 | 34 |

Caveat: `ArrayTaskThrottle` is not persistent. A cancel-and-resubmit reverts to
the profile's `max_concurrent`. For 1.7, fix `max_concurrent` in the profile
instead — but only at submission time, never mid-build (§6).

---

## 2a. Acquisition is link-limited, so its parallelism belongs across hosts

Acquisition ran at 126 MB/s with `max_workers: 4`. Raising it to 24 — 157 live
threads — produced **124 MB/s**. No throttling from the Hub, load average 1.97
on 384 cores. The extra concurrency bought exactly nothing, and the reason was
visible in one number the whole time: 126 MB/s is 1008 Mbps.

```
bond0        25000 Mbps  (up)   internal fabric
hsn0/hsn1   200000 Mbps  (up)   Slingshot interconnect
ens2f3        1000 Mbps  (up)   default route  <- all external traffic
```

A login node with 384 cores, a 25 Gbps bond and a 200 Gbps Slingshot fabric
reaches the internet through a 1 Gbps management NIC. The fast interfaces are
for Lustre and MPI. **Check `ip route` against interface speeds before tuning
any download concurrency**, because saturation looks identical to a tuning
problem, and every knob inside the host is the wrong knob.

The right axis is more hosts. login1 is an identical node with its own 1 Gbps
uplink, and it needed no code: `metisctl download-task --profile P --task-index N`
is a per-task entry point that skips completed tasks and does not take the
supervisor's singleton lock. Running it on login1 descending from the last task
while login2's supervisor ascends from the first took aggregate throughput to
**231 MB/s**, and the two never collide because `StateStore.task_lock` is a
mkdir mutex on Lustre — atomic, and deliberately unwilling to reclaim a lock
whose `OWNER.json` names a different host.

**The caveat that matters operationally:** that same refusal means a helper
process dying on login1 leaves a lock its own host can clear and login2 cannot.
`metisctl unlock-stale` exists for exactly this, and a multi-host acquisition
should expect to need it.

**For 1.7 at 30 T:** 1 Gbps is 10.8 TB/day per host. A 30 T-token corpus is
tens of terabytes of candidates, so acquisition is measured in host-days and the
only lever is how many hosts pull at once. Design the supervisor for it —
`--shard i/n` as a first-class flag rather than a hand-driven loop — and confirm
whether a data-transfer node with a real uplink exists before assuming the login
nodes are the ceiling.

---

## 3. Profiles demanded evidence publishers never ship

This was the dominant *correctness* failure class, worth ~45B usable tokens.
Each case is the same shape: a gate asserts something about the data that nobody
checked against the data, and fail-closed turns "we could not measure this" into
"this is bad."

| source | symptom | actual cause |
|---|---|---|
| `nemotron_specialized_fact_seeking` | 44/60 `missing_language_probability` | detector returned `None` below 100 letters / 30 words; rows run 18–55 words. Short is *uncertain*, not *unmeasurable* |
| `openstax` | reported `0/1` | preflight sampled only the **first input file** per source; openstax is 76 books in 76 files, so "one file" was one record |
| `finepdfs_edu_english` | 38.3% vs a 76.7% ceiling | `reading_order_passed` internally required `repeated_page_edges <= 0.08` **and** the profile gated the same fraction again. Relaxing either alone changed nothing, and the redundancy hid the cause |
| `open_law_usgpo` | 38/60 `personal_data` | agency office numbers in Federal Register notices — (817) 222-5110 is FAA Fort Worth |
| `nemotron_math_proofs` | 7/60 language rejections | ships `lean.jsonl` with `formal_statement`/`lean_header` and **no** `ext` field, so the formal-language test never fired and Lean was scored as English prose |
| `megamath_unique` | 54/60 `math_score_minimum` | **units**: MegaMath-web states a 0–1 probability, FineMath a 0–5 integer, gate threshold is 3. A row rated 1.00 scored 1.0 against 3 |

**Lesson for 1.7:** before writing a gate, dump the actual columns of the pinned
config and confirm the field exists, is populated, and is on the scale the
threshold assumes. `preflight-profiles` catches this in a minute; it was written
for exactly this and is the highest-value tool in the repo.

**Corollary — categories that repeat lines by construction:** code, worked
mathematics, books, and LaTeX papers all repeat lines structurally. The repo had
already raised `maximum_repeated_line_fraction` for the first three;
`scientific_paper_v1` still sat at the 0.20 default and passed only 51.1% of
arXiv papers (0.45 passes 97.2%). Check the whole family when fixing one.

---

## 4. `allow_patterns` on multi-config Hugging Face repos

**Seven of the pinned sources used unrestricted patterns.** Every one either was
broken or needed auditing. This was the largest single class of defect.

- **`cosmopedia_v2`** — the id 307-redirects to `HuggingFaceTB/smollm-corpus`,
  which holds three configs. `**/*.parquet` matched all 673 GB and the resolver
  filled its byte target from `fineweb-edu-dedup`: the raw CommonCrawl corpus
  Cosmopedia was *generated from*, not Cosmopedia. Zero yield, and a duplicate of
  the separately pinned `fineweb_edu`.
- **`proof_pile2_math` / `proof_pile2_science`** — same repo, same revision, same
  pattern. Both resolved against the identical 482-file set; all 48 of science's
  files were also among math's 73.
- **`finemath_unique`** — `finemath-4plus` is a nested subset of `finemath-3plus`
  (confirmed against the data: every `int_score>=4` row carries raw score >= 3.5,
  exactly 4plus's published floor). The pattern counted those rows twice.
- **`megamath_unique`** — redirects to `IFM/MegaMath`; `megamath-web-pro` is a
  model-rewritten refinement of `megamath-web`, so taking both double-counts and
  contradicts `provenance.generated=false`.

**Lesson for 1.7:** pin the config, never the repo. Check for redirects — two of
seven repo ids silently redirected elsewhere.

### 4a. The glob trap that produces a silent zero-token source

`manifest.matches_any` is `fnmatch`-based, and `fnmatch` treats `/` literally.
The `**/` fallback only strips a **leading** `**/`.

```
finemath-3plus/**            -> 128 files   CORRECT
finemath-3plus/**/*.parquet  ->   0 files   MATCHES NOTHING
```

Whether the second form works depends on whether files sit directly under the
config dir or one level deeper — `megamath-web/**/*.parquet` *does* match,
because that config nests. **Always validate a pattern against the real tree
using the repo's own matcher before committing it.** A pattern that matches
nothing produces a zero-token source, not an error.

---

## 5. Silent-failure mechanics worth knowing

**`_row_metadata` pre-seeds metadata and `_set_evidence` will not overwrite.**
Any adjustment in the profile block was discarded for exactly the rows that ship
the field the adjustment is for. This had silently disabled the
`nemotron_cc_math_4plus` and `_unique_3` partition floors long before I touched
anything. Fixed by marking derived scores and passing `overwrite=`.

**A self-confirming constant.** `compressed_bytes_per_token: 0.75` for Cosmopedia
was wrong by 5x (measured 3.86 from parquet column-chunk metadata). It cannot
self-detect: `_select_files` stops once selected bytes reach
`candidate_tokens * ratio`, and the shortfall check divides those same bytes by
the same ratio. It would have taken 8 of 104 shards, landed ~2.4B tokens against
an 8B budget, and reported `candidate_target_met`. **Measure bytes-per-token from
the actual files, per source.**

**Attestation vs. rubber stamp.** Cosmopedia ships no generator column at all, so
`require_genealogy` rejected 705 of 720 rows. The fix is a manifest attestation
naming the pinned generator with a written basis — legitimate, because the
generator is a property of the release. But note `seed_data` is the seed *corpus
name* (9 distinct values across 720 rows), so hashing it into
`source_document_id` satisfies the grounding gate with evidence that grounds
nothing. The repo's own comment warns about this: "one value shared by every row
identifies nothing." **For 1.7, prefer hashing the actual seed text (in `prompt`)
over a corpus label.**

**Directory names can lie.** In `nvidia/Nemotron-Pretraining-Legal-v1`, the
payloads of the two largest subsets are swapped: `Case-Law-Summary/` contains the
CaseHOLD *task* ("select the correct holding statement"), and `CaseHOLD/`
contains ordinary case-law narrative. Read payloads, never names.

---

## 6. Immutability ordering — the rule that cost three failed submissions

Three consecutive `submit build` attempts failed, each on a different binding:

1. `The data manifest changed after acquisition` — a stale `ACQUISITION_READY.json`
   short-circuits `write_acquisition_handoff`, which validates the *old* handoff
   against the new manifest instead of writing a new one.
2. `The repository commit changed after the immutable source lock was created` —
   the lock binds `HEAD`.
3. `The immutable source lock changed after acquisition` — the handoff binds
   `sha256(sources.lock.json)`.

**The rule: make every code and manifest change first, then `resolve` LAST, then
rebind the handoff, then submit. Any commit after resolve invalidates the lock.**

**`rehandoff` semantics.** It seals `artifact_count` / `artifact_bytes` and
refuses when the acquired data itself moved. It is:
- the **wrong** tool after a re-pin that downloaded new files (it correctly refuses)
- the **right** tool when only the commit/lock moved and no bytes changed

When bytes genuinely changed, archive `ACQUISITION_READY.json`, `HANDOFF_VERIFIED.json`
and the `handoff_signature`/`handoff_verify` completion markers (they embed
`handoff_sha256` and fail two commands later otherwise), then re-run acquisition;
it re-verifies and attests without re-downloading.

### 6a. The scheduler block is inside the execution contract

`_stage_execution_contract` hashes:

```python
"state_artifacts": {sha256 of sources.lock.json, build.inputs.json, ACQUISITION_READY.json},
"scheduler": profile.get("scheduler", {}),
"gates": profile.get("gates", {}),
```

Consequences:
- **Changing any `tasks_per_job` or `max_concurrent` invalidates every completed
  task in every stage.** normalize deletes its output and redoes the task
  (line 1267); filtering stages raise "completion belongs to stale inputs or
  policy" (line 387).
- **Re-resolving mid-build does the same**, because the lock's sha256 is in the
  contract.

**Tune concurrency at submission time only.** Mid-build, `ArrayTaskThrottle` is
the only safe lever. Source-code changes are safe (not hashed) but pulling under
running jobs risks partially-written files — there are lazy imports in the hot
path (`from .handoff_verification import ...` inside the task function).

**For 1.7:** consider hashing only the scheduler keys that affect *results*
(none of them do) rather than the whole block. Concurrency is an operational
knob and should not be provenance.

---

## 7. Slurm operational notes

**Held tasks are invisible to failure monitoring.** Tasks failed to launch with
`user_env_retrieval_failed_requeued_held` (Slurm runs a login shell to retrieve
the environment under `--export=ALL`; it timed out), were requeued, and **held**.
A held task is neither `RUNNING` nor `FAILED`. Under an `afterok` graph it stalls
every downstream stage in silence — a monitor that greps for failures shows
nothing while the build sits dead. One task hit `Restarts=4`.

**Any watcher must check for held/launch-failed states and release them**
(`scontrol release <jobid>`), and must alarm on "nothing running while jobs are
queued." Match `PENDING` only — a `COMPLETING` job keeps the reason string that
held it, and acting on that fires forever.

**Environment:** Slurm reports local time while `date -u` reports UTC; a
`StartTime` that looks five hours stale is probably a timezone confusion.

---

## 8. Contamination: reformulations defeat n-gram decontamination

`nemotron_legal_v1` pulled five subsets that are Qwen3 **reformulations** of tasks
from `legalbench` and `lex_glue` — both pinned in `eval-holdouts.yaml`.

The decontamination policy matches on 13-grams (`explicit_genealogy_match`,
`remove_entire_document`). **A reformulation need share no 13-gram with the
benchmark it came from.** Decontamination is not a backstop for
model-rewritten benchmark data; exclusion at the pin is.

**For 1.7:** treat "synthetic data derived from a task in our eval set" as a
distinct risk from "text that overlaps our eval set," and audit synthetic corpora
against the holdout registry by *provenance*, not by n-gram.

---

## 9. Gate design principles that earned their keep

- **Measure the thing the gate is aimed at.** Repeated page edges do not measure
  reading order; they measure pagination. Currency `$` is not a math delimiter.
  A role mailbox (`support@openstax.org`) is not a person's contact details. An
  RFC 2606 placeholder (`firstname.lastname@example.org`) can never route to
  anyone.
- **Count distinct, not occurrences.** A running footer repeating one address on
  every page is one contact, not forty. `personal_data_maximum_contacts: 4`
  admits a document with a contact block and still rejects a staff directory —
  measured shape: 258 of 300 records carry no contact, 31 carry one or two, a
  thin tail runs higher.
- **Scope exemptions to the profile that needs them,** and test that they do not
  leak. `legal_primary_v1` may keep agency phone numbers; `web_general_v1` must
  not.
- **Never rewrite the training text to satisfy a gate.** The currency fix changes
  only the balance *count*; a test pins that `extract_training_text` is unchanged.
- **Regex needs testing, not reading.** A negative lookahead only suppresses the
  match starting where it is anchored, so `no-reply@x.org` was skipped at `no`
  and matched again at `reply`. And `\$\d[\d,]*` backtracks: `$20` matches as
  `$2`, so the test inspects `0 \times` instead of ` \times` and real mathematics
  reads as a price. Both were found by tests, not by review.

---

## 10a. A producer and a consumer that disagreed about the same flag

`stage_runner` demands the frozen Common Crawl opt-out snapshot whenever a
source declares `provenance.common_crawl_derived` -- deliberately, and the code
says why: a packaged extraction like FreshWeb is Common Crawl text a third party
filtered at build time, which makes it *more* exposed to a later publisher
withdrawal, not less. But `handoff._uses_common_crawl` only wrote the snapshot
when some source had `driver == "common_crawl_ranges"`.

Those two conditions agreed for as long as any source used that driver. When the
`common_crawl_ranges` sources were withdrawn, no source had the driver, so no
snapshot was ever written -- while `metis_freshweb_2025` still declared
`common_crawl_derived` and failed all 34 of its normalization tasks on the
missing block. **Every archived handoff on disk lacks it**, twelve of them.

The part that should worry you most: `verify-handoff` is *supposed* to catch
exactly this (`_verify_final_common_crawl_policy` raises when the block is
missing and Common Crawl is used) -- but it is gated on the same wrong
`_uses_common_crawl`. **A guard and the thing it guards shared a broken
predicate, so the guard passed twelve times on handoffs that were missing the
data.** 2,983 publisher opt-out domains and 118 URL rules were silently not
being honoured.

**For 1.7:** when a check and the code it checks derive from the same predicate,
they cannot catch each other. Assert the *outcome* independently -- "if any
source is common_crawl_derived, the handoff has the block" -- rather than
re-deriving the condition.

## 10b. Fail-closed at task level is the wrong granularity

`normalize` raised when a task accepted zero records. The intent was to catch a
profile demanding evidence no publisher ships. In practice it caught that
**zero** times and stopped the build **three** times on properties of one file:

- a 13-byte zstd frame the publisher released empty
- an OpenStax book, because that source is one document per task, so one
  rejected book is a task that accepted nothing
- `lean_proofsteps.jsonl.zst`, where a proof state plus one tactic repeats lines
  by construction: median `repeated_line_fraction` 0.795, so 4% of rows clear
  `proof_v1`'s 0.50

Each time, a failed task failed its array element and `afterok` stopped all 49
downstream jobs -- in the last case over 12.7MB of a 7.58GB source.

"Every record rejected" is a **source**-level signal and is meaningless at task
level, where a task may hold one document. The zero-yield fact is now recorded
(`zero_yield` plus the rejection histogram, in every report) and the systematic
case is caught where it is visible: `preflight-profiles` before submission, and
`minimum_unique_tokens` at `select`.

**For 1.7:** keep the check, move it. After normalize, assert per *source* that
not every task yielded zero. That catches the real failure without letting one
file stop 49 jobs.

## 10c. An array element reported 34 failures and exited 0

Element 14's own summary said `'failed': 34, 'ran': 48`, and `sacct` recorded
`COMPLETED 0:0`.

**This was originally written up as an exit-code bug. That diagnosis was
probably wrong, and chasing it is instructive.** The batch runner does return
non-zero on failures, and `stage.sbatch` uses `exec`, so the interpreter's
status *is* the job's status — no pipe to swallow it, no trailing command to
overwrite it. Both predate the incident, so neither can explain it. The likely
story is `--requeue`: the element was requeued, the retry recomputed its pending
set from completion markers, ran only the 34 still-unfinished tasks, and passed.
`sacct` reports the final attempt. The log showing `failed: 34` was the first.

Two lessons, and the second is the one worth keeping.

**Reading a requeued array element's log is not reading its outcome.** Slurm
reports the last attempt; `%A_%a.out` may be from any of them. Compare
`sacct -j <id> --duplicates` before concluding anything about an array.

**The exposure was real even though the bug was not.** Everything deciding the
exit code was in-process accounting: a worker reports ok, the parent counts it,
the count becomes the status. Every link is somewhere a task can be lost without
anyone lying — a swallowed exception, a requeued attempt, a refactor that
returns early. `afterok` sees only the status, so a stage that drops tasks and
exits 0 advances the graph over a corpus with holes, and nothing downstream
looks again.

**Fixed by making the exit code an observation rather than a tally.** After a
task group runs, every index it owned is re-stat'ed; a task counts as done when
its completion marker exists on disk, not when it reports that it does. A worker
that returns `ok` and writes nothing now fails the element and names the tasks.

**For 1.7:** prefer deriving status from durable artifacts over in-process
counters anywhere a scheduler makes a branching decision on it. The counter and
the artifact agree right up until the moment it matters.

### 10c-postscript. The fix was worse than the bug, and why

The marker check above was written, shipped, and reverted within a day, because
on the first real submission it failed a stage that had done all of its work:

```
FAIL stage handoff_signature: 16 of 16 tasks left no completion marker
```

against a stage holding **3,280 completion markers**. Completion markers are not
named uniformly, and `StateStore.is_complete` is an exact filename match:

```
handoff_signature   task-00000000-117b1cffc8bb3828.json   8 digits + digest
download            download-000000-d5b1bb2fd544f305.json stage-prefixed
holdouts            task-000000.json                      plain 6 digits
```

The check looked up `task-<index:06d>` and could never find the first two forms.
The whole 37-stage graph then sat on `DependencyNeverSatisfied`.

The same wrong assumption was already present, one line above, in the `pending`
computation that decides which tasks to skip — so stages whose markers carry a
digest have always re-run instead of resuming. Wasteful, harmless, and years
old. Promoting it to an exit code made it fatal.

Three things worth keeping:

- **A safety check is production code.** This one was reasoned about carefully
  and tested against a synthetic state store that used the naming the check
  assumed. The test confirmed the check's own premise instead of the codebase's
  behaviour, which is the easiest test to write and the least useful.
- **Weight the evidence for the bug you are defending against.** This defended
  against 10c, which the section above had already downgraded to "probably a
  requeue artifact." Defending hard against a bug that likely does not exist,
  with a mechanism that can fail closed, is a bad trade.
- **Before asserting on a convention, verify the codebase actually guarantees
  it.** `ls` on one completion directory would have shown the digest suffix in
  seconds. Any future attempt needs one canonical completion key per stage —
  ask the state store what a task's marker is called rather than guess its
  shape.

## 10d. Task granularity is one file, and files are not the same size

Input files range from 13 bytes to 29.53GB. Task granularity is one file, so the
tail of every parallel stage is set by the largest single file, and no amount of
cluster is relevant to it. Measured: after the YAML fix the bulk of normalize
finished in about 30 minutes, while the tail ran for hours on a handful of
monsters -- one element spent **4h14m** on a single file.

The skew is inherited from upstream packaging, not created here:

```
median      356 MB        44 files (2.4%) hold 15.7% of all bytes
p90       2,146 MB        nvidia/Nemotron-Math-Proofs-v1 ships its entire
p99       3,378 MB          corpus as one 29.53GB data/lean.jsonl
max      29.53 GB        txt360 ships a single 22.85GB chunk
```

`nemotron_math_proofs` has no sharded alternative in the repo at all -- the
publisher offers exactly one file. So sharding has to happen on our side.

Nothing about the work is serial: a 22.85GB file is ~10 million independent
per-row decisions. It is the *task contract* that is serial, not the
computation.

This also breaks rate-based ETAs. Aggregate throughput says nothing about
completion when the finish time is set by one indivisible unit; I predicted "30
minutes" twice and was wrong twice for this reason.

**For 1.7:** shard large inputs at acquisition, or make task granularity
sub-file (byte ranges / row groups). Either removes the tail and makes ETAs
meaningful.

## 10f. The dedup stages do not record how much they removed

`exact_filter` completion markers carry `stage`, `task_index`, `completed_at`
and the execution-contract hash. No counts. `cleanup_exact` then deletes the
intermediate artifacts, so after a pass completes **there is no way to answer
"how many documents did exact dedup remove"** -- not from the receipts, not from
the leftovers.

Nor can it be inferred from directory sizes: `normalized/` is reclaimed by
cleanup, so there is no before-and-after to compare.

This is the single most important operational number in the whole build. The
`minimum_unique_tokens` gate fires at `select`, stage 33, and whether it passes
depends entirely on how much the four dedup passes remove. Today that is
unknowable until `token_count` at stage 27 -- by which point 26 stages and
several days have already been spent.

**For 1.7:** every filtering stage should write `records_in`, `records_out` and
`bytes_in`, `bytes_out` into its completion marker. `normalize` already writes
`counts` and `rejection_reasons`; the dedup filters should do the same. It costs
nothing and turns a late hard gate into an early prediction.

## 10e. Every manifest edit cascades through three identities

A single-line manifest change invalidates, in order:

1. `manifest_contract_sha256` -> every stage completion is stale by contract
2. the source lock -> the handoff no longer matches it
3. **download task IDs** -> acquisition markers no longer match, and `rehandoff`
   refuses with "Acquisition task is incomplete" even though every byte is
   present and size-matched

The third is the surprising one. The recovery is always: archive handoff +
handoff-bound markers -> resolve -> re-run acquisition (idempotent; it
re-verifies and re-attests without downloading) -> submit.

Regenerating the handoff **also** invalidates stage completions, because
`sha256(ACQUISITION_READY.json)` is inside the execution contract. So there is no
way to fix a manifest bug without redoing normalize. **Batch manifest changes;
never make them one at a time.**

This is what made 1.6 feel like an endless loop: fail-fast means bugs surface
strictly one at a time, and full invalidation means each one costs a complete
re-run. Four bugs became four restarts of a multi-hour stage. There is no
fix-and-resume.

**The fix for 1.7, and it is cheap now and enormous later:** put only what
determines a record's *content* into `_stage_execution_contract`. It currently
hashes the entire manifest and the entire `scheduler` block, so correcting a
licence expression -- metadata about how a record is treated, never what it
contains -- or raising a concurrency limit discards the whole corpus. Hash the
quality profiles, the gates that actually filter, and the input identities;
leave licence text, labels and scheduler tuning out. At 1 T that mistake costs
an afternoon. At 30 T it costs days per typo.

## 11. Numbers from this build, for calibration

| | start | end |
|---|---|---|
| zero-yield sources | 2 | **0** |
| usable tokens | 928.3 B | **973.7 B** |
| margin vs `minimum_unique_tokens: 950B` | **-21.7 B** | **+23.7 B** |

The build was **below its own gate** at the start and nobody knew, because the
gate fires at `select` — stage 33 of 37, days in. **For 1.7: compute projected
usable tokens (candidate x measured yield, capped at exposure) before submitting,
and treat it as a submission gate rather than a late one.**

Remaining gaps at submission, for reference: `finepdfs_edu_english` 10.3 B,
`nemotron_cc_v2_organic` 4.5 B (the only source that genuinely needs more data
rather than better gates), `megamath_unique` 3.7 B, `nemotron_math_proofs` 2.5 B.
