# AGENTS.md

Guidance for AI agents working in this repository.

Metis is a from-scratch frontier-model training stack: data acquisition and
curation, tokenizer, pretraining, post-training, and release. The goal is a
model that **pushes** the frontier. Every default in here should be read in that
light.

---

## 1. Research must be recent, and it must be frontier

This is the rule that matters most, and it is not optional.

**Do research often.** Any time something is broken, slow, surprising, or merely
"the way it has always been done here", stop and check what the frontier
actually does now. Do not pattern-match to what you already know. Your training
data is older than the frontier.

**When you research, it must be recent and it must be frontier.**

- Prefer the **last 6-12 months**. Prefer the current and previous year.
  Anything older than ~2 years is suspect and must be justified explicitly.
- Cite **primary sources**: technical reports, model cards, dataset papers,
  official lab blogs. Not blog-summaries-of-blogs, not tutorials, not Stack
  Overflow, not a model's own recollection.
- **Give the date** (month + year) next to every claim. If you cannot date it,
  you cannot use it.
- Go to the **frontier and the exotic edges**: the labs actually training
  frontier models, and the live research fringe next to them. Open, transparent
  labs (AI2/OLMo, HuggingFace/FineWeb, NVIDIA/Nemotron) publish the most
  pipeline detail and are the best primary sources; the most capable models
  (DeepSeek, Qwen, Kimi, Gemini, Llama, GLM, MiniMax) set the bar even when they
  disclose less.
- **Say what is undisclosed.** "DeepSeek does not publish this" is a finding.
  Guessing and presenting it as practice is not.
- Distinguish **consensus** from **contested** from **outdated**. Report all
  three. An outdated-practices list is often the most useful output.

**Do not bring back GPT-3-era methods.** Techniques that were standard in
2020-2023 and have since been measured as suboptimal are worse than no
recommendation, because they look authoritative. Examples of the failure mode:
recommending C4/Gopher heuristics as the primary quality gate, global
cross-snapshot dedup for English, a 32k-50k vocabulary, KenLM perplexity as the
main quality filter, or flat single-stage pretraining with no annealing. All of
these were correct once. None of them are the frontier now.

If research contradicts something already built here, **say so plainly**. A
finding that the current design is behind the frontier is exactly the finding
worth having. Do not soften it.

---

## 2. Silent wrongness is the failure mode that matters

This pipeline is provenance-sealed: manifests are hashed, inputs are frozen,
outputs are checksummed. That machinery catches corruption. It does **not** catch
a stage that runs, verifies, and quietly does the wrong thing.

The failures that have actually cost time here look like this:

- A partition that drops task indices. Every shard that *is* processed verifies
  perfectly; the corpus is simply missing the ones that were never assigned.
- A hand-written twin of a reference implementation that drifts. Both run, both
  verify, they disagree, and nothing compares them.
- A filter tuned so aggressively it removes far more than intended, biased
  toward exactly the documents you most wanted to keep.

So:

- When you write a partition, a split, or a shard assignment, **test coverage
  explicitly**: every unit covered exactly once, no gaps, no duplicates.
- When two implementations of the same rule exist (a readable reference and a
  fast ndarray twin, say), **pin them to each other with a differential test**.
  If one is unused, delete it rather than leave it to drift.
- Prefer a test that would **fail if the behaviour changed** over a test that
  merely exercises the code. Mutate your own logic and check the test catches
  it; a test that survives every mutation is decoration.

---

## 3. Do not disturb a live pipeline

Runs take days and cost real allocation. Before changing anything:

- **Check what is running.** `squeue`, `sacct`, and the stage logs under
  `${lustre_root}/logs/`. Jobs already in the queue carry their `--export`
  environment from submission time; code on disk changes under them.
- **Editing a source file does not affect a running Python process.** It has
  already imported the module. A deploy is for the *next* launch.
- **Slow is not the same as wrong.** Measure before you kill anything. Work that
  is 95% done and correct is worth more than a restart on faster code.
- **New defaults must not break in-flight work.** `build.inputs.json` is frozen
  per release and re-derived on every submission, so changing how inputs are
  enumerated mid-release blocks resubmission. Ship such changes disabled and
  document when to enable them.
- Prefer deploying by **commit and pull**, not by copying files onto the
  cluster. Provenance is the point of this repository.

---

## 4. Measure on the real thing

Synthetic microbenchmarks mislead, especially here, where the working set is tens
of gigabytes and memory latency dominates.

- Profile the **actual running process** (`py-spy dump --pid ...`) before
  theorising about where time goes.
- Benchmark against the **real index and real documents**, at real scale. A
  2M-row stand-in for a 72M-row array gives the wrong answer about cache
  behaviour, and therefore the wrong optimisation.
- For any change that alters filtering or selection, prove the decisions are
  **identical** on real data before claiming it is only a performance change.

---

## 5. Conventions

- Commit directly to `main`; history is linear.
- Commit messages explain **why**, in prose, at length. Look at `git log` before
  writing one. They read as engineering notes, not changelogs.
- Comments explain **why**, never what. Most code needs none.
- Tests are `unittest` classes under `tests/`, run with `pytest`.
- The CPU data runtime does not have `torch`; the training tests cannot run
  there. Run the data suite with the GPU-dependent modules ignored.
- Never commit secrets, credentials, or data samples.
