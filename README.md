# Hollowstar

An end-to-end language-model research and training stack.

Hollowstar is the codename for an active research system spanning corpus
acquisition and curation, tokenizer construction, pretraining, post-training,
evaluation, and release. The repository is public for operational reasons. Its
visibility is not a release announcement, a stability guarantee, or a claim
that every experimental path is ready for outside use.

## Status

This is a working research repository, not a packaged framework. Interfaces,
recipes, and infrastructure may change as experiments close open questions.

- `main` reflects current development rather than a supported release.
- Production runs are defined by frozen manifests and executable configuration,
  not by estimates in prose documents.
- Data and release stages are provenance-sealed and fail closed when required
  inputs, checksums, or environment facts cannot be verified.
- Hardware fit, throughput, and quality are treated as measurements tied to a
  specific configuration and runtime.

## What is here

The repository covers the full path from raw sources to releasable weights:

- source acquisition, license policy, normalization, quality filtering,
  deduplication, decontamination, exact selection, and packing;
- tokenizer sampling, training, validation, and immutable handoff artifacts;
- recurrent, attention, routed, and conditional-memory model experiments;
- distributed pretraining, context extension, checkpointing, and recovery;
- supervised, distillation, preference, and reasoning post-training stages;
- evaluation, export, release verification, and operational telemetry;
- correctness tests for partition coverage, implementation parity, frozen
  inputs, release integrity, and failure recovery.

## Source of truth

When two parts of the repository disagree, use this order:

1. frozen inputs and release manifests for the specific run;
2. executable profiles under `configs/`;
3. the implementation under `src/`, `ops/`, and `slurm/`;
4. tests that pin the intended behavior;
5. plans and narrative documents.

Plans preserve design reasoning, but they are not automatically current. Never
derive a launch configuration from a prose document when an executable profile
exists.

## Repository map

- `configs/` — model-family, data, runtime, and site profiles
- `manifests/` — source policy, release contracts, holdouts, and replacements
- `src/` — data, model, training, orchestration, and release packages
- `ops/` — operator entry points and environment bootstrap
- `slurm/` — scheduler launchers and staged job definitions
- `tests/` — behavioral, differential, integrity, and recovery tests
- `docs/` — runbooks, research records, and historical plans
- `scripts/` — local experiments, conversion tools, and legacy workflows

Generated corpora, credentials, checkpoints, and release payloads do not belong
in Git.

## Local development

Python 3.11 or 3.12 is required. A general development environment can be
created with:

```bash
git clone https://github.com/lernex/Hollowstar.git
cd Hollowstar
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the test suite with:

```bash
python -m pytest tests
```

GPU-dependent training tests require the matching training runtime. CPU data
environments should run the data suite with GPU-only modules excluded rather
than installing an unverified accelerator stack.

Production environments use their committed, hash-locked dependency set and
site profile. The editable setup above is for development; it is not a
substitute for the production bootstrap.

## Operating boundary

Before changing or launching a production stage:

1. inspect the scheduler and current stage logs;
2. identify the exact commit and frozen input contract already in use;
3. confirm the selected site profile and storage root;
4. run the profile doctor and required preflight checks;
5. submit through the checked-in operator path;
6. verify produced manifests, checksums, and handoff reports before advancing.

Editing a source file does not update an already-running Python process, while
queued jobs may retain environment values captured at submission. Changes are
for the next launch unless a run is deliberately and safely restarted.

## Research standard

Architecture and training decisions should be based on recent primary research
and measurements on representative workloads. Older defaults are not accepted
merely because they are familiar. Every optimization that claims behavioral
equivalence should prove it against the readable reference implementation, and
every partitioning change should prove exact coverage with no gaps or
duplicates.

## Security and provenance

Do not commit credentials, access tokens, private data samples, cluster paths
containing secrets, or generated model artifacts. Preserve the chain from
source manifests through frozen inputs, checksummed outputs, and release
verification; a successful process exit is not proof that the intended corpus
or model was produced.
