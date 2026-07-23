# Metis-1.6 login2, Rhea, and Portage operator runbook

The three environments have deliberately separate responsibilities:

| Environment | Responsibility | Profile |
|---|---|---|
| `login2` | Authenticate, acquire raw candidates and evaluation holdouts onto Lustre | `login2` |
| Rhea | Normalize, filter, deduplicate, decontaminate, train the tokenizer, tokenize, select, and shard | `rhea` |
| Portage | Consume the verified immutable release for model pretraining | trainer configuration, not a data-prep profile |

## Information confirmed by the account owner

- The account is `vollmerc` in group `sumusa`. The acquisition root is
  `/lus/lustre1/vollmerc/metis-1.6`; the launcher creates it if needed. `/lus/lustre1` itself must
  never be used as the data root.
- `login2` is the permitted acquisition host.
- GNU Screen, `/usr/bin/python3.11`, Cray Python modules, Apptainer, and the Lustre tools are present.
- Hugging Face, GitHub, and Common Crawl HTTPS endpoints are reachable without a proxy.
- A `0/0` default quota report does not prove that usable capacity has been assigned. Capacity confirmation
  remains an operator prerequisite.
- Rhea's scheduler account, partition, QoS policy, array limit, node memory, wall-time limit, and
  view of the Lustre filesystem are not yet known. Its profile therefore fails closed.

Before the first login2 launch, confirm three remaining account-level prerequisites:

- the account owner can clone the private `lernex/Metis` repository;
- the user's Hugging Face account has accepted the gated HLE and GPQA dataset terms as well as the
  gated NVIDIA source terms (the preflight checks every one and stops before large downloads);
- a read-only GitHub token with repository Metadata access is available, or `gh auth` is already
  configured for the account, so FreshGitHub can reject forks and mirrors fail-closed;
- `pypi.org`/Python package-file HTTPS is reachable for the first hash-locked runtime install, and
  at least 3TB free for acquisition (5TB recommended) is genuinely available despite the ambiguous
  `0/0` report. Rhea later requires 8TB free (12TB recommended) while it builds the release.

Portage's `parry` partition, `MaxArraySize=1001`, 10,000-job limit, five-day wall-time limit, and
roughly 512,000MB nodes are Portage facts. They are not copied into the Rhea profile.

## Clone once, then one login2 command

From the account owner's home directory:

```bash
git clone git@github.com:lernex/Metis.git
cd Metis
```

Then the only acquisition command is:

```bash
./ops/start-acquisition.sh \
  --lustre-root /lus/lustre1/vollmerc/metis-1.6 \
  --quota-acknowledgement administrator-confirmed
```

Use `administrator-confirmed` only after the Lustre administrator has confirmed at least the
required usable capacity for this account. If the administrator explicitly confirms that the
project has no hard quota, use `--quota-acknowledgement unlimited` instead. The same value may be
provided through `METIS_LUSTRE_QUOTA_ACKNOWLEDGEMENT`; an ambiguous `0/0` report without either
explicit acknowledgement is a production preflight failure.

The launcher prompts invisibly for the user's Hugging Face read token only when no existing local
credential is available. It also reuses `GITHUB_TOKEN`, `GH_TOKEN`, or `gh auth`, and otherwise
prompts invisibly for a read-only GitHub token. Neither token is placed in the command line, Screen
session name, state files, or logs. A discovered Hugging Face token file must be owned by the current
account, must not be a symlink, and must have no group or world permissions (mode `0600` or stricter).
Never paste either credential into email, Git, YAML, or a shell command argument. GH Archive and
codeload provide events and source bytes; the authenticated metadata check is what makes the
fork/mirror exclusion fail closed. GitHub rate-limit reset headers are honored automatically.

The launcher creates the directory, starts one `metis16-acquisition` GNU Screen session, installs
the pinned Python runtime, runs the full acquisition doctor, resolves the immutable source lock,
and runs the restart-safe supervisor in the foreground inside Screen. The runtime is installed from
`requirements-metis16-data.lock`, which pins the complete transitive dependency graph and includes
SHA-256 hashes for every accepted distribution. Bootstrap uses `--require-hashes` and
`--only-binary=:all:`; it neither performs an unbounded pip upgrade nor compiles unreviewed source
packages on the login host. It returns immediately, so the SSH connection may close without
stopping acquisition.

The runtime contract supports CPython 3.11 and 3.12. Its Linux x86_64 wheels were resolved for both
ABIs; `login2` uses the confirmed `/usr/bin/python3.11`. If the direct input file, generated lock,
Python ABI range, or installed package set differs, bootstrap rebuilds the dedicated virtual
environment before doing any data work. The human-edited input file is not an installation surface:
it exists only to regenerate and review the transitive lock.

Acquisition advances in dependency-safe waves: packaged Hugging Face/index payloads first; Common
Crawl, pinned repository, canonical-source, and recent-GitHub materializers second; and engineering
discussion extraction only after the repository-license cache exists. A failed wave prevents every
dependent wave from starting. Network concurrency is bounded independently for Hugging Face,
Common Crawl, canonical sites, and public GitHub archives.

```bash
screen -r metis16-acquisition
METIS_LUSTRE_ROOT=/lus/lustre1/vollmerc/metis-1.6 ./metisctl status --profile login2
tail -f /lus/lustre1/vollmerc/metis-1.6/logs/metis-1.6-data-r1/acquisition/screen.log
```

Rerunning the launcher is the resume operation. Completed content-addressed tasks are skipped, and
the singleton lock prevents two supervisors from writing the same acquisition concurrently.

The command remains fail-closed if the root is unsafe, capacity is insufficient, a credential or
gated source is unavailable, a materializer has not passed its fixture, a source remains a remote
plan, the repository is dirty, holdouts are incomplete, or an artifact hash/size no longer matches.

## Immutable login2-to-Rhea handoff

Successful acquisition emits `state/metis-1.6-data-r1/ACQUISITION_READY.json`. It binds:

- the fully expanded data manifest;
- the immutable source lock;
- the hash-locked Python dependency contract and acquisition interpreter identity;
- every download completion marker;
- materialized artifact paths, byte sizes, and SHA-256 hashes;
- the evaluation holdout bundle;
- the clean repository commit used for acquisition.

It also reports measured per-source candidate counts and the deterministic replacement allocation.
If one source is short, compatible donor surplus is assigned automatically without changing the
source category, phase, or freshness target. The handoff stops if those reserves are insufficient.
For the three singleton fresh Common Crawl routes, acquisition first widens the five preferred 2026
crawls and then automatically activates `CC-MAIN-2026-04`; it never substitutes historical generic
web. No operator flag is needed for either path.

When a materializer represents its output as a directory, the directory inode is never treated as
an artifact. The handoff records and verifies its acquisition receipt plus every shard named by the
receipt.

Rhea verifies that handoff before submitting any CPU work. A changed manifest, dependency lock,
supported Python ABI contract, source lock, completion marker, holdout bundle, artifact inventory,
or repository commit is fatal. Rhea may use either CPython 3.11 or 3.12, but must install the exact
same dependency lock. The artifacts are bound by paths relative to the acquisition root, so Rhea
may expose the same Lustre content at a different absolute mount prefix.

The submission command performs only fast structural and size checks. It then places a restartable
`handoff_signature` Slurm array ahead of normalization: one task hashes one frozen acquisition
artifact, evaluation holdout artifact, or Common Crawl opt-out artifact. Each successful task writes
an immutable marker bound to the handoff hash. A reducer validates every marker and current file
stat, then writes `HANDOFF_VERIFIED.json`; normalization refuses to run without that exact marker.
Resubmission schedules only missing hash tasks. This avoids making the operator synchronously reread
the entire acquisition before Slurm can accept the build.

The explicit command below remains available as an optional synchronous diagnostic, but it is not
required before `submit build`:

```bash
METIS_LUSTRE_ROOT=/path-visible-on-rhea ./metisctl verify-handoff --profile rhea --deep
```

## Rhea remains intentionally sealed

Once Rhea exists, fill its confirmed Slurm values and verify that it sees the same Lustre directory.
Then bootstrap its own Python environment and submit the build:

```bash
export METIS_LUSTRE_ROOT=/exact/path-visible-on-rhea
export METIS_SLURM_ACCOUNT=CONFIRMED
export METIS_SLURM_PARTITION=CONFIRMED
export METIS_SLURM_MAX_ARRAY_SIZE=CONFIRMED
./ops/bootstrap.sh --profile rhea --role compute --lustre-root "$METIS_LUSTRE_ROOT"
./metisctl doctor --profile rhea --role compute
./metisctl verify-handoff --profile rhea
./metisctl submit build --profile rhea
```

Do not set `scheduler.site_values_confirmed: true` until those values are measured on Rhea. If Rhea
cannot mount the same acquisition content at any path, stop: a separately verified transfer/staging
backend is required before CPU preparation. Do not substitute Portage scheduler values.
