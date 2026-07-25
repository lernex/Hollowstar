# Metis-1.6 autonomous Portage training

This is the Portage-side handoff after Rhea has emitted the immutable verified
`metis-1.6-data-r1` release. Portage does not acquire, normalize, tokenize, select, or repack
training data.

## The one operator command

From a clean clone on a Portage login host:

```bash
./ops/start-portage-training.sh
```

This command is intentionally the only Portage training command, but it is not
a data-authoring command. Before it is run, Lustre must contain:

- the immutable `metis-1.6-data-r1` release built on Rhea;
- the sealed post-training static datasets, teachers, verifiers, evaluation
  suite, and pinned generators described below; and
- either a compatible Portage site software stack, a SHA-256-sidecar-pinned
  Apptainer image, or a local self-hashed runtime bundle.

The launcher discovers and validates those inputs. It cannot manufacture
licensed teacher outputs, human preferences, private tests, a site ROCm build,
or a Slurm account. Their absence is a pre-allocation failure, not a late
training crash.

The default project root is `/lus/lustre1/vollmerc/metis-1.6`. An alternate confirmed
user-owned child may be supplied with `--lustre-root`. The launcher is idempotent: running the same
command again reports the live campaign rather than submitting a second copy.

The command performs only bounded login-side work. It first finds a policy-compatible Python
3.11/3.12 plus PyYAML. If the active interpreter cannot load it, the standard-library bootstrap
tries only live-listed site modules and then a local `metis.portage-runtime-bundle/v1` containing an
exact-ABI, SHA-256-pinned PyYAML wheel. Installation is offline (`--no-index --no-deps`) and writes a
self-hashed receipt. It never fetches an unpinned login dependency.

The launcher then verifies the cluster identity, `parry` partition, 128-node/512-APU GPU TRES,
five-day wall limit, Lustre mount, clean Git commit, base-data release, and post-training release
umbrella. The umbrella hash-pins one backend-native index per family. Static artifacts must already
be sealed. The release builder installs the repository's single byte-pinned distributed
materialization hook; it executes only stage adapters already sealed inside a DeepSeek capability,
verifier, OPD capability, preference, or evaluation artifact. Generated
output is bound to the live parent and, where applicable, the frozen
overall-SFT reference, all five specialist checkpoints, or the frozen reward
model. Missing static data, adapter bytes, or lineage fails before the
expensive stage.
The benchmark suite is static, but its result bundle is generated only after
preference alignment and is hash-bound to that exact family checkpoint.

The login preflight also runs `sbatch --test-only` with the exact 128-node,
512-task production envelope. This validates the current user's account,
partition, QoS, reservation, GPU, and wall-time access without submitting a
job. If the site default association is valid, no account flag is needed.
Otherwise set `METIS_SLURM_ACCOUNT` and, when required,
`METIS_SLURM_QOS` or `METIS_SLURM_RESERVATION`; an unusable combination fails
before the first bring-up job is submitted.

The Rhea-side release builder is:

```bash
./metis-posttraining-release \
  --config configs/metis16/portage-training.yaml \
  --write-template /path/to/posttraining-build.json

# Fill the convention-based paths with the actual sealed artifacts. Static
# capability/verifier manifests carry metis.generation-adapter/v1 executables;
# the template has already installed and referenced the common hook.
./metis-posttraining-release \
  --config configs/metis16/portage-training.yaml \
  --spec /path/to/posttraining-build.json \
  --json-output /path/to/posttraining-build-result.json
```

The build specification uses
`metis.posttraining-release-build/v1` and contains exactly `praxis` and
`logos`, then every enabled `stage -> requirement` in
`configs/metis16/posttraining.yaml`. Static records name a complete
`metis.sealed-artifact/v1` manifest. Checkpoint-bound records must be deferred
and name the installed pinned hook, bounded adapter-source arguments and
timeout, an output
manifest, a reducer receipt, a rank-receipt directory, and either
`distributed_family_v1` or explicitly bounded `rank0_only_v1` execution. The
builder rejects any manually claimed tokenizer envelope: it copies and seals
the exact tokenizer bytes from the verified base release and binds the
canonical-ID sidecar itself.

The builder hashes every static payload byte and writes
`DEEP_VERIFICATION.json`, two family indexes, and the umbrella. Portage login
preflight checks paths, self-hashes, sizes, and immutable bindings without
rereading the whole release. The later 128-rank compute audit independently
rehashes every static post-training payload exactly once before production.

No package, wheel, ROCm build, container, module name, Slurm account, QoS, reservation, kernel, or
performance number is guessed. The compute runtime resolver tries a compatible active stack,
live-listed site modules, a SHA-256-sidecar-pinned Apptainer image, and finally the local self-hashed
wheel/source bundle. It records the exact Python ABI, PyTorch/ROCm build, loaded modules, container
or bundle hash, Transformer Engine, Mamba SSM, causal-conv1d, FlashAttention, AITER, hipBLASLt,
Composable Kernel paths, and RCCL facts. If no complete runtime passes, it stops with the attempted
inventory.

## Autonomous dependency chain

Slurm receives one dependency-safe chain:

1. compute-node ROCm, PyTorch, RCCL, NUMA, Lustre, CXI, and software inventory;
2. one-MI300A gfx942/BF16/hipBLASLt checks, exact-family-width GEMMs, real Transformer Engine FP8
   forward/backward when available, fused Mamba-2 packed-reset forward/backward, and packed causal
   varlen FlashAttention GQA isolation checks;
3. four-APU intra-node all-reduce, all-to-all, and variable-count all-to-all correctness/timing;
4. sixteen-APU multi-node collective correctness, bandwidth, rank variation, and CXI checks;
5. a 128-rank distributed deep audit in which every base-data binary/index
   shard and every sealed static post-training payload is hashed exactly once;
6. one exclusive 128-node allocation split into simultaneous Praxis and Logos steps.

Every stage has a self-hashed report and completion marker. `afterok` dependencies prevent later
jobs from starting when a gate fails. A bounded retry handles transient step/RCCL initialization
faults, but deterministic failures are never rewritten as successes.

The distributed data marker is:

```text
$METIS_PORTAGE_STATE_ROOT/campaigns/<campaign-id>/gates/release_verification.json
```

It uses `metis.portage-release-verification/v1` and binds the exact `RELEASE.json`, release
self-hash, pretraining contract, shard manifest, canonical shard inventory, phase totals, and every
rank receipt. Its nested
`metis.portage-posttraining-release-verification/v1` marker binds the
post-training preflight, deep inventory, byte total, and every post-training
rank receipt. Both trainers re-check the cheap bindings before reading
training tokens.

## Measured tuning, not static guesses

Praxis and Logos manifests bound every permitted micro-batch, accumulation count, learning rate,
precision profile, compile mode, and dispatch-overlap choice. The launcher cannot try a value
outside those lists.

On the exact production rank counts it:

- tests memory headroom, finite loss, zero MoE token drop, collective health, BF16 parity, and
  measured tokens/second;
- uses FP8 whenever its real compute smoke and numerical canary pass; BF16 remains the correctness
  oracle and fail-closed fallback rather than replacing a safe FP8 lane merely because a short BF16
  canary timed faster;
- measures replicated and row-sharded N-gram table layouts against the same candidate, permits row
  sharding only after a zero-drop/loss-parity check, and selects placement from measured HBM and
  throughput;
- runs bounded optimizer canaries and prefers the manifest's intended learning rate only when its
  gradient, update-ratio, non-finite, and short-loss gates pass;
- races complementary contiguous and interleaved node mappings while preserving
  `Praxis EP=128 × 1` and `Logos EP=192 × 2`;
- self-hashes the selected profile against the release marker, model manifest, Git commit, GPU,
  ROCm, PyTorch, memory, and loaded-module fingerprint.

PyTorch TunableOp/hipBLASLt results are stored per rank so nodes never race while writing a shared
CSV. A software/runtime change invalidates both PyTorch's kernel records and the higher-level Metis
profile.

## Simultaneous production and recovery

The family allocation uses:

- Praxis: 32 nodes / 128 MI300A APUs / `EP=128`, one expert replica;
- Logos: 96 nodes / 384 MI300A APUs / `EP=192`, two expert replicas.

They are concurrent disjoint `srun --exact` steps inside one 128-node job, so they cannot start on
different reservations. Both consume the same immutable shard order and token-axis phase contract.

Slurm signals the batch shell 15 minutes before its segment limit. The signal is forwarded through
the supervisor and synchronous `srun` steps to both trainers. A job is requeued only after the
trainers return the checkpoint-safe code and the supervisor writes a self-hashed `resume_safe`
marker. Node failure and preemption use Slurm requeue. A measured base-training OOM rejects that
exact autotune candidate, runs a canary for a smaller manifest-bounded micro-batch with compensating
accumulation, preserves the exact effective token batch, deep-hashes the latest checkpoint
artifacts, and writes a profile-migration receipt. The trainer accepts that receipt only when every
other profile field and the parameter/optimizer layout are unchanged. Context-extension and
post-training batches use an analogous stage-bound override receipt; sealed dataset manifests are
never edited. Automatic restarts are capped; unknown trainer failures stop closed.

Before DPD-pilot and every RLVR stage, the live family allocation also replays every bounded
quality candidate from the same parent checkpoint and RNG state against separately sealed held-out
arrays. Claimed profile metrics alone are rejected. The trainer first benchmarks the
manifest-bounded micro-batch/token-chunk/group candidates, preserving effective batch and token
budget, then records the fastest safe working set. It next runs the two-step DPD or GSPO trials,
recomputes evaluator accumulators, and promotes only a gate-passing reproduced result. Both
decisions are atomic, self-hashed, topology/runtime/precision-bound, and safe to reuse after
requeue; an active stage without its complete matching receipts stops closed.

The trainer's `--stage all` sequence owns base phases A/B/C, context extension, and post-training.
Each stage consumes the previous stage's durable checkpoint and lineage receipt.

Context extension consumes exactly 18B active tokens at a 163,840-token train
length. It writes durable checkpoints at the first optimizer boundary after
6B, 12B, and 18B, evaluates each on the same 384 training-disjoint 131K
records, and restores the highest-scoring passing checkpoint. Its source
mixture and Rhea build are documented in
[`metis16_context_extension_data.md`](metis16_context_extension_data.md).

Post-training then runs mixed-length cold SFT, mixed-length overall SFT,
cross-tokenizer DeepSeek sequence DPD, five independent Metis specialist
branches, same-tokenizer OPD consolidation, pairwise reward modeling,
preference GSPO, evaluation, and a local publish gate. Reasoning specialists
use 60% correctness-only optimization before enabling the two-sided adaptive
thinking budget for the final 40%.

## Telemetry and completion

One CPU-only sampler per node records ROCm utilization, memory, clocks, power, temperature, and
selected Cassini/CXI congestion/retry counters throughout the run. The trainer writes one telemetry
stream per rank with tokens/second, FLOPs, MFU, all-to-all bytes/time, expert-load variation, loss,
token cursor, and overflow drops.

`COMPLETE.json` is not written unless:

- all 128 Praxis and all 384 Logos rank streams exist;
- their last rows are finite and reach at least the 1T base-token cursor;
- all-to-all traffic was measured;
- every recorded MoE overflow-drop count is zero;
- before/after telemetry covers all 128 allocated nodes.

To inspect the campaign without changing it:

```bash
./ops/start-portage-training.sh --status
```

The output includes campaign paths, stage gates, live Slurm queue rows, accounting rows, and any
fail-closed report.
