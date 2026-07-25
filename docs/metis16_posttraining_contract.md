# Metis-1.6 context extension and post-training contract

`configs/metis16/posttraining.yaml` is the immutable stage contract.
`metis_training.posttraining` supplies the loss functions, artifact verification,
checkpoint lineage, resume logic, and backend orchestration. It deliberately
does not claim that unavailable post-training data, teachers, or reward
environments exist.

## Locked sequence

For Praxis and Logos independently:

1. direct 4,096 -> 163,840-token continued-pretraining jump over exactly 18B
   active tokens, deploying at 131,072 tokens;
2. cold-start SFT with 92% of examples in 8K-32K and a deliberately small 8%
   in 32K-64K;
3. overall SFT with a 65% 8K-32K, 25% 32K-64K, and 10% 64K-131K mix;
4. a 10% cross-tokenizer DeepSeek-DPD pilot and live promotion gate;
5. full cross-tokenizer DeepSeek-DPD;
6. five independent GSPO specialist branches from that same unified
   checkpoint: reasoning, code, knowledge, writing, and agentic;
7. same-tokenizer OPD that consolidates all five specialists into the
   untouched unified student;
8. a side-branch pairwise reward model;
9. frozen-reward GSPO preference alignment;
10. evaluation and a local publish-readiness gate.

The reward-model checkpoint never replaces the policy checkpoint in lineage.
Likewise, specialist branches never become the unified policy directly. OPD
starts from the exact DeepSeek-DPD checkpoint and sees all five specialist
checkpoint hashes at once.
The final gate creates a sealed local candidate only; it does not upload model
weights.

## Production in-process backend

`metis-posttrain run` launches the Portage campaign. After base pre-training,
every initialized family rank enters
`metis_training.stage_backend.run_posttraining_campaign` from `train.py` with
the live model, optimizer, parallel topology, signal coordinator, and
`CheckpointManager`. There is no nested trainer command and no rank-zero-only
model update: policy forwards, backwards, expert-parallel collectives, gradient
synchronization, and distributed checkpoint writes remain collective across
the family allocation.

Checkpoint-producing stages write native
`metis.distributed-checkpoint/v1` manifests. Their `extra_state` binds
`posttraining_stage`, `parent_checkpoint_sha256`, and
`stage_config_sha256`; campaign state and receipts also bind the pipeline,
family manifest, tokenizer, canonical-ID sidecar, measured autotune selection,
precision-role plan, and optimizer-state policy. Resume rehashes those
contracts and refuses lineage or byte drift. Evaluation and publish stages
write sealed `metis.evaluation-results/v1` and
`metis.publish-candidate/v1` artifacts.

### Explicit legacy interface

Only `metis-posttrain legacy-run` uses the deprecated external-backend
orchestrator. That compatibility path invokes `METIS_POSTTRAIN_BACKEND` with
`--runtime-spec /absolute/path/to/RUNTIME.json` and expects a self-hashed
`metis.stage-output/v1` receipt. It is not used by the autonomous production
campaign.

## Required inputs

The environment variables in the YAML must point to complete, hash-sealed
manifests. Among other checks, the contract requires:

- both direct and `<think>` answer modes in both SFT stages;
- identity, safety, abstention, deduplication, and contamination audits;
- DeepSeek positives, fresh Metis negatives, a verified sequence-preference
  margin, and log probabilities from the frozen overall-SFT reference;
- strict 10%-90% avg@16 prompt filtering before each RLVR domain;
- deterministic sandboxed STEM/code verifiers, source-pinned knowledge and
  writing judges, and a pinned agent environment;
- single-use unified-student OPD trajectories plus reverse-KL logits over the
  union of the student and routed specialist top-32 token sets;
- human-preference provenance and position-balanced reward-model pairs;
- per-candidate preference-alignment scores produced once by the frozen
  pairwise reward backbone and head at the exact parent policy checkpoint;
- a sealed evaluation suite with thresholds and holdouts.

The orchestrator refuses missing inputs even in dry-run mode. This is
intentional: a training process cannot autonomously manufacture trustworthy
teachers, private tests, human preferences, licenses, or benchmark holdouts.

Context-extension and SFT artifacts are shared by Praxis and Logos. DPD
negatives, specialist avg@16 selections, OPD trajectories, preference
candidates, and final evaluation results are not: each generated manifest
names its family and exact generating checkpoint hash. The static evaluation
suite is shared, but its sealed adapter runs only after preference alignment
and emits results bound to that live family checkpoint. The two
families may share the underlying prompt pool, but cannot share policy-dependent
negative samples or filtered selections.

The Portage one-command path packages those inputs as one
`metis.posttraining-release-umbrella/v1`. The umbrella self-hash and exact
post-training YAML file hash pin separate Praxis and Logos
`metis.posttraining-release-index/v1` files. Each family index contains a
structured, file-hashed tokenizer manifest entry and stage/name requirement
records. A requirement is either:

- `state: sealed`, with its manifest file hash and sealed-envelope self-hash; or
- `state: deferred`, with a safe future manifest path and a generation hook
  whose executable SHA-256, arguments, timeout, and receipt path are pinned.

All paths are relative to the family index root; absolute paths, `..`, and
symlinks are rejected. The release builder installs one byte-pinned
`metis16-posttraining-materialize` hook for every production deferred stage.
That hook will execute only a `metis.generation-adapter/v1` executable already
contained in, hashed by, and marked executable in a static sealed requirement.
This makes the DeepSeek endpoint client, verifier code, prompt source,
same-tokenizer OPD router, reward scorer, and final evaluator part of the Rhea
audit surface instead of an
untracked command supplied on Portage.

The trainer never starts a nested generator while its
family allocation is occupied. Instead, every rank collectively seals one
`metis.deferred-materialization-request/v1`, exits with the supervisor handoff
status, and `FamilySupervisor` runs the hash-pinned hook after the trainer step
has released the nodes. `distributed_family_v1` launches one task per family
rank; `rank0_only_v1` launches exactly one task with its declared GPU count.

Every hook task writes a self-hashed
`metis.generation-hook-rank-receipt/v1`. The supervisor reducer requires exact,
duplicate-free rank coverage and writes
`metis.generation-hook-receipt/v2`. The request, rank receipts, reducer receipt,
and sealed output all bind the family, stage, requirement, parent-checkpoint
SHA-256, immutable `stage_bindings`, release-index file and self hashes,
requirement-record hash, deep-verification receipt, executable hash, execution
protocol, world size, sealed adapter contract, and output file and self
hashes. Every generated stage binds `parent_policy_checkpoint`. DPD also binds
`dpd_reference_checkpoint` to overall SFT. OPD additionally binds
`unified_student_checkpoint` and all five `specialist_checkpoints`.
Preference alignment binds `reward_model_manifest`. The supervisor
deep-verifies all those distributed checkpoints and receipts before allocating
the adapter step. This is how on-policy DPD, specialist RL, OPD, and
reward-scored artifacts are created at the correct point in lineage without
pretending that a future checkpoint exists. The family process receives only
the validated umbrella through `METIS_POSTTRAINING_RELEASE_INDEX`; direct
artifact environment variables are not the autonomous production path.

The tokenizer entry is not accepted on a claimed hash alone. Its sealed
`tokenizer_file` bytes must hash-identically match the tokenizer artifact in
the verified 1T base-data release. Metadata also binds the base release hash,
the tokenizer and canonical-sidecar paths relative to that release, and both
canonical-sidecar hashes. Portage recomputes the canonical map semantics from
the live base tokenizer before post-training begins.

`./metis-posttrain` is the repository-local operator surface for this production path:
`release-build` seals the Rhea handoff, `release-status` verifies it without an
allocation, `status` inspects the campaign, and `run` launches or resumes the
Portage campaign whose family trainers call the in-process backend. The older
external-backend orchestrator is retained only as the explicitly named
`legacy-run` command; it is not the default execution path.

Preference-alignment rewards are never recomputed through the policy being
optimized. Its mmap bundle carries a `metis.frozen-reward-scores/v1` contract
that self-hashes the score array and candidate IDs, the pairwise reward-model
manifest, its frozen parent policy checkpoint, and the canonical-ID sidecar.
The backend verifies those bytes before GSPO and consumes the immutable
`reward_scores` array. Any score, candidate, checkpoint, or reward-model drift
fails before the first optimizer step.

The cold-start SFT floor is 30 million source QA instructions, not 30 million
fully padded tensors. Its sealed metadata therefore proves
`source_instruction_count` separately, declares document-isolated packing,
and proves the 92%/8% length-bucket mass. Overall SFT separately proves its
65%/25%/10% mix, including genuine 64K-131K examples. Shorter SFT examples are
intentional: a 131K-capable model still needs far more ordinary conversations
than exceptionally long ones, while the 10% long bucket teaches the chat and
tool formats at the deployed limit.

## Important DPD and OPD boundary

Nanbeige DPD includes token-level probability distillation, which assumes a
shared token probability space. DeepSeek-V4-Flash and Metis do not share a
tokenizer, so the production YAML explicitly does **not** claim token KD:
both token-distillation weights are zero. The local DeepSeek service generates
verified positive sequences; the frozen Metis policy generates negatives and
scores both sides after Metis retokenization; training uses the sequence-level
preference margin. This is cross-tokenizer sequence DPD, not fake vocabulary
alignment.

Dense probability distillation happens later where it is valid. All five
Metis specialists share the Metis tokenizer and architecture. OPD samples
single-use trajectories from the current unified Metis student, routes each
prompt to its sealed domain specialist, and computes reverse KL over the union
of the student and teacher top-32 token sets (up to 64 unique IDs). The
trainer validates the student and every specialist checkpoint hash before the
first OPD update.

## Research-to-code mapping

- [Nanbeige4-3B](https://arxiv.org/abs/2512.06266): two-stage SFT, DPD,
  strict avg@16 filtering, STEM -> code -> preference RL, and the dedicated
  pairwise reward model.
- [Nanbeige4.1-3B](https://arxiv.org/abs/2602.13367): turn-level agent credit,
  correctness-gated code-efficiency rewards, and swap consistency.
- [GSPO](https://arxiv.org/abs/2507.18071): length-normalized sequence
  likelihood ratio, sequence clipping, GSPO-token, and the MoE stability
  rationale. The 3e-4/4e-4 clips are paper starting values, not guaranteed
  optima for Metis.
- [DAPO](https://arxiv.org/abs/2503.14476): no KL, asymmetric clipping,
  dynamic sampling, and masking truncated responses. Metis uses Nanbeige's
  stricter 10%-90% band rather than DAPO's exact-zero/exact-one filter.
- [DAST](https://arxiv.org/abs/2503.04472): the difficulty-adaptive token
  budget. Metis adapts that budget into a two-sided,
  correctness-dominant reward: correct answers too far below or above the
  difficulty-conditioned budget are penalized, while wrong answers receive no
  length shaping. This adaptation is not attributed to DAST.
- [MiniCPM5-1B](https://github.com/OpenBMB/MiniCPM): independent
  domain-specialist RL, same-tokenizer top-k-union reverse-KL OPD, reuse of
  specialist prompt domains, and a two-stage reasoning-length schedule.
- [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash):
  local high-capability bootstrap teacher and the independent-specialist ->
  unified-OPD structure. Metis does not copy its tokenizer or treat a
  generation endpoint as token-level logits.
- [Nemotron Nano 2](https://arxiv.org/abs/2508.14444) and
  [Nemotron 3](https://arxiv.org/abs/2512.20856): direct NoPE context
  extension, overshoot, synthetic long-context data, and concurrent
  short-context replay.

The context data rationale and exact source table are in
[`metis16_context_extension_data.md`](metis16_context_extension_data.md).
The 163,840-token train length is the conservative 1.25x overshoot for a
131,072-token deployment target. The 18B budget has durable 6B/12B/18B
checkpoints, each scored on 384 disjoint 131K records; after the final gate,
the trainer restores the best passing checkpoint rather than automatically
using 18B. Portage memory and throughput probes may still reject the shape; no
paper result substitutes for a real Metis/MI300A canary.

Reasoning, code, knowledge, and agentic specialists spend the first 60% of
their optimizer steps on correctness only. The last 40% enables the selected
difficulty-adaptive coefficient. This prevents early length shaping from
teaching short wrong answers, while the two-sided target prevents both
underthinking on hard prompts and overthinking on easy prompts. Writing keeps
the coefficient at zero and relies on its calibrated judge and deterministic
style constraints.

The papers do not establish optimal DPD beta/loss weights, GSPO clips, or
length-shaping coefficients for a small recursive MoE. The YAML treats its
listed values as bounded candidates: the DPD pilot selects its full-stage
profile, and each RLVR stage runs a short stability/quality selection before
committing. A failed or non-finite candidate cannot be promoted. The backend
must put the passing selection and its `selected_profile_sha256` in the stage
output receipt; resume rejects a missing or changed selection.

Those selections are live Portage trials, not trusted numbers copied into a
data manifest. Every DPD-pilot and RLVR mmap bundle must seal a
`profile_selection.live_autotune` contract and held-out arrays. The contract
is self-hashed, bound by the profile-selection receipt, and must use the
pipeline's exact two-step canary policy. The family ranks restore the same
parent distributed checkpoint, optimizer transition, and RNG seed before
every candidate; run the candidate update on the ordinary EP graph; replay
the held-out evaluator; compare its accumulators with the independently
sealed evaluator receipt within the pipeline tolerance; and restore the
parent again before the promoted stage starts. An atomic, self-hashed receipt
under `posttraining/<family>/autotune/` binds the candidate set, parent,
bundle, evaluator arrays, precision-role plan, base autotune profile, rank
topology, runtime inventory, timings, peak HBM, and selected profile. A
complete matching receipt is reusable after requeue. A partial receipt can
continue only before policy training starts; an active policy checkpoint
without a complete matching receipt fails closed.

The DPD replay implementation is
`metis.dpd-preference-replay/v1`. Its separate evaluation arrays are:

- `autotune_evaluation_{positive,negative}_input_ids`
- `autotune_evaluation_{positive,negative}_attention_mask`
- `autotune_evaluation_{positive,negative}_response_mask`
- `autotune_evaluation_role`, with all three sealed roles represented:
  primary, reasoning, and self-correction

It compares length-normalized positive-versus-negative preference scores from
the parent and candidate. The RLVR replay implementation is
`metis.rlvr-offline-policy-replay/v1`. It seals candidate IDs, attention and
response masks, verifier `correctness`, and truncation arrays (plus
`efficiency_reward` for code). It recomputes the policy distribution over the
16 fixed verified candidates, expected reward, entropy, and correct-response
NLL. Truncated candidates are excluded. Thus no new quality labels are
invented inside the trainer: preference roles and verifier results originate
in the sealed evaluator payload, while the live policy probabilities are
recomputed on the exact family ranks.

DPD and RLVR training bundles must also contain `split_fingerprint`, and their
live evaluator must contain `autotune_evaluation_split_fingerprint`. Each is an
`[records, 32]` uint8 array of unique SHA-256 rows. The trainer does not trust
those rows as labels: for DPD it recomputes each hash from the complete
positive/negative token, attention, and response-mask record; for RLVR it
recomputes each hash from the prompt prefix shared by all 16 candidates. It
then rejects any training/evaluator intersection. Forged fingerprints,
duplicate prompts, or held-out leakage therefore fail before a candidate
update.

Each mmap `working_set` must also carry a self-hashed
`metis.posttraining-working-set-autotune/v1` candidate list. Candidates may
change `micro_batch_size`, `token_chunk_size`, and, for grouped RLVR,
`candidate_micro_group_size`; they must preserve the sealed effective local
batch through compensating gradient accumulation and preserve the exact
record/token budget. One warmup and three measured live forward/backward
canaries run on every family rank. Candidates that exceed recomputed
HBM/host limits or actually OOM are rejected, and the fastest safe median
throughput wins with p95 and canary hashes recorded. A later measured stage
OOM still uses the existing checkpoint-safe downward migration path.
