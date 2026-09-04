# Metis-1.6 post-training contract

Status: locked for Metis-1.6
Last revised: September 2026

The executable source of truth is `configs/metis16/posttraining.yaml`. This
document explains the decisions that the manifest and validators enforce.

## Final sequence

```text
context extension
  -> cold-start SFT
  -> overall SFT
  -> shared hybrid-mode GSPO
  -> five parallel capability GSPO specialists
  -> same-tokenizer OPD consolidation
  -> evaluation
  -> local publish gate
```

The five specialists are reasoning, code, knowledge, writing, and agentic
tool use. They are capability specialists, not reasoning-mode specialists.
Every specialist trains all three reasoning modes.

DeepSeek preference distillation is not part of Metis-1.6. The external
teacher and Metis use different tokenizers, making aligned token-level
distillation unnecessarily complex for this generation. There is also no
standalone scalar reward model or terminal general-preference stage. Verifiable
and rubric-scored GSPO supplies the reinforcement signal, and the release ends
on the consolidated policy.

## Public reasoning modes

The tokenizer reserves three immutable control tokens:

| Public mode | Token | Required response |
|---|---|---|
| `direct` | `<|direct|>` | Answer immediately; no visible reasoning trace |
| `think` | `<|think|>` | Use a normal, efficient reasoning trace |
| `think_max` | `<|think_max|>` | Use the largest useful reasoning budget; optimize correctness, not verbosity |

The mode is an explicit prompt/data attribute. Unknown or ambiguous mode labels
are quarantined rather than guessed.

Mode does not control MoRE compute. The routed architecture keeps the same
targets in every mode:

| Mode | Target mean depth | Target mean routed k |
|---|---:|---:|
| `direct` | 2.0 | 4.0 |
| `think` | 2.0 | 4.0 |
| `think_max` | 2.0 | 4.0 |

The difference between modes is the response contract and generation budget,
not a hidden depth or expert-routing contract. Evaluation records depth and k
per mode and fails if this invariance drifts.

## Data construction

Standard datasets do not need to use Metis names. Raw examples remain
immutable; a versioned transformation layer emits `metis.sft-data/v2` or
`metis.rlvr-data/v2` records with an explicit `reasoning_mode` field and the
matching control token.

The transformation pipeline must:

1. classify or deliberately assign a mode using provenance-backed rules;
2. render the Metis chat template and mode token;
3. validate the trace-format contract;
4. retain same-prompt examples across all three modes for a measured overlap
   subset;
5. deduplicate and decontaminate after rendering;
6. audit prompt share and target-token share by domain and mode;
7. quarantine uncertain labels instead of silently mapping them.

Prompt counts alone are not a sufficient balance metric because `think_max`
targets are much longer. Manifests therefore seal actual target-token share by
mode. Same-prompt overlap is also mandatory: it gives training and evaluation a
controlled test of whether the mode token changes behavior without changing
the task.

## SFT

Cold-start SFT installs correct reasoning, self-correction, and trace
boundaries. Its target mix is reasoning-heavy:

- `direct`: 15%
- `think`: 60%
- `think_max`: 25%

Overall SFT broadens chat, knowledge, code, writing, safety, abstention, and
tool use while keeping mode switching reliable:

- `direct`: 45%
- `think`: 40%
- `think_max`: 15%

These are prompt-share targets. Target-token share is measured separately.
Both SFT stages require identity scrub, safety calibration, abstention,
deduplication, contamination checks, mode balance, explicit mode labels, and
same-prompt overlap.

## Shared hybrid-mode GSPO

A short shared GSPO stage follows overall SFT. It reinforces mode selection,
trace format, termination, repetition control, and general behavior before any
capability branch diverges.

Each prompt produces 16 on-policy candidates from the exact parent checkpoint.
Prompts pass only when avg@16 correctness is strictly between 10% and 90%.
Rollouts are single-use and checkpoint-bound. The objective is sequence-ratio
GSPO with no KL penalty and masked truncated samples.

Reward is composed from task correctness or a calibrated rubric judge plus
strict mode compliance. A standalone learned scalar reward model is forbidden.
Length shaping applies only to `think`. `direct` receives no trace-length
reward, and `think_max` receives no positive reward for being longer.

## Parallel capability specialists

All five specialists branch independently from the sealed shared
`hybrid_mode_gspo` checkpoint. Optimizer state resets for each branch, and the
live unified policy is restored after each specialist finishes.

Each specialist sees all three modes, including a same-prompt overlap subset:

| Specialist | `direct` | `think` | `think_max` |
|---|---:|---:|---:|
| Reasoning | 15% | 50% | 35% |
| Code | 25% | 45% | 30% |
| Knowledge | 60% | 30% | 10% |
| Writing | 70% | 25% | 5% |
| Agentic | 20% | 50% | 30% |

These are prompt mixtures, not independent models per mode. The code branch
first optimizes verified correctness and only then adds correctness-gated
efficiency. Agentic training can use turn-level credit. Knowledge and writing
may use calibrated rubric or pairwise judges, but judge scores remain data-side
rewards; they do not create a standalone RM stage.

## OPD consolidation

OPD starts from the same shared hybrid-mode checkpoint and consumes all five
same-origin specialist checkpoints. It is same-tokenizer distillation using
the union of student and teacher top-k support.

Capability is the teacher-selection axis. Reasoning mode is preserved from the
prompt and may not select a different specialist or a different MoRE depth.
The OPD bundle seals specialist checkpoint hashes, the unified student hash,
reasoning mode per row, and domain-by-mode target-token balance.

## Evaluation and release

Evaluation runs against the exact OPD checkpoint and gates:

- `direct`, `think`, and `think_max` format compliance;
- direct-mode trace leakage;
- hard-task gain from `think_max`;
- quality and safety by mode and capability;
- mean MoRE depth and routed k for every mode;
- standard capability, agentic, grounding, abstention, and safety metrics.

The publish gate only seals a local release candidate after evaluation passes.
It performs no external upload and carries no reward-model artifact.

## Executable artifact contracts

- SFT data: `metis.sft-data/v2`
- GSPO data: `metis.rlvr-data/v2`
- OPD data: `metis.opd-data/v1`
- Evaluation results: `metis.evaluation-results/v1`
- Control tokens: declared in both the tokenizer manifest and post-training
  pipeline
- Deferred generation bindings: exact parent checkpoint for GSPO; exact shared
  student plus all five specialist checkpoints for OPD

Every split fingerprint must cover each unit exactly once. GSPO overlap groups
must contain exactly one row for each mode and one shared base-prompt
fingerprint. Manifests self-hash, seal array hashes, and bind tokenizer, family,
checkpoint, and generation lineage.

## Research basis

The recipe is a synthesis; no source discloses this exact Metis sequence.

- MiniCPM5 (May 2026) publishes deep-then-hybrid SFT, capability RL specialists,
  and same-origin OPD:
  https://github.com/OpenBMB/MiniCPM
- UltraData-SFT-2605 (May 2026) publishes explicit thinking/non-thinking
  partitions and mode-aware domain data:
  https://huggingface.co/datasets/openbmb/UltraData-SFT-2605
- A.X K1 (January 2026) reports same-prompt mode overlap, paired-mode SFT/GSPO,
  and format rewards:
  https://arxiv.org/abs/2601.09200
- Nanbeige4.1-3B (February 2026) reports correctness-first,
  correctness-gated code efficiency and agentic RL:
  https://arxiv.org/abs/2602.13367
- Nanbeige4.2-3B (July 2026) reports hybrid-mode RL followed by reasoning RL
  with length control:
  https://arxiv.org/abs/2607.22083
- MOPD (June 2026) reports parallel same-origin capability specialists and
  consolidation:
  https://arxiv.org/abs/2606.30406
- Open-MOPD (August 2026) reports explicit token-share balancing during
  multi-specialist distillation:
  https://arxiv.org/abs/2608.19098

What remains undisclosed by these sources includes Metis-specific mode
mixtures, exact token budgets, and the transfer behavior of these recipes to a
small recursive MoE. Those values are contracts to measure, not claims of
published consensus.
