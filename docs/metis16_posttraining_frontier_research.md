# Metis-1.6 post-training frontier review

Date: September 2026
Decision scope: Metis-1.6 only

## Executive decision

Metis-1.6 will use three public response modes—`direct`, `think`, and
`think_max`—inside every broad SFT and capability-RL stage. It will not train
one specialist per reasoning mode. It will not condition MoRE depth or routed k
on the mode token.

The final recipe is:

```text
single-jump context CPT
-> cold-start mode-aware SFT
-> broad mode-aware SFT
-> shared hybrid-mode GSPO
-> parallel capability GSPO specialists
-> same-tokenizer OPD
-> evaluation and local release gate
```

Cross-tokenizer DeepSeek preference distillation is cut. A standalone pairwise
reward model is also cut.

## What the 2026 frontier actually supports

### Explicit reasoning modes are installed before RL

A.X K1 (January 2026) trains overlapping prompts in multiple reasoning modes
during SFT and then reinforces mode behavior with paired GSPO and format
rewards. This is the closest disclosed precedent for teaching reliable mode
switching rather than hoping an inference-time instruction creates it:
https://arxiv.org/abs/2601.09200

OpenBMB's UltraData-SFT-2605 (May 2026) publishes explicit thinking and
non-thinking partitions across domains. The dataset demonstrates the practical
answer to incompatible upstream naming: preserve raw data, then publish a
derived mode label and rendered template in the training layer:
https://huggingface.co/datasets/openbmb/UltraData-SFT-2605

MiniCPM5 (May 2026) describes deep SFT followed by hybrid SFT before
capability-specialist reinforcement learning. That supports establishing
reasoning competence first and broad mode behavior second:
https://github.com/OpenBMB/MiniCPM

Conclusion: SFT owns the initial response grammar and competent examples for
all modes. RL does not replace that work; it makes the policy obey the mode
under sampling and optimizes verified outcomes.

### Hybrid mode is a conditioning axis, not a specialist taxonomy

Nanbeige4.2-3B (July 2026) reports hybrid-mode RL followed by a reasoning-focused
phase with explicit length control:
https://arxiv.org/abs/2607.22083

MOPD (June 2026) and Open-MOPD (August 2026) organize parallel experts around
capabilities and consolidate them into a same-origin student. Open-MOPD also
makes token-share balance explicit:
https://arxiv.org/abs/2606.30406
https://arxiv.org/abs/2608.19098

Conclusion: Metis branches reasoning, code, knowledge, writing, and agentic
capabilities. Every branch contains `direct`, `think`, and `think_max`.
Creating three additional 'mode specialists' would conflate task capability
with response policy and make consolidation harder to interpret.

### Code efficiency must be correctness-gated

Nanbeige4.1-3B (February 2026) applies code correctness reinforcement before
efficiency shaping and gates efficiency on verified correctness:
https://arxiv.org/abs/2602.13367

Conclusion: the code specialist trains all modes on executable correctness.
Only verified-correct candidates receive an efficiency term. `think_max`
never receives a positive reward merely for producing more tokens.

### Same-origin consolidation is the relevant distillation path

MiniCPM5 (May 2026), MOPD (June 2026), and Open-MOPD (August 2026) support
parallel capability policies followed by same-origin consolidation. These
teachers share the student's tokenizer, architecture, and control-token
semantics.

Conclusion: OPD is retained because it avoids the exact alignment problem that
makes an external DeepSeek DPD stage unattractive. It consolidates capability,
while copying the prompt's reasoning mode unchanged.

## Mode contracts

### `direct`

- no visible reasoning trace;
- compact final answer;
- full task correctness and safety requirements;
- no length reward;
- not a lower-MoRE-compute mode.

### `think`

- normal visible reasoning;
- difficulty-aware efficiency shaping;
- correctness dominates reward;
- average reasoning length is optimized, not maximized.

### `think_max`

- largest useful reasoning budget;
- correctness-only expansion;
- no positive reward for length or verbosity;
- evaluated for incremental hard-task gain over `think`;
- not a higher-MoRE-depth mode.

All three modes target mean MoRE depth 2.0 and mean routed k 4.0. This invariant
is a conservative Metis-1.6 choice: MoRE is itself new in this generation, so
changing both response policy and recursive compute by mode would confound the
first measurement.

## How standardized datasets become Metis data

Upstream field names are irrelevant. The data compiler creates a versioned
Metis view while preserving source provenance.

For every example it emits:

- immutable source and revision identifiers;
- base-prompt fingerprint;
- rendered prompt with one control token;
- `reasoning_mode` enum;
- response and trace-boundary metadata;
- domain and capability tags;
- target-token count;
- transformation version;
- decontamination and quality-audit results.

Mode assignment must be deliberate:

1. Existing trace-free, self-contained answers can become `direct`.
2. High-quality concise reasoning traces can become `think`.
3. Verified hard-task traces that use additional search, checking, or
   self-correction can become `think_max`.
4. Ambiguous examples are quarantined or regenerated; they are not guessed
   into a mode.
5. A controlled prompt subset is regenerated in all three modes so mode
   switching can be measured on identical tasks.

For SFT, balance is audited twice: prompt share and target-token share. This is
necessary because a 15% `think_max` prompt share can dominate tokens.

For GSPO, one task may be rolled out under each requested mode. Each row still
contains a 16-sample on-policy group for GSPO. An overlap ID connects the three
mode rows to the same base prompt. Rewards combine correctness with strict
mode compliance.

## Shared GSPO versus capability GSPO

The shared hybrid-mode GSPO pass is small and behavioral. It stabilizes:

- control-token obedience;
- direct-mode trace suppression;
- valid trace boundaries;
- termination and repetition;
- switching between normal and maximum reasoning.

Capability specialists then reinforce domain outcomes. Their public mode mixes
differ because domains genuinely need different response budgets, but no mode
is absent. The mixture is measured on target tokens as well as prompts.

## Reward strategy

Consensus supported by the cited 2026 systems:

- use verifiers where correctness is executable or exact;
- use sequence-level GSPO for the MoE policy;
- filter uninformative on-policy groups;
- mask truncated samples;
- make length shaping correctness-dominant;
- keep capability branches independent before consolidation.

Contested or model-specific:

- exact GSPO clip ranges;
- strict 10-90% filtering at Metis scale;
- optimum `think_max` prompt and token share;
- transfer of large-model hybrid-mode recipes to Metis's small active
  parameter count;
- benefit of a shared hybrid GSPO pass before specialist RL.

These are live canary and evaluation questions, not facts to assume.

## Practices rejected for Metis-1.6

- Cross-tokenizer token/logit distillation from DeepSeek.
- Treating `direct` as 'no compute' or `think_max` as 'deeper MoRE'.
- Three mode-only RL specialists.
- Assigning every dataset a mode by a brittle heuristic with no quarantine.
- Balancing only examples while ignoring target-token volume.
- Rewarding `think_max` for verbosity.
- A naive global length penalty.
- A terminal scalar-RM stage that can soften verified capability gains.
- Consolidation that drops or rewrites the prompt's mode token.

## Unknowns that must stay labeled unknown

The cited labs do not publish Metis's exact three-mode mixture, token budgets,
mode-compliance weights, or MoRE behavior. They also do not establish that
`think_max` should improve every task; the release gate should require gains
on a declared hard-task slice and allow no material regression elsewhere.

The safest frontier posture is therefore an explicit, auditable contract plus
small live canaries—not claiming that borrowed ratios are universal.
