# Shared-cost utility routing (experimental, September 2026)

This is an opt-in research path, not a replacement for the running Wave-1
campaign. Existing defaults, exact-marginal controls, and production Praxis and
Logos remain unchanged. A feasible allocation or a fitted utility head is not
evidence that the model has improved.

**Causality finding, September 2026.** The original sequence-wide quota
rankings and this document's original batch-global price/repair allocator can
change earlier predictions when only future tokens change. A causal attention
mask does not make those routing decisions causal. The global allocator is an
offline allocation oracle, not a qualified autoregressive serving policy.
The separate `causal_compute_budget` path uses prefix-local admission instead.
Training through the global oracle is now rejected; full-model quality still
has to qualify the causal replacement. A new utility head alone does not
repair this issue.

**September 4, 2026 assessment: not qualified for a campaign restart.** Heads
fitted for 32, 256, and 1,024 updates improved the existing allocation in some
held-out comparisons, but did not establish superiority over uniform
depth-two/k-four execution. At the real six-by-4,096-token per-APU layout, the
learned allocation beat shuffled equal-work trajectories on several windows,
yet the uniform control remained at least as strong. This is evidence of some
allocation signal, not a successful replacement recipe.

The initial frozen backbone was the original Core step-5,000 checkpoint.
A later step-10,000 checkpoint with a separately fitted head showed the same
qualification failure on an independent full-layout window. These were
head-only adaptation experiments, not new 50B-token pretraining runs, and do
not establish RM quality or final convergence.
Three full-geometry FP8 optimizer steps also exercised actual five-pass
execution within the modeled cap; they establish execution feasibility, not
model quality or 80-APU throughput.

Aggregate artifacts and source-bound utility checkpoints are preserved under
`/lus/lustre1/vollmerc/more-credit-repair-20260904/`; the local experiment ledger
is `reports/more-credit-20260904/results.tsv`. Natural GPU routing variability
and forced-plan replay parity are recorded explicitly. Failed or
noise-unresolved comparisons must not be quoted as allocation improvements.

The final review found and corrected duplicated LM-head charges in the probe's
observation-mode nominal forward-cost reports. Historical raw reports remain
unchanged for provenance: their affected forward-cost totals must not be used
as measured cost evidence. Losses, controller-inclusive budget gates, teacher
token counts, and wall times are unaffected; the no-promotion decision stands.

## Historical offline oracle: budget and execution

The original joint allocator enables a separate outcome-prediction head.
The reference budget is the modeled training cost of the original model at
its reference mean depth and width (normally two passes and k=4), not two
independent quotas. Mean depth and mean width may trade against each other.

A mandatory first pass uses the reference k to construct a contextual state.
At each subsequent boundary, the router predicts the value of possible
remaining depths and per-pass/per-layer widths. A common cost price allocates
the remaining budget; only the next pass is executed before replanning from the
updated state, recurrent memory, state difference, and route history. Expert
identities are still selected from the current layer state.

The allocator has a hard integer cost cap. Its price search and primal repair
are not an exact solution to every multiple-choice knapsack problem; the
reported dual bound concerns the **predicted utility objective**, not actual
language-model quality. Discrete unused budget is reported rather than burned
on padding. Neither perfect allocation nor a quality advantage is assumed.

The ledger charges pass-dependent backbone/LM-head/memory cost, actual routed
expert invocations, and the utility head. Each possible future pass is priced
together with its required controller call; already executed controller work
is subtracted before replanning. It does not withhold controller work for
tokens that stop. The cost reference excludes the new head, so adding a
controller does not silently increase the comparison budget.
These are modeled training FLOPs, not a promise of identical wall-clock time.
Kernel launch overhead, packing, and small expert batches still matter.

The first-pass k is deliberately not optimized by this initial implementation.
Joint routing currently supports replicated experts without context
parallelism. The five-pass architectural cap can be used, but a checkpoint
trained only at depths 1--3 does not establish useful depth-4/5 behavior.

## Historical observed-improvement learning signal

The utility head predicts reductions in next-token cross entropy along a
future trajectory. Its depth and width predictions share the same loss units.
For each actually visited later exit, the target is the observed loss at the
earlier context minus the observed loss at that later exit. Width adjustments
are centered on the reference k, separating them from the depth intercept.

Unvisited continuations receive **no target**. In particular, stopping does
not create an imaginary continuation with zero prediction loss. The head does
not train to reproduce its own hard decisions, and hard allocation is not
presented as differentiable. Training-time utility perturbations explore
alternative feasible trajectories without relaxing the cost cap.

Context features and targets are detached from the backbone. Backbone training
uses the ordinary selected-exit LM loss; utility regression updates only the
new head, with independent gradient clipping. The original continuation and
k-policy heads are frozen in this mode. Residual/expert-identity learning is
not frozen.

Observed token-loss improvements are conditional observations, not an
unbiased estimate of every counterfactual or of cross-token externalities.
Exploration, held-out outcome prediction, actual equal-cost interventions,
and complete learning curves remain necessary. The additive trajectory value
model can also miss interactions between widths at different layers/passes.

## Causal training candidates

`--compute-allocation-mode causal` selects the prefix-invariant implementation.
It commits to a trajectory from first-pass causal context with a price fixed
before the sequence. Each sequence earns its own prefix credit; an earlier
token cannot borrow compute from later tokens or another sequence. There is
no sequence-wide ranking or current-batch price search. Exploration is keyed
to token position rather than batch-shaped random draws.

The reference remains fixed depth two and k four, but the old depth/width
histograms are not imposed. Unused credit is reported rather than hidden.
Neither a larger maximum depth nor a full budget is evidence that allocating
that work improves the model. Packing still removes inactive tokens from
later attention and SSM context; causal admission does not make pruning
equivalent to keeping all earlier states available.

The observed-improvement candidate retains losses at visited exits for its
critic targets. The separate `--terminal-action-critic` candidate instead
predicts the negative terminal CE of the admitted depth/width trajectory,
including a learned halt value. It labels only executed actions and prepays
one vocabulary projection per token; intermediate non-exiting tokens do not
need extra vocabulary projections. This objective has a different head shape
and checkpoint identity, not a compatible interpretation of an older fitted
improvement head. Both candidates charge their critic against the same cap;
neither obtains free policy-learning work.

`--compute-allocation-mode causal-fixed` is the lean fixed depth-two/k-four
reference, without a trained adaptive critic or unused intermediate readouts.
The `loop-fixed` and `loop-pathway-frozen` rows distinguish rerouted from
frozen expert identities. Execution settings and actual replay work must also
be matched or reported, rather than attributing every difference to routing.
The initial causal qualification lane is Core with recurrent-memory
contributions disabled; it is not an RM qualification.

## Interfaces

Training uses the existing ablation trainer and fresh output directories:

```bash
python -m metis_ablation.train \
  --row more-core \
  --output /path/to/new-campaign \
  --release-root /path/to/sealed-release \
  --compute-allocation-mode causal \
  --causal-compute-price 0 \
  --joint-max-passes 5 \
  --joint-router-exploration 0.05 \
  --joint-utility-coefficient 1.0 \
  --stop-after-steps 1000
```

Add `--terminal-action-critic --terminal-critic-exploration 0.05` only for
the separately identified terminal-value experiment. Use a new output root
for a different policy or objective.

The flags are recorded in curriculum/model identity. They must not be
introduced into an existing campaign by weakening its resume checks. Fixed
and random rows cannot silently become learned joint-routing rows.

`--diagnostic-apus N --stop-after-steps M` permits a bounded, explicitly
identified canary on spare ranks without truncating the scientific horizon.
It retains the row's micro-batch and derives accumulation to preserve the
480-sequence global batch exactly. An explicit `--diagnostic-micro-batch`
override must preserve that same global batch. The original lane's
throughput estimate is removed; a smaller-rank canary is not an 80-APU
throughput result.

For controlled probes, `force_routed_k` has shape
`[max_passes, physical_layers, batch, sequence]` and complements the existing
`force_depth` override. Active widths must be valid integers. These explicit
interventions bypass the learned width surrogate.

For end-to-end training comparisons, `--evaluation-gap-blocks 8` selects a
window inside an unsampled stride gap rather than a block that training will
eventually consume. Inputs and target lookahead are checked against the
preceding/next training blocks and the phase boundary. The window is tied to
the declared sampler identity and budget; changing that budget requires a new
holdout definition. This is token-disjoint evaluation, not a claim of
deduplicated document holdout. The default zero preserves the earlier
checkpoint-relative future-window probes, and training input enumeration is
unchanged.

With an enabled utility head, supervised calls using an explicit depth/width
trajectory can request `return_router_observations=True`. They expose utility
observations in memory and the `joint_utility` auxiliary loss. A diagnostic
trainer can freeze the backbone, fit the head, call `mark_trained()` after an
actual update, and evaluate on separate documents. All teacher/probe work must
be accounted for separately; it is not free pretraining.

Terminal-mode observations use `return_terminal_router_observations=True`
instead. The old observation request is rejected rather than silently mixing
the two objectives. Terminal mode also rejects a supervised request for final
logits that would introduce an unbudgeted second vocabulary projection.

Evaluation rejects a head with no recorded updates unless
`allow_untrained_joint_router=True` explicitly marks an untrained diagnostic.
An update counter is a guard against accidental use, not a quality certificate.

Joint telemetry includes:

- `joint_budget_flops`, `joint_model_flops`, and `joint_router_flops`;
- `joint_unused_budget_flops` and `joint_budget_enforced`;
- `joint_utility_observations` and `joint_utility_upper_gap`.

Actual vocabulary work is separate: `lm_head_forward_rows` and
`lm_head_forward_flops` include real projections, synchronized dummy rows, and
any explicitly requested extra logits. `lm_head_recompute_rows` and
`lm_head_recompute_flops` count internal head-checkpoint replay. Replay
counters are live until backward completes: snapshot them after each
microbatch's backward, not immediately after forward. They remain zero in
evaluation without backward. These are logical projection FLOPs, not padded
GPU instruction counts. Do not add the same head replay again through a
blanket full-model recomputation multiplier.

Teacher interventions can exceed the reference cap and report that fact with
`joint_budget_enforced=0`; they must never be reported as budget-matched model
runs. Aggregate global costs in the trainer include all accumulation steps and
ranks. Existing jobs/checkpoints are preserved until real-data evidence
supports a separately identified restart.

## Related evidence, not validation of this design

- [MUTO, ACL July 2026](https://aclanthology.org/2026.acl-long.1386/) uses
  observed ground-truth likelihood changes to assess the utility of generated
  reasoning prefixes. It motivates outcome-based targets, but does not
  establish a shared depth/expert-width controller.
- [POISE, May 2026](https://arxiv.org/html/2605.07579v2) studies internal-state
  value prediction and policy-gradient baselines. Its independence and
  cross-rollout considerations reinforce that detaching a target does not
  automatically make a statistical estimator unbiased.

The single-price, receding-horizon combination here is an experimental design.
Neither source proves its allocation optimality, convergence, or superiority.

[Mixture-of-Recursions, October 2025 revision,
Appendix A.2](https://arxiv.org/html/2507.10524v3#A1.SS2) explicitly distinguishes
future-aware training-time top-k routing from causal per-token deployment
rules. Its attention masking and auxiliary BCE do not by themselves make a
sequence-wide ranking causal. This is relevant to the reproduced prefix
violation, not evidence that the current implementation already satisfies a
causal deployment contract.
