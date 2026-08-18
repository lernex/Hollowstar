from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from metis_ablation.analysis import RoutingAnalyzer
from metis_ablation.sampler import AblationSampleStream
from metis_ablation.specs import (
    ABLATION_LADDER,
    GLOBAL_BATCH_SEQUENCES,
    GLOBAL_BATCH_TOKENS,
    AblationSpec,
    dense_control_report,
    spec_by_name,
    validate_allocation,
)
from metis_ablation.train import AblationSchedule
from metis_mamba.optim import (
    MuonAdamWHybrid,
    _dequantize_blockwise_int8,
    _quantize_blockwise_int8,
    _zeropower_via_newton_schulz5,
)
from metis_training.data import PHASE_STARTS, PHASE_TOKENS
from metis_training.metrics import estimate_hardware_flops
from metis_training.model import (
    BudgetController,
    CurriculumState,
    Metis16ForCausalLM,
    _memory_attention_combine,
    _memory_attention_scores,
    _stream_gate_logits,
    expert_segment_plan,
    packed_expert_rows,
    geometric_continue_probability,
    max_entropy_categorical,
)
from metis_training.model_config import Metis16Config, load_family_config
from metis_training.precision import bucketed_row_count


# --------------------------------------------------------------------------
# production must not move


@pytest.mark.parametrize(
    ("family", "stored", "active"),
    [
        ("praxis", 3_545_482_071, 470_844_375),
        ("logos", 12_000_001_676, 1_189_454_220),
    ],
)
def test_production_audits_survive_the_ablation_family(family, stored, active):
    audit = load_family_config(family).logical_parameter_audit()
    assert audit.stored_total == stored
    assert audit.active_per_pass_mean == active


def test_production_families_still_reject_relaxed_settings():
    config = load_family_config("praxis")
    with pytest.raises(ValueError, match="dense feed-forward"):
        replace(config, ffn_mode="dense").validate()
    with pytest.raises(ValueError, match="five passes"):
        replace(config, max_passes=3).validate()
    with pytest.raises(ValueError, match="four persistent mHC streams"):
        replace(config, n_streams=1).validate()


def test_expert_execution_defaults_to_the_production_loop():
    assert load_family_config("praxis").expert_execution == "loop"
    assert load_family_config("logos").expert_execution == "loop"


def test_ablation_fp8_uses_blockwise_scaling_without_moving_production():
    ablation = spec_by_name("more-core").model_config(
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    )
    assert ablation.precision.fp8_scaling == "blockwise"
    assert "mhc_controller" not in ablation.precision.fp8_roles
    assert "mhc_controller" in ablation.precision.bf16_roles
    assert load_family_config("praxis").precision.fp8_scaling == "delayed"
    assert load_family_config("logos").precision.fp8_scaling == "delayed"


# --------------------------------------------------------------------------
# ladder invariants


def test_every_row_consumes_an_identical_global_batch():
    for spec in ABLATION_LADDER:
        assert spec.apus * spec.micro_batch * spec.grad_accum == GLOBAL_BATCH_SEQUENCES


def test_more_core_uses_the_measured_eight_sequence_micro_batch():
    primary = spec_by_name("more-core")
    assert (primary.apus, primary.micro_batch, primary.grad_accum) == (28, 8, 2)
    for name in ("more-core-xs", "more-core-xxs", "more-core-seed2"):
        spec = spec_by_name(name)
        assert spec.micro_batch == 8
        assert spec.apus * spec.micro_batch * spec.grad_accum == GLOBAL_BATCH_SEQUENCES


def test_primary_proxy_is_the_parameter_matched_shallow_recurrent_block():
    config = spec_by_name("more-core").model_config(
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    )
    assert config.d_model == 4096
    assert config.n_layers == 2
    assert config.attention_indices == (1,)
    assert config.latent_dim == 2048
    assert config.n_routed_experts == 72
    assert config.expert_intermediate_dim == 1152
    assert config.budgeted_depth_values == (1, 2, 3)
    assert config.ngram_memory.injection_layers == (0, 1)
    assert all(
        0 <= layer < config.n_layers
        for layer in config.ngram_memory.injection_layers
    )
    with pytest.raises(ValueError, match="N-gram injection layer"):
        replace(
            config,
            ngram_memory=replace(
                config.ngram_memory,
                injection_layers=(config.n_layers,),
            ),
        ).validate()
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(config, budgeted_depth_values=(1, 3, 2)).validate()
    with pytest.raises(ValueError, match="inside budgeted_depth_values"):
        replace(config, budgeted_depth_values=(3, 4)).validate()


def test_rank_counts_are_whole_nodes_and_fit_the_allocation():
    report = validate_allocation()
    assert report["rows"] == 13
    assert report["spare_apus"] >= 4
    for spec in ABLATION_LADDER:
        assert spec.apus % 4 == 0, f"{spec.name} is not a whole number of nodes"


def test_iso_flop_rows_really_are_iso_flop():
    """Rows 1 and 5-13 must land within a percent of each other, or the paper's
    matched-compute claim is decoration."""

    costs = {}
    for spec in ABLATION_LADDER:
        if not spec.iso_flop:
            continue
        config = spec.model_config(
            mhc_backend="torch_reference",
            mamba_backend="torch_reference",
            attention_backend="torch_reference",
        )
        depth = 1.0 if spec.continuation_mode == "depth_one" else 2.0
        width = (
            float(spec.fixed_routed_k) if spec.routed_k_mode == "fixed" else 4.0
        )
        costs[spec.name] = estimate_hardware_flops(
            config, tokens=1, observed_mean_passes=depth, observed_mean_routed_k=width
        )
    assert len(costs) >= 10
    low, high = min(costs.values()), max(costs.values())
    assert (high - low) / high < 0.01, costs


def test_pathway_rows_are_exactly_matched():
    """Rows 5 and 6 differ only in pathway, so their cost must be identical."""

    def cost(name: str) -> float:
        spec = spec_by_name(name)
        config = spec.model_config(
            mhc_backend="torch_reference",
            mamba_backend="torch_reference",
            attention_backend="torch_reference",
        )
        return estimate_hardware_flops(
            config, tokens=1, observed_mean_passes=2.0, observed_mean_routed_k=4.0
        )

    assert cost("loop-fixed") == cost("loop-pathway-frozen")


def test_learned_rows_use_exact_budgeted_depth_and_width():
    assert spec_by_name("mor-dense-ffn").continuation_mode == "budgeted"
    assert spec_by_name("mor-fixed-k").continuation_mode == "budgeted"
    assert spec_by_name("fixed-depth-adaptive-k").routed_k_mode == "budgeted"
    assert spec_by_name("more-core").continuation_mode == "budgeted"
    assert spec_by_name("more-core").routed_k_mode == "budgeted"
    assert spec_by_name("more-rm").continuation_mode == "budgeted"
    assert spec_by_name("more-rm").routed_k_mode == "budgeted"
    assert spec_by_name("random-k").continuation_mode == "budgeted"
    assert spec_by_name("random-depth").routed_k_mode == "budgeted"


def test_dense_controls_are_matched_to_their_stated_objective():
    report = dense_control_report()
    reference = report["more_core"]
    stored_error = abs(
        report["dense_param_matched"]["stored_total"] - reference["stored_total"]
    ) / reference["stored_total"]
    flop_error = abs(
        report["dense_flop_matched"]["flops_per_token"] - reference["flops_per_token"]
    ) / reference["flops_per_token"]
    assert stored_error < 0.01, report["dense_param_matched"]
    assert flop_error < 0.01, report["dense_flop_matched"]


def test_dense_rows_report_all_parameters_active():
    spec = spec_by_name("dense-flop-matched")
    audit = spec.model_config(
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    ).logical_parameter_audit()
    assert audit.routed_experts == 0
    assert audit.active_per_pass_min == audit.active_per_pass_max


def test_global_batch_mismatch_is_rejected_at_construction():
    with pytest.raises(ValueError, match="identical token sets"):
        AblationSpec(
            index=99,
            name="broken",
            title="broken",
            isolates="nothing",
            apus=28,
            micro_batch=4,
            grad_accum=3,
        )


# --------------------------------------------------------------------------
# random-policy controls


def test_max_entropy_categorical_hits_its_mean():
    weights = max_entropy_categorical(range(1, 9), 4.0)
    assert pytest.approx(sum(weights), abs=1e-9) == 1.0
    mean = sum(w * v for w, v in zip(weights, range(1, 9)))
    assert pytest.approx(mean, abs=1e-6) == 4.0
    # Maximum entropy subject to a mean below the midpoint tilts down, but must
    # never collapse onto a single value: that would be a fixed-k row wearing a
    # random-k label.
    assert min(weights) > 0.0


def test_geometric_continue_probability_hits_its_mean_depth():
    for target in (1.5, 2.0, 3.0, 4.5):
        p = geometric_continue_probability(5, target)
        realized = (1.0 - p**5) / (1.0 - p) if p < 1.0 else 5.0
        assert pytest.approx(realized, abs=1e-6) == target


# --------------------------------------------------------------------------
# sampler


class _FakeStream:
    sequence_length = 4_096

    def __init__(self):
        self.requests: list[tuple[int, int, int, int]] = []

    def batch(self, *, global_token_cursor, rank, world_size, micro_batch_size):
        self.requests.append((global_token_cursor, rank, world_size, micro_batch_size))
        return None


def _sampler(budget=50_000_000_000):
    return AblationSampleStream(
        _FakeStream(),
        budget_tokens=budget,
        block_tokens=GLOBAL_BATCH_TOKENS,
        phase_starts=dict(PHASE_STARTS),
        phase_tokens=dict(PHASE_TOKENS),
    )


def test_sampler_preserves_release_phase_proportions():
    sampler = _sampler()
    fractions = [plan.sampled_fraction for plan in sampler.plans]
    # The property being tested is that every phase is thinned by the *same*
    # fraction, so the ablation corpus keeps the release's phase mix. The value
    # of that fraction is budget over corpus, and the corpus moved from the
    # aspirational 1T to the 804.8B that was actually built; pinning 0.05 here
    # tested the old corpus size rather than the sampler.
    expected = sampler.budget_tokens / sum(PHASE_TOKENS.values())
    for fraction in fractions:
        assert pytest.approx(fraction, abs=1e-4) == expected
    assert sampler.dropped_tokens() >= 0
    assert sampler.sampled_tokens <= sampler.budget_tokens


def test_sampler_never_crosses_a_phase_boundary():
    sampler = _sampler()
    boundaries = sorted(PHASE_STARTS.values())[1:]
    for step in range(0, sampler.total_blocks, 97):
        cursor = sampler.release_cursor(step)
        end = cursor + sampler.block_tokens
        for boundary in boundaries:
            assert not (cursor < boundary < end), (step, cursor, boundary)


def test_identical_token_windows_across_different_world_sizes():
    """The heart of the comparability claim: a 16-APU row and a 56-APU row must
    read the same tokens for the same optimizer step."""

    windows = {}
    for apus, micro, accum in ((16, 4, 7), (28, 4, 4), (56, 2, 4)):
        sampler = _sampler()
        covered: set[int] = set()
        for rank in range(apus):
            list(
                sampler.micro_batches(
                    step=11,
                    rank=rank,
                    world_size=apus,
                    micro_batch_size=micro,
                    grad_accum=accum,
                )
            )
        for cursor, rank, world, micro_size in sampler.stream.requests:
            span = micro_size * sampler.stream.sequence_length
            base = cursor + rank * span
            covered.update(range(base, base + span, sampler.stream.sequence_length))
        windows[apus] = covered
    reference = windows[16]
    assert len(reference) == GLOBAL_BATCH_SEQUENCES
    for apus, covered in windows.items():
        assert covered == reference, f"world size {apus} read a different window"


def test_sampler_rejects_a_batch_that_does_not_tile_the_block():
    sampler = _sampler()
    with pytest.raises(ValueError, match="identical global batch"):
        list(
            sampler.micro_batches(
                step=0, rank=0, world_size=28, micro_batch_size=4, grad_accum=3
            )
        )


# --------------------------------------------------------------------------
# model paths, exercised end to end on the tiny config


def _tiny(**overrides):
    config = Metis16Config.tiny_for_tests()
    if overrides:
        config = replace(config, **overrides)
        config._validate_tiny() if config.family == "tiny" else config.validate()
    return config


def _curriculum(**overrides):
    """Tiny's routed-k ceiling is 3, so every curriculum here must stay inside it."""

    overrides.setdefault("fixed_routed_k", 2)
    overrides.setdefault("stochastic_routing", False)
    return CurriculumState(**overrides)


def _tiny_batch(config, *, batch=2, length=8):
    generator = torch.Generator().manual_seed(20_260_725)
    input_ids = torch.randint(0, config.vocab_size, (batch, length), generator=generator)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    return input_ids, labels


def test_grouped_expert_execution_matches_the_loop_exactly():
    """The grouped path is a scheduling change, not a numerical one."""

    torch.manual_seed(7)
    loop_config = _tiny()
    grouped_config = replace(loop_config, expert_execution="grouped")
    grouped_config._validate_tiny()

    loop_model = Metis16ForCausalLM(loop_config)
    grouped_model = Metis16ForCausalLM(grouped_config)
    grouped_model.load_state_dict(loop_model.state_dict())
    loop_model.eval()
    grouped_model.eval()

    input_ids, labels = _tiny_batch(loop_config)
    curriculum = _curriculum()
    with torch.no_grad():
        left = loop_model(input_ids, labels, curriculum=curriculum)
        right = grouped_model(input_ids, labels, curriculum=curriculum)
    torch.testing.assert_close(left.loss, right.loss, rtol=1e-5, atol=1e-6)


def test_the_halting_gate_keeps_a_gradient_when_it_saturates():
    """A gate the budget cannot move is a budget that does not exist.

    More depth always lowers the loss, so the continuation logit runs away
    within a few steps. Unbounded, the sigmoid then saturates and its
    derivative goes to zero -- and a Lagrange multiplier scales a gradient, so
    once that gradient is zero no multiplier at any size can bring the policy
    back. That is the whole explanation for the campaign's asymmetry: the width
    policy is a softmax over eight choices and converges on its target from
    7.35 to 3.45 in fifty steps, while the depth policy pins at the maximum by
    step ten with the identical controller attached.

    So the gate must still respond when it is pushed hard. This drives it far
    past saturation and asserts that the derivative survives.
    """

    torch.manual_seed(3)
    model = Metis16ForCausalLM(_tiny())
    controller = model.continuation
    width = controller.hidden.in_features
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(6, width, generator=generator)
    pieces = torch.split(
        features,
        [
            model.config.d_model,
            model.config.memory_dim,
            model.config.d_model,
            width - 2 * model.config.d_model - model.config.memory_dim,
        ],
        dim=-1,
    )
    with torch.no_grad():
        controller.output.bias.fill_(60.0)
    probability = controller(*pieces)
    assert float(probability.mean().detach()) > 0.9, "the gate should be saturated here"
    gradient = torch.autograd.grad(probability.sum(), controller.output.bias)[0]
    assert float(gradient.abs().sum()) > 1e-4, (
        "the saturated gate has no gradient left for a budget to act on"
    )


def test_budget_controller_binds_against_a_loss_that_rewards_more_compute():
    """More depth always lowers the loss, so only a binding budget stops it.

    This is the canary's failure in miniature. Depth helps, so an unconstrained
    policy climbs to the ceiling -- measured, from its intended mean of 1.86 to
    the 5.0 maximum within nine steps, with halt_collapse reaching 1.000, which
    is to say no token ever stopped. The fixed coefficient could not hold it:
    at 1e-2 against a task term of any real size the equilibrium is far past
    the target.

    The point of the constraint is not that depth is bad. It is that rows 5, 8,
    9 and 10 of the campaign are supposed to execute identical FLOPs per token,
    so that the comparison is about *where* compute goes rather than how much.
    A multiplier that finds its own strength restores that.
    """

    def settled_mean(rate, coefficient, steps=4000, reward=5.0):
        controller = BudgetController(
            2.0, coefficient=coefficient, rate=rate, limit=1.0e3
        )
        controller.train()
        value = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([value], lr=0.01)
        history = []
        for _ in range(steps):
            optimizer.zero_grad()
            # A task that always prefers more compute, exactly as the real one
            # does, plus the budget that is supposed to stand against it.
            (controller.penalty(value) - reward * value).backward()
            optimizer.step()
            history.append(float(value.detach()))
        return sum(history[-500:]) / 500

    controlled = settled_mean(0.05, 1.0)
    fixed = settled_mean(0.0, 0.01)
    assert abs(controlled - 2.0) < 0.25, controlled
    assert fixed > 20.0, f"the fixed coefficient bound after all: {fixed}"


def test_the_budget_multiplier_does_not_wind_up_while_the_policy_is_pinned():
    """A stored-up multiplier arrives all at once, and that is an oscillation.

    Measured on MoRE-Core, with no leak: depth pinned at the 5.00 ceiling from
    step 20 to step 60 while the multiplier accumulated against an error it
    could not act on, broke through to 2.01 at step 70, and was driven to 1.00 --
    the floor -- by step 80, with the width policy rebounding from 2.6 to 7.5 to
    compensate. Neither end of that is a policy.

    So the multiplier must forget. This holds a controller against a constant
    error it cannot fix, as a saturated policy does, and asserts what it stores
    stays bounded.
    """

    controller = BudgetController(
        2.0, coefficient=1.0, rate=20.0, limit=1.0e9, leak=0.02
    )
    controller.train()
    pinned = torch.tensor(5.0)  # a policy stuck at the ceiling
    for _ in range(2000):
        controller.penalty(pinned)
    stored = abs(float(controller.multiplier))
    # rate * error / leak is where a leaky integrator settles.
    assert stored < 1.5 * (20.0 * 3.0 / 0.02), stored

    unleaked = BudgetController(
        2.0, coefficient=1.0, rate=20.0, limit=1.0e9, leak=0.0
    )
    unleaked.train()
    for _ in range(2000):
        unleaked.penalty(pinned)
    assert abs(float(unleaked.multiplier)) > 10 * stored, "the leak changed nothing"


def test_budget_controller_pushes_back_up_when_the_policy_undershoots():
    """An equality constraint, not a squeeze towards always-shallow.

    The worry a mean budget deserves is that it teaches the model never to go
    deep. It cannot: the multiplier is signed, so a policy that undershoots its
    target drives the multiplier down through zero and is pushed back up. The
    budget fixes how much compute is spent, never where.
    """

    controller = BudgetController(2.0, coefficient=1.0, rate=0.05, limit=1.0e3)
    controller.train()
    value = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([value], lr=0.01)
    history = []
    for _ in range(4000):
        optimizer.zero_grad()
        # A task that prefers *less* compute, the opposite failure.
        (controller.penalty(value) + 5.0 * value).backward()
        optimizer.step()
        history.append(float(value.detach()))
    assert float(controller.multiplier) < 0.0, "the multiplier never turned around"
    assert abs(sum(history[-500:]) / 500 - 2.0) < 0.25, sum(history[-500:]) / 500


def test_budget_controller_is_off_unless_a_rate_is_declared():
    """Production Praxis and Logos must see the penalty they already saw."""

    controller = BudgetController(2.0, coefficient=0.01, rate=0.0, limit=1e3)
    controller.train()
    realized = torch.tensor(5.0)
    torch.testing.assert_close(
        controller.penalty(realized), (realized - 2.0).square() * 0.01
    )
    assert float(controller.multiplier) == 0.0, "an inactive controller must not drift"


def test_replicated_dispatch_matches_the_general_path():
    """The fast path is the general path with the no-ops removed, exactly.

    With one member in the expert group the sort is over a constant key, the
    count exchange is a collective with itself, and the all-to-alls move data
    from a rank to itself. Removing them is only safe if the result is
    identical, and the general path is still live code for production expert
    parallelism -- so this pins one to the other rather than trusting the
    argument.
    """

    torch.manual_seed(31)
    model = Metis16ForCausalLM(replace(_tiny(), expert_execution="grouped_gemm"))
    moe = model.layers[0].moe
    assert moe.world_size == 1

    experts = moe.config.n_routed_experts
    tokens, latent = 12, moe.config.latent_dim
    generator = torch.Generator().manual_seed(4)
    hidden = torch.randn(tokens * 2, latent, generator=generator)
    indices = torch.randint(0, experts, (tokens * 2,), generator=generator)
    weights = torch.rand(tokens * 2, generator=generator)
    token_indices = torch.randint(0, tokens, (tokens * 2,), generator=generator)

    with torch.no_grad():
        fast = moe._dispatch_replicated(hidden, indices, weights, token_indices, tokens)
        general = moe._dispatch_general(hidden, indices, weights, token_indices, tokens)
    assert torch.equal(fast[0], general[0]), "combined expert output moved"
    assert torch.equal(fast[1], general[1]), "processed-assignment count moved"


def test_ep_drop_accounting_uses_returned_source_assignments(
    monkeypatch: pytest.MonkeyPatch,
):
    """Receive-side EP load is not a source-side drop count."""

    import metis_training.model as model_module

    model = Metis16ForCausalLM(_tiny())
    moe = model.layers[0].moe
    moe.world_size = 4
    monkeypatch.setattr(model_module.dist, "all_reduce", lambda *args, **kwargs: None)

    def imbalanced_dispatch(
        hidden_states,
        expert_indices,
        weights,
        token_indices,
        token_count,
    ):
        del expert_indices, weights, token_indices
        return (
            hidden_states.new_zeros(token_count, hidden_states.shape[-1]),
            torch.tensor(1, device=hidden_states.device),
            torch.zeros((), device=hidden_states.device, dtype=torch.long),
            torch.zeros((), device=hidden_states.device),
        )

    monkeypatch.setattr(moe, "_dispatch", imbalanced_dispatch)
    batch, length = 2, 6
    hidden = torch.randn(batch, length, moe.config.d_model)
    route_features = torch.randn(batch, length, moe.config.route_feature_dim)
    active = torch.ones(batch, length, dtype=torch.bool)
    _output, state = moe(
        hidden,
        route_features=route_features,
        active_mask=active,
        curriculum=_curriculum(),
    )
    assert int(state.assignments) > 1
    assert torch.equal(state.processed_assignments, state.assignments)


def test_packed_buffer_size_does_not_depend_on_where_the_routing_went():
    """The packed buffer must be sized from shapes, not from the counts.

    Reading the segment lengths back to the host to size the buffer is a
    pipeline stall per layer per pass -- two hundred of them per micro-step,
    measured at 10.8 s of wall clock. Sizing for the worst case removes the
    read entirely, which is only sound if the worst case really is a bound and
    really is independent of the distribution.
    """

    generator = torch.Generator().manual_seed(20_260_817)
    for experts, rows, multiple in ((7, 61, 16), (96, 4096, 16), (5, 0, 8), (3, 7, 1)):
        expected = packed_expert_rows(rows, experts, multiple)
        for trial in range(6):
            if trial == 0:  # everything to one expert, the worst imbalance
                indices = torch.zeros(rows, dtype=torch.long)
            else:
                indices = torch.randint(0, experts, (rows,), generator=generator)
            slots, padded = expert_segment_plan(indices, experts, multiple)
            assert int(padded.sum()) == expected, "buffer size moved with the routing"
            assert bool((padded >= torch.bincount(indices, minlength=experts)).all())
            assert int(torch.unique(slots).numel()) == rows
            assert bool((slots < expected).all())


def test_grouped_gemm_expert_execution_matches_the_loop_exactly():
    """Contracting the bank in one GEMM must not move a single number.

    The loop path applies expert ``i`` to its assigned rows in ascending
    original order; the grouped-GEMM path packs those same rows, in that same
    order, into expert ``i``'s segment. So this is an exact equality, not a
    tolerance -- anything looser would pass while rows were being routed to the
    wrong expert.
    """

    torch.manual_seed(7)
    loop_config = _tiny()
    grouped_config = replace(loop_config, expert_execution="grouped_gemm")
    grouped_config._validate_tiny()

    loop_model = Metis16ForCausalLM(loop_config)
    grouped_model = Metis16ForCausalLM(grouped_config)
    _copy_expert_bank(loop_model, grouped_model)
    loop_model.eval()
    grouped_model.eval()

    input_ids, labels = _tiny_batch(loop_config)
    curriculum = _curriculum()
    with torch.no_grad():
        left = loop_model(input_ids, labels, curriculum=curriculum)
        right = grouped_model(input_ids, labels, curriculum=curriculum)
    assert torch.equal(left.loss, right.loss)


def test_grouped_gemm_seeds_each_expert_the_same_way_the_loop_does():
    """Expert ``i`` is the same expert in both layouts, from the same seed.

    The per-expert seeding is what makes a rank's expert weights a function of
    the *global* expert identity. If the grouped bank consumed its random
    stream in a different order the two layouts would train different models
    from the same manifest, and nothing downstream would notice.
    """

    torch.manual_seed(7)
    loop_model = Metis16ForCausalLM(_tiny())
    torch.manual_seed(7)
    grouped_model = Metis16ForCausalLM(
        replace(_tiny(), expert_execution="grouped_gemm")
    )
    loop_moe = loop_model.layers[0].moe
    grouped_bank = grouped_model.layers[0].moe.local_experts
    for index, expert in enumerate(loop_moe.local_experts):
        assert torch.equal(
            expert.gate_up.weight, grouped_bank.gate_up.weight[index]
        )
        assert torch.equal(expert.down.weight, grouped_bank.down.weight[index])


def test_grouped_gemm_backward_matches_the_loop():
    """Gradients too -- a forward-only check would miss a wrong scatter.

    The grouped contraction can accumulate products in a different order from
    the expert loop, so pin it at near-roundoff rather than requiring bitwise
    identity. The exact assignment test above still catches a moved row.
    """

    torch.manual_seed(7)
    loop_model = Metis16ForCausalLM(_tiny())
    grouped_model = Metis16ForCausalLM(
        replace(_tiny(), expert_execution="grouped_gemm")
    )
    _copy_expert_bank(loop_model, grouped_model)
    input_ids, labels = _tiny_batch(loop_model.config)
    curriculum = _curriculum()

    loop_output = loop_model(input_ids, labels, curriculum=curriculum)
    (loop_output.loss + loop_output.auxiliary_loss).backward()
    grouped_output = grouped_model(input_ids, labels, curriculum=curriculum)
    (grouped_output.loss + grouped_output.auxiliary_loss).backward()

    loop_moe = loop_model.layers[0].moe
    grouped_bank = grouped_model.layers[0].moe.local_experts
    for index, expert in enumerate(loop_moe.local_experts):
        torch.testing.assert_close(
            expert.gate_up.weight.grad,
            grouped_bank.gate_up.weight.grad[index],
            rtol=2.0e-4,
            atol=2.0e-9,
        )
    torch.testing.assert_close(
        loop_model.embedding.weight.grad,
        grouped_model.embedding.weight.grad,
        rtol=1e-6,
        atol=1e-7,
    )


def test_expert_segment_plan_covers_every_assignment_exactly_once():
    """No assignment dropped, none written twice, none in a neighbour's segment.

    A packing bug here is the archetypal silent failure: every GEMM still runs,
    every shape still checks, and a slice of the batch is quietly computed by
    the wrong expert or not at all.
    """

    generator = torch.Generator().manual_seed(20_260_816)
    for expert_count, rows, multiple in (
        (7, 61, 1),
        (7, 61, 16),
        (96, 1024, 16),
        (4, 0, 16),
        (5, 9, 8),
    ):
        indices = torch.randint(0, expert_count, (rows,), generator=generator)
        slots, padded = expert_segment_plan(indices, expert_count, multiple)

        assert int(padded.numel()) == expert_count
        assert torch.equal(padded % multiple, torch.zeros_like(padded))
        assert bool((padded >= multiple).all()), "every expert must own a block"
        counts = torch.bincount(indices, minlength=expert_count)
        assert bool((padded >= counts).all()), "a segment lost an assignment"

        assert int(slots.numel()) == rows
        assert int(torch.unique(slots).numel()) == rows, "two rows share a slot"
        assert bool((slots >= 0).all()) and bool((slots < int(padded.sum())).all())

        # The slot each row lands in must belong to that row's own expert.
        starts = torch.cumsum(padded, 0) - padded
        owner = torch.bucketize(slots, starts, right=True) - 1
        assert torch.equal(owner, indices), "a row landed in another expert's segment"


def test_expert_segment_plan_rejects_an_index_past_the_bank():
    """Routing and the bank must agree on how many experts exist."""

    with pytest.raises(RuntimeError):
        expert_segment_plan(torch.tensor([0, 1, 4]), 3, 16)


def test_row_bucketing_bounds_the_waste_and_the_number_of_shapes():
    """Two properties, and the tension between them is the whole design.

    Too fine a bucket and Transformer Engine sees a row count it has never
    tuned for on every step, which measured at 908 ms per call against 0.70 ms
    for a repeated shape. Too coarse and the padding is real arithmetic. So:
    never pad by more than an eighth, and never let the whole range of packed
    row counts open up more than a couple of hundred distinct shapes.
    """

    multiple = 16
    for rows in list(range(1, 600)) + [1024, 4097, 7568, 16384, 30224, 65536]:
        padded = bucketed_row_count(rows, multiple=multiple)
        assert padded >= rows
        assert padded % multiple == 0
        assert padded <= max(multiple, rows * 9 // 8 + multiple), rows

    shapes = {bucketed_row_count(rows) for rows in range(0, 65_537)}
    assert len(shapes) < 160, len(shapes)
    assert bucketed_row_count(0) == multiple, "an empty call still needs a block"


def test_row_bucketing_does_not_change_the_result():
    """Zero rows must contribute nothing beyond the GEMM's own reassociation.

    A bias-free projection maps a zero row to a zero row, and the padded rows
    are sliced off before anything reads them, so the mathematical result does
    not depend on the bucket. What does move is the last bit or two, because
    changing M lets the GEMM block and accumulate differently -- and that was
    already true of the sixteen-row padding this replaces. What must never
    happen is a bucket that changes the answer in any way a reader would
    notice.
    """

    torch.manual_seed(19)
    projection = torch.nn.Linear(24, 12, bias=False)
    values = torch.randn(37, 24)
    reference = projection(values)
    for padded in (bucketed_row_count(37, multiple=16), 64, 256):
        widened = torch.nn.functional.pad(values, (0, 0, 0, padded - 37))
        torch.testing.assert_close(
            projection(widened)[:37], reference, rtol=1e-4, atol=1e-6
        )
        # The padding cannot raise an absolute maximum, which is what FP8
        # delayed scaling reduces over.
        assert torch.equal(widened.abs().amax(), values.abs().amax())


def test_contractions_agree_with_the_einsums_they_replace():
    """Multiply-and-reduce must be the same contraction, not merely a fast one.

    These three sites were rewritten because einsum lowers them to degenerate
    batched GEMMs that rocBLAS answers with hundreds of milliseconds of host
    time. Rewriting a contraction by hand is an easy place to transpose an axis
    and get a plausible tensor of the right shape, so each is pinned to the
    einsum it replaced.
    """

    generator = torch.Generator().manual_seed(20_260_816)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    streams, vectors = randn(3, 7, 4, 32), randn(4, 32)
    torch.testing.assert_close(
        _stream_gate_logits(streams, vectors),
        torch.einsum("...sd,sd->...s", streams, vectors),
    )

    query, key = randn(3, 7, 4, 16), randn(3, 7, 5, 16)
    torch.testing.assert_close(
        _memory_attention_scores(query, key),
        torch.einsum("...sh,...mh->...sm", query, key),
    )

    weights, value = randn(3, 7, 4, 5), randn(3, 7, 5, 16)
    torch.testing.assert_close(
        _memory_attention_combine(weights, value),
        torch.einsum("...sm,...mh->...sh", weights, value),
    )


def test_batched_newton_schulz_matches_the_matrix_at_a_time_version():
    """Stacking experts must not change what Muon does to any one of them.

    The grouped bank hands Muon one ``[experts, out, in]`` parameter where the
    per-expert bank handed it ``experts`` separate matrices. If the batched
    Newton-Schulz reduced over the expert axis anywhere -- the norm is the easy
    place to get this wrong -- every expert would be orthogonalized against the
    others and the two layouts would train different models while both looked
    healthy.
    """

    generator = torch.Generator().manual_seed(20_260_816)
    for shape in ((8, 64, 32), (5, 24, 96), (3, 16, 16)):
        stack = torch.randn(*shape, generator=generator, dtype=torch.float32)
        batched = _zeropower_via_newton_schulz5(stack, steps=5)
        for index in range(shape[0]):
            one = _zeropower_via_newton_schulz5(stack[index], steps=5)
            torch.testing.assert_close(batched[index], one, rtol=1e-6, atol=1e-6)


def test_muon_steps_a_stacked_expert_bank_like_separate_experts():
    """End to end: the same gradients move the same weights either way."""

    torch.manual_seed(11)
    experts, out_features, in_features = 6, 32, 16
    reference = [
        torch.nn.Parameter(torch.randn(out_features, in_features))
        for _ in range(experts)
    ]
    stacked = torch.nn.Parameter(torch.stack([p.detach().clone() for p in reference]))
    gradients = [torch.randn(out_features, in_features) for _ in range(experts)]
    for parameter, gradient in zip(reference, gradients):
        parameter.grad = gradient.clone()
    stacked.grad = torch.stack(gradients)

    def optimizer_for(params):
        return MuonAdamWHybrid(
            [
                {
                    "params": params,
                    "names": ["w"] * len(params),
                    "optimizer": "muon",
                    "weight_decay": 0.01,
                }
            ],
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            muon_beta=0.95,
            muon_ns_steps=5,
            muon_nesterov=True,
        )

    optimizer_for(reference).step()
    optimizer_for([stacked]).step()
    for index, parameter in enumerate(reference):
        torch.testing.assert_close(stacked[index], parameter, rtol=1e-5, atol=1e-6)


def test_blockwise_int8_muon_state_respects_its_error_bound():
    generator = torch.Generator().manual_seed(20_260_817)
    values = torch.randn(73, 91, generator=generator, dtype=torch.float32)
    quantized, scales = _quantize_blockwise_int8(values, block_size=2048)
    restored = _dequantize_blockwise_int8(
        quantized,
        scales,
        shape=values.shape,
    )

    flat_values = values.flatten()
    flat_restored = restored.flatten()
    for block_index, scale in enumerate(scales):
        start = block_index * 2048
        end = min(start + 2048, flat_values.numel())
        error = (flat_restored[start:end] - flat_values[start:end]).abs()
        assert float(error.max()) <= float(scale) * 0.5001 + 1.0e-7


def test_int8_muon_keeps_only_quantized_momentum_and_tracks_fp32():
    torch.manual_seed(12)
    initial = torch.randn(64, 32)
    full = torch.nn.Parameter(initial.clone())
    quantized = torch.nn.Parameter(initial.clone())

    def optimizer_for(parameter, state_bits):
        return MuonAdamWHybrid(
            [{"params": [parameter], "optimizer": "muon", "weight_decay": 0.01}],
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            muon_beta=0.95,
            muon_ns_steps=5,
            muon_nesterov=True,
            muon_state_bits=state_bits,
        )

    full_optimizer = optimizer_for(full, 32)
    quantized_optimizer = optimizer_for(quantized, 8)
    for _ in range(4):
        gradient = torch.randn_like(initial)
        full.grad = gradient.clone()
        quantized.grad = gradient.clone()
        full_optimizer.step()
        quantized_optimizer.step()

    state = quantized_optimizer.state[quantized]
    assert state["momentum_buffer"].dtype == torch.int8
    assert state["momentum_scale"].dtype == torch.float32
    assert state["momentum_buffer"].numel() == 2048
    torch.testing.assert_close(quantized, full, rtol=2e-4, atol=2e-5)


def _copy_expert_bank(loop_model, grouped_model):
    """Give the grouped model the loop model's weights, bank layout aside."""

    loop_state = loop_model.state_dict()
    grouped_state = grouped_model.state_dict()
    for name, value in loop_state.items():
        if ".local_experts." not in name:
            grouped_state[name] = value
    for layer_index, layer in enumerate(loop_model.layers):
        prefix = f"layers.{layer_index}.moe.local_experts"
        for index, expert in enumerate(layer.moe.local_experts):
            grouped_state[f"{prefix}.gate_up.weight"][index] = (
                loop_state[f"{prefix}.{index}.gate_up.weight"]
            )
            grouped_state[f"{prefix}.down.weight"][index] = (
                loop_state[f"{prefix}.{index}.down.weight"]
            )
    grouped_model.load_state_dict(grouped_state)


def test_dense_ffn_row_trains():
    config = replace(
        _tiny(),
        ffn_mode="dense",
        dense_ffn_intermediate_dim=24,
        n_routed_experts=0,
        n_shared_experts=0,
        expert_parallel_size=1,
        world_size=1,
        expert_replicas=1,
    )
    config._validate_tiny()
    model = Metis16ForCausalLM(config)
    input_ids, labels = _tiny_batch(config)
    output = model(input_ids, labels, curriculum=_curriculum())
    (output.loss + output.auxiliary_loss).backward()
    assert torch.isfinite(output.loss)
    # N-gram table gradients are sparse tensors, which have no ``isfinite``;
    # check their values instead of skipping them.
    gradients = [
        g.coalesce().values() if g.is_sparse else g
        for g in (p.grad for p in model.parameters())
        if g is not None
    ]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_pathway_freeze_reuses_pass_one_experts():
    torch.manual_seed(11)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config)

    seen: list[torch.Tensor] = []
    layer = model.layers[0].moe
    original = layer.forward

    def spy(hidden_states, **kwargs):
        result = original(hidden_states, **kwargs)
        seen.append(kwargs.get("pass_index", 1))
        return result

    layer.forward = spy  # type: ignore[method-assign]
    curriculum = _curriculum(pathway_mode="frozen", continuation_mode="fixed_max")
    with torch.no_grad():
        output = model(input_ids, labels, curriculum=curriculum)
    layer.forward = original  # type: ignore[method-assign]
    assert torch.isfinite(output.loss)
    assert max(seen) > 1, "the frozen-pathway test never reached a second pass"


def test_pathway_freeze_changes_the_result_but_not_the_shape():
    torch.manual_seed(13)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config)
    with torch.no_grad():
        per_pass = model(
            input_ids,
            labels,
            curriculum=_curriculum(
                continuation_mode="fixed_max", pathway_mode="per_pass"
            ),
        )
        frozen = model(
            input_ids,
            labels,
            curriculum=_curriculum(
                continuation_mode="fixed_max", pathway_mode="frozen"
            ),
        )
    assert per_pass.loss.shape == frozen.loss.shape
    assert not torch.allclose(per_pass.loss, frozen.loss), (
        "freezing the pathway changed nothing, so the pathway axis is inert on "
        "this fixture and row 6 would measure noise"
    )


def test_random_width_control_spends_the_target_budget():
    torch.manual_seed(17)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config, batch=8, length=16)
    curriculum = _curriculum(
        routed_k_mode="random", target_mean_routed_k=2.0, random_policy_seed=99
    )
    with torch.no_grad():
        output = model(input_ids, labels, curriculum=curriculum)
    observed = float(output.telemetry["mean_routed_k"])
    assert 1.4 < observed < 2.6, observed


def test_random_depth_control_spends_the_target_budget():
    torch.manual_seed(19)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config, batch=16, length=16)
    curriculum = _curriculum(
        continuation_mode="random", target_mean_depth=2.0, random_policy_seed=101
    )
    with torch.no_grad():
        output = model(input_ids, labels, curriculum=curriculum)
    observed = float(output.telemetry["mean_depth"])
    assert 1.5 < observed < 2.6, observed


def test_random_policies_are_reproducible_from_the_seed():
    config = _tiny()
    curriculum = _curriculum(routed_k_mode="random", random_policy_seed=5)
    losses = []
    for _ in range(2):
        torch.manual_seed(23)
        model = Metis16ForCausalLM(config)
        model.eval()
        input_ids, labels = _tiny_batch(config)
        with torch.no_grad():
            losses.append(model(input_ids, labels, curriculum=curriculum).loss)
    torch.testing.assert_close(losses[0], losses[1])


def test_dense_config_rejects_width_and_pathway_curricula():
    config = replace(
        _tiny(),
        ffn_mode="dense",
        dense_ffn_intermediate_dim=24,
        n_routed_experts=0,
        n_shared_experts=0,
    )
    with pytest.raises(ValueError, match="no width or pathway decision"):
        CurriculumState(routed_k_mode="fixed", fixed_routed_k=2).validate(config)


# --------------------------------------------------------------------------
# analysis


def test_routing_analyzer_collects_transitions_and_correlation(tmp_path: Path):
    torch.manual_seed(29)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config, batch=4, length=16)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    analyzer = RoutingAnalyzer(config, max_passes=config.max_passes)
    with analyzer.capture(model):
        with torch.no_grad():
            output = model(
                input_ids,
                labels,
                curriculum=_curriculum(continuation_mode="fixed_max"),
            )
        analyzer.observe(output, attention_mask)

    report = analyzer.report()
    assert report["observations"] == 1
    assert len(report["depth_distribution"]) == config.max_passes + 1
    assert len(report["transition_off_diagonal_mass"]) == config.max_passes - 1
    path = analyzer.flush(tmp_path / "routing.json", step=3)
    payload = json.loads(path.read_text())
    assert payload["step"] == 3
    # flush resets, so a second report must be empty rather than double-counting
    assert analyzer.report()["observations"] == 0


def test_analyzer_capture_leaves_no_residue():
    config = _tiny()
    model = Metis16ForCausalLM(config)
    analyzer = RoutingAnalyzer(config, max_passes=config.max_passes)
    with analyzer.capture(model):
        pass
    for layer in model.layers:
        assert layer.moe.capture_selection is False
        assert layer.moe._analysis_last_selection is None


def test_halt_calibration_bins_are_well_formed():
    config = _tiny()
    analyzer = RoutingAnalyzer(config, max_passes=config.max_passes)
    predicted = torch.tensor([0.1, 0.4, 0.9, 0.95])
    continued = torch.tensor([0.0, 0.0, 1.0, 1.0])
    valid = torch.ones(4, dtype=torch.bool)
    analyzer.observe_calibration(predicted=predicted, continued=continued, valid=valid)
    curve = analyzer.calibration_curve()
    assert curve
    assert sum(row["count"] for row in curve) == 4
    for row in curve:
        assert 0.0 <= row["mean_predicted"] <= 1.0
        assert 0.0 <= row["mean_realized"] <= 1.0


# --------------------------------------------------------------------------
# schedule


def test_schedule_warms_up_then_decays_to_the_floor():
    schedule = AblationSchedule(total_steps=1_000, base_learning_rate=1.0e-4)
    assert schedule.learning_rate(0) < schedule.learning_rate(10)
    peak = max(schedule.learning_rate(step) for step in range(1_000))
    assert pytest.approx(peak, rel=1e-6) == 1.0e-4
    assert schedule.learning_rate(999) < 0.2 * 1.0e-4


def test_schedule_is_identical_across_rows():
    """Nothing about the schedule may differ between rows, or it becomes a
    candidate explanation for a difference between them."""

    schedules = {
        spec.name: AblationSchedule(total_steps=27_246, base_learning_rate=1.8e-4)
        for spec in ABLATION_LADDER
    }
    values = {name: s.learning_rate(5_000) for name, s in schedules.items()}
    assert len(set(values.values())) == 1


# --------------------------------------------------------------------------
# waves 2 and 3


def test_every_wave_allocation_fits_and_uses_whole_nodes():
    from metis_ablation.specs import WAVES

    for wave, specs in WAVES.items():
        report = validate_allocation(specs)
        assert report["spare_apus"] >= 0, wave
        for spec in specs:
            assert spec.apus % 4 == 0, (wave, spec.name)
            assert (
                spec.apus * spec.micro_batch * spec.grad_accum
                == GLOBAL_BATCH_SEQUENCES
            ), (wave, spec.name)


def test_row_names_are_globally_unique():
    from metis_ablation.specs import ALL_SPECS

    names = [spec.name for spec in ALL_SPECS]
    assert len(set(names)) == len(names)


def test_scaling_ladder_is_monotone_in_size():
    """Three distinct scale points, or it is not a scaling curve."""

    from metis_ablation.specs import WAVES, spec_by_name

    sizes = []
    for name in ("more-core-xxs", "more-core-xs", "more-core"):
        audit = spec_by_name(name).model_config(
            mhc_backend="torch_reference",
            mamba_backend="torch_reference",
            attention_backend="torch_reference",
        ).logical_parameter_audit()
        sizes.append(audit.stored_total)
    assert sizes == sorted(sizes)
    # Each step should be a real jump, not a rounding difference.
    assert sizes[1] > 1.5 * sizes[0]
    assert sizes[2] > 1.5 * sizes[1]
    assert len(WAVES["2"]) == 8


def test_scaling_ladder_keeps_the_ngram_table_proportional():
    """A fixed 0.30B table would be most of the smallest model, and the scaling
    curve would mostly be measuring a constant lookup table."""

    from metis_ablation.specs import spec_by_name

    fractions = []
    for name in ("more-core-xxs", "more-core-xs", "more-core"):
        audit = spec_by_name(name).model_config(
            mhc_backend="torch_reference",
            mamba_backend="torch_reference",
            attention_backend="torch_reference",
        ).logical_parameter_audit()
        fractions.append(audit.ngram_tables / audit.stored_total)
    assert max(fractions) - min(fractions) < 0.10, fractions
    assert max(fractions) < 0.30, fractions


def test_scaling_dense_controls_stay_parameter_matched():
    from metis_ablation.specs import spec_by_name

    for scale in ("-xs", "-xxs"):
        def stored(name: str) -> int:
            return spec_by_name(name).model_config(
                mhc_backend="torch_reference",
                mamba_backend="torch_reference",
                attention_backend="torch_reference",
            ).logical_parameter_audit().stored_total

        dense = stored(f"dense-param-matched{scale}")
        sparse = stored(f"more-core{scale}")
        assert abs(dense - sparse) / sparse < 0.02, (scale, dense, sparse)


def test_seed_wave_mirrors_its_parents_and_changes_only_the_seed(tmp_path: Path):
    from metis_ablation.campaign import emit_slurm
    from metis_ablation.specs import SECOND_SEED, WAVES, spec_by_name

    for spec in WAVES["3"]:
        parent = spec_by_name(spec.name.removesuffix("-seed2"))
        assert spec.ffn_mode == parent.ffn_mode
        assert spec.continuation_mode == parent.continuation_mode
        assert spec.routed_k_mode == parent.routed_k_mode
        assert spec.depth_memory == parent.depth_memory

    emit_slurm(
        tmp_path,
        wave="3",
        repo_root="/repo",
        output_root="/out",
        release_root="/release",
        seed=1234,
    )
    for path in (tmp_path / "wave3").glob("*.sbatch"):
        body = path.read_text()
        assert f"--seed {SECOND_SEED}" in body, path.name
        assert "--seed 1234" not in body, path.name


def test_wave_one_launchers_keep_the_requested_seed(tmp_path: Path):
    from metis_ablation.campaign import emit_slurm

    emit_slurm(
        tmp_path,
        wave="1",
        repo_root="/repo",
        output_root="/out",
        release_root="/release",
        seed=1234,
    )
    body = (tmp_path / "wave1" / "10-more-core.sbatch").read_text()
    assert "--seed 1234" in body
    assert "export WORLD_SIZE=28" in body
    assert "srun --kill-on-bad-exit=1 --network=disable_rdzv_get" in body


def test_every_task_in_a_launcher_gets_its_own_rank_and_apu(tmp_path: Path):
    """The step body, not the batch shell, must resolve RANK and LOCAL_RANK.

    Exporting them from the batch shell is the failure this pins: SLURM_PROCID
    only exists per task, so a batch-shell export launches every task as rank 0
    against APU 0.  Both forms pass a string match for "RANK", so the launcher
    is executed here with a stub ``srun`` that fans out four tasks and a stub
    ``python`` that records what each one actually saw.
    """

    import subprocess

    from metis_ablation.campaign import emit_slurm

    emit_slurm(
        tmp_path,
        wave="1",
        repo_root="/repo",
        output_root=str(tmp_path / "out"),
        release_root="/release",
    )
    body = (tmp_path / "wave1" / "10-more-core.sbatch").read_text()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    recorded = tmp_path / "ranks.txt"
    (bin_dir / "scontrol").write_text("#!/bin/bash\necho node0\n")
    (bin_dir / "srun").write_text(
        "#!/bin/bash\n"
        # Drop srun's own flags; what remains is `bash -c <step body>`.
        "while [[ \"$1\" == -* ]]; do shift; done\n"
        "for i in 0 1 2 3; do\n"
        f"  SLURM_PROCID=$i SLURM_LOCALID=$i \"$@\" || exit 1\n"
        "done\n"
    )
    (bin_dir / "python").write_text(
        "#!/bin/bash\n"
        f"echo \"${{RANK:-unset}} ${{LOCAL_RANK:-unset}}\" >> {recorded}\n"
    )
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)

    runtime = tmp_path / "runtime.sh"
    sourced = tmp_path / "sourced.txt"
    runtime.write_text(f'echo "${{SLURM_PROCID:-unset}}" >> {sourced}\n')

    script = tmp_path / "row.sbatch"
    script.write_text(body)
    result = subprocess.run(
        ["bash", str(script)],
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "METIS_ABLATION_RUNTIME": str(runtime),
            "SLURM_JOB_NODELIST": "node[0-6]",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    seen = [line.split() for line in recorded.read_text().split("\n") if line.strip()]
    assert seen == [["0", "0"], ["1", "1"], ["2", "2"], ["3", "3"]], seen

    # The runtime derives per-rank scratch paths from SLURM_PROCID, so it has to
    # be sourced inside the step, once per task, not once in the batch shell.
    activations = [
        line.strip() for line in sourced.read_text().split("\n") if line.strip()
    ]
    assert activations == ["0", "1", "2", "3"], activations


def test_launchers_do_not_request_a_gpu_gres(tmp_path: Path):
    """Portage's parry partition defines no GPU gres; asking for one is fatal.

    ``sbatch --gpus-per-task=1`` is rejected outright there ("Invalid generic
    resource (gres) specification"), so the four MI300A APUs on a node are
    addressed by local task id instead.
    """

    from metis_ablation.campaign import emit_slurm, emit_sweep

    written = emit_slurm(
        tmp_path,
        wave="1",
        repo_root="/repo",
        output_root="/out",
        release_root="/release",
    )
    written += emit_sweep(
        tmp_path,
        repo_root="/repo",
        output_root="/out",
        release_root="/release",
    )
    for path in written:
        if path.suffix != ".sbatch":
            continue
        body = path.read_text()
        assert "--gpus-per-task" not in body, path.name
        assert "--gres" not in body, path.name
        # Four tasks per node is what maps a task onto an APU.
        assert "#SBATCH --ntasks-per-node=4" in body, path.name


def test_sweep_covers_every_archetype_at_every_rate(tmp_path: Path):
    from metis_ablation.campaign import (
        SWEEP_ARCHETYPES,
        SWEEP_LEARNING_RATES,
        emit_sweep,
    )

    written = emit_sweep(
        tmp_path, repo_root="/repo", output_root="/out", release_root="/release"
    )
    scripts = [path for path in written if path.suffix == ".sbatch"]
    assert len(scripts) == len(SWEEP_ARCHETYPES) * len(SWEEP_LEARNING_RATES)
    bodies = "\n".join(path.read_text() for path in scripts)
    for rate in SWEEP_LEARNING_RATES:
        assert f"--learning-rate {rate:g}" in bodies
    # A sweep must not resume a previous sweep's checkpoint.
    assert bodies.count("--no-resume") == len(scripts)


def test_campaign_plan_runs_for_every_wave():
    from metis_ablation.campaign import plan

    for wave in ("1", "2", "3", "all"):
        payload = plan(wave=wave)
        assert payload["rows"], wave
        assert payload["campaign_exaflops"] > 0, wave


# --------------------------------------------------------------------------
# the memory-gate fix: MoRE-Core must not route on retrieved memory


def test_disabling_depth_memory_also_silences_the_routing_path():
    """``memory_gate_scale=0`` must zero the summary that feeds the continuation,
    width, and pathway heads -- not merely the representation path.

    If only the representation path were gated, MoRE-Core would still route on
    retrieved memory and the MoRE-Core / MoRE-RM comparison would measure
    something narrower than it claims."""

    torch.manual_seed(31)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config, batch=2, length=16)

    summaries: list[torch.Tensor] = []
    original = model.depth_memory.retrieve

    def spy(bank, streams, *, active_mask, gate_scale):
        fused, summary, weights = original(
            bank, streams, active_mask=active_mask, gate_scale=gate_scale
        )
        summaries.append(summary.detach().clone())
        return fused, summary, weights

    model.depth_memory.retrieve = spy  # type: ignore[method-assign]
    with torch.no_grad():
        model(
            input_ids,
            labels,
            curriculum=_curriculum(
                continuation_mode="fixed_max", memory_gate_scale=0.0
            ),
        )
    model.depth_memory.retrieve = original  # type: ignore[method-assign]

    assert summaries, "the depth memory was never consulted"
    for summary in summaries:
        assert torch.all(summary == 0), (
            "retrieved memory still reaches the routing heads with the gate closed"
        )


def test_depth_memory_summary_is_untouched_at_full_gate():
    """The gate fix must be exactly a no-op at the production scale of 1.0."""

    torch.manual_seed(37)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config, batch=2, length=16)

    captured: list[torch.Tensor] = []
    original = model.depth_memory.retrieve

    def spy(bank, streams, *, active_mask, gate_scale):
        result = original(bank, streams, active_mask=active_mask, gate_scale=gate_scale)
        captured.append(result[1].detach().clone())
        return result

    model.depth_memory.retrieve = spy  # type: ignore[method-assign]
    with torch.no_grad():
        model(input_ids, labels, curriculum=_curriculum(continuation_mode="fixed_max"))
    model.depth_memory.retrieve = original  # type: ignore[method-assign]
    assert captured
    assert any(bool((summary != 0).any()) for summary in captured), (
        "the memory summary is zero even at full gate; the ablation pair would "
        "be comparing two identical models"
    )


def test_more_core_and_more_rm_specs_actually_differ():
    from metis_ablation.specs import spec_by_name

    core = spec_by_name("more-core").curriculum()
    memory = spec_by_name("more-rm").curriculum()
    assert core.memory_gate_scale == 0.0
    assert memory.memory_gate_scale == 1.0
    assert core.continuation_mode == memory.continuation_mode
    assert core.routed_k_mode == memory.routed_k_mode


# --------------------------------------------------------------------------
# transition alignment under packing


def test_transitions_are_aligned_when_later_passes_are_packed():
    """A packed pass is a subsequence of its predecessor, not a prefix.

    Comparing them positionally would pair unrelated tokens and manufacture
    off-diagonal mass out of nothing, so a model whose coalition never changes
    must still report exactly zero."""

    torch.manual_seed(41)
    config = _tiny()
    model = Metis16ForCausalLM(config)
    model.eval()
    input_ids, labels = _tiny_batch(config, batch=4, length=16)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    analyzer = RoutingAnalyzer(config, max_passes=config.max_passes)
    with analyzer.capture(model):
        with torch.no_grad():
            output = model(
                input_ids,
                labels,
                curriculum=_curriculum(pathway_mode="frozen"),
            )
        analyzer.observe(output, attention_mask)

    # Adaptive depth means later passes are packed. With the pathway frozen the
    # top expert cannot change, so every transition must sit on the diagonal.
    packed = output.active_masks.sum(dim=(1, 2))
    assert int(packed[0]) > int(packed[-1]), "no pass was actually packed"
    for mass in analyzer.transition_off_diagonal_mass():
        assert mass == pytest.approx(0.0, abs=1e-12), (
            "frozen pathways produced off-diagonal transitions, so packed passes "
            "are being compared positionally"
        )


# --------------------------------------------------------------------------
# trainer plumbing


def _smoke_spec(name: str):
    """A one-rank version of a row, for CPU end-to-end tests."""

    base = spec_by_name(name)
    spec = AblationSpec(
        **{**base.__dict__, "apus": 1, "micro_batch": GLOBAL_BATCH_SEQUENCES,
           "grad_accum": 1}
    )
    object.__setattr__(spec, "micro_batch", 1)
    object.__setattr__(spec, "grad_accum", 2)
    return spec


@pytest.fixture
def tiny_proxy(monkeypatch):
    """Shrink the proxy geometry so a real trainer run finishes in seconds."""

    import metis_ablation.specs as specs_module

    original = specs_module.proxy_config

    def small(*, world_size, ffn_mode="moe", ngram_slots_per_head=None,
              overrides=None, **kwargs):
        base = dict(overrides or {})
        base.update(
            d_model=128, n_heads=2, head_dim=64, n_kv_heads=1, n_layers=2,
            attention_indices=(1,), latent_dim=64, expert_intermediate_dim=32,
            n_routed_experts=8, mamba_ngroups=1, mamba_head_dim=64,
            mamba_chunk_size=16, sequence_length=64, final_context_length=64,
            context_extension_train_length=64, vocab_size=256,
            mhc_pass_embedding_dim=8, route_feature_dim=16, memory_dim=16,
            memory_heads=2, max_routed_k=8, target_mean_routed_k=4.0,
            activation_recompute_policy="none",
        )
        if ffn_mode == "dense":
            base.update(n_routed_experts=0, n_shared_experts=0)
        kwargs.update(
            mhc_backend="torch_reference",
            mamba_backend="torch_reference",
            attention_backend="torch_reference",
        )
        return original(
            world_size=world_size, ffn_mode=ffn_mode, ngram_slots_per_head=257,
            overrides=base, **kwargs
        )

    monkeypatch.setattr(specs_module, "proxy_config", small)
    return small


def _train(spec, root: Path, **kwargs):
    from metis_ablation.train import train_row

    defaults = dict(
        release_root=None, budget_tokens=20_000_000, learning_rate=1.0e-4,
        seed=1, telemetry_every=100, analysis_every=0, checkpoint_every=0,
        device_override="cpu", synthetic=True,
    )
    defaults.update(kwargs)
    return train_row(spec, output_root=root, **defaults)


def test_trainer_runs_every_row_of_every_wave(tiny_proxy, tmp_path: Path):
    from metis_ablation.specs import WAVES

    for wave, specs in WAVES.items():
        for spec in specs:
            summary = _train(_smoke_spec(spec.name), tmp_path, max_steps=1)
            assert summary["final_loss"] == summary["final_loss"], (wave, spec.name)
            assert (tmp_path / spec.name / "run.json").exists()


def test_throughput_canary_can_skip_the_terminal_checkpoint(
    tiny_proxy,
    tmp_path: Path,
):
    spec = _smoke_spec("more-core")
    _train(spec, tmp_path, max_steps=1, final_checkpoint=False)

    manifest = json.loads((tmp_path / spec.name / "run.json").read_text())
    assert manifest["final_checkpoint"] is False
    assert list((tmp_path / spec.name / "checkpoints").iterdir()) == []


def test_resume_picks_up_where_it_stopped(tiny_proxy, tmp_path: Path):
    spec = _smoke_spec("more-rm")
    first = _train(spec, tmp_path, checkpoint_every=2, max_steps=4)
    assert first["start_step"] == 0

    second = _train(spec, tmp_path, checkpoint_every=2, max_steps=4)
    assert second["start_step"] == 4
    assert second["resumed_from"] is not None

    fresh = _train(spec, tmp_path, checkpoint_every=2, max_steps=4, resume=False)
    assert fresh["start_step"] == 0


def test_resume_refuses_a_changed_schedule(tiny_proxy, tmp_path: Path):
    """Resuming a cosine decay against a different horizon would silently train
    a different model, so it is an error rather than a warning."""

    spec = _smoke_spec("more-core")
    _train(spec, tmp_path, checkpoint_every=2, max_steps=4)
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        _train(spec, tmp_path, checkpoint_every=2, max_steps=3)
    with pytest.raises(RuntimeError, match="Refusing to resume"):
        _train(spec, tmp_path, checkpoint_every=2, max_steps=4, learning_rate=9.0e-4)


def test_current_scaling_restore_discards_only_delayed_fp8_metadata(
    tiny_proxy,
    tmp_path: Path,
):
    """Delayed TE history is transient state, not a current-scaling parameter.

    Transformer Engine accepts delayed ``_extra_state`` into a current-scaling
    module, then changes its metadata inventory on the first recompute. The
    failure arrives in backward, long after state_dict claimed the restore was
    valid, so the checkpoint loader has to remove that recipe-owned state
    explicitly while keeping every real weight strict.
    """

    from metis_ablation.train import RunPaths, _restore_checkpoint

    class ExtraStateLinear(torch.nn.Linear):
        def get_extra_state(self):
            return {"delayed_amax": torch.ones(1)}

        def set_extra_state(self, state):
            self.delayed_amax = state

    class CurrentScalingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = ExtraStateLinear(3, 2)
            self.config = SimpleNamespace(
                precision=SimpleNamespace(fp8_scaling="current")
            )

    source = CurrentScalingModel()
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    paths = RunPaths(tmp_path)
    paths.prepare()
    checkpoint = paths.checkpoints / "step-0000002"
    checkpoint.mkdir()
    torch.save(
        {
            "model": source.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 2,
            "total_steps": 4,
            "base_learning_rate": 0.1,
        },
        checkpoint / "state.pt",
    )

    target = CurrentScalingModel()
    restored = _restore_checkpoint(
        checkpoint,
        model=target,
        optimizer=torch.optim.SGD(target.parameters(), lr=0.1),
        device=torch.device("cpu"),
        total_steps=4,
        learning_rate=0.1,
    )
    assert restored == 2
    torch.testing.assert_close(target.projection.weight, source.projection.weight)
    torch.testing.assert_close(target.projection.bias, source.projection.bias)
    assert not hasattr(target.projection, "delayed_amax")


def test_checkpoints_are_pruned_and_atomic(tiny_proxy, tmp_path: Path):
    spec = _smoke_spec("more-core")
    _train(spec, tmp_path, checkpoint_every=1, max_steps=6)
    kept = sorted((tmp_path / spec.name / "checkpoints").glob("step-*"))
    assert 0 < len(kept) <= 2, [path.name for path in kept]
    for path in kept:
        assert (path / "state.pt").exists()
        assert not (path / "state.pt.partial").exists()


def test_storage_policy_is_applied_to_routers(tiny_proxy, tmp_path: Path):
    """Routers must reach FP32 storage; a silent BF16 router would only show up
    as slightly noisier routing, which no loss curve would reveal."""

    from metis_ablation.specs import spec_by_name as lookup

    spec = _smoke_spec("more-core")
    config = spec.model_config(
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    )
    model = Metis16ForCausalLM(config)
    model.apply_parameter_storage_policy(device=torch.device("cpu"))
    routers = [
        (name, parameter.dtype)
        for name, parameter in model.named_parameters()
        if ".expert_router." in name or ".k_router." in name
        or name.startswith("continuation.")
    ]
    assert routers
    for name, dtype in routers:
        assert dtype == torch.float32, (name, dtype)

    from metis_ablation.train import _assert_storage_policy

    _assert_storage_policy(model)


def test_storage_policy_assertion_catches_a_bf16_router(tiny_proxy):
    from metis_ablation.train import _assert_storage_policy

    spec = _smoke_spec("more-core")
    config = spec.model_config(
        mhc_backend="torch_reference",
        mamba_backend="torch_reference",
        attention_backend="torch_reference",
    )
    model = Metis16ForCausalLM(config)
    # Straight ``.to(bfloat16)`` is exactly the mistake the assertion exists to
    # catch: it looks like it worked and quietly demotes every router.
    model.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="storage policy"):
        _assert_storage_policy(model)


def test_run_manifest_records_everything_needed_to_reproduce(tiny_proxy, tmp_path: Path):
    spec = _smoke_spec("more-rm")
    _train(spec, tmp_path, max_steps=1)
    manifest = json.loads((tmp_path / spec.name / "run.json").read_text())
    for key in (
        "spec", "model", "optimizer", "parameters", "schedule", "curriculum",
        "global_batch_tokens", "total_steps", "world_size", "precision_profile",
        "sampler",
    ):
        assert key in manifest, key
    assert manifest["global_batch_tokens"] == GLOBAL_BATCH_TOKENS
    assert manifest["curriculum"]["memory_gate_scale"] == 1.0


def test_telemetry_carries_the_paper_axes(tiny_proxy, tmp_path: Path):
    spec = _smoke_spec("more-core")
    _train(spec, tmp_path, telemetry_every=1, max_steps=2)
    lines = (tmp_path / spec.name / "telemetry" / "rank-00000.jsonl").read_text()
    record = json.loads(lines.splitlines()[-1])
    for key in (
        "loss", "learning_rate", "grad_norm", "cumulative_tokens",
        "estimated_hardware_flops", "estimated_mfu", "depth_histogram",
        "tokens_per_second", "fp8_parity_relative_error",
    ):
        assert key in record, key
    for key in ("mean_depth", "mean_routed_k", "expert_entropy_ratio"):
        assert key in record["telemetry"], key


def test_every_release_inventory_attribute_the_tree_names_exists():
    """A misspelled loader survives every test that passes ``--synthetic``.

    ``metis_ablation.train`` called ``ReleaseInventory.load`` where the class
    only defines ``from_release_root``.  Nothing caught it, because the only
    path the suite exercises is the synthetic one, and the real branch is the
    first thing a 28-rank job does after it has paid for its allocation.
    """

    import re

    from metis_training.data import ReleaseInventory

    root = Path(__file__).resolve().parent.parent / "src"
    pattern = re.compile(r"\bReleaseInventory\.([A-Za-z_][A-Za-z0-9_]*)")
    seen: set[tuple[str, str]] = set()
    for path in root.rglob("*.py"):
        for attribute in pattern.findall(path.read_text(encoding="utf-8")):
            seen.add((str(path.relative_to(root)), attribute))

    assert seen, "no ReleaseInventory usages found; the guard has gone stale"
    missing = [
        f"{where} names ReleaseInventory.{attribute}"
        for where, attribute in sorted(seen)
        if not hasattr(ReleaseInventory, attribute)
    ]
    assert not missing, missing


def test_random_controls_draw_identically_when_the_pass_is_replayed():
    """Pass-level recompute runs each pass twice; both must route the same.

    The controls used a cached torch.Generator, which advances on every call.
    torch.utils.checkpoint restores the default RNG but not user-created
    generators, so the recomputed forward drew fresh randomness and the
    backward differentiated a coalition the forward never chose. On rows 12 and
    13 that is not a crash, it is a control that measures nothing -- which is
    the one thing those rows exist to rule out.
    """

    import torch
    from types import SimpleNamespace

    from metis_training.model import (
        AdaptiveDroplessMoE,
        CurriculumState,
        Metis16ForCausalLM,
    )

    device = torch.device("cpu")
    curriculum = CurriculumState(random_policy_seed=4_242, random_policy_step=7)

    def draw(fn, owner, state, pass_index):
        generator = fn(owner, state, device, pass_index)
        assert generator is not None
        return torch.rand(16, generator=generator, device=device)

    for fn, owner in (
        (AdaptiveDroplessMoE._random_policy_generator, SimpleNamespace(layer_idx=3)),
        (Metis16ForCausalLM._random_depth_generator, SimpleNamespace()),
    ):
        first = draw(fn, owner, curriculum, 1)
        replayed = draw(fn, owner, curriculum, 1)
        torch.testing.assert_close(first, replayed)

        # ... while still moving with the step and the pass, or the control
        # would spend the whole run on one frozen coalition.
        later_step = draw(
            fn, owner, replace(curriculum, random_policy_step=8), 1
        )
        later_pass = draw(fn, owner, curriculum, 2)
        assert not torch.allclose(first, later_step)
        assert not torch.allclose(first, later_pass)


def test_random_policy_draws_differ_across_layers():
    """Otherwise a per-token random width becomes a per-token constant."""

    import torch
    from types import SimpleNamespace

    from metis_training.model import AdaptiveDroplessMoE, CurriculumState

    device = torch.device("cpu")
    curriculum = CurriculumState(random_policy_seed=4_242, random_policy_step=7)
    draws = []
    for layer_idx in (0, 1, 2):
        generator = AdaptiveDroplessMoE._random_policy_generator(
            SimpleNamespace(layer_idx=layer_idx), curriculum, device, 0
        )
        draws.append(torch.rand(16, generator=generator, device=device))
    assert not torch.allclose(draws[0], draws[1])
    assert not torch.allclose(draws[1], draws[2])


def test_recompute_replay_keeps_the_expert_assignment_shape():
    """A replayed pass must route where the forward routed, not where it would.

    Router logits are not bitwise reproducible across the two executions of a
    pass-level recompute -- reductions over atomics reassociate -- so tokens
    near a top-k tie land differently. That changes the number of assignments
    and therefore the shape of the packed expert input, and
    torch.utils.checkpoint rejects the replay. The tape fixes the identities;
    this asserts that a second call with *perturbed* logits still produces the
    forward's assignment, which is a strictly harder condition than the tiny
    perturbations the hardware actually produces.
    """

    from metis_training.model import RoutingReplayTape
    from metis_training.model_config import Metis16Config

    config = Metis16Config.tiny_for_tests()
    torch.manual_seed(16_062_026)

    tape = RoutingReplayTape()
    top_indices = torch.tensor([[[0, 2, 1], [3, 1, 0]]])
    chosen_k = torch.tensor([[2, 1]])
    tape.record(0, 0, top_indices, chosen_k)

    # The replay of the same pass and layer gets the recorded identities.
    replayed = tape.lookup(0, 0)
    assert replayed is not None
    torch.testing.assert_close(replayed[0], top_indices)
    torch.testing.assert_close(replayed[1], chosen_k)

    # Write-once: a second record for the same pass and layer -- which is what
    # the replay itself would do -- must not overwrite the forward's decision.
    tape.record(0, 0, torch.zeros_like(top_indices), torch.zeros_like(chosen_k))
    again = tape.lookup(0, 0)
    assert again is not None
    torch.testing.assert_close(again[0], top_indices)
    torch.testing.assert_close(again[1], chosen_k)

    # Different passes and layers are independent; taping them together would
    # freeze the pathway axis by accident and silently turn every row into
    # row 6.
    assert tape.lookup(1, 0) is None
    assert tape.lookup(0, 1) is None

    tape.clear()
    assert tape.lookup(0, 0) is None


def test_replay_tape_exists_only_where_a_pass_is_actually_replayed():
    """No tape without pass recompute: it would be memory held for nothing."""

    from metis_training.model import CurriculumState, Metis16ForCausalLM
    from metis_training.model_config import Metis16Config

    config = Metis16Config.tiny_for_tests()
    assert config.activation_recompute_policy == "none"
    model = Metis16ForCausalLM(config)
    model.train()
    input_ids = torch.zeros((1, 8), dtype=torch.long)
    curriculum = CurriculumState(fixed_routed_k=config.max_routed_k)
    model(input_ids, curriculum=curriculum)
    assert model._routing_replay_tape is None

    model.set_activation_recompute_policy("pass")
    model(input_ids, curriculum=curriculum)
    assert model._routing_replay_tape is not None

    # Inference replays nothing, so it should not tape either.
    model.eval()
    with torch.no_grad():
        model(input_ids, curriculum=curriculum)
    assert model._routing_replay_tape is None
