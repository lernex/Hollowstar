from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

import metis_training.model as model_module
from metis_training.model import (
    CurriculumState,
    Metis16ForCausalLM,
    MetisProcessGroups,
    PLACEMENT_ROW_SHARDED_TABLE,
    sinkhorn_doubly_stochastic,
)
from metis_training.model_config import (
    Metis16Config,
    default_manifest_path,
    load_family_config,
)


def _batch(config: Metis16Config, *, batch: int = 2, length: int = 8):
    generator = torch.Generator().manual_seed(16062026)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch, length),
        generator=generator,
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    reset_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    reset_mask[:, 0] = True
    reset_mask[:, length // 2] = True
    return input_ids, labels, reset_mask


def _fixed_curriculum(k: int = 2, *, target_depth: float = 2.0):
    return CurriculumState(
        routed_k_mode="fixed",
        fixed_routed_k=k,
        stochastic_routing=False,
        target_mean_depth=target_depth,
        target_mean_routed_k=float(k),
    )


@pytest.mark.parametrize(
    ("family", "stored", "active", "world", "ep", "replicas"),
    [
        ("praxis", 3_545_482_071, 470_844_375, 160, 32, 5),
        ("logos", 12_000_001_676, 1_189_454_220, 352, 32, 11),
    ],
)
def test_production_manifests_are_locked_and_audited(
    family: str,
    stored: int,
    active: int,
    world: int,
    ep: int,
    replicas: int,
) -> None:
    path = default_manifest_path(family)
    assert path == Path("configs/metis16").resolve() / f"{family}.yaml"
    config = load_family_config(path, family=family)
    audit = config.logical_parameter_audit()
    assert config.schema == "metis.model-family/v1"
    assert config.world_size == world
    assert config.expert_parallel_size == ep
    assert config.expert_replicas == replicas
    assert config.world_size == config.expert_parallel_size * config.expert_replicas
    assert config.attention_backend == "varlen_fused_required"
    assert config.context_extension_train_length == 163_840
    assert config.ngram_memory.table_mode in {"replicated", "row_sharded"}
    assert config.ngram_memory.retrieved_rows_per_token == 16
    assert audit.stored_total == stored
    assert audit.active_per_pass_mean == active
    assert config.expected_parameter_audit == audit.to_dict()
    assert tuple(sorted(config.autotune.micro_batch_sizes, reverse=True)) == (
        config.autotune.micro_batch_sizes
    )
    assert config.autotune.preferred_learning_rate in config.autotune.learning_rates
    assert config.autotune.precision_profiles == ("fp8", "bf16")
    assert config.autotune.dispatch_overlap == ("on", "off")
    assert config.expert_balance_coefficient == 0.0
    assert config.expert_balance_bias_update_rate == pytest.approx(1.0e-3)


def test_tiny_parameter_audit_matches_instantiated_unique_parameters() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    local = model.local_parameter_audit()
    instantiated = sum(
        parameter.numel()
        for parameter in {id(value): value for value in model.parameters()}.values()
    )
    assert local["local_total"] == instantiated
    assert local["logical_total"] == instantiated
    assert config.logical_parameter_audit().stored_total == instantiated


def test_shapes_dropless_dynamic_k_and_backward() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    input_ids, labels, reset_mask = _batch(config)
    output = model(
        input_ids,
        labels,
        reset_mask=reset_mask,
        force_depth=2,
        curriculum=_fixed_curriculum(2),
    )
    assert output.logits is None
    assert output.final_hidden_state.shape == (*input_ids.shape, config.d_model)
    assert output.chosen_depths.shape == input_ids.shape
    assert output.active_masks.shape == (config.max_passes, *input_ids.shape)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.auxiliary_loss)
    assert int(output.telemetry["moe_assignments"]) > 0
    assert torch.equal(
        output.telemetry["moe_assignments"],
        output.telemetry["moe_processed_assignments"],
    )
    assert int(output.telemetry["moe_dropped_assignments"]) == 0
    selection_counts = output.telemetry["expert_selection_counts"]
    assert selection_counts.shape == (
        config.n_layers,
        config.n_routed_experts,
    )
    assert int(selection_counts.sum()) == int(output.telemetry["moe_assignments"])
    assert int(output.telemetry["overflow_drop_tokens"]) == 0
    assert "expert_load_cv" in output.telemetry
    assert 0.0 <= float(output.telemetry["expert_entropy_ratio"].detach()) <= 1.0
    assert float(output.telemetry["halt_collapse_fraction"]) == 0.0
    assert "all_to_all_bytes" in output.telemetry
    assert "all_to_all_seconds" in output.telemetry
    (output.loss + output.auxiliary_loss).backward()
    assert model.embedding.weight.grad is not None
    assert torch.isfinite(model.embedding.weight.grad).all()


def test_aux_loss_free_selection_bias_update_is_layerwise_and_checkpointed() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    counts = torch.ones(config.n_layers, config.n_routed_experts, dtype=torch.long)
    counts[:, 0] = 100

    model.update_expert_selection_biases(counts)

    for layer in model.layers:
        assert float(layer.moe.selection_bias[0]) < 0.0
        assert torch.all(layer.moe.selection_bias[1:] > 0.0)
    state = model.state_dict()
    assert "layers.0.moe.selection_bias" in state
    torch.testing.assert_close(
        state["layers.0.moe.selection_bias"],
        model.layers[0].moe.selection_bias,
    )


def test_selection_bias_changes_only_expert_ids_not_combine_affinities() -> None:
    config = Metis16Config.tiny_for_tests()
    moe = Metis16ForCausalLM(config, dtype=torch.float32).layers[0].moe
    logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    chosen_k = torch.tensor([[2]])
    active = torch.ones((1, 1), dtype=torch.bool)

    base_indices, base_weights, _ = moe._select_experts(logits, chosen_k, active)
    moe.selection_bias.copy_(torch.tensor([0.0, 0.9, 0.0, 0.0]))
    same_indices, same_weights, _ = moe._select_experts(logits, chosen_k, active)

    assert torch.equal(base_indices[..., :2], same_indices[..., :2])
    torch.testing.assert_close(base_weights, same_weights)

    moe.selection_bias.copy_(torch.tensor([0.0, 0.0, 3.0, 0.0]))
    changed_indices, changed_weights, _ = moe._select_experts(
        logits, chosen_k, active
    )
    assert set(changed_indices[0, 0, :2].tolist()) == {0, 2}
    gathered_unbiased = logits.gather(-1, changed_indices)
    expected = torch.softmax(gathered_unbiased[..., :2], dim=-1)
    torch.testing.assert_close(changed_weights[..., :2], expected)


def test_forced_depth_is_exact_and_active_sets_are_monotonic() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    input_ids, labels, _reset_mask = _batch(config)
    forced_depth = torch.tensor(
        [
            [1, 2, 3, 1, 2, 3, 1, 2],
            [3, 2, 1, 3, 2, 1, 3, 2],
        ]
    )
    output = model(
        input_ids,
        labels,
        force_depth=forced_depth,
        curriculum=_fixed_curriculum(2),
    )
    assert torch.equal(output.chosen_depths, forced_depth)
    assert torch.all(output.active_masks[1:] <= output.active_masks[:-1])
    reconstructed = output.active_masks.sum(dim=0)
    assert torch.equal(reconstructed, forced_depth)
    assert output.telemetry["mean_passes"] == pytest.approx(2.0)
    assert int(output.telemetry["executed_active_tokens"]) == int(forced_depth.sum())
    assert int(output.telemetry["dense_pass_fallback_tokens"]) == 0
    assert int(output.telemetry["packed_continuation_enabled"]) == 1


def test_bucketed_active_tokens_preserve_the_exact_model_result() -> None:
    base = replace(
        Metis16Config.tiny_for_tests(),
        activation_recompute_policy="pass",
    )
    exact = Metis16ForCausalLM(base, dtype=torch.float32).train()
    bucketed = Metis16ForCausalLM(
        replace(base, active_token_bucket_shift=0),
        dtype=torch.float32,
    ).train()
    bucketed.load_state_dict(exact.state_dict())
    input_ids, labels, reset_mask = _batch(base)
    forced_depth = torch.tensor(
        [
            [1, 3, 1, 2, 1, 3, 1, 2],
            [2, 1, 3, 1, 2, 1, 3, 1],
        ]
    )
    kwargs = {
        "labels": labels,
        "reset_mask": reset_mask,
        "force_depth": forced_depth,
        "curriculum": _fixed_curriculum(2),
        "return_logits": False,
    }
    reference = exact(input_ids, **kwargs)
    observed = bucketed(input_ids, **kwargs)
    torch.testing.assert_close(observed.loss, reference.loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        observed.final_hidden_state,
        reference.final_hidden_state,
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.equal(observed.chosen_depths, reference.chosen_depths)
    assert int(observed.telemetry["packed_continuation_padding_tokens"]) > 0
    (reference.loss + reference.auxiliary_loss).backward()
    (observed.loss + observed.auxiliary_loss).backward()
    for (left_name, left), (right_name, right) in zip(
        exact.named_parameters(),
        bucketed.named_parameters(),
        strict=True,
    ):
        assert left_name == right_name
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None, left_name
            continue
        torch.testing.assert_close(
            right.grad,
            left.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda message, name=left_name: f"{name}: {message}",
        )


def test_chunked_loss_matches_explicit_logits_and_uses_aligned_labels() -> None:
    config = replace(Metis16Config.tiny_for_tests(), lm_head_chunk_size=3)
    model = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    input_ids, labels, _reset_mask = _batch(config)
    kwargs = {
        "force_depth": 1,
        "curriculum": _fixed_curriculum(1, target_depth=1.0),
    }
    explicit = model(input_ids, labels, return_logits=True, **kwargs)
    chunked = model(input_ids, labels, return_logits=False, **kwargs)
    assert explicit.logits is not None
    assert chunked.logits is None
    torch.testing.assert_close(chunked.loss, explicit.loss)

    hand_logits = torch.full((1, 3, 4), -20.0)
    aligned = torch.tensor([[2, 1, -100]])
    hand_logits[0, 0, 2] = 20.0
    hand_logits[0, 1, 1] = 20.0
    assert float(model._causal_loss(hand_logits, aligned)) < 1.0e-6


def test_all_ignored_labels_produce_finite_graph_connected_zero() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    input_ids, labels, _reset_mask = _batch(config)
    labels.fill_(-100)
    output = model(
        input_ids,
        labels,
        force_depth=1,
        curriculum=_fixed_curriculum(1, target_depth=1.0),
    )
    assert output.loss is not None
    assert output.loss.item() == 0.0
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.embedding.weight.grad is not None


def test_document_reset_prevents_cross_document_state_leakage() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    suffix = torch.tensor([[7, 8, 9, 10]])
    first = torch.cat((torch.tensor([[1, 2, 3, 4]]), suffix), dim=1)
    second = torch.cat((torch.tensor([[20, 21, 22, 23]]), suffix), dim=1)
    reset = torch.zeros_like(first, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 4] = True
    kwargs = {
        "reset_mask": reset,
        "force_depth": 1,
        "curriculum": _fixed_curriculum(1, target_depth=1.0),
        "return_logits": False,
    }
    first_output = model(first, **kwargs).final_hidden_state[:, 4:]
    second_output = model(second, **kwargs).final_hidden_state[:, 4:]
    torch.testing.assert_close(first_output, second_output, rtol=1.0e-5, atol=1.0e-5)


def test_packed_later_passes_preserve_document_resets_across_active_gaps() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    suffix = torch.tensor([[7, 8, 9, 10]])
    first = torch.cat((torch.tensor([[1, 2, 3, 4]]), suffix), dim=1)
    second = torch.cat((torch.tensor([[20, 21, 22, 23]]), suffix), dim=1)
    reset = torch.zeros_like(first, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 4] = True
    forced_depth = torch.tensor([[1, 2, 1, 2, 2, 3, 2, 3]])
    kwargs = {
        "reset_mask": reset,
        "force_depth": forced_depth,
        "curriculum": _fixed_curriculum(2),
        "return_logits": False,
    }
    first_output = model(first, **kwargs)
    second_output = model(second, **kwargs)
    assert int(first_output.telemetry["packed_continuation_passes"]) == 2
    torch.testing.assert_close(
        first_output.final_hidden_state[:, 4:],
        second_output.final_hidden_state[:, 4:],
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_row_sharded_table_placement_is_explicit() -> None:
    base = Metis16Config.tiny_for_tests(table_mode="row_sharded")
    model = Metis16ForCausalLM(base, dtype=torch.float32)
    table_placements = {
        placement
        for name, placement in model.parameter_placements().items()
        if name.startswith("ngram_memory.tables.")
    }
    assert table_placements == {PLACEMENT_ROW_SHARDED_TABLE}


def test_sinkhorn_projection_is_doubly_stochastic() -> None:
    logits = torch.randn(4, 4, generator=torch.Generator().manual_seed(7))
    matrix = sinkhorn_doubly_stochastic(logits, iterations=32)
    torch.testing.assert_close(matrix.sum(dim=0), torch.ones(4), atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(4), atol=1.0e-5, rtol=1.0e-5)


class _RecordingPrecisionPolicy:
    def __init__(self) -> None:
        self.roles: list[str] = []
        self.context_entries = 0

    def linear(self, in_features: int, out_features: int, *, bias: bool, role: str, **kwargs):
        self.roles.append(role)
        return nn.Linear(in_features, out_features, bias=bias, **kwargs)

    def execution_context(self):
        self.context_entries += 1
        return nullcontext()

    def is_fp8_role(self, role: str) -> bool:
        return False


class _StrictRepeatedModulePrecisionPolicy:
    def __init__(self) -> None:
        self.active_region: int | None = None
        self.next_region = 0
        self.region_modules: dict[int, set[int]] = {}

    def linear(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        role: str,
        **kwargs,
    ):
        policy = self

        class StrictLinear(nn.Linear):
            def forward(self, values):
                if policy.active_region is None:
                    raise AssertionError(f"{role} executed outside a precision region")
                seen = policy.region_modules[policy.active_region]
                if id(self) in seen:
                    raise AssertionError(
                        f"{role} executed twice in one precision region"
                    )
                seen.add(id(self))
                return super().forward(values)

        return StrictLinear(
            in_features,
            out_features,
            bias=bias,
            **kwargs,
        )

    @contextmanager
    def execution_context(self):
        if self.active_region is not None:
            raise AssertionError("precision regions must not be nested")
        region = self.next_region
        self.next_region += 1
        self.active_region = region
        self.region_modules[region] = set()
        try:
            yield
        finally:
            self.active_region = None

    def is_fp8_role(self, _role: str) -> bool:
        return False


def test_precision_policy_factory_and_execution_context_are_used() -> None:
    policy = _RecordingPrecisionPolicy()
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, precision_policy=policy, dtype=torch.float32)
    input_ids, labels, _reset_mask = _batch(config, batch=1)
    model(
        input_ids,
        labels,
        force_depth=1,
        curriculum=_fixed_curriculum(1, target_depth=1.0),
    )
    assert {
        "mamba_in_projection",
        "mamba_out_projection",
        "attention_qkv_projection",
        "attention_out_projection",
        "attention_pass_lora_down",
        "attention_pass_lora_up",
        "expert_gate_up_projection",
        "expert_down_projection",
        "latent_down_projection",
        "latent_up_projection",
        "memory_state_write_projection",
        "memory_metadata_write_projection",
        "memory_query_projection",
        "memory_key_projection",
        "memory_value_projection",
        "memory_output_projection",
        "memory_route_projection",
        "mhc_controller",
        "ngram_projection",
        "lm_head",
    } <= set(policy.roles)
    assert policy.context_entries > 0


def test_recursive_passes_and_lm_chunks_use_distinct_precision_regions() -> None:
    policy = _StrictRepeatedModulePrecisionPolicy()
    config = replace(
        Metis16Config.tiny_for_tests(),
        lm_head_chunk_size=3,
    )
    model = Metis16ForCausalLM(
        config,
        precision_policy=policy,
        dtype=torch.float32,
    )
    input_ids, labels, _reset_mask = _batch(config, batch=1)
    output = model(
        input_ids,
        labels,
        force_depth=3,
        curriculum=_fixed_curriculum(2),
    )
    (output.loss + output.auxiliary_loss).backward()
    assert policy.next_region > config.max_passes * config.n_layers


def test_empty_act_rank_executes_world_aligned_lm_head_dummy_chunks() -> None:
    policy = _StrictRepeatedModulePrecisionPolicy()
    config = replace(Metis16Config.tiny_for_tests(), lm_head_chunk_size=3)
    model = Metis16ForCausalLM(
        config,
        precision_policy=policy,
        dtype=torch.float32,
    )
    world = mock.sentinel.world
    model.process_groups = MetisProcessGroups(world=world)
    hidden = torch.randn(1, 8, config.d_model, requires_grad=True)
    labels = torch.full((1, 8), -100, dtype=torch.long)
    weights = torch.zeros((1, 8))
    compute_mask = torch.zeros((1, 8), dtype=torch.bool)
    before = policy.next_region

    def all_reduce(value, *, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is world
        value.fill_(5)

    with (
        mock.patch.object(
            model_module,
            "_group_world_size",
            side_effect=lambda group=None: 2 if group is world else 1,
        ),
        mock.patch.object(model_module.dist, "all_reduce", side_effect=all_reduce),
    ):
        loss = model._chunked_weighted_causal_loss_sum(
            hidden,
            labels,
            weights,
            compute_mask=compute_mask,
        )
    loss.backward()
    assert float(loss.detach()) == 0.0
    # Two aligned LM-head chunks execute once in the forward and once during
    # non-reentrant activation recomputation.
    assert policy.next_region - before == 4


def test_storage_policy_keeps_router_and_mamba_sensitive_state_fp32() -> None:
    model = Metis16ForCausalLM(Metis16Config.tiny_for_tests(), dtype=torch.float32)
    model.apply_parameter_storage_policy(torch.device("cpu"))
    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.layers[0].moe.expert_router.weight.dtype == torch.float32
    assert model.layers[0].moe.k_router.weight.dtype == torch.float32
    assert model.continuation.hidden.weight.dtype == torch.float32
    reference_mamba = model.layers[0].mixer.impl
    assert reference_mamba.A_log.dtype == torch.float32
    assert reference_mamba.dt_bias.dtype == torch.float32


def test_dispatch_overlap_setter_reaches_every_moe() -> None:
    model = Metis16ForCausalLM(Metis16Config.tiny_for_tests(), dtype=torch.float32)
    model.set_dispatch_overlap(True)
    assert model.dispatch_overlap_enabled
    assert all(layer.moe.dispatch_overlap for layer in model.layers)
    model.set_dispatch_overlap(False)
    assert not model.dispatch_overlap_enabled
    assert all(not layer.moe.dispatch_overlap for layer in model.layers)


def test_expert_initialization_is_keyed_by_global_identity() -> None:
    config = Metis16Config.tiny_for_tests()
    torch.manual_seed(1)
    first = Metis16ForCausalLM(config, dtype=torch.float32)
    torch.manual_seed(999)
    second = Metis16ForCausalLM(config, dtype=torch.float32)
    first_expert = first.layers[0].moe.local_experts[0].gate_up.weight
    matching_copy = second.layers[0].moe.local_experts[0].gate_up.weight
    other_expert = first.layers[0].moe.local_experts[1].gate_up.weight
    torch.testing.assert_close(first_expert, matching_copy)
    assert not torch.equal(first_expert, other_expert)


def test_pass_recompute_matches_forward_and_backward() -> None:
    config = Metis16Config.tiny_for_tests()
    torch.manual_seed(77)
    reference = Metis16ForCausalLM(config, dtype=torch.float32)
    recomputed = Metis16ForCausalLM(config, dtype=torch.float32)
    recomputed.load_state_dict(reference.state_dict())
    recomputed.set_activation_recompute_policy("pass")
    input_ids, labels, _reset_mask = _batch(config, batch=1, length=6)
    forced_depth = torch.tensor([[1, 2, 3, 1, 2, 3]])
    kwargs = {
        "labels": labels,
        "force_depth": forced_depth,
        "curriculum": _fixed_curriculum(2),
        "return_logits": True,
    }
    reference_output = reference(input_ids, **kwargs)
    recomputed_output = recomputed(input_ids, **kwargs)
    torch.testing.assert_close(recomputed_output.loss, reference_output.loss)
    torch.testing.assert_close(
        recomputed_output.final_hidden_state,
        reference_output.final_hidden_state,
    )
    torch.testing.assert_close(
        recomputed_output.telemetry["expert_selection_counts"],
        reference_output.telemetry["expert_selection_counts"],
    )
    assert int(reference_output.telemetry["activation_recompute_enabled"]) == 0
    assert int(recomputed_output.telemetry["activation_recompute_enabled"]) == 1

    reference_output.loss.backward()
    recomputed_output.loss.backward()
    for parameter_name in (
        "embedding.weight",
        "layers.0.moe.k_router.weight",
        "continuation.output.weight",
    ):
        reference_gradient = dict(reference.named_parameters())[parameter_name].grad
        recomputed_gradient = dict(recomputed.named_parameters())[parameter_name].grad
        assert reference_gradient is not None
        assert recomputed_gradient is not None
        torch.testing.assert_close(recomputed_gradient, reference_gradient)

    recomputed.eval()
    with torch.no_grad():
        evaluation = recomputed(
            input_ids,
            labels,
            force_depth=2,
            curriculum=_fixed_curriculum(2),
        )
    assert int(evaluation.telemetry["activation_recompute_enabled"]) == 0


def test_hard_continuation_physically_compacts_mixer_tokens() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    input_ids, labels, _reset_mask = _batch(config, batch=2, length=6)
    forced_depth = torch.tensor(
        [
            [1, 2, 3, 1, 2, 3],
            [3, 2, 1, 3, 2, 1],
        ]
    )
    mixer_token_counts: list[int] = []

    def record_shape(_module, args, _kwargs) -> None:
        hidden_states = args[0]
        mixer_token_counts.append(hidden_states.shape[0] * hidden_states.shape[1])

    handle = model.layers[0].mixer.register_forward_pre_hook(
        record_shape,
        with_kwargs=True,
    )
    try:
        output = model(
            input_ids,
            labels,
            force_depth=forced_depth,
            curriculum=_fixed_curriculum(2),
            return_logits=True,
        )
    finally:
        handle.remove()

    assert mixer_token_counts == [12, 8, 4]
    assert int(output.telemetry["executed_active_tokens"]) == 24
    assert int(output.telemetry["dense_envelope_tokens"]) == 36
    assert int(output.telemetry["packed_continuation_passes"]) == 2
    assert float(output.telemetry["ponder_exit_mass_error"]) == 0.0
    assert float(output.telemetry["ponder_exit_mass_max_error"]) == 0.0
    assert output.logits is not None
    torch.testing.assert_close(
        output.loss,
        model._causal_loss(output.logits, labels),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_causal_loss_credits_k_and_continuation_routers() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    input_ids, labels, _reset_mask = _batch(config, batch=1, length=6)
    output = model(
        input_ids,
        labels,
        force_depth=2,
        curriculum=_fixed_curriculum(2),
    )
    assert output.loss is not None
    output.loss.backward()

    k_gradients = [
        layer.moe.k_router.weight.grad
        for layer in model.layers
    ]
    assert all(gradient is not None for gradient in k_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in k_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in k_gradients) > 0.0
    continuation_gradient = model.continuation.output.weight.grad
    assert continuation_gradient is not None
    assert torch.isfinite(continuation_gradient).all()
    assert float(continuation_gradient.abs().sum()) > 0.0


def test_token_difficulty_priors_increase_width_and_depth() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    moe = model.layers[0].moe
    hidden = torch.zeros(1, 3, config.d_model)
    route_features = torch.zeros(1, 3, config.route_feature_dim)
    active = torch.ones(1, 3, dtype=torch.bool)
    curriculum = CurriculumState(
        routed_k_mode="adaptive",
        stochastic_routing=False,
    )
    with torch.no_grad():
        moe.k_router.weight.zero_()
        moe.k_router.bias.zero_()
        moe.expert_router.weight.zero_()
        moe.expert_router.bias.zero_()
        _, uncertain = moe(
            hidden,
            route_features=route_features,
            active_mask=active,
            curriculum=curriculum,
        )
        moe.expert_router.bias.fill_(-10.0)
        moe.expert_router.bias[0] = 10.0
        _, confident = moe(
            hidden,
            route_features=route_features,
            active_mask=active,
            curriculum=curriculum,
        )
    assert float(uncertain.token_difficulty.mean()) > float(
        confident.token_difficulty.mean()
    )
    assert float(uncertain.expected_k.mean()) > float(confident.expected_k.mean())

    controller = model.continuation
    with torch.no_grad():
        controller.hidden.weight.zero_()
        controller.hidden.bias.zero_()
        controller.output.weight.zero_()
        controller.output.bias.zero_()
        state = torch.ones(1, 3, config.d_model)
        memory = torch.zeros_like(state)
        route = torch.zeros(1, 3, config.route_feature_dim)
        easy = controller(state, memory, torch.zeros_like(state), route)
        hard = controller(state, memory, 4.0 * torch.ones_like(state), route)
    assert torch.all(hard > easy)


def test_adaptive_halt_collapse_telemetry_detects_single_depth() -> None:
    config = Metis16Config.tiny_for_tests()
    model = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    with torch.no_grad():
        model.continuation.output.weight.zero_()
        model.continuation.output.bias.fill_(-100.0)
    input_ids, labels, _reset_mask = _batch(config, batch=1, length=6)
    output = model(
        input_ids,
        labels,
        curriculum=CurriculumState(
            routed_k_mode="fixed",
            fixed_routed_k=2,
            stochastic_routing=False,
            continuation_mode="adaptive",
        ),
    )
    assert torch.equal(output.chosen_depths, torch.ones_like(input_ids))
    assert float(output.telemetry["halt_collapse_fraction"]) == 1.0
    assert int(output.telemetry["executed_active_tokens"]) == input_ids.numel()


def test_zero_local_active_rank_keeps_ep_pass_sequence_without_dummy_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Metis16Config.tiny_for_tests()
    torch.manual_seed(91)
    reference = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    sentinel = Metis16ForCausalLM(config, dtype=torch.float32).eval()
    sentinel.load_state_dict(reference.state_dict())
    input_ids, labels, _reset_mask = _batch(config, batch=1, length=6)
    kwargs = {
        "labels": labels,
        "force_depth": 1,
        "curriculum": _fixed_curriculum(2, target_depth=1.0),
        "return_logits": False,
    }
    reference_output = reference(input_ids, **kwargs)

    active_counts: list[int] = []

    def record_active(_module, _args, call_kwargs) -> None:
        active_counts.append(int(call_kwargs["active_mask"].sum()))

    handle = sentinel.layers[0].moe.register_forward_pre_hook(
        record_active,
        with_kwargs=True,
    )
    monkeypatch.setattr(
        model_module,
        "_group_any_active",
        lambda _active_mask, *, group, groups=(): True,
    )
    try:
        sentinel_output = sentinel(input_ids, **kwargs)
    finally:
        handle.remove()

    assert active_counts == [input_ids.numel(), 0, 0]
    assert int(sentinel_output.telemetry["executed_active_tokens"]) == input_ids.numel()
    assert int(sentinel_output.telemetry["dense_pass_fallback_tokens"]) == 0
    assert float(sentinel_output.telemetry["ponder_exit_mass_max_error"]) == 0.0
    torch.testing.assert_close(
        sentinel_output.final_hidden_state,
        reference_output.final_hidden_state,
    )
    torch.testing.assert_close(sentinel_output.loss, reference_output.loss)


def _loopback_all_to_all(monkeypatch: pytest.MonkeyPatch, world_size: int = 2) -> None:
    """Run collective code paths in-process against a split-preserving loopback."""

    monkeypatch.setattr(
        model_module,
        "_group_world_size",
        lambda group=None: world_size,
    )

    def all_to_all_single(output, values, output_split_sizes=None, input_split_sizes=None, **_):
        if list(output_split_sizes or []) != list(input_split_sizes or []):
            raise AssertionError("loopback requires symmetric splits")
        output.copy_(values)

    monkeypatch.setattr(model_module.dist, "all_to_all_single", all_to_all_single)


def test_fp8_expert_wire_round_trips_through_the_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _loopback_all_to_all(monkeypatch)
    torch.manual_seed(3)
    # Row scales must survive a four-order-of-magnitude spread across rows,
    # which is exactly what a per-tensor scale would destroy.
    values = (
        torch.randn(64, 1024, dtype=torch.float32)
        * torch.logspace(-2, 2, 64).unsqueeze(-1)
    ).bfloat16().requires_grad_(True)
    splits = [32, 32]

    received = model_module._variable_all_to_all(
        values,
        input_splits=splits,
        output_splits=splits,
        group=object(),
        wire="fp8",
    )
    assert received.dtype == values.dtype
    relative = (
        (received.float() - values.float()).norm(dim=-1)
        / values.float().norm(dim=-1).clamp_min(1e-12)
    )
    # E4M3 carries three mantissa bits, so ~3% per row is the floor, not a bug.
    assert float(relative.detach().amax()) < 0.05

    received.float().square().sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_bf16_profile_forces_a_bf16_expert_wire() -> None:
    config = Metis16Config.tiny_for_tests()
    config = replace(
        config,
        precision=replace(config.precision, expert_collective_wire="fp8"),
    )

    class _Policy:
        fp8_enabled = False

        def linear(self, in_features, out_features, bias=True, *, role, **kwargs):
            del role
            return nn.Linear(in_features, out_features, bias=bias, **kwargs)

    fp8_model = Metis16ForCausalLM(config, dtype=torch.float32)
    assert fp8_model.layers[0].moe.dispatch_wire == "fp8"
    assert fp8_model.layers[0].moe.combine_wire == "fp8"

    bf16_model = Metis16ForCausalLM(config, precision_policy=_Policy(), dtype=torch.float32)
    assert bf16_model.layers[0].moe.dispatch_wire == "bfloat16"
    assert bf16_model.layers[0].moe.combine_wire == "bfloat16"


def test_fp8_wire_probe_error_stays_inside_the_bringup_gate() -> None:
    torch.manual_seed(11)
    values = torch.randn(512, 1024, dtype=torch.bfloat16)
    assert model_module.expert_collective_wire_error(values) < 0.08
    assert model_module.expert_collective_wire_error(values, gradient=True) < 0.15


@pytest.mark.parametrize("family", ["praxis", "logos"])
def test_locked_manifests_select_the_low_traffic_collective_plan(family: str) -> None:
    config = load_family_config(family)
    # Replicated tables all-gather every touched row across the whole family on
    # every backward; row sharding replaces that with a per-row exchange.
    assert config.ngram_memory.table_mode == "row_sharded"
    assert config.precision.expert_collective_wire == "fp8"
    assert config.autotune.ngram_table_modes[0] == "row_sharded"


class _StubWork:
    """Loopback stand-in for a torch.distributed async work handle."""

    def __init__(self, apply) -> None:
        self._apply = apply
        self.waited = False

    def wait(self) -> None:
        self._apply()
        self.waited = True


def _loopback_collectives(
    monkeypatch: pytest.MonkeyPatch,
    *,
    world_size: int = 2,
    trace: list[str] | None = None,
) -> None:
    """World-size-N loopback that preserves split structure and async semantics.

    Every rank in a single-process loopback is this rank, so a symmetric split
    plan copies straight through.  Deferring the copy until ``wait`` is what
    catches a pipeline that reads a buffer before its transfer completes.
    """

    monkeypatch.setattr(model_module, "_group_world_size", lambda group=None: world_size)

    def all_to_all_single(
        output, values, output_split_sizes=None, input_split_sizes=None,
        group=None, async_op=False,
    ):
        if list(output_split_sizes or []) != list(input_split_sizes or []):
            raise AssertionError("loopback requires symmetric splits")
        if trace is not None:
            trace.append(f"a2a{tuple(values.shape)}")
        copy = values.detach().clone()

        def apply() -> None:
            output.copy_(copy)

        if async_op:
            return _StubWork(apply)
        apply()
        return None

    monkeypatch.setattr(model_module.dist, "all_to_all_single", all_to_all_single)


def _tiny_expert_parallel(chunks: int, *, expert_parallel_size: int = 4):
    """Tiny config with a real EP group, so the dispatch path is not skipped."""

    config = Metis16Config.tiny_for_tests()
    return replace(
        config,
        moe_dispatch_chunks=chunks,
        expert_parallel_size=expert_parallel_size,
        world_size=expert_parallel_size,
    )


def _moe_dispatch_case(config, chunks: int, trace: list[str] | None = None):
    torch.manual_seed(17)
    config = replace(config, moe_dispatch_chunks=chunks)
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    moe = model.layers[0].moe
    experts = 12
    latent = config.latent_dim
    hidden = torch.randn(experts, latent, dtype=torch.float32, requires_grad=True)
    expert_ids = torch.tensor([0, 3, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2])
    expert_ids = expert_ids % config.n_routed_experts
    weights = torch.rand(experts)
    token_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    combined, processed, wire_bytes, _elapsed = moe._dispatch(
        hidden, expert_ids, weights, token_indices, 6
    )
    combined.square().sum().backward()
    return combined, processed, wire_bytes, hidden.grad, moe


def test_pipelined_dispatch_matches_the_serial_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_expert_parallel(1)
    trace: list[str] = []
    _loopback_collectives(
        monkeypatch,
        world_size=config.expert_parallel_size,
        trace=trace,
    )

    serial, serial_processed, serial_bytes, serial_grad, _ = _moe_dispatch_case(config, 1)
    serial_collectives = len(trace)
    assert serial_collectives > 0, "serial dispatch never reached the collective path"
    for chunks in (2, 3, 4):
        trace.clear()
        piped, processed, wire_bytes, grad, moe = _moe_dispatch_case(config, chunks)
        assert moe.dispatch_chunks == chunks
        assert len(trace) > serial_collectives, "pipeline did not chunk its transfers"
        torch.testing.assert_close(piped, serial)
        torch.testing.assert_close(grad, serial_grad)
        # Telemetry must not drift with the pipeline depth: the same rows are
        # processed and the same bytes cross the wire, just at different times.
        assert int(processed) == int(serial_processed)
        assert int(wire_bytes) == int(serial_bytes)


def test_pipelined_dispatch_issues_a_load_independent_collective_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Divergent collective counts across ranks would deadlock the EP group."""

    config = _tiny_expert_parallel(1)
    counts = []
    for seed_shift in range(3):
        trace: list[str] = []
        _loopback_collectives(
            monkeypatch,
            world_size=config.expert_parallel_size,
            trace=trace,
        )
        torch.manual_seed(101 + seed_shift)
        chunked = replace(config, moe_dispatch_chunks=3)
        moe = Metis16ForCausalLM(chunked, dtype=torch.float32).layers[0].moe
        rows = 9 + seed_shift * 5
        hidden = torch.randn(rows, config.latent_dim)
        # Deliberately lopsided routing, including a destination with no rows.
        expert_ids = torch.zeros(rows, dtype=torch.long)
        expert_ids[: rows // 2] = 1 % config.n_routed_experts
        moe._dispatch(
            hidden,
            expert_ids,
            torch.rand(rows),
            torch.arange(rows) % 4,
            4,
        )
        counts.append(len(trace))
    assert len(set(counts)) == 1, f"collective count varies with local load: {counts}"


def test_pipeline_defers_reads_until_its_transfers_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A carrier read before its wait would silently train on empty buffers."""

    config = _tiny_expert_parallel(1)
    _loopback_collectives(monkeypatch, world_size=config.expert_parallel_size)
    combined, _processed, _bytes, grad, _moe = _moe_dispatch_case(config, 3)
    assert torch.isfinite(combined).all() and combined.abs().sum() > 0
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Overlapped gradient reduction
# ---------------------------------------------------------------------------


def _reducer_topology(world_size: int, replicas: int = 1):
    from metis_training.distributed import ParallelTopology

    return ParallelTopology(
        family="tiny",
        world_size=world_size,
        rank=0,
        local_rank=0,
        expert_parallel_size=world_size // replicas,
        expert_replica_count=replicas,
        expert_group=object(),
        expert_group_ranks=tuple(range(world_size // replicas)),
        expert_data_group=object() if replicas > 1 else None,
        expert_data_group_ranks=tuple(range(replicas)),
        dense_data_group=object(),
    )


def _loopback_all_reduce(monkeypatch: pytest.MonkeyPatch, order: list[int] | None = None):
    """Sum-preserving single-process all_reduce that records issue order."""

    import metis_training.distributed as distributed_module

    def all_reduce(tensor, group=None, async_op=False):
        if order is not None:
            order.append(int(tensor.numel()))
        # One rank looping back: reducing across a group of identical ranks
        # multiplies by the group size, which the divisor then undoes.
        tensor.mul_(_LOOPBACK_RANKS)
        if async_op:
            return _StubWork(lambda: None)
        return None

    monkeypatch.setattr(distributed_module.dist, "all_reduce", all_reduce)


_LOOPBACK_RANKS = 4


def _tiny_grads(seed: int = 5):
    config = Metis16Config.tiny_for_tests(table_mode="row_sharded")
    torch.manual_seed(seed)
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    input_ids, labels, _reset = _batch(config, batch=2, length=8)
    model(
        input_ids,
        labels=labels,
        return_logits=False,
        curriculum=_fixed_curriculum(2),
    ).loss.backward()
    return model


def test_overlapped_reduction_matches_the_synchronous_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metis_training.distributed import (
        OverlappedGradientReducer,
        synchronize_gradients,
    )

    topology = _reducer_topology(_LOOPBACK_RANKS)
    _loopback_all_reduce(monkeypatch)

    reference = _tiny_grads()
    synchronize_gradients(reference, topology)
    expected = {name: p.grad.clone() for name, p in reference.named_parameters()}

    overlapped = _tiny_grads()
    reducer = OverlappedGradientReducer(overlapped, topology)
    reducer.arm()
    # Gradients already exist, so nothing fired from a hook; finalize must
    # still flush every bucket in order and produce the identical result.
    synchronize_gradients(overlapped, topology, reducer=reducer)
    for name, parameter in overlapped.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], msg=name)


def test_reducer_fires_during_backward_and_only_when_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metis_training.distributed import OverlappedGradientReducer

    config = Metis16Config.tiny_for_tests(table_mode="row_sharded")
    topology = _reducer_topology(_LOOPBACK_RANKS)
    issued: list[int] = []
    _loopback_all_reduce(monkeypatch, order=issued)

    torch.manual_seed(5)
    model = Metis16ForCausalLM(config, dtype=torch.float32)
    reducer = OverlappedGradientReducer(model, topology)
    input_ids, labels, _reset = _batch(config, batch=2, length=8)

    # An unarmed backward is an accumulation micro-step: nothing may reduce.
    kwargs = {
        "labels": labels,
        "return_logits": False,
        "curriculum": _fixed_curriculum(2),
    }
    model(input_ids, **kwargs).loss.backward()
    assert issued == []

    reducer.arm()
    model(input_ids, **kwargs).loss.backward()
    assert issued, "armed backward issued no reduction before finalize"
    reducer.finalize()


def test_reducer_issue_order_is_fixed_regardless_of_gradient_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranks that issue buckets in different orders deadlock RCCL."""

    from metis_training.distributed import OverlappedGradientReducer

    topology = _reducer_topology(_LOOPBACK_RANKS)
    orders = []
    for seed in (5, 11, 23):
        issued: list[int] = []
        _loopback_all_reduce(monkeypatch, order=issued)
        model = _tiny_grads(seed)
        reducer = OverlappedGradientReducer(model, topology)
        reducer.arm()
        reducer.finalize()
        orders.append(issued)
    assert orders[0] == orders[1] == orders[2], f"bucket order drifted: {orders}"


def test_reducer_flushes_buckets_whose_parameters_never_got_a_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing can leave a parameter gradient-free; the bucket must still fire."""

    from metis_training.distributed import OverlappedGradientReducer

    topology = _reducer_topology(_LOOPBACK_RANKS, replicas=2)
    issued: list[int] = []
    _loopback_all_reduce(monkeypatch, order=issued)

    model = _tiny_grads()
    reducer = OverlappedGradientReducer(model, topology)
    total_buckets = len(reducer._buckets)
    reducer.arm()
    assert reducer.finalize() is True
    assert len(issued) == total_buckets
    # N-gram tables are sparse or row-owned and deliberately sit outside the
    # dense reducer, so only reducer-managed placements are guaranteed a
    # materialised gradient here.
    placements = model.parameter_placements()
    managed = {"replicated", "expert_sharded"}
    for name, parameter in model.named_parameters():
        if placements[name] in managed:
            assert parameter.grad is not None, name


def test_production_topology_constants_agree_across_the_repository() -> None:
    """distributed._production_shape duplicates the manifests; keep them equal."""

    from metis_training.distributed import _production_shape

    for family in ("praxis", "logos"):
        config = load_family_config(family)
        assert _production_shape(family) == (
            config.world_size,
            config.expert_parallel_size,
            config.expert_replicas,
        )
