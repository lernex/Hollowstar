from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

import metis_training.train as train_module
from metis_data.ngram_canonical import (
    CANONICAL_IDS_BINARY,
    CANONICAL_IDS_MANIFEST,
    build_canonical_id_sidecar,
    canonicalize_decoded_token,
    validate_canonical_id_sidecar,
)
from metis_training.checkpointing import CheckpointManager
from metis_training.contracts import (
    AutotuneSelection,
    canonical_json_sha256,
    sha256_file,
)
from metis_training.data import (
    TOTAL_TOKENS,
    DeterministicReleaseStream,
    ReleaseInventory,
    ReleaseShard,
    TrainingBatch,
)
from metis_training.distributed import ParallelTopology, Runtime
from metis_training.metrics import (
    StepMetrics,
    enforce_health_gates,
    peak_memory_evidence,
)
from metis_training.model_config import PrecisionConfig
from metis_training.optimizers import (
    FP32MasterSparseAdam,
    OptimizerBundle,
    clip_grad_norm_,
)
from metis_training.precision import PrecisionPolicy
from metis_training.schedule import TokenSchedule
from metis_training.train import _restore_if_requested, _run_steps


def _runtime_manifest() -> dict:
    return {
        "schema": "metis.training-runtime/v1",
        "schedule": {
            "token_axis": "globally_emitted_non_padding_tokens",
            "phases": [
                {
                    "id": "phase_a",
                    "start_token": 0,
                    "end_token_exclusive": 700_000_000_000,
                    "warmup_tokens": 2_000_000_000,
                    "end_lr_ratio": 0.55,
                },
                {
                    "id": "phase_b",
                    "start_token": 700_000_000_000,
                    "end_token_exclusive": 950_000_000_000,
                    "start_lr_ratio": 0.55,
                    "end_lr_ratio": 0.20,
                },
                {
                    "id": "phase_c",
                    "start_token": 950_000_000_000,
                    "end_token_exclusive": 1_000_000_000_000,
                    "start_lr_ratio": 0.20,
                    "end_lr_ratio": 0.02,
                },
            ],
            "curriculum": {
                "warm_start_fraction": 0.075,
                "initial_depth": 1,
                "initial_routed_k": 4,
                "initial_memory_gate": 0.0,
                "target_mean_depth": 2.0,
                "target_mean_routed_k": 4.0,
                "ramp_fraction": 0.075,
                "max_passes": 5,
                "routed_k_min": 1,
                "routed_k_max": 8,
            },
        },
    }


def test_token_schedule_is_continuous_and_model_curriculum_is_typed() -> None:
    schedule = TokenSchedule(_runtime_manifest(), base_learning_rate=2.0e-4)
    before = schedule.state(700_000_000_000 - 1)
    after = schedule.state(700_000_000_000)
    assert before.learning_rate == pytest.approx(after.learning_rate, rel=1e-10)
    assert schedule.state(0).force_depth == 1
    ramped = schedule.state(200_000_000_000)
    assert ramped.force_depth is None
    assert set(ramped.model_curriculum()) == {
        "continuation_mode",
        "routed_k_mode",
        "fixed_routed_k",
        "memory_gate_scale",
        "ngram_gate_scale",
        "max_passes",
        "stochastic_routing",
        "temperature",
        "target_mean_depth",
        "target_mean_routed_k",
    }


def test_precision_policy_bf16_never_claims_fp8_on_cpu() -> None:
    config = PrecisionConfig(backend="bf16", require_fp8_validation=False)
    policy = PrecisionPolicy(
        config,
        requested_profile="bf16",
        device=torch.device("cpu"),
        production=False,
    )
    layer = policy.linear(4, 3, role="expert_projection")
    assert isinstance(layer, torch.nn.Linear)
    assert policy.audit.effective_profile == "bf16"
    assert policy.validate_execution()["ok"] is True


def test_sparse_optimizer_uses_fp32_master_and_only_touched_rows() -> None:
    table = torch.nn.Parameter(torch.zeros(5, 3, dtype=torch.bfloat16))
    indices = torch.tensor([[1, 3]])
    values = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]], dtype=torch.bfloat16)
    with torch.sparse.check_sparse_tensor_invariants():
        table.grad = torch.sparse_coo_tensor(
            indices,
            values,
            size=table.shape,
        )
    optimizer = FP32MasterSparseAdam([table], lr=1.0e-2)
    optimizer.step()
    assert table[0].eq(0).all()
    assert table[2].eq(0).all()
    assert not table[1].eq(0).all()
    state = optimizer.state[table]
    assert state["master_param"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


class _PlacedClipModel(torch.nn.Module):
    def __init__(self, *, expert_grad: float, row_grad: float) -> None:
        super().__init__()
        self.replicated = torch.nn.Parameter(torch.ones(2))
        self.expert = torch.nn.Parameter(torch.ones(1))
        self.sparse_table = torch.nn.Parameter(torch.ones(2))
        self.row_table = torch.nn.Parameter(torch.ones(2))
        self.replicated.grad = torch.tensor([3.0, 4.0])
        self.expert.grad = torch.tensor([expert_grad])
        self.sparse_table.grad = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([7.0]),
            size=self.sparse_table.shape,
        )
        self.row_table.grad = torch.sparse_coo_tensor(
            torch.tensor([[1]]),
            torch.tensor([row_grad]),
            size=self.row_table.shape,
        )

    def parameter_placements(self) -> dict[str, str]:
        return {
            "replicated": "replicated",
            "expert": "expert_sharded",
            "sparse_table": "sparse_table",
            "row_table": "row_sharded_table",
        }


def _mock_distributed_topology(rank: int) -> ParallelTopology:
    return ParallelTopology(
        family="logos",
        world_size=4,
        rank=rank,
        local_rank=rank,
        expert_parallel_size=2,
        expert_replica_count=2,
        expert_group=mock.sentinel.expert_group,
        expert_group_ranks=(0, 1) if rank < 2 else (2, 3),
        expert_data_group=mock.sentinel.expert_data_group,
        expert_data_group_ranks=(rank % 2, rank % 2 + 2),
        dense_data_group=mock.sentinel.world_group,
    )


def test_global_clip_counts_logical_placements_once_and_keeps_replicas_equal() -> None:
    # Ranks 0/2 and 1/3 are duplicate expert replicas. Replicated and sparse
    # tensors have four copies. The logical squared norm is therefore:
    # 3^2 + 4^2 + 7^2 + 12^2 + 5^2 + 8^2 + 15^2 = 532.
    expected_global_squared = 532.0
    expected_local_squared = (122.5, 143.5, 122.5, 143.5)
    clipped_replicated_weights: list[torch.Tensor] = []
    observed_norms: list[float] = []

    for rank in range(4):
        expert_grad = 12.0 if rank % 2 == 0 else 5.0
        row_grad = 8.0 if rank % 2 == 0 else 15.0
        model = _PlacedClipModel(expert_grad=expert_grad, row_grad=row_grad)

        def all_reduce(
            value: torch.Tensor,
            *,
            op: object,
            group: object,
        ) -> None:
            assert op == torch.distributed.ReduceOp.SUM
            assert group is mock.sentinel.world_group
            assert float(value.item()) == pytest.approx(expected_local_squared[rank])
            value.fill_(expected_global_squared)

        with mock.patch(
            "metis_training.optimizers.dist.all_reduce",
            side_effect=all_reduce,
        ):
            norm = clip_grad_norm_(
                model,
                10.0,
                topology=_mock_distributed_topology(rank),
            )
        observed_norms.append(float(norm.item()))
        torch.optim.SGD([model.replicated], lr=0.1).step()
        clipped_replicated_weights.append(model.replicated.detach().clone())

    expected_norm = expected_global_squared**0.5
    assert observed_norms == pytest.approx([expected_norm] * 4)
    for weight in clipped_replicated_weights[1:]:
        torch.testing.assert_close(weight, clipped_replicated_weights[0])
    coefficient = 10.0 / (expected_norm + 1.0e-6)
    torch.testing.assert_close(
        clipped_replicated_weights[0],
        torch.ones(2) - 0.1 * torch.tensor([3.0, 4.0]) * coefficient,
    )


class _TinyPlacedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("counter", torch.tensor(7))

    def parameter_placements(self) -> dict[str, str]:
        return {name: "replicated" for name, _ in self.named_parameters()}


class _TinyExtraStateModel(_TinyPlacedModel):
    def __init__(self) -> None:
        super().__init__()
        self.delayed_scaling_state = torch.tensor(
            [3, 1, 4, 1, 5],
            dtype=torch.uint8,
        )
        self.extra_state_loads = 0

    def get_extra_state(self) -> torch.Tensor:
        # Return a detached copy just like TE's serialized uint8 payload. It is
        # deliberately not a live buffer view.
        return self.delayed_scaling_state.detach().clone()

    def set_extra_state(self, state: torch.Tensor) -> None:
        self.delayed_scaling_state = state.detach().clone()
        self.extra_state_loads += 1


def _single_topology() -> ParallelTopology:
    return ParallelTopology(
        family="praxis",
        world_size=1,
        rank=0,
        local_rank=0,
        expert_parallel_size=1,
        expert_replica_count=1,
        expert_group=None,
        expert_group_ranks=(0,),
        expert_data_group=None,
        expert_data_group_ranks=(0,),
        dense_data_group=None,
    )


def test_checkpoint_roundtrip_is_lineage_bound_and_idempotent(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = _TinyPlacedModel()
    dense = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    bundle = OptimizerBundle(dense, None)
    loss = model.linear(torch.ones(2, 3)).square().mean()
    loss.backward()
    dense.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    manager = CheckpointManager(tmp_path, topology=_single_topology(), keep_last=2)
    checkpoint = manager.save(
        model=model,
        optimizer=bundle,
        global_token_cursor=10,
        optimizer_step=1,
        phase="phase_a",
        shard_order_seed=123,
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        family_manifest_sha256="c" * 64,
        runtime_manifest_sha256="d" * 64,
        autotune_profile_sha256="e" * 64,
        precision_role_plan_sha256="f" * 64,
        precision_audit={"profile": "bf16"},
        extra_state={
            "data_position": {
                "global_token_cursor": 10,
                "phase": "phase_a",
                "shard_phase_index": 0,
                "offset_in_shard": 10,
            }
        },
    )
    assert manager.save(
        model=model,
        optimizer=bundle,
        global_token_cursor=10,
        optimizer_step=1,
        phase="phase_a",
        shard_order_seed=123,
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        family_manifest_sha256="c" * 64,
        runtime_manifest_sha256="d" * 64,
        autotune_profile_sha256="e" * 64,
        precision_role_plan_sha256="f" * 64,
        precision_audit={"profile": "bf16"},
        extra_state={
            "data_position": {
                "global_token_cursor": 10,
                "phase": "phase_a",
                "shard_phase_index": 0,
                "offset_in_shard": 10,
            }
        },
    ) == checkpoint
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    resume = manager.load(
        checkpoint,
        model=model,
        optimizer=bundle,
        expected_release_sha256="a" * 64,
        expected_shard_manifest_sha256="b" * 64,
        expected_family_manifest_sha256="c" * 64,
        expected_runtime_manifest_sha256="d" * 64,
        expected_autotune_profile_sha256="e" * 64,
        expected_precision_role_plan_sha256="f" * 64,
    )
    assert resume.global_token_cursor == 10
    assert resume.precision_role_plan_sha256 == "f" * 64
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


class _PostStepSignalModel(torch.nn.Module):
    def __init__(
        self,
        coordinator: SimpleNamespace,
        *,
        signal_on_bias_update: bool = True,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("selection_bias_updates", torch.zeros((), dtype=torch.int64))
        self._coordinator = coordinator
        self._signal_on_bias_update = signal_on_bias_update

    def forward(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "loss": self.weight.square(),
            "telemetry": {
                "expert_selection_counts": torch.ones((1, 1)),
                "moe_assignments": 1.0,
                "moe_processed_assignments": 1.0,
                "moe_dropped_assignments": 0.0,
                "expert_entropy_ratio": 1.0,
                "expert_load_cv": 0.0,
                "halt_collapse_fraction": 0.0,
                "sinkhorn_max_marginal_error": 0.0,
                "ponder_exit_mass_max_error": 0.0,
                "mhc_stream_diversity": 1.0,
            },
        }

    def update_expert_selection_biases(self, counts: torch.Tensor) -> None:
        assert counts.shape == (1, 1)
        self.selection_bias_updates.add_(1)
        if self._signal_on_bias_update:
            self._coordinator.requested = True
            self._coordinator.reason = "SIGTERM"


class _SingleBatchStream:
    def __init__(self, batch: TrainingBatch) -> None:
        self.batch = batch
        self.consumed = False

    def next(self, *, expected_cursor: int) -> TrainingBatch:
        assert not self.consumed
        assert expected_cursor == self.batch.global_token_cursor
        self.consumed = True
        return self.batch

    @staticmethod
    def position(cursor: int) -> dict[str, object]:
        return {
            "global_token_cursor": cursor,
            "phase": TokenSchedule.phase_at(cursor),
            "shard_phase_index": 0,
            "offset_in_shard": cursor,
        }


class _SingleBatchPrefetcher:
    def __init__(
        self,
        stream: _SingleBatchStream,
        **_kwargs: object,
    ) -> None:
        self.stream = stream

    def __enter__(self) -> "_SingleBatchPrefetcher":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def next(self, *, expected_cursor: int) -> TrainingBatch:
        return self.stream.next(expected_cursor=expected_cursor)


class _BF16PolicyStub:
    fp8_enabled = False
    effective_profile = "bf16"

    class _Audit:
        @staticmethod
        def to_dict() -> dict[str, str]:
            return {"effective_profile": "bf16"}

    audit = _Audit()


def _signal_training_batch(start_cursor: int, end_cursor: int) -> TrainingBatch:
    phase = TokenSchedule.phase_at(start_cursor)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    return TrainingBatch(
        input_ids=input_ids,
        canonical_ids=input_ids.clone(),
        labels=torch.tensor([[2, 3, 4, -100]], dtype=torch.long),
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        document_ids=torch.zeros_like(input_ids, dtype=torch.int32),
        reset_mask=torch.tensor([[True, False, False, False]]),
        phase=phase,
        global_token_cursor=start_cursor,
        next_global_token_cursor=end_cursor,
        non_padding_tokens=end_cursor - start_cursor,
        supervised_tokens=3,
    )


@pytest.mark.parametrize(
    ("start_cursor", "end_cursor", "interval_tokens", "phase_boundary"),
    [
        (0, 4, 4, False),
        (TOTAL_TOKENS - 4, TOTAL_TOKENS, TOTAL_TOKENS, True),
    ],
    ids=("interval-checkpoint-step", "final-phase-boundary-step"),
)
@pytest.mark.parametrize(
    "signal_timing",
    (
        "during-bias-update",
        "after-post-step-poll",
        "during-ordinary-save",
    ),
)
def test_checkpoint_eligible_signal_races_save_once_and_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start_cursor: int,
    end_cursor: int,
    interval_tokens: int,
    phase_boundary: bool,
    signal_timing: str,
) -> None:
    coordinator = SimpleNamespace(requested=False, reason=None)
    model = _PostStepSignalModel(
        coordinator,
        signal_on_bias_update=signal_timing == "during-bias-update",
    )
    dense = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = OptimizerBundle(dense, None)
    topology = _single_topology()
    manager = CheckpointManager(tmp_path, topology=topology, keep_last=2)
    batch = _signal_training_batch(start_cursor, end_cursor)
    stream = _SingleBatchStream(batch)
    inventory = ReleaseInventory(
        root=tmp_path,
        tokenizer=tmp_path / "tokenizer.json",
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        shards=(),
    )
    manifest_path = tmp_path / "family.yaml"
    runtime_path = tmp_path / "runtime.yaml"
    manifest_path.write_text("family: praxis\n", encoding="utf-8")
    runtime_path.write_text("schema: metis.training-runtime/v1\n", encoding="utf-8")
    runtime_manifest = _runtime_manifest()
    runtime_manifest.update(
        {
            "data": {
                "prefetch_micro_batches": 1,
                "shard_order_seed": 123,
            },
            "optimizer": {"grad_clip": 100.0},
            "checkpoint": {
                "interval_tokens": interval_tokens,
            },
            "health": {
                "abort_on_nonfinite": True,
                "abort_on_token_drop": True,
                "minimum_expert_entropy_ratio": 0.2,
                "maximum_expert_load_cv": 2.0,
                "maximum_halt_collapse_fraction": 0.98,
                "require_structural_telemetry": True,
                "maximum_sinkhorn_marginal_error": 0.005,
                "maximum_ponder_exit_mass_error": 0.001,
                "minimum_mhc_stream_diversity": 1.0e-12,
                "telemetry_interval_steps": 10,
            },
        }
    )
    config = SimpleNamespace(
        sequence_length=4,
        vocab_size=8,
        autotune=SimpleNamespace(
            gates=SimpleNamespace(max_grad_norm=100.0)
        ),
    )
    selection = AutotuneSelection(
        family="praxis",
        micro_batch=1,
        grad_accum=1,
        learning_rate=0.1,
        precision_profile="bf16",
        compile_mode="eager",
        dispatch_overlap=False,
        ngram_table_mode="replicated",
        profile_sha256="e" * 64,
        environment_sha256="environment",
        release_marker_sha256="release",
        precision_role_plan_sha256="f" * 64,
    )
    runtime = Runtime(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
    )
    monkeypatch.setattr(
        train_module,
        "ReleaseBatchPrefetcher",
        _SingleBatchPrefetcher,
    )
    monkeypatch.setattr(
        train_module,
        "estimate_train_flops",
        lambda *_args, **_kwargs: 1.0,
    )
    original_global_any = train_module.global_any
    global_any_calls = 0

    def injected_global_any(
        flag: bool,
        observed_topology: ParallelTopology,
        device: torch.device,
    ) -> bool:
        nonlocal global_any_calls
        global_any_calls += 1
        observed = original_global_any(flag, observed_topology, device)
        if (
            signal_timing == "after-post-step-poll"
            and global_any_calls == 2
        ):
            coordinator.requested = True
            coordinator.reason = "SIGTERM"
        return observed

    monkeypatch.setattr(train_module, "global_any", injected_global_any)
    original_save_checkpoint = train_module._save_checkpoint
    save_signal_reasons: list[str | None] = []

    def injected_save_checkpoint(*args: object, **kwargs: object) -> Path:
        save_signal_reasons.append(kwargs.get("signal_reason"))
        saved = original_save_checkpoint(*args, **kwargs)
        if signal_timing == "during-ordinary-save":
            coordinator.requested = True
            coordinator.reason = "SIGTERM"
        return saved

    monkeypatch.setattr(
        train_module,
        "_save_checkpoint",
        injected_save_checkpoint,
    )
    result, checkpoint = _run_steps(
        args=SimpleNamespace(probe=False, output=tmp_path),
        config=config,
        runtime_manifest=runtime_manifest,
        runtime=runtime,
        topology=topology,
        policy=_BF16PolicyStub(),
        model=model,
        forward_model=model,
        optimizer=optimizer,
        schedule=TokenSchedule(runtime_manifest, base_learning_rate=0.1),
        stream=stream,
        inventory=inventory,
        selection=selection,
        cursor=start_cursor,
        optimizer_step=0,
        checkpoint_manager=manager,
        signal_coordinator=coordinator,
        manifest_path=manifest_path,
        runtime_path=runtime_path,
    )

    assert result == {"requeue": True, "cursor": end_cursor, "step": 1}
    assert checkpoint is not None
    assert len(save_signal_reasons) == 1
    checkpoints = sorted(manager.root.glob("tokens-*"))
    assert checkpoints == [checkpoint]
    assert not list(manager.root.glob(".incomplete-*"))
    manifest = json.loads(
        (checkpoint / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["global_token_cursor"] == end_cursor
    assert manifest["optimizer_step"] == 1
    expected_signal_reason = (
        None if signal_timing == "during-ordinary-save" else "SIGTERM"
    )
    assert manifest["signal_reason"] == expected_signal_reason
    assert manifest["phase_boundary"] is phase_boundary
    assert (
        manifest["extra_state"]["data_position"]["global_token_cursor"]
        == end_cursor
    )
    saved_weight = model.weight.detach().clone()

    resumed_coordinator = SimpleNamespace(requested=False, reason=None)
    resumed_model = _PostStepSignalModel(resumed_coordinator)
    resumed_dense = torch.optim.SGD(resumed_model.parameters(), lr=9.0)
    resumed_optimizer = OptimizerBundle(resumed_dense, None)
    resume = CheckpointManager(
        tmp_path,
        topology=topology,
        keep_last=2,
    ).load(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_release_sha256=inventory.release_sha256,
        expected_shard_manifest_sha256=inventory.shard_manifest_sha256,
        expected_family_manifest_sha256=sha256_file(manifest_path),
        expected_runtime_manifest_sha256=sha256_file(runtime_path),
        expected_autotune_profile_sha256=selection.profile_sha256,
        expected_precision_role_plan_sha256=(
            selection.precision_role_plan_sha256
        ),
    )
    assert (resume.global_token_cursor, resume.optimizer_step) == (end_cursor, 1)
    assert resume.signal_reason == expected_signal_reason
    torch.testing.assert_close(resumed_model.weight, saved_weight)
    assert int(resumed_model.selection_bias_updates.item()) == 1


def test_chunked_checkpoint_dispatches_module_extra_state_on_restore(
    tmp_path: Path,
) -> None:
    model = _TinyExtraStateModel()
    optimizer = OptimizerBundle(
        torch.optim.AdamW(model.parameters(), lr=1.0e-3),
        None,
    )
    expected_extra_state = model.delayed_scaling_state.clone()
    manager = CheckpointManager(
        tmp_path,
        topology=_single_topology(),
        max_staging_bytes=2,
    )
    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        global_token_cursor=10,
        optimizer_step=1,
        phase="phase_a",
        shard_order_seed=123,
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        family_manifest_sha256="c" * 64,
        runtime_manifest_sha256="d" * 64,
        autotune_profile_sha256="e" * 64,
        precision_audit={"profile": "fp8"},
        extra_state={
            "data_position": {
                "global_token_cursor": 10,
                "phase": "phase_a",
                "shard_phase_index": 0,
                "offset_in_shard": 10,
            }
        },
    )
    model.delayed_scaling_state.zero_()
    loads_before_restore = model.extra_state_loads
    manager.load(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_release_sha256="a" * 64,
        expected_shard_manifest_sha256="b" * 64,
        expected_family_manifest_sha256="c" * 64,
        expected_runtime_manifest_sha256="d" * 64,
        expected_autotune_profile_sha256="e" * 64,
    )
    assert model.extra_state_loads == loads_before_restore + 1
    assert torch.equal(model.delayed_scaling_state, expected_extra_state)


def test_final_step_signal_checkpoint_promotes_after_zero_work_resume_and_survives_prune(
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    model = _TinyPlacedModel()
    dense = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer = OptimizerBundle(dense, None)
    model.linear(torch.ones(2, 3)).square().mean().backward()
    dense.step()
    final_cursor = 101
    final_step = 7
    incomplete_extra = {
        "posttraining_stage": "specialist_code",
        "parent_checkpoint_sha256": "1" * 64,
        "stage_config_sha256": "2" * 64,
        "bundle_sha256": "3" * 64,
        "stage_epoch": 1,
        "stage_next_global_batch": 0,
        "stage_optimizer_step": final_step,
        "campaign_token_cursor": final_cursor,
        "stage_complete": False,
        "optimizer_state_policy": "preserve",
        "runtime_batch": {
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
        },
        "last_metrics": {
            "loss": 0.125,
            "grad_norm": 0.75,
        },
    }
    common = {
        "model": model,
        "optimizer": optimizer,
        "global_token_cursor": final_cursor,
        "optimizer_step": final_step,
        "phase": "specialist_code",
        "shard_order_seed": 123,
        "release_sha256": "a" * 64,
        "shard_manifest_sha256": "b" * 64,
        "family_manifest_sha256": "c" * 64,
        "runtime_manifest_sha256": "d" * 64,
        "autotune_profile_sha256": "e" * 64,
        "precision_role_plan_sha256": "f" * 64,
        "precision_audit": {"profile": "bf16"},
        "signal_reason": "SIGTERM",
    }
    manager = CheckpointManager(
        tmp_path,
        topology=_single_topology(),
        keep_last=1,
    )
    checkpoint = manager.save(
        **common,
        phase_boundary=False,
        extra_state=incomplete_extra,
    )
    incomplete_manifest = json.loads(
        (checkpoint / "MANIFEST.json").read_text(encoding="utf-8")
    )
    incomplete_sha256 = incomplete_manifest["checkpoint_sha256"]
    assert incomplete_manifest["phase_boundary"] is False
    assert incomplete_manifest["extra_state"]["stage_complete"] is False

    # This is the requeue path: a fresh process fully restores the checkpoint,
    # discovers that the final optimizer step already consumed all stage data,
    # and therefore performs no further model or optimizer work.
    resumed_model = _TinyPlacedModel()
    resumed_dense = torch.optim.AdamW(resumed_model.parameters(), lr=9.0e-2)
    resumed_optimizer = OptimizerBundle(resumed_dense, None)
    resumed_manager = CheckpointManager(
        tmp_path,
        topology=_single_topology(),
        keep_last=1,
    )
    resume = resumed_manager.load(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_release_sha256="a" * 64,
        expected_shard_manifest_sha256="b" * 64,
        expected_family_manifest_sha256="c" * 64,
        expected_runtime_manifest_sha256="d" * 64,
        expected_autotune_profile_sha256="e" * 64,
        expected_precision_role_plan_sha256="f" * 64,
    )
    assert (resume.global_token_cursor, resume.optimizer_step) == (
        final_cursor,
        final_step,
    )
    promoted = resumed_manager.save(
        **{
            **common,
            "model": resumed_model,
            "optimizer": resumed_optimizer,
        },
        phase_boundary=True,
        extra_state={**incomplete_extra, "stage_complete": True},
    )
    assert promoted == checkpoint
    final_manifest = json.loads(
        (checkpoint / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert final_manifest["phase_boundary"] is True
    assert final_manifest["extra_state"]["stage_complete"] is True
    assert final_manifest["extra_state"]["last_metrics"] == {
        "loss": 0.125,
        "grad_norm": 0.75,
    }
    assert final_manifest["checkpoint_sha256"] != incomplete_sha256
    assert final_manifest["checkpoint_sha256"] == canonical_json_sha256(
        {
            key: value
            for key, value in final_manifest.items()
            if key != "checkpoint_sha256"
        }
    )
    for artifact in final_manifest["artifacts"]:
        artifact_path = checkpoint / artifact["path"]
        assert artifact_path.stat().st_size == artifact["bytes"]
        assert sha256_file(artifact_path) == artifact["sha256"]

    # A receipt created after finalization binds the promoted self-hash.
    receipt = {
        "schema": "metis.checkpoint-receipt/v1",
        "checkpoint_manifest": str(checkpoint / "MANIFEST.json"),
        "checkpoint_sha256": final_manifest["checkpoint_sha256"],
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )
    assert receipt["checkpoint_sha256"] == final_manifest["checkpoint_sha256"]
    assert receipt["receipt_sha256"] == canonical_json_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )

    for cursor in (202, 303):
        resumed_manager.save(
            **{
                **common,
                "model": resumed_model,
                "optimizer": resumed_optimizer,
                "global_token_cursor": cursor,
                "optimizer_step": final_step + cursor,
                "phase": "later_stage",
                "signal_reason": None,
            },
            phase_boundary=False,
            extra_state={
                "posttraining_stage": "later_stage",
                "campaign_token_cursor": cursor,
                "stage_complete": False,
            },
        )
    assert checkpoint.is_dir()
    assert not (
        resumed_manager.root / "tokens-0000000000202"
    ).exists()
    assert (
        resumed_manager.root / "tokens-0000000000303"
    ).is_dir()


def test_same_cursor_checkpoint_rejects_incompatible_requests_and_unproven_promotion(
    tmp_path: Path,
) -> None:
    model = _TinyPlacedModel()
    dense = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    optimizer = OptimizerBundle(dense, None)
    manager = CheckpointManager(tmp_path, topology=_single_topology())
    common = {
        "model": model,
        "optimizer": optimizer,
        "global_token_cursor": 77,
        "optimizer_step": 4,
        "phase": "overall_sft",
        "shard_order_seed": 123,
        "release_sha256": "a" * 64,
        "shard_manifest_sha256": "b" * 64,
        "family_manifest_sha256": "c" * 64,
        "runtime_manifest_sha256": "d" * 64,
        "autotune_profile_sha256": "e" * 64,
        "precision_role_plan_sha256": "f" * 64,
        "precision_audit": {"profile": "bf16"},
        "signal_reason": None,
        "phase_boundary": False,
        "extra_state": {
            "posttraining_stage": "overall_sft",
            "stage_optimizer_step": 4,
            "campaign_token_cursor": 77,
            "stage_complete": False,
        },
    }
    manager.save(**common)

    with pytest.raises(RuntimeError, match="optimizer_step"):
        manager.save(**{**common, "optimizer_step": 5})
    with pytest.raises(RuntimeError, match="release_sha256"):
        manager.save(**{**common, "release_sha256": "9" * 64})
    with pytest.raises(RuntimeError, match="extra_state"):
        manager.save(
            **{
                **common,
                "extra_state": {
                    **common["extra_state"],
                    "campaign_token_cursor": 76,
                },
            }
        )
    with pytest.raises(RuntimeError, match="fully loads and validates"):
        manager.save(
            **{
                **common,
                "phase_boundary": True,
                "extra_state": {
                    **common["extra_state"],
                    "stage_complete": True,
                },
            }
        )


def test_checkpoint_streams_tensor_chunks_and_restores_optimizer_exactly(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    model = _TinyPlacedModel()
    dense = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    bundle = OptimizerBundle(dense, None)
    model.linear(torch.ones(3, 3)).square().mean().backward()
    dense.step()
    expected_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    expected_optimizer = {
        name: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in dense.state[parameter].items()
        }
        for name, parameter in model.named_parameters()
    }
    manager = CheckpointManager(
        tmp_path,
        topology=_single_topology(),
        keep_last=2,
        max_staging_bytes=8,
    )
    checkpoint = manager.save(
        model=model,
        optimizer=bundle,
        global_token_cursor=12,
        optimizer_step=1,
        phase="phase_a",
        shard_order_seed=123,
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        family_manifest_sha256="c" * 64,
        runtime_manifest_sha256="d" * 64,
        autotune_profile_sha256="e" * 64,
        precision_audit={"profile": "bf16"},
        extra_state={
            "data_position": {
                "global_token_cursor": 12,
                "phase": "phase_a",
                "shard_phase_index": 0,
                "offset_in_shard": 12,
            }
        },
    )
    manifest = json.loads(
        (checkpoint / "MANIFEST.json").read_text(encoding="utf-8")
    )
    state_records = [
        row for row in manifest["artifacts"] if row.get("kind") == "state_shard"
    ]
    assert manifest["layout"] == "metis.tensor-chunks/v1"
    assert len(state_records) > 1
    assert all(int(row["staged_tensor_bytes"]) <= 8 for row in state_records)
    assert not (checkpoint / "replicated.pt").exists()

    restored_model = _TinyPlacedModel()
    restored_dense = torch.optim.AdamW(restored_model.parameters(), lr=9.0e-2)
    restored_bundle = OptimizerBundle(restored_dense, None)
    manager.load(
        checkpoint,
        model=restored_model,
        optimizer=restored_bundle,
        expected_release_sha256="a" * 64,
        expected_shard_manifest_sha256="b" * 64,
        expected_family_manifest_sha256="c" * 64,
        expected_runtime_manifest_sha256="d" * 64,
        expected_autotune_profile_sha256="e" * 64,
    )
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    for name, parameter in restored_model.named_parameters():
        restored_state = restored_dense.state[parameter]
        assert set(restored_state) == set(expected_optimizer[name])
        for key, expected in expected_optimizer[name].items():
            observed = restored_state[key]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(observed, expected)
            else:
                assert observed == expected
    assert restored_dense.param_groups[0]["lr"] == pytest.approx(3.0e-4)


def test_checkpoint_latest_uses_newest_manifest_when_pointer_is_stale(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path, topology=_single_topology())
    older = manager.root / "tokens-0000000000010"
    newer = manager.root / "tokens-0000000000020"
    older.mkdir()
    newer.mkdir()
    (older / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (newer / "MANIFEST.json").write_text("{}\n", encoding="utf-8")
    (manager.root / "LATEST.json").write_text(
        json.dumps(
            {
                "schema": "metis.checkpoint-latest/v1",
                "checkpoint": older.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert manager.latest() == newer


class _MigrationCheckpointManager:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.expected_profile_sha256: str | None = None

    def latest(self) -> Path:
        return self.checkpoint

    def load(self, _checkpoint: Path, **kwargs: object) -> SimpleNamespace:
        assert _checkpoint == self.checkpoint
        self.expected_profile_sha256 = str(
            kwargs["expected_autotune_profile_sha256"]
        )
        return SimpleNamespace(global_token_cursor=91, optimizer_step=7)


def _migration_restore_fixture(
    tmp_path: Path,
) -> tuple[
    _MigrationCheckpointManager,
    ReleaseInventory,
    Path,
    Path,
    AutotuneSelection,
    Path,
    dict,
]:
    old_profile_sha = "1" * 64
    new_profile_sha = "2" * 64
    precision_role_plan_sha = "9" * 64
    checkpoint = tmp_path / "checkpoint-00000000000000000091"
    checkpoint.mkdir()
    checkpoint_manifest = {
        "checkpoint_sha256": "3" * 64,
        "autotune_profile_sha256": old_profile_sha,
        "precision_role_plan_sha256": precision_role_plan_sha,
    }
    checkpoint_manifest_path = checkpoint / "MANIFEST.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "praxis.yaml"
    runtime_path = tmp_path / "runtime.yaml"
    manifest_path.write_text("family: praxis\n", encoding="utf-8")
    runtime_path.write_text("schema: test\n", encoding="utf-8")
    old_selected = {
        "micro_batch_size": 2,
        "grad_accum_steps": 8,
        "learning_rate": 2.0e-4,
        "precision_profile": "fp8",
        "compile_mode": "default",
        "dispatch_overlap": True,
        "ngram_table_mode": "row_sharded",
    }
    new_selected = {
        **old_selected,
        "micro_batch_size": 1,
        "grad_accum_steps": 16,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "metis.portage-autotune/v1",
                "family": "praxis",
                "selected": new_selected,
                "precision_role_plan_sha256": precision_role_plan_sha,
                "profile_sha256": new_profile_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "metis.autotune-profile-migration/v1",
        "family": "praxis",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "old_profile_sha256": old_profile_sha,
        "new_profile_sha256": new_profile_sha,
        "new_profile_path": str(profile_path.resolve()),
        "precision_role_plan_sha256": precision_role_plan_sha,
        "old_selected": old_selected,
        "new_selected": new_selected,
        "allowed_changes": ["micro_batch_size", "grad_accum_steps"],
        "global_token_batch_unchanged": True,
        "state_conversion": "none_parameter_and_optimizer_layout_unchanged",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    selection = AutotuneSelection(
        family="praxis",
        micro_batch=1,
        grad_accum=16,
        learning_rate=2.0e-4,
        precision_profile="fp8",
        compile_mode="default",
        dispatch_overlap=True,
        ngram_table_mode="row_sharded",
        profile_sha256=new_profile_sha,
        environment_sha256="4" * 64,
        release_marker_sha256="5" * 64,
        precision_role_plan_sha256=precision_role_plan_sha,
    )
    inventory = ReleaseInventory(
        root=tmp_path,
        tokenizer=tmp_path / "tokenizer.json",
        release_sha256="6" * 64,
        shard_manifest_sha256="7" * 64,
        shards=(),
    )
    return (
        _MigrationCheckpointManager(checkpoint),
        inventory,
        manifest_path,
        runtime_path,
        selection,
        profile_path,
        receipt,
    )


def test_restore_accepts_exact_oom_profile_migration_and_loads_old_profile(
    tmp_path: Path,
) -> None:
    (
        manager,
        inventory,
        manifest_path,
        runtime_path,
        selection,
        profile_path,
        receipt,
    ) = _migration_restore_fixture(tmp_path)
    receipt_path = tmp_path / "profile-migration.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with mock.patch.dict(
        "os.environ",
        {"METIS_AUTOTUNE_PROFILE_MIGRATION": str(receipt_path)},
        clear=False,
    ):
        cursor, step = _restore_if_requested(
            mode="auto",
            manager=manager,  # type: ignore[arg-type]
            model=object(),
            optimizer=object(),  # type: ignore[arg-type]
            inventory=inventory,
            manifest_path=manifest_path,
            runtime_path=runtime_path,
            selection=selection,
            profile_path=profile_path,
        )
    assert (cursor, step) == (91, 7)
    assert manager.expected_profile_sha256 == "1" * 64


@pytest.mark.parametrize("semantic_tamper", [False, True])
def test_restore_rejects_tampered_or_batch_changing_profile_migration(
    tmp_path: Path,
    semantic_tamper: bool,
) -> None:
    (
        manager,
        inventory,
        manifest_path,
        runtime_path,
        selection,
        profile_path,
        receipt,
    ) = _migration_restore_fixture(tmp_path)
    if semantic_tamper:
        receipt["new_selected"]["grad_accum_steps"] = 15
        receipt["receipt_sha256"] = canonical_json_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    else:
        receipt["checkpoint_sha256"] = "8" * 64
    receipt_path = tmp_path / "profile-migration.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with mock.patch.dict(
        "os.environ",
        {"METIS_AUTOTUNE_PROFILE_MIGRATION": str(receipt_path)},
        clear=False,
    ), pytest.raises(RuntimeError, match="migration"):
        _restore_if_requested(
            mode="auto",
            manager=manager,  # type: ignore[arg-type]
            model=object(),
            optimizer=object(),  # type: ignore[arg-type]
            inventory=inventory,
            manifest_path=manifest_path,
            runtime_path=runtime_path,
            selection=selection,
            profile_path=profile_path,
        )


def _write_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<|endoftext|>": 1,
                "a": 2,
                "b": 3,
                "c": 4,
                "d": 5,
                "e": 6,
                "f": 7,
            },
            unk_token="<pad>",
        )
    )
    tokenizer.save(str(path))


def test_release_stream_emits_next_token_aligned_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import metis_training.data as data_module

    phase_tokens = {"phase_a": 8, "phase_b": 4, "phase_c": 4}
    phase_starts = {"phase_a": 0, "phase_b": 8, "phase_c": 12}
    monkeypatch.setattr(data_module, "PHASE_TOKENS", phase_tokens)
    monkeypatch.setattr(data_module, "PHASE_STARTS", phase_starts)
    monkeypatch.setattr(data_module, "TOTAL_TOKENS", 16)
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    values = {
        "phase_a": [2, 3, 4, 5, 6, 7, 2, 3],
        "phase_b": [4, 5, 6, 7],
        "phase_c": [2, 3, 4, 5],
    }
    shards = []
    for phase_index, (phase, tokens) in enumerate(values.items()):
        binary = tmp_path / f"{phase}.bin"
        np.asarray(tokens, dtype="<u2").tofile(binary)
        index = tmp_path / f"{phase}.index.jsonl"
        index.write_text("", encoding="utf-8")
        shards.append(
            ReleaseShard(
                phase=phase,
                phase_index=0,
                binary=binary,
                index=index,
                tokens=len(tokens),
            )
        )
    inventory = ReleaseInventory(
        root=tmp_path,
        tokenizer=tokenizer_path,
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        shards=tuple(shards),
    )
    with DeterministicReleaseStream(
        inventory,
        sequence_length=4,
        shard_order_seed=7,
        expected_vocabulary_size=8,
        require_canonical_ids=False,
    ) as stream:
        batch = stream.batch(
            global_token_cursor=0,
            rank=0,
            world_size=1,
            micro_batch_size=1,
        )
    assert batch.input_ids.tolist() == [[2, 3, 4, 5]]
    assert batch.canonical_ids.tolist() == [[2, 3, 4, 5]]
    assert batch.labels.tolist() == [[3, 4, 5, 6]]
    assert batch.next_global_token_cursor == 4


def test_engram_canonical_normalization_preserves_only_declared_boundaries() -> None:
    assert canonicalize_decoded_token("Ａpple", raw_token="raw-a") == "apple"
    assert canonicalize_decoded_token(" Café\t", raw_token="raw-b") == "cafe"
    assert canonicalize_decoded_token(" \t\r\n ", raw_token="raw-c") == " "
    assert canonicalize_decoded_token("\uFFFDApple", raw_token="<raw-replacement>") == (
        "<raw-replacement>"
    )
    assert canonicalize_decoded_token(
        "\uFFFDApple", raw_token="<raw-replacement>"
    ) != canonicalize_decoded_token("Apple", raw_token="Apple")


def _canonical_sidecar_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    vocab = {
        "<eos>": 0,
        "Apple": 1,
        "apple": 2,
        "Ａpple": 3,
        "café": 4,
        "cafe": 5,
        "\uFFFDbad": 6,
        "plain": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=None))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    descriptor = build_canonical_id_sidecar(
        tokenizer,
        tokenizer_path=tokenizer_path,
        output_dir=tmp_path,
    )
    return (
        tokenizer_path,
        tmp_path / CANONICAL_IDS_MANIFEST,
        tmp_path / CANONICAL_IDS_BINARY,
        descriptor,
    )


def test_canonical_sidecar_is_uint16_contiguous_and_tokenizer_bound(
    tmp_path: Path,
) -> None:
    tokenizer_path, manifest_path, binary_path, descriptor = (
        _canonical_sidecar_fixture(tmp_path)
    )
    validated, canonical_ids = validate_canonical_id_sidecar(
        manifest_path=manifest_path,
        binary_path=binary_path,
        tokenizer_path=tokenizer_path,
        expected_vocabulary_size=8,
        expected_manifest_sha256=descriptor["manifest_sha256"],
        expected_binary_sha256=descriptor["binary_sha256"],
        recompute_from_tokenizer=True,
    )
    assert canonical_ids.dtype == np.dtype("<u2")
    assert canonical_ids.shape == (8,)
    assert canonical_ids[1] == canonical_ids[2] == canonical_ids[3]
    assert canonical_ids[4] == canonical_ids[5]
    assert canonical_ids[6] != canonical_ids[1]
    assert np.array_equal(
        np.unique(canonical_ids),
        np.arange(validated["canonical_vocabulary_size"], dtype=np.uint16),
    )


def test_canonical_sidecar_rejects_binary_corruption(tmp_path: Path) -> None:
    tokenizer_path, manifest_path, binary_path, descriptor = (
        _canonical_sidecar_fixture(tmp_path)
    )
    payload = bytearray(binary_path.read_bytes())
    payload[-1] ^= 1
    binary_path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="descriptor or lineage"):
        validate_canonical_id_sidecar(
            manifest_path=manifest_path,
            binary_path=binary_path,
            tokenizer_path=tokenizer_path,
            expected_vocabulary_size=8,
            expected_manifest_sha256=descriptor["manifest_sha256"],
            expected_binary_sha256=descriptor["binary_sha256"],
        )


def test_release_stream_requires_canonical_sidecar_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import metis_training.data as data_module

    monkeypatch.setattr(
        data_module,
        "PHASE_TOKENS",
        {"phase_a": 1, "phase_b": 1, "phase_c": 1},
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    shards = []
    for phase in ("phase_a", "phase_b", "phase_c"):
        binary = tmp_path / f"{phase}.bin"
        np.asarray([1], dtype="<u2").tofile(binary)
        index = tmp_path / f"{phase}.index.jsonl"
        index.write_text("", encoding="utf-8")
        shards.append(ReleaseShard(phase, 0, binary, index, 1))
    inventory = ReleaseInventory(
        root=tmp_path,
        tokenizer=tokenizer_path,
        release_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        shards=tuple(shards),
    )
    with pytest.raises(RuntimeError, match="requires.*canonical-ID sidecar"):
        DeterministicReleaseStream(
            inventory,
            sequence_length=1,
            shard_order_seed=7,
            expected_vocabulary_size=8,
        )


def test_telemetry_row_contains_portage_completion_fields() -> None:
    row = StepMetrics(
        optimizer_step=1,
        global_token_cursor=10,
        phase="phase_a",
        loss=2.0,
        learning_rate=1.0e-4,
        global_non_padding_tokens=100,
        global_supervised_tokens=99,
        step_time_s=1.0,
        tokens_per_second=100.0,
        estimated_train_flops=1.0e12,
        estimated_mfu=0.5,
        grad_norm=1.0,
        update_to_weight_ratio=None,
        peak_hbm_bytes=1,
        precision_profile="bf16",
        overflow_drop_tokens=0,
        collective_errors=0,
        telemetry={
            "all_to_all_bytes": 8,
            "all_to_all_seconds": 0.1,
            "expert_load_cv": 0.2,
        },
    ).to_dict()
    for field in (
        "tokens_per_second",
        "estimated_train_flops",
        "step_time_s",
        "mfu",
        "all_to_all_bytes",
        "all_to_all_seconds",
        "overflow_drop_tokens",
        "expert_load_cv",
        "loss",
        "global_token_cursor",
    ):
        assert field in row


def test_hbm_evidence_gates_on_reserved_or_allocated_peak() -> None:
    with (
        mock.patch("torch.cuda.max_memory_allocated", return_value=70),
        mock.patch("torch.cuda.max_memory_reserved", return_value=90),
    ):
        evidence = peak_memory_evidence(torch.device("cuda"))
    assert evidence == {
        "peak_hbm_bytes": 90,
        "peak_hbm_allocated_bytes": 70,
        "peak_hbm_reserved_bytes": 90,
    }


def _healthy_structural_telemetry() -> dict[str, float]:
    return {
        "moe_assignments": 128.0,
        "moe_processed_assignments": 128.0,
        "moe_dropped_assignments": 0.0,
        "expert_entropy_ratio": 0.8,
        "expert_load_cv": 0.25,
        "halt_collapse_fraction": 0.4,
        "sinkhorn_max_marginal_error": 1.0e-5,
        "ponder_exit_mass_max_error": 1.0e-6,
        "mhc_stream_diversity": 1.0e-3,
    }


def _enforce_structural_health(telemetry: dict[str, float]) -> None:
    enforce_health_gates(
        loss=2.0,
        grad_norm_value=1.0,
        telemetry=telemetry,
        maximum_grad_norm=100.0,
        abort_on_nonfinite=True,
        abort_on_token_drop=True,
        minimum_expert_entropy_ratio=0.2,
        maximum_expert_load_cv=2.0,
        maximum_halt_collapse_fraction=0.98,
        require_structural_telemetry=True,
        maximum_sinkhorn_marginal_error=0.005,
        maximum_ponder_exit_mass_error=0.001,
        minimum_mhc_stream_diversity=1.0e-12,
    )


def test_structural_health_gates_accept_consistent_model_telemetry() -> None:
    _enforce_structural_health(_healthy_structural_telemetry())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("moe_processed_assignments", 127.0, "assignment accounting"),
        ("sinkhorn_max_marginal_error", 0.1, "Sinkhorn"),
        ("ponder_exit_mass_max_error", 0.1, "Ponder"),
        ("mhc_stream_diversity", 0.0, "streams collapsed"),
    ],
)
def test_structural_health_gates_fail_closed(
    field: str,
    value: float,
    message: str,
) -> None:
    telemetry = _healthy_structural_telemetry()
    telemetry[field] = value
    with pytest.raises(RuntimeError, match=message):
        _enforce_structural_health(telemetry)


def test_structural_health_gates_require_complete_telemetry() -> None:
    telemetry = _healthy_structural_telemetry()
    del telemetry["ponder_exit_mass_max_error"]
    with pytest.raises(RuntimeError, match="telemetry is incomplete"):
        _enforce_structural_health(telemetry)
