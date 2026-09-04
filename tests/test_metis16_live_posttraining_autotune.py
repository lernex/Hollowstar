from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn

from metis_training.distributed import ParallelTopology, Runtime
from metis_training.stage_backend import (
    LIVE_PROFILE_AUTOTUNE_RECEIPT_SCHEMA,
    MMapStageBundle,
    ProfileSelection,
    StageBackendError,
    _canonical_hash,
    _metrics_reproduce,
    _main_optimizer_step,
    _run_live_profile_autotune,
    _run_supervised_stage,
    _working_set_candidate_bundle,
)


def _topology() -> ParallelTopology:
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


def _trial(profile: dict[str, float], metrics: dict[str, float]) -> dict[str, object]:
    payload: dict[str, object] = {
        "profile": profile,
        "profile_sha256": _canonical_hash(profile),
        "metrics": metrics,
        "metrics_sha256": _canonical_hash(metrics),
        "trial_sha256": "",
    }
    payload["trial_sha256"] = _canonical_hash(
        payload, omit={"trial_sha256"}
    )
    return payload


class _ModeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.mode = "parent"


class _BiasUpdateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.optimizer_stepped = False
        self.bias_updates: list[torch.Tensor] = []

    def update_expert_selection_biases(self, counts: torch.Tensor) -> None:
        assert self.optimizer_stepped
        self.bias_updates.append(counts.clone())


def test_live_gspo_trials_restore_parent_and_resume_from_self_hashed_receipt() -> None:
    default = {
        "clip_low": 0.0003,
        "clip_high": 0.0004,
        "length_coefficient": 0.0,
    }
    winner = {**default, "length_coefficient": 0.05}
    first_metrics = {
        "reward_gain": 0.1,
        "entropy_delta": 0.0,
        "evaluation_regression": 0.0,
        "nonfinite_steps": 0.0,
        "evaluation_records": 3.0,
        "rollout_prompts": 3.0,
    }
    winner_metrics = {
        "reward_gain": 0.2,
        "entropy_delta": 0.0,
        "evaluation_regression": 0.0,
        "nonfinite_steps": 0.0,
        "evaluation_records": 3.0,
        "rollout_prompts": 3.0,
    }
    evaluator = {
        "records": 3,
        "reproduction_tolerance": 1.0e-8,
        "evaluator_sha256": "e" * 64,
        "dataset_sha256": "d" * 64,
    }
    live = {
        "training_optimizer_steps": 1,
        "seed": 17,
        "evaluator": evaluator,
        "live_autotune_sha256": "l" * 64,
    }
    evidence: dict[str, object] = {
        "trials": [
            _trial(default, first_metrics),
            _trial(winner, winner_metrics),
        ],
        "live_autotune": live,
        "selection_sha256": "",
    }
    evidence["selection_sha256"] = _canonical_hash(
        evidence, omit={"selection_sha256"}
    )
    bundle = SimpleNamespace(
        family="praxis",
        stage_id="hybrid_mode_gspo",
        manifest={"profile_selection": evidence},
        manifest_sha256="b" * 64,
        records=2,
        training={
            "epochs": 1,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
        },
        batch_migration_sha256=None,
        working_set_autotune_sha256="w" * 64,
        working_set={"token_chunk_size": 1},
    )
    stage = {
        "id": "hybrid_mode_gspo",
        "autotune": {
            "live_canary": {
                "schema": "metis.posttraining-live-canary-policy/v1",
                "training_optimizer_steps": 1,
                "evaluator_implementation": "metis.rlvr-offline-policy-replay/v1",
                "minimum_evaluation_records": 3,
                "maximum_reproduction_tolerance": 1.0e-8,
                "restore_parent_between_trials": True,
                "restore_rng_between_trials": True,
            }
        },
        "objective": {
            "clip_low": default["clip_low"],
            "clip_high": default["clip_high"],
        },
        "reward": {"length_coefficient": 0.0},
        "autotune_gate": {
            "minimum_reward_gain": 0.0,
            "maximum_entropy_delta": 0.25,
            "maximum_evaluation_regression": 0.0,
            "maximum_nonfinite_steps": 0,
        },
    }
    model = _ModeModel()
    optimizer = SimpleNamespace(zero_grad=lambda **_kwargs: None)
    runtime = Runtime(
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
    )
    restore_calls: list[str] = []

    def restore_parent() -> None:
        model.mode = "parent"
        restore_calls.append("restore")

    def train_candidate(**kwargs: object) -> dict[str, object]:
        selection = kwargs["selection_override"]
        assert isinstance(selection, ProfileSelection)
        model.mode = f"{selection.profile['length_coefficient']:.2f}"
        return {}

    def evaluate(**_kwargs: object) -> dict[str, float]:
        score = {"parent": 0.5, "0.00": 0.6, "0.05": 0.7}[model.mode]
        return {
            "expected_reward": score,
            "entropy": 1.0,
            "correct_response_nll": 0.5,
            "evaluation_records": 3.0,
            "rollout_prompts": 3.0,
        }

    with tempfile.TemporaryDirectory() as raw, mock.patch.multiple(
        "metis_training.stage_backend",
        _selected_gspo_profile=mock.DEFAULT,
        _gspo_profile_candidates=mock.DEFAULT,
        _run_gspo_stage=mock.DEFAULT,
        _evaluate_live_gspo_profile=mock.DEFAULT,
        _profile_trial_state_fingerprint=mock.DEFAULT,
        _runtime_rank_inventory=mock.DEFAULT,
        _apply_optimizer_state_transition=mock.DEFAULT,
    ) as patches:
        patches["_selected_gspo_profile"].return_value = ProfileSelection(
            default, _canonical_hash(default), "s" * 64, {}, "external"
        )
        gate = {
            "name": "stable_reward",
            "minimum_reward_gain": 0.0,
            "maximum_entropy_delta": 0.25,
            "maximum_evaluation_regression": 0.0,
            "maximum_nonfinite_steps": 0,
        }
        patches["_gspo_profile_candidates"].return_value = (
            default,
            [default, winner],
            gate,
        )
        patches["_run_gspo_stage"].side_effect = train_candidate
        patches["_evaluate_live_gspo_profile"].side_effect = evaluate
        patches["_profile_trial_state_fingerprint"].side_effect = (
            lambda *_args, **_kwargs: _canonical_hash({"mode": model.mode})
        )
        patches["_runtime_rank_inventory"].return_value = (
            [{"rank": 0, "device": "cpu"}],
            "r" * 64,
        )
        patches["_apply_optimizer_state_transition"].return_value = None

        output = Path(raw)
        selected = _run_live_profile_autotune(
            stage=stage,
            stage_config_sha256="c" * 64,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            forward_model=model,
            runtime=runtime,
            topology=_topology(),
            output_root=output,
            parent_checkpoint_sha256="p" * 64,
            precision_role_plan_sha256="q" * 64,
            autotune_profile_sha256="a" * 64,
            compile_mode="eager",
            optimizer_state_policy="preserve",
            restore_parent=restore_parent,
            signal_coordinator=None,
        )
        assert selected.profile == winner
        receipt_path = output / "autotune" / "hybrid_mode_gspo.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["schema"] == LIVE_PROFILE_AUTOTUNE_RECEIPT_SCHEMA
        assert receipt["complete"] is True
        assert len(receipt["trials"]) == 2
        assert receipt["receipt_sha256"] == _canonical_hash(
            receipt, omit={"receipt_sha256"}
        )
        calls_after_first_run = len(restore_calls)

        resumed = _run_live_profile_autotune(
            stage=stage,
            stage_config_sha256="c" * 64,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            forward_model=model,
            runtime=runtime,
            topology=_topology(),
            output_root=output,
            parent_checkpoint_sha256="p" * 64,
            precision_role_plan_sha256="q" * 64,
            autotune_profile_sha256="a" * 64,
            compile_mode="eager",
            optimizer_state_policy="preserve",
            restore_parent=restore_parent,
            signal_coordinator=None,
            active_resume=True,
        )
        assert resumed.profile == winner
        assert len(restore_calls) == calls_after_first_run


def test_live_metrics_must_reproduce_the_sealed_evaluator() -> None:
    with torch.no_grad():
        try:
            _metrics_reproduce(
                {"reward_gain": 0.2},
                {"reward_gain": 0.1},
                tolerance=1.0e-4,
            )
        except StageBackendError as exception:
            assert "could not reproduce" in str(exception)
        else:
            raise AssertionError("metric drift must fail closed")


def test_working_set_candidate_preserves_effective_batch_and_token_budget() -> None:
    bundle = MMapStageBundle(
        stage_id="specialist_reasoning",
        family="praxis",
        root=Path("/tmp"),
        manifest={},
        manifest_sha256="m" * 64,
        arrays={},
        specs={},
        teacher_distributions={},
        sealed_training={
            "micro_batch_size": 4,
            "gradient_accumulation": 2,
        },
        training={
            "epochs": 1,
            "micro_batch_size": 4,
            "gradient_accumulation": 2,
            "shuffle_seed": 7,
            "checkpoint_interval_steps": 0,
            "learning_rate": 1.0e-4,
            "minimum_learning_rate_ratio": 0.1,
            "warmup_steps": 0,
            "gradient_clip": 1.0,
        },
        batch_migration_path=None,
        batch_migration_sha256=None,
        batch_migration_chain=((4, 2),),
        working_set_autotune_sha256=None,
        canonical_id_lookup=np.arange(8, dtype="<u2"),
        canonical_map_self_sha256="c" * 64,
        canonical_ids_sha256="i" * 64,
        canonical_lookup_tensor=None,
        working_set={
            "token_chunk_size": 128,
            "candidate_micro_group_size": 4,
        },
        records=16,
        sequence_length=4,
    )
    revised = _working_set_candidate_bundle(
        bundle,
        {
            "micro_batch_size": 2,
            "token_chunk_size": 256,
            "candidate_micro_group_size": 8,
        },
        topology=_topology(),
    )
    assert revised.training["micro_batch_size"] == 2
    assert revised.training["gradient_accumulation"] == 4
    assert (
        revised.training["micro_batch_size"]
        * revised.training["gradient_accumulation"]
        == 8
    )
    assert revised.working_set["token_chunk_size"] == 256
    assert revised.working_set["candidate_micro_group_size"] == 8


def test_policy_optimizer_updates_selection_bias_exactly_once_after_step() -> None:
    model = _BiasUpdateModel()
    (model.weight.square() * 2.0).backward()

    class _Optimizer:
        def step(self) -> None:
            model.optimizer_stepped = True

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            model.weight.grad = None

    counts = torch.tensor([[3.0, 1.0]], dtype=torch.float32)
    _main_optimizer_step(
        model=model,
        optimizer=_Optimizer(),  # type: ignore[arg-type]
        topology=_topology(),
        runtime=Runtime(
            device=torch.device("cpu"),
            rank=0,
            local_rank=0,
            world_size=1,
            distributed=False,
        ),
        local_weight=1,
        gradient_clip=10.0,
        expert_selection_counts=counts,
    )
    assert len(model.bias_updates) == 1
    assert torch.equal(model.bias_updates[0], counts)


def test_zero_work_final_step_resume_preserves_last_metrics() -> None:
    bundle = SimpleNamespace(
        stage_id="overall_sft",
        sequence_length=4,
        records=2,
        manifest={},
        training={
            "epochs": 1,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "checkpoint_interval_steps": 0,
        },
        iter_rank_batches=lambda *_args, **_kwargs: iter(()),
    )
    model = nn.Linear(2, 2)
    optimizer = SimpleNamespace(zero_grad=lambda **_kwargs: None)
    result = _run_supervised_stage(
        stage={
            "id": "overall_sft",
            "sequence_length": 4,
            "objective": {"name": "causal_lm"},
        },
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        runtime=Runtime(
            device=torch.device("cpu"),
            rank=0,
            local_rank=0,
            world_size=1,
            distributed=False,
        ),
        topology=_topology(),
        start_epoch=1,
        start_global_batch=0,
        start_optimizer_step=2,
        start_cursor=99,
        checkpoint_callback=lambda *_args: None,
        resume_metrics={"loss": 1.25, "grad_norm": 0.75},
    )
    assert result["optimizer_steps"] == 2
    assert result["campaign_token_cursor"] == 99
    assert result["loss"] == 1.25
    assert result["grad_norm"] == 0.75
