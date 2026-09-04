from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import yaml

from .checkpointing import CheckpointManager, SignalCoordinator
from .contracts import (
    AutotuneSelection,
    canonical_json_sha256,
    load_autotune_selection,
    load_family_manifest,
    require_release_verification,
    sha256_file,
)
from .data import (
    PHASE_STARTS,
    PHASE_TOKENS,
    TOTAL_TOKENS,
    DeterministicReleaseStream,
    ReleaseBatchPrefetcher,
    ReleaseInventory,
    TrainingBatch,
)
from .distributed import (
    ParallelTopology,
    Runtime,
    all_reduce_sum,
    barrier,
    broadcast_initial_parameters,
    build_parallel_topology,
    destroy_runtime,
    global_any,
    initialize_runtime,
    normalize_summed_gradients,
    OverlappedGradientReducer,
    synchronize_gradients,
)
from .metrics import (
    MetricsWriter,
    StepMetrics,
    enforce_health_gates,
    estimate_train_flops,
    estimated_mfu,
    peak_memory_evidence,
    scalar_telemetry,
)
from .optimizers import (
    OptimizerBundle,
    build_training_optimizers,
    clip_grad_norm_,
    sample_parameters,
    sampled_update_to_weight_ratio,
)
from .precision import PrecisionPolicy, build_precision_policy
from .precision_plan import load_precision_role_plan, measured_role_dtype_map
from .schedule import TokenSchedule, set_optimizer_learning_rate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_MANIFEST = REPOSITORY_ROOT / "configs" / "metis16" / "training-runtime.yaml"
DEFAULT_POSTTRAINING_MANIFEST = REPOSITORY_ROOT / "configs" / "metis16" / "posttraining.yaml"
REQUEUE_EXIT_CODE = 75


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _load_runtime(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "metis.training-runtime/v1":
        raise RuntimeError("Invalid Metis-1.6 training runtime manifest")
    return payload


def _output_value(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _combined_training_loss(output: Any) -> torch.Tensor:
    causal = _output_value(output, "loss")
    if causal is None or not isinstance(causal, torch.Tensor):
        raise RuntimeError("Metis model forward must return a tensor causal loss")
    auxiliary = _output_value(output, "auxiliary_loss")
    if auxiliary is None:
        return causal
    if not isinstance(auxiliary, torch.Tensor):
        raise RuntimeError("Metis model auxiliary_loss must be a tensor")
    return causal + auxiliary


def _normalized_telemetry(output: Any) -> dict[str, Any]:
    telemetry = scalar_telemetry(_output_value(output, "telemetry", {}))
    if "mean_passes" not in telemetry and "mean_depth" in telemetry:
        telemetry["mean_passes"] = telemetry["mean_depth"]
    if "overflow_drop_tokens" not in telemetry:
        telemetry["overflow_drop_tokens"] = int(
            telemetry.get(
                "moe_dropped_assignments",
                telemetry.get("dropped_tokens", 0),
            )
        )
    telemetry.setdefault("all_to_all_bytes", 0)
    telemetry.setdefault("all_to_all_seconds", 0.0)
    telemetry.setdefault("expert_load_cv", 0.0)
    return telemetry


def _set_seed(seed: int, *, rank: int) -> None:
    # Dense initialization must match before broadcast. Expert modules derive
    # their own global-expert seed inside the model implementation.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _all_reduce_numbers(
    values: Sequence[float],
    topology: ParallelTopology,
    device: torch.device,
) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    all_reduce_sum(tensor, topology)
    return [float(item) for item in tensor.cpu().tolist()]


def _model_forward(
    model: Any,
    batch: TrainingBatch,
    *,
    schedule_state: Any,
) -> Any:
    return model(
        batch.input_ids,
        labels=batch.labels,
        curriculum=schedule_state.model_curriculum(),
        attention_mask=batch.attention_mask,
        document_ids=batch.document_ids,
        reset_mask=batch.reset_mask,
        canonical_ids=batch.canonical_ids,
        max_passes=schedule_state.max_passes,
        force_depth=schedule_state.force_depth,
    )


def _synthetic_batch(
    *,
    cursor: int,
    rank: int,
    world_size: int,
    micro_batch: int,
    sequence_length: int,
    vocab_size: int,
) -> TrainingBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(16_062_026 + cursor + rank * 1_000_003)
    input_ids = torch.randint(
        0,
        vocab_size,
        (micro_batch, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    reset = torch.zeros_like(mask)
    reset[:, 0] = True
    document_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    emitted = world_size * micro_batch * sequence_length
    next_cursor = cursor + emitted
    return TrainingBatch(
        input_ids=input_ids,
        # Synthetic probes are the only intentional identity-canonicalization
        # path. Production batches must come from the released sidecar.
        canonical_ids=input_ids,
        labels=labels,
        attention_mask=mask,
        document_ids=document_ids,
        reset_mask=reset,
        phase="phase_a",
        global_token_cursor=cursor,
        next_global_token_cursor=next_cursor,
        non_padding_tokens=micro_batch * sequence_length,
        supervised_tokens=micro_batch * (sequence_length - 1),
    )


def _precision_storage_move(model: Any, device: torch.device) -> None:
    mover = getattr(model, "apply_parameter_storage_policy", None)
    if callable(mover):
        mover(device=device)
    else:
        model.to(device=device, dtype=torch.bfloat16)
    bad = [
        (name, str(parameter.dtype))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.dtype != torch.bfloat16
        and not (
            getattr(parameter, "metis_storage_dtype", None) == "float32"
            and parameter.dtype == torch.float32
        )
    ]
    if bad:
        raise RuntimeError(
            "Trainable model parameters must use BF16 storage; FP32 masters are "
            f"optimizer-owned. Invalid examples: {bad[:8]}"
        )


def _configure_sparse_sync(model: Any, topology: ParallelTopology, table_mode: str) -> None:
    enabler = getattr(model, "enable_managed_sparse_gradient_sync", None)
    if table_mode == "replicated" and topology.distributed:
        if not callable(enabler):
            raise RuntimeError("Replicated N-gram memory has no managed sparse synchronizer")
        enabler(topology.dense_data_group)
    elif (
        table_mode == "row_sharded"
        and topology.expert_replica_count > 1
    ):
        if not callable(enabler):
            raise RuntimeError("Row-sharded N-gram memory has no replica synchronizer")
        enabler(topology.expert_data_group)


def _compile_forward(model: Any, mode: str) -> Any:
    if mode in {"eager", "none"}:
        return model
    if mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise RuntimeError(f"Unsupported torch.compile mode: {mode}")
    if not hasattr(torch, "compile"):
        raise RuntimeError("Selected candidate requires torch.compile")
    return torch.compile(
        model,
        mode=mode,
        dynamic=True,
        fullgraph=False,
    )


def _selected_from_probe_args(args: argparse.Namespace, config: Any) -> AutotuneSelection:
    bounds = config.autotune
    learning_rate = float(
        args.lr_candidate
        if args.lr_candidate is not None
        else getattr(bounds, "preferred_learning_rate", bounds.learning_rates[0])
    )
    if not args.precision_role_plan:
        raise RuntimeError("Trainer probe requires a sealed precision role plan")
    role_plan = load_precision_role_plan(
        args.precision_role_plan,
        config=config,
    )
    measured_map = measured_role_dtype_map(role_plan)
    if args.precision_profile == "fp8" and "fp8" not in set(
        measured_map.values()
    ):
        raise RuntimeError("FP8 probe requested but no exact role qualified for FP8")
    return AutotuneSelection(
        family=args.family,
        micro_batch=int(args.micro_batch),
        grad_accum=int(args.grad_accum),
        learning_rate=learning_rate,
        precision_profile=str(args.precision_profile),
        compile_mode=str(args.compile_mode),
        dispatch_overlap=str(args.overlap_dispatch) == "on",
        ngram_table_mode=str(args.ngram_table_mode),
        profile_sha256="probe",
        environment_sha256="probe",
        release_marker_sha256="",
        precision_role_plan_sha256=str(role_plan["plan_sha256"]),
        precision_role_inventory_sha256=str(role_plan["inventory_sha256"]),
        measured_precision_role_map=tuple(sorted(measured_map.items())),
    )


def _validate_autotune_lineage(
    profile_path: str | Path,
    *,
    selection: AutotuneSelection,
    manifest_path: str | Path,
    training_contract_path: str | Path,
    release_marker: Mapping[str, Any],
    topology: ParallelTopology,
) -> None:
    payload = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    expected = {
        "manifest_sha256": sha256_file(manifest_path),
        "training_contract_sha256": sha256_file(training_contract_path),
        "release_marker_sha256": release_marker.get("marker_sha256"),
        "world_size": topology.world_size,
        "expert_parallel_size": topology.expert_parallel_size,
        "expert_replicas": topology.expert_replica_count,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(f"Autotune profile lineage mismatch for {field}")
    if selection.release_marker_sha256 != release_marker.get("marker_sha256"):
        raise RuntimeError("Autotune selection is not bound to this release audit")


def _construct_model(
    *,
    config: Any,
    topology: ParallelTopology,
    policy: PrecisionPolicy,
) -> Any:
    from .mhc_kernels import require_mhc_backend
    from .model import Metis16ForCausalLM, MetisProcessGroups

    require_mhc_backend(
        backend=config.mhc_backend,
        family=config.family,
        device=policy.device,
    )
    model = Metis16ForCausalLM(
        config,
        process_groups=MetisProcessGroups(
            world=topology.dense_data_group,
            expert=topology.expert_group,
            expert_data=topology.expert_data_group,
            table_lookup=topology.expert_group,
            table_gradient=(
                topology.dense_data_group
                if config.ngram_memory.table_mode == "replicated"
                else topology.expert_data_group
            ),
            context=topology.context_group,
        ),
        precision_policy=policy,
    )
    _precision_storage_move(model, policy.device)
    _configure_sparse_sync(model, topology, config.ngram_memory.table_mode)
    broadcast_initial_parameters(model, topology)
    return model


def _build_optimizer(
    model: Any,
    runtime_manifest: Mapping[str, Any],
    selection: AutotuneSelection,
    config: Any,
) -> tuple[OptimizerBundle, dict[str, Any]]:
    optimizer_config = runtime_manifest["optimizer"]
    muon = optimizer_config["muon"]
    optimizer, summary = build_training_optimizers(
        model,
        learning_rate=selection.learning_rate,
        beta1=float(optimizer_config["beta1"]),
        beta2=float(optimizer_config["beta2"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        sparse_learning_rate_scale=float(config.ngram_memory.learning_rate_scale),
        muon_beta=float(muon["beta"]),
        muon_ns_steps=int(muon["newton_schulz_steps"]),
        muon_nesterov=bool(muon["nesterov"]),
        include_routed_experts=bool(muon["include_routed_experts"]),
        muon_state_bits=int(muon.get("state_bits", 32)),
    )
    return optimizer, summary.to_dict()


def _write_base_checkpoint_receipt(
    *,
    checkpoint: Path,
    family: str,
    output_root: Path,
) -> Path:
    manifest_path = checkpoint / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt: dict[str, Any] = {
        "schema": "metis.checkpoint-receipt/v1",
        "family": family,
        "stage": "base_pretraining",
        "checkpoint_manifest": str(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "parent_checkpoint_sha256": None,
        "config_sha256": manifest["family_manifest_sha256"],
        "precision_role_plan_sha256": manifest[
            "precision_role_plan_sha256"
        ],
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = canonical_json_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    path = output_root / "BASE_PRETRAINING_RECEIPT.json"
    _atomic_json(path, receipt)
    return path


def _save_checkpoint(
    manager: CheckpointManager,
    *,
    model: Any,
    optimizer: OptimizerBundle,
    cursor: int,
    step: int,
    phase: str,
    shard_order_seed: int,
    inventory: ReleaseInventory,
    manifest_path: Path,
    runtime_path: Path,
    selection: AutotuneSelection,
    policy: PrecisionPolicy,
    signal_reason: str | None,
    phase_boundary: bool,
    extra_state: Mapping[str, Any] | None = None,
) -> Path:
    return manager.save(
        model=model,
        optimizer=optimizer,
        global_token_cursor=cursor,
        optimizer_step=step,
        phase=phase,
        shard_order_seed=shard_order_seed,
        release_sha256=inventory.release_sha256,
        shard_manifest_sha256=inventory.shard_manifest_sha256,
        family_manifest_sha256=sha256_file(manifest_path),
        runtime_manifest_sha256=sha256_file(runtime_path),
        autotune_profile_sha256=selection.profile_sha256,
        precision_role_plan_sha256=selection.precision_role_plan_sha256,
        precision_audit=policy.audit.to_dict(),
        signal_reason=signal_reason,
        phase_boundary=phase_boundary,
        extra_state=extra_state,
    )


def _restore_if_requested(
    *,
    mode: str,
    manager: CheckpointManager,
    model: Any,
    optimizer: OptimizerBundle,
    inventory: ReleaseInventory,
    manifest_path: Path,
    runtime_path: Path,
    selection: AutotuneSelection,
    profile_path: str | Path,
) -> tuple[int, int]:
    latest = manager.latest()
    if mode == "never":
        if latest is not None:
            raise RuntimeError("--resume never refuses an output directory with checkpoints")
        return 0, 0
    if latest is None:
        if mode == "required":
            raise RuntimeError("--resume required found no valid checkpoint")
        return 0, 0
    checkpoint_manifest_path = latest / "MANIFEST.json"
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint_profile_sha256 = str(
        checkpoint_manifest.get("autotune_profile_sha256", "")
    )
    if (
        checkpoint_manifest.get("precision_role_plan_sha256")
        != selection.precision_role_plan_sha256
    ):
        raise RuntimeError(
            "Checkpoint precision role plan differs from the selected exact-role plan"
        )
    expected_profile_sha256 = selection.profile_sha256
    if checkpoint_profile_sha256 != selection.profile_sha256:
        migration_raw = os.environ.get("METIS_AUTOTUNE_PROFILE_MIGRATION")
        if not migration_raw:
            raise RuntimeError(
                "Checkpoint autotune profile differs from the selected profile "
                "and no measured migration receipt was supplied"
            )
        migration_path = Path(migration_raw).expanduser().resolve()
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        unsigned_migration = {
            key: value
            for key, value in migration.items()
            if key != "receipt_sha256"
        }
        if (
            migration.get("schema")
            != "metis.autotune-profile-migration/v1"
            or migration.get("receipt_sha256")
            != canonical_json_sha256(unsigned_migration)
        ):
            raise RuntimeError("Autotune profile migration receipt is invalid")
        resolved_profile = Path(profile_path).expanduser().resolve()
        resolved_receipt_profile = Path(
            str(migration.get("new_profile_path", ""))
        ).expanduser().resolve()
        current_profile_payload = json.loads(
            resolved_profile.read_text(encoding="utf-8")
        )
        old_selected = migration.get("old_selected")
        new_selected = migration.get("new_selected")
        if not isinstance(old_selected, Mapping) or not isinstance(
            new_selected, Mapping
        ):
            raise RuntimeError("Autotune migration omits selected candidate state")
        changed_fields = {
            key
            for key in set(old_selected) | set(new_selected)
            if old_selected.get(key) != new_selected.get(key)
        }
        allowed_changes = migration.get("allowed_changes")
        expected_changes = {"micro_batch_size", "grad_accum_steps"}
        if (
            migration.get("family") != selection.family
            or Path(str(migration.get("checkpoint", ""))).expanduser().resolve()
            != latest.resolve()
            or migration.get("checkpoint_sha256")
            != checkpoint_manifest.get("checkpoint_sha256")
            or migration.get("checkpoint_manifest_sha256")
            != sha256_file(checkpoint_manifest_path)
            or migration.get("old_profile_sha256")
            != checkpoint_profile_sha256
            or resolved_receipt_profile != resolved_profile
            or migration.get("new_profile_sha256")
            != selection.profile_sha256
            or migration.get("precision_role_plan_sha256")
            != selection.precision_role_plan_sha256
            or current_profile_payload.get("profile_sha256")
            != selection.profile_sha256
            or current_profile_payload.get("selected") != dict(new_selected)
            or allowed_changes
            != ["micro_batch_size", "grad_accum_steps"]
            or changed_fields != expected_changes
            or migration.get("global_token_batch_unchanged") is not True
            or migration.get("state_conversion")
            != "none_parameter_and_optimizer_layout_unchanged"
            or int(old_selected["micro_batch_size"])
            * int(old_selected["grad_accum_steps"])
            != int(new_selected["micro_batch_size"])
            * int(new_selected["grad_accum_steps"])
        ):
            raise RuntimeError(
                "Autotune profile migration is not bound to this exact "
                "checkpoint and checkpoint-compatible micro-batch change"
            )
        expected_profile_sha256 = checkpoint_profile_sha256
    state = manager.load(
        latest,
        model=model,
        optimizer=optimizer,
        expected_release_sha256=inventory.release_sha256,
        expected_shard_manifest_sha256=inventory.shard_manifest_sha256,
        expected_family_manifest_sha256=sha256_file(manifest_path),
        expected_runtime_manifest_sha256=sha256_file(runtime_path),
        expected_autotune_profile_sha256=expected_profile_sha256,
        expected_precision_role_plan_sha256=(
            selection.precision_role_plan_sha256
        ),
    )
    return state.global_token_cursor, state.optimizer_step


def _run_steps(
    *,
    args: argparse.Namespace,
    config: Any,
    runtime_manifest: Mapping[str, Any],
    runtime: Runtime,
    topology: ParallelTopology,
    policy: PrecisionPolicy,
    model: Any,
    forward_model: Any,
    optimizer: OptimizerBundle,
    schedule: TokenSchedule,
    stream: DeterministicReleaseStream | None,
    inventory: ReleaseInventory | None,
    selection: AutotuneSelection,
    cursor: int,
    optimizer_step: int,
    checkpoint_manager: CheckpointManager | None,
    signal_coordinator: SignalCoordinator,
    manifest_path: Path,
    runtime_path: Path,
) -> tuple[dict[str, Any], Path | None]:
    health = runtime_manifest["health"]
    checkpoint_config = runtime_manifest["checkpoint"]
    data_config = runtime_manifest["data"]
    probe = bool(args.probe)
    target_steps = int(args.probe_steps) if probe else None
    start_cursor = cursor
    initial_loss: float | None = None
    final_loss: float | None = None
    bf16_reference_loss: float | None = None
    fp8_actual_loss: float | None = None
    max_grad_norm = 0.0
    nonfinite_steps = 0
    total_step_seconds = 0.0
    timed_tokens = 0
    total_estimated_flops = 0.0
    collective_errors = 0
    overflow_drop_tokens = 0
    update_before = sample_parameters(model) if probe and args.lr_candidate is not None else {}
    # Reducing dense gradients during backward, rather than in a serial phase
    # after it, hides the whole replicated all-reduce behind compute.
    gradient_reducer = (
        OverlappedGradientReducer(model, topology) if topology.distributed else None
    )
    last_checkpoint: Path | None = None
    last_checkpoint_cursor: int | None = None
    next_checkpoint = (
        ((cursor // int(checkpoint_config["interval_tokens"])) + 1)
        * int(checkpoint_config["interval_tokens"])
    )
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)

    # Ranks that jointly own one sequence must read the same batch and then
    # keep different windows of it, so the loader is sharded by
    # context-parallel *group* and the split happens after the fetch.  With
    # context parallelism off these are the runtime's own rank and world size.
    context_parallel_size = topology.context_parallel_size
    data_rank = runtime.rank // context_parallel_size
    data_world_size = runtime.world_size // context_parallel_size

    prefetch_context: Any
    if stream is not None:
        prefetch_context = ReleaseBatchPrefetcher(
            stream,
            start_cursor=cursor,
            rank=data_rank,
            world_size=data_world_size,
            micro_batch_size=selection.micro_batch,
            depth=int(data_config["prefetch_micro_batches"]),
        )
    else:
        prefetch_context = nullcontext(None)

    metrics = MetricsWriter(
        args.output,
        enabled=not probe,
        rank=runtime.rank,
    )
    with prefetch_context as prefetcher, metrics:
        while (probe and optimizer_step < target_steps) or (
            not probe and cursor < TOTAL_TOKENS
        ):
            signal_seen = global_any(
                signal_coordinator.requested, topology, runtime.device
            )
            if signal_seen and not probe:
                if (
                    last_checkpoint is not None
                    and last_checkpoint_cursor == cursor
                ):
                    return {
                        "requeue": True,
                        "cursor": cursor,
                        "step": optimizer_step,
                    }, last_checkpoint
                if (
                    checkpoint_manager is None
                    or inventory is None
                    or stream is None
                ):
                    raise RuntimeError(
                        "Signal checkpoint requested without production checkpoint state"
                    )
                data_position = stream.position(cursor)
                last_checkpoint = _save_checkpoint(
                    checkpoint_manager,
                    model=model,
                    optimizer=optimizer,
                    cursor=cursor,
                    step=optimizer_step,
                    phase=str(data_position["phase"]),
                    shard_order_seed=int(data_config["shard_order_seed"]),
                    inventory=inventory,
                    manifest_path=manifest_path,
                    runtime_path=runtime_path,
                    selection=selection,
                    policy=policy,
                    signal_reason=signal_coordinator.reason or "remote-rank-signal",
                    phase_boundary=cursor
                    in {
                        PHASE_STARTS["phase_b"],
                        PHASE_STARTS["phase_c"],
                        TOTAL_TOKENS,
                    },
                    extra_state={"data_position": data_position},
                )
                last_checkpoint_cursor = cursor
                return {"requeue": True, "cursor": cursor, "step": optimizer_step}, last_checkpoint

            optimizer.zero_grad(set_to_none=True)
            state = schedule.state(cursor)
            set_optimizer_learning_rate(optimizer, state.learning_rate)
            step_phase = state.phase
            phase_end = PHASE_STARTS[step_phase] + PHASE_TOKENS[step_phase]
            maximum_step_tokens = (
                runtime.world_size
                * selection.micro_batch
                * config.sequence_length
                * selection.grad_accum
            )
            sample_collective_timing = bool(
                probe
                or (optimizer_step + 1)
                % int(health["telemetry_interval_steps"])
                == 0
                or cursor + maximum_step_tokens >= phase_end
            )
            collective_timing_setter = getattr(
                model, "enable_collective_event_timing", None
            )
            if callable(collective_timing_setter):
                collective_timing_setter(sample_collective_timing)
            global_supervised = 0
            global_non_padding = 0
            global_loss_numerator = 0.0
            step_telemetry: dict[str, Any] = {}
            accumulated_telemetry = {
                "all_to_all_bytes": 0.0,
                "all_to_all_seconds": 0.0,
                "all_to_all_enqueue_seconds": 0.0,
                "moe_assignments": 0.0,
                "moe_processed_assignments": 0.0,
                "moe_dropped_assignments": 0.0,
                "overflow_drop_tokens": 0.0,
            }
            expert_selection_counts: torch.Tensor | None = None
            accumulation_count = 0
            step_started = time.perf_counter()
            for _accumulation_index in range(selection.grad_accum):
                if (
                    gradient_reducer is not None
                    and _accumulation_index == selection.grad_accum - 1
                ):
                    # Only the last micro-step holds the final gradient, so only
                    # its backward may reduce.  An accumulation loop that exits
                    # early never arms the reducer and falls back to the
                    # synchronous path inside synchronize_gradients.
                    gradient_reducer.arm()
                if stream is None:
                    cpu_batch = _synthetic_batch(
                        cursor=cursor,
                        rank=data_rank,
                        world_size=data_world_size,
                        micro_batch=selection.micro_batch,
                        sequence_length=config.sequence_length,
                        vocab_size=config.vocab_size,
                    )
                else:
                    cpu_batch = prefetcher.next(expected_cursor=cursor)
                if cpu_batch.phase != step_phase:
                    raise RuntimeError("An optimizer batch attempted to cross a phase boundary")
                cpu_batch = cpu_batch.shard_for_context_parallel(
                    topology.context_rank, context_parallel_size
                )
                batch = cpu_batch.to(runtime.device)

                if (
                    probe
                    and policy.fp8_enabled
                    and bf16_reference_loss is None
                ):
                    cpu_rng = torch.get_rng_state()
                    cuda_rng = (
                        torch.cuda.get_rng_state(runtime.device)
                        if runtime.device.type == "cuda"
                        else None
                    )
                    with torch.no_grad(), policy.bf16_reference_context():
                        reference_output = _model_forward(
                            model, batch, schedule_state=state
                        )
                    reference_loss = _combined_training_loss(reference_output)
                    ref_sum, ref_count = _all_reduce_numbers(
                        [
                            float(reference_loss.detach().float().item())
                            * batch.supervised_tokens,
                            float(batch.supervised_tokens),
                        ],
                        topology,
                        runtime.device,
                    )
                    bf16_reference_loss = ref_sum / max(ref_count, 1.0)
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, runtime.device)

                output = _model_forward(forward_model, batch, schedule_state=state)
                loss = _combined_training_loss(output)
                raw_telemetry = _output_value(output, "telemetry", {})
                batch_selection_counts = (
                    raw_telemetry.get("expert_selection_counts")
                    if isinstance(raw_telemetry, Mapping)
                    else None
                )
                if batch_selection_counts is not None:
                    if not isinstance(batch_selection_counts, torch.Tensor):
                        raise RuntimeError(
                            "expert_selection_counts telemetry must be a tensor"
                        )
                    detached_counts = batch_selection_counts.detach()
                    expert_selection_counts = (
                        detached_counts.clone()
                        if expert_selection_counts is None
                        else expert_selection_counts + detached_counts
                    )
                telemetry = _normalized_telemetry(output)
                step_telemetry.update(telemetry)
                for telemetry_name in accumulated_telemetry:
                    accumulated_telemetry[telemetry_name] += float(
                        telemetry.get(telemetry_name, 0.0)
                    )
                local_loss = float(loss.detach().float().item())
                local_values = [
                    local_loss * batch.supervised_tokens,
                    float(batch.supervised_tokens),
                    float(batch.non_padding_tokens),
                ]
                loss_sum, supervised, non_padding = _all_reduce_numbers(
                    local_values, topology, runtime.device
                )
                global_loss_numerator += loss_sum
                global_supervised += int(supervised)
                global_non_padding += int(non_padding)
                if (
                    bf16_reference_loss is not None
                    and fp8_actual_loss is None
                    and supervised > 0
                ):
                    fp8_actual_loss = loss_sum / supervised
                if batch.supervised_tokens > 0:
                    (loss * float(batch.supervised_tokens)).backward()
                else:
                    (loss * 0.0).backward()
                expected_delta = cpu_batch.next_global_token_cursor - cursor
                if int(non_padding) != expected_delta:
                    raise RuntimeError(
                        "Global emitted-token accounting diverged from the immutable stream"
                    )
                cursor = cpu_batch.next_global_token_cursor
                accumulation_count += 1
                if cursor >= phase_end or (probe and accumulation_count >= selection.grad_accum):
                    break

            for telemetry_name, telemetry_value in accumulated_telemetry.items():
                step_telemetry[telemetry_name] = telemetry_value
            if callable(collective_timing_setter):
                collective_timing_setter(False)
            if global_supervised <= 0:
                optimizer.zero_grad(set_to_none=True)
                if cursor >= TOTAL_TOKENS:
                    break
                raise RuntimeError("Optimizer batch contained no supervised tokens")
            try:
                synchronize_gradients(model, topology, reducer=gradient_reducer)
            except RuntimeError:
                collective_errors += 1
                raise
            normalize_summed_gradients(
                model,
                topology,
                global_supervised_tokens=global_supervised,
            )
            grad_norm_tensor = clip_grad_norm_(
                model,
                float(runtime_manifest["optimizer"]["grad_clip"]),
                topology=topology,
            )
            grad_norm_value = float(grad_norm_tensor.detach().item())
            max_grad_norm = max(max_grad_norm, grad_norm_value)
            step_loss = global_loss_numerator / global_supervised
            if not math.isfinite(step_loss) or not math.isfinite(grad_norm_value):
                nonfinite_steps += 1
            overflow_drop_tokens += int(
                step_telemetry.get(
                    "overflow_drop_tokens",
                    step_telemetry.get("dropped_tokens", 0),
                )
            )
            enforce_health_gates(
                loss=step_loss,
                grad_norm_value=grad_norm_value,
                telemetry=step_telemetry,
                maximum_grad_norm=float(config.autotune.gates.max_grad_norm),
                abort_on_nonfinite=bool(health["abort_on_nonfinite"]),
                abort_on_token_drop=bool(health["abort_on_token_drop"]),
                minimum_expert_entropy_ratio=float(health["minimum_expert_entropy_ratio"]),
                maximum_expert_load_cv=float(health["maximum_expert_load_cv"]),
                maximum_halt_collapse_fraction=float(
                    health["maximum_halt_collapse_fraction"]
                ),
                require_structural_telemetry=bool(
                    health["require_structural_telemetry"]
                ),
                maximum_sinkhorn_marginal_error=float(
                    health["maximum_sinkhorn_marginal_error"]
                ),
                maximum_ponder_exit_mass_error=float(
                    health["maximum_ponder_exit_mass_error"]
                ),
                minimum_mhc_stream_diversity=float(
                    health["minimum_mhc_stream_diversity"]
                ),
            )
            optimizer.step()
            selection_bias_updater = getattr(
                model, "update_expert_selection_biases", None
            )
            if callable(selection_bias_updater):
                if expert_selection_counts is None:
                    raise RuntimeError(
                        "Metis routing did not return per-layer expert selection counts"
                    )
                selection_bias_updater(expert_selection_counts)
            optimizer_step += 1
            post_step_signal_seen = global_any(
                signal_coordinator.requested, topology, runtime.device
            )
            if post_step_signal_seen and not probe:
                if (
                    checkpoint_manager is None
                    or inventory is None
                    or stream is None
                ):
                    raise RuntimeError(
                        "Signal checkpoint requested without production checkpoint state"
                    )
                phase_boundary = cursor in {
                    PHASE_STARTS["phase_b"],
                    PHASE_STARTS["phase_c"],
                    TOTAL_TOKENS,
                }
                data_position = stream.position(cursor)
                last_checkpoint = _save_checkpoint(
                    checkpoint_manager,
                    model=model,
                    optimizer=optimizer,
                    cursor=cursor,
                    step=optimizer_step,
                    phase=str(data_position["phase"]),
                    shard_order_seed=int(data_config["shard_order_seed"]),
                    inventory=inventory,
                    manifest_path=manifest_path,
                    runtime_path=runtime_path,
                    selection=selection,
                    policy=policy,
                    signal_reason=(
                        signal_coordinator.reason or "remote-rank-signal"
                    ),
                    phase_boundary=phase_boundary,
                    extra_state={"data_position": data_position},
                )
                last_checkpoint_cursor = cursor
                return {
                    "requeue": True,
                    "cursor": cursor,
                    "step": optimizer_step,
                }, last_checkpoint
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            elapsed = time.perf_counter() - step_started
            estimated_flops = estimate_train_flops(
                config,
                tokens=global_non_padding,
                observed_mean_passes=step_telemetry.get("mean_passes"),
                observed_mean_routed_k=step_telemetry.get("mean_routed_k"),
            )
            warmup_step = probe and optimizer_step == 1
            if not warmup_step:
                total_step_seconds += elapsed
                timed_tokens += global_non_padding
                total_estimated_flops += estimated_flops
            initial_loss = step_loss if initial_loss is None else initial_loss
            final_loss = step_loss
            if (
                not probe
                and (
                    optimizer_step % int(health["telemetry_interval_steps"]) == 0
                    or cursor
                    in {
                        PHASE_STARTS["phase_b"],
                        PHASE_STARTS["phase_c"],
                        TOTAL_TOKENS,
                    }
                )
            ):
                memory_evidence = peak_memory_evidence(runtime.device)
                metrics.write(
                    StepMetrics(
                        optimizer_step=optimizer_step,
                        global_token_cursor=cursor,
                        phase=step_phase,
                        loss=step_loss,
                        learning_rate=state.learning_rate,
                        global_non_padding_tokens=global_non_padding,
                        global_supervised_tokens=global_supervised,
                        step_time_s=elapsed,
                        tokens_per_second=global_non_padding / max(elapsed, 1.0e-9),
                        estimated_train_flops=estimated_flops,
                        estimated_mfu=estimated_mfu(
                            estimated_flops=estimated_flops,
                            elapsed_seconds=elapsed,
                            world_size=runtime.world_size,
                            precision_profile=policy.effective_profile,
                        ),
                        grad_norm=grad_norm_value,
                        update_to_weight_ratio=None,
                        peak_hbm_bytes=memory_evidence["peak_hbm_bytes"],
                        precision_profile=policy.effective_profile,
                        overflow_drop_tokens=int(
                            step_telemetry.get("overflow_drop_tokens", 0)
                        ),
                        collective_errors=collective_errors,
                        telemetry=step_telemetry,
                        peak_hbm_allocated_bytes=memory_evidence[
                            "peak_hbm_allocated_bytes"
                        ],
                        peak_hbm_reserved_bytes=memory_evidence[
                            "peak_hbm_reserved_bytes"
                        ],
                    )
                )

            phase_boundary = cursor in {
                PHASE_STARTS["phase_b"],
                PHASE_STARTS["phase_c"],
                TOTAL_TOKENS,
            }
            if (
                not probe
                and checkpoint_manager is not None
                and inventory is not None
                and (cursor >= next_checkpoint or phase_boundary)
            ):
                pre_checkpoint_signal_seen = global_any(
                    signal_coordinator.requested, topology, runtime.device
                )
                if pre_checkpoint_signal_seen:
                    if stream is None:
                        raise RuntimeError(
                            "Signal checkpoint requested without a production stream"
                        )
                    data_position = stream.position(cursor)
                    last_checkpoint = _save_checkpoint(
                        checkpoint_manager,
                        model=model,
                        optimizer=optimizer,
                        cursor=cursor,
                        step=optimizer_step,
                        phase=str(data_position["phase"]),
                        shard_order_seed=int(data_config["shard_order_seed"]),
                        inventory=inventory,
                        manifest_path=manifest_path,
                        runtime_path=runtime_path,
                        selection=selection,
                        policy=policy,
                        signal_reason=(
                            signal_coordinator.reason or "remote-rank-signal"
                        ),
                        phase_boundary=phase_boundary,
                        extra_state={"data_position": data_position},
                    )
                    last_checkpoint_cursor = cursor
                    return {
                        "requeue": True,
                        "cursor": cursor,
                        "step": optimizer_step,
                    }, last_checkpoint
                if stream is None:
                    raise RuntimeError(
                        "Ordinary production checkpoint has no release stream"
                    )
                data_position = stream.position(cursor)
                last_checkpoint = _save_checkpoint(
                    checkpoint_manager,
                    model=model,
                    optimizer=optimizer,
                    cursor=cursor,
                    step=optimizer_step,
                    phase=str(data_position["phase"]),
                    shard_order_seed=int(data_config["shard_order_seed"]),
                    inventory=inventory,
                    manifest_path=manifest_path,
                    runtime_path=runtime_path,
                    selection=selection,
                    policy=policy,
                    signal_reason=None,
                    phase_boundary=phase_boundary,
                    extra_state={"data_position": data_position},
                )
                last_checkpoint_cursor = cursor
                while next_checkpoint <= cursor:
                    next_checkpoint += int(checkpoint_config["interval_tokens"])
                post_checkpoint_signal_seen = global_any(
                    signal_coordinator.requested, topology, runtime.device
                )
                if post_checkpoint_signal_seen:
                    return {
                        "requeue": True,
                        "cursor": cursor,
                        "step": optimizer_step,
                    }, last_checkpoint

    update_ratio = (
        sampled_update_to_weight_ratio(update_before, model) if update_before else 0.0
    )
    relative_error = (
        abs(fp8_actual_loss - bf16_reference_loss)
        / max(abs(bf16_reference_loss), 1.0e-12)
        if bf16_reference_loss is not None and fp8_actual_loss is not None
        else 0.0
    )
    memory_evidence = peak_memory_evidence(runtime.device)
    result = {
        "requeue": False,
        "cursor": cursor,
        "step": optimizer_step,
        "initial_loss": float(initial_loss or 0.0),
        "final_loss": float(final_loss or 0.0),
        "max_grad_norm": max_grad_norm,
        "nonfinite_steps": nonfinite_steps,
        "update_to_weight_ratio": update_ratio,
        "step_time_s": total_step_seconds,
        "non_padding_tokens": timed_tokens,
        "estimated_train_flops": total_estimated_flops,
        **memory_evidence,
        "overflow_drop_tokens": overflow_drop_tokens,
        "collective_errors": collective_errors,
        "finite_loss": nonfinite_steps == 0,
        "loss_relative_error_vs_bf16": relative_error,
        "bf16_reference_loss": bf16_reference_loss,
        "fp8_actual_loss": fp8_actual_loss,
        "start_cursor": start_cursor,
    }
    return result, last_checkpoint


def _audit(args: argparse.Namespace) -> int:
    from .model import load_family_config

    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = load_family_manifest(manifest_path)
    config = load_family_config(
        manifest_path,
        family=args.family,
        materialize_tables=False,
    )
    if os.environ.get("METIS_PORTAGE_PREFLIGHT_ONLY") == "1":
        marker: Mapping[str, Any] = {
            "status": "pending_distributed_gate",
            "training_permitted": False,
        }
    else:
        marker = require_release_verification(args.data_release)
    report = {
        "schema": "metis.trainer-audit/v1",
        "ok": True,
        "family": args.family,
        "world_size": int(payload["topology"]["world_size"]),
        "expert_parallel_size": int(payload["topology"]["expert_parallel_size"]),
        "expert_replicas": int(payload["topology"]["expert_replica_count"]),
        "manifest_sha256": sha256_file(manifest_path),
        "release": marker,
        "parameter_audit": config.logical_parameter_audit().to_dict(),
        "precision": dataclasses.asdict(config.precision),
        "autotune": dataclasses.asdict(config.autotune),
    }
    _atomic_json(args.json_output, report)
    return 0


def _run(args: argparse.Namespace) -> int:
    from .model import load_family_config

    manifest_path = Path(args.manifest).expanduser().resolve()
    runtime_path = Path(args.runtime_manifest).expanduser().resolve()
    family_manifest = load_family_manifest(manifest_path)
    config = load_family_config(
        manifest_path,
        family=args.family,
        materialize_tables=True,
    )
    runtime_manifest = _load_runtime(runtime_path)
    runtime = initialize_runtime()
    production_shape = runtime.world_size == int(family_manifest["topology"]["world_size"])
    if not production_shape and not (args.probe and args.synthetic_probe):
        raise RuntimeError(
            f"{args.family} launch has {runtime.world_size} ranks but its production "
            f"manifest requires {family_manifest['topology']['world_size']}"
        )
    topology = build_parallel_topology(
        runtime,
        family=args.family,
        routed_experts=config.n_routed_experts,
        production=production_shape,
        context_parallel_size=config.context_parallel_size,
    )
    _set_seed(16_062_026, rank=runtime.rank)

    if args.probe:
        selection = _selected_from_probe_args(args, config)
        release_marker: Mapping[str, Any] = {}
    else:
        if not args.autotune_profile:
            raise RuntimeError("Production training requires a measured autotune profile")
        selection = load_autotune_selection(
            args.autotune_profile,
            family_manifest=family_manifest,
        )
        release_marker = {}

    if args.ngram_table_mode is not None and (
        str(args.ngram_table_mode) != selection.ngram_table_mode
    ):
        raise RuntimeError(
            "--ngram-table-mode disagrees with the measured autotune profile"
        )
    config = dataclasses.replace(
        config,
        ngram_memory=dataclasses.replace(
            config.ngram_memory,
            table_mode=selection.ngram_table_mode,
        ),
    )
    config.validate()

    if args.synthetic_probe:
        if not args.probe:
            raise RuntimeError("--synthetic-probe is legal only with --probe")
        inventory = None
        stream = None
    else:
        marker_path = os.environ.get("METIS_RELEASE_VERIFICATION_MARKER")
        release_marker = require_release_verification(
            args.data_release,
            marker_path=marker_path,
        )
        inventory = ReleaseInventory.from_release_root(args.data_release)
        stream = DeterministicReleaseStream(
            inventory,
            sequence_length=config.sequence_length,
            shard_order_seed=int(runtime_manifest["data"]["shard_order_seed"]),
            mmap_cache_shards=int(runtime_manifest["data"]["mmap_cache_shards"]),
            mask_cross_document_targets=bool(
                runtime_manifest["data"]["mask_cross_document_targets"]
            ),
        )
        if not args.probe:
            marker_payload = json.loads(Path(marker_path).read_text(encoding="utf-8"))
            _validate_autotune_lineage(
                args.autotune_profile,
                selection=selection,
                manifest_path=manifest_path,
                training_contract_path=REPOSITORY_ROOT
                / str(runtime_manifest["data_contract"]),
                release_marker=marker_payload,
                topology=topology,
            )

    policy = build_precision_policy(
        config.precision,
        profile=selection.precision_profile,
        device=runtime.device,
        production=not args.probe,
        permit_fallback=False,
        measured_role_dtypes=selection.measured_role_dtypes,
        precision_role_plan_sha256=selection.precision_role_plan_sha256,
    )
    if policy.fp8_enabled:
        policy.validate_execution()
    model = _construct_model(config=config, topology=topology, policy=policy)
    overlap_setter = getattr(model, "set_dispatch_overlap", None)
    if callable(overlap_setter):
        overlap_setter(selection.dispatch_overlap)
    optimizer, optimizer_summary = _build_optimizer(
        model, runtime_manifest, selection, config
    )
    forward_model = _compile_forward(model, selection.compile_mode)
    schedule = TokenSchedule(runtime_manifest, base_learning_rate=selection.learning_rate)
    output_root = Path(args.output).expanduser().resolve()
    if runtime.rank == 0:
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output_root / "RUN_CONFIG.json",
            {
                "schema": "metis.training-run/v1",
                "family": args.family,
                "manifest_sha256": sha256_file(manifest_path),
                "runtime_manifest_sha256": sha256_file(runtime_path),
                "autotune": selection.__dict__,
                "precision": policy.audit.to_dict(),
                "optimizer": optimizer_summary,
            },
        )
    barrier(topology)

    checkpoint_manager = (
        CheckpointManager(
            output_root,
            topology=topology,
            keep_last=int(runtime_manifest["checkpoint"]["keep_last"]),
            max_staging_bytes=int(
                runtime_manifest["checkpoint"]["max_staging_bytes"]
            ),
        )
        if not args.probe
        else None
    )
    if args.probe:
        cursor, optimizer_step = 0, 0
    else:
        assert checkpoint_manager is not None and inventory is not None
        cursor, optimizer_step = _restore_if_requested(
            mode=args.resume,
            manager=checkpoint_manager,
            model=model,
            optimizer=optimizer,
            inventory=inventory,
            manifest_path=manifest_path,
            runtime_path=runtime_path,
            selection=selection,
            profile_path=args.autotune_profile,
        )
    model.train()
    with SignalCoordinator() as signals:
        result, last_checkpoint = _run_steps(
            args=args,
            config=config,
            runtime_manifest=runtime_manifest,
            runtime=runtime,
            topology=topology,
            policy=policy,
            model=model,
            forward_model=forward_model,
            optimizer=optimizer,
            schedule=schedule,
            stream=stream,
            inventory=inventory,
            selection=selection,
            cursor=cursor,
            optimizer_step=optimizer_step,
            checkpoint_manager=checkpoint_manager,
            signal_coordinator=signals,
            manifest_path=manifest_path,
            runtime_path=runtime_path,
        )
    if stream is not None:
        stream.close()

    if args.probe:
        report = {
            "schema": "metis.trainer-probe/v1",
            "ok": bool(
                result["finite_loss"]
                and result["overflow_drop_tokens"] == 0
                and result["collective_errors"] == 0
            ),
            "family": args.family,
            "precision_profile": policy.effective_profile,
            "precision_role_plan_sha256": (
                selection.precision_role_plan_sha256
            ),
            "measured_precision_role_map": (
                selection.measured_role_dtypes
            ),
            "ngram_table_mode": config.ngram_memory.table_mode,
            **result,
        }
        if runtime.rank == 0:
            _atomic_json(args.json_output, report)
        barrier(topology)
        return 0 if report["ok"] else 2

    if result["requeue"]:
        return REQUEUE_EXIT_CODE
    if last_checkpoint is None and checkpoint_manager is not None:
        last_checkpoint = checkpoint_manager.latest()
    if result["cursor"] != TOTAL_TOKENS or last_checkpoint is None:
        raise RuntimeError("Base pretraining ended without the exact 1T final checkpoint")
    receipt: Path | None = None
    if runtime.rank == 0:
        receipt = _write_base_checkpoint_receipt(
            checkpoint=last_checkpoint,
            family=args.family,
            output_root=output_root,
        )
    barrier(topology)
    posttraining_result: dict[str, Any] | None = None
    if args.stage == "all":
        from .stage_backend import run_posttraining_campaign

        with SignalCoordinator() as posttraining_signals:
            posttraining_result = run_posttraining_campaign(
                args=args,
                config=config,
                model=model,
                optimizer=optimizer,
                policy=policy,
                runtime=runtime,
                topology=topology,
                family_manifest=family_manifest,
                base_checkpoint=last_checkpoint,
                base_receipt=output_root / "BASE_PRETRAINING_RECEIPT.json",
                posttraining_manifest=Path(args.posttraining_manifest),
                signal_coordinator=posttraining_signals,
            )
        if not bool(posttraining_result.get("complete")):
            raise RuntimeError("Post-training campaign returned without completion")
    if runtime.rank == 0:
        completion: dict[str, Any] = {
            "schema": "metis.training-complete/v1",
            "family": args.family,
            "stage": args.stage,
            "global_token_cursor": TOTAL_TOKENS,
            "base_checkpoint": str(last_checkpoint),
            "base_checkpoint_receipt": str(
                output_root / "BASE_PRETRAINING_RECEIPT.json"
            ),
            "ok": True,
        }
        if posttraining_result is not None:
            completion["posttraining"] = {
                "policy_checkpoint_sha256": posttraining_result.get(
                    "policy_checkpoint_sha256"
                ),
                "policy_checkpoint_receipt": posttraining_result.get(
                    "policy_checkpoint_receipt"
                ),
                "evaluation_receipt": posttraining_result.get("evaluation_receipt"),
                "release_candidate": posttraining_result.get("release_candidate"),
            }
        _atomic_json(
            output_root / "COMPLETE.json",
            completion,
        )
    barrier(topology)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Metis-1.6 FP8-first autonomous Portage trainer."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", choices=("praxis", "logos"), required=True)
    parser.add_argument(
        "--stage",
        choices=("pretrain", "all"),
        default="pretrain",
        help=(
            "Run exact 1T base pretraining only, or the lineage-bound full "
            "base -> context-extension -> post-training campaign. Partial "
            "context/post-training entry is intentionally rejected until an "
            "explicit parent-checkpoint handoff is supplied."
        ),
    )
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--autotune-profile")
    parser.add_argument("--precision-role-plan")
    parser.add_argument("--runtime-manifest", default=str(DEFAULT_RUNTIME_MANIFEST))
    parser.add_argument(
        "--posttraining-manifest", default=str(DEFAULT_POSTTRAINING_MANIFEST)
    )
    parser.add_argument("--audit-config", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--precision-profile", choices=("fp8", "bf16"))
    parser.add_argument(
        "--compile-mode",
        choices=("eager", "default", "reduce-overhead", "max-autotune"),
    )
    parser.add_argument("--overlap-dispatch", choices=("on", "off"))
    parser.add_argument(
        "--ngram-table-mode",
        choices=("replicated", "row_sharded"),
    )
    parser.add_argument("--lr-candidate", type=float)
    parser.add_argument("--synthetic-probe", action="store_true")
    parser.add_argument("--json-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.audit_config:
        if not args.json_output:
            raise SystemExit("--audit-config requires --json-output")
        return _audit(args)
    if args.probe:
        required = {
            "--json-output": args.json_output,
            "--micro-batch": args.micro_batch,
            "--grad-accum": args.grad_accum,
            "--precision-profile": args.precision_profile,
            "--compile-mode": args.compile_mode,
            "--overlap-dispatch": args.overlap_dispatch,
            "--ngram-table-mode": args.ngram_table_mode,
            "--precision-role-plan": args.precision_role_plan,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise SystemExit("probe is missing: " + ", ".join(missing))
        if args.probe_steps <= 0:
            raise SystemExit("--probe-steps must be positive")
    try:
        return _run(args)
    except BaseException as exc:
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        if args.probe and args.json_output and rank == 0:
            _atomic_json(
                args.json_output,
                {
                    "schema": "metis.trainer-probe/v1",
                    "ok": False,
                    "finite_loss": False,
                    "failure": f"{type(exc).__name__}: {exc}",
                },
            )
        raise
    finally:
        destroy_runtime()


if __name__ == "__main__":
    raise SystemExit(main())
