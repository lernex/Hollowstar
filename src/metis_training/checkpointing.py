from __future__ import annotations

import json
import os
import random
import re
import shutil
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from .contracts import canonical_json_sha256, sha256_file
from .distributed import ParallelTopology, barrier
from .optimizers import OptimizerBundle


CHECKPOINT_SCHEMA = "metis.distributed-checkpoint/v1"
CHECKPOINT_LAYOUT = "metis.tensor-chunks/v1"
DEFAULT_MAX_STAGING_BYTES = 256 * 1024 * 1024
_CHECKPOINT_PATTERN = re.compile(r"^tokens-(\d{13})$")


@dataclass(frozen=True)
class ResumeState:
    global_token_cursor: int
    optimizer_step: int
    phase: str
    shard_order_seed: int
    checkpoint_path: str
    checkpoint_sha256: str
    autotune_profile_sha256: str
    precision_role_plan_sha256: str
    release_sha256: str
    signal_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _LoadedCheckpointProof:
    checkpoint_path: str
    checkpoint_sha256: str
    global_token_cursor: int
    optimizer_step: int
    phase: str


def _placements(model: nn.Module) -> dict[str, str]:
    provider = getattr(model, "parameter_placements", None)
    if not callable(provider):
        return {name: "replicated" for name, _ in model.named_parameters()}
    result = {str(name): str(value) for name, value in provider().items()}
    names = {name for name, _ in model.named_parameters()}
    if names != set(result):
        raise RuntimeError("Model parameter placement inventory is incomplete")
    return result


def _atomic_torch_save(
    payload: Any,
    path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": (
            str(path.relative_to(artifact_root))
            if artifact_root is not None
            else path.name
        ),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _optimizer_state_by_name(
    optimizer: OptimizerBundle,
    named_parameters: Mapping[str, nn.Parameter],
    selected_names: set[str],
) -> dict[str, Any]:
    reverse = {id(parameter): name for name, parameter in named_parameters.items()}
    output: dict[str, Any] = {}
    for label, component in (("dense", optimizer.dense), ("sparse", optimizer.sparse)):
        if component is None:
            continue
        for parameter, state in component.state.items():
            name = reverse.get(id(parameter))
            if name in selected_names:
                output[name] = {"component": label, "state": _to_cpu(state)}
    return output


def _optimizer_state_index(
    optimizer: OptimizerBundle,
    named_parameters: Mapping[str, nn.Parameter],
) -> dict[str, tuple[str, Mapping[Any, Any]]]:
    reverse = {id(parameter): name for name, parameter in named_parameters.items()}
    output: dict[str, tuple[str, Mapping[Any, Any]]] = {}
    for label, component in (("dense", optimizer.dense), ("sparse", optimizer.sparse)):
        if component is None:
            continue
        for parameter, state in component.state.items():
            name = reverse.get(id(parameter))
            if name is None:
                raise RuntimeError(
                    "Optimizer state references a parameter outside the model inventory"
                )
            if name in output:
                raise RuntimeError(f"Optimizer state duplicates parameter {name}")
            output[name] = (label, state)
    return output


def _optimizer_components_by_name(
    optimizer: OptimizerBundle,
    named_parameters: Mapping[str, nn.Parameter],
) -> dict[str, tuple[str, torch.optim.Optimizer]]:
    reverse = {id(parameter): name for name, parameter in named_parameters.items()}
    output: dict[str, tuple[str, torch.optim.Optimizer]] = {}
    for label, component in (("dense", optimizer.dense), ("sparse", optimizer.sparse)):
        if component is None:
            continue
        for group in component.param_groups:
            for parameter in group["params"]:
                name = reverse.get(id(parameter))
                if name is None:
                    raise RuntimeError(
                        "Optimizer parameter group references a parameter outside the model"
                    )
                if name in output:
                    raise RuntimeError(
                        f"Optimizer parameter groups duplicate parameter {name}"
                    )
                output[name] = (label, component)
    return output


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _dtype_from_name(value: str) -> torch.dtype:
    prefix = "torch."
    if not value.startswith(prefix):
        raise RuntimeError(f"Checkpoint contains an invalid tensor dtype {value!r}")
    dtype = getattr(torch, value[len(prefix) :], None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"Checkpoint contains an unsupported tensor dtype {value!r}")
    return dtype


def _copy_tensor_chunk_to_cpu(
    tensor: torch.Tensor,
    *,
    start: int,
    end: int,
    maximum_bytes: int,
) -> torch.Tensor:
    source = tensor.detach()
    total_bytes = source.numel() * source.element_size()
    if not source.is_contiguous() and total_bytes > maximum_bytes:
        raise RuntimeError(
            "A non-contiguous checkpoint tensor exceeds the bounded staging limit; "
            "make the model/optimizer state contiguous before production training"
        )
    flattened = source.view(-1) if source.is_contiguous() else source.reshape(-1)
    # copy=True is important even for CPU-backed state: torch.save otherwise
    # serializes the slice's complete underlying storage.
    return flattened[start:end].to(device="cpu", copy=True)


def _restore_optimizer_state(
    optimizer: OptimizerBundle,
    named_parameters: Mapping[str, nn.Parameter],
    payload: Mapping[str, Any],
) -> None:
    for name, row in payload.items():
        if name not in named_parameters:
            raise RuntimeError(f"Checkpoint optimizer state references unknown parameter {name}")
        parameter = named_parameters[name]
        component_name = str(row["component"])
        component = optimizer.dense if component_name == "dense" else optimizer.sparse
        if component is None:
            raise RuntimeError(f"Checkpoint expects unavailable optimizer component {component_name}")
        component.state[parameter] = _to_device(row["state"], parameter.device)


def _owned_name_sets(
    model: nn.Module,
    topology: ParallelTopology,
) -> tuple[set[str], set[str], set[str]]:
    placements = _placements(model)
    replicated = {
        name
        for name, placement in placements.items()
        if placement in {"replicated", "sparse_table"}
    }
    expert = {
        name for name, placement in placements.items() if placement == "expert_sharded"
    }
    tables = {
        name for name, placement in placements.items() if placement == "row_sharded_table"
    }
    return replicated, expert, tables


class SignalCoordinator:
    """Turn preemption signals into a safe checkpoint request."""

    def __init__(self) -> None:
        self.requested = False
        self.reason: str | None = None
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.requested = True
        try:
            self.reason = signal.Signals(signum).name
        except ValueError:
            self.reason = str(signum)

    def install(self) -> None:
        for candidate in (signal.SIGUSR1, signal.SIGTERM):
            self._previous[int(candidate)] = signal.getsignal(candidate)
            signal.signal(candidate, self._handler)

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()

    def __enter__(self) -> "SignalCoordinator":
        self.install()
        return self

    def __exit__(self, *_args: object) -> None:
        self.restore()


class CheckpointManager:
    def __init__(
        self,
        output_root: str | Path,
        *,
        topology: ParallelTopology,
        keep_last: int = 3,
        max_staging_bytes: int = DEFAULT_MAX_STAGING_BYTES,
    ) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        if max_staging_bytes < 1:
            raise ValueError("max_staging_bytes must be positive")
        self.root = Path(output_root).expanduser().resolve() / "checkpoints"
        self.topology = topology
        self.keep_last = keep_last
        self.max_staging_bytes = int(max_staging_bytes)
        self._loaded_checkpoint_proof: _LoadedCheckpointProof | None = None
        if topology.rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
        barrier(topology)

    def _gather_records(self, local: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.topology.distributed:
            return local
        gathered: list[list[dict[str, Any]] | None] = [None] * self.topology.world_size
        dist.all_gather_object(gathered, local)
        flattened = [record for rows in gathered if rows for record in rows]
        paths = [record["path"] for record in flattened]
        if len(paths) != len(set(paths)):
            raise RuntimeError("Checkpoint writers produced duplicate artifact paths")
        return flattened

    def _gather_state_inventory(
        self, local: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not self.topology.distributed:
            flattened = local
        else:
            gathered: list[list[dict[str, Any]] | None] = [
                None
            ] * self.topology.world_size
            dist.all_gather_object(gathered, local)
            flattened = [row for rows in gathered if rows for row in rows]
        keys = [
            (
                row.get("owner"),
                row.get("kind"),
                row.get("name"),
                row.get("component"),
                row.get("state_key"),
            )
            for row in flattened
        ]
        if len(keys) != len(set(keys)):
            raise RuntimeError("Checkpoint state inventory contains duplicate targets")
        return flattened

    def _write_state_shards(
        self,
        *,
        checkpoint_root: Path,
        owner: str,
        model_state: Mapping[str, torch.Tensor],
        model_names: set[str],
        optimizer_state: Mapping[str, tuple[str, Mapping[Any, Any]]],
        optimizer_names: set[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        inventory: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        staged_tensor_bytes = 0
        shard_index = 0

        def flush() -> None:
            nonlocal items, staged_tensor_bytes, shard_index
            if not items:
                return
            payload = {
                "schema": CHECKPOINT_LAYOUT,
                "owner": owner,
                "shard_index": shard_index,
                "items": items,
            }
            target = (
                checkpoint_root
                / "state"
                / f"{owner}-shard-{shard_index:05d}.pt"
            )
            record = _atomic_torch_save(
                payload,
                target,
                artifact_root=checkpoint_root,
            )
            record.update(
                {
                    "kind": "state_shard",
                    "owner": owner,
                    "item_count": len(items),
                    "staged_tensor_bytes": staged_tensor_bytes,
                }
            )
            records.append(record)
            items = []
            staged_tensor_bytes = 0
            shard_index += 1

        def add_tensor(
            *,
            kind: str,
            name: str,
            tensor: torch.Tensor,
            component: str | None = None,
            state_key: str | None = None,
        ) -> None:
            nonlocal staged_tensor_bytes
            shape = [int(value) for value in tensor.shape]
            dtype = str(tensor.dtype)
            total_numel = int(tensor.numel())
            inventory.append(
                {
                    "owner": owner,
                    "kind": kind,
                    "name": name,
                    "component": component,
                    "state_key": state_key,
                    "shape": shape,
                    "dtype": dtype,
                    "numel": total_numel,
                }
            )
            elements_per_chunk = max(
                1, self.max_staging_bytes // max(1, tensor.element_size())
            )
            boundaries = (
                [(0, 0)]
                if total_numel == 0
                else [
                    (start, min(total_numel, start + elements_per_chunk))
                    for start in range(0, total_numel, elements_per_chunk)
                ]
            )
            for start, end in boundaries:
                chunk_bytes = (end - start) * tensor.element_size()
                if items and staged_tensor_bytes + chunk_bytes > self.max_staging_bytes:
                    flush()
                cpu_chunk = _copy_tensor_chunk_to_cpu(
                    tensor,
                    start=start,
                    end=end,
                    maximum_bytes=self.max_staging_bytes,
                )
                items.append(
                    {
                        "kind": kind,
                        "name": name,
                        "component": component,
                        "state_key": state_key,
                        "shape": shape,
                        "dtype": dtype,
                        "total_numel": total_numel,
                        "start": start,
                        "end": end,
                        "tensor": cpu_chunk,
                    }
                )
                staged_tensor_bytes += chunk_bytes
                if staged_tensor_bytes >= self.max_staging_bytes:
                    flush()

        for name in sorted(model_names):
            value = model_state.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Model state {name} is not a tensor")
            add_tensor(kind="model_tensor", name=name, tensor=value)

        for name in sorted(optimizer_names):
            row = optimizer_state.get(name)
            if row is None:
                continue
            component, state = row
            state_keys = [str(key) for key in state]
            if len(state_keys) != len(set(state_keys)) or any(
                not isinstance(key, str) for key in state
            ):
                raise RuntimeError(
                    f"Optimizer state for {name} requires unique string keys"
                )
            for state_key in sorted(state_keys):
                value = state[state_key]
                if isinstance(value, torch.Tensor):
                    add_tensor(
                        kind="optimizer_tensor",
                        name=name,
                        tensor=value,
                        component=component,
                        state_key=state_key,
                    )
                    continue
                if _contains_tensor(value):
                    raise RuntimeError(
                        f"Nested tensor optimizer state is unsupported for {name}.{state_key}"
                    )
                if items and staged_tensor_bytes + 4096 > self.max_staging_bytes:
                    flush()
                inventory.append(
                    {
                        "owner": owner,
                        "kind": "optimizer_value",
                        "name": name,
                        "component": component,
                        "state_key": state_key,
                    }
                )
                items.append(
                    {
                        "kind": "optimizer_value",
                        "name": name,
                        "component": component,
                        "state_key": state_key,
                        "value": value,
                    }
                )
        flush()
        return records, inventory

    @staticmethod
    def _validate_manifest_self_hash(manifest: Mapping[str, Any]) -> str:
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("Existing checkpoint has an unexpected schema")
        unsigned = {
            key: value
            for key, value in manifest.items()
            if key != "checkpoint_sha256"
        }
        observed = str(manifest.get("checkpoint_sha256", ""))
        if observed != canonical_json_sha256(unsigned):
            raise RuntimeError("Existing checkpoint manifest self-hash is invalid")
        return observed

    def _existing_checkpoint_action(
        self,
        *,
        final: Path,
        global_token_cursor: int,
        optimizer_step: int,
        phase: str,
        shard_order_seed: int,
        release_sha256: str,
        shard_manifest_sha256: str,
        family_manifest_sha256: str,
        runtime_manifest_sha256: str,
        autotune_profile_sha256: str,
        precision_audit: Mapping[str, Any],
        precision_role_plan_sha256: str | None,
        signal_reason: str | None,
        phase_boundary: bool,
        extra_state: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        manifest_path = final / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_sha256 = self._validate_manifest_self_hash(manifest)
        requested_immutable: dict[str, Any] = {
            "layout": CHECKPOINT_LAYOUT,
            "max_staging_bytes": self.max_staging_bytes,
            "family": self.topology.family,
            "world_size": self.topology.world_size,
            "expert_parallel_size": self.topology.expert_parallel_size,
            "expert_replica_count": self.topology.expert_replica_count,
            "global_token_cursor": int(global_token_cursor),
            "optimizer_step": int(optimizer_step),
            "phase": phase,
            "shard_order_seed": int(shard_order_seed),
            "release_sha256": release_sha256,
            "shard_manifest_sha256": shard_manifest_sha256,
            "family_manifest_sha256": family_manifest_sha256,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "autotune_profile_sha256": autotune_profile_sha256,
            "precision_role_plan_sha256": precision_role_plan_sha256,
            "precision_audit": dict(precision_audit),
            "signal_reason": signal_reason,
        }
        mismatches = [
            field
            for field, requested in requested_immutable.items()
            if manifest.get(field) != requested
        ]
        if mismatches:
            raise RuntimeError(
                "Existing same-cursor checkpoint is incompatible with the "
                f"requested state: {', '.join(sorted(mismatches))}"
            )
        match = _CHECKPOINT_PATTERN.fullmatch(final.name)
        if match is None or int(match.group(1)) != int(global_token_cursor):
            raise RuntimeError(
                "Existing checkpoint path does not encode its requested token cursor"
            )

        existing_boundary = bool(manifest.get("phase_boundary", False))
        requested_boundary = bool(phase_boundary)
        existing_extra_raw = manifest.get("extra_state")
        if not isinstance(existing_extra_raw, Mapping):
            raise RuntimeError("Existing checkpoint extra_state is invalid")
        existing_extra = dict(existing_extra_raw)
        requested_extra = dict(extra_state or {})

        if existing_boundary == requested_boundary:
            if existing_extra != requested_extra:
                raise RuntimeError(
                    "Existing same-cursor checkpoint extra_state is incompatible"
                )
            return "return", manifest
        if existing_boundary or not requested_boundary:
            raise RuntimeError(
                "Existing same-cursor checkpoint boundary state cannot be downgraded"
            )

        existing_complete = existing_extra.pop("stage_complete", None)
        requested_complete = requested_extra.pop("stage_complete", None)
        if existing_extra != requested_extra:
            raise RuntimeError(
                "Existing same-cursor checkpoint finalization metadata is incompatible"
            )
        if (
            existing_complete is not None
            or requested_complete is not None
        ) and not (
            existing_complete is False and requested_complete is True
        ):
            raise RuntimeError(
                "Existing same-cursor checkpoint stage_complete transition is not "
                "the exact false-to-true promotion"
            )

        proof = self._loaded_checkpoint_proof
        if (
            proof is None
            or Path(proof.checkpoint_path).resolve() != final.resolve()
            or proof.checkpoint_sha256 != checkpoint_sha256
            or proof.global_token_cursor != int(global_token_cursor)
            or proof.optimizer_step != int(optimizer_step)
            or proof.phase != phase
        ):
            raise RuntimeError(
                "Existing same-cursor checkpoint can be promoted only after this "
                "manager fully loads and validates that exact saved state"
            )
        return "promote", manifest

    def _promote_existing_checkpoint(
        self,
        *,
        final: Path,
        validated_manifest: Mapping[str, Any],
        extra_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        manifest_path = final / "MANIFEST.json"
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_sha256 = self._validate_manifest_self_hash(current)
        if current_sha256 != validated_manifest.get("checkpoint_sha256"):
            raise RuntimeError(
                "Existing checkpoint changed while final boundary promotion was pending"
            )
        promoted = dict(current)
        promoted["phase_boundary"] = True
        promoted["extra_state"] = dict(extra_state or {})
        promoted.pop("checkpoint_sha256", None)
        promoted["checkpoint_sha256"] = canonical_json_sha256(promoted)
        _atomic_json(promoted, manifest_path)
        directory_fd = os.open(final, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._write_latest(final.name)
        self._prune()
        root_fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        return promoted

    def _resolve_existing_checkpoint(
        self,
        *,
        final: Path,
        global_token_cursor: int,
        optimizer_step: int,
        phase: str,
        shard_order_seed: int,
        release_sha256: str,
        shard_manifest_sha256: str,
        family_manifest_sha256: str,
        runtime_manifest_sha256: str,
        autotune_profile_sha256: str,
        precision_audit: Mapping[str, Any],
        precision_role_plan_sha256: str | None,
        signal_reason: str | None,
        phase_boundary: bool,
        extra_state: Mapping[str, Any] | None,
    ) -> Path:
        try:
            action, manifest = self._existing_checkpoint_action(
                final=final,
                global_token_cursor=global_token_cursor,
                optimizer_step=optimizer_step,
                phase=phase,
                shard_order_seed=shard_order_seed,
                release_sha256=release_sha256,
                shard_manifest_sha256=shard_manifest_sha256,
                family_manifest_sha256=family_manifest_sha256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                autotune_profile_sha256=autotune_profile_sha256,
                precision_audit=precision_audit,
                precision_role_plan_sha256=precision_role_plan_sha256,
                signal_reason=signal_reason,
                phase_boundary=phase_boundary,
                extra_state=extra_state,
            )
            local_result = {
                "action": action,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "error": None,
            }
        except Exception as exc:
            manifest = {}
            local_result = {
                "action": None,
                "checkpoint_sha256": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if self.topology.distributed:
            gathered: list[dict[str, Any] | None] = [
                None
            ] * self.topology.world_size
            dist.all_gather_object(gathered, local_result)
            results = [row for row in gathered if row is not None]
        else:
            results = [local_result]
        errors = sorted(
            {
                str(row["error"])
                for row in results
                if row.get("error") is not None
            }
        )
        if errors:
            raise RuntimeError(
                "Existing checkpoint request validation failed: "
                + " | ".join(errors)
            )
        outcomes = {
            (str(row["action"]), str(row["checkpoint_sha256"]))
            for row in results
        }
        if len(outcomes) != 1:
            raise RuntimeError(
                "Existing checkpoint request produced inconsistent rank-local validation"
            )
        action, old_checkpoint_sha256 = next(iter(outcomes))
        if action == "return":
            return final
        if action != "promote":
            raise RuntimeError(f"Unknown existing checkpoint action {action!r}")

        promotion_error: str | None = None
        promoted: dict[str, Any] | None = None
        if self.topology.rank == 0:
            try:
                promoted = self._promote_existing_checkpoint(
                    final=final,
                    validated_manifest=manifest,
                    extra_state=extra_state,
                )
            except Exception as exc:
                promotion_error = f"{type(exc).__name__}: {exc}"
        if self.topology.distributed:
            payload: list[Any] = [
                {
                    "error": promotion_error,
                    "checkpoint_sha256": (
                        promoted.get("checkpoint_sha256")
                        if promoted is not None
                        else None
                    ),
                }
                if self.topology.rank == 0
                else None
            ]
            dist.broadcast_object_list(payload, src=0)
            promotion_result = dict(payload[0])
        else:
            promotion_result = {
                "error": promotion_error,
                "checkpoint_sha256": (
                    promoted.get("checkpoint_sha256")
                    if promoted is not None
                    else None
                ),
            }
        if promotion_result["error"] is not None:
            raise RuntimeError(
                "Existing checkpoint final boundary promotion failed: "
                f"{promotion_result['error']}"
            )
        new_checkpoint_sha256 = str(promotion_result["checkpoint_sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", new_checkpoint_sha256):
            raise RuntimeError("Promoted checkpoint did not seal a valid self-hash")
        proof = self._loaded_checkpoint_proof
        if (
            proof is not None
            and Path(proof.checkpoint_path).resolve() == final.resolve()
            and proof.checkpoint_sha256 == old_checkpoint_sha256
        ):
            self._loaded_checkpoint_proof = _LoadedCheckpointProof(
                checkpoint_path=proof.checkpoint_path,
                checkpoint_sha256=new_checkpoint_sha256,
                global_token_cursor=proof.global_token_cursor,
                optimizer_step=proof.optimizer_step,
                phase=proof.phase,
            )
        barrier(self.topology)
        visible = json.loads(
            (final / "MANIFEST.json").read_text(encoding="utf-8")
        )
        if self._validate_manifest_self_hash(visible) != new_checkpoint_sha256:
            raise RuntimeError(
                "Promoted checkpoint manifest is not consistently visible"
            )
        return final

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: OptimizerBundle,
        global_token_cursor: int,
        optimizer_step: int,
        phase: str,
        shard_order_seed: int,
        release_sha256: str,
        shard_manifest_sha256: str,
        family_manifest_sha256: str,
        runtime_manifest_sha256: str,
        autotune_profile_sha256: str,
        precision_audit: Mapping[str, Any],
        precision_role_plan_sha256: str | None = None,
        signal_reason: str | None = None,
        phase_boundary: bool = False,
        extra_state: Mapping[str, Any] | None = None,
    ) -> Path:
        if (
            self.topology.world_size > 1
            and precision_role_plan_sha256 is None
        ):
            raise RuntimeError(
                "Distributed production checkpoint requires a precision role plan hash"
            )
        if precision_role_plan_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", precision_role_plan_sha256
        ):
            raise RuntimeError("Checkpoint precision role plan hash is invalid")
        final = self.root / f"tokens-{global_token_cursor:013d}"
        incomplete = self.root / f".incomplete-tokens-{global_token_cursor:013d}"
        barrier(self.topology)
        existing = (final / "MANIFEST.json").is_file()
        if self.topology.distributed:
            observed: list[bool | None] = [None] * self.topology.world_size
            dist.all_gather_object(observed, existing)
            if any(bool(value) for value in observed):
                if not all(bool(value) for value in observed):
                    raise RuntimeError("Checkpoint visibility is inconsistent across ranks")
                return self._resolve_existing_checkpoint(
                    final=final,
                    global_token_cursor=global_token_cursor,
                    optimizer_step=optimizer_step,
                    phase=phase,
                    shard_order_seed=shard_order_seed,
                    release_sha256=release_sha256,
                    shard_manifest_sha256=shard_manifest_sha256,
                    family_manifest_sha256=family_manifest_sha256,
                    runtime_manifest_sha256=runtime_manifest_sha256,
                    autotune_profile_sha256=autotune_profile_sha256,
                    precision_audit=precision_audit,
                    precision_role_plan_sha256=precision_role_plan_sha256,
                    signal_reason=signal_reason,
                    phase_boundary=phase_boundary,
                    extra_state=extra_state,
                )
        elif existing:
            return self._resolve_existing_checkpoint(
                final=final,
                global_token_cursor=global_token_cursor,
                optimizer_step=optimizer_step,
                phase=phase,
                shard_order_seed=shard_order_seed,
                release_sha256=release_sha256,
                shard_manifest_sha256=shard_manifest_sha256,
                family_manifest_sha256=family_manifest_sha256,
                runtime_manifest_sha256=runtime_manifest_sha256,
                autotune_profile_sha256=autotune_profile_sha256,
                precision_audit=precision_audit,
                precision_role_plan_sha256=precision_role_plan_sha256,
                signal_reason=signal_reason,
                phase_boundary=phase_boundary,
                extra_state=extra_state,
            )
        self._loaded_checkpoint_proof = None
        if self.topology.rank == 0:
            if final.exists():
                raise RuntimeError(f"Checkpoint target exists without a manifest: {final}")
            if incomplete.exists():
                shutil.rmtree(incomplete)
            incomplete.mkdir(parents=True)
        barrier(self.topology)

        named_parameters = dict(model.named_parameters())
        replicated_names, expert_names, table_names = _owned_name_sets(model, self.topology)
        state = model.state_dict()
        optimizer_state = _optimizer_state_index(optimizer, named_parameters)
        local_records: list[dict[str, Any]] = []
        local_state_inventory: list[dict[str, Any]] = []
        if self.topology.rank == 0:
            replicated_model_names = {
                name
                for name in state
                if name in replicated_names or name not in named_parameters
            }
            records, inventory = self._write_state_shards(
                checkpoint_root=incomplete,
                owner="replicated",
                model_state=state,
                model_names=replicated_model_names,
                optimizer_state=optimizer_state,
                optimizer_names=replicated_names,
            )
            local_records.extend(records)
            local_state_inventory.extend(inventory)
            group_payload = {
                "schema": "metis.optimizer-groups/v1",
                "dense_param_groups": [
                    {key: value for key, value in group.items() if key != "params"}
                    for group in optimizer.dense.param_groups
                ],
                "sparse_param_groups": (
                    [
                        {key: value for key, value in group.items() if key != "params"}
                        for group in optimizer.sparse.param_groups
                    ]
                    if optimizer.sparse is not None
                    else None
                ),
            }
            group_record = _atomic_torch_save(
                group_payload,
                incomplete / "optimizer-groups.pt",
                artifact_root=incomplete,
            )
            group_record.update({"kind": "optimizer_groups", "owner": "replicated"})
            local_records.append(group_record)

        # Expert shards and row-sharded memory are duplicated only across Logos'
        # expert replicas. Replica zero is the canonical checkpoint owner.
        if self.topology.expert_replica_rank == 0:
            if expert_names:
                owner = f"experts-ep-{self.topology.expert_rank:04d}"
                records, inventory = self._write_state_shards(
                    checkpoint_root=incomplete,
                    owner=owner,
                    model_state=state,
                    model_names=expert_names,
                    optimizer_state=optimizer_state,
                    optimizer_names=expert_names,
                )
                local_records.extend(records)
                local_state_inventory.extend(inventory)
            if table_names:
                owner = f"tables-ep-{self.topology.expert_rank:04d}"
                records, inventory = self._write_state_shards(
                    checkpoint_root=incomplete,
                    owner=owner,
                    model_state=state,
                    model_names=table_names,
                    optimizer_state=optimizer_state,
                    optimizer_names=table_names,
                )
                local_records.extend(records)
                local_state_inventory.extend(inventory)

        rng_payload = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(self.topology.local_rank)
            if torch.cuda.is_available()
            else None,
        }
        rng_record = _atomic_torch_save(
            rng_payload,
            incomplete / f"rng-rank-{self.topology.rank:04d}.pt",
            artifact_root=incomplete,
        )
        rng_record.update({"kind": "rng", "owner": f"rank-{self.topology.rank:04d}"})
        local_records.append(rng_record)
        records = self._gather_records(local_records)
        state_inventory = self._gather_state_inventory(local_state_inventory)
        barrier(self.topology)
        if self.topology.rank == 0:
            manifest: dict[str, Any] = {
                "schema": CHECKPOINT_SCHEMA,
                "layout": CHECKPOINT_LAYOUT,
                "max_staging_bytes": self.max_staging_bytes,
                "family": self.topology.family,
                "world_size": self.topology.world_size,
                "expert_parallel_size": self.topology.expert_parallel_size,
                "expert_replica_count": self.topology.expert_replica_count,
                "global_token_cursor": int(global_token_cursor),
                "optimizer_step": int(optimizer_step),
                "phase": phase,
                "shard_order_seed": int(shard_order_seed),
                "release_sha256": release_sha256,
                "shard_manifest_sha256": shard_manifest_sha256,
                "family_manifest_sha256": family_manifest_sha256,
                "runtime_manifest_sha256": runtime_manifest_sha256,
                "autotune_profile_sha256": autotune_profile_sha256,
                "precision_role_plan_sha256": precision_role_plan_sha256,
                "precision_audit": dict(precision_audit),
                "signal_reason": signal_reason,
                "phase_boundary": bool(phase_boundary),
                "created_unix": time.time(),
                "artifacts": sorted(records, key=lambda row: row["path"]),
                "state_inventory": sorted(
                    state_inventory,
                    key=lambda row: (
                        str(row.get("owner")),
                        str(row.get("kind")),
                        str(row.get("name")),
                        str(row.get("component")),
                        str(row.get("state_key")),
                    ),
                ),
                "extra_state": dict(extra_state or {}),
            }
            manifest["checkpoint_sha256"] = canonical_json_sha256(manifest)
            _atomic_json(manifest, incomplete / "MANIFEST.json")
            directory_fd = os.open(incomplete, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(incomplete, final)
            self._write_latest(final.name)
            self._prune()
        barrier(self.topology)
        return final

    def _write_latest(self, name: str) -> None:
        _atomic_json(
            {"schema": "metis.checkpoint-latest/v1", "checkpoint": name},
            self.root / "LATEST.json",
        )

    def _prune(self) -> None:
        candidates: list[tuple[int, Path, bool]] = []
        for path in self.root.iterdir():
            match = _CHECKPOINT_PATTERN.match(path.name)
            manifest_path = path / "MANIFEST.json"
            if not match or not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates.append(
                (int(match.group(1)), path, bool(manifest.get("phase_boundary", False)))
            )
        candidates.sort(reverse=True)
        retained = {path for _cursor, path, _boundary in candidates[: self.keep_last]}
        retained.update(path for _cursor, path, boundary in candidates if boundary)
        for _cursor, path, _boundary in candidates:
            if path in retained:
                continue
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe checkpoint prune target: {path}") from exc
            shutil.rmtree(path)

    def latest(self) -> Path | None:
        pointer = self.root / "LATEST.json"
        candidates: set[Path] = set()
        if pointer.is_file():
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            path = (self.root / str(payload.get("checkpoint", ""))).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise RuntimeError("LATEST checkpoint pointer escapes checkpoint root") from exc
            if (path / "MANIFEST.json").is_file():
                candidates.add(path)
        candidates.update(
            path
            for path in self.root.iterdir()
            if _CHECKPOINT_PATTERN.match(path.name)
            and (path / "MANIFEST.json").is_file()
        )
        return (
            max(
                candidates,
                key=lambda path: int(
                    _CHECKPOINT_PATTERN.match(path.name).group(1)
                ),
            )
            if candidates
            else None
        )

    def _load_chunked_state(
        self,
        *,
        manifest: Mapping[str, Any],
        records: Mapping[str, Mapping[str, Any]],
        model: nn.Module,
        optimizer: OptimizerBundle,
        load_artifact: Callable[[str], Mapping[str, Any]],
    ) -> None:
        named_parameters = dict(model.named_parameters())
        replicated, experts, tables = _owned_name_sets(model, self.topology)
        model_state = model.state_dict()
        expected_model_names = {
            name
            for name in model_state
            if name in replicated
            or name not in named_parameters
            or name in experts
            or name in tables
        }
        owners = {"replicated"}
        if experts:
            owners.add(f"experts-ep-{self.topology.expert_rank:04d}")
        if tables:
            owners.add(f"tables-ep-{self.topology.expert_rank:04d}")

        inventory = manifest.get("state_inventory")
        if not isinstance(inventory, list) or not inventory:
            raise RuntimeError("Chunked checkpoint has no state inventory")

        def target_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            return (
                row.get("owner"),
                row.get("kind"),
                row.get("name"),
                row.get("component"),
                row.get("state_key"),
            )

        relevant_inventory: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for raw in inventory:
            if not isinstance(raw, Mapping):
                raise RuntimeError("Checkpoint state inventory row is invalid")
            key = target_key(raw)
            if key in relevant_inventory:
                raise RuntimeError("Checkpoint state inventory duplicates a target")
            if raw.get("owner") in owners:
                relevant_inventory[key] = raw
        inventory_model_names = {
            str(row.get("name"))
            for row in relevant_inventory.values()
            if row.get("kind") == "model_tensor"
        }
        if inventory_model_names != expected_model_names:
            raise RuntimeError(
                "Chunked checkpoint model inventory is incomplete; "
                f"missing={sorted(expected_model_names - inventory_model_names)[:8]} "
                f"unexpected={sorted(inventory_model_names - expected_model_names)[:8]}"
            )

        components = _optimizer_components_by_name(optimizer, named_parameters)
        optimizer_parameters = {
            str(row.get("name"))
            for row in relevant_inventory.values()
            if row.get("kind") in {"optimizer_tensor", "optimizer_value"}
        }
        for name in optimizer_parameters:
            component_row = components.get(name)
            if component_row is None:
                raise RuntimeError(
                    f"Checkpoint optimizer state references unmanaged parameter {name}"
                )
            _label, component = component_row
            component.state[named_parameters[name]].clear()

        intervals: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
        seen_values: set[tuple[Any, ...]] = set()
        relevant_records = [
            record
            for record in records.values()
            if record.get("kind") == "state_shard"
            and record.get("owner") in owners
        ]
        if not relevant_records:
            raise RuntimeError("Chunked checkpoint has no state shards for this rank")
        with torch.no_grad():
            for record in sorted(
                relevant_records, key=lambda row: str(row.get("path"))
            ):
                payload = load_artifact(str(record["path"]))
                if (
                    payload.get("schema") != CHECKPOINT_LAYOUT
                    or payload.get("owner") != record.get("owner")
                    or not isinstance(payload.get("items"), list)
                    or len(payload["items"]) != int(record.get("item_count", -1))
                ):
                    raise RuntimeError(
                        f"Checkpoint state shard payload is invalid: {record['path']}"
                    )
                for item in payload["items"]:
                    if not isinstance(item, Mapping):
                        raise RuntimeError("Checkpoint state-shard item is invalid")
                    key = target_key({**item, "owner": payload["owner"]})
                    expected = relevant_inventory.get(key)
                    if expected is None:
                        raise RuntimeError(
                            f"Checkpoint state shard contains undeclared target {key}"
                        )
                    kind = str(item.get("kind"))
                    name = str(item.get("name"))
                    if kind == "optimizer_value":
                        if key in seen_values or _contains_tensor(item.get("value")):
                            raise RuntimeError(
                                f"Checkpoint scalar optimizer state duplicates {key}"
                            )
                        component_label, component = components[name]
                        if component_label != item.get("component"):
                            raise RuntimeError(
                                f"Optimizer component mismatch for {name}"
                            )
                        component.state[named_parameters[name]][
                            str(item.get("state_key"))
                        ] = item.get("value")
                        seen_values.add(key)
                        continue
                    tensor = item.get("tensor")
                    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                        raise RuntimeError("Checkpoint tensor chunk is not CPU-backed")
                    shape = tuple(int(value) for value in item.get("shape", []))
                    dtype = _dtype_from_name(str(item.get("dtype")))
                    total_numel = int(item.get("total_numel", -1))
                    start = int(item.get("start", -1))
                    end = int(item.get("end", -1))
                    if (
                        list(shape) != expected.get("shape")
                        or str(dtype) != expected.get("dtype")
                        or total_numel != int(expected.get("numel", -1))
                        or start < 0
                        or end < start
                        or end > total_numel
                        or tensor.dtype != dtype
                        or tensor.numel() != end - start
                    ):
                        raise RuntimeError(
                            f"Checkpoint tensor chunk metadata is invalid for {key}"
                        )
                    if kind == "model_tensor":
                        target = model_state.get(name)
                        if (
                            target is None
                            or tuple(target.shape) != shape
                            or target.dtype != dtype
                        ):
                            raise RuntimeError(
                                f"Checkpoint model tensor shape/dtype mismatch for {name}"
                            )
                    elif kind == "optimizer_tensor":
                        component_label, component = components[name]
                        if component_label != item.get("component"):
                            raise RuntimeError(
                                f"Optimizer component mismatch for {name}"
                            )
                        state_key = str(item.get("state_key"))
                        state = component.state[named_parameters[name]]
                        target = state.get(state_key)
                        if target is None:
                            target = torch.empty(
                                shape,
                                dtype=dtype,
                                device=named_parameters[name].device,
                            )
                            state[state_key] = target
                        if (
                            not isinstance(target, torch.Tensor)
                            or tuple(target.shape) != shape
                            or target.dtype != dtype
                        ):
                            raise RuntimeError(
                                f"Checkpoint optimizer tensor mismatch for {name}.{state_key}"
                            )
                    else:
                        raise RuntimeError(
                            f"Checkpoint contains unknown state item kind {kind!r}"
                        )
                    if not target.is_contiguous():
                        if start != 0 or end != total_numel:
                            raise RuntimeError(
                                f"Non-contiguous checkpoint target was split for {key}"
                            )
                        target.copy_(tensor.reshape(shape))
                    else:
                        target.view(-1)[start:end].copy_(tensor.view(-1))
                    intervals.setdefault(key, []).append((start, end))

        tensor_keys = {
            key
            for key, row in relevant_inventory.items()
            if row.get("kind") in {"model_tensor", "optimizer_tensor"}
        }
        if set(intervals) != tensor_keys:
            raise RuntimeError("Checkpoint tensor chunks do not cover the state inventory")
        value_keys = {
            key
            for key, row in relevant_inventory.items()
            if row.get("kind") == "optimizer_value"
        }
        if seen_values != value_keys:
            raise RuntimeError("Checkpoint optimizer values do not cover the state inventory")
        for key, spans in intervals.items():
            total = int(relevant_inventory[key]["numel"])
            cursor = 0
            for start, end in sorted(spans):
                if start != cursor or end < start:
                    raise RuntimeError(
                        f"Checkpoint tensor chunks overlap or have a gap for {key}"
                    )
                cursor = end
            if cursor != total:
                raise RuntimeError(f"Checkpoint tensor chunks are incomplete for {key}")

        # ``state_dict()`` returns live views for ordinary parameters/buffers,
        # so the bounded chunk copies above restore those tensors in place.
        # Module extra state is different: e.g. Transformer Engine serializes
        # DelayedScaling amax/scale history in ``._extra_state`` and only
        # applies it from ``set_extra_state`` during ``load_state_dict``.
        # Re-load the now-complete mapping to dispatch every extra-state value
        # through its owning module without staging a second model copy.
        incompatible = model.load_state_dict(model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint model restore is incomplete after extra-state dispatch; "
                f"missing={incompatible.missing_keys[:8]} "
                f"unexpected={incompatible.unexpected_keys[:8]}"
            )

        group_records = [
            row
            for row in records.values()
            if row.get("kind") == "optimizer_groups"
        ]
        if len(group_records) != 1:
            raise RuntimeError("Chunked checkpoint requires one optimizer-group artifact")
        groups = load_artifact(str(group_records[0]["path"]))
        if groups.get("schema") != "metis.optimizer-groups/v1":
            raise RuntimeError("Checkpoint optimizer-group payload is invalid")

        def restore_groups(
            current: list[dict[str, Any]],
            saved: Any,
            *,
            label: str,
        ) -> None:
            if not isinstance(saved, list) or len(current) != len(saved):
                raise RuntimeError(f"Checkpoint {label} parameter groups are incompatible")
            for current_group, saved_group in zip(current, saved, strict=True):
                if not isinstance(saved_group, Mapping) or "params" in saved_group:
                    raise RuntimeError(
                        f"Checkpoint {label} parameter-group metadata is invalid"
                    )
                current_group.update(saved_group)

        restore_groups(
            optimizer.dense.param_groups,
            groups.get("dense_param_groups"),
            label="dense",
        )
        if optimizer.sparse is None:
            if groups.get("sparse_param_groups") is not None:
                raise RuntimeError("Checkpoint expects a sparse optimizer")
        else:
            restore_groups(
                optimizer.sparse.param_groups,
                groups.get("sparse_param_groups"),
                label="sparse",
            )

    def load(
        self,
        checkpoint: str | Path,
        *,
        model: nn.Module,
        optimizer: OptimizerBundle,
        expected_release_sha256: str,
        expected_shard_manifest_sha256: str,
        expected_family_manifest_sha256: str,
        expected_runtime_manifest_sha256: str,
        expected_autotune_profile_sha256: str,
        expected_precision_role_plan_sha256: str | None = None,
    ) -> ResumeState:
        # A failed or partial restore must never leave an earlier checkpoint
        # eligible for metadata-only finalization.
        self._loaded_checkpoint_proof = None
        path = Path(checkpoint).expanduser().resolve()
        manifest_path = path / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("Unexpected checkpoint schema")
        unsigned = {key: value for key, value in manifest.items() if key != "checkpoint_sha256"}
        if manifest.get("checkpoint_sha256") != canonical_json_sha256(unsigned):
            raise RuntimeError("Checkpoint manifest self-hash is invalid")
        expected = {
            "release_sha256": expected_release_sha256,
            "shard_manifest_sha256": expected_shard_manifest_sha256,
            "family_manifest_sha256": expected_family_manifest_sha256,
            "runtime_manifest_sha256": expected_runtime_manifest_sha256,
            "autotune_profile_sha256": expected_autotune_profile_sha256,
        }
        if expected_precision_role_plan_sha256 is not None:
            if not re.fullmatch(
                r"[0-9a-f]{64}", expected_precision_role_plan_sha256
            ):
                raise RuntimeError("Expected precision role plan hash is invalid")
            expected["precision_role_plan_sha256"] = (
                expected_precision_role_plan_sha256
            )
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise RuntimeError(f"Checkpoint lineage mismatch for {field}")
        extra_state = manifest.get("extra_state", {})
        data_position = extra_state.get("data_position")
        if "posttraining_stage" not in extra_state and (
            not isinstance(data_position, Mapping)
            or int(data_position.get("global_token_cursor", -1))
            != int(manifest.get("global_token_cursor", -2))
            or data_position.get("phase") != manifest.get("phase")
        ):
            raise RuntimeError("Checkpoint is missing its exact phase/shard/offset position")
        if (
            int(manifest.get("world_size", 0)) != self.topology.world_size
            or int(manifest.get("expert_parallel_size", 0))
            != self.topology.expert_parallel_size
            or int(manifest.get("expert_replica_count", 0))
            != self.topology.expert_replica_count
        ):
            raise RuntimeError("Checkpoint topology does not match this launch")

        artifact_rows = manifest.get("artifacts")
        if not isinstance(artifact_rows, list) or not artifact_rows:
            raise RuntimeError("Checkpoint has no artifact inventory")
        records: dict[str, Mapping[str, Any]] = {}
        for row in artifact_rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("Checkpoint artifact record is invalid")
            name = str(row.get("path", ""))
            if not name or Path(name).is_absolute() or name in records:
                raise RuntimeError("Checkpoint artifact path is unsafe or duplicated")
            records[name] = row

        def load_artifact(name: str) -> Mapping[str, Any]:
            artifact = (path / name).resolve()
            try:
                artifact.relative_to(path)
            except ValueError as exc:
                raise RuntimeError(
                    f"Checkpoint artifact escapes its root: {name}"
                ) from exc
            record = records.get(name)
            if record is None or artifact.is_symlink() or not artifact.is_file():
                raise RuntimeError(f"Checkpoint artifact is missing: {name}")
            if artifact.stat().st_size != int(record["bytes"]) or sha256_file(artifact) != record["sha256"]:
                raise RuntimeError(f"Checkpoint artifact failed integrity validation: {name}")
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Checkpoint artifact payload is invalid: {name}")
            return payload

        if manifest.get("layout") == CHECKPOINT_LAYOUT:
            self._load_chunked_state(
                manifest=manifest,
                records=records,
                model=model,
                optimizer=optimizer,
                load_artifact=load_artifact,
            )
        elif manifest.get("layout") is None:
            # Backward-compatible restore for pre-streaming v1 checkpoints.
            named_parameters = dict(model.named_parameters())
            replicated, experts, tables = _owned_name_sets(model, self.topology)
            replicated_payload = load_artifact("replicated.pt")
            partial_model: dict[str, torch.Tensor] = dict(
                replicated_payload["model"]
            )
            if experts:
                expert_payload = load_artifact(
                    f"experts-ep-{self.topology.expert_rank:04d}.pt"
                )
                partial_model.update(expert_payload["model"])
            else:
                expert_payload = {"optimizer": {}}
            if tables:
                table_payload = load_artifact(
                    f"tables-ep-{self.topology.expert_rank:04d}.pt"
                )
                partial_model.update(table_payload["model"])
            else:
                table_payload = {"optimizer": {}}
            missing, unexpected = model.load_state_dict(partial_model, strict=False)
            unresolved = [
                name
                for name in missing
                if name in replicated
                or name in experts
                or name in tables
                or name not in named_parameters
            ]
            if unresolved or unexpected:
                raise RuntimeError(
                    f"Checkpoint model restore is incomplete; missing={unresolved[:8]} "
                    f"unexpected={unexpected[:8]}"
                )
            _restore_optimizer_state(
                optimizer, named_parameters, replicated_payload.get("optimizer", {})
            )
            _restore_optimizer_state(
                optimizer, named_parameters, expert_payload.get("optimizer", {})
            )
            _restore_optimizer_state(
                optimizer, named_parameters, table_payload.get("optimizer", {})
            )
        else:
            raise RuntimeError(
                f"Unsupported checkpoint layout: {manifest.get('layout')!r}"
            )
        rng = load_artifact(f"rng-rank-{self.topology.rank:04d}.pt")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state(rng["torch_cuda"], self.topology.local_rank)
        barrier(self.topology)
        self._loaded_checkpoint_proof = _LoadedCheckpointProof(
            checkpoint_path=str(path),
            checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            global_token_cursor=int(manifest["global_token_cursor"]),
            optimizer_step=int(manifest["optimizer_step"]),
            phase=str(manifest["phase"]),
        )
        return ResumeState(
            global_token_cursor=int(manifest["global_token_cursor"]),
            optimizer_step=int(manifest["optimizer_step"]),
            phase=str(manifest["phase"]),
            shard_order_seed=int(manifest["shard_order_seed"]),
            checkpoint_path=str(path),
            checkpoint_sha256=str(manifest["checkpoint_sha256"]),
            autotune_profile_sha256=str(manifest["autotune_profile_sha256"]),
            precision_role_plan_sha256=str(
                manifest.get("precision_role_plan_sha256") or ""
            ),
            release_sha256=str(manifest["release_sha256"]),
            signal_reason=manifest.get("signal_reason"),
        )
