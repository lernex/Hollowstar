from __future__ import annotations

import datetime as _datetime
import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


@dataclass(frozen=True)
class ParallelTopology:
    family: str
    world_size: int
    rank: int
    local_rank: int
    expert_parallel_size: int
    expert_replica_count: int
    expert_group: dist.ProcessGroup | None
    expert_group_ranks: tuple[int, ...]
    expert_data_group: dist.ProcessGroup | None
    expert_data_group_ranks: tuple[int, ...]
    dense_data_group: dist.ProcessGroup | None

    @property
    def expert_rank(self) -> int:
        return self.rank % self.expert_parallel_size

    @property
    def expert_replica_rank(self) -> int:
        return self.rank // self.expert_parallel_size

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def as_model_groups(self) -> dict[str, object]:
        return {
            "world": self.dense_data_group,
            "dense_data": self.dense_data_group,
            "expert": self.expert_group,
            "expert_data": self.expert_data_group,
            "expert_parallel_size": self.expert_parallel_size,
            "expert_replica_count": self.expert_replica_count,
            "expert_rank": self.expert_rank,
            "expert_replica_rank": self.expert_replica_rank,
            "global_rank": self.rank,
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool


def initialize_runtime(
    *,
    device: str | None = None,
    timeout_minutes: int = 30,
) -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Portage training requires ROCm-visible torch.cuda devices")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            # PyTorch deliberately uses the backend name NCCL on ROCm; the
            # installed implementation is RCCL.
            dist.init_process_group(
                backend="nccl",
                timeout=_datetime.timedelta(minutes=timeout_minutes),
            )
        return Runtime(
            device=torch.device("cuda", local_rank),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            distributed=True,
        )
    if device:
        selected = torch.device(device)
    elif torch.cuda.is_available():
        selected = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        selected = torch.device("mps")
    else:
        selected = torch.device("cpu")
    return Runtime(
        device=selected,
        rank=0,
        local_rank=0,
        world_size=1,
        distributed=False,
    )


def destroy_runtime() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _production_shape(family: str) -> tuple[int, int, int]:
    shapes = {
        "praxis": (128, 128, 1),
        "logos": (384, 192, 2),
    }
    try:
        return shapes[family.lower()]
    except KeyError as exc:
        raise ValueError("family must be praxis or logos") from exc


def build_parallel_topology(
    runtime: Runtime,
    *,
    family: str,
    routed_experts: int,
    production: bool,
) -> ParallelTopology:
    family = family.lower()
    expected_world, expected_ep, expected_replicas = _production_shape(family)
    if production:
        if runtime.world_size != expected_world:
            raise RuntimeError(
                f"{family} production launch requires {expected_world} ranks; "
                f"observed {runtime.world_size}"
            )
        ep_size = expected_ep
        replicas = expected_replicas
    else:
        if runtime.world_size > routed_experts or routed_experts % runtime.world_size != 0:
            raise RuntimeError(
                "Probe world size must divide the routed expert count so every "
                "rank owns the same non-zero number of experts"
            )
        ep_size = runtime.world_size
        replicas = 1

    if runtime.world_size != ep_size * replicas:
        raise RuntimeError("World size does not equal EP size times expert replicas")
    if routed_experts % ep_size != 0:
        raise RuntimeError("Routed expert count must be divisible by EP size")
    if not runtime.distributed:
        return ParallelTopology(
            family=family,
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

    expert_group = None
    expert_group_ranks: tuple[int, ...] = ()
    for replica in range(replicas):
        ranks = tuple(range(replica * ep_size, (replica + 1) * ep_size))
        group = dist.new_group(ranks=list(ranks), backend="nccl")
        if runtime.rank in ranks:
            expert_group = group
            expert_group_ranks = ranks

    expert_data_group = None
    expert_data_group_ranks: tuple[int, ...] = ()
    for expert_rank in range(ep_size):
        ranks = tuple(expert_rank + replica * ep_size for replica in range(replicas))
        # A one-rank group is represented as None to avoid needless RCCL work.
        group = dist.new_group(ranks=list(ranks), backend="nccl") if len(ranks) > 1 else None
        if runtime.rank in ranks:
            expert_data_group = group
            expert_data_group_ranks = ranks
    if expert_group is None or not expert_group_ranks or not expert_data_group_ranks:
        raise RuntimeError("Failed to place the current rank in Portage process groups")
    return ParallelTopology(
        family=family,
        world_size=runtime.world_size,
        rank=runtime.rank,
        local_rank=runtime.local_rank,
        expert_parallel_size=ep_size,
        expert_replica_count=replicas,
        expert_group=expert_group,
        expert_group_ranks=expert_group_ranks,
        expert_data_group=expert_data_group,
        expert_data_group_ranks=expert_data_group_ranks,
        dense_data_group=dist.group.WORLD,
    )


def _placements(model: nn.Module) -> dict[str, str]:
    provider = getattr(model, "parameter_placements", None)
    if callable(provider):
        raw = provider()
        if not isinstance(raw, Mapping):
            raise RuntimeError("model.parameter_placements() must return a mapping")
        placements = {str(name): str(value) for name, value in raw.items()}
    else:
        placements = {name: "replicated" for name, _ in model.named_parameters()}
    valid = {"replicated", "expert_sharded", "sparse_table", "row_sharded_table"}
    unknown = sorted(set(placements.values()) - valid)
    if unknown:
        raise RuntimeError(f"Unknown parameter placement tags: {unknown}")
    names = {name for name, _ in model.named_parameters()}
    missing = sorted(names - set(placements))
    if missing:
        raise RuntimeError(f"Model omitted parameter placement tags: {missing[:8]}")
    return placements


def broadcast_initial_parameters(model: nn.Module, topology: ParallelTopology) -> None:
    if not topology.distributed:
        return
    placements = _placements(model)
    for name, parameter in model.named_parameters():
        placement = placements[name]
        if placement in {"replicated", "sparse_table"}:
            dist.broadcast(parameter.data, src=0, group=topology.dense_data_group)
        elif placement == "expert_sharded" and topology.expert_replica_count > 1:
            source = topology.expert_data_group_ranks[0]
            dist.broadcast(parameter.data, src=source, group=topology.expert_data_group)
        elif placement == "row_sharded_table" and topology.expert_replica_count > 1:
            # The same row partition is duplicated only across Logos expert
            # replicas; partitions still intentionally differ by EP rank.
            source = topology.expert_data_group_ranks[0]
            dist.broadcast(parameter.data, src=source, group=topology.expert_data_group)
        # Row-sharded tables intentionally differ by owner EP rank.
    for name, buffer in model.named_buffers():
        if buffer.numel() == 0:
            continue
        # Router/load-balancing and controller buffers must begin identical.
        if "expert" not in name or "local" not in name:
            dist.broadcast(buffer, src=0, group=topology.dense_data_group)


def _parameter_buckets(
    rows: Iterable[tuple[str, nn.Parameter]],
    *,
    maximum_bucket_bytes: int,
) -> list[list[nn.Parameter]]:
    buckets: list[list[nn.Parameter]] = []
    current: list[nn.Parameter] = []
    current_bytes = 0
    current_key: tuple[torch.dtype, torch.device] | None = None
    for _name, parameter in rows:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise RuntimeError("Sparse gradient reached the dense gradient synchronizer")
        key = (parameter.grad.dtype, parameter.grad.device)
        size = parameter.grad.numel() * parameter.grad.element_size()
        if current and (key != current_key or current_bytes + size > maximum_bucket_bytes):
            buckets.append(current)
            current = []
            current_bytes = 0
        current_key = key
        current.append(parameter)
        current_bytes += size
    if current:
        buckets.append(current)
    return buckets


def _all_reduce_buckets(
    buckets: Iterable[list[nn.Parameter]],
    *,
    group: dist.ProcessGroup,
    divisor: int,
) -> None:
    pending: list[tuple[object, torch.Tensor, list[nn.Parameter]]] = []
    for parameters in buckets:
        flattened = _flatten_dense_tensors([parameter.grad for parameter in parameters])
        work = dist.all_reduce(flattened, group=group, async_op=True)
        pending.append((work, flattened, parameters))
    for work, flattened, parameters in pending:
        work.wait()
        flattened.div_(float(divisor))
        for parameter, reduced in zip(
            parameters,
            _unflatten_dense_tensors(flattened, [parameter.grad for parameter in parameters]),
            strict=True,
        ):
            parameter.grad.copy_(reduced)


def synchronize_gradients(
    model: nn.Module,
    topology: ParallelTopology,
    *,
    maximum_bucket_bytes: int = 128 * 1024 * 1024,
) -> None:
    if not topology.distributed:
        return
    placements = _placements(model)
    named = list(model.named_parameters())
    sparse = [
        (name, parameter)
        for name, parameter in named
        if placements[name] == "sparse_table" and parameter.grad is not None and parameter.grad.is_sparse
    ]
    if sparse:
        # The model owns coalesced sparse-row synchronization because it knows
        # table hashing/partition semantics. A normal dense all-reduce would
        # either fail or silently materialize multi-billion-row gradients.
        checker = getattr(model, "sparse_gradient_sync_enabled", None)
        if not callable(checker) or not bool(checker()):
            raise RuntimeError(
                "Replicated N-gram sparse gradients are not synchronized; call "
                "model.enable_managed_sparse_gradient_sync(group) before training"
            )
    replicated_rows = [
        (name, parameter)
        for name, parameter in named
        if placements[name] in {"replicated", "sparse_table"}
        and not (parameter.grad is not None and parameter.grad.is_sparse)
    ]
    expert_rows = [
        (name, parameter)
        for name, parameter in named
        if placements[name] == "expert_sharded"
    ]
    replicated_buckets = _parameter_buckets(
        replicated_rows, maximum_bucket_bytes=maximum_bucket_bytes
    )
    _all_reduce_buckets(
        replicated_buckets,
        group=topology.dense_data_group,
        divisor=topology.world_size,
    )
    if topology.expert_replica_count > 1:
        expert_buckets = _parameter_buckets(
            expert_rows, maximum_bucket_bytes=maximum_bucket_bytes
        )
        _all_reduce_buckets(
            expert_buckets,
            group=topology.expert_data_group,
            divisor=topology.expert_replica_count,
        )


def normalize_summed_gradients(
    model: nn.Module,
    topology: ParallelTopology,
    *,
    global_supervised_tokens: int,
) -> None:
    """Normalize token-summed gradients after placement-aware synchronization.

    Dense replicated parameters receive one local token sum per rank and are
    averaged over the full world. Expert and row-sharded table owners already
    receive dispatched work from their entire EP group, then average only over
    expert replicas. The two placement classes therefore need different scale
    factors to become an exact global supervised-token mean.
    """

    if global_supervised_tokens <= 0:
        raise RuntimeError("Cannot normalize gradients with zero supervised tokens")
    placements = _placements(model)
    dense_scale = topology.world_size / float(global_supervised_tokens)
    sharded_scale = topology.expert_replica_count / float(global_supervised_tokens)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        scale = (
            sharded_scale
            if placements[name] in {"expert_sharded", "row_sharded_table"}
            else dense_scale
        )
        if parameter.grad.is_sparse:
            parameter.grad._values().mul_(scale)
        else:
            parameter.grad.mul_(scale)


def all_reduce_sum(value: torch.Tensor, topology: ParallelTopology) -> torch.Tensor:
    if topology.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=topology.dense_data_group)
    return value


def global_any(flag: bool, topology: ParallelTopology, device: torch.device) -> bool:
    if not topology.distributed:
        return flag
    value = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=topology.dense_data_group)
    return bool(value.item())


def barrier(topology: ParallelTopology) -> None:
    if topology.distributed:
        dist.barrier(group=topology.dense_data_group)
