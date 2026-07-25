from __future__ import annotations

import os
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @property
    def is_root(self) -> bool:
        return self.rank == 0


def normalize_slurm_environment() -> None:
    mappings = {
        "RANK": "SLURM_PROCID",
        "LOCAL_RANK": "SLURM_LOCALID",
        "WORLD_SIZE": "SLURM_NTASKS",
    }
    for target, source in mappings.items():
        if target not in os.environ and source in os.environ:
            os.environ[target] = os.environ[source]
    os.environ.setdefault("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ:
        # Single-rank local probes do not need rendezvous.  Multi-rank Slurm
        # scripts must set MASTER_ADDR from the allocation's first hostname.
        os.environ["MASTER_ADDR"] = "127.0.0.1"


def initialize_distributed(*, require_gpu: bool = True) -> DistributedContext:
    normalize_slurm_environment()
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("Portage distributed probe requires a visible ROCm GPU")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible devices={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(local_rank)
    initialized = False
    if world_size > 1:
        if os.environ.get("MASTER_ADDR") in {"", "127.0.0.1"}:
            raise RuntimeError("Multi-rank Slurm probe is missing a non-loopback MASTER_ADDR")
        dist.init_process_group(
            backend="nccl" if require_gpu else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        initialized = True
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        initialized=initialized,
    )


def destroy_distributed(context: DistributedContext) -> None:
    if not context.initialized:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def node_identity() -> str:
    return socket.getfqdn()
