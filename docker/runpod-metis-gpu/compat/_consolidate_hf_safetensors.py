from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch.distributed as dist


def _copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def consolidate_safetensors_files_on_every_rank(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[object, int] | None = None,
    num_threads: int = 1,
) -> None:
    del fqn_to_index_mapping, num_threads

    src = Path(input_dir)
    dst = Path(output_dir)
    if not src.exists():
        raise FileNotFoundError(f"Input safetensor shard directory not found: {src}")

    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        _copy_tree_contents(src, dst)

    if is_dist:
        dist.barrier()
