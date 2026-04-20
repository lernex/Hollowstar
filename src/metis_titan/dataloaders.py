from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader


class MemmapTokenDataset(IterableDataset, Stateful):
    """Stream already-tokenized memmap shards without re-tokenizing on GPU."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        split: str,
        seq_len: int,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool,
    ) -> None:
        root = Path(dataset_path)
        meta = json.loads((root / "meta.json").read_text())
        dtype = np.dtype(meta["dtype"])
        bin_path = root / f"{split}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing memmap split: {bin_path}")

        self.dataset_path = root
        self.split = split
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self.chunk_len = seq_len + 1
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self._stride_tokens = self.chunk_len * max(dp_world_size, 1)
        self._initial_offset = self.chunk_len * dp_rank
        self._cursor = self._initial_offset
        self._epoch = 0

        if len(self.data) < self.chunk_len:
            raise ValueError(
                f"{bin_path} only has {len(self.data)} tokens, which is too small for seq_len={seq_len}."
            )

    def __iter__(self):
        while True:
            while self._cursor + self.chunk_len <= len(self.data):
                window = np.asarray(
                    self.data[self._cursor : self._cursor + self.chunk_len],
                    dtype=np.int64,
                )
                tokens = torch.from_numpy(window.copy())
                positions = torch.arange(self.seq_len, dtype=torch.long)
                self._cursor += self._stride_tokens
                yield {"input": tokens[:-1], "positions": positions}, tokens[1:]

            if not self.infinite:
                break

            self._epoch += 1
            self._cursor = self._initial_offset

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": self._cursor,
            "epoch": self._epoch,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._cursor = int(state_dict.get("cursor", self._initial_offset))
        self._epoch = int(state_dict.get("epoch", 0))


class MemmapTokenDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = "memmap_tokens"
        dataset_path: str | None = None
        split: str = "train"
        infinite: bool = True

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        **kwargs,
    ):
        if not config.dataset_path:
            raise ValueError("MemmapTokenDataLoader requires dataset_path to be set.")

        dataset = MemmapTokenDataset(
            dataset_path=config.dataset_path,
            split=config.split,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "batch_size": local_batch_size,
        }
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )

