from __future__ import annotations

import bisect
import json
import queue
import random
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from tokenizers import Tokenizer

from metis_data.ngram_canonical import validate_canonical_id_sidecar


PHASE_ORDER = ("phase_a", "phase_b", "phase_c")
PHASE_DIRECTORIES = {
    "phase_a": "phase-a",
    "phase_b": "phase-b",
    "phase_c": "phase-c",
}
PHASE_STARTS = {
    "phase_a": 0,
    "phase_b": 700_000_000_000,
    "phase_c": 950_000_000_000,
}
PHASE_TOKENS = {
    "phase_a": 700_000_000_000,
    "phase_b": 250_000_000_000,
    "phase_c": 50_000_000_000,
}
TOTAL_TOKENS = 1_000_000_000_000


@dataclass(frozen=True)
class ReleaseShard:
    phase: str
    phase_index: int
    binary: Path
    index: Path
    tokens: int


@dataclass(frozen=True)
class ReleaseInventory:
    root: Path
    tokenizer: Path
    release_sha256: str
    shard_manifest_sha256: str
    shards: tuple[ReleaseShard, ...]
    ngram_canonical_map: Path | None = None
    ngram_canonical_ids: Path | None = None
    ngram_canonical_map_self_sha256: str = ""
    ngram_canonical_ids_sha256: str = ""

    @classmethod
    def from_release_root(cls, release_root: str | Path) -> "ReleaseInventory":
        root = Path(release_root).expanduser().resolve()
        descriptor_path = root / "RELEASE.json"
        if not descriptor_path.is_file():
            raise RuntimeError(f"Release descriptor is missing: {descriptor_path}")
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        if (
            descriptor.get("schema") != "metis.data-release/v2"
            or descriptor.get("token_dtype") != "uint16"
            or descriptor.get("token_endianness") != "little"
            or int(descriptor.get("target_tokens", 0)) != TOTAL_TOKENS
            or descriptor.get("phase_tokens") != PHASE_TOKENS
            or descriptor.get("verification", {}).get("ok") is not True
        ):
            raise RuntimeError("Release descriptor is not the verified Metis-1.6 1T contract")
        artifacts = descriptor.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise RuntimeError("Release descriptor has no artifact map")

        def artifact(field: str) -> Path:
            raw = artifacts.get(field)
            if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
                raise RuntimeError(f"Unsafe release artifact path: {field}")
            path = (root / raw).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Release artifact escapes root: {field}") from exc
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Release artifact is missing or a symlink: {path}")
            return path

        tokenizer = artifact("tokenizer")
        ngram_canonical_map = artifact("ngram_canonical_map")
        ngram_canonical_ids = artifact("ngram_canonical_ids")
        tokenizer_contract = descriptor.get("tokenizer_contract")
        if not isinstance(tokenizer_contract, Mapping):
            raise RuntimeError("Release descriptor has no tokenizer contract")
        canonical_descriptor, _canonical_ids = validate_canonical_id_sidecar(
            manifest_path=ngram_canonical_map,
            binary_path=ngram_canonical_ids,
            tokenizer_path=tokenizer,
            expected_vocabulary_size=65_536,
            expected_manifest_sha256=descriptor.get(
                "ngram_canonical_map_self_sha256"
            ),
            expected_binary_sha256=descriptor.get("ngram_canonical_ids_sha256"),
            recompute_from_tokenizer=False,
        )
        if (
            descriptor.get("ngram_canonical_map_manifest_sha256")
            != tokenizer_contract.get("ngram_canonical_map_manifest_sha256")
            or descriptor.get("ngram_canonical_map_self_sha256")
            != tokenizer_contract.get("ngram_canonical_map_self_sha256")
            or descriptor.get("ngram_canonical_ids_sha256")
            != tokenizer_contract.get("ngram_canonical_ids_sha256")
            or tokenizer_contract.get("ngram_canonicalization_algorithm")
            != canonical_descriptor.get("algorithm")
            or tokenizer_contract.get("ngram_canonical_dtype") != "uint16"
            or tokenizer_contract.get("ngram_canonical_endianness") != "little"
            or int(tokenizer_contract.get("ngram_canonical_entry_count", -1))
            != 65_536
        ):
            raise RuntimeError(
                "Released canonical-ID sidecar disagrees with the tokenizer contract"
            )
        shard_manifest = artifact("shard_manifest")
        rows = [
            json.loads(line)
            for line in shard_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        shards: list[ReleaseShard] = []
        phase_totals = {phase: 0 for phase in PHASE_ORDER}
        phase_indices = {phase: set() for phase in PHASE_ORDER}
        for row in rows:
            phase = str(row.get("phase", ""))
            if phase not in PHASE_ORDER:
                raise RuntimeError(f"Unknown phase in shard inventory: {phase!r}")
            phase_index = int(row.get("phase_index", -1))
            tokens = int(row.get("tokens", 0))
            if phase_index < 0 or tokens <= 0:
                raise RuntimeError("Invalid phase index or token count in shard inventory")

            def shard_artifact(field: str, suffix: str) -> Path:
                raw = row.get(field)
                if not isinstance(raw, str) or Path(raw).is_absolute():
                    raise RuntimeError(f"Unsafe shard {field} path")
                path = (root / raw).resolve()
                phase_root = (root / PHASE_DIRECTORIES[phase]).resolve()
                try:
                    path.relative_to(phase_root)
                except ValueError as exc:
                    raise RuntimeError(f"Shard {field} escapes its phase directory") from exc
                expected = f"shard-{phase_index:05d}.{suffix}"
                if path.name != expected or path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"Shard {field} does not match inventory: {path}")
                return path

            binary = shard_artifact("binary", "bin")
            index = shard_artifact("index", "index.jsonl")
            if binary.stat().st_size != tokens * 2:
                raise RuntimeError(f"Shard byte length is not uint16-exact: {binary}")
            if phase_index in phase_indices[phase]:
                raise RuntimeError(f"Duplicate shard phase index: {phase}/{phase_index}")
            phase_indices[phase].add(phase_index)
            phase_totals[phase] += tokens
            shards.append(
                ReleaseShard(
                    phase=phase,
                    phase_index=phase_index,
                    binary=binary,
                    index=index,
                    tokens=tokens,
                )
            )
        if phase_totals != PHASE_TOKENS:
            raise RuntimeError(f"Shard phase totals are not exact: {phase_totals}")
        for phase in PHASE_ORDER:
            if phase_indices[phase] != set(range(len(phase_indices[phase]))):
                raise RuntimeError(f"Shard indices for {phase} are not contiguous")
        return cls(
            root=root,
            tokenizer=tokenizer,
            release_sha256=str(descriptor.get("release_sha256", "")),
            shard_manifest_sha256=str(descriptor.get("shard_manifest_sha256", "")),
            shards=tuple(shards),
            ngram_canonical_map=ngram_canonical_map,
            ngram_canonical_ids=ngram_canonical_ids,
            ngram_canonical_map_self_sha256=str(
                descriptor.get("ngram_canonical_map_self_sha256", "")
            ),
            ngram_canonical_ids_sha256=str(
                descriptor.get("ngram_canonical_ids_sha256", "")
            ),
        )


@dataclass
class TrainingBatch:
    input_ids: torch.Tensor
    canonical_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    document_ids: torch.Tensor
    reset_mask: torch.Tensor
    phase: str
    global_token_cursor: int
    next_global_token_cursor: int
    non_padding_tokens: int
    supervised_tokens: int

    def to(self, device: torch.device, *, non_blocking: bool = True) -> "TrainingBatch":
        return TrainingBatch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            canonical_ids=self.canonical_ids.to(
                device, non_blocking=non_blocking
            ),
            labels=self.labels.to(device, non_blocking=non_blocking),
            attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
            document_ids=self.document_ids.to(device, non_blocking=non_blocking),
            reset_mask=self.reset_mask.to(device, non_blocking=non_blocking),
            phase=self.phase,
            global_token_cursor=self.global_token_cursor,
            next_global_token_cursor=self.next_global_token_cursor,
            non_padding_tokens=self.non_padding_tokens,
            supervised_tokens=self.supervised_tokens,
        )

    def pin_memory(self) -> "TrainingBatch":
        if not torch.cuda.is_available():
            return self
        return TrainingBatch(
            input_ids=self.input_ids.pin_memory(),
            canonical_ids=self.canonical_ids.pin_memory(),
            labels=self.labels.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            document_ids=self.document_ids.pin_memory(),
            reset_mask=self.reset_mask.pin_memory(),
            phase=self.phase,
            global_token_cursor=self.global_token_cursor,
            next_global_token_cursor=self.next_global_token_cursor,
            non_padding_tokens=self.non_padding_tokens,
            supervised_tokens=self.supervised_tokens,
        )


class _MMapCache:
    def __init__(self, maximum_open: int) -> None:
        if maximum_open <= 0:
            raise ValueError("maximum_open must be positive")
        self.maximum_open = maximum_open
        self._cache: OrderedDict[Path, np.memmap] = OrderedDict()

    def get(self, shard: ReleaseShard) -> np.memmap:
        existing = self._cache.pop(shard.binary, None)
        if existing is not None:
            self._cache[shard.binary] = existing
            return existing
        mapping = np.memmap(shard.binary, dtype="<u2", mode="r", shape=(shard.tokens,))
        self._cache[shard.binary] = mapping
        while len(self._cache) > self.maximum_open:
            _path, old = self._cache.popitem(last=False)
            mmap = getattr(old, "_mmap", None)
            if mmap is not None:
                mmap.close()
        return mapping

    def close(self) -> None:
        for mapping in self._cache.values():
            mmap = getattr(mapping, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._cache.clear()


class DeterministicReleaseStream:
    """Rank-addressable stream over the immutable phase shards.

    Every family walks the same phase-local permutation and the same token
    sequence. Different world/global batch sizes only change optimizer-step
    boundaries, not which one-trillion input tokens are consumed.
    """

    def __init__(
        self,
        inventory: ReleaseInventory,
        *,
        sequence_length: int,
        shard_order_seed: int,
        mmap_cache_shards: int = 4,
        mask_cross_document_targets: bool = True,
        expected_vocabulary_size: int = 65_536,
        require_canonical_ids: bool = True,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.inventory = inventory
        self.sequence_length = int(sequence_length)
        self.mask_cross_document_targets = bool(mask_cross_document_targets)
        self._cache = _MMapCache(mmap_cache_shards)
        tokenizer = Tokenizer.from_file(str(inventory.tokenizer))
        tokenizer_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
        if tokenizer_size != int(expected_vocabulary_size):
            raise RuntimeError(
                "Released tokenizer vocabulary drifted from the uint16 Metis-1.6 "
                f"contract: expected {int(expected_vocabulary_size):,}, "
                f"observed {tokenizer_size:,}"
            )
        eos_candidates = ("<|endoftext|>", "<eos>", "</s>")
        pad_candidates = ("<|padding|>", "<pad>", "<|pad|>")
        self.eos_token_id = next(
            (tokenizer.token_to_id(token) for token in eos_candidates if tokenizer.token_to_id(token) is not None),
            None,
        )
        self.pad_token_id = next(
            (tokenizer.token_to_id(token) for token in pad_candidates if tokenizer.token_to_id(token) is not None),
            None,
        )
        if self.eos_token_id is None:
            raise RuntimeError("Released tokenizer has no recognized EOS token")
        if self.pad_token_id is None:
            # Padding is always masked and never counted. EOS is a safe physical
            # fill value when the tokenizer intentionally has no pad token.
            self.pad_token_id = int(self.eos_token_id)
        if (
            inventory.ngram_canonical_map is None
            or inventory.ngram_canonical_ids is None
        ):
            if require_canonical_ids:
                raise RuntimeError(
                    "Production training requires the released canonical-ID sidecar"
                )
            self._canonical_id_lookup = np.arange(
                expected_vocabulary_size, dtype="<u2"
            )
        else:
            _descriptor, canonical_ids = validate_canonical_id_sidecar(
                manifest_path=inventory.ngram_canonical_map,
                binary_path=inventory.ngram_canonical_ids,
                tokenizer_path=inventory.tokenizer,
                expected_vocabulary_size=expected_vocabulary_size,
                expected_manifest_sha256=(
                    inventory.ngram_canonical_map_self_sha256 or None
                ),
                expected_binary_sha256=(
                    inventory.ngram_canonical_ids_sha256 or None
                ),
                recompute_from_tokenizer=False,
            )
            self._canonical_id_lookup = canonical_ids

        self._phase_shards: dict[str, list[ReleaseShard]] = {}
        self._prefix: dict[str, list[int]] = {}
        for phase_position, phase in enumerate(PHASE_ORDER):
            rows = sorted(
                (shard for shard in inventory.shards if shard.phase == phase),
                key=lambda shard: shard.phase_index,
            )
            random.Random(int(shard_order_seed) + 104_729 * (phase_position + 1)).shuffle(rows)
            prefix = [0]
            for row in rows:
                prefix.append(prefix[-1] + row.tokens)
            if prefix[-1] != PHASE_TOKENS[phase]:
                raise RuntimeError(f"Phase {phase} prefix does not match its token contract")
            self._phase_shards[phase] = rows
            self._prefix[phase] = prefix

    @staticmethod
    def phase_for_cursor(global_token_cursor: int) -> str:
        if not 0 <= global_token_cursor < TOTAL_TOKENS:
            raise ValueError("global_token_cursor must be within the 1T pretraining range")
        if global_token_cursor < PHASE_STARTS["phase_b"]:
            return "phase_a"
        if global_token_cursor < PHASE_STARTS["phase_c"]:
            return "phase_b"
        return "phase_c"

    def position(self, global_token_cursor: int) -> dict[str, Any]:
        """Resolve an exact resumable phase/shard/offset position."""

        if global_token_cursor == TOTAL_TOKENS:
            phase = "phase_c"
            rows = self._phase_shards[phase]
            return {
                "global_token_cursor": global_token_cursor,
                "phase": phase,
                "phase_offset": PHASE_TOKENS[phase],
                "shard_phase_index": rows[-1].phase_index,
                "shard_binary": str(rows[-1].binary.relative_to(self.inventory.root)),
                "offset_in_shard": rows[-1].tokens,
                "at_end": True,
            }
        phase = self.phase_for_cursor(global_token_cursor)
        phase_offset = global_token_cursor - PHASE_STARTS[phase]
        prefix = self._prefix[phase]
        shard_position = bisect.bisect_right(prefix, phase_offset) - 1
        rows = self._phase_shards[phase]
        if shard_position == len(rows):
            shard_position -= 1
        shard = rows[shard_position]
        return {
            "global_token_cursor": global_token_cursor,
            "phase": phase,
            "phase_offset": phase_offset,
            "shard_phase_index": shard.phase_index,
            "shard_binary": str(shard.binary.relative_to(self.inventory.root)),
            "offset_in_shard": phase_offset - prefix[shard_position],
            "at_end": False,
        }

    def _read(self, phase: str, phase_offset: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.int64)
        phase_size = PHASE_TOKENS[phase]
        if phase_offset < 0 or phase_offset + count > phase_size:
            raise ValueError("Requested token range crosses a phase boundary")
        prefix = self._prefix[phase]
        shards = self._phase_shards[phase]
        output = np.empty((count,), dtype=np.int64)
        written = 0
        cursor = phase_offset
        while written < count:
            shard_position = bisect.bisect_right(prefix, cursor) - 1
            if shard_position < 0 or shard_position >= len(shards):
                raise RuntimeError("Could not resolve phase token offset to a shard")
            shard = shards[shard_position]
            within = cursor - prefix[shard_position]
            take = min(count - written, shard.tokens - within)
            mapping = self._cache.get(shard)
            output[written : written + take] = mapping[within : within + take]
            written += take
            cursor += take
        return output

    def batch(
        self,
        *,
        global_token_cursor: int,
        rank: int,
        world_size: int,
        micro_batch_size: int,
    ) -> TrainingBatch:
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be within world_size")
        if world_size <= 0 or micro_batch_size <= 0:
            raise ValueError("world_size and micro_batch_size must be positive")
        phase = self.phase_for_cursor(global_token_cursor)
        phase_start = PHASE_STARTS[phase]
        phase_offset = global_token_cursor - phase_start
        remaining = PHASE_TOKENS[phase] - phase_offset
        global_capacity = world_size * micro_batch_size * self.sequence_length
        next_cursor = global_token_cursor + min(global_capacity, remaining)

        shape = (micro_batch_size, self.sequence_length)
        input_ids = torch.full(shape, int(self.pad_token_id), dtype=torch.long)
        canonical_ids = torch.full(
            shape,
            int(self._canonical_id_lookup[int(self.pad_token_id)]),
            dtype=torch.long,
        )
        labels = torch.full(shape, -100, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.bool)
        reset_mask = torch.zeros(shape, dtype=torch.bool)
        document_ids = torch.zeros(shape, dtype=torch.int32)
        local_valid = 0
        local_supervised = 0

        rank_base = rank * micro_batch_size * self.sequence_length
        for sample_index in range(micro_batch_size):
            relative = rank_base + sample_index * self.sequence_length
            valid = max(0, min(self.sequence_length, remaining - relative))
            if valid <= 0:
                reset_mask[sample_index, 0] = True
                continue
            lookahead = 1 if relative + valid < remaining else 0
            values = self._read(phase, phase_offset + relative, valid + lookahead)
            ids = torch.from_numpy(values[:valid].copy()).to(torch.long)
            input_ids[sample_index, :valid] = ids
            canonical_ids[sample_index, :valid] = torch.from_numpy(
                self._canonical_id_lookup[values[:valid]].astype(
                    np.int64, copy=True
                )
            )
            attention_mask[sample_index, :valid] = True
            reset_mask[sample_index, 0] = True
            if valid > 1:
                reset_mask[sample_index, 1:valid] = ids[:-1].eq(int(self.eos_token_id))
            document_ids[sample_index] = torch.cumsum(
                reset_mask[sample_index].to(torch.int32), dim=0
            ) - 1
            if valid > 1:
                labels[sample_index, : valid - 1] = ids[1:]
            if lookahead:
                labels[sample_index, valid - 1] = int(values[valid])
            if self.mask_cross_document_targets:
                labels[sample_index, :valid].masked_fill_(
                    ids.eq(int(self.eos_token_id)), -100
                )
            local_valid += valid
            local_supervised += int(labels[sample_index].ne(-100).sum().item())

        return TrainingBatch(
            input_ids=input_ids,
            canonical_ids=canonical_ids,
            labels=labels,
            attention_mask=attention_mask,
            document_ids=document_ids,
            reset_mask=reset_mask,
            phase=phase,
            global_token_cursor=global_token_cursor,
            next_global_token_cursor=next_cursor,
            non_padding_tokens=local_valid,
            supervised_tokens=local_supervised,
        )

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> "DeterministicReleaseStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class ReleaseBatchPrefetcher:
    """One-producer bounded queue that overlaps Lustre reads with GPU work."""

    def __init__(
        self,
        stream: DeterministicReleaseStream,
        *,
        start_cursor: int,
        rank: int,
        world_size: int,
        micro_batch_size: int,
        depth: int = 2,
    ) -> None:
        if depth <= 0:
            raise ValueError("prefetch depth must be positive")
        self.stream = stream
        self.rank = rank
        self.world_size = world_size
        self.micro_batch_size = micro_batch_size
        self._queue: queue.Queue[TrainingBatch | BaseException | None] = queue.Queue(depth)
        self._stop = threading.Event()
        self._cursor = start_cursor
        self._thread = threading.Thread(
            target=self._produce,
            name=f"metis-release-prefetch-rank-{rank}",
            daemon=True,
        )
        self._thread.start()

    def _produce(self) -> None:
        try:
            while not self._stop.is_set() and self._cursor < TOTAL_TOKENS:
                batch = self.stream.batch(
                    global_token_cursor=self._cursor,
                    rank=self.rank,
                    world_size=self.world_size,
                    micro_batch_size=self.micro_batch_size,
                ).pin_memory()
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.25)
                        break
                    except queue.Full:
                        continue
                self._cursor = batch.next_global_token_cursor
            while not self._stop.is_set():
                try:
                    self._queue.put(None, timeout=0.25)
                    break
                except queue.Full:
                    continue
        except BaseException as exc:
            while not self._stop.is_set():
                try:
                    self._queue.put(exc, timeout=0.25)
                    break
                except queue.Full:
                    continue

    def next(self, *, expected_cursor: int) -> TrainingBatch:
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            raise RuntimeError("Release prefetch worker failed") from item
        if item.global_token_cursor != expected_cursor:
            raise RuntimeError(
                f"Prefetch stream cursor mismatch: expected {expected_cursor}, "
                f"received {item.global_token_cursor}"
            )
        return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Release prefetch worker did not stop")

    def __enter__(self) -> "ReleaseBatchPrefetcher":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def assert_same_stream_prefix(
    first: DeterministicReleaseStream,
    second: DeterministicReleaseStream,
    *,
    positions: Iterable[int],
    count: int = 128,
) -> None:
    """Test helper proving two jobs resolve the same immutable data order."""

    for position in positions:
        phase = first.phase_for_cursor(position)
        phase_offset = position - PHASE_STARTS[phase]
        available = min(count, PHASE_TOKENS[phase] - phase_offset)
        left = first._read(phase, phase_offset, available)
        right = second._read(phase, phase_offset, available)
        if not np.array_equal(left, right):
            raise AssertionError(f"Streams differ at global token cursor {position}")
