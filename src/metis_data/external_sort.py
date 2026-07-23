from __future__ import annotations

import heapq
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


Record = tuple[Any, ...]


def iter_fixed_records(
    path: Path,
    record: struct.Struct,
    *,
    chunk_records: int = 16_384,
) -> Iterator[Record]:
    """Stream a fixed-width binary file and fail on every truncated record."""

    size = path.stat().st_size
    if size % record.size:
        raise RuntimeError(
            f"Corrupt fixed-record file {path}: {size} is not divisible by {record.size}"
        )
    with path.open("rb") as handle:
        while payload := handle.read(record.size * chunk_records):
            yield from record.iter_unpack(payload)


def _write_records(path: Path, record: struct.Struct, rows: Iterable[Record]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(record.pack(*row))


def _merge_group(
    paths: Sequence[Path],
    destination: Path,
    record: struct.Struct,
    key: Callable[[Record], Any],
) -> None:
    streams = [iter_fixed_records(path, record) for path in paths]
    _write_records(destination, record, heapq.merge(*streams, key=key))


def external_sort_records(
    paths: Iterable[Path],
    *,
    record: struct.Struct,
    key: Callable[[Record], Any],
    temporary_directory: Path,
    chunk_records: int = 250_000,
    merge_fan_in: int = 64,
) -> Iterator[Record]:
    """Bounded-memory external sort for fixed-width binary records.

    Input files are consumed sequentially, sorted runs are written atomically
    beneath the requested scratch directory, and multi-pass merges cap both
    open descriptors and heap memory. The caller never has to construct a
    corpus-sized Python list or database index.
    """

    if chunk_records < 1 or merge_fan_in < 2:
        raise ValueError("chunk_records must be positive and merge_fan_in at least two")
    temporary_directory.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="metis-external-sort-", dir=temporary_directory))
    try:
        runs: list[Path] = []
        buffer: list[Record] = []
        run_index = 0
        for path in paths:
            for row in iter_fixed_records(Path(path), record):
                buffer.append(row)
                if len(buffer) >= chunk_records:
                    buffer.sort(key=key)
                    run = work / f"run-{run_index:08d}.bin"
                    _write_records(run, record, buffer)
                    runs.append(run)
                    buffer.clear()
                    run_index += 1
        if buffer:
            buffer.sort(key=key)
            run = work / f"run-{run_index:08d}.bin"
            _write_records(run, record, buffer)
            runs.append(run)
            buffer.clear()

        generation = 0
        while len(runs) > merge_fan_in:
            merged: list[Path] = []
            for group_index, offset in enumerate(range(0, len(runs), merge_fan_in)):
                group = runs[offset : offset + merge_fan_in]
                destination = work / f"merge-{generation:04d}-{group_index:08d}.bin"
                _merge_group(group, destination, record, key)
                merged.append(destination)
                for path in group:
                    path.unlink()
            runs = merged
            generation += 1
        if not runs:
            return
        streams = [iter_fixed_records(path, record) for path in runs]
        yield from heapq.merge(*streams, key=key)
    finally:
        shutil.rmtree(work, ignore_errors=True)
