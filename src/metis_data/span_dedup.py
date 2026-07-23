from __future__ import annotations

import contextlib
import copy
import hashlib
import heapq
import json
import os
import re
import shutil
import struct
import tempfile
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Sequence


# The on-disk formats are deliberately fixed width.  The first pass writes a
# 128-bit SHA-256 prefix and source location (28 bytes); the larger exact record
# is materialized only for cross-document prefix hits.  A prefix collision is
# harmless: the second pass recomputes the full SHA-256, and the exact finder
# keeps distinct full digests separate.
PREFILTER_SCHEMA = "metis.span-prefilter-signatures/v1"
CANDIDATE_SCHEMA = "metis.span-prefilter-candidates/v1"
SIGNATURE_SCHEMA = "metis.span-signatures/v2"
REMOVAL_SCHEMA = "metis.span-removals/v2"
PREFILTER_RECORD = struct.Struct("<16sIII")
CANDIDATE_RECORD = struct.Struct("<III16s")
SIGNATURE_RECORD = struct.Struct("<32sIIIq32s")
REMOVAL_RECORD = struct.Struct("<II32s")
UNSORTED_REMOVAL_RECORD = struct.Struct("<III32s")
UINT32_MAX = (1 << 32) - 1
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str
    normalized: str
    words: int


@dataclass(frozen=True)
class SpanSignature:
    digest: bytes
    sentence_start: int


@dataclass(frozen=True)
class SpanStripResult:
    text: str
    requested_spans: int
    removed_sentences: int
    remaining_words: int
    remaining_sentences: int


def _validate_policy(sentence_count: int, minimum_span_words: int) -> None:
    if sentence_count < 1:
        raise ValueError("sentence_count must be at least 1")
    if minimum_span_words < 1:
        raise ValueError("minimum_span_words must be at least 1")


def _priority(value: Any) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Document priority must be an integer, got {value!r}") from exc
    return max(INT64_MIN, min(INT64_MAX, priority))


def stable_document_tie(document_id: Any) -> bytes:
    """Return the deterministic, fixed-width tie key derived from a stable id."""

    return hashlib.sha256(str(document_id).encode("utf-8")).digest()


def _canonical_sentence(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def sentence_spans(text: str) -> tuple[SentenceSpan, ...]:
    """Split text deterministically while preserving source character offsets.

    Full stops, question marks, exclamation marks, and line boundaries terminate
    a sentence.  This intentionally conservative splitter behaves predictably
    for prose, Markdown, lists, and extracted web text; exact matching does not
    depend on a mutable language model or external sentence-segmentation package.
    """

    spans: list[SentenceSpan] = []
    length = len(text)
    cursor = 0

    def skip_space(position: int) -> int:
        while position < length and text[position].isspace():
            position += 1
        return position

    start = skip_space(0)
    cursor = start
    while cursor < length:
        character = text[cursor]
        boundary = -1
        if character == "\n":
            boundary = cursor
        elif character in ".!?":
            end = cursor + 1
            while end < length and text[end] in ".!?":
                end += 1
            if end == length or text[end].isspace():
                boundary = end
                cursor = end - 1

        if boundary >= 0:
            raw = text[start:boundary]
            stripped = raw.strip()
            if stripped:
                leading = len(raw) - len(raw.lstrip())
                trailing = len(raw.rstrip())
                absolute_start = start + leading
                absolute_end = start + trailing
                normalized = _canonical_sentence(text[absolute_start:absolute_end])
                if normalized:
                    spans.append(
                        SentenceSpan(
                            start=absolute_start,
                            end=absolute_end,
                            text=text[absolute_start:absolute_end],
                            normalized=normalized,
                            words=len(WORD_RE.findall(normalized)),
                        )
                    )
            start = skip_space(boundary + (1 if character == "\n" else 0))
            cursor = start
            continue
        cursor += 1

    if start < length:
        raw = text[start:]
        stripped = raw.strip()
        if stripped:
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            absolute_start = start + leading
            absolute_end = start + trailing
            normalized = _canonical_sentence(text[absolute_start:absolute_end])
            if normalized:
                spans.append(
                    SentenceSpan(
                        start=absolute_start,
                        end=absolute_end,
                        text=text[absolute_start:absolute_end],
                        normalized=normalized,
                        words=len(WORD_RE.findall(normalized)),
                    )
                )
    return tuple(spans)


def span_digest(sentences: Sequence[SentenceSpan], sentence_start: int, sentence_count: int) -> bytes:
    if sentence_start < 0 or sentence_start + sentence_count > len(sentences):
        raise IndexError(
            f"Sentence window [{sentence_start}, {sentence_start + sentence_count}) "
            f"is outside a {len(sentences)}-sentence document"
        )
    payload = "\x1e".join(
        sentence.normalized for sentence in sentences[sentence_start : sentence_start + sentence_count]
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def iter_span_signatures(
    text: str,
    *,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
) -> Iterator[SpanSignature]:
    """Yield exact normalized sentence-window signatures for one document."""

    _validate_policy(sentence_count, minimum_span_words)
    sentences = sentence_spans(text)
    if len(sentences) < sentence_count:
        return
    window_words = sum(sentence.words for sentence in sentences[:sentence_count])
    for sentence_start in range(len(sentences) - sentence_count + 1):
        if sentence_start:
            window_words += sentences[sentence_start + sentence_count - 1].words
            window_words -= sentences[sentence_start - 1].words
        if window_words < minimum_span_words:
            continue
        yield SpanSignature(
            digest=span_digest(sentences, sentence_start, sentence_count),
            sentence_start=sentence_start,
        )


@dataclass(frozen=True)
class _ManifestedFile:
    path: Path
    sha256: str | None = None


class _BoundedFilePool:
    """LRU file pool that also builds manifests without rereading output."""

    def __init__(self, root: Path, maximum_open_files: int, *, suffix: str) -> None:
        if maximum_open_files < 1:
            raise ValueError("maximum_open_files must be at least 1")
        self.root = root
        self.maximum_open_files = maximum_open_files
        self.suffix = suffix
        self.handles: OrderedDict[int, BinaryIO] = OrderedDict()
        self.records: dict[int, int] = {}
        self.bytes: dict[int, int] = {}
        self.hashers: dict[int, Any] = {}

    def path(self, key: int) -> Path:
        return self.root / f"{key:04d}.{self.suffix}"

    def write(self, key: int, payload: bytes) -> None:
        handle = self.handles.pop(key, None)
        if handle is None:
            if len(self.handles) >= self.maximum_open_files:
                _, oldest = self.handles.popitem(last=False)
                oldest.close()
            handle = self.path(key).open("ab")
        self.handles[key] = handle
        handle.write(payload)
        self.records[key] = self.records.get(key, 0) + 1
        self.bytes[key] = self.bytes.get(key, 0) + len(payload)
        hasher = self.hashers.setdefault(key, hashlib.sha256())
        hasher.update(payload)

    def close(self) -> None:
        while self.handles:
            _, handle = self.handles.popitem(last=False)
            handle.close()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read repeated-span manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Repeated-span manifest {path} must contain a JSON object")
    return payload


def _require_fields(path: Path, payload: dict[str, Any], expected: dict[str, Any]) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"Repeated-span manifest mismatch at {path}: {field}="
                f"{payload.get(field)!r}, expected {value!r}"
            )


def _output_entry(
    *,
    key: int,
    relative_path: str,
    records: int,
    byte_count: int,
    sha256: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "path": relative_path,
        "present": records > 0,
        "records": records,
        "bytes": byte_count,
        "sha256": sha256 if records else EMPTY_SHA256,
    }


def _publish_rank_pool(
    pool: _BoundedFilePool,
    stage: Path,
    output_root: Path,
    *,
    rank: int,
    finder_workers: int,
    suffix: str,
) -> dict[str, dict[str, Any]]:
    pool.close()
    outputs: dict[str, dict[str, Any]] = {}
    for bucket in range(finder_workers):
        relative = f"{bucket:04d}/{rank:06d}.{suffix}"
        destination = output_root / relative
        source = stage / f"{bucket:04d}.{suffix}"
        records = pool.records.get(bucket, 0)
        byte_count = pool.bytes.get(bucket, 0)
        if records:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            digest = pool.hashers[bucket].hexdigest()
        else:
            destination.unlink(missing_ok=True)
            digest = EMPTY_SHA256
        outputs[f"{bucket:04d}"] = _output_entry(
            key=bucket,
            relative_path=relative,
            records=records,
            byte_count=byte_count,
            sha256=digest,
        )
    return outputs


def _validate_output(
    root: Path,
    manifest_path: Path,
    entry: Any,
    *,
    expected_key: int,
    expected_relative_path: str,
    record_size: int,
) -> _ManifestedFile | None:
    if not isinstance(entry, dict):
        raise RuntimeError(f"Output entry in {manifest_path} must be an object")
    expected = {
        "key": expected_key,
        "path": expected_relative_path,
    }
    _require_fields(manifest_path, entry, expected)
    try:
        records = int(entry["records"])
        byte_count = int(entry["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid output counts in {manifest_path}") from exc
    present = entry.get("present")
    digest = entry.get("sha256")
    if records < 0 or byte_count != records * record_size:
        raise RuntimeError(
            f"Invalid output size in {manifest_path} for {expected_relative_path}: "
            f"{records} records and {byte_count} bytes"
        )
    if present is not (records > 0):
        raise RuntimeError(
            f"Invalid present flag in {manifest_path} for {expected_relative_path}"
        )
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(
            f"Invalid SHA-256 in {manifest_path} for {expected_relative_path}"
        )
    if records == 0 and digest != EMPTY_SHA256:
        raise RuntimeError(
            f"Empty output in {manifest_path} has a non-empty SHA-256"
        )
    path = root / expected_relative_path
    if records == 0:
        if path.exists():
            raise RuntimeError(
                f"Manifest {manifest_path} marks {path} empty, but the file exists"
            )
        return None
    if not path.is_file():
        raise RuntimeError(f"Manifest {manifest_path} references missing output {path}")
    actual_size = path.stat().st_size
    if actual_size != byte_count:
        raise RuntimeError(
            f"Output size mismatch for {path}: {actual_size}, expected {byte_count}"
        )
    return _ManifestedFile(path=path, sha256=digest)


def _manifest_ranks(root: Path, total_ranks: int | None) -> tuple[list[int], int]:
    manifest_root = root / "_manifests"
    ranks: set[int] = set()
    for path in manifest_root.glob("*.json"):
        try:
            ranks.add(int(path.stem))
        except ValueError:
            continue
    if total_ranks is None:
        if not ranks:
            raise RuntimeError(f"No repeated-span rank manifests found in {manifest_root}")
        total_ranks = max(ranks) + 1
    if total_ranks < 1:
        raise ValueError("total_ranks must be at least 1")
    expected = set(range(total_ranks))
    if ranks != expected:
        missing = sorted(expected - ranks)
        extra = sorted(ranks - expected)
        raise RuntimeError(
            f"Incomplete repeated-span rank manifest inventory in {manifest_root}; "
            f"missing={missing[:16]}, extra={extra[:16]}"
        )
    return list(range(total_ranks)), total_ranks


def _rank_bucket_inputs(
    root: Path,
    *,
    schema: str,
    suffix: str,
    record_size: int,
    bucket: int,
    finder_workers: int | None,
    sentence_count: int,
    minimum_span_words: int,
    total_ranks: int | None,
) -> tuple[list[_ManifestedFile], int, int]:
    ranks, resolved_total = _manifest_ranks(root, total_ranks)
    inputs: list[_ManifestedFile] = []
    resolved_workers = finder_workers
    for rank in ranks:
        manifest_path = root / "_manifests" / f"{rank:06d}.json"
        manifest = _read_json(manifest_path)
        if resolved_workers is None:
            try:
                resolved_workers = int(manifest["finder_workers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid finder_workers in {manifest_path}") from exc
        if resolved_workers < 1 or bucket < 0 or bucket >= resolved_workers:
            raise ValueError(f"bucket {bucket} is outside finder_workers={resolved_workers}")
        expected_fields = {
            "schema": schema,
            "rank": rank,
            "finder_workers": resolved_workers,
            "sentence_count": sentence_count,
            "minimum_span_words": minimum_span_words,
            "record_size": record_size,
        }
        if schema == PREFILTER_SCHEMA:
            expected_fields["digest_prefix_bytes"] = 16
        _require_fields(manifest_path, manifest, expected_fields)
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {
            f"{value:04d}" for value in range(resolved_workers)
        }:
            raise RuntimeError(f"Incomplete bucket inventory in {manifest_path}")
        manifested = _validate_output(
            root,
            manifest_path,
            outputs[f"{bucket:04d}"],
            expected_key=bucket,
            expected_relative_path=f"{bucket:04d}/{rank:06d}.{suffix}",
            record_size=record_size,
        )
        if manifested is not None:
            inputs.append(manifested)
    assert resolved_workers is not None
    return inputs, resolved_workers, resolved_total


def _iter_fixed_records(
    source: _ManifestedFile | Path,
    record: struct.Struct,
    chunk_records: int,
) -> Iterator[tuple[Any, ...]]:
    path = source.path if isinstance(source, _ManifestedFile) else source
    expected_hash = source.sha256 if isinstance(source, _ManifestedFile) else None
    size = path.stat().st_size
    if size % record.size:
        raise RuntimeError(
            f"Corrupt fixed-record file {path}: {size} bytes is not divisible by "
            f"record size {record.size}"
        )
    hasher = hashlib.sha256() if expected_hash is not None else None
    with path.open("rb") as handle:
        while payload := handle.read(record.size * chunk_records):
            if hasher is not None:
                hasher.update(payload)
            yield from record.iter_unpack(payload)
    if hasher is not None and hasher.hexdigest() != expected_hash:
        raise RuntimeError(
            f"SHA-256 mismatch for repeated-span artifact {path}: "
            f"{hasher.hexdigest()}, expected {expected_hash}"
        )


def _write_run(path: Path, rows: Sequence[tuple[Any, ...]], record: struct.Struct) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(record.pack(*row))


def _merge_runs(
    paths: Sequence[Path],
    destination: Path,
    *,
    record: struct.Struct,
    key: Any,
) -> None:
    iterators = [_iter_fixed_records(path, record, 8_192) for path in paths]
    heap: list[tuple[Any, int, tuple[Any, ...]]] = []
    for index, iterator in enumerate(iterators):
        row = next(iterator, None)
        if row is not None:
            heapq.heappush(heap, (key(row), index, row))
    with destination.open("wb") as handle:
        while heap:
            _, index, row = heapq.heappop(heap)
            handle.write(record.pack(*row))
            following = next(iterators[index], None)
            if following is not None:
                heapq.heappush(heap, (key(following), index, following))


@contextlib.contextmanager
def _external_sorted_records(
    inputs: Sequence[_ManifestedFile | Path],
    *,
    record: struct.Struct,
    key: Any,
    work_root: Path,
    prefix: str,
    chunk_records: int,
    maximum_open_runs: int,
) -> Iterator[tuple[Iterator[tuple[Any, ...]], dict[str, int]]]:
    """Yield a bounded-memory global sort over fixed-width input records."""

    if chunk_records < 1:
        raise ValueError("chunk_records must be at least 1")
    if maximum_open_runs < 2:
        raise ValueError("maximum_open_runs must be at least 2")
    work_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{prefix}-", dir=work_root))
    rows: list[tuple[Any, ...]] = []
    runs: list[Path] = []
    input_records = 0

    def flush() -> None:
        nonlocal rows
        if not rows:
            return
        rows.sort(key=key)
        path = stage / f"run-{len(runs):08d}.bin"
        _write_run(path, rows, record)
        runs.append(path)
        rows = []

    try:
        for source in inputs:
            for row in _iter_fixed_records(source, record, chunk_records):
                rows.append(row)
                input_records += 1
                if len(rows) >= chunk_records:
                    flush()
        flush()
        initial_runs = len(runs)
        merge_passes = 0
        generation = 0
        if not runs:
            empty = stage / "empty.bin"
            empty.touch()
            runs = [empty]
        while len(runs) > 1:
            merged: list[Path] = []
            for group_index in range(0, len(runs), maximum_open_runs):
                group = runs[group_index : group_index + maximum_open_runs]
                if len(group) == 1:
                    merged.append(group[0])
                    continue
                destination = stage / f"merge-{generation:04d}-{len(merged):08d}.bin"
                _merge_runs(group, destination, record=record, key=key)
                for path in group:
                    path.unlink()
                merged.append(destination)
            runs = merged
            generation += 1
            merge_passes += 1
        report = {
            "input_records": input_records,
            "initial_runs": initial_runs,
            "merge_passes": merge_passes,
            "maximum_open_runs": maximum_open_runs,
            "chunk_records": chunk_records,
        }
        yield _iter_fixed_records(runs[0], record, chunk_records), report
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def write_span_prefilter_signatures(
    documents: Iterable[Any],
    output_root: Path,
    *,
    rank: int,
    finder_workers: int,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
    maximum_open_files: int = 32,
) -> dict[str, Any]:
    """Write the compact first-pass span inventory for one corpus rank."""

    _validate_policy(sentence_count, minimum_span_words)
    if finder_workers < 1:
        raise ValueError("finder_workers must be at least 1")
    if rank < 0 or rank > UINT32_MAX:
        raise ValueError(f"rank must fit uint32, got {rank}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".span-prefilter-{rank:06d}-", dir=output_root))
    pool = _BoundedFilePool(stage, maximum_open_files, suffix="compact")
    documents_seen = 0
    records_written = 0
    documents_with_records = 0
    try:
        for document_index, document in enumerate(documents):
            if document_index > UINT32_MAX:
                raise OverflowError(f"rank {rank} contains more than uint32-addressable documents")
            documents_seen += 1
            found = False
            for signature in iter_span_signatures(
                str(document.text),
                sentence_count=sentence_count,
                minimum_span_words=minimum_span_words,
            ):
                found = True
                bucket = int.from_bytes(signature.digest[:8], "little") % finder_workers
                pool.write(
                    bucket,
                    PREFILTER_RECORD.pack(
                        signature.digest[:16],
                        rank,
                        document_index,
                        signature.sentence_start,
                    ),
                )
                records_written += 1
            documents_with_records += int(found)
        outputs = _publish_rank_pool(
            pool,
            stage,
            output_root,
            rank=rank,
            finder_workers=finder_workers,
            suffix="compact",
        )
        report: dict[str, Any] = {
            "schema": PREFILTER_SCHEMA,
            "rank": rank,
            "finder_workers": finder_workers,
            "sentence_count": sentence_count,
            "minimum_span_words": minimum_span_words,
            "record_size": PREFILTER_RECORD.size,
            "digest_prefix_bytes": 16,
            "documents": documents_seen,
            "documents_with_signatures": documents_with_records,
            "signatures": records_written,
            "populated_buckets": sum(entry["present"] for entry in outputs.values()),
            "outputs": outputs,
        }
        _atomic_json(output_root / "_manifests" / f"{rank:06d}.json", report)
        return report
    finally:
        pool.close()
        shutil.rmtree(stage, ignore_errors=True)


def _publish_candidate_outputs(
    sorted_candidates: Iterator[tuple[Any, ...]],
    candidate_root: Path,
    stage: Path,
    *,
    bucket: int,
    total_ranks: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    counts: dict[int, int] = {}
    bytes_by_rank: dict[int, int] = {}
    hashers: dict[int, Any] = {}
    current_rank: int | None = None
    handle: BinaryIO | None = None
    total = 0
    try:
        for rank, document, sentence_start, digest in sorted_candidates:
            rank = int(rank)
            if rank < 0 or rank >= total_ranks:
                raise RuntimeError(f"Candidate record contains out-of-range rank {rank}")
            if current_rank != rank:
                if handle is not None:
                    handle.close()
                current_rank = rank
                handle = (stage / f"{rank:06d}.candidate").open("wb")
            payload = CANDIDATE_RECORD.pack(
                rank,
                int(document),
                int(sentence_start),
                bytes(digest),
            )
            assert handle is not None
            handle.write(payload)
            counts[rank] = counts.get(rank, 0) + 1
            bytes_by_rank[rank] = bytes_by_rank.get(rank, 0) + len(payload)
            hashers.setdefault(rank, hashlib.sha256()).update(payload)
            total += 1
    finally:
        if handle is not None:
            handle.close()

    destination_folder = candidate_root / f"{bucket:04d}"
    destination_folder.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for rank in range(total_ranks):
        relative = f"{bucket:04d}/{rank:06d}.candidate"
        source = stage / f"{rank:06d}.candidate"
        destination = candidate_root / relative
        count = counts.get(rank, 0)
        if count:
            source.replace(destination)
            digest = hashers[rank].hexdigest()
        else:
            destination.unlink(missing_ok=True)
            digest = EMPTY_SHA256
        outputs[f"{rank:06d}"] = _output_entry(
            key=rank,
            relative_path=relative,
            records=count,
            byte_count=bytes_by_rank.get(rank, 0),
            sha256=digest,
        )
    for stale in destination_folder.glob("*.candidate"):
        try:
            stale_rank = int(stale.stem)
        except ValueError:
            stale.unlink()
            continue
        if stale_rank >= total_ranks:
            stale.unlink()
    return outputs, total


def find_repeated_span_candidates(
    prefilter_root: Path,
    candidate_root: Path,
    *,
    bucket: int,
    finder_workers: int | None = None,
    total_ranks: int | None = None,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
    chunk_records: int = 250_000,
    maximum_open_runs: int = 64,
    temporary_directory: Path | None = None,
) -> dict[str, Any]:
    """Globally select spans present in at least two distinct documents."""

    _validate_policy(sentence_count, minimum_span_words)
    prefilter_root = Path(prefilter_root)
    candidate_root = Path(candidate_root)
    inputs, resolved_workers, resolved_total = _rank_bucket_inputs(
        prefilter_root,
        schema=PREFILTER_SCHEMA,
        suffix="compact",
        record_size=PREFILTER_RECORD.size,
        bucket=bucket,
        finder_workers=finder_workers,
        sentence_count=sentence_count,
        minimum_span_words=minimum_span_words,
        total_ranks=total_ranks,
    )
    work_root = (
        Path(temporary_directory)
        if temporary_directory is not None
        else candidate_root / ".prefilter-work"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".candidate-{bucket:04d}-", dir=work_root))
    unsorted_candidates = stage / "candidates.unsorted"
    repeated_digests = 0
    try:
        with _external_sorted_records(
            inputs,
            record=PREFILTER_RECORD,
            key=lambda row: (row[0], row[1], row[2], row[3]),
            work_root=stage,
            prefix="compact-sort",
            chunk_records=chunk_records,
            maximum_open_runs=maximum_open_runs,
        ) as (records, compact_sort):
            with unsorted_candidates.open("wb") as output:
                current_digest: bytes | None = None
                first_identity: tuple[int, int] | None = None
                buffered: list[tuple[Any, ...]] = []
                spool: BinaryIO | None = None
                qualified = False
                buffer_limit = min(max(chunk_records, 1), 4_096)

                def discard_buffer() -> None:
                    nonlocal buffered, spool
                    buffered = []
                    if spool is not None:
                        spool.close()
                        spool = None

                def buffer_record(row: tuple[Any, ...]) -> None:
                    nonlocal buffered, spool
                    if spool is None and len(buffered) < buffer_limit:
                        buffered.append(row)
                        return
                    if spool is None:
                        spool = tempfile.TemporaryFile(dir=stage)
                        for saved in buffered:
                            spool.write(PREFILTER_RECORD.pack(*saved))
                        buffered = []
                    spool.write(PREFILTER_RECORD.pack(*row))

                def emit(row: tuple[Any, ...]) -> None:
                    digest, rank, document, sentence_start = row
                    output.write(
                        CANDIDATE_RECORD.pack(
                            int(rank),
                            int(document),
                            int(sentence_start),
                            bytes(digest),
                        )
                    )

                def flush_buffer() -> None:
                    nonlocal buffered, spool
                    if spool is not None:
                        spool.seek(0)
                        while payload := spool.read(PREFILTER_RECORD.size * 8_192):
                            for saved in PREFILTER_RECORD.iter_unpack(payload):
                                emit(saved)
                        spool.close()
                        spool = None
                    else:
                        for saved in buffered:
                            emit(saved)
                    buffered = []

                for row in records:
                    digest = bytes(row[0])
                    identity = (int(row[1]), int(row[2]))
                    if digest != current_digest:
                        discard_buffer()
                        current_digest = digest
                        first_identity = identity
                        qualified = False
                        buffer_record(row)
                        continue
                    if not qualified and identity == first_identity:
                        buffer_record(row)
                        continue
                    if not qualified:
                        flush_buffer()
                        qualified = True
                        repeated_digests += 1
                    emit(row)
                discard_buffer()

        with _external_sorted_records(
            [unsorted_candidates],
            record=CANDIDATE_RECORD,
            key=lambda row: (row[0], row[1], row[2], row[3]),
            work_root=stage,
            prefix="candidate-sort",
            chunk_records=chunk_records,
            maximum_open_runs=maximum_open_runs,
        ) as (records, candidate_sort):
            publish_stage = stage / "publish"
            publish_stage.mkdir()
            outputs, candidate_records = _publish_candidate_outputs(
                records,
                candidate_root,
                publish_stage,
                bucket=bucket,
                total_ranks=resolved_total,
            )
        report: dict[str, Any] = {
            "schema": CANDIDATE_SCHEMA,
            "bucket": bucket,
            "finder_workers": resolved_workers,
            "total_ranks": resolved_total,
            "sentence_count": sentence_count,
            "minimum_span_words": minimum_span_words,
            "prefilter_record_size": PREFILTER_RECORD.size,
            "candidate_record_size": CANDIDATE_RECORD.size,
            "digest_prefix_bytes": 16,
            "prefilter_files": len(inputs),
            "prefilter_records": compact_sort["input_records"],
            "repeated_digests": repeated_digests,
            "candidate_records": candidate_records,
            "compact_sort": compact_sort,
            "candidate_sort": candidate_sort,
            "outputs": outputs,
        }
        _atomic_json(candidate_root / "_manifests" / f"{bucket:04d}.json", report)
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _candidate_inputs_for_rank(
    candidate_root: Path,
    *,
    rank: int,
    finder_workers: int,
    total_ranks: int | None,
    sentence_count: int,
    minimum_span_words: int,
) -> tuple[list[_ManifestedFile], int]:
    inputs: list[_ManifestedFile] = []
    resolved_total = total_ranks
    if resolved_total is not None and (
        resolved_total < 1 or rank < 0 or rank >= resolved_total
    ):
        raise ValueError(f"rank {rank} is outside total_ranks={resolved_total}")
    for bucket in range(finder_workers):
        manifest_path = candidate_root / "_manifests" / f"{bucket:04d}.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing repeated-span candidate manifest {manifest_path}")
        manifest = _read_json(manifest_path)
        if resolved_total is None:
            try:
                resolved_total = int(manifest["total_ranks"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid total_ranks in {manifest_path}") from exc
            if resolved_total < 1 or rank < 0 or rank >= resolved_total:
                raise ValueError(f"rank {rank} is outside total_ranks={resolved_total}")
        _require_fields(
            manifest_path,
            manifest,
            {
                "schema": CANDIDATE_SCHEMA,
                "bucket": bucket,
                "finder_workers": finder_workers,
                "total_ranks": resolved_total,
                "sentence_count": sentence_count,
                "minimum_span_words": minimum_span_words,
                "candidate_record_size": CANDIDATE_RECORD.size,
                "digest_prefix_bytes": 16,
            },
        )
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {
            f"{value:06d}" for value in range(resolved_total)
        }:
            raise RuntimeError(f"Incomplete rank inventory in {manifest_path}")
        manifested = _validate_output(
            candidate_root,
            manifest_path,
            outputs[f"{rank:06d}"],
            expected_key=rank,
            expected_relative_path=f"{bucket:04d}/{rank:06d}.candidate",
            record_size=CANDIDATE_RECORD.size,
        )
        if manifested is not None:
            inputs.append(manifested)
    assert resolved_total is not None
    if rank < 0 or rank >= resolved_total:
        raise ValueError(f"rank {rank} is outside total_ranks={resolved_total}")
    return inputs, resolved_total


def _iter_candidate_locations(
    inputs: Sequence[_ManifestedFile],
    *,
    rank: int,
) -> Iterator[tuple[int, int, bytes]]:
    streams: list[Iterator[tuple[int, int, bytes]]] = []
    for source in inputs:
        def stream(current: _ManifestedFile = source) -> Iterator[tuple[int, int, bytes]]:
            for recorded_rank, document, sentence_start, digest in _iter_fixed_records(
                current, CANDIDATE_RECORD, 8_192
            ):
                if int(recorded_rank) != rank:
                    raise RuntimeError(
                        f"Candidate file {current.path} contains rank {recorded_rank}, expected {rank}"
                    )
                yield int(document), int(sentence_start), bytes(digest)

        streams.append(stream())
    previous: tuple[int, int, bytes] | None = None
    for candidate in heapq.merge(*streams):
        if candidate == previous:
            raise RuntimeError(f"Duplicate repeated-span candidate location {candidate[:2]}")
        yield candidate
        previous = candidate


def write_span_signatures(
    documents: Iterable[Any],
    output_root: Path,
    *,
    rank: int,
    finder_workers: int,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
    maximum_open_files: int = 32,
    candidate_root: Path | None = None,
    total_ranks: int | None = None,
) -> dict[str, Any]:
    """Write full signatures, optionally restricted to global prefilter hits."""

    _validate_policy(sentence_count, minimum_span_words)
    if finder_workers < 1:
        raise ValueError("finder_workers must be at least 1")
    if rank < 0 or rank > UINT32_MAX:
        raise ValueError(f"rank must fit uint32, got {rank}")
    candidate_inputs: list[_ManifestedFile] = []
    resolved_total = total_ranks
    if candidate_root is not None:
        candidate_inputs, resolved_total = _candidate_inputs_for_rank(
            Path(candidate_root),
            rank=rank,
            finder_workers=finder_workers,
            total_ranks=total_ranks,
            sentence_count=sentence_count,
            minimum_span_words=minimum_span_words,
        )
    candidates = (
        _iter_candidate_locations(candidate_inputs, rank=rank)
        if candidate_root is not None
        else iter(())
    )
    pending = next(candidates, None)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".span-rank-{rank:06d}-", dir=output_root))
    pool = _BoundedFilePool(stage, maximum_open_files, suffix="sig")
    documents_seen = 0
    signatures_written = 0
    documents_with_signatures = 0
    try:
        for document_index, document in enumerate(documents):
            if document_index > UINT32_MAX:
                raise OverflowError(f"rank {rank} contains more than uint32-addressable documents")
            if pending is not None and pending[0] < document_index:
                raise RuntimeError(
                    f"Candidate references missing document {pending[0]} before "
                    f"document {document_index} on rank {rank}"
                )
            documents_seen += 1
            priority = _priority(document.metadata.get("priority", 1))
            tie = stable_document_tie(document.id)
            found = False
            for signature in iter_span_signatures(
                str(document.text),
                sentence_count=sentence_count,
                minimum_span_words=minimum_span_words,
            ):
                if candidate_root is not None:
                    if (
                        pending is not None
                        and pending[0] == document_index
                        and pending[1] < signature.sentence_start
                    ):
                        raise RuntimeError(
                            f"Candidate start {pending[1]} is absent from document "
                            f"{document_index} on rank {rank}"
                        )
                    if (
                        pending is None
                        or pending[0] != document_index
                        or pending[1] != signature.sentence_start
                    ):
                        continue
                    if pending[2] != signature.digest[:16]:
                        raise RuntimeError(
                            f"Candidate digest mismatch for rank {rank}, document "
                            f"{document_index}, sentence start {signature.sentence_start}"
                        )
                    pending = next(candidates, None)
                found = True
                bucket = int.from_bytes(signature.digest[:8], "little") % finder_workers
                pool.write(
                    bucket,
                    SIGNATURE_RECORD.pack(
                        signature.digest,
                        rank,
                        document_index,
                        signature.sentence_start,
                        priority,
                        tie,
                    ),
                )
                signatures_written += 1
            if pending is not None and pending[0] == document_index:
                raise RuntimeError(
                    f"Candidate start {pending[1]} is absent from document "
                    f"{document_index} on rank {rank}"
                )
            documents_with_signatures += int(found)
        if pending is not None:
            raise RuntimeError(
                f"Candidate references document {pending[0]} beyond the end of rank {rank}"
            )
        outputs = _publish_rank_pool(
            pool,
            stage,
            output_root,
            rank=rank,
            finder_workers=finder_workers,
            suffix="sig",
        )
        report: dict[str, Any] = {
            "schema": SIGNATURE_SCHEMA,
            "rank": rank,
            "finder_workers": finder_workers,
            "total_ranks": resolved_total if resolved_total is not None else -1,
            "sentence_count": sentence_count,
            "minimum_span_words": minimum_span_words,
            "record_size": SIGNATURE_RECORD.size,
            "documents": documents_seen,
            "documents_with_signatures": documents_with_signatures,
            "signatures": signatures_written,
            "prefiltered": candidate_root is not None,
            "candidate_files": len(candidate_inputs),
            "populated_buckets": sum(entry["present"] for entry in outputs.values()),
            "outputs": outputs,
        }
        _atomic_json(output_root / "_manifests" / f"{rank:06d}.json", report)
        return report
    finally:
        pool.close()
        shutil.rmtree(stage, ignore_errors=True)


def _publish_removal_outputs(
    sorted_removals: Iterator[tuple[Any, ...]],
    removal_root: Path,
    stage: Path,
    *,
    bucket: int,
    total_ranks: int,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    counts: dict[int, int] = {}
    bytes_by_rank: dict[int, int] = {}
    hashers: dict[int, Any] = {}
    current_rank: int | None = None
    handle: BinaryIO | None = None
    previous: tuple[int, int, int, bytes] | None = None
    previous_document: tuple[int, int] | None = None
    removal_starts = 0
    affected_documents = 0
    try:
        for rank, document, sentence_start, digest in sorted_removals:
            row = (int(rank), int(document), int(sentence_start), bytes(digest))
            if row == previous:
                continue
            previous = row
            if row[0] < 0 or row[0] >= total_ranks:
                raise RuntimeError(f"Removal record contains out-of-range rank {row[0]}")
            if current_rank != row[0]:
                if handle is not None:
                    handle.close()
                current_rank = row[0]
                handle = (stage / f"{row[0]:06d}.remove").open("wb")
            payload = REMOVAL_RECORD.pack(row[1], row[2], row[3])
            assert handle is not None
            handle.write(payload)
            counts[row[0]] = counts.get(row[0], 0) + 1
            bytes_by_rank[row[0]] = bytes_by_rank.get(row[0], 0) + len(payload)
            hashers.setdefault(row[0], hashlib.sha256()).update(payload)
            removal_starts += 1
            identity = (row[0], row[1])
            if identity != previous_document:
                affected_documents += 1
                previous_document = identity
    finally:
        if handle is not None:
            handle.close()

    destination_folder = removal_root / f"{bucket:04d}"
    destination_folder.mkdir(parents=True, exist_ok=True)
    for stale in destination_folder.glob("*.remove"):
        stale.unlink()
    outputs: dict[str, dict[str, Any]] = {}
    for rank in range(total_ranks):
        relative = f"{bucket:04d}/{rank:06d}.remove"
        source = stage / f"{rank:06d}.remove"
        destination = removal_root / relative
        count = counts.get(rank, 0)
        if count:
            source.replace(destination)
            digest = hashers[rank].hexdigest()
        else:
            digest = EMPTY_SHA256
        outputs[f"{rank:06d}"] = _output_entry(
            key=rank,
            relative_path=relative,
            records=count,
            byte_count=bytes_by_rank.get(rank, 0),
            sha256=digest,
        )
    return outputs, removal_starts, affected_documents


def find_span_duplicates(
    signature_root: Path,
    removal_root: Path,
    *,
    bucket: int,
    finder_workers: int | None = None,
    total_ranks: int | None = None,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
    sqlite_cache_mb: int = 128,
    chunk_records: int = 250_000,
    maximum_open_runs: int = 64,
    temporary_directory: Path | None = None,
) -> dict[str, Any]:
    """Resolve exact duplicates using bounded fixed-record external sorts.

    ``sqlite_cache_mb`` remains accepted so existing operators do not break, but
    SQLite is no longer used.  The winner is the greatest priority, followed by
    stable document tie key and source position.
    """

    _validate_policy(sentence_count, minimum_span_words)
    if sqlite_cache_mb < 1:
        raise ValueError("sqlite_cache_mb must be positive")
    signature_root = Path(signature_root)
    removal_root = Path(removal_root)
    inputs, resolved_workers, resolved_total = _rank_bucket_inputs(
        signature_root,
        schema=SIGNATURE_SCHEMA,
        suffix="sig",
        record_size=SIGNATURE_RECORD.size,
        bucket=bucket,
        finder_workers=finder_workers,
        sentence_count=sentence_count,
        minimum_span_words=minimum_span_words,
        total_ranks=total_ranks,
    )
    work_root = (
        Path(temporary_directory)
        if temporary_directory is not None
        else removal_root / ".finder-work"
    )
    work_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".finder-{bucket:04d}-", dir=work_root))
    unsorted_removals = stage / "removals.unsorted"
    duplicate_groups = 0
    try:
        with _external_sorted_records(
            inputs,
            record=SIGNATURE_RECORD,
            key=lambda row: (row[0], -int(row[4]), row[5], row[1], row[2], row[3]),
            work_root=stage,
            prefix="signature-sort",
            chunk_records=chunk_records,
            maximum_open_runs=maximum_open_runs,
        ) as (records, signature_sort):
            with unsorted_removals.open("wb") as output:
                current_digest: bytes | None = None
                keeper: tuple[int, int] | None = None
                group_has_removals = False
                for digest, rank, document, sentence_start, _priority_value, _tie in records:
                    digest = bytes(digest)
                    identity = (int(rank), int(document))
                    if digest != current_digest:
                        duplicate_groups += int(group_has_removals)
                        current_digest = digest
                        keeper = identity
                        group_has_removals = False
                        continue
                    if identity == keeper:
                        continue
                    group_has_removals = True
                    output.write(
                        UNSORTED_REMOVAL_RECORD.pack(
                            identity[0],
                            identity[1],
                            int(sentence_start),
                            digest,
                        )
                    )
                duplicate_groups += int(group_has_removals)

        with _external_sorted_records(
            [unsorted_removals],
            record=UNSORTED_REMOVAL_RECORD,
            key=lambda row: (row[0], row[1], row[2], row[3]),
            work_root=stage,
            prefix="removal-sort",
            chunk_records=chunk_records,
            maximum_open_runs=maximum_open_runs,
        ) as (records, removal_sort):
            publish_stage = stage / "publish"
            publish_stage.mkdir()
            outputs, removal_starts, affected_documents = _publish_removal_outputs(
                records,
                removal_root,
                publish_stage,
                bucket=bucket,
                total_ranks=resolved_total,
            )
        report: dict[str, Any] = {
            "schema": REMOVAL_SCHEMA,
            "bucket": bucket,
            "finder_workers": resolved_workers,
            "total_ranks": resolved_total,
            "sentence_count": sentence_count,
            "minimum_span_words": minimum_span_words,
            "signature_files": len(inputs),
            "signature_records": signature_sort["input_records"],
            "duplicate_groups": duplicate_groups,
            "removal_starts": removal_starts,
            "affected_documents": affected_documents,
            "removal_record_size": REMOVAL_RECORD.size,
            "signature_sort": signature_sort,
            "removal_sort": removal_sort,
            "outputs": outputs,
        }
        _atomic_json(removal_root / "_manifests" / f"{bucket:04d}.json", report)
        return report
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _iter_removal_file(
    source: _ManifestedFile,
    chunk_records: int = 8_192,
) -> Iterator[tuple[int, int, bytes]]:
    for document, sentence_start, digest in _iter_fixed_records(
        source, REMOVAL_RECORD, chunk_records
    ):
        yield int(document), int(sentence_start), bytes(digest)


def iter_span_removals(
    removal_root: Path,
    *,
    rank: int,
    finder_workers: int,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
) -> Iterator[tuple[int, int, bytes]]:
    """Merge sorted finder outputs for one rank with bounded memory."""

    _validate_policy(sentence_count, minimum_span_words)
    if finder_workers < 1:
        raise ValueError("finder_workers must be at least 1")
    removal_root = Path(removal_root)
    streams: list[Iterator[tuple[int, int, bytes]]] = []
    for bucket in range(finder_workers):
        manifest_path = removal_root / "_manifests" / f"{bucket:04d}.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Missing repeated-span finder manifest: {manifest_path}")
        manifest = _read_json(manifest_path)
        expected = {
            "schema": REMOVAL_SCHEMA,
            "bucket": bucket,
            "finder_workers": finder_workers,
            "sentence_count": sentence_count,
            "minimum_span_words": minimum_span_words,
            "removal_record_size": REMOVAL_RECORD.size,
        }
        _require_fields(manifest_path, manifest, expected)
        try:
            total_ranks = int(manifest["total_ranks"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid total_ranks in {manifest_path}") from exc
        if rank < 0 or rank >= total_ranks:
            raise ValueError(f"rank {rank} is outside total_ranks={total_ranks}")
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != {
            f"{value:06d}" for value in range(total_ranks)
        }:
            raise RuntimeError(f"Incomplete rank inventory in {manifest_path}")
        source = _validate_output(
            removal_root,
            manifest_path,
            outputs[f"{rank:06d}"],
            expected_key=rank,
            expected_relative_path=f"{bucket:04d}/{rank:06d}.remove",
            record_size=REMOVAL_RECORD.size,
        )
        if source is not None:
            streams.append(_iter_removal_file(source))

    previous: tuple[int, int, bytes] | None = None
    for removal in heapq.merge(*streams):
        if removal != previous:
            yield removal
            previous = removal


def strip_duplicate_spans(
    text: str,
    sentence_starts: Iterable[int],
    *,
    sentence_count: int = 3,
) -> SpanStripResult:
    """Remove sentence windows and merge overlaps without rewriting other text."""

    if sentence_count < 1:
        raise ValueError("sentence_count must be at least 1")
    sentences = sentence_spans(text)
    requested = sorted(set(int(value) for value in sentence_starts))
    intervals: list[tuple[int, int]] = []
    for start in requested:
        if start < 0 or start + sentence_count > len(sentences):
            raise RuntimeError(
                f"Repeated-span removal start {start} is invalid for a "
                f"{len(sentences)}-sentence document"
            )
        end = start + sentence_count
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))

    character_ranges: list[tuple[int, int]] = []
    for sentence_start, sentence_end in intervals:
        start = sentences[sentence_start].start
        end = sentences[sentence_end - 1].end
        while start > 0 and text[start - 1] in " \t":
            start -= 1
        while end < len(text) and text[end] in " \t":
            end += 1
        character_ranges.append((start, end))

    output: list[str] = []
    cursor = 0
    for start, end in character_ranges:
        output.append(text[cursor:start])
        left = output[-1][-1:] if output[-1] else ""
        right = text[end : end + 1]
        if left and right and not left.isspace() and not right.isspace():
            output.append(" ")
        cursor = end
    output.append(text[cursor:])
    filtered = "".join(output).strip()
    remaining = sentence_spans(filtered)
    return SpanStripResult(
        text=filtered,
        requested_spans=len(requested),
        removed_sentences=sum(end - start for start, end in intervals),
        remaining_words=len(WORD_RE.findall(_canonical_sentence(filtered))),
        remaining_sentences=len(remaining),
    )


def build_span_dedup_filter(
    removal_root: Path,
    *,
    finder_workers: int,
    sentence_count: int = 3,
    minimum_span_words: int = 24,
    minimum_remaining_words: int = 50,
    minimum_remaining_sentences: int = 3,
    quarantine_writer: Any = None,
) -> Any:
    """Build a DataTrove step that strips repeated spans and audits mutations."""

    from datatrove.pipeline.base import PipelineStep

    _validate_policy(sentence_count, minimum_span_words)
    if minimum_remaining_words < 0 or minimum_remaining_sentences < 0:
        raise ValueError("remaining document minimums must be non-negative")

    class RepeatedSpanFilter(PipelineStep):
        name = "Metis exact repeated sentence-span/template deduplication"
        type = "SPAN-DEDUP"

        def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Iterator[Any]:
            removals = iter_span_removals(
                removal_root,
                rank=rank,
                finder_workers=finder_workers,
                sentence_count=sentence_count,
                minimum_span_words=minimum_span_words,
            )
            pending = next(removals, None)
            with quarantine_writer if quarantine_writer else contextlib.nullcontext() as writer:
                for document_index, document in enumerate(data):
                    if pending is not None and pending[0] < document_index:
                        raise RuntimeError(
                            f"Repeated-span removal references missing document {pending[0]} "
                            f"before current document {document_index} on rank {rank}"
                        )
                    starts: list[int] = []
                    expected_digests: list[bytes] = []
                    while pending is not None and pending[0] == document_index:
                        starts.append(pending[1])
                        expected_digests.append(pending[2])
                        pending = next(removals, None)
                    if not starts:
                        yield document
                        continue

                    original_text = str(document.text)
                    original_sentences = sentence_spans(original_text)
                    for start, expected_digest in zip(starts, expected_digests, strict=True):
                        actual_digest = span_digest(original_sentences, start, sentence_count)
                        if actual_digest != expected_digest:
                            raise RuntimeError(
                                f"Repeated-span removal digest mismatch for rank {rank}, "
                                f"document {document_index}, sentence start {start}; "
                                "the filter input no longer matches the signed corpus"
                            )
                    result = strip_duplicate_spans(
                        original_text,
                        starts,
                        sentence_count=sentence_count,
                    )
                    dropped = (
                        result.remaining_words < minimum_remaining_words
                        or result.remaining_sentences < minimum_remaining_sentences
                    )
                    reason = (
                        "repeated_span_below_minimum"
                        if dropped
                        else "repeated_span_content_removed"
                    )
                    audit = {
                        "filter_reason": reason,
                        "span_dedup_action": "dropped" if dropped else "modified",
                        "span_dedup_requested_spans": result.requested_spans,
                        "span_dedup_removed_sentences": result.removed_sentences,
                        "span_dedup_original_sha256": hashlib.sha256(
                            original_text.encode("utf-8")
                        ).hexdigest(),
                    }
                    if writer:
                        quarantined = copy.copy(document)
                        quarantined.metadata = {**document.metadata, **audit}
                        writer.write(quarantined, rank)
                    if dropped:
                        continue
                    document.text = result.text
                    document.metadata.update(audit)
                    document.metadata["span_dedup_remaining_words"] = result.remaining_words
                    document.metadata["span_dedup_remaining_sentences"] = result.remaining_sentences
                    yield document

                if pending is not None:
                    raise RuntimeError(
                        f"Repeated-span removal references document {pending[0]} beyond "
                        f"the end of rank {rank}"
                    )

    return RepeatedSpanFilter()
