from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

from .dedup import canonical_text
from .external_sort import external_sort_records, iter_fixed_records
from .state import atomic_json


SIGNATURE_SCHEMA = "metis.sha256-signatures/v2"
REMOVAL_SCHEMA = "metis.sha256-removals/v2"
SIGNATURE_RECORD = struct.Struct("<32sIIq32s")
REMOVAL_RECORD = struct.Struct("<I")
UINT32_MAX = (1 << 32) - 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(text: str) -> bytes:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).digest()


class _FilePool:
    def __init__(self, root: Path, maximum_open: int = 32) -> None:
        self.root = root
        self.maximum_open = maximum_open
        self.handles: OrderedDict[int, BinaryIO] = OrderedDict()
        self.counts: dict[int, int] = {}

    def write(self, bucket: int, payload: bytes) -> None:
        handle = self.handles.pop(bucket, None)
        if handle is None:
            if len(self.handles) >= self.maximum_open:
                _, oldest = self.handles.popitem(last=False)
                oldest.close()
            handle = (self.root / f"{bucket:04d}.sig").open("ab")
        self.handles[bucket] = handle
        handle.write(payload)
        self.counts[bucket] = self.counts.get(bucket, 0) + 1

    def close(self) -> None:
        while self.handles:
            _, handle = self.handles.popitem(last=False)
            handle.close()


def write_final_signatures(
    documents: Iterable[Any],
    output_root: Path,
    *,
    rank: int,
    finder_workers: int,
) -> dict[str, Any]:
    """Stream full SHA-256 signatures and publish a completeness manifest."""

    if finder_workers < 1 or rank < 0 or rank > UINT32_MAX:
        raise ValueError("finder_workers must be positive and rank must fit uint32")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".sha256-rank-{rank:06d}-", dir=output_root))
    # Sized to the bucket count, not left at the default. Buckets are assigned
    # by `digest % finder_workers`, so a pool smaller than the bucket count
    # evicts and reopens a file on roughly every write, and each of those is a
    # Lustre metadata round trip. The same 64-bucket/32-handle mismatch in the
    # span writer held that stage at 10-32% CPU for 11.5 hours.
    pool = _FilePool(stage, maximum_open=max(32, finder_workers))
    documents_seen = 0
    try:
        for document_index, document in enumerate(documents):
            if document_index > UINT32_MAX:
                raise OverflowError(f"rank {rank} contains too many documents")
            digest = content_sha256(str(document.text))
            priority = int(document.metadata.get("priority", 1))
            tie = hashlib.sha256(str(document.id).encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "little") % finder_workers
            pool.write(
                bucket,
                SIGNATURE_RECORD.pack(digest, rank, document_index, -priority, tie),
            )
            documents_seen += 1
        pool.close()

        bucket_manifest: dict[str, dict[str, Any]] = {}
        for bucket in range(finder_workers):
            destination = output_root / f"{bucket:04d}" / f"{rank:06d}.sig"
            source = stage / f"{bucket:04d}.sig"
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                bucket_manifest[str(bucket)] = {
                    "records": int(pool.counts[bucket]),
                    "size": destination.stat().st_size,
                    "sha256": _sha256_file(destination),
                }
            else:
                destination.unlink(missing_ok=True)
        report: dict[str, Any] = {
            "schema": SIGNATURE_SCHEMA,
            "rank": rank,
            "finder_workers": finder_workers,
            "record_size": SIGNATURE_RECORD.size,
            "documents": documents_seen,
            "buckets": bucket_manifest,
        }
        atomic_json(output_root / "_manifests" / f"{rank:06d}.json", report)
        return report
    finally:
        pool.close()
        shutil.rmtree(stage, ignore_errors=True)


def _signature_manifests(
    signature_root: Path, finder_workers: int | None, expected_ranks: int | None
) -> list[dict[str, Any]]:
    manifest_root = signature_root / "_manifests"
    paths = sorted(manifest_root.glob("*.json")) if manifest_root.exists() else []
    if expected_ranks is None:
        expected_ranks = len(paths)
    expected_names = {f"{rank:06d}.json" for rank in range(expected_ranks)}
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"SHA-256 signature manifests incomplete: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    manifests: list[dict[str, Any]] = []
    for rank, path in enumerate(sorted(paths)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != SIGNATURE_SCHEMA
            or int(payload.get("rank", -1)) != rank
            or int(payload.get("record_size", -1)) != SIGNATURE_RECORD.size
        ):
            raise RuntimeError(f"Invalid SHA-256 signature manifest: {path}")
        workers = int(payload.get("finder_workers", -1))
        if finder_workers is not None and workers != finder_workers:
            raise RuntimeError(f"SHA-256 signature worker mismatch in {path}")
        for raw_bucket, record in payload.get("buckets", {}).items():
            bucket = int(raw_bucket)
            candidate = signature_root / f"{bucket:04d}" / f"{rank:06d}.sig"
            if (
                not candidate.is_file()
                or candidate.stat().st_size != int(record.get("size", -1))
                or candidate.stat().st_size != int(record.get("records", -1)) * SIGNATURE_RECORD.size
                or _sha256_file(candidate) != record.get("sha256")
            ):
                raise RuntimeError(f"SHA-256 signature partition missing or corrupt: {candidate}")
        manifests.append(payload)
    return manifests


def find_final_duplicates(
    signature_root: Path,
    removal_root: Path,
    *,
    bucket: int,
    finder_workers: int | None = None,
    expected_ranks: int | None = None,
    temporary_directory: Path | None = None,
    chunk_records: int = 250_000,
) -> dict[str, Any]:
    """Externally sort one SHA-256 bucket and emit every rank, including empty ones."""

    signature_root = Path(signature_root)
    removal_root = Path(removal_root)
    manifests = _signature_manifests(signature_root, finder_workers, expected_ranks)
    expected_ranks = len(manifests)
    if bucket < 0 or (finder_workers is not None and bucket >= finder_workers):
        raise ValueError("SHA-256 finder bucket is outside the configured range")
    paths = [
        signature_root / f"{bucket:04d}" / f"{rank:06d}.sig"
        for rank, manifest in enumerate(manifests)
        if str(bucket) in manifest.get("buckets", {})
    ]
    scratch = Path(temporary_directory or (removal_root / ".sort-work"))
    by_rank: dict[int, list[int]] = {}
    groups = 0
    removed = 0
    current_digest: bytes | None = None
    first = True
    seen_nodes: set[tuple[int, int]] = set()
    for digest, rank, document, _priority_sort, _tie in external_sort_records(
        paths,
        record=SIGNATURE_RECORD,
        key=lambda row: (row[0], row[3], row[4], row[1], row[2]),
        temporary_directory=scratch,
        chunk_records=chunk_records,
    ):
        digest = bytes(digest)
        node = (int(rank), int(document))
        if digest != current_digest:
            current_digest = digest
            first = True
            seen_nodes.clear()
        if node in seen_nodes:
            continue
        seen_nodes.add(node)
        if first:
            first = False
            continue
        if len(seen_nodes) == 2:
            groups += 1
        by_rank.setdefault(node[0], []).append(node[1])
        removed += 1

    destination = removal_root / f"{bucket:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".sha256-remove-{bucket:04d}-", dir=destination))
    outputs: dict[str, dict[str, Any]] = {}
    try:
        for rank in range(expected_ranks):
            values = sorted(set(by_rank.get(rank, [])))
            path = stage / f"{rank:06d}.remove"
            with path.open("wb") as handle:
                for document in values:
                    handle.write(REMOVAL_RECORD.pack(document))
            outputs[str(rank)] = {
                "records": len(values),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        for stale in destination.glob("*.remove"):
            stale.unlink()
        for path in stage.glob("*.remove"):
            path.replace(destination / path.name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    report: dict[str, Any] = {
        "schema": REMOVAL_SCHEMA,
        "bucket": bucket,
        "finder_workers": finder_workers if finder_workers is not None else -1,
        "expected_ranks": expected_ranks,
        "duplicate_groups": groups,
        "removed": removed,
        "outputs": outputs,
    }
    atomic_json(removal_root / "_manifests" / f"{bucket:04d}.json", report)
    return report


def load_final_removals(
    removal_root: Path, *, rank: int, finder_workers: int
) -> set[int]:
    removed: set[int] = set()
    removal_root = Path(removal_root)
    for bucket in range(finder_workers):
        manifest_path = removal_root / "_manifests" / f"{bucket:04d}.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing SHA-256 removal manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != REMOVAL_SCHEMA or int(manifest.get("bucket", -1)) != bucket:
            raise RuntimeError(f"Invalid SHA-256 removal manifest: {manifest_path}")
        record = manifest.get("outputs", {}).get(str(rank))
        if not isinstance(record, dict):
            raise RuntimeError(f"SHA-256 removal manifest omits rank {rank}: {manifest_path}")
        path = removal_root / f"{bucket:04d}" / f"{rank:06d}.remove"
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("size", -1))
            or path.stat().st_size != int(record.get("records", -1)) * REMOVAL_RECORD.size
            or _sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"SHA-256 removal output missing or corrupt: {path}")
        for (document,) in iter_fixed_records(path, REMOVAL_RECORD):
            removed.add(int(document))
    return removed


def build_sha256_filter(
    removal_root: Path,
    *,
    finder_workers: int,
    reason: str,
    annotate_hash: bool,
    exclusion_writer: Any = None,
) -> Any:
    from datatrove.pipeline.base import PipelineStep

    class Sha256Filter(PipelineStep):
        name = f"Metis full SHA-256 exact deduplication ({reason})"
        type = "SHA256-DEDUP"

        def run(self, data: Iterable[Any], rank: int = 0, world_size: int = 1) -> Iterator[Any]:
            removed = load_final_removals(
                removal_root, rank=rank, finder_workers=finder_workers
            )
            with exclusion_writer if exclusion_writer else contextlib.nullcontext() as writer:
                for document_index, document in enumerate(data):
                    if document_index in removed:
                        document.metadata["filter_reason"] = reason
                        if writer:
                            writer.write(document, rank)
                    else:
                        if annotate_hash:
                            document.metadata["final_content_sha256"] = content_sha256(
                                str(document.text)
                            ).hex()
                        yield document

    return Sha256Filter()


def build_final_hash_filter(
    removal_root: Path, *, finder_workers: int, exclusion_writer: Any = None
) -> Any:
    return build_sha256_filter(
        removal_root,
        finder_workers=finder_workers,
        reason="final_sha256_duplicate",
        annotate_hash=True,
        exclusion_writer=exclusion_writer,
    )
