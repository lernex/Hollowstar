from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

from .config import load_yaml, repository_root
from .decontaminate import (
    ordered_ngram_hashes,
    required_matches,
    ContaminationIndex,
    benchmark_genealogy_match,
    code_ngram_hashes,
    code_skeleton_ngram_hashes,
    looks_like_code,
    ngram_hashes,
)
from .dedup import canonical_text


POSTING_DTYPE = np.dtype([("hash", "<u8"), ("group", "V16")])
MINHASH_PAIR_STRUCT = struct.Struct("<4I")
MINHASH_MEMBER_STRUCT = struct.Struct("<IQ")
MINHASH_CANDIDATE_STRUCT = struct.Struct("<QQq32s")
MINHASH_REMOVAL_STRUCT = struct.Struct("<I")

PRIORITY_CLUSTER_SCHEMA = "metis.priority-minhash-clusters/v1"
PRIORITY_CANDIDATE_SCHEMA = "metis.priority-minhash-rank-candidates/v1"
PRIORITY_RESOLVER_SCHEMA = "metis.priority-minhash-bucket-resolution/v1"
PRIORITY_FINALIZE_SCHEMA = "metis.priority-minhash-rank-removals/v1"
PRIORITY_COMPLETE_SCHEMA = "metis.priority-minhash-complete/v1"
MINHASH_BUCKET_OUTPUT_SCHEMA = "metis.minhash-bucket-output/v1"
MINHASH_PAIR_BYTES = MINHASH_PAIR_STRUCT.size
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CONTAMINATION_INDEX_SCHEMA = "metis.contamination-index/v5"
HOLDOUT_BUNDLE_SCHEMA = "metis.holdout-bundle/v4"
CONTAMINATION_ARRAY_NAMES = (
    "exact",
    "ngram_postings",
    "short_ngram_postings",
    "code_ngram_postings",
    "code_skeleton_ngram_postings",
)
CONTAMINATION_POLICY_FIELDS = (
    "ngram_size",
    "minimum_matching_ngrams",
    "short_ngram_size",
    "minimum_short_matching_ngrams",
    "code_ngram_size",
    "minimum_code_matching_ngrams",
    "code_skeleton_ngram_size",
    "minimum_code_skeleton_matching_ngrams",
    "maximum_shingle_rows",
    # Detection tuning must round-trip. The disk index defaults these to 0 when
    # absent, so omitting them here would let decontam_index build with the
    # configured values and decontam_filter silently run without them.
    "match_fraction",
    "contiguous_run_minimum",
)


def build_regex_word_tokenizer() -> Any:
    """Return a dependency-light tokenizer for multilingual/code MinHash shingles."""
    from datatrove.utils.word_tokenizers import WordTokenizer

    class MetisRegexWordTokenizer(WordTokenizer):
        _word = re.compile(r"\w+|[^\w\s]", re.UNICODE)
        _sentence = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)", re.UNICODE)

        def word_tokenize(self, text: str) -> list[str]:
            return self._word.findall(text)

        def sent_tokenize(self, text: str) -> list[str]:
            return [match.group(0).strip() for match in self._sentence.finditer(text) if match.group(0).strip()]

        def span_tokenize(self, text: str) -> list[tuple[int, int]]:
            return [(match.start(), match.end()) for match in self._sentence.finditer(text)]

    return MetisRegexWordTokenizer()


def _priority_json_digest(value: Any) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "manifest_sha256"}
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_priority_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {**payload, "status": "complete"}
    complete["manifest_sha256"] = _priority_json_digest(complete)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(complete, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return complete


def _load_priority_manifest(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Completeness manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid completeness manifest: {path}") from exc
    if (
        payload.get("schema") != schema
        or payload.get("status") != "complete"
        or payload.get("manifest_sha256") != _priority_json_digest(payload)
    ):
        raise RuntimeError(f"Completeness manifest failed validation: {path}")
    return payload


def _stream_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _priority_artifact(
    path: Path,
    *,
    base: Path,
    records: int,
) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(base.resolve())),
        "size": path.stat().st_size,
        "sha256": _stream_file_sha256(path),
        "records": int(records),
    }


def _priority_artifact_path(base: Path, record: dict[str, Any]) -> Path:
    path = (base / str(record["path"])).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Priority MinHash artifact escaped its work directory: {path}") from exc
    return path


def _validate_priority_artifact(base: Path, record: dict[str, Any]) -> Path:
    path = _priority_artifact_path(base, record)
    if not path.is_file():
        raise RuntimeError(f"Priority MinHash artifact is missing: {path}")
    expected_size = int(record["size"])
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"Priority MinHash artifact size changed: {path}")
    if _stream_file_sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"Priority MinHash artifact hash changed: {path}")
    return path


def _replace_priority_directory(temporary: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def _encode_minhash_node(rank: int, document_id: int, total_tasks: int) -> int:
    if not (0 <= rank < total_tasks <= 0x80000000):
        raise RuntimeError(
            f"MinHash rank {rank} is outside the configured task range 0..{total_tasks - 1}"
        )
    if not (0 <= document_id <= 0xFFFFFFFF):
        raise RuntimeError(f"MinHash document index does not fit uint32: {document_id}")
    return (rank << 32) | document_id


def _decode_minhash_node(node: int) -> tuple[int, int]:
    return node >> 32, node & 0xFFFFFFFF


def _component_bucket(component: int, bucket_count: int) -> int:
    # SplitMix64 makes buckets independent of rank/doc-id bit layout.
    value = (component + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return int(value % bucket_count)


def _iter_fixed_records(
    path: Path,
    record: struct.Struct,
    *,
    records_per_chunk: int = 65_536,
) -> Iterator[tuple[Any, ...]]:
    chunk_bytes = max(1, records_per_chunk) * record.size
    with path.open("rb") as handle:
        while payload := handle.read(chunk_bytes):
            if len(payload) % record.size:
                raise RuntimeError(f"Corrupt fixed-record MinHash artifact: {path}")
            yield from struct.iter_unpack(record.format, payload)


def write_minhash_bucket_output_manifest(
    duplicate_folder: str | Path,
    inventory_folder: str | Path,
    *,
    bucket: int,
    expected_buckets: int,
) -> dict[str, Any]:
    """Seal one DataTrove MinHash bucket output, including an empty output.

    The production graph currently launches exactly one DataTrove worker for
    each MinHash band bucket. DataTrove names that worker's output
    ``{bucket:05d}_00.dups`` and creates the file even when it contains zero
    pairs. Requiring that explicit empty file distinguishes a genuinely empty
    bucket from a missing/failed task.
    """

    if expected_buckets <= 0:
        raise ValueError("expected_buckets must be positive")
    if not (0 <= bucket < expected_buckets):
        raise ValueError(f"bucket must be in 0..{expected_buckets - 1}")
    duplicates = Path(duplicate_folder).resolve()
    inventory = Path(inventory_folder).resolve()
    if not duplicates.is_dir():
        raise RuntimeError(f"MinHash duplicate output directory is missing: {duplicates}")
    expected_name = f"{bucket:05d}_00.dups"
    expected_path = duplicates / expected_name
    matching = sorted(
        path
        for path in duplicates.glob(f"{bucket:05d}_*.dups")
        if path.is_file()
    )
    if matching != [expected_path]:
        observed = [path.name for path in matching]
        raise RuntimeError(
            f"MinHash bucket {bucket} output inventory is invalid; "
            f"expected [{expected_name!r}], found {observed!r}"
        )
    size = expected_path.stat().st_size
    if size % MINHASH_PAIR_BYTES:
        raise RuntimeError(
            f"Corrupt MinHash duplicate output {expected_path}: {size} bytes is not "
            f"divisible by {MINHASH_PAIR_BYTES}"
        )
    payload = {
        "schema": MINHASH_BUCKET_OUTPUT_SCHEMA,
        "duplicate_folder": str(duplicates),
        "bucket": bucket,
        "expected_buckets": expected_buckets,
        "output": {
            "path": expected_name,
            "present": True,
            "empty": size == 0,
            "size": size,
            "records": size // MINHASH_PAIR_BYTES,
            "sha256": _stream_file_sha256(expected_path),
        },
    }
    return _write_priority_manifest(
        inventory / f"bucket-{bucket:06d}.json",
        payload,
    )


def verified_minhash_bucket_inventory(
    duplicate_folder: str | Path,
    inventory_folder: str | Path,
    *,
    expected_buckets: int,
) -> tuple[list[Path], list[dict[str, Any]], str]:
    """Validate the exact configured DataTrove bucket set and its file hashes."""

    if expected_buckets <= 0:
        raise ValueError("expected_buckets must be positive")
    duplicates = Path(duplicate_folder).resolve()
    inventory = Path(inventory_folder).resolve()
    expected_manifest_names = {
        f"bucket-{bucket:06d}.json" for bucket in range(expected_buckets)
    }
    actual_manifest_names = {
        path.name for path in inventory.glob("bucket-*.json") if path.is_file()
    }
    if actual_manifest_names != expected_manifest_names:
        raise RuntimeError(
            "MinHash bucket manifest inventory is incomplete; "
            f"missing={sorted(expected_manifest_names - actual_manifest_names)[:16]}, "
            f"unexpected={sorted(actual_manifest_names - expected_manifest_names)[:16]}"
        )

    expected_output_names = {
        f"{bucket:05d}_00.dups" for bucket in range(expected_buckets)
    }
    actual_output_names = {
        str(path.relative_to(duplicates))
        for path in duplicates.rglob("*.dups")
        if path.is_file()
    }
    if actual_output_names != expected_output_names:
        raise RuntimeError(
            "MinHash duplicate file inventory does not match the configured bucket set; "
            f"missing={sorted(expected_output_names - actual_output_names)[:16]}, "
            f"unexpected={sorted(actual_output_names - expected_output_names)[:16]}"
        )

    paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for bucket in range(expected_buckets):
        marker = inventory / f"bucket-{bucket:06d}.json"
        manifest = _load_priority_manifest(marker, MINHASH_BUCKET_OUTPUT_SCHEMA)
        if (
            manifest.get("duplicate_folder") != str(duplicates)
            or int(manifest.get("bucket", -1)) != bucket
            or int(manifest.get("expected_buckets", -1)) != expected_buckets
        ):
            raise RuntimeError(f"MinHash bucket manifest is bound to another run: {marker}")
        output = manifest.get("output")
        if not isinstance(output, dict):
            raise RuntimeError(f"MinHash bucket manifest has no output record: {marker}")
        expected_name = f"{bucket:05d}_00.dups"
        path = duplicates / expected_name
        size = path.stat().st_size
        if (
            output.get("path") != expected_name
            or output.get("present") is not True
            or bool(output.get("empty")) is not (size == 0)
            or int(output.get("size", -1)) != size
            or int(output.get("records", -1)) * MINHASH_PAIR_BYTES != size
            or str(output.get("sha256")) != _stream_file_sha256(path)
        ):
            raise RuntimeError(f"MinHash bucket output changed after sealing: {path}")
        paths.append(path)
        rows.append(
            {
                "bucket": bucket,
                "manifest_sha256": manifest["manifest_sha256"],
                "path": expected_name,
                "size": size,
                "records": int(output["records"]),
                "sha256": str(output["sha256"]),
            }
        )
    return paths, rows, _priority_json_digest(rows)


def _duplicate_inventory(duplicates: Path) -> tuple[list[Path], list[dict[str, Any]], str]:
    files = sorted(path for path in duplicates.glob("**/*.dups") if path.is_file())
    inventory = [
        {
            "path": str(path.resolve().relative_to(duplicates.resolve())),
            "size": path.stat().st_size,
            "sha256": _stream_file_sha256(path),
        }
        for path in files
    ]
    return files, inventory, _priority_json_digest(inventory)


def _configure_priority_sqlite(path: Path, cache_mb: int) -> sqlite3.Connection:
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA cache_size={-max(8, int(cache_mb)) * 1024}")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    return connection


def _priority_sqlite_path(
    *,
    prefix: str,
    default_directory: Path,
    temporary_directory: str | Path | None,
) -> Path:
    directory = (
        Path(temporary_directory).expanduser().resolve()
        if temporary_directory is not None
        else default_directory
    )
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".sqlite3",
        dir=directory,
    )
    os.close(descriptor)
    return Path(name)


def _remove_priority_sqlite(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _union_find_root(
    connection: sqlite3.Connection,
    node: int,
    *,
    create: bool,
) -> int:
    row = connection.execute("SELECT parent FROM nodes WHERE node=?", (node,)).fetchone()
    if row is None:
        if not create:
            raise RuntimeError(f"Disk union-find lost node {node}")
        connection.execute(
            "INSERT INTO nodes(node,parent,size) VALUES(?,?,1)",
            (node, node),
        )
        return node
    current = node
    root = int(row[0])
    path: list[int] = []
    while root != current:
        path.append(current)
        current = root
        next_row = connection.execute(
            "SELECT parent FROM nodes WHERE node=?",
            (current,),
        ).fetchone()
        if next_row is None:
            raise RuntimeError(f"Disk union-find has a dangling parent {current}")
        root = int(next_row[0])
    if path:
        connection.executemany(
            "UPDATE nodes SET parent=? WHERE node=?",
            ((root, member) for member in path),
        )
    return root


def _union_find_merge(connection: sqlite3.Connection, left: int, right: int) -> None:
    left_root = _union_find_root(connection, left, create=True)
    right_root = _union_find_root(connection, right, create=True)
    if left_root == right_root:
        return
    left_size_row = connection.execute(
        "SELECT size FROM nodes WHERE node=?",
        (left_root,),
    ).fetchone()
    right_size_row = connection.execute(
        "SELECT size FROM nodes WHERE node=?",
        (right_root,),
    ).fetchone()
    if left_size_row is None or right_size_row is None:
        raise RuntimeError("Disk union-find root metadata is incomplete")
    left_size = int(left_size_row[0])
    right_size = int(right_size_row[0])
    if left_size < right_size or (
        left_size == right_size and right_root < left_root
    ):
        left_root, right_root = right_root, left_root
        left_size, right_size = right_size, left_size
    connection.execute(
        "UPDATE nodes SET parent=? WHERE node=?",
        (left_root, right_root),
    )
    connection.execute(
        "UPDATE nodes SET size=? WHERE node=?",
        (left_size + right_size, left_root),
    )


def _validate_cluster_manifest_artifacts(
    work: Path,
    manifest: dict[str, Any],
) -> None:
    observed_members = 0
    total_tasks = int(manifest.get("total_tasks", 0))
    member_files = manifest.get("member_files")
    if not isinstance(member_files, dict):
        raise RuntimeError("Cluster manifest member_files must be an object")
    expected_paths: set[str] = set()
    for raw_rank, record in member_files.items():
        rank = int(raw_rank)
        if not (0 <= rank < total_tasks):
            raise RuntimeError(f"Cluster member index has out-of-range rank {rank}")
        expected_path = f"clusters/members/{rank:06d}.members"
        if record.get("path") != expected_path:
            raise RuntimeError(f"Cluster member index has a noncanonical path for rank {rank}")
        _validate_priority_artifact(work, record)
        records = int(record["records"])
        if int(record["size"]) != records * MINHASH_MEMBER_STRUCT.size:
            raise RuntimeError("Cluster member index has an invalid fixed-record size")
        observed_members += records
        expected_paths.add(expected_path)
    members_root = work / "clusters" / "members"
    actual_paths = {
        str(path.resolve().relative_to(work.resolve()))
        for path in members_root.glob("*.members")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise RuntimeError("Cluster member file inventory does not match CLUSTERS.json")
    if observed_members != int(manifest["component_members"]):
        raise RuntimeError("Cluster member indexes are incomplete")


def cluster_priority_minhash_pairs(
    duplicate_folder: str | Path,
    work_folder: str | Path,
    *,
    total_tasks: int,
    bucket_count: int = 256,
    sqlite_cache_mb: int = 256,
    transaction_rows: int = 100_000,
    bucket_inventory_folder: str | Path | None = None,
    expected_duplicate_buckets: int | None = None,
    temporary_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Stream `.dups` edges into disk union-find and emit rank member indexes."""

    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    duplicates = Path(duplicate_folder).resolve()
    work = Path(work_folder).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (bucket_inventory_folder is None) is not (expected_duplicate_buckets is None):
        raise ValueError(
            "bucket_inventory_folder and expected_duplicate_buckets must be provided together"
        )
    if bucket_inventory_folder is not None:
        pair_files, inventory, inventory_sha256 = verified_minhash_bucket_inventory(
            duplicates,
            bucket_inventory_folder,
            expected_buckets=int(expected_duplicate_buckets),
        )
        input_contract = {
            "mode": "sealed_datatrove_buckets",
            "bucket_manifest_folder": str(Path(bucket_inventory_folder).resolve()),
            "expected_duplicate_buckets": int(expected_duplicate_buckets),
        }
    else:
        pair_files, inventory, inventory_sha256 = _duplicate_inventory(duplicates)
        input_contract = {"mode": "legacy_unsealed_inventory"}
    cluster_dir = work / "clusters"
    manifest_path = cluster_dir / "CLUSTERS.json"
    if manifest_path.is_file():
        try:
            existing = _load_priority_manifest(manifest_path, PRIORITY_CLUSTER_SCHEMA)
            if (
                existing.get("input_inventory_sha256") == inventory_sha256
                and int(existing.get("total_tasks", -1)) == total_tasks
                and int(existing.get("bucket_count", -1)) == bucket_count
                and existing.get("input_contract") == input_contract
            ):
                _validate_cluster_manifest_artifacts(work, existing)
                return existing
        except RuntimeError:
            pass

    temporary = Path(
        tempfile.mkdtemp(prefix=".clusters-", dir=work)
    )
    database = _priority_sqlite_path(
        prefix=".minhash-union-find-",
        default_directory=temporary,
        temporary_directory=temporary_directory,
    )
    members_dir = temporary / "members"
    members_dir.mkdir()
    try:
        connection = _configure_priority_sqlite(database, sqlite_cache_mb)
    except BaseException:
        _remove_priority_sqlite(database)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    duplicate_pairs = 0
    batch_size = max(1, int(transaction_rows))
    try:
        connection.execute(
            "CREATE TABLE nodes("
            "node INTEGER PRIMARY KEY,"
            "parent INTEGER NOT NULL,"
            "size INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        connection.execute("BEGIN IMMEDIATE")
        for pair_path in pair_files:
            for rank_a, doc_a, rank_b, doc_b in _iter_fixed_records(
                pair_path,
                MINHASH_PAIR_STRUCT,
            ):
                left = _encode_minhash_node(int(rank_a), int(doc_a), total_tasks)
                right = _encode_minhash_node(int(rank_b), int(doc_b), total_tasks)
                _union_find_merge(connection, left, right)
                duplicate_pairs += 1
                if duplicate_pairs % batch_size == 0:
                    connection.execute("COMMIT")
                    connection.execute("BEGIN IMMEDIATE")
        connection.execute("COMMIT")

        last_node = -1
        while True:
            rows = connection.execute(
                "SELECT node FROM nodes WHERE node>? ORDER BY node LIMIT ?",
                (last_node, batch_size),
            ).fetchall()
            if not rows:
                break
            connection.execute("BEGIN IMMEDIATE")
            for (raw_node,) in rows:
                _union_find_root(connection, int(raw_node), create=False)
            connection.execute("COMMIT")
            last_node = int(rows[-1][0])

        component_members = int(
            connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        )
        components = int(
            connection.execute(
                "SELECT COUNT(*) FROM nodes WHERE node=parent"
            ).fetchone()[0]
        )

        member_files: dict[str, dict[str, Any]] = {}
        cursor = connection.execute("SELECT node,parent FROM nodes ORDER BY node")
        active_rank: int | None = None
        active_handle: BinaryIO | None = None
        active_path: Path | None = None
        active_count = 0
        active_hash = hashlib.sha256()

        def close_member_file() -> None:
            nonlocal active_handle, active_path, active_count, active_hash
            if active_handle is None or active_path is None or active_rank is None:
                return
            active_handle.flush()
            os.fsync(active_handle.fileno())
            active_handle.close()
            member_files[str(active_rank)] = {
                "path": f"clusters/members/{active_path.name}",
                "size": active_path.stat().st_size,
                "sha256": active_hash.hexdigest(),
                "records": active_count,
            }
            active_handle = None
            active_path = None
            active_count = 0
            active_hash = hashlib.sha256()

        for rows in iter(lambda: cursor.fetchmany(batch_size), []):
            for raw_node, raw_root in rows:
                node = int(raw_node)
                root = int(raw_root)
                rank, document_id = _decode_minhash_node(node)
                if rank >= total_tasks:
                    raise RuntimeError(f"Union-find contains out-of-range rank {rank}")
                if active_rank != rank:
                    close_member_file()
                    active_rank = rank
                    active_path = members_dir / f"{rank:06d}.members"
                    active_handle = active_path.open("wb")
                payload = MINHASH_MEMBER_STRUCT.pack(document_id, root)
                active_handle.write(payload)
                active_hash.update(payload)
                active_count += 1
        close_member_file()
        connection.execute("PRAGMA optimize")
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        connection.close()
        _remove_priority_sqlite(database)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    connection.close()
    # The immutable, hashed rank-member indexes are the consumed cluster
    # artifact. The much larger SQLite working database has served its purpose
    # and is removed to avoid retaining a second corpus-sized representation.
    _remove_priority_sqlite(database)

    payload = {
        "schema": PRIORITY_CLUSTER_SCHEMA,
        "total_tasks": total_tasks,
        "bucket_count": bucket_count,
        "duplicate_pairs": duplicate_pairs,
        "component_members": component_members,
        "components": components,
        "input_inventory": inventory,
        "input_inventory_sha256": inventory_sha256,
        "input_contract": input_contract,
        "member_files": member_files,
        "union_find": {
            "backend": "sqlite",
            "disk_backed": True,
            "retained_after_member_index": False,
        },
    }
    _write_priority_manifest(temporary / "CLUSTERS.json", payload)
    _replace_priority_directory(temporary, cluster_dir)
    return _load_priority_manifest(manifest_path, PRIORITY_CLUSTER_SCHEMA)


class _BoundedCandidateWriters:
    def __init__(self, folder: Path, max_open_files: int) -> None:
        self.folder = folder
        self.max_open_files = max(1, int(max_open_files))
        self.handles: OrderedDict[int, BinaryIO] = OrderedDict()
        self.paths: dict[int, Path] = {}

    def write(self, bucket: int, payload: bytes) -> None:
        handle = self.handles.pop(bucket, None)
        if handle is None:
            path = self.paths.setdefault(
                bucket,
                self.folder / f"bucket-{bucket:06d}.candidates",
            )
            handle = path.open("ab")
        self.handles[bucket] = handle
        handle.write(payload)
        if len(self.handles) > self.max_open_files:
            _, evicted = self.handles.popitem(last=False)
            evicted.close()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def write_priority_minhash_rank_candidates(
    document_folder: str | Path,
    work_folder: str | Path,
    *,
    rank: int,
    total_tasks: int,
    max_open_files: int = 32,
) -> dict[str, Any]:
    """Write one rank's priority candidates, partitioned by component bucket."""

    from datatrove.pipeline.readers import JsonlReader

    if not (0 <= rank < total_tasks):
        raise ValueError(f"rank must be in 0..{total_tasks - 1}")
    work = Path(work_folder).resolve()
    cluster = _load_priority_manifest(
        work / "clusters" / "CLUSTERS.json",
        PRIORITY_CLUSTER_SCHEMA,
    )
    if int(cluster["total_tasks"]) != total_tasks:
        raise RuntimeError("Candidate writer task count does not match cluster manifest")
    cluster_hash = str(cluster["manifest_sha256"])
    documents = Path(document_folder).resolve()
    destination = work / "candidates" / f"rank-{rank:06d}"
    marker = destination / "CANDIDATES.json"
    if marker.is_file():
        try:
            existing = _load_priority_manifest(marker, PRIORITY_CANDIDATE_SCHEMA)
            if (
                existing.get("cluster_manifest_sha256") == cluster_hash
                and int(existing.get("rank", -1)) == rank
                and int(existing.get("total_tasks", -1)) == total_tasks
                and existing.get("document_folder") == str(documents)
            ):
                observed = 0
                for artifact in existing.get("shards", []):
                    _validate_priority_artifact(work, artifact)
                    records = int(artifact["records"])
                    if int(artifact["size"]) != records * MINHASH_CANDIDATE_STRUCT.size:
                        raise RuntimeError("Candidate shard has an invalid fixed-record size")
                    observed += records
                if (
                    observed != int(existing["candidate_count"])
                    or observed != int(existing["member_count"])
                ):
                    raise RuntimeError("Rank candidate completeness counts do not reconcile")
                return existing
        except RuntimeError:
            pass

    member_record = cluster.get("member_files", {}).get(str(rank))
    member_count = int(member_record["records"]) if member_record else 0
    member_path: Path | None = None
    if member_record:
        member_path = _validate_priority_artifact(work, member_record)

    candidates_parent = work / "candidates"
    candidates_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".rank-{rank:06d}-", dir=candidates_parent)
    )
    writers = _BoundedCandidateWriters(temporary, max_open_files)
    bucket_counts: dict[int, int] = {}
    candidate_count = 0
    try:
        if member_path is not None:
            members = iter(_iter_fixed_records(member_path, MINHASH_MEMBER_STRUCT))
            current = next(members, None)
            reader = JsonlReader(str(documents), shuffle_files=False)
            for document_index, document in enumerate(
                reader.run(rank=rank, world_size=total_tasks)
            ):
                if current is None:
                    break
                member_document_id, component = map(int, current)
                if member_document_id < document_index:
                    raise RuntimeError(
                        f"MinHash member {rank}:{member_document_id} has no corresponding document"
                    )
                if member_document_id != document_index:
                    continue
                priority = int(document.metadata.get("priority", 1))
                if not (-(1 << 63) <= priority < (1 << 63)):
                    raise RuntimeError(
                        f"Document priority does not fit signed int64 at {rank}:{document_index}"
                    )
                node = _encode_minhash_node(rank, document_index, total_tasks)
                tie_break = hashlib.sha256(
                    str(document.id).encode("utf-8")
                ).digest()
                bucket = _component_bucket(component, int(cluster["bucket_count"]))
                writers.write(
                    bucket,
                    MINHASH_CANDIDATE_STRUCT.pack(
                        component,
                        node,
                        priority,
                        tie_break,
                    ),
                )
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
                candidate_count += 1
                current = next(members, None)
            if current is not None:
                member_document_id = int(current[0])
                raise RuntimeError(
                    f"MinHash member {rank}:{member_document_id} has no corresponding document"
                )
        writers.close()
        if candidate_count != member_count:
            raise RuntimeError(
                f"Rank {rank} wrote {candidate_count} candidates for {member_count} component members"
            )
        shards = [
            {
                **_priority_artifact(
                    path,
                    base=work,
                    records=bucket_counts[bucket],
                ),
                "bucket": bucket,
            }
            for bucket, path in sorted(writers.paths.items())
        ]
        payload = {
            "schema": PRIORITY_CANDIDATE_SCHEMA,
            "cluster_manifest_sha256": cluster_hash,
            "rank": rank,
            "total_tasks": total_tasks,
            "document_folder": str(documents),
            "member_count": member_count,
            "candidate_count": candidate_count,
            "member_index_sha256": (
                str(member_record["sha256"]) if member_record else None
            ),
            "shards": shards,
        }
        # Paths above refer to the final rank directory, not the temporary name.
        for shard in payload["shards"]:
            shard["path"] = (
                f"candidates/rank-{rank:06d}/"
                f"bucket-{int(shard['bucket']):06d}.candidates"
            )
        _write_priority_manifest(temporary / "CANDIDATES.json", payload)
        _replace_priority_directory(temporary, destination)
    except BaseException:
        writers.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _load_priority_manifest(marker, PRIORITY_CANDIDATE_SCHEMA)


def _candidate_inputs_for_bucket(
    work: Path,
    cluster: dict[str, Any],
    *,
    bucket: int,
    total_tasks: int,
) -> tuple[list[tuple[int, Path, dict[str, Any]]], str]:
    inputs: list[tuple[int, Path, dict[str, Any]]] = []
    fingerprint: list[dict[str, Any]] = []
    for rank in range(total_tasks):
        marker = work / "candidates" / f"rank-{rank:06d}" / "CANDIDATES.json"
        candidate_manifest = _load_priority_manifest(
            marker,
            PRIORITY_CANDIDATE_SCHEMA,
        )
        if (
            candidate_manifest.get("cluster_manifest_sha256")
            != cluster["manifest_sha256"]
            or int(candidate_manifest.get("rank", -1)) != rank
            or int(candidate_manifest.get("total_tasks", -1)) != total_tasks
        ):
            raise RuntimeError(f"Rank {rank} candidate manifest is bound to another cluster")
        if (
            sum(int(shard["records"]) for shard in candidate_manifest.get("shards", []))
            != int(candidate_manifest["candidate_count"])
            or int(candidate_manifest["candidate_count"])
            != int(candidate_manifest["member_count"])
        ):
            raise RuntimeError(f"Rank {rank} candidate completeness counts do not reconcile")
        selected = [
            shard
            for shard in candidate_manifest.get("shards", [])
            if int(shard["bucket"]) == bucket
        ]
        if len(selected) > 1:
            raise RuntimeError(f"Rank {rank} has duplicate candidate shards for bucket {bucket}")
        shard_record = selected[0] if selected else None
        fingerprint.append(
            {
                "rank": rank,
                "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
                "shard_sha256": shard_record["sha256"] if shard_record else None,
                "records": int(shard_record["records"]) if shard_record else 0,
            }
        )
        if shard_record:
            if (
                int(shard_record["size"])
                != int(shard_record["records"]) * MINHASH_CANDIDATE_STRUCT.size
            ):
                raise RuntimeError(f"Rank {rank} candidate shard has an invalid size")
            inputs.append(
                (
                    rank,
                    _validate_priority_artifact(work, shard_record),
                    shard_record,
                )
            )
    return inputs, _priority_json_digest(fingerprint)


def resolve_priority_minhash_bucket(
    work_folder: str | Path,
    *,
    bucket: int,
    total_tasks: int,
    sqlite_cache_mb: int = 256,
    transaction_rows: int = 100_000,
    temporary_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one component bucket with disk-backed winner/member tables."""

    work = Path(work_folder).resolve()
    cluster = _load_priority_manifest(
        work / "clusters" / "CLUSTERS.json",
        PRIORITY_CLUSTER_SCHEMA,
    )
    bucket_count = int(cluster["bucket_count"])
    if not (0 <= bucket < bucket_count):
        raise ValueError(f"bucket must be in 0..{bucket_count - 1}")
    if int(cluster["total_tasks"]) != total_tasks:
        raise RuntimeError("Bucket resolver task count does not match cluster manifest")
    inputs, candidate_set_sha256 = _candidate_inputs_for_bucket(
        work,
        cluster,
        bucket=bucket,
        total_tasks=total_tasks,
    )
    destination = work / "resolved" / f"bucket-{bucket:06d}"
    marker = destination / "RESOLVED.json"
    if marker.is_file():
        try:
            existing = _load_priority_manifest(marker, PRIORITY_RESOLVER_SCHEMA)
            if (
                existing.get("cluster_manifest_sha256")
                == cluster["manifest_sha256"]
                and existing.get("candidate_set_sha256") == candidate_set_sha256
                and int(existing.get("bucket", -1)) == bucket
            ):
                observed_removals = 0
                for artifact in existing.get("removal_fragments", []):
                    _validate_priority_artifact(work, artifact)
                    records = int(artifact["records"])
                    if int(artifact["size"]) != records * MINHASH_REMOVAL_STRUCT.size:
                        raise RuntimeError("Removal fragment has an invalid fixed-record size")
                    observed_removals += records
                if (
                    observed_removals != int(existing["removed"])
                    or int(existing["removed"])
                    != int(existing["candidate_count"]) - int(existing["components"])
                ):
                    raise RuntimeError("Bucket resolution completeness counts do not reconcile")
                return existing
        except RuntimeError:
            pass

    resolved_parent = work / "resolved"
    resolved_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".bucket-{bucket:06d}-", dir=resolved_parent)
    )
    database = _priority_sqlite_path(
        prefix=f".minhash-resolver-{bucket:06d}-",
        default_directory=temporary,
        temporary_directory=temporary_directory,
    )
    try:
        connection = _configure_priority_sqlite(database, sqlite_cache_mb)
    except BaseException:
        _remove_priority_sqlite(database)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    batch_size = max(1, int(transaction_rows))
    candidate_count = 0
    try:
        connection.execute(
            "CREATE TABLE members("
            "node INTEGER PRIMARY KEY,"
            "component INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE winners("
            "component INTEGER PRIMARY KEY,"
            "priority INTEGER NOT NULL,"
            "tie_break BLOB NOT NULL,"
            "node INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        connection.execute("BEGIN IMMEDIATE")
        for rank, candidate_path, shard_record in inputs:
            observed = 0
            previous_document_id = -1
            for component_raw, node_raw, priority_raw, tie_break in _iter_fixed_records(
                candidate_path,
                MINHASH_CANDIDATE_STRUCT,
            ):
                component = int(component_raw)
                node = int(node_raw)
                priority = int(priority_raw)
                node_rank, document_id = _decode_minhash_node(node)
                if node_rank != rank or document_id <= previous_document_id:
                    raise RuntimeError(
                        f"Candidate shard ordering/ownership is invalid: {candidate_path}"
                    )
                if _component_bucket(component, bucket_count) != bucket:
                    raise RuntimeError(
                        f"Candidate component was written to the wrong bucket: {component}"
                    )
                previous_document_id = document_id
                try:
                    connection.execute(
                        "INSERT INTO members(node,component) VALUES(?,?)",
                        (node, component),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError(f"Duplicate MinHash candidate node {node}") from exc
                connection.execute(
                    "INSERT INTO winners(component,priority,tie_break,node) "
                    "VALUES(?,?,?,?) "
                    "ON CONFLICT(component) DO UPDATE SET "
                    "priority=excluded.priority,"
                    "tie_break=excluded.tie_break,"
                    "node=excluded.node "
                    "WHERE excluded.priority>winners.priority "
                    "OR (excluded.priority=winners.priority "
                    "AND excluded.tie_break>winners.tie_break) "
                    "OR (excluded.priority=winners.priority "
                    "AND excluded.tie_break=winners.tie_break "
                    "AND excluded.node<winners.node)",
                    (component, priority, sqlite3.Binary(tie_break), node),
                )
                candidate_count += 1
                observed += 1
                if candidate_count % batch_size == 0:
                    connection.execute("COMMIT")
                    connection.execute("BEGIN IMMEDIATE")
            if observed != int(shard_record["records"]):
                raise RuntimeError(
                    f"Candidate shard record count changed: {candidate_path}"
                )
        connection.execute("COMMIT")
        components = int(
            connection.execute("SELECT COUNT(*) FROM winners").fetchone()[0]
        )
        if candidate_count < components:
            raise RuntimeError("Winner table contains more components than candidates")

        removal_fragments: list[dict[str, Any]] = []
        query = connection.execute(
            "SELECT members.node "
            "FROM members JOIN winners USING(component) "
            "WHERE members.node<>winners.node "
            "ORDER BY members.node"
        )
        active_rank: int | None = None
        active_path: Path | None = None
        active_handle: BinaryIO | None = None
        active_hash = hashlib.sha256()
        active_count = 0

        def close_fragment() -> None:
            nonlocal active_path, active_handle, active_hash, active_count
            if active_handle is None or active_path is None or active_rank is None:
                return
            active_handle.flush()
            os.fsync(active_handle.fileno())
            active_handle.close()
            removal_fragments.append(
                {
                    "path": (
                        f"resolved/bucket-{bucket:06d}/"
                        f"rank-{active_rank:06d}.remove.part"
                    ),
                    "size": active_path.stat().st_size,
                    "sha256": active_hash.hexdigest(),
                    "records": active_count,
                    "rank": active_rank,
                }
            )
            active_path = None
            active_handle = None
            active_hash = hashlib.sha256()
            active_count = 0

        for rows in iter(lambda: query.fetchmany(batch_size), []):
            for (raw_node,) in rows:
                rank, document_id = _decode_minhash_node(int(raw_node))
                if active_rank != rank:
                    close_fragment()
                    active_rank = rank
                    active_path = temporary / f"rank-{rank:06d}.remove.part"
                    active_handle = active_path.open("wb")
                packed = MINHASH_REMOVAL_STRUCT.pack(document_id)
                active_handle.write(packed)
                active_hash.update(packed)
                active_count += 1
        close_fragment()
        removed = sum(int(record["records"]) for record in removal_fragments)
        if removed != candidate_count - components:
            raise RuntimeError(
                "Bucket removal count does not equal candidates minus components"
            )
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        connection.close()
        _remove_priority_sqlite(database)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    connection.close()
    _remove_priority_sqlite(database)
    payload = {
        "schema": PRIORITY_RESOLVER_SCHEMA,
        "cluster_manifest_sha256": cluster["manifest_sha256"],
        "candidate_set_sha256": candidate_set_sha256,
        "bucket": bucket,
        "bucket_count": bucket_count,
        "total_tasks": total_tasks,
        "candidate_count": candidate_count,
        "components": components,
        "removed": removed,
        "removal_fragments": removal_fragments,
    }
    _write_priority_manifest(temporary / "RESOLVED.json", payload)
    _replace_priority_directory(temporary, destination)
    return _load_priority_manifest(marker, PRIORITY_RESOLVER_SCHEMA)


def _resolved_manifests(
    work: Path,
    cluster: dict[str, Any],
    *,
    total_tasks: int,
) -> tuple[list[dict[str, Any]], str]:
    manifests: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    for bucket in range(int(cluster["bucket_count"])):
        marker = (
            work
            / "resolved"
            / f"bucket-{bucket:06d}"
            / "RESOLVED.json"
        )
        payload = _load_priority_manifest(marker, PRIORITY_RESOLVER_SCHEMA)
        if (
            payload.get("cluster_manifest_sha256") != cluster["manifest_sha256"]
            or int(payload.get("bucket", -1)) != bucket
            or int(payload.get("total_tasks", -1)) != total_tasks
        ):
            raise RuntimeError(f"Bucket {bucket} resolution belongs to another cluster")
        if (
            sum(int(row["records"]) for row in payload.get("removal_fragments", []))
            != int(payload["removed"])
            or int(payload["removed"])
            != int(payload["candidate_count"]) - int(payload["components"])
        ):
            raise RuntimeError(f"Bucket {bucket} resolution counts do not reconcile")
        manifests.append(payload)
        fingerprints.append(
            {
                "bucket": bucket,
                "manifest_sha256": payload["manifest_sha256"],
            }
        )
    return manifests, _priority_json_digest(fingerprints)


def finalize_priority_minhash_rank_removals(
    work_folder: str | Path,
    output_folder: str | Path,
    *,
    rank: int,
    total_tasks: int,
    sqlite_cache_mb: int = 128,
    transaction_rows: int = 100_000,
    temporary_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Merge one rank's bucket fragments into a sorted DataTrove `.remove` file."""

    if not (0 <= rank < total_tasks):
        raise ValueError(f"rank must be in 0..{total_tasks - 1}")
    work = Path(work_folder).resolve()
    output = Path(output_folder).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cluster = _load_priority_manifest(
        work / "clusters" / "CLUSTERS.json",
        PRIORITY_CLUSTER_SCHEMA,
    )
    if int(cluster["total_tasks"]) != total_tasks:
        raise RuntimeError("Removal finalizer task count does not match cluster manifest")
    resolved, resolver_set_sha256 = _resolved_manifests(
        work,
        cluster,
        total_tasks=total_tasks,
    )
    fragments: list[dict[str, Any]] = []
    for manifest in resolved:
        selected = [
            record
            for record in manifest.get("removal_fragments", [])
            if int(record["rank"]) == rank
        ]
        if len(selected) > 1:
            raise RuntimeError(
                f"Bucket {manifest['bucket']} has duplicate removal fragments for rank {rank}"
            )
        if selected:
            _validate_priority_artifact(work, selected[0])
            fragments.append(selected[0])

    finalized = work / "finalized"
    finalized.mkdir(parents=True, exist_ok=True)
    marker = finalized / f"rank-{rank:06d}.json"
    output_path = output / f"{rank:06d}.remove"
    if marker.is_file():
        try:
            existing = _load_priority_manifest(marker, PRIORITY_FINALIZE_SCHEMA)
            output_record = existing.get("output")
            output_valid = (
                output_record is None
                and not output_path.exists()
                and int(existing["removal_count"]) == 0
                or output_record is not None
                and output_path.is_file()
                and output_path.stat().st_size == int(output_record["size"])
                and int(output_record["size"])
                == int(output_record["records"]) * MINHASH_REMOVAL_STRUCT.size
                and int(output_record["records"])
                == int(existing["removal_count"])
                and _stream_file_sha256(output_path)
                == str(output_record["sha256"])
            )
            if (
                existing.get("cluster_manifest_sha256")
                == cluster["manifest_sha256"]
                and existing.get("resolver_set_sha256") == resolver_set_sha256
                and existing.get("output_folder") == str(output)
                and output_valid
            ):
                return existing
        except RuntimeError:
            pass

    database = _priority_sqlite_path(
        prefix=f".rank-{rank:06d}-",
        default_directory=finalized,
        temporary_directory=temporary_directory,
    )
    try:
        connection = _configure_priority_sqlite(database, sqlite_cache_mb)
    except BaseException:
        _remove_priority_sqlite(database)
        raise
    batch_size = max(1, int(transaction_rows))
    inserted = 0
    try:
        connection.execute(
            "CREATE TABLE removals(doc_id INTEGER PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute("BEGIN IMMEDIATE")
        for fragment in fragments:
            fragment_path = _priority_artifact_path(work, fragment)
            previous = -1
            observed = 0
            for (raw_document_id,) in _iter_fixed_records(
                fragment_path,
                MINHASH_REMOVAL_STRUCT,
            ):
                document_id = int(raw_document_id)
                if document_id <= previous:
                    raise RuntimeError(
                        f"Removal fragment is not strictly sorted: {fragment_path}"
                    )
                previous = document_id
                try:
                    connection.execute(
                        "INSERT INTO removals(doc_id) VALUES(?)",
                        (document_id,),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RuntimeError(
                        f"Document {rank}:{document_id} appeared in multiple component buckets"
                    ) from exc
                inserted += 1
                observed += 1
                if inserted % batch_size == 0:
                    connection.execute("COMMIT")
                    connection.execute("BEGIN IMMEDIATE")
            if observed != int(fragment["records"]):
                raise RuntimeError(
                    f"Removal fragment record count changed: {fragment_path}"
                )
        connection.execute("COMMIT")
        removal_count = int(
            connection.execute("SELECT COUNT(*) FROM removals").fetchone()[0]
        )
        if removal_count != inserted:
            raise RuntimeError("Removal finalizer lost or duplicated records")

        output_record: dict[str, Any] | None = None
        if removal_count:
            temporary_output = output_path.with_name(
                f".{output_path.name}.{os.getpid()}.tmp"
            )
            digest = hashlib.sha256()
            with temporary_output.open("wb") as handle:
                cursor = connection.execute(
                    "SELECT doc_id FROM removals ORDER BY doc_id"
                )
                for rows in iter(lambda: cursor.fetchmany(batch_size), []):
                    for (document_id,) in rows:
                        packed = MINHASH_REMOVAL_STRUCT.pack(int(document_id))
                        handle.write(packed)
                        digest.update(packed)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_output, output_path)
            output_record = {
                "path": output_path.name,
                "size": output_path.stat().st_size,
                "sha256": digest.hexdigest(),
                "records": removal_count,
            }
        else:
            output_path.unlink(missing_ok=True)
    except BaseException:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        connection.close()
        _remove_priority_sqlite(database)
        raise
    connection.close()
    _remove_priority_sqlite(database)
    payload = {
        "schema": PRIORITY_FINALIZE_SCHEMA,
        "cluster_manifest_sha256": cluster["manifest_sha256"],
        "resolver_set_sha256": resolver_set_sha256,
        "rank": rank,
        "total_tasks": total_tasks,
        "output_folder": str(output),
        "fragment_count": len(fragments),
        "removal_count": removal_count,
        "output": output_record,
    }
    return _write_priority_manifest(marker, payload)


def _priority_manifest_inventory(
    folder: Path,
    pattern: str,
    expected_names: set[str],
    *,
    label: str,
) -> None:
    actual = {path.name for path in folder.glob(pattern) if path.is_file()}
    if actual != expected_names:
        raise RuntimeError(
            f"Priority MinHash {label} inventory is incomplete; "
            f"missing={sorted(expected_names - actual)[:16]}, "
            f"unexpected={sorted(actual - expected_names)[:16]}"
        )


def _validate_priority_candidate_manifest(
    work: Path,
    cluster: dict[str, Any],
    *,
    rank: int,
    total_tasks: int,
) -> dict[str, Any]:
    marker = work / "candidates" / f"rank-{rank:06d}" / "CANDIDATES.json"
    payload = _load_priority_manifest(marker, PRIORITY_CANDIDATE_SCHEMA)
    if (
        payload.get("cluster_manifest_sha256") != cluster["manifest_sha256"]
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("total_tasks", -1)) != total_tasks
    ):
        raise RuntimeError(f"Rank {rank} candidate manifest belongs to another run")
    observed = 0
    seen_buckets: set[int] = set()
    for artifact in payload.get("shards", []):
        bucket = int(artifact.get("bucket", -1))
        if bucket in seen_buckets or not (0 <= bucket < int(cluster["bucket_count"])):
            raise RuntimeError(f"Rank {rank} candidate shard bucket inventory is invalid")
        seen_buckets.add(bucket)
        if (
            artifact.get("path")
            != f"candidates/rank-{rank:06d}/bucket-{bucket:06d}.candidates"
        ):
            raise RuntimeError(f"Rank {rank} candidate shard has a noncanonical path")
        path = _validate_priority_artifact(work, artifact)
        records = int(artifact["records"])
        if (
            path.stat().st_size != records * MINHASH_CANDIDATE_STRUCT.size
            or int(artifact["size"]) != records * MINHASH_CANDIDATE_STRUCT.size
        ):
            raise RuntimeError(f"Rank {rank} candidate shard has an invalid fixed-record size")
        observed += records
    if (
        observed != int(payload.get("candidate_count", -1))
        or observed != int(payload.get("member_count", -1))
    ):
        raise RuntimeError(f"Rank {rank} candidate completeness counts do not reconcile")
    return payload


def _validate_priority_resolver_manifest(
    work: Path,
    cluster: dict[str, Any],
    *,
    bucket: int,
    total_tasks: int,
    candidate_set_sha256: str,
) -> dict[str, Any]:
    marker = work / "resolved" / f"bucket-{bucket:06d}" / "RESOLVED.json"
    payload = _load_priority_manifest(marker, PRIORITY_RESOLVER_SCHEMA)
    if (
        payload.get("cluster_manifest_sha256") != cluster["manifest_sha256"]
        or int(payload.get("bucket", -1)) != bucket
        or int(payload.get("bucket_count", -1)) != int(cluster["bucket_count"])
        or int(payload.get("total_tasks", -1)) != total_tasks
        or payload.get("candidate_set_sha256") != candidate_set_sha256
    ):
        raise RuntimeError(f"Bucket {bucket} resolver manifest belongs to another run")
    observed = 0
    seen_ranks: set[int] = set()
    for artifact in payload.get("removal_fragments", []):
        rank = int(artifact.get("rank", -1))
        if rank in seen_ranks or not (0 <= rank < total_tasks):
            raise RuntimeError(f"Bucket {bucket} removal fragment rank inventory is invalid")
        seen_ranks.add(rank)
        if (
            artifact.get("path")
            != f"resolved/bucket-{bucket:06d}/rank-{rank:06d}.remove.part"
        ):
            raise RuntimeError(
                f"Bucket {bucket} removal fragment has a noncanonical path"
            )
        path = _validate_priority_artifact(work, artifact)
        records = int(artifact["records"])
        if (
            path.stat().st_size != records * MINHASH_REMOVAL_STRUCT.size
            or int(artifact["size"]) != records * MINHASH_REMOVAL_STRUCT.size
        ):
            raise RuntimeError(f"Bucket {bucket} removal fragment has an invalid size")
        observed += records
    if (
        observed != int(payload.get("removed", -1))
        or int(payload.get("removed", -1))
        != int(payload.get("candidate_count", -1)) - int(payload.get("components", -1))
    ):
        raise RuntimeError(f"Bucket {bucket} resolver counts do not reconcile")
    return payload


def _validate_priority_finalizer_manifest(
    work: Path,
    output: Path,
    cluster: dict[str, Any],
    *,
    rank: int,
    total_tasks: int,
    resolver_set_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker = work / "finalized" / f"rank-{rank:06d}.json"
    payload = _load_priority_manifest(marker, PRIORITY_FINALIZE_SCHEMA)
    if (
        payload.get("cluster_manifest_sha256") != cluster["manifest_sha256"]
        or payload.get("resolver_set_sha256") != resolver_set_sha256
        or int(payload.get("rank", -1)) != rank
        or int(payload.get("total_tasks", -1)) != total_tasks
        or payload.get("output_folder") != str(output)
    ):
        raise RuntimeError(f"Rank {rank} finalizer manifest belongs to another run")
    removal_count = int(payload.get("removal_count", -1))
    if removal_count < 0:
        raise RuntimeError(f"Rank {rank} finalizer has an invalid removal count")
    output_path = output / f"{rank:06d}.remove"
    output_record = payload.get("output")
    if removal_count == 0:
        if output_record is not None or output_path.exists():
            raise RuntimeError(
                f"Rank {rank} declares zero removals but has a removal artifact"
            )
        inventory = {
            "rank": rank,
            "present": False,
            "path": output_path.name,
            "records": 0,
            "size": 0,
            "sha256": EMPTY_SHA256,
            "finalizer_manifest_sha256": payload["manifest_sha256"],
        }
        return payload, inventory
    if not isinstance(output_record, dict):
        raise RuntimeError(f"Rank {rank} finalizer omits its removal artifact")
    if output_record.get("path") != output_path.name or not output_path.is_file():
        raise RuntimeError(f"Rank {rank} removal artifact is missing: {output_path}")
    expected_size = removal_count * MINHASH_REMOVAL_STRUCT.size
    if (
        int(output_record.get("records", -1)) != removal_count
        or int(output_record.get("size", -1)) != expected_size
        or output_path.stat().st_size != expected_size
        or _stream_file_sha256(output_path) != str(output_record.get("sha256"))
    ):
        raise RuntimeError(f"Rank {rank} removal artifact changed after finalization")
    inventory = {
        "rank": rank,
        "present": True,
        "path": output_path.name,
        "records": removal_count,
        "size": expected_size,
        "sha256": str(output_record["sha256"]),
        "finalizer_manifest_sha256": payload["manifest_sha256"],
    }
    return payload, inventory


def verify_priority_minhash_completion(
    work_folder: str | Path,
    output_folder: str | Path,
    *,
    total_tasks: int,
) -> dict[str, Any]:
    """Reconcile the partitioned priority MinHash pipeline and seal COMPLETE.json.

    This singleton reducer is intentionally metadata-heavy and corpus-light. It
    streams hashes for the already-finalized removal artifacts, validates every
    rank/bucket manifest, and proves that exactly one document survives each
    connected component before filters are allowed to consume the removals.
    """

    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")
    work = Path(work_folder).resolve()
    output = Path(output_folder).resolve()
    cluster = _load_priority_manifest(
        work / "clusters" / "CLUSTERS.json",
        PRIORITY_CLUSTER_SCHEMA,
    )
    if int(cluster.get("total_tasks", -1)) != total_tasks:
        raise RuntimeError("Completion verifier task count does not match cluster manifest")
    bucket_count = int(cluster.get("bucket_count", 0))
    if bucket_count <= 0:
        raise RuntimeError("Cluster manifest has an invalid component bucket count")

    candidate_marker_paths = {
        str(path.relative_to(work / "candidates"))
        for path in (work / "candidates").glob("rank-*/CANDIDATES.json")
        if path.is_file()
    }
    expected_candidate_markers = {
        f"rank-{rank:06d}/CANDIDATES.json" for rank in range(total_tasks)
    }
    if candidate_marker_paths != expected_candidate_markers:
        raise RuntimeError(
            "Priority MinHash candidate marker inventory is incomplete; "
            f"missing={sorted(expected_candidate_markers - candidate_marker_paths)[:16]}, "
            f"unexpected={sorted(candidate_marker_paths - expected_candidate_markers)[:16]}"
        )
    candidates = [
        _validate_priority_candidate_manifest(
            work,
            cluster,
            rank=rank,
            total_tasks=total_tasks,
        )
        for rank in range(total_tasks)
    ]
    candidate_bucket_fingerprints: list[str] = []
    for bucket in range(bucket_count):
        fingerprint: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates):
            selected = [
                shard
                for shard in candidate.get("shards", [])
                if int(shard["bucket"]) == bucket
            ]
            if len(selected) > 1:
                raise RuntimeError(
                    f"Rank {rank} has duplicate candidate shards for bucket {bucket}"
                )
            shard = selected[0] if selected else None
            fingerprint.append(
                {
                    "rank": rank,
                    "candidate_manifest_sha256": candidate["manifest_sha256"],
                    "shard_sha256": shard["sha256"] if shard else None,
                    "records": int(shard["records"]) if shard else 0,
                }
            )
        candidate_bucket_fingerprints.append(_priority_json_digest(fingerprint))

    resolver_marker_paths = {
        str(path.relative_to(work / "resolved"))
        for path in (work / "resolved").glob("bucket-*/RESOLVED.json")
        if path.is_file()
    }
    expected_resolver_markers = {
        f"bucket-{bucket:06d}/RESOLVED.json" for bucket in range(bucket_count)
    }
    if resolver_marker_paths != expected_resolver_markers:
        raise RuntimeError(
            "Priority MinHash resolver marker inventory is incomplete; "
            f"missing={sorted(expected_resolver_markers - resolver_marker_paths)[:16]}, "
            f"unexpected={sorted(resolver_marker_paths - expected_resolver_markers)[:16]}"
        )
    resolvers = [
        _validate_priority_resolver_manifest(
            work,
            cluster,
            bucket=bucket,
            total_tasks=total_tasks,
            candidate_set_sha256=candidate_bucket_fingerprints[bucket],
        )
        for bucket in range(bucket_count)
    ]
    resolver_set_sha256 = _priority_json_digest(
        [
            {"bucket": bucket, "manifest_sha256": manifest["manifest_sha256"]}
            for bucket, manifest in enumerate(resolvers)
        ]
    )

    _priority_manifest_inventory(
        work / "finalized",
        "rank-*.json",
        {f"rank-{rank:06d}.json" for rank in range(total_tasks)},
        label="finalizer marker",
    )
    finalizers: list[dict[str, Any]] = []
    removal_inventory: list[dict[str, Any]] = []
    for rank in range(total_tasks):
        finalizer, removal = _validate_priority_finalizer_manifest(
            work,
            output,
            cluster,
            rank=rank,
            total_tasks=total_tasks,
            resolver_set_sha256=resolver_set_sha256,
        )
        finalizers.append(finalizer)
        removal_inventory.append(removal)

    expected_removal_names = {
        row["path"] for row in removal_inventory if row["present"]
    }
    actual_removal_names = {
        path.name for path in output.glob("*.remove") if path.is_file()
    }
    if actual_removal_names != expected_removal_names:
        raise RuntimeError(
            "Priority MinHash final removal inventory is incomplete; "
            f"missing={sorted(expected_removal_names - actual_removal_names)[:16]}, "
            f"unexpected={sorted(actual_removal_names - expected_removal_names)[:16]}"
        )

    candidate_count = sum(int(row["candidate_count"]) for row in candidates)
    resolver_candidates = sum(int(row["candidate_count"]) for row in resolvers)
    components = sum(int(row["components"]) for row in resolvers)
    resolver_removed = sum(int(row["removed"]) for row in resolvers)
    finalized_removed = sum(int(row["removal_count"]) for row in finalizers)
    if (
        candidate_count != int(cluster["component_members"])
        or resolver_candidates != candidate_count
        or components != int(cluster["components"])
        or resolver_removed != candidate_count - components
        or finalized_removed != resolver_removed
    ):
        raise RuntimeError(
            "Priority MinHash global completeness counts do not reconcile: "
            f"cluster_members={cluster['component_members']}, candidates={candidate_count}, "
            f"resolver_candidates={resolver_candidates}, components={components}, "
            f"resolver_removed={resolver_removed}, finalized_removed={finalized_removed}"
        )

    payload = {
        "schema": PRIORITY_COMPLETE_SCHEMA,
        "cluster_manifest_sha256": cluster["manifest_sha256"],
        "total_tasks": total_tasks,
        "bucket_count": bucket_count,
        "output_folder": str(output),
        "candidate_set_sha256": _priority_json_digest(
            [
                {"rank": rank, "manifest_sha256": manifest["manifest_sha256"]}
                for rank, manifest in enumerate(candidates)
            ]
        ),
        "resolver_set_sha256": resolver_set_sha256,
        "finalizer_set_sha256": _priority_json_digest(
            [
                {"rank": rank, "manifest_sha256": manifest["manifest_sha256"]}
                for rank, manifest in enumerate(finalizers)
            ]
        ),
        "removal_inventory_sha256": _priority_json_digest(removal_inventory),
        "removal_files": removal_inventory,
        "duplicate_pairs": int(cluster["duplicate_pairs"]),
        "component_members": candidate_count,
        "components": components,
        "removed": finalized_removed,
    }
    return _write_priority_manifest(work / "COMPLETE.json", payload)


def require_verified_priority_minhash_rank(
    work_folder: str | Path,
    output_folder: str | Path,
    *,
    rank: int,
    total_tasks: int,
) -> dict[str, Any]:
    """Validate COMPLETE.json and the exact removal artifact consumed by one rank."""

    if not (0 <= rank < total_tasks):
        raise ValueError(f"rank must be in 0..{total_tasks - 1}")
    work = Path(work_folder).resolve()
    output = Path(output_folder).resolve()
    complete = _load_priority_manifest(
        work / "COMPLETE.json",
        PRIORITY_COMPLETE_SCHEMA,
    )
    if (
        int(complete.get("total_tasks", -1)) != total_tasks
        or complete.get("output_folder") != str(output)
    ):
        raise RuntimeError("Priority MinHash COMPLETE.json belongs to another run")
    removal_files = complete.get("removal_files")
    if (
        not isinstance(removal_files, list)
        or len(removal_files) != total_tasks
        or [int(row.get("rank", -1)) for row in removal_files]
        != list(range(total_tasks))
        or complete.get("removal_inventory_sha256")
        != _priority_json_digest(removal_files)
    ):
        raise RuntimeError("Priority MinHash COMPLETE.json has an invalid rank inventory")
    expected_finalizer_set_sha256 = _priority_json_digest(
        [
            {
                "rank": int(row["rank"]),
                "manifest_sha256": row.get("finalizer_manifest_sha256"),
            }
            for row in removal_files
        ]
    )
    if (
        complete.get("finalizer_set_sha256") != expected_finalizer_set_sha256
        or sum(int(row.get("records", -1)) for row in removal_files)
        != int(complete.get("removed", -1))
        or int(complete.get("component_members", -1))
        - int(complete.get("components", -1))
        != int(complete.get("removed", -1))
    ):
        raise RuntimeError("Priority MinHash COMPLETE.json global counts do not reconcile")
    cluster = _load_priority_manifest(
        work / "clusters" / "CLUSTERS.json",
        PRIORITY_CLUSTER_SCHEMA,
    )
    if complete.get("cluster_manifest_sha256") != cluster["manifest_sha256"]:
        raise RuntimeError("Priority MinHash cluster changed after global verification")
    finalizer, observed = _validate_priority_finalizer_manifest(
        work,
        output,
        cluster,
        rank=rank,
        total_tasks=total_tasks,
        resolver_set_sha256=str(complete["resolver_set_sha256"]),
    )
    expected = removal_files[rank]
    if (
        finalizer["manifest_sha256"] != expected.get("finalizer_manifest_sha256")
        or observed != expected
    ):
        raise RuntimeError(f"Priority MinHash rank {rank} changed after global verification")
    return {
        "rank": rank,
        "removal_count": int(observed["records"]),
        "removal_path": str(output / observed["path"]) if observed["present"] else None,
        "complete_manifest_sha256": complete["manifest_sha256"],
        "finalizer_manifest_sha256": finalizer["manifest_sha256"],
        "removal_sha256": observed["sha256"],
    }


def build_priority_minhash_removals(
    duplicate_folder: str | Path,
    output_folder: str | Path,
    document_folder: str | Path,
    *,
    total_tasks: int,
    work_folder: str | Path | None = None,
    bucket_count: int = 256,
    sqlite_cache_mb: int = 256,
    transaction_rows: int = 100_000,
    temporary_directory: str | Path | None = None,
) -> dict[str, int]:
    """Compatibility orchestrator for the partitionable disk-backed pipeline.

    Production schedulers should call the four public stage functions directly
    (one global cluster, one candidate task per rank, one resolver per bucket,
    then one finalizer per rank). This wrapper preserves the existing local
    call site and runs the same stages serially.
    """

    output = Path(output_folder).resolve()
    work = (
        Path(work_folder).resolve()
        if work_folder is not None
        else output / "_priority_minhash_work"
    )
    cluster = cluster_priority_minhash_pairs(
        duplicate_folder,
        work,
        total_tasks=total_tasks,
        bucket_count=bucket_count,
        sqlite_cache_mb=sqlite_cache_mb,
        transaction_rows=transaction_rows,
        temporary_directory=temporary_directory,
    )
    candidate_manifests = [
        write_priority_minhash_rank_candidates(
            document_folder,
            work,
            rank=rank,
            total_tasks=total_tasks,
        )
        for rank in range(total_tasks)
    ]
    resolved_manifests = [
        resolve_priority_minhash_bucket(
            work,
            bucket=bucket,
            total_tasks=total_tasks,
            sqlite_cache_mb=sqlite_cache_mb,
            transaction_rows=transaction_rows,
            temporary_directory=temporary_directory,
        )
        for bucket in range(bucket_count)
    ]
    finalized_manifests = [
        finalize_priority_minhash_rank_removals(
            work,
            output,
            rank=rank,
            total_tasks=total_tasks,
            sqlite_cache_mb=max(8, sqlite_cache_mb // 2),
            transaction_rows=transaction_rows,
            temporary_directory=temporary_directory,
        )
        for rank in range(total_tasks)
    ]
    # Keep the local variables above explicit: this compatibility path executes
    # the same public partitioned stages production submits independently.
    if not candidate_manifests or not resolved_manifests or not finalized_manifests:
        raise RuntimeError("Priority MinHash compatibility pipeline produced no stage manifests")
    summary = verify_priority_minhash_completion(
        work,
        output,
        total_tasks=total_tasks,
    )
    return {
        "duplicate_pairs": int(summary["duplicate_pairs"]),
        "component_members": int(summary["component_members"]),
        "components": int(summary["components"]),
        "removed": int(summary["removed"]),
    }


def _postings_array(postings: Any) -> np.ndarray:
    count = sum(len(groups) for groups in postings.values())
    output = np.empty(count, dtype=POSTING_DTYPE)
    cursor = 0
    for shingle, groups in sorted(postings.items()):
        for group in sorted(groups):
            output[cursor] = (np.uint64(shingle), np.void(group))
            cursor += 1
    return output


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contamination_manifest_sha256(payload: Mapping[str, Any]) -> str:
    return _canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _relative_artifact_record(path: Path, *, base: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Contamination artifact must be stored beside its index: {resolved}"
        ) from exc
    if path.is_symlink():
        raise RuntimeError(f"Contamination artifact may not be a symbolic link: {path}")
    return {
        "path": str(relative),
        "size": resolved.stat().st_size,
        "sha256": _stream_file_sha256(resolved),
    }


def _safe_relative_artifact(
    base: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"{label} has no valid relative path")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise RuntimeError(f"{label} path must be relative to the index: {raw_path}")
    unresolved = base.resolve() / relative
    if unresolved.is_symlink():
        raise RuntimeError(f"{label} may not be a symbolic link: {unresolved}")
    path = unresolved.resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} path escapes the contamination directory: {raw_path}") from exc
    if not path.is_file():
        raise RuntimeError(f"{label} is missing or is not a regular file: {path}")
    try:
        expected_size = int(record["size"])
        expected_sha256 = str(record["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has an invalid size/hash contract") from exc
    if expected_size < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError(f"{label} has an invalid size/hash contract")
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"{label} size changed after the index was built: {path}")
    if _stream_file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{label} hash changed after the index was built: {path}")
    return path


def _registry_relative_path(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(repository_root().resolve()))
    except ValueError:
        return None


def _find_registry_for_save(
    report: Mapping[str, Any],
    explicit_path: str | Path | None,
) -> Path:
    expected_sha256 = str(report.get("registry_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("Holdout report has no valid registry_sha256")
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser())
    else:
        candidates.append(
            repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml"
        )
        recorded = report.get("registry_path")
        if isinstance(recorded, str) and recorded:
            candidates.append(Path(recorded).expanduser())
    seen: set[Path] = set()
    mismatched: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            continue
        actual = _stream_file_sha256(resolved)
        if actual == expected_sha256:
            return resolved
        mismatched.append(str(resolved))
        if explicit_path is not None:
            break
    if mismatched:
        raise RuntimeError(
            "Pinned benchmark registry hash differs from the holdout report: "
            + ", ".join(mismatched)
        )
    raise RuntimeError(
        "Pinned benchmark registry used to build the holdouts is unavailable"
    )


def _policy_contract(value: Any) -> dict[str, int]:
    return {
        field: int(getattr(value, field))
        for field in CONTAMINATION_POLICY_FIELDS
    }


def _validate_policy_contract(
    policy: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for field in CONTAMINATION_POLICY_FIELDS:
        if field not in policy:
            raise RuntimeError(
                f"Pinned benchmark decontamination policy is missing {field}"
            )
        try:
            actual_value = int(policy[field])
            expected_value = int(expected[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Pinned benchmark decontamination policy has invalid {field}"
            ) from exc
        if actual_value != expected_value:
            raise RuntimeError(
                "Contamination index is stale for the pinned benchmark policy: "
                f"{field} is {actual_value}, index expects {expected_value}"
            )


def _validate_holdout_report(
    report: Mapping[str, Any],
    *,
    holdouts: Path,
    registry: Path,
) -> None:
    def valid_job(job: Any) -> bool:
        if not isinstance(job, Mapping):
            return False
        try:
            return (
                int(job.get("source_rows", 0)) > 0
                and int(job.get("fragments", 0)) > 0
            )
        except (TypeError, ValueError):
            return False

    if report.get("schema") != HOLDOUT_BUNDLE_SCHEMA:
        raise RuntimeError(
            f"Holdout report must use {HOLDOUT_BUNDLE_SCHEMA}"
        )
    if report.get("training_use") != "forbidden":
        raise RuntimeError("Holdout report must mark evaluation data as forbidden for training")
    if int(report.get("output_size", -1)) != holdouts.stat().st_size:
        raise RuntimeError("Holdout report output size does not match holdouts.jsonl")
    if report.get("output_sha256") != _stream_file_sha256(holdouts):
        raise RuntimeError("Holdout report output hash does not match holdouts.jsonl")
    if report.get("registry_sha256") != _stream_file_sha256(registry):
        raise RuntimeError("Holdout report registry hash does not match the pinned registry")
    jobs = report.get("jobs")
    try:
        job_count = int(report.get("job_count", -1))
        total_fragments = int(report.get("total_fragments", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Holdout report has invalid job/fragment counts") from exc
    if (
        job_count <= 0
        or total_fragments <= 0
        or not isinstance(jobs, list)
        or len(jobs) != job_count
        or any(not valid_job(job) for job in jobs)
    ):
        raise RuntimeError("Holdout report does not prove complete non-empty benchmark jobs")


def _contamination_inputs_for_save(
    output: Path,
    *,
    holdouts_path: str | Path | None,
    holdout_report_path: str | Path | None,
    benchmark_registry_path: str | Path | None,
    policy_contract: Mapping[str, Any],
) -> dict[str, Any]:
    holdouts = (
        Path(holdouts_path)
        if holdouts_path is not None
        else output.parent / "holdouts.jsonl"
    ).resolve()
    report_path = (
        Path(holdout_report_path)
        if holdout_report_path is not None
        else output.parent / "HOLDOUTS.json"
    ).resolve()
    report = _read_json_mapping(report_path, "Holdout report")
    registry_path = _find_registry_for_save(report, benchmark_registry_path)
    registry = load_yaml(registry_path)
    if registry.get("schema") != "metis.contamination-registry/v2":
        raise RuntimeError("Pinned benchmark registry has an unsupported schema")
    policy = registry.get("policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("Pinned benchmark registry has no policy mapping")
    benchmarks = registry.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise RuntimeError("Pinned benchmark registry has no benchmark inventory")
    if not holdouts.is_file() or holdouts.is_symlink():
        raise RuntimeError(f"Benchmark holdout bundle is missing: {holdouts}")
    _validate_holdout_report(report, holdouts=holdouts, registry=registry_path)
    _validate_policy_contract(policy, policy_contract)
    inputs: dict[str, Any] = {
        "holdouts": _relative_artifact_record(holdouts, base=output.parent),
        "holdout_report": _relative_artifact_record(
            report_path, base=output.parent
        ),
        "registry": {
            "path": str(registry_path),
            "repository_relative_path": _registry_relative_path(registry_path),
            "size": registry_path.stat().st_size,
            "sha256": _stream_file_sha256(registry_path),
            "canonical_sha256": _canonical_json_sha256(registry),
        },
        "policy_sha256": _canonical_json_sha256(policy),
        "policy_contract": dict(policy_contract),
    }
    inputs["bundle_sha256"] = _canonical_json_sha256(inputs)
    return inputs


def _atomic_contamination_manifest(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_contamination_index(
    index: ContaminationIndex,
    path: str | Path,
    *,
    holdouts_path: str | Path | None = None,
    holdout_report_path: str | Path | None = None,
    benchmark_registry_path: str | Path | None = None,
) -> None:
    """Seal a disk index to the exact holdout bundle and pinned policy.

    The default paths match the production layout used by ``stage_runner``.
    All provenance inputs are mandatory; a free-floating index cannot be
    published because filtering with it would not prove which benchmarks and
    policy produced its arrays.
    """

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_contract = _policy_contract(index)
    inputs = _contamination_inputs_for_save(
        output,
        holdouts_path=holdouts_path,
        holdout_report_path=holdout_report_path,
        benchmark_registry_path=benchmark_registry_path,
        policy_contract=policy_contract,
    )
    arrays = {
        "exact": np.asarray(
            sorted(value.encode("ascii") for value in index.exact), dtype="S64"
        ),
        "ngram_postings": _postings_array(index.ngram_postings),
        "short_ngram_postings": _postings_array(index.short_ngram_postings),
        "code_ngram_postings": _postings_array(index.code_ngram_postings),
        "code_skeleton_ngram_postings": _postings_array(
            index.code_skeleton_ngram_postings
        ),
    }
    array_files: dict[str, dict[str, Any]] = {}
    for name, values in arrays.items():
        destination = output.with_suffix(f".{name}.npy")
        temporary_array = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary_array.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_array, destination)
        finally:
            temporary_array.unlink(missing_ok=True)
        array_files[name] = {
            **_relative_artifact_record(destination, base=output.parent),
            "count": int(len(values)),
            "shape": [int(value) for value in values.shape],
            "dtype": str(values.dtype),
            "dtype_sha256": _canonical_json_sha256(
                np.lib.format.dtype_to_descr(values.dtype)
            ),
        }
    payload: dict[str, Any] = {
        "schema": CONTAMINATION_INDEX_SCHEMA,
        "status": "complete",
        # Keep the path-only inventory for release tooling while the signed
        # artifact records below provide the fail-closed integrity contract.
        "arrays": {
            name: str(record["path"]) for name, record in array_files.items()
        },
        "array_artifacts": array_files,
        "arrays_sha256": _canonical_json_sha256(array_files),
        "counts": {name: int(len(values)) for name, values in arrays.items()},
        "distinct_shingles": {
            "ngrams": len(index.ngram_postings),
            "short_ngrams": len(index.short_ngram_postings),
            "code_ngrams": len(index.code_ngram_postings),
            "code_skeleton_ngrams": len(index.code_skeleton_ngram_postings),
        },
        "suppressed_shingles": dict(index.suppressed_shingles),
        **policy_contract,
        "inputs": inputs,
        "inputs_sha256": _canonical_json_sha256(inputs),
    }
    payload["manifest_sha256"] = _contamination_manifest_sha256(payload)
    _atomic_contamination_manifest(output, payload)


def _sorted_contains(values: np.ndarray, value: Any) -> bool:
    position = int(np.searchsorted(values, value))
    return position < len(values) and bool(values[position] == value)


def _matching_one_group(values: np.ndarray, candidates: set[int], minimum: int) -> bool:
    group_counts: dict[bytes, int] = {}
    hashes = values["hash"]
    for candidate in candidates:
        value = np.uint64(candidate)
        left = int(np.searchsorted(hashes, value, side="left"))
        right = int(np.searchsorted(hashes, value, side="right"))
        for raw_group in values["group"][left:right]:
            group = bytes(raw_group)
            count = group_counts.get(group, 0) + 1
            if count >= minimum:
                return True
            group_counts[group] = count
    return False


def _longest_run_ndarray(values: np.ndarray, ordered: list[int]) -> int:
    """Longest unbroken run of matching n-grams sharing one evaluation row.

    The sorted-ndarray twin of decontaminate.longest_contiguous_run. Copied text
    forms a continuous span; incidental agreement never does, however much of it
    accumulates. Carries the active run length per row in a single pass.
    """

    hashes = values["hash"]
    groups_col = values["group"]
    best = 0
    active: dict[bytes, int] = {}
    for shingle in ordered:
        value = np.uint64(shingle)
        left = int(np.searchsorted(hashes, value, side="left"))
        right = int(np.searchsorted(hashes, value, side="right"))
        if left == right:
            active = {}
            continue
        extended: dict[bytes, int] = {}
        for raw_group in groups_col[left:right]:
            group = bytes(raw_group)
            length = active.get(group, 0) + 1
            extended[group] = length
            if length > best:
                best = length
        active = extended
    return best


@dataclass(frozen=True)
class DiskContaminationIndex:
    exact: np.ndarray
    ngram_postings: np.ndarray
    short_ngram_postings: np.ndarray
    code_ngram_postings: np.ndarray
    ngram_size: int
    minimum_matching_ngrams: int
    short_ngram_size: int
    minimum_short_matching_ngrams: int
    code_ngram_size: int
    minimum_code_matching_ngrams: int
    code_skeleton_ngram_postings: np.ndarray
    code_skeleton_ngram_size: int
    minimum_code_skeleton_matching_ngrams: int
    maximum_shingle_rows: int
    match_fraction: float = 0.0
    contiguous_run_minimum: int = 0

    def reason(self, text: str) -> str | None:
        normalized = canonical_text(text)
        digest = np.bytes_(hashlib.sha256(normalized.encode()).hexdigest().encode("ascii"))
        if _sorted_contains(self.exact, digest):
            return "benchmark_exact"
        run_minimum = int(getattr(self, "contiguous_run_minimum", 0) or 0)
        if run_minimum > 0 and _longest_run_ndarray(
            self.ngram_postings, ordered_ngram_hashes(normalized, self.ngram_size)
        ) >= run_minimum:
            return "benchmark_contiguous_run"
        if _matching_one_group(
            self.ngram_postings,
            ngram_hashes(normalized, self.ngram_size),
            required_matches(self.minimum_matching_ngrams,
                             len(ngram_hashes(normalized, self.ngram_size)),
                             getattr(self, "match_fraction", 0.0)),
        ):
            return "benchmark_ngram"
        if looks_like_code(text):
            if _matching_one_group(
                self.code_ngram_postings,
                code_ngram_hashes(text, self.code_ngram_size),
                required_matches(self.minimum_code_matching_ngrams,
                                 len(code_ngram_hashes(text, self.code_ngram_size)),
                                 getattr(self, "match_fraction", 0.0)),
            ):
                return "benchmark_code_ngram"
            if _matching_one_group(
                self.code_skeleton_ngram_postings,
                code_skeleton_ngram_hashes(text, self.code_skeleton_ngram_size),
                required_matches(self.minimum_code_skeleton_matching_ngrams,
                                 len(code_skeleton_ngram_hashes(text, self.code_skeleton_ngram_size)),
                                 getattr(self, "match_fraction", 0.0)),
            ):
                return "benchmark_code_skeleton_ngram"
        if _matching_one_group(
            self.short_ngram_postings,
            ngram_hashes(normalized, self.short_ngram_size),
            required_matches(self.minimum_short_matching_ngrams,
                             len(ngram_hashes(normalized, self.short_ngram_size)),
                             getattr(self, "match_fraction", 0.0)),
        ):
            return "benchmark_short_ngram"
        return None


def _resolve_registry_for_load(
    record: Mapping[str, Any],
    explicit_path: str | Path | None,
) -> Path:
    try:
        expected_size = int(record.get("size", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Contamination index has an invalid registry contract"
        ) from exc
    expected_sha256 = str(record.get("sha256") or "")
    if expected_size < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("Contamination index has an invalid registry contract")
    if explicit_path is not None:
        unresolved = Path(explicit_path).expanduser()
    else:
        relative = record.get("repository_relative_path")
        if isinstance(relative, str) and relative:
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(
                    "Contamination index registry path escapes the repository"
                )
            unresolved = repository_root() / relative_path
        else:
            recorded = record.get("path")
            if not isinstance(recorded, str) or not recorded:
                raise RuntimeError(
                    "Contamination index has no resolvable pinned benchmark registry"
                )
            unresolved = Path(recorded).expanduser()
    if unresolved.is_symlink():
        raise RuntimeError(
            f"Pinned benchmark registry may not be a symbolic link: {unresolved}"
        )
    candidate = unresolved.resolve()
    if not candidate.is_file():
        raise RuntimeError(f"Pinned benchmark registry is unavailable: {candidate}")
    if candidate.stat().st_size != expected_size:
        raise RuntimeError(
            f"Pinned benchmark registry size changed after index construction: {candidate}"
        )
    if _stream_file_sha256(candidate) != expected_sha256:
        raise RuntimeError(
            f"Pinned benchmark registry hash changed after index construction: {candidate}"
        )
    return candidate


def _validate_contamination_inputs(
    index_path: Path,
    payload: Mapping[str, Any],
    *,
    benchmark_registry: Mapping[str, Any] | None,
    benchmark_registry_path: str | Path | None,
) -> None:
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("Contamination index has no benchmark input contract")
    if payload.get("inputs_sha256") != _canonical_json_sha256(inputs):
        raise RuntimeError("Contamination index benchmark input contract is corrupt")
    claimed_bundle = inputs.get("bundle_sha256")
    unsigned_inputs = {
        key: value for key, value in inputs.items() if key != "bundle_sha256"
    }
    if claimed_bundle != _canonical_json_sha256(unsigned_inputs):
        raise RuntimeError("Contamination index benchmark bundle digest is corrupt")
    holdout_record = inputs.get("holdouts")
    report_record = inputs.get("holdout_report")
    registry_record = inputs.get("registry")
    if (
        not isinstance(holdout_record, Mapping)
        or not isinstance(report_record, Mapping)
        or not isinstance(registry_record, Mapping)
    ):
        raise RuntimeError("Contamination index benchmark artifact inventory is incomplete")
    holdouts = _safe_relative_artifact(
        index_path.parent,
        holdout_record,
        label="Benchmark holdout bundle",
    )
    report_path = _safe_relative_artifact(
        index_path.parent,
        report_record,
        label="Benchmark holdout report",
    )
    registry_path = _resolve_registry_for_load(
        registry_record, benchmark_registry_path
    )
    report = _read_json_mapping(report_path, "Benchmark holdout report")
    _validate_holdout_report(report, holdouts=holdouts, registry=registry_path)
    registry = load_yaml(registry_path)
    if registry.get("schema") != "metis.contamination-registry/v2":
        raise RuntimeError("Pinned benchmark registry has an unsupported schema")
    if (
        registry_record.get("canonical_sha256")
        != _canonical_json_sha256(registry)
    ):
        raise RuntimeError(
            "Pinned benchmark registry canonical content changed after index construction"
        )
    if benchmark_registry is not None:
        if _canonical_json_sha256(benchmark_registry) != _canonical_json_sha256(
            registry
        ):
            raise RuntimeError(
                "Runtime benchmark registry does not match the index input contract"
            )
    policy = registry.get("policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("Pinned benchmark registry has no policy mapping")
    if inputs.get("policy_sha256") != _canonical_json_sha256(policy):
        raise RuntimeError(
            "Pinned benchmark decontamination policy changed after index construction"
        )
    policy_contract = inputs.get("policy_contract")
    if not isinstance(policy_contract, Mapping):
        raise RuntimeError("Contamination index has no effective policy contract")
    _validate_policy_contract(policy, policy_contract)
    _validate_policy_contract(
        policy_contract,
        {field: payload.get(field) for field in CONTAMINATION_POLICY_FIELDS},
    )


def _load_contamination_arrays(
    index_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    paths = payload.get("arrays")
    records = payload.get("array_artifacts")
    if not isinstance(paths, Mapping) or set(paths) != set(
        CONTAMINATION_ARRAY_NAMES
    ):
        raise RuntimeError("Contamination index path inventory is incomplete")
    if not isinstance(records, Mapping) or set(records) != set(
        CONTAMINATION_ARRAY_NAMES
    ):
        raise RuntimeError("Contamination index array inventory is incomplete")
    if payload.get("arrays_sha256") != _canonical_json_sha256(records):
        raise RuntimeError("Contamination index array inventory is corrupt")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(
        CONTAMINATION_ARRAY_NAMES
    ):
        raise RuntimeError("Contamination index has no array counts")
    arrays: dict[str, np.ndarray] = {}
    for name in CONTAMINATION_ARRAY_NAMES:
        record = records[name]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Contamination array {name} has no artifact contract")
        expected_name = index_path.with_suffix(f".{name}.npy").name
        if record.get("path") != expected_name or paths.get(name) != expected_name:
            raise RuntimeError(
                f"Contamination array {name} does not use its canonical filename"
            )
        array_path = _safe_relative_artifact(
            index_path.parent,
            record,
            label=f"Contamination array {name}",
        )
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Contamination array {name} is not a valid NPY file") from exc
        expected_dtype = np.dtype("S64") if name == "exact" else POSTING_DTYPE
        if array.dtype != expected_dtype or array.ndim != 1:
            raise RuntimeError(
                f"Contamination array {name} has an unexpected dtype or shape"
            )
        expected_shape = record.get("shape")
        if expected_shape != [int(value) for value in array.shape]:
            raise RuntimeError(f"Contamination array {name} shape changed")
        if record.get("dtype") != str(array.dtype) or record.get(
            "dtype_sha256"
        ) != _canonical_json_sha256(np.lib.format.dtype_to_descr(array.dtype)):
            raise RuntimeError(f"Contamination array {name} dtype contract changed")
        try:
            expected_count = int(record["count"])
            manifest_count = int(counts[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Contamination array {name} has invalid counts"
            ) from exc
        if expected_count != len(array) or manifest_count != len(array):
            raise RuntimeError(f"Contamination array {name} count changed")
        arrays[name] = array
    return arrays


def load_contamination_index(
    path: str | Path,
    *,
    benchmark_registry: Mapping[str, Any] | None = None,
    benchmark_registry_path: str | Path | None = None,
) -> ContaminationIndex | DiskContaminationIndex:
    index_path = Path(path).resolve()
    payload = _read_json_mapping(index_path, "Contamination index manifest")
    if payload.get("schema") != CONTAMINATION_INDEX_SCHEMA:
        raise RuntimeError(
            "Contamination index lacks sealed array and benchmark-input provenance; "
            f"rebuild it with schema {CONTAMINATION_INDEX_SCHEMA} before filtering"
        )
    if (
        payload.get("status") != "complete"
        or payload.get("manifest_sha256")
        != _contamination_manifest_sha256(payload)
    ):
        raise RuntimeError("Contamination index manifest failed integrity validation")
    _validate_contamination_inputs(
        index_path,
        payload,
        benchmark_registry=benchmark_registry,
        benchmark_registry_path=benchmark_registry_path,
    )
    arrays = _load_contamination_arrays(index_path, payload)
    return DiskContaminationIndex(
        exact=arrays["exact"],
        ngram_postings=arrays["ngram_postings"],
        short_ngram_postings=arrays["short_ngram_postings"],
        code_ngram_postings=arrays["code_ngram_postings"],
        ngram_size=int(payload["ngram_size"]),
        minimum_matching_ngrams=int(payload["minimum_matching_ngrams"]),
        short_ngram_size=int(payload["short_ngram_size"]),
        minimum_short_matching_ngrams=int(payload["minimum_short_matching_ngrams"]),
        code_ngram_size=int(payload["code_ngram_size"]),
        minimum_code_matching_ngrams=int(payload["minimum_code_matching_ngrams"]),
        code_skeleton_ngram_postings=arrays["code_skeleton_ngram_postings"],
        code_skeleton_ngram_size=int(payload["code_skeleton_ngram_size"]),
        minimum_code_skeleton_matching_ngrams=int(
            payload["minimum_code_skeleton_matching_ngrams"]
        ),
        maximum_shingle_rows=int(payload["maximum_shingle_rows"]),
        match_fraction=float(payload.get("match_fraction", 0.0)),
        contiguous_run_minimum=int(payload.get("contiguous_run_minimum", 0)),
    )


def build_datatrove_decontamination_filter(
    index_path: str | Path,
    exclusion_writer: Any = None,
    benchmark_registry: dict[str, Any] | None = None,
) -> Any:
    from datatrove.pipeline.filters.base_filter import BaseFilter

    class MetisDecontaminationFilter(BaseFilter):
        name = "Metis benchmark decontamination"
        type = "DECONT"

        def __init__(self) -> None:
            super().__init__(exclusion_writer=exclusion_writer)
            self.index = load_contamination_index(
                index_path,
                benchmark_registry=benchmark_registry,
            )

        def filter(self, doc: Any) -> bool | tuple[bool, str]:
            if benchmark_registry is not None:
                benchmark_id = benchmark_genealogy_match(doc.metadata, benchmark_registry)
                if benchmark_id:
                    doc.metadata["benchmark_genealogy_match"] = benchmark_id
                    doc.metadata["decontamination_reason"] = "benchmark_genealogy"
                    return False, "benchmark_genealogy"
            reason = self.index.reason(str(doc.text))
            if reason:
                doc.metadata["decontamination_reason"] = reason
                return False, reason
            return True

    return MetisDecontaminationFilter()
