"""Incremental exact dedup over immutable eligible shards, not copied text.

Each batch preserves sorted occurrence/provenance leaves. Active size-tiered
runs retain every distinct occurrence, including losing sources. A winner is
the associative/commutative minimum of ``winner_key``; arrival times and ranks
are deliberately absent. Readers take a committed metadata snapshot.

Ingest scans only its supplied shards. Compaction reads only metadata, merging
at most ``max_fan_in`` similarly sized runs; a row crosses O(log N) tiers.
Current runs and permanent occurrence leaves occupy O(N) metadata. Kernel-owned
read leases protect snapshots while superseded compacted caches are reclaimed.
Locks recover after worker/node failure without age-based reclamation.
Exact counts are an explicit metadata
scan in ``dedup_status``, not a corpus scan per ingest.
"""
from __future__ import annotations

import contextlib
import logging
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Iterator, Sequence

from .common import digest_json, read_receipt, under_root, write_receipt
from .dedup_runs import (
    COMPARATOR, _alive, bucket_for, canonical_rows, input_contract, merge_rows, metadata_lock, positive_integer,
    prepare_inputs, prepared_rows, read_reference_metadata, receipt_file_pin, sort_key, winner_key, write_run,
)
from .dedup_signatures import generate_signatures, signature_status
from .dedup_storage import (
    bind_working_budget, namespace_name, quota_receipt, quota_rmtree, quota_unlink, storage_namespace,
)


SCHEMA = "metis17.exact-index/v1"
FAN_IN = 16
LOCK_TIMEOUT = 3600
DEFERRED_RUN_LIMIT = 256
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CommitView:
    sha256: str
    runs: dict[str, dict[str, Any]]


def _lock(root: Path, name: str):
    return metadata_lock(
        root / "locks" / name, timeout=None if name == "ingest-publication" else LOCK_TIMEOUT,
    )


def _batch_key(batch_id: str) -> str:
    if not isinstance(batch_id, str) or not batch_id or len(batch_id.encode("utf-8")) > 1024:
        raise ValueError("batch_id must be a nonempty, bounded stable identity")
    return digest_json(batch_id)


def _initialize(root: Path, bucket_count: int) -> dict[str, Any]:
    positive_integer(bucket_count, "bucket_count")
    root.mkdir(parents=True, exist_ok=True)
    expected = {
        "schema": SCHEMA, "bucket_count": bucket_count, "partition": "full-sha256-modulo/v1",
        "comparator": COMPARATOR,
        "occurrence_identity": "source,object,doc-id,raw-content-sha256/v1",
        "input_identity": "immutable-shard-path,sha256/v1",
        "stage_receipt_hash": "canonical-payload-sha256/v1",
    }
    with _lock(root, "publication"):
        path = root / "INDEX.json"
        if path.exists():
            existing = read_receipt(path)
            existing.setdefault("stage_receipt_hash", "canonical-payload-sha256/v1")
            if existing != expected:
                raise ValueError("Dedup layout/policy changed; use a new index, not a new task count")
        else:
            write_receipt(path, expected)
    return expected


def _configuration(root: Path) -> dict[str, Any]:
    value = read_receipt(root / "INDEX.json")
    if value.get("schema") != SCHEMA or value.get("comparator") != COMPARATOR:
        raise ValueError("Unsupported exact-dedup index/comparator")
    positive_integer(value["bucket_count"], "bucket_count")
    if value.get("partition") != "full-sha256-modulo/v1":
        raise ValueError("Unsupported exact-dedup partition")
    if value.get("stage_receipt_hash", "canonical-payload-sha256/v1") != "canonical-payload-sha256/v1":
        raise ValueError("Dedup receipt-hash contract changed; use a new index")
    return value


def _bucket_path(root: Path, bucket: int) -> Path:
    return root / "buckets" / f"{bucket:06d}" / "CURRENT.json"


def _visible_runs(
    root: Path, bucket: int, cache: dict[str, _CommitView | None] | None = None,
) -> list[dict[str, Any]]:
    """Reuse proofs only inside one publication transaction, not across snapshots."""
    path = _bucket_path(root, bucket)
    if not path.exists():
        return []
    state = read_receipt(path)
    if state.get("schema") != "metis17.exact-bucket/v1" or state.get("bucket") != bucket:
        raise ValueError("Incorrect bucket receipt or partition")
    cache = {} if cache is None else cache
    result, seen = [], set()
    for run in state["runs"]:
        key = run["commit"]
        if key not in cache:
            commit = under_root(root, key)
            if commit.exists():
                manifest = read_receipt(commit)
                indexed = {entry["run_id"]: entry for entry in manifest["runs"]}
                if len(indexed) != len(manifest["runs"]):
                    raise ValueError("A committed manifest repeats a run identity")
                cache[key] = _CommitView(digest_json(manifest), indexed)
            else:
                cache[key] = None
        view = cache[key]
        if view is None:
            if run.get("commit_kind") == "batch":
                continue
            raise ValueError("A published compaction receipt is missing")
        if view.sha256 != run["commit_sha256"]:
            raise ValueError("Run publication receipt digest mismatch")
        unwrapped = {key: value for key, value in run.items()
                     if key not in {"commit", "commit_sha256", "commit_kind"}}
        if view.runs.get(run["run_id"]) != unwrapped or run["bucket"] != bucket:
            raise ValueError("Run is not covered by its committed manifest/bucket")
        if run["run_id"] in seen:
            raise ValueError("Duplicate active run; refusing to double corpus accounting")
        seen.add(run["run_id"])
        result.append(run)
    return result


def _publish_bucket(
    root: Path, bucket: int, runs: list[dict[str, Any]], *,
    retired: list[dict[str, Any]] | None = None,
) -> None:
    if retired is None:
        path = _bucket_path(root, bucket)
        retired = read_receipt(path).get("retired", []) if path.exists() else []
    write_receipt(_bucket_path(root, bucket), {
        "schema": "metis17.exact-bucket/v1", "bucket": bucket, "runs": runs, "retired": retired,
    })


def _selected_buckets(config: dict[str, Any], bucket_ids: Sequence[int] | None) -> list[int]:
    buckets = list(range(config["bucket_count"])) if bucket_ids is None else list(bucket_ids)
    if any(type(bucket) is not int or not 0 <= bucket < config["bucket_count"] for bucket in buckets):
        raise ValueError("bucket_ids must be valid index buckets, not worker/rank identifiers")
    if len(set(buckets)) != len(buckets):
        raise ValueError("Duplicate bucket_ids would double coverage")
    return sorted(buckets)


@contextlib.contextmanager
def _snapshot(
    root: Path, bucket_ids: Sequence[int] | None,
) -> Iterator[tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]]:
    lease = root / "readers" / f"{uuid.uuid4().hex}.json"
    lease_lock = lease.with_suffix(".lock")
    try:
        with metadata_lock(lease_lock):
            with _lock(root, "publication"):
                config = _configuration(root)
                cache: dict[str, _CommitView | None] = {}
                buckets = {bucket: _visible_runs(root, bucket, cache)
                           for bucket in _selected_buckets(config, bucket_ids)}
                write_receipt(lease, {
                    "schema": "metis17.exact-read-lease/v2",
                    "lock": str(lease_lock.relative_to(root)),
                    "host": socket.gethostname(), "pid": os.getpid(),
                    "run_ids": [run["run_id"] for runs in buckets.values() for run in runs],
                })
            try:
                yield config, buckets
            finally:
                with _lock(root, "publication"):
                    lease.unlink(missing_ok=True)
    finally:
        if not lease.exists():
            lease_lock.unlink(missing_ok=True)


def _reclaim_caches(
    root: Path, bucket: int, runs: list[dict[str, Any]], working_budget: Any = None,
) -> None:
    """Caller holds bucket + publication locks; permanent batch leaves stay intact."""
    path = _bucket_path(root, bucket)
    if not path.exists():
        return
    retired = read_receipt(path).get("retired", [])
    if not retired:
        return
    protected = set()
    for lease in (root / "readers").glob("*.json"):
        value = read_receipt(lease)
        if value.get("schema") == "metis17.exact-read-lease/v2":
            lease_lock = under_root(root, value["lock"])
            if lease_lock != lease.with_suffix(".lock"):
                raise ValueError("Reader lease references a different lock")
            try:
                with metadata_lock(lease_lock, timeout=0, create=False):
                    lease.unlink()
            except TimeoutError:
                protected.update(value["run_ids"])
            else:
                lease_lock.unlink(missing_ok=True)
        elif value.get("schema") != "metis17.exact-read-lease/v1":
            raise ValueError("Unrecognized reader lease; refusing metadata reclamation")
        elif value["host"] == socket.gethostname() and not _alive(value["pid"]):
            lease.unlink()
        else:
            protected.update(value["run_ids"])
    kept = []
    for run in retired:
        if run["run_id"] in protected:
            kept.append(run)
            continue
        if run["commit_kind"] != "compaction":
            raise ValueError("Permanent source occurrence leaves cannot be reclaimed")
        if working_budget is None:
            for name in ("occurrences", "winners"):
                quota_unlink(None, under_root(root, run[name]["path"]))
        else:
            storage = run.get("storage")
            directory = under_root(root, storage["directory"]) if storage else root / "compactions"
            kind = storage["kind"] if storage else "legacy-compactions"
            if storage and storage["namespace"] != namespace_name(kind, directory):
                raise ValueError("Compaction storage namespace differs from its receipt")
            with storage_namespace(working_budget, kind, directory) as quota:
                for name in ("occurrences", "winners"):
                    quota_unlink(quota, under_root(root, run[name]["path"]))
    if len(kept) != len(retired):
        _publish_bucket(root, bucket, runs, retired=kept)


def _wrapped(run: dict[str, Any], root: Path, path: Path, manifest: dict[str, Any], kind: str):
    return {
        **run, "commit": str(path.relative_to(root)), "commit_sha256": digest_json(manifest),
        "commit_kind": kind,
    }


def _matching_tier(runs: list[dict[str, Any]], fan_in: int, minimum_fan_in: int = 2) -> list[dict[str, Any]]:
    tiers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        tiers[run["level"]].append(run)
    for level in sorted(tiers):
        if len(tiers[level]) >= minimum_fan_in:
            return sorted(tiers[level], key=lambda run: run["run_id"])[:fan_in]
    return []


def _remove_work_run(root: Path, run: dict[str, Any], quota: Any = None) -> None:
    for name in ("occurrences", "winners"):
        quota_unlink(quota, under_root(root, run[name]["path"]))


def _build_batch_runs(
    root: Path, work: Path, inputs: list[dict[str, Any]], bucket_count: int, batch_size: int,
    quota: Any = None,
) -> list[dict[str, Any]]:
    buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    runs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sequence = 0

    def merge(bucket: int, selected: list[dict[str, Any]]) -> None:
        nonlocal sequence
        sequence += 1
        merged = write_run(
            root, work / f"sort-{sequence:09d}",
            merge_rows(root, selected, batch_size=batch_size),
            bucket=bucket, weight=sum(run["weight"] for run in selected),
            batch_size=batch_size, preserve_deliveries=True,
            quota=quota,
        )
        runs[bucket] = [run for run in runs[bucket] if run not in selected] + [merged]
        for run in selected:
            _remove_work_run(root, run, quota)

    def flush(bucket: int) -> None:
        nonlocal sequence
        buffer = buffers[bucket]
        if not buffer:
            return
        buffer.sort(key=sort_key)
        sequence += 1
        run = write_run(
            root, work / f"sort-{sequence:09d}", buffer,
            bucket=bucket, weight=len(buffer), batch_size=batch_size, preserve_deliveries=True,
            quota=quota,
        )
        runs[bucket].append(run)
        buffer.clear()
        while selected := _matching_tier(runs[bucket], FAN_IN):
            merge(bucket, selected)

    for artifact in inputs:
        with contextlib.closing(prepared_rows(artifact, batch_size)) as rows:
            for row, _ in rows:
                bucket = bucket_for(row["dedup_hash"], bucket_count)
                buffers[bucket].append(row)
                if len(buffers[bucket]) >= batch_size:
                    flush(bucket)
    for bucket in list(buffers):
        flush(bucket)
        while len(runs[bucket]) > 1:
            merge(bucket, sorted(runs[bucket], key=lambda run: run["weight"])[:FAN_IN])
    result = [run for bucket in sorted(runs) for run in runs[bucket]]
    if sum(run["weight"] for run in result) != sum(item["rows"] for item in inputs):
        raise ValueError("Batch partition lost or duplicated prepared rows")
    return result


def _claimed_input(root: Path, path: Path, input_id: str) -> bool:
    if not path.exists():
        return False
    claim = read_receipt(path)
    commit = under_root(root, claim["commit"])
    if not commit.exists():
        return False
    manifest = read_receipt(commit)
    if (
        claim["input_id"] != input_id or digest_json(manifest) != claim["commit_sha256"]
        or input_id not in manifest["admitted_inputs"]
    ):
        raise ValueError("Input admission claim is not covered by a committed batch")
    return True


def _publish_batch(
    root: Path, runs: list[dict[str, Any]], manifest: dict[str, Any], commit: Path,
    new_inputs: list[dict[str, Any]], quota: Any, working_budget: Any, *, defer_compaction: bool = False,
) -> None:
    if quota is not None:
        quota.reconcile()
    reported_backpressure = False
    while True:
        with contextlib.ExitStack() as bucket_locks:
            if defer_compaction:
                # Publish is already serial; queue its callers away from the
                # shared lock so hundreds of ingesters cannot starve maintenance.
                bucket_locks.enter_context(_lock(root, "ingest-publication"))
            else:
                for bucket in sorted(run["bucket"] for run in runs):
                    bucket_locks.enter_context(_lock(root, f"bucket-{bucket:06d}"))
            with _lock(root, "publication"):
                cache: dict[str, _CommitView | None] = {}
                current = {run["bucket"]: _visible_runs(root, run["bucket"], cache) for run in runs}
                overdue = [
                    bucket for bucket, visible in current.items()
                    if (
                        defer_compaction and len(visible) >= DEFERRED_RUN_LIMIT
                        or not defer_compaction and working_budget is not None and _matching_tier(visible, FAN_IN)
                    )
                ]
                if not overdue:
                    for run in runs:
                        bucket = run["bucket"]
                        _publish_bucket(root, bucket, [
                            *current[bucket], _wrapped(run, root, commit, manifest, "batch"),
                        ])
                    for item in new_inputs:
                        input_id = item["input_id"]
                        write_receipt(root / "inputs" / input_id[:2] / f"{input_id}.json", {
                            "input_id": input_id, "sha256": item["sha256"],
                            "commit": str(commit.relative_to(root)),
                            "commit_sha256": digest_json(manifest),
                        })
                    # Failed publication has neither visible rows nor training credits.
                    quota_receipt(quota, commit, manifest)
                    return
        if defer_compaction:
            if not reported_backpressure:
                _LOG.warning("Exact-index publication is waiting for compaction in buckets %s", overdue)
                reported_backpressure = True
            time.sleep(0.1)
            continue
        # Release bucket/publication locks before helping a concurrent compactor.
        # A capacity pause here bounds active streams instead of accumulating runs.
        compact_dedup(root, bucket_ids=overdue, working_budget=working_budget)


def ingest_eligible(
    parquet_paths: Sequence[Path], output_dir: Path, *, batch_id: str,
    bucket_count: int = 64, batch_size: int = 4096,
    stage_receipt_path: Path | None = None, stage_receipt_sha256: str | None = None,
    stage_receipt_file_sha256: str | None = None,
    receipt_file_sha256: str | None = None,
    working_budget: Any = None,
    defer_compaction: bool = False,
) -> dict[str, Any]:
    """Admit finalized shards once, including overlapping/repartitioned retries.

    ``stage_receipt_sha256`` is ``digest_json(read_receipt(path))``, the embedded
    canonical receipt seal. ``receipt_file_sha256`` optionally pins exact JSON
    bytes; ``stage_receipt_file_sha256`` remains its compatibility spelling.
    Production chunk receipts require both explicit path and canonical seal,
    plus raw-object completion evidence. Immutable proof copies survive pointer
    replay. Only whole-object convenience receipts may be auto-discovered.
    ``batch_size``/worker assignments are not part of batch identity. Returned
    admission counts are not training credits: consume current winners.

    ``working_budget`` is a shared ``storage.WorkingBudget``, not an entered
    whole-index quota. Production RUN/limits auto-enable it if omitted. Bulk
    metadata is metered before writes; small control files use its reserved
    metadata allowance. Attaching a budget adopts existing outputs once.
    ``defer_compaction`` moves only cross-batch maintenance off the caller's
    critical path; a bounded independent compactor must service those runs.
    """
    if type(defer_compaction) is not bool:
        raise ValueError("defer_compaction must be a boolean")
    positive_integer(batch_size, "batch_size")
    positive_integer(bucket_count, "bucket_count")
    stage_receipt_file_sha256 = receipt_file_pin(receipt_file_sha256, stage_receipt_file_sha256)
    key = _batch_key(batch_id)
    root = Path(output_dir).resolve()
    working_budget = bind_working_budget(root, working_budget)
    stage_proofs: list[dict[str, Any]] = []
    inputs = prepare_inputs(
        parquet_paths, stage_receipt_path=stage_receipt_path,
        stage_receipt_sha256=stage_receipt_sha256,
        receipt_snapshot_dir=root / "receipts", stage_proofs=stage_proofs,
        stage_receipt_file_sha256=stage_receipt_file_sha256,
        working_budget=working_budget,
    )
    _initialize(root, bucket_count)
    identity = digest_json({
        **input_contract(inputs, stage_proofs, stage_receipt_file_sha256), "bucket_count": bucket_count,
    })
    batch_root = root / "batches" / key
    commit = batch_root / "COMMITTED.json"
    with _lock(root, f"batch-{key}"), storage_namespace(working_budget, "exact-batch", batch_root) as quota:
        if commit.exists():
            existing = read_receipt(commit)
            legacy_identities = {
                digest_json({
                    "inputs": existing["inputs"], "bucket_count": bucket_count,
                    "empty_stage_sha256": stage_proofs[0]["sha256"]
                    if stage_receipt_path is not None and not inputs else None,
                }),
            }
            if inputs or not stage_proofs:
                legacy_identities.add(digest_json({
                    "inputs": existing["inputs"], "bucket_count": bucket_count,
                }))
            legacy = (
                "stage_receipts" not in existing and stage_receipt_file_sha256 is None
                and existing["input_sha256"] in legacy_identities
                and len(existing["inputs"]) == len(inputs)
                and all({"path", "sha256", "rows", "stage_receipt_sha256"}.issubset(old)
                        and {
                            key: current.get("stage_receipt_origin", current.get(key))
                            if key == "stage_receipt_path" else current.get(key)
                            for key in old
                        } == old
                        for old, current in zip(existing["inputs"], inputs))
            )
            if (existing["input_sha256"] != identity and not legacy) or existing["batch_id"] != batch_id:
                raise ValueError("The same batch_id was reused with different inputs")
            if not defer_compaction:
                compact_dedup(root, bucket_ids=[run["bucket"] for run in existing["runs"]],
                              working_budget=working_budget)
            return existing
        intent_path = batch_root / "INTENT.json"
        intent = {"batch_id": batch_id, "input_sha256": identity}
        if intent_path.exists():
            if read_receipt(intent_path) != intent:
                raise ValueError("The same batch_id was reused with different inputs")
            # An interrupted attempt may already have installed invisible bucket pointers.
            # A retry can now have no new inputs if another batch admitted the same shards.
            for bucket in range(bucket_count):
                with _lock(root, f"bucket-{bucket:06d}"):
                    with _lock(root, "publication"):
                        path = _bucket_path(root, bucket)
                        if path.exists():
                            state = read_receipt(path)
                            kept = [run for run in state["runs"]
                                    if run["commit"] != str(commit.relative_to(root))]
                            if len(kept) != len(state["runs"]):
                                _publish_bucket(root, bucket, kept)
        else:
            quota_receipt(quota, intent_path, intent)
        for abandoned in batch_root.glob("attempt-*"):
            if abandoned.is_dir():
                quota_rmtree(quota, abandoned)
        with contextlib.ExitStack() as stack:
            by_id = {item["input_id"]: item for item in inputs}
            new_inputs = []
            for input_id in sorted(by_id):
                stack.enter_context(_lock(root, f"input-{input_id}"))
                claim = root / "inputs" / input_id[:2] / f"{input_id}.json"
                if not _claimed_input(root, claim, input_id):
                    new_inputs.append(by_id[input_id])
            work = batch_root / f"attempt-{uuid.uuid4().hex}"
            work.mkdir(parents=True)
            published = False
            try:
                runs = _build_batch_runs(root, work, new_inputs, bucket_count, batch_size, quota)
                manifest = {
                    "schema": "metis17.exact-batch/v1", "status": "complete",
                    "batch_id": batch_id, "input_sha256": identity, "inputs": inputs,
                    "stage_receipts": stage_proofs,
                    "admitted_inputs": sorted(item["input_id"] for item in new_inputs),
                    "requested_input_rows": sum(item["rows"] for item in inputs),
                    "admitted_input_rows": sum(item["rows"] for item in new_inputs),
                    "runs": runs, "exact_complete": True,
                    "near_span_code_decisions": "require_closed_comparison_scope",
                }
                _publish_batch(
                    root, runs, manifest, commit, new_inputs, quota, working_budget,
                    defer_compaction=defer_compaction,
                )
                published = True
            finally:
                if not published and not commit.exists():
                    quota_rmtree(quota, work)
    if not defer_compaction:
        compact_dedup(root, bucket_ids=[run["bucket"] for run in runs], working_budget=working_budget)
    return manifest


def _recover_compaction(root: Path, bucket: int, working_budget: Any) -> None:
    pending = _bucket_path(root, bucket).with_name("COMPACTION_PENDING.json")
    if not pending.exists():
        return
    receipt = read_receipt(pending)
    directory = under_root(root, receipt["directory"])
    expected_parent = root / "compaction-runs" / f"{bucket:06d}"
    if (
        receipt.get("schema") != "metis17.compaction-pending/v1"
        or directory.parent != expected_parent or len(directory.name) != 32
        or any(character not in "0123456789abcdef" for character in directory.name)
    ):
        raise ValueError("Unsafe interrupted compaction directory")
    with _lock(root, "publication"):
        active = _visible_runs(root, bucket)
        state_path = _bucket_path(root, bucket)
        retired = read_receipt(state_path).get("retired", []) if state_path.exists() else []
        committed = any(
            under_root(root, run["commit"]).parent == directory for run in [*active, *retired]
        )
    if not committed and directory.exists():
        with storage_namespace(working_budget, "compaction", directory) as quota:
            quota_rmtree(quota, directory)
    pending.unlink()


def compact_dedup(
    output_dir: Path, *, bucket_ids: Sequence[int] | None = None, max_fan_in: int = 16,
    working_budget: Any = None, max_merges: int | None = None, minimum_fan_in: int = 2,
) -> dict[str, Any]:
    """Size-tier, quota-metered compaction; never scan old prepared text.

    Each merge gets a separately recoverable budget namespace. Quota pauses
    leave committed runs authoritative and retain a journal for partial cleanup.
    """
    positive_integer(max_fan_in, "max_fan_in", minimum=2)
    positive_integer(minimum_fan_in, "minimum_fan_in", minimum=2)
    if minimum_fan_in > max_fan_in:
        raise ValueError("minimum_fan_in exceeds max_fan_in")
    if max_merges is not None:
        positive_integer(max_merges, "max_merges")
    root = Path(output_dir).resolve()
    working_budget = bind_working_budget(root, working_budget)
    config = _configuration(root)
    buckets = _selected_buckets(config, bucket_ids)
    merges, read_rows, written_rows = 0, 0, 0
    for bucket in buckets:
        with _lock(root, f"bucket-{bucket:06d}"):
            _recover_compaction(root, bucket, working_budget)
            with _lock(root, "publication"):
                runs = _visible_runs(root, bucket)
            while selected := _matching_tier(
                runs, max_fan_in, 2 if len(runs) >= DEFERRED_RUN_LIMIT // 2 else minimum_fan_in,
            ):
                pending = None
                if working_budget is None:
                    prefix = root / "compactions" / uuid.uuid4().hex
                    directory = prefix.parent
                    commit = prefix.with_suffix(".json")
                else:
                    directory = root / "compaction-runs" / f"{bucket:06d}" / uuid.uuid4().hex
                    prefix = directory / "run"
                    commit = directory / "COMPLETE.json"
                    pending = _bucket_path(root, bucket).with_name("COMPACTION_PENDING.json")
                    write_receipt(pending, {
                        "schema": "metis17.compaction-pending/v1",
                        "directory": str(directory.relative_to(root)),
                    })
                with storage_namespace(working_budget, "compaction", directory) as quota:
                    merged = write_run(
                        root, prefix, merge_rows(root, selected), bucket=bucket,
                        weight=sum(run["weight"] for run in selected), batch_size=4096,
                        preserve_deliveries=False, quota=quota, storage_kind="compaction",
                    )
                    manifest = {
                        "schema": "metis17.exact-compaction/v1", "status": "complete",
                        "input_run_ids": sorted(run["run_id"] for run in selected),
                        "input_commit_sha256": sorted(run["commit_sha256"] for run in selected),
                        "runs": [merged], "metadata_only": True,
                    }
                    quota_receipt(quota, commit, manifest)
                with _lock(root, "publication"):
                    # Deferred publishers may append while metadata is merged.
                    # Only the pinned inputs are replaced, never the old snapshot.
                    current = _visible_runs(root, bucket)
                    selected_by_id = {run["run_id"]: run for run in selected}
                    current_by_id = {run["run_id"]: run for run in current}
                    if any(current_by_id.get(key) != run for key, run in selected_by_id.items()):
                        raise RuntimeError("Compaction inputs changed while their bucket was leased")
                    runs = [run for run in current if run["run_id"] not in selected_by_id]
                    runs.append(_wrapped(merged, root, commit, manifest, "compaction"))
                    retired = read_receipt(_bucket_path(root, bucket)).get("retired", [])
                    retired.extend(run for run in selected if run["commit_kind"] == "compaction")
                    _publish_bucket(root, bucket, runs, retired=retired)
                    if pending is not None:
                        pending.unlink()
                    _reclaim_caches(root, bucket, runs, working_budget)
                merges += 1
                read_rows += sum(run["occurrences"]["rows"] for run in selected)
                written_rows += merged["occurrences"]["rows"]
                if max_merges is not None and merges >= max_merges:
                    break
            with _lock(root, "publication"):
                runs = _visible_runs(root, bucket)
                _reclaim_caches(root, bucket, runs, working_budget)
        if max_merges is not None and merges >= max_merges:
            break
    return {
        "schema": "metis17.exact-compaction-progress/v1", "buckets": buckets,
        "merges": merges, "metadata_rows_read": read_rows, "metadata_rows_written": written_rows,
        "max_fan_in": max_fan_in, "metadata_only": True,
        "merge_budget_exhausted": max_merges is not None and merges >= max_merges,
    }


def iter_survivors(
    output_dir: Path, *, bucket_ids: Sequence[int] | None = None, batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Stream winner references from a consistent committed snapshot.

    Occurrence counts in individual run indexes cannot be summed across runs:
    redelivery under a different shard layout may share canonical occurrences.
    This iterator intentionally omits that count; ``dedup_status`` computes it.
    ``prepared_row`` is zero-based across the whole immutable shard, not a row
    group. Exhaust or close the iterator to release its metadata-cache lease.
    Parquet readers close before lease release, including on early termination.
    Do not read live compacted caches outside this lease protocol.
    """
    positive_integer(batch_size, "batch_size")
    root = Path(output_dir).resolve()
    with _snapshot(root, bucket_ids) as (config, buckets):
        for bucket, runs in buckets.items():
            with contextlib.closing(merge_rows(root, runs, winners=True, batch_size=batch_size)) as rows:
                for digest, group in groupby(rows, lambda row: row["dedup_hash"]):
                    if bucket_for(digest, config["bucket_count"]) != bucket:
                        raise ValueError("Survivor belongs to a different bucket")
                    best = min(group, key=winner_key)
                    yield {key: value for key, value in best.items() if key != "occurrence_count"}


def iter_occurrences(
    output_dir: Path, *, bucket_ids: Sequence[int] | None = None, batch_size: int = 4096,
) -> Iterator[dict[str, Any]]:
    """Stream canonical occurrences, including every lower-quality contender."""
    positive_integer(batch_size, "batch_size")
    root = Path(output_dir).resolve()
    with _snapshot(root, bucket_ids) as (config, buckets):
        for bucket, runs in buckets.items():
            with contextlib.closing(merge_rows(root, runs, batch_size=batch_size)) as rows:
                for row in canonical_rows(rows):
                    if bucket_for(row["dedup_hash"], config["bucket_count"]) != bucket:
                        raise ValueError("Occurrence belongs to a different bucket")
                    yield row


def dedup_status(output_dir: Path) -> dict[str, Any]:
    """Compute authoritative exact counts with one explicit metadata-only scan."""
    root = Path(output_dir).resolve()
    if not (root / "INDEX.json").exists():
        return {"schema": "metis17.exact-status/v1", "initialized": False,
                "input_rows": 0, "raw_occurrences": 0, "unique_winners": 0}
    with _snapshot(root, None) as (config, buckets):
        raw, unique, raw_characters, winner_characters = 0, 0, 0, 0
        for bucket, runs in buckets.items():
            with contextlib.closing(merge_rows(root, runs)) as rows:
                for digest, group in groupby(canonical_rows(rows), lambda row: row["dedup_hash"]):
                    if bucket_for(digest, config["bucket_count"]) != bucket:
                        raise ValueError("Metadata partition has missing/misassigned coverage")
                    best = None
                    for row in group:
                        raw += 1
                        raw_characters += row["character_count"]
                        if best is None or winner_key(row) < winner_key(best):
                            best = row
                    unique += 1
                    winner_characters += best["character_count"]
        input_rows = sum(run["weight"] for runs in buckets.values() for run in runs)
        return {
            "schema": "metis17.exact-status/v1", "initialized": True,
            "bucket_count": config["bucket_count"],
            "snapshot_sha256": digest_json(buckets),
            "input_rows": input_rows, "raw_occurrences": raw, "unique_winners": unique,
            "repeated_occurrence_deliveries": input_rows - raw, "duplicate_occurrences": raw - unique,
            "raw_characters": raw_characters, "winner_characters": winner_characters,
            "active_runs": sum(map(len, buckets.values())), "exact_complete": True,
            "near_span_code_decisions": "require_closed_comparison_scope",
            "source_text_deleted": False,
        }
