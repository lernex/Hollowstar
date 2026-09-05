from __future__ import annotations

import hashlib
import contextlib
import itertools
import json
import multiprocessing
import os
import shutil
import socket
import unittest
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from metis_data17 import dedup, dedup_locks, dedup_runs, dedup_storage
from metis_data17.common import digest_json, read_receipt, sha256_file, write_receipt
from metis_data17.dedup import (
    compact_dedup, dedup_status, generate_signatures, ingest_eligible,
    iter_occurrences, iter_survivors, signature_status,
)
from metis_data17.dedup_runs import PREP_SCHEMA, bucket_for, metadata_lock, winner_key
from metis_data17.dedup_signatures import _shingles, minhash_bands, minhash_signature


def _ingest_process(arguments):
    path, output, batch_id, buckets = arguments
    return ingest_eligible([Path(path)], Path(output), batch_id=batch_id,
                           bucket_count=buckets, batch_size=3)["admitted_input_rows"]


def _hold_metadata_lock(path, connection):
    with metadata_lock(Path(path)):
        connection.send("locked")
        connection.recv()


def _hold_reader_lease(root, connection):
    with patch("metis_data17.dedup.socket.gethostname", return_value="failed-compute-node"):
        reader = iter_survivors(Path(root), batch_size=1)
        next(reader)
        connection.send("reading")
        connection.recv()
        reader.close()


def _ingest_budget_process(arguments):
    from metis_data17.storage import WorkingBudget

    path, output, batch_id, release = arguments
    return ingest_eligible(
        [Path(path)], Path(output), batch_id=batch_id, bucket_count=1, batch_size=3,
        working_budget=WorkingBudget(Path(release), allocation_bytes=4096),
    )["admitted_input_rows"]


class Metis17DedupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".metis17-dedup-test-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.output = self.root / "index"

    def tearDown(self):
        shutil.rmtree(self.root)

    def row(self, doc_id, *, text=None, priority=50, score=-1.0, source="source",
            object_id="object", category="web", language="en"):
        text = text if text is not None else f"document {doc_id} has useful unique text"
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return {
            "doc_id": str(doc_id), "text": text, "content_hash": content_hash,
            "dedup_hash": hashlib.sha256((category + "\0" + text).encode()).hexdigest(),
            "source_id": source, "object_id": object_id, "priority": priority,
            "quality_score": score, "language": language, "category": category,
            "character_count": len(text), "metadata_json": json.dumps({"source": source}),
        }

    def shard(self, name, rows, *, status="ELIGIBLE", schema=None):
        directory = self.root / "prepared" / name / "eligible" / digest_json(name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-000000.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema or PREP_SCHEMA), path,
                       row_group_size=3, compression="zstd")
        receipt = {
            "schema": "metis17.prepared-object/v1", "status": status,
            "eligible": status == "ELIGIBLE", "training_ready": status == "ELIGIBLE",
            "pending_reasons": [] if status == "ELIGIBLE" else ["decontamination"],
            "receipt_path": str((directory / "ELIGIBLE.json").relative_to(self.root)),
            "chunks": [{"path": str(path.relative_to(self.root)), "sha256": sha256_file(path),
                        "byte_count": path.stat().st_size, "records": len(rows)}],
            "eligible_documents": len(rows),
        }
        write_receipt(directory / "ELIGIBLE.json", receipt)
        return path

    def production_chunk(self, name, rows):
        chunk_id = digest_json(["chunk", name])
        fingerprint = digest_json(["normalized", name])
        directory = self.root / "prepared" / name / "chunks" / chunk_id
        eligible = directory / "eligible" / digest_json(["eligibility", name])
        eligible.mkdir(parents=True)
        path = eligible / "part-000000.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=PREP_SCHEMA), path)
        normalized = self.root / "normalized" / name / fingerprint
        normalized.mkdir(parents=True)
        base_path = normalized / "part-000000.parquet"
        ready_path = normalized / "part-000000.READY.json"
        completion_path = normalized / "NORMALIZED.json"
        source = rows[0]["source_id"] if rows else "source"
        object_id = rows[0]["object_id"] if rows else "object"
        chunk = {
            "path": str(base_path.relative_to(self.root)),
            "ready_receipt": str(ready_path.relative_to(self.root)),
            "chunk_id": chunk_id, "sha256": "a" * 64, "records": max(1, len(rows)),
            "byte_count": 123, "document_start": 0, "document_stop": max(1, len(rows)),
        }
        ready = {
            "schema": "metis17.base-chunk/v1", "status": "NORMALIZED_CHUNK_READY",
            "source_id": source, "object_id": object_id, "normalization_fingerprint": fingerprint,
            "chunk_id": chunk_id, "chunk_index": 0, "chunk": chunk,
        }
        write_receipt(ready_path, ready)
        completion = {
            "schema": "metis17.normalized-object/v1", "status": "NORMALIZED",
            "reblock_complete": True, "normalization_fingerprint": fingerprint,
            "source_id": source, "object_id": object_id,
            "chunks": [chunk], "chunk_receipts": [chunk["ready_receipt"]],
        }
        write_receipt(completion_path, completion)
        stage_path = directory / "PREP_COMPLETE.json"
        outputs = [{
            "path": str(path.relative_to(self.root)), "records": len(rows),
            "byte_count": path.stat().st_size, "sha256": sha256_file(path),
        }]
        receipt = {
            "schema": "metis17.prepared-chunk/v1", "status": "ELIGIBLE",
            "eligible": True, "training_ready": True, "object_complete": True,
            "source_id": source, "object_id": object_id, "pending_reasons": [],
            "chunk_id": chunk_id, "chunk_index": 0, "normalization_fingerprint": fingerprint,
            "base_chunk": chunk["path"], "base_chunk_receipt": chunk["ready_receipt"],
            "object_completion": {
                "path": str(completion_path.relative_to(self.root)),
                "receipt_sha256": digest_json(completion),
            },
            "inputs": {"base_chunk_receipt_sha256": digest_json(ready)},
            "chunks": outputs, "screened_chunks": outputs, "normalized_chunks": [chunk],
            "eligible_documents": len(rows), "receipt_path": str(stage_path.relative_to(self.root)),
        }
        write_receipt(stage_path, receipt)
        write_receipt(eligible / "FILTERED.json", {
            **receipt, "schema": "metis17.filtered-chunk/v1", "status": "FILTERED",
            "eligible": False, "training_ready": False,
            "receipt_path": str((eligible / "FILTERED.json").relative_to(self.root)),
        })
        return path, stage_path, completion_path

    def ingest(self, path, batch_id, *, buckets=3, output=None, batch_size=3):
        return ingest_eligible([path], output or self.output, batch_id=batch_id,
                               bucket_count=buckets, batch_size=batch_size)

    def survivors(self, output=None):
        return {row["dedup_hash"]: row for row in iter_survivors(output or self.output)}

    def budget(self, derived_bytes=4_000_000):
        from metis_data17.storage import WorkingBudget

        release = self.root / "working-release"
        release.mkdir(exist_ok=True)
        write_receipt(release / "limits.json", {
            "capacity_confirmation": "pending", "max_raw_bytes": 1,
            "max_working_bytes": derived_bytes + 524_289,
            "policy_and_metadata_reserve_bytes": 524_288,
            "filesystem_free_floor_bytes": 0,
        })
        return WorkingBudget(release, allocation_bytes=4096)

    def test_late_higher_quality_replaces_winner_without_losing_occurrences(self):
        low = self.shard("low", [self.row("low", text="shared exact text", priority=20)])
        high = self.shard("high", [self.row("high", text="shared exact text", priority=90)])
        checksums = [sha256_file(low), sha256_file(high)]
        first = self.ingest(low, "first")
        self.assertEqual(first["admitted_input_rows"], 1)
        self.assertEqual(next(iter(self.survivors().values()))["doc_id"], "low")
        self.ingest(high, "later")
        winner = next(iter(self.survivors().values()))
        self.assertEqual(winner["doc_id"], "high")
        self.assertEqual((winner["prepared_path"], winner["prepared_row"]), (str(high), 0))
        self.assertNotIn("text", winner)
        self.assertEqual({row["doc_id"] for row in iter_occurrences(self.output)}, {"high", "low"})
        status = dedup_status(self.output)
        self.assertEqual((status["input_rows"], status["raw_occurrences"], status["unique_winners"]), (2, 2, 1))
        self.assertEqual(status["duplicate_occurrences"], 1)
        self.assertEqual([sha256_file(low), sha256_file(high)], checksums)
        for manifest in (self.output / "batches").glob("*/COMMITTED.json"):
            self.assertTrue(read_receipt(manifest)["exact_complete"])

    def test_arrival_order_and_exact_ties_are_deterministic(self):
        rows = [
            self.row("low", text="same", priority=4, score=1000.0),
            self.row("unknown", text="same", priority=8),
            self.row("z-known", text="same", priority=8, score=0.0),
            self.row("a-known", text="same", priority=8, score=0.0),
        ]
        paths = [self.shard(str(index), [row]) for index, row in enumerate(rows)]
        # Reverse arrival plus mixed interleavings catch timestamp/first-seen and unstable tie rules.
        for trial, order in enumerate(((0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1))):
            output = self.root / f"order-{trial}"
            for index in order:
                self.ingest(paths[index], str(index), output=output, buckets=1)
            winners = list(self.survivors(output).values())
            self.assertEqual([row["doc_id"] for row in winners], ["a-known"])
            self.assertEqual(dedup_status(output)["raw_occurrences"], 4)

    def test_comparator_matches_independent_reference_and_min_is_associative(self):
        path = self.shard("comparator", [
            self.row(f"{priority}-{score}", priority=priority, score=score)
            for priority in (-5, 0, 5) for score in (-1.0, 0.0, 0.8, 3.0)
        ])
        self.ingest(path, "comparator", buckets=1)
        rows = list(iter_occurrences(self.output))
        expected = sorted(rows, key=lambda row: (
            -row["priority"], -(row["quality_score"] if row["quality_score"] >= 0 else -1),
            row["doc_id"],
        ))
        self.assertEqual([row["doc_id"] for row in sorted(rows, key=winner_key)],
                         [row["doc_id"] for row in expected])
        self.assertEqual(
            winner_key({**rows[0], "metadata_json": '{"download_timestamp":"1900-01-01"}'}),
            winner_key({**rows[0], "metadata_json": '{"download_timestamp":"2999-01-01"}'}),
        )
        for a, b, c in itertools.product(rows[:5], repeat=3):
            choose = lambda x, y: min((x, y), key=winner_key)
            self.assertEqual(choose(a, b), choose(b, a))
            self.assertEqual(choose(choose(a, b), c), choose(a, choose(b, c)))

    def test_repeat_batch_and_new_delivery_never_double_counts(self):
        path = self.shard("repeat", [self.row("1"), self.row("2")])
        first = self.ingest(path, "stable", buckets=1)
        again = self.ingest(path, "stable", buckets=1, batch_size=17)
        self.assertEqual(first, again)
        redelivery = self.ingest(path, "new-worker-layout", buckets=1)
        self.assertEqual(redelivery["admitted_input_rows"], 0)
        status = dedup_status(self.output)
        self.assertEqual((status["input_rows"], status["raw_occurrences"], status["unique_winners"]), (2, 2, 2))
        copied = self.shard("copied", [self.row("1"), self.row("2")])
        self.ingest(copied, "copied-delivery", buckets=1)
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 2)
        self.assertEqual(dedup_status(self.output)["unique_winners"], 2)

    def test_repartitioned_canonical_occurrences_are_not_credited_twice(self):
        rows = [self.row(str(index)) for index in range(9)]
        initial = self.shard("initial", rows)
        self.ingest(initial, "original", buckets=1)
        split1, split2 = self.shard("split1", rows[:4]), self.shard("split2", rows[4:])
        self.ingest(split1, "split1", buckets=1)
        self.ingest(split2, "split2", buckets=1)
        status = dedup_status(self.output)
        self.assertEqual((status["input_rows"], status["raw_occurrences"], status["unique_winners"]), (18, 9, 9))
        self.assertEqual(status["repeated_occurrence_deliveries"], 9)
        self.assertEqual(status["winner_characters"], sum(row["character_count"] for row in rows))
        self.assertEqual(len(list(iter_occurrences(self.output))), 9)

    def test_identical_input_aliases_choose_the_same_reference_in_either_order(self):
        rows = [self.row("same")]
        paths = [self.shard("alias-a", rows), self.shard("alias-z", rows)]
        self.assertEqual(sha256_file(paths[0]), sha256_file(paths[1]))
        winners = []
        for trial, order in enumerate((paths, paths[::-1])):
            output = self.root / f"alias-order-{trial}"
            for index, path in enumerate(order):
                self.ingest(path, str(index), buckets=1, output=output)
            winners.append(next(iter(self.survivors(output).values())))
            status = dedup_status(output)
            self.assertEqual((status["raw_occurrences"], status["unique_winners"]), (1, 1))
            self.assertEqual(status["input_rows"], 2)
        self.assertEqual(winners[0], winners[1])
        self.assertEqual(winners[0]["prepared_path"], str(paths[0]))

    def test_batch_conflict_and_changed_layout_are_rejected(self):
        path = self.shard("conflict", [self.row("first")])
        self.ingest(path, "stable")
        different = self.shard("different", [self.row("second")])
        with self.assertRaisesRegex(ValueError, "same batch_id"):
            self.ingest(different, "stable")
        with self.assertRaisesRegex(ValueError, "layout/policy changed"):
            self.ingest(path, "rank-count-is-not-bucket-count", buckets=7)
        self.shard("conflict", [self.row("changed", priority=88)])
        with self.assertRaisesRegex(ValueError, "same batch_id"):
            self.ingest(path, "stable")

    def test_partition_coverage_and_bucket_count_invariance(self):
        rows = [self.row(str(index), text=f"distinct key {index % 37}", priority=index)
                for index in range(111)]
        path = self.shard("coverage", rows)
        expected = {row["dedup_hash"]: row for row in rows}
        for buckets in (1, 7):
            output = self.root / f"bucket-count-{buckets}"
            self.ingest(path, "all", output=output, buckets=buckets, batch_size=5)
            winners = self.survivors(output)
            self.assertEqual({key: row["doc_id"] for key, row in winners.items()},
                             {key: row["doc_id"] for key, row in expected.items()})
            occurrences = list(iter_occurrences(output))
            self.assertEqual({(row["prepared_path"], row["prepared_row"]) for row in occurrences},
                             {(str(path), index) for index in range(len(rows))})
            self.assertEqual(len(occurrences), len(rows))
            claimed = []
            for bucket in range(buckets):
                part = list(iter_survivors(output, bucket_ids=[bucket]))
                for row in part:
                    self.assertEqual(int(row["dedup_hash"], 16) % buckets, bucket)
                    self.assertEqual(bucket_for(row["dedup_hash"], buckets), bucket)
                claimed.extend(row["doc_id"] for row in part)
            self.assertEqual(len(claimed), len(set(claimed)))
            self.assertEqual(set(claimed), {row["doc_id"] for row in expected.values()})
        with self.assertRaises(ValueError):
            compact_dedup(self.root / "bucket-count-7", bucket_ids=[1, 1])
        with self.assertRaises(ValueError):
            compact_dedup(self.root / "bucket-count-7", bucket_ids=[7])

    def test_external_multirun_compaction_is_bounded_and_metadata_only(self):
        paths = [self.shard(f"many-{index}", [
            self.row(f"{index}-{offset}", text=f"shared key {offset % 13}", priority=index)
            for offset in range(31)
        ]) for index in range(9)]
        with patch.object(dedup, "compact_dedup", return_value={}):
            for index, path in enumerate(paths):
                self.ingest(path, str(index), buckets=1, batch_size=4)
        self.assertEqual(dedup_status(self.output)["active_runs"], 9)
        original_merge = dedup.merge_rows
        fan_ins = []
        original_file = dedup_runs.pq.ParquetFile
        projections = []

        def merged(root, runs, **kwargs):
            fan_ins.append(len(runs))
            return original_merge(root, runs, **kwargs)

        class MetadataFile:
            def __init__(self, path, *args, **kwargs):
                if Path(path) in paths:
                    raise AssertionError("Compaction reread a prepared text shard")
                self.inner = original_file(path, *args, **kwargs)
                self.schema_arrow = self.inner.schema_arrow

            def iter_batches(self, *args, **kwargs):
                projections.append(kwargs.get("columns"))
                if not kwargs.get("columns") or "text" in kwargs["columns"]:
                    raise AssertionError("Compaction projected full text")
                return self.inner.iter_batches(*args, **kwargs)

            def close(self):
                self.inner.close()

        with patch.object(dedup, "merge_rows", side_effect=merged), \
             patch.object(dedup_runs.pq, "ParquetFile", MetadataFile), \
             patch.object(dedup, "prepared_rows", side_effect=AssertionError("text source read")):
            report = compact_dedup(self.output, max_fan_in=3)
        self.assertGreater(report["merges"], 0)
        self.assertTrue(projections)
        self.assertTrue(fan_ins and max(fan_ins) <= 3)
        status = dedup_status(self.output)
        self.assertEqual((status["raw_occurrences"], status["unique_winners"]), (279, 13))
        self.assertTrue(all(row["priority"] == 8 for row in self.survivors().values()))
        state = read_receipt(self.output / "buckets/000000/CURRENT.json")
        levels = [run["level"] for run in state["runs"]]
        self.assertEqual(len(levels), len(set(levels)))
        self.assertLessEqual(len(levels), status["input_rows"].bit_length())
        # All committed occurrence leaves remain independently verifiable after compaction.
        for manifest in (self.output / "batches").glob("*/COMMITTED.json"):
            receipt = read_receipt(manifest)
            self.assertEqual(sum(run["occurrences"]["rows"] for run in receipt["runs"]), 31)
            for run in receipt["runs"]:
                artifact = run["occurrences"]
                self.assertEqual(sha256_file(self.output / artifact["path"]), artifact["sha256"])

    def test_interrupted_publication_is_invisible_and_retry_safe(self):
        path = self.shard("interrupted", [self.row(str(index)) for index in range(11)])
        original = dedup.write_receipt

        def interrupted(path, value):
            if path.name == "COMMITTED.json":
                raise OSError("interrupted before final publication")
            return original(path, value)

        with patch.object(dedup, "write_receipt", side_effect=interrupted), \
             patch.object(dedup_storage, "write_receipt", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.ingest(path, "interrupted", buckets=5)
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 0)
        self.assertEqual(list(iter_survivors(self.output)), [])
        self.ingest(path, "another-worker", buckets=5)
        resumed = self.ingest(path, "interrupted", buckets=5, batch_size=13)
        self.assertEqual(resumed["admitted_input_rows"], 0)
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 11)
        self.assertEqual(len(list(iter_survivors(self.output))), 11)

    def test_reader_snapshot_survives_later_compaction(self):
        first = self.shard("snapshot-first", [
            self.row("first-a", text="A", priority=1), self.row("first-b", text="B", priority=1),
        ])
        later = self.shard("snapshot-later", [
            self.row("later-a", text="A", priority=5), self.row("later-b", text="B", priority=5),
        ])
        self.ingest(first, "first", buckets=1)
        iterator = iter_survivors(self.output, batch_size=1)
        old = [next(iterator)]
        self.ingest(later, "later", buckets=1)
        old.extend(iterator)
        self.assertEqual({row["priority"] for row in old}, {1})
        self.assertEqual({row["priority"] for row in iter_survivors(self.output)}, {5})

    def test_compacted_caches_are_bounded_and_reader_leases_protect_them(self):
        paths = [self.shard(f"cache-{index}", [
            self.row(f"{index}-a", text="A", priority=index),
            self.row(f"{index}-b", text="B", priority=index),
        ]) for index in range(8)]
        for index in range(2):
            self.ingest(paths[index], f"cache-{index}", buckets=1)
        state_path = self.output / "buckets/000000/CURRENT.json"
        old_run = read_receipt(state_path)["runs"][0]
        self.assertEqual(old_run["commit_kind"], "compaction")
        old_files = [self.output / old_run[name]["path"] for name in ("occurrences", "winners")]
        reader = iter_survivors(self.output, batch_size=1)
        first = next(reader)
        for index in range(2, 8):
            self.ingest(paths[index], f"cache-{index}", buckets=1)
        self.assertTrue(all(path.exists() for path in old_files))
        old = [first, *reader]
        self.assertEqual({row["priority"] for row in old}, {1})
        compact_dedup(self.output)
        self.assertFalse(any(path.exists() for path in old_files))
        state = read_receipt(state_path)
        self.assertEqual(state["retired"], [])
        active = [run for run in state["runs"] if run["commit_kind"] == "compaction"]
        self.assertEqual(len(list((self.output / "compactions").glob("*.occurrences.parquet"))),
                         len(active))
        self.assertEqual(len(list((self.output / "batches").glob("*/attempt-*/*.occurrences.parquet"))), 8)
        self.assertFalse(list((self.output / "readers").glob("*.json")))
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 16)

    def test_parquet_readers_close_before_lease_release_and_failed_merge_return(self):
        path = self.shard("reader-close-order", [self.row(str(index)) for index in range(8)])
        self.ingest(path, "reader-close-order", buckets=1)
        original_file = dedup_runs.pq.ParquetFile
        original_snapshot = dedup._snapshot
        observed = []

        class ObservedParquet:
            def __init__(self, path, *args, **kwargs):
                self.inner = original_file(path, *args, **kwargs)
                self.schema_arrow = self.inner.schema_arrow
                self.closed = False
                observed.append(self)

            def iter_batches(self, *args, **kwargs):
                return self.inner.iter_batches(*args, **kwargs)

            def close(self):
                self.inner.close()
                self.closed = True

        @contextlib.contextmanager
        def checked_snapshot(*args, **kwargs):
            with original_snapshot(*args, **kwargs) as snapshot:
                try:
                    yield snapshot
                finally:
                    self.assertTrue(observed)
                    self.assertTrue(all(reader.closed for reader in observed),
                                    "The lease must outlive every opened Parquet reader")

        with patch.object(dedup_runs.pq, "ParquetFile", ObservedParquet), \
             patch.object(dedup, "_snapshot", checked_snapshot):
            for factory in (iter_survivors, iter_occurrences):
                observed.clear()
                iterator = factory(self.output, batch_size=1)
                next(iterator)
                self.assertTrue(any(not reader.closed for reader in observed))
                iterator.close()
                self.assertTrue(all(reader.closed for reader in observed))
        observed.clear()
        runs = read_receipt(self.output / "buckets/000000/CURRENT.json")["runs"]
        with patch.object(dedup_runs.pq, "ParquetFile", ObservedParquet), \
             patch.object(dedup_runs.ParquetSink, "append", side_effect=RuntimeError("failed merge")), \
             contextlib.closing(dedup_runs.merge_rows(self.output, runs, batch_size=1)) as rows:
            with self.assertRaisesRegex(RuntimeError, "failed merge"):
                dedup_runs.write_run(
                    self.output, self.output / "failed-merge", rows, bucket=0,
                    weight=8, batch_size=1, preserve_deliveries=False,
                )
            self.assertTrue(observed)
            self.assertTrue(all(reader.closed for reader in observed))

    def test_parallel_threads_same_batch_are_idempotent(self):
        path = self.shard("threads", [self.row(str(index)) for index in range(16)])
        with ThreadPoolExecutor(max_workers=4) as pool:
            reports = list(pool.map(lambda _: self.ingest(path, "same", buckets=2), range(4)))
        self.assertTrue(all(report == reports[0] for report in reports))
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 16)

    def test_parallel_process_writers_and_overlapping_inputs(self):
        rows = [self.row(str(index), text=f"duplicate {index % 6}", priority=index) for index in range(18)]
        paths = [self.shard(f"process-{index}", rows[index::3]) for index in range(3)]
        arguments = [(str(path), str(self.output), f"process-{index}", 3)
                     for index, path in enumerate(paths)]
        arguments.append((str(paths[0]), str(self.output), "repeat-process", 3))
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
            admitted = list(pool.map(_ingest_process, arguments))
        self.assertEqual(sum(admitted), 18)
        status = dedup_status(self.output)
        self.assertEqual((status["raw_occurrences"], status["unique_winners"]), (18, 6))
        self.assertEqual({row["priority"] for row in self.survivors().values()}, set(range(12, 18)))

    def test_corrupt_receipts_artifacts_and_ineligible_inputs_fail_closed(self):
        pending = self.shard("pending", [self.row("bad")], status="NORMALIZED_PENDING_DECONTAMINATION")
        with self.assertRaisesRegex(ValueError, "finalized eligible"):
            self.ingest(pending, "pending")
        path = self.shard("good", [self.row("good")])
        receipt = path.parent / "ELIGIBLE.json"
        with self.assertRaisesRegex(ValueError, "receipt digest"):
            ingest_eligible([path], self.output, batch_id="wrong-proof", bucket_count=3,
                            stage_receipt_path=receipt, stage_receipt_sha256="0" * 64)
        self.ingest(path, "good")
        current = next((self.output / "buckets").glob("*/CURRENT.json"))
        original = current.read_bytes()
        value = json.loads(original)
        value["bucket"] += 1
        current.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Receipt hash mismatch"):
            dedup_status(self.output)
        current.write_bytes(original)
        state = read_receipt(current)
        artifact = self.output / state["runs"][0]["occurrences"]["path"]
        with artifact.open("r+b") as stream:
            stream.seek(4)
            old = stream.read(1)
            stream.seek(4)
            stream.write(bytes([old[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            dedup_status(self.output)

    def test_invalid_metadata_never_publishes_a_batch(self):
        changes = [
            {"quality_score": float("nan")}, {"quality_score": float("inf")},
            {"quality_score": -0.1}, {"metadata_json": "[]"},
            {"metadata_json": '{"score": NaN}'}, {"metadata_json": "{"},
            {"character_count": -1}, {"doc_id": ""}, {"language": None},
            {"content_hash": "wrong"}, {"dedup_hash": "wrong"},
        ]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                path = self.shard(f"invalid-{index}", [{**self.row("bad"), **change}])
                output = self.root / f"invalid-index-{index}"
                with self.assertRaises((ValueError, TypeError)):
                    self.ingest(path, "invalid", output=output, buckets=1)
                self.assertFalse(list((output / "batches").glob("*/COMMITTED.json")))
                self.assertEqual(dedup_status(output)["raw_occurrences"], 0)
        wrong_schema = pa.schema([
            pa.field(field.name, pa.int64() if field.name == "priority" else field.type)
            for field in PREP_SCHEMA
        ])
        path = self.shard("wrong-schema", [self.row("wrong")], schema=wrong_schema)
        with self.assertRaisesRegex(ValueError, "priority"):
            self.ingest(path, "wrong-schema")

    def test_empty_shard_and_empty_inputs_are_valid_without_fake_winners(self):
        empty = self.shard("empty", [])
        manifest = self.ingest(empty, "empty")
        self.assertEqual(manifest["admitted_input_rows"], 0)
        self.assertEqual(manifest["runs"], [])
        self.assertEqual(list(iter_survivors(self.output)), [])
        self.assertEqual(dedup_status(self.output)["unique_winners"], 0)
        self.assertEqual(compact_dedup(self.output)["merges"], 0)
        ingest_eligible([], self.output, batch_id="empty-subset", bucket_count=3,
                        stage_receipt_path=empty.parent / "ELIGIBLE.json")
        with self.assertRaises(ValueError):
            ingest_eligible([], self.output, batch_id="bad-size", bucket_count=3, batch_size=0)

    def test_prep_complete_alias_resolves_to_the_immutable_eligible_receipt(self):
        path = self.shard("prep-manifest", [self.row("document")])
        sealed = path.parent / "ELIGIBLE.json"
        payload = read_receipt(sealed)
        alias = path.parents[2] / "PREP_COMPLETE.json"
        write_receipt(alias, payload)
        first = ingest_eligible(
            [path], self.output, batch_id="manifest", bucket_count=1,
            stage_receipt_path=alias, stage_receipt_sha256=digest_json(payload),
        )
        self.assertEqual(first["inputs"][0]["stage_receipt_path"], str(sealed))
        again = ingest_eligible(
            [path], self.output, batch_id="manifest", bucket_count=1,
            stage_receipt_path=sealed, stage_receipt_sha256=digest_json(payload),
        )
        self.assertEqual(first, again)
        write_receipt(alias, {
            **payload, "status": "NORMALIZED_PENDING_DECONTAMINATION",
            "eligible": False, "training_ready": False, "pending_reasons": ["decontamination"],
        })
        with self.assertRaisesRegex(ValueError, "finalized eligible"):
            ingest_eligible([path], self.output, batch_id="pending-pointer", bucket_count=1,
                            stage_receipt_path=alias)

    def test_production_chunk_completion_and_distinct_receipt_hash_domains(self):
        path, stage, completion = self.production_chunk("production", [self.row("production")])
        payload = read_receipt(stage)
        file_sha = sha256_file(stage)
        self.assertNotEqual(file_sha, digest_json(payload))
        with self.assertRaisesRegex(ValueError, "canonical seal"):
            ingest_eligible([path], self.output, batch_id="wrong-hash-domain", bucket_count=1,
                            stage_receipt_path=stage, stage_receipt_sha256=file_sha)
        with self.assertRaisesRegex(ValueError, "full-file"):
            ingest_eligible([path], self.output, batch_id="wrong-file-domain", bucket_count=1,
                            stage_receipt_path=stage, stage_receipt_sha256=digest_json(payload),
                            stage_receipt_file_sha256=digest_json(payload))
        first = ingest_eligible([path], self.output, batch_id="production", bucket_count=1,
                                stage_receipt_path=stage, stage_receipt_sha256=digest_json(payload),
                                stage_receipt_file_sha256=file_sha)
        proof = first["stage_receipts"][0]
        self.assertEqual((proof["sha256"], proof["payload_sha256"]),
                         (digest_json(payload), digest_json(payload)))
        self.assertEqual(proof["file_sha256"], file_sha)
        self.assertEqual(read_receipt(Path(proof["snapshot"]["path"])), payload)
        self.assertEqual(len(proof["completion_proofs"]), 2)
        self.assertFalse((completion.parent / "part-000000.parquet").exists())
        winner = next(iter_survivors(self.output))
        self.assertEqual(winner["stage_receipt_sha256"], digest_json(payload))
        write_receipt(stage, {
            **payload, "status": "NORMALIZED_PENDING_DECONTAMINATION",
            "eligible": False, "training_ready": False, "chunks": [], "eligible_documents": 0,
            "pending_reasons": ["policy_replay"],
        })
        # Explicitly pinned retries use the retained proof, never the mutable new-policy pointer.
        again = ingest_eligible([path], self.output, batch_id="production", bucket_count=1,
                                stage_receipt_path=stage, stage_receipt_sha256=digest_json(payload),
                                stage_receipt_file_sha256=file_sha)
        self.assertEqual(first, again)
        self.assertEqual(sha256_file(Path(proof["snapshot"]["path"])), file_sha)
        with self.assertRaisesRegex(ValueError, "finalized eligible"):
            self.ingest(path, "new-policy", buckets=1)

    def test_canonical_pin_survives_json_formatting_and_pointer_replay(self):
        path, stage, _ = self.production_chunk("canonical-proof", [self.row("document")])
        payload = read_receipt(stage)
        seal = digest_json(payload)
        first = ingest_eligible([path], self.output, batch_id="canonical", bucket_count=1,
                                stage_receipt_path=stage, stage_receipt_sha256=seal)
        original_file_sha = sha256_file(stage)
        stage.write_text(json.dumps(json.loads(stage.read_text()), separators=(",", ":")))
        self.assertNotEqual(sha256_file(stage), original_file_sha)
        self.assertEqual(digest_json(read_receipt(stage)), seal)
        second = ingest_eligible([path], self.output, batch_id="canonical", bucket_count=1,
                                 stage_receipt_path=stage, stage_receipt_sha256=seal)
        self.assertEqual(first, second)
        write_receipt(stage, {
            **payload, "status": "ELIGIBLE_PENDING_OBJECT_COMPLETION",
            "eligible": False, "training_ready": False, "object_complete": False,
            "object_completion": None, "chunks": [], "eligible_documents": 0,
        })
        third = ingest_eligible([path], self.output, batch_id="canonical", bucket_count=1,
                                stage_receipt_path=stage, stage_receipt_sha256=seal)
        self.assertEqual(first, third)
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 1)

    def test_original_canonical_batches_remain_idempotent_after_proof_archiving(self):
        path = self.shard("original-canonical", [self.row("original")])
        original = self.ingest(path, "original", buckets=1)
        legacy = {key: value for key, value in original.items() if key != "stage_receipts"}
        added_fields = {
            "stage_receipt_file_sha256", "stage_receipt_payload_sha256",
            "stage_receipt_snapshot", "completion_proofs",
        }
        legacy["inputs"] = [
            {key: value for key, value in item.items() if key not in added_fields}
            for item in original["inputs"]
        ]
        legacy["input_sha256"] = digest_json({
            "inputs": legacy["inputs"], "bucket_count": 1, "empty_stage_sha256": None,
        })
        batch_path = self.output / "batches" / digest_json("original") / "COMMITTED.json"
        write_receipt(batch_path, legacy)
        bucket_path = self.output / "buckets/000000/CURRENT.json"
        state = read_receipt(bucket_path)
        state["runs"][0]["commit_sha256"] = digest_json(legacy)
        write_receipt(bucket_path, state)
        index = read_receipt(self.output / "INDEX.json")
        index.pop("stage_receipt_hash")
        write_receipt(self.output / "INDEX.json", index)
        self.assertEqual(self.ingest(path, "original", buckets=1), legacy)
        other = self.shard("different-canonical", [self.row("different")])
        with self.assertRaisesRegex(ValueError, "same batch_id"):
            self.ingest(other, "original", buckets=1)
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 1)

    def test_production_discovery_ignores_filtered_and_pending_inventories(self):
        path, stage, _ = self.production_chunk("production-pending", [self.row("pending")])
        payload = read_receipt(stage)
        filtered = path.parent / "FILTERED.json"
        with self.assertRaisesRegex(ValueError, "finalized eligible"):
            ingest_eligible([path], self.output, batch_id="internal-filtered", bucket_count=1,
                            stage_receipt_path=filtered, stage_receipt_sha256=digest_json(read_receipt(filtered)))
        write_receipt(stage, {
            **payload, "status": "ELIGIBLE_PENDING_OBJECT_COMPLETION",
            "eligible": False, "training_ready": False, "object_complete": False,
            "object_completion": None, "chunks": [], "eligible_documents": 0,
            "pending_reasons": ["object_reblock_incomplete"],
        })
        with self.assertRaisesRegex(ValueError, "finalized eligible"):
            self.ingest(path, "pending-object", buckets=1)
        write_receipt(stage, {**payload, "chunks": [], "eligible_documents": 0})
        with self.assertRaisesRegex(ValueError, "not attested"):
            ingest_eligible(
                [path], self.output, batch_id="screened-fallback-forbidden", bucket_count=1,
                stage_receipt_path=stage, stage_receipt_sha256=digest_json(read_receipt(stage)),
            )
        empty = ingest_eligible([], self.output, batch_id="empty-production", bucket_count=1,
                                stage_receipt_path=stage, stage_receipt_sha256=digest_json(read_receipt(stage)))
        self.assertEqual(empty["admitted_input_rows"], 0)
        self.assertEqual(len(empty["stage_receipts"]), 1)
        write_receipt(stage, payload)
        ingest_eligible([path], self.output, batch_id="promoted", bucket_count=1,
                        stage_receipt_path=stage, stage_receipt_sha256=digest_json(payload))
        self.assertEqual(dedup_status(self.output)["unique_winners"], 1)

    def test_production_chunks_cannot_use_discovery_or_an_unpinned_pointer(self):
        path, stage, _ = self.production_chunk("explicit-production", [self.row("document")])
        with self.assertRaisesRegex(ValueError, "explicit stage_receipt_path"):
            self.ingest(path, "no-explicit-stage", buckets=1)
        with self.assertRaisesRegex(ValueError, "canonical stage_receipt_sha256"):
            ingest_eligible([path], self.output, batch_id="unpinned", bucket_count=1,
                            stage_receipt_path=stage)
        with self.assertRaisesRegex(ValueError, "canonical stage_receipt_sha256"):
            generate_signatures(
                [path], self.root / "unpinned-signatures", batch_id="unpinned",
                snapshot="2026-09", semantic_namespace="web", stage_receipt_path=stage,
            )
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 0)

    def test_parent_accepted_snapshot_never_requires_the_mutable_pointer(self):
        path, stage, _ = self.production_chunk("accepted-snapshot", [self.row("document")])
        payload = read_receipt(stage)
        seal = digest_json(payload)
        accepted = self.root / "accepted-receipts" / f"{seal}.json"
        write_receipt(accepted, payload)
        stage.unlink()
        result = ingest_eligible(
            [path], self.output, batch_id="accepted", bucket_count=1,
            stage_receipt_path=accepted, stage_receipt_sha256=seal,
            receipt_file_sha256=sha256_file(accepted),
        )
        self.assertEqual(result["inputs"][0]["stage_receipt_path"], str(accepted))
        self.assertEqual(result["stage_receipts"][0]["path"], str(accepted))
        self.assertEqual(next(iter_survivors(self.output))["stage_receipt_sha256"], seal)
        signatures = generate_signatures(
            [path], self.root / "accepted-signatures", batch_id="accepted",
            snapshot="2026-09", semantic_namespace="web",
            stage_receipt_path=accepted, stage_receipt_sha256=seal,
            receipt_file_sha256=sha256_file(accepted),
        )
        self.assertEqual(signatures["documents"], 1)
        self.assertEqual(signatures["stage_receipts"][0]["path"], str(accepted))

    def test_receipt_relocation_to_parent_snapshot_preserves_batch_identity(self):
        path, stage, _ = self.production_chunk("relocated-snapshot", [self.row("document")])
        payload = read_receipt(stage)
        seal = digest_json(payload)
        first = ingest_eligible(
            [path], self.output, batch_id="relocated", bucket_count=1,
            stage_receipt_path=stage, stage_receipt_sha256=seal,
        )
        accepted = self.root / "accepted-receipts" / f"{seal}.json"
        write_receipt(accepted, payload)
        second = ingest_eligible(
            [path], self.output, batch_id="relocated", bucket_count=1,
            stage_receipt_path=accepted, stage_receipt_sha256=seal,
        )
        self.assertEqual(first, second)
        self.assertEqual(dedup_status(self.output)["raw_occurrences"], 1)

    def test_parent_accepted_snapshot_name_and_seal_are_not_bypassed_by_cache(self):
        path, stage, _ = self.production_chunk("accepted-corruption", [self.row("document")])
        payload = read_receipt(stage)
        seal = digest_json(payload)
        accepted = self.root / "accepted-receipts" / f"{seal}.json"
        write_receipt(accepted, payload)
        ingest_eligible([path], self.output, batch_id="accepted", bucket_count=1,
                        stage_receipt_path=accepted, stage_receipt_sha256=seal)
        wrong_name = accepted.with_name("0" * 64 + ".json")
        write_receipt(wrong_name, payload)
        with self.assertRaisesRegex(ValueError, "filename"):
            ingest_eligible([path], self.output, batch_id="wrong-name", bucket_count=1,
                            stage_receipt_path=wrong_name, stage_receipt_sha256=seal)
        write_receipt(accepted, {**payload, "eligible": False})
        with self.assertRaisesRegex(ValueError, "canonical seal"):
            ingest_eligible([path], self.output, batch_id="corrupted", bucket_count=1,
                            stage_receipt_path=accepted, stage_receipt_sha256=seal)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ingest_eligible(
                [path], self.output, batch_id="conflicting-pin", bucket_count=1,
                stage_receipt_path=accepted, stage_receipt_sha256=seal,
                receipt_file_sha256="0" * 64, stage_receipt_file_sha256="1" * 64,
            )

    def test_production_completion_evidence_is_required_and_verified(self):
        path, stage, completion = self.production_chunk("bad-completion", [self.row("document")])
        payload = read_receipt(stage)
        mutations = [
            {"object_complete": False},
            {"object_completion": None},
            {"object_completion": {**payload["object_completion"], "receipt_sha256": "0" * 64}},
            {"chunk_index": 1},
            {"chunk_id": "other-chunk"},
        ]
        for index, change in enumerate(mutations):
            write_receipt(stage, {**payload, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.ingest(path, f"bad-{index}", buckets=1)
        original = read_receipt(completion)
        write_receipt(completion, {**original, "reblock_complete": False})
        write_receipt(stage, {
            **payload, "object_completion": {
                **payload["object_completion"], "receipt_sha256": digest_json(read_receipt(completion)),
            },
        })
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.ingest(path, "not-complete", buckets=1)
        self.assertFalse(list((self.output / "batches").glob("*/COMMITTED.json")))

    def test_signature_generation_accepts_the_production_chunk_gate(self):
        path, stage, _ = self.production_chunk("signature-chunk", [self.row("document")])
        output = self.root / "production-signatures"
        receipt = generate_signatures(
            [path], output, batch_id="production", snapshot="2026-09",
            semantic_namespace="web", stage_receipt_path=stage,
            stage_receipt_sha256=digest_json(read_receipt(stage)),
            stage_receipt_file_sha256=sha256_file(stage),
        )
        self.assertEqual(receipt["documents"], 1)
        self.assertEqual(receipt["stage_receipts"][0]["sha256"], digest_json(read_receipt(stage)))
        self.assertEqual(receipt["stage_receipts"][0]["file_sha256"], sha256_file(stage))
        self.assertTrue(Path(receipt["stage_receipts"][0]["snapshot"]["path"]).exists())

    def test_real_reblock_and_preparation_receipts_feed_dedup_without_schema_twins(self):
        from metis_data17.prep import prepare_chunk, reblock_object
        from tests.test_metis17_prep import Metis17PreparationTests

        fixture = Metis17PreparationTests("test_chunk_jobs_cover_every_reblocked_record_exactly_once")
        fixture.setUp()
        try:
            texts = [fixture.SAFE + " Variant A.", fixture.SAFE + " Variant B."]
            source = [{"text": text} for text in (texts[0], texts[1], fixture.HOLDOUT, *texts)]
            spec, raw, output = fixture._object(source, wire_format="jsonl_gzip")
            normalized = reblock_object(spec, raw, output, {**fixture.config, "output_chunk_bytes": 2000})
            self.assertGreater(len(normalized["chunks"]), 1)
            accepted = 0
            for index, chunk in enumerate(reversed(normalized["chunks"])):
                prepared = prepare_chunk(
                    fixture.root / chunk["path"], fixture.root / "chunk-jobs", fixture.config,
                )
                self.assertEqual(prepared["status"], "ELIGIBLE")
                sealed = fixture.root / prepared["receipt_path"]
                stage = (
                    fixture.root / "chunk-jobs" / "chunks" / prepared["chunk_id"] / "PREP_COMPLETE.json"
                    if index % 2 else sealed
                )
                with self.assertRaisesRegex(ValueError, "canonical seal"):
                    ingest_eligible(
                        [fixture.root / item["path"] for item in prepared["chunks"]],
                        self.output, batch_id=f"wrong-file-domain-{prepared['chunk_id']}",
                        bucket_count=3, stage_receipt_path=stage,
                        stage_receipt_sha256=sha256_file(stage),
                    )
                result = ingest_eligible(
                    [fixture.root / item["path"] for item in prepared["chunks"]],
                    self.output, batch_id=prepared["chunk_id"], bucket_count=3, batch_size=2,
                    stage_receipt_path=stage, stage_receipt_sha256=digest_json(read_receipt(stage)),
                )
                self.assertEqual(result["stage_receipts"][0]["path"], str(sealed))
                stage_seal = digest_json(read_receipt(stage))
                locator = self.output / "receipts" / "payloads" / stage_seal[:2] / f"{stage_seal}.json"
                cached = read_receipt(locator)
                write_receipt(locator, {**cached, "sha256": stage_seal})
                # An old locator's hash-domain mixup must not become an implicit file-byte pin.
                sealed_retry = ingest_eligible(
                    [fixture.root / item["path"] for item in prepared["chunks"]],
                    self.output, batch_id=prepared["chunk_id"], bucket_count=3,
                    stage_receipt_path=sealed, stage_receipt_sha256=stage_seal,
                )
                self.assertEqual(result, sealed_retry)
                self.assertEqual(read_receipt(locator)["sha256"], sha256_file(sealed))
                accepted_receipt = fixture.root / "accepted-receipts" / f"{stage_seal}.json"
                write_receipt(accepted_receipt, read_receipt(stage))
                accepted_result = ingest_eligible(
                    [fixture.root / item["path"] for item in prepared["chunks"]],
                    self.output, batch_id=prepared["chunk_id"], bucket_count=3,
                    stage_receipt_path=accepted_receipt, stage_receipt_sha256=stage_seal,
                )
                self.assertEqual(result, accepted_result)
                accepted += prepared["eligible_documents"]
            self.assertEqual(accepted, 4)
            status = dedup_status(self.output)
            self.assertEqual((status["raw_occurrences"], status["unique_winners"]), (4, 2))
            self.assertEqual({row["content_hash"] for row in iter_survivors(self.output)},
                             {hashlib.sha256(text.encode()).hexdigest() for text in texts})
        finally:
            fixture.doCleanups()

    def test_legacy_directory_lock_requires_quiescent_migration_not_ttl(self):
        lock = self.root / "foreign-lock"
        lock.mkdir()
        owner = {"host": "not-" + socket.gethostname(), "pid": 2147483647, "nonce": "foreign"}
        write_receipt(lock / "owner.json", owner)
        os.utime(lock, (1, 1))
        with self.assertRaisesRegex(RuntimeError, "quiescent migration"):
            with metadata_lock(lock, timeout=0.01):
                self.fail("A foreign-host lock must never be age-reclaimed")
        self.assertEqual(read_receipt(lock / "owner.json"), owner)

    def test_file_lock_recovers_after_worker_death_without_host_or_age_checks(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        path = self.root / "recoverable.lock"
        worker = context.Process(target=_hold_metadata_lock, args=(str(path), child))
        worker.start()
        try:
            self.assertTrue(parent.poll(15))
            self.assertEqual(parent.recv(), "locked")
            with self.assertRaises(TimeoutError):
                with metadata_lock(path, timeout=0.02):
                    self.fail("A live holder must retain its lock")
            worker.terminate()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            with patch("socket.gethostname", return_value="replacement-compute-node"):
                with metadata_lock(path, timeout=0.1):
                    self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)
            parent.close()
            child.close()

    def test_failed_foreign_reader_does_not_pin_metadata_forever(self):
        paths = [self.shard(f"reader-crash-{index}", [
            self.row(f"{index}-a", text="A", priority=index),
            self.row(f"{index}-b", text="B", priority=index),
        ]) for index in range(4)]
        for index in range(2):
            self.ingest(paths[index], str(index), buckets=1)
        old = read_receipt(self.output / "buckets/000000/CURRENT.json")["runs"][0]
        old_path = self.output / old["occurrences"]["path"]
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        worker = context.Process(target=_hold_reader_lease, args=(str(self.output), child))
        worker.start()
        try:
            self.assertTrue(parent.poll(15))
            self.assertEqual(parent.recv(), "reading")
            for index in range(2, 4):
                self.ingest(paths[index], str(index), buckets=1)
            self.assertTrue(old_path.exists())
            worker.terminate()
            worker.join(10)
            with patch("metis_data17.dedup.socket.gethostname", return_value="replacement-compute-node"):
                compact_dedup(self.output)
            self.assertFalse(old_path.exists())
            self.assertFalse(list((self.output / "readers").glob("*.json")))
        finally:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)
            parent.close()
            child.close()

    def test_local_only_lustre_locks_are_rejected(self):
        with patch.object(dedup_locks, "_mounts", return_value=(
            (self.root, "lustre", frozenset({"rw", "localflock"})),
        )):
            with self.assertRaisesRegex(RuntimeError, "distributed Lustre"):
                with metadata_lock(self.root / "unsafe.lock"):
                    self.fail("Node-local flock must not masquerade as a distributed lock")

    def test_parquet_quota_stops_growth_before_the_limit(self):
        from metis_data17.acquisition import CapacityPending

        budget = self.budget(2048)
        path = budget.root / "stream-test" / "metadata.parquet"
        with self.assertRaises(CapacityPending):
            with budget.quota("parquet-stream-test", path.parent) as quota:
                sink = dedup_runs.ParquetSink(path, pa.schema([("value", pa.binary())]), 1, quota=quota)
                try:
                    sink.append({"value": hashlib.shake_256(b"incompressible metadata").digest(65536)})
                finally:
                    with contextlib.suppress(CapacityPending):
                        sink.close()
        self.assertLessEqual(path.stat().st_size, 2048)
        self.assertLessEqual(budget.snapshot()["reserved_bytes"], 2048)

    def test_budgeted_ingest_compaction_and_signatures_are_metered_and_idempotent(self):
        budget = self.budget()
        output = budget.root / "exact"
        paths = [self.shard(f"quota-{index}", [
            self.row(f"{index}-{row}", text=f"duplicate value {row % 5}", priority=index)
            for row in range(12)
        ]) for index in range(4)]
        for index, path in enumerate(paths):
            ingest_eligible([path], output, batch_id=str(index), bucket_count=1, batch_size=3,
                            working_budget=budget)
        status = dedup_status(output)
        self.assertEqual((status["raw_occurrences"], status["unique_winners"]), (48, 5))
        self.assertTrue(all(row["priority"] == 3 for row in iter_survivors(output)))
        before = budget.snapshot()["committed_bytes"]
        ingest_eligible([paths[0]], output, batch_id="0", bucket_count=1, working_budget=budget)
        self.assertEqual(budget.snapshot()["committed_bytes"], before)
        with self.assertRaisesRegex(ValueError, "WorkingBudget"):
            ingest_eligible([paths[0]], output, batch_id="unmetered", bucket_count=1)
        with self.assertRaisesRegex(ValueError, "WorkingBudget"):
            compact_dedup(output)
        with budget.quota("wrong-parent-quota", budget.root / "wrong-parent-quota") as quota:
            with self.assertRaisesRegex(TypeError, "not an entered WorkingQuota"):
                compact_dedup(output, working_budget=quota)
        signature_root = budget.root / "signatures"
        result = generate_signatures(
            [paths[0]], signature_root, batch_id="signature", snapshot="2026-09",
            semantic_namespace="web", working_budget=budget,
        )
        self.assertEqual(result["documents"], 12)
        directories = [
            *output.glob("batches/*"), *output.glob("compaction-runs/*/*"),
            *output.glob("receipt-blobs/*/*"), *signature_root.glob("scopes/*/batches/*"),
            *signature_root.glob("receipt-blobs/*/*"),
        ]
        actual = sum(file.stat().st_size for directory in directories
                     for file in directory.rglob("*") if file.is_file())
        self.assertEqual(budget.snapshot()["committed_bytes"], actual)
        self.assertLessEqual(budget.snapshot()["reserved_bytes"], budget.snapshot()["derived_limit_bytes"])

    def test_release_limits_automatically_enable_quota_and_exhaustion_is_unpublished(self):
        from metis_data17.acquisition import CapacityPending

        budget = self.budget(2000)
        write_receipt(budget.root / "RUN.json", {"schema": "metis17.run/v1"})
        path = self.shard("quota-exhausted", [self.row("document")])
        output = budget.root / "exact"
        with self.assertRaises(CapacityPending):
            ingest_eligible([path], output, batch_id="limited", bucket_count=1)
        self.assertEqual(dedup_status(output)["unique_winners"], 0)
        self.assertFalse(list((output / "batches").glob("*/COMMITTED.json")))
        self.assertLessEqual(budget.snapshot()["reserved_bytes"], 2000)

    def test_signature_quota_exhaustion_never_publishes_completion(self):
        from metis_data17.acquisition import CapacityPending
        from metis_data17.storage import WorkingQuota

        budget = self.budget(4000)
        output = budget.root / "signatures"
        path = self.shard("signature-budget", [self.row("document")])
        original_open = WorkingQuota.open
        parquet_writes = []

        def opening(quota, target, mode="wb"):
            if Path(target).suffix == ".parquet":
                parquet_writes.append(Path(target).name)
            return original_open(quota, target, mode)

        with patch.object(WorkingQuota, "open", opening), self.assertRaises(CapacityPending):
            generate_signatures(
                [path], output, batch_id="limited", snapshot="2026-09", semantic_namespace="web",
                working_budget=budget,
            )
        self.assertEqual(set(parquet_writes), {"near.parquet", "spans.parquet"})
        self.assertFalse(list(output.glob("scopes/*/batches/*/COMMITTED.json")))
        self.assertLessEqual(budget.snapshot()["reserved_bytes"], 4000)

    def test_compaction_quota_pause_preserves_old_view_and_recovers_its_partial_run(self):
        from metis_data17.acquisition import CapacityPending

        budget = self.budget()
        output = budget.root / "exact"
        paths = [self.shard(f"compaction-budget-{index}", [
            self.row(f"{index}-{row}", text=f"shared {row}", priority=index) for row in range(10)
        ]) for index in range(2)]
        with patch.object(dedup, "compact_dedup", return_value={}):
            for index, path in enumerate(paths):
                ingest_eligible([path], output, batch_id=str(index), bucket_count=1, working_budget=budget)
        before = dedup_status(output)
        used = budget.snapshot()["committed_bytes"]
        limits = read_receipt(budget.root / "limits.json")
        write_receipt(budget.root / "limits.json", {**limits, "max_working_bytes": used + 524_289 + 100})
        with self.assertRaises(CapacityPending):
            compact_dedup(output, working_budget=budget)
        self.assertEqual(dedup_status(output)["snapshot_sha256"], before["snapshot_sha256"])
        self.assertEqual(dedup_status(output)["raw_occurrences"], 20)
        pending = output / "buckets/000000/COMPACTION_PENDING.json"
        self.assertTrue(pending.exists())
        abandoned = output / read_receipt(pending)["directory"]
        self.assertLessEqual(budget.snapshot()["reserved_bytes"], used + 100)
        write_receipt(budget.root / "limits.json", limits)
        compact_dedup(output, working_budget=budget)
        self.assertFalse(pending.exists())
        self.assertEqual(list(abandoned.iterdir()), [])
        self.assertEqual(dedup_status(output)["unique_winners"], 10)
        self.assertEqual(dedup_status(output)["active_runs"], 1)

    def test_budget_attachment_adopts_existing_metadata_once(self):
        release = self.root / "working-release"
        output = release / "exact"
        for index in range(2):
            path = self.shard(f"old-budget-{index}", [self.row(str(index), text="same", priority=index)])
            self.ingest(path, str(index), output=output, buckets=1)
        data_directories = [output / name for name in ("batches", "compactions", "receipts")]
        prior_bytes = sum(file.stat().st_size for directory in data_directories
                          for file in directory.rglob("*") if file.is_file())
        budget = self.budget()
        compact_dedup(output, working_budget=budget)
        self.assertEqual(budget.snapshot()["committed_bytes"], prior_bytes)
        compact_dedup(output, working_budget=budget)
        self.assertEqual(budget.snapshot()["committed_bytes"], prior_bytes)
        self.assertEqual(dedup_status(output)["unique_winners"], 1)

    def test_concurrent_budgeted_writers_share_the_cap_without_duplicate_credits(self):
        budget = self.budget()
        output = budget.root / "concurrent"
        paths = [self.shard(f"concurrent-quota-{index}", [
            self.row(f"{index}-{row}", text=f"same {row}", priority=index) for row in range(8)
        ]) for index in range(4)]
        arguments = [(str(path), str(output), str(index), str(budget.root))
                     for index, path in enumerate(paths)]
        arguments.append((str(paths[0]), str(output), "redelivery", str(budget.root)))
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
            admitted = list(pool.map(_ingest_budget_process, arguments))
        self.assertEqual(sum(admitted), 32)
        self.assertEqual(dedup_status(output)["raw_occurrences"], 32)
        self.assertEqual(dedup_status(output)["unique_winners"], 8)
        self.assertLessEqual(budget.snapshot()["reserved_bytes"], budget.snapshot()["derived_limit_bytes"])

    def test_budgeted_proof_blobs_support_pinned_pointer_replay(self):
        budget = self.budget()
        output = budget.root / "exact"
        path, stage, _ = self.production_chunk("quota-replay", [self.row("document")])
        payload = read_receipt(stage)
        seal = digest_json(payload)
        first = ingest_eligible(
            [path], output, batch_id="pinned", bucket_count=1, working_budget=budget,
            stage_receipt_path=stage, stage_receipt_sha256=seal,
        )
        before = budget.snapshot()["committed_bytes"]
        write_receipt(stage, {
            **payload, "status": "NORMALIZED_PENDING_DECONTAMINATION",
            "eligible": False, "training_ready": False, "chunks": [], "eligible_documents": 0,
            "pending_reasons": ["policy_replay"],
        })
        second = ingest_eligible(
            [path], output, batch_id="pinned", bucket_count=1, working_budget=budget,
            stage_receipt_path=stage, stage_receipt_sha256=seal,
        )
        self.assertEqual(first, second)
        self.assertEqual(budget.snapshot()["committed_bytes"], before)

    def test_compaction_recovery_never_deletes_an_already_published_run(self):
        budget = self.budget()
        output = budget.root / "exact"
        with patch.object(dedup, "compact_dedup", return_value={}):
            for index in range(2):
                path = self.shard(f"published-compaction-{index}", [
                    self.row(str(index), text="same", priority=index),
                ])
                ingest_eligible([path], output, batch_id=str(index), bucket_count=1, working_budget=budget)
        original = dedup._publish_bucket

        def interrupted(root, bucket, runs, **kwargs):
            original(root, bucket, runs, **kwargs)
            if any(run["commit_kind"] == "compaction" for run in runs):
                raise OSError("interrupted after publication")

        with patch.object(dedup, "_publish_bucket", side_effect=interrupted), self.assertRaises(OSError):
            compact_dedup(output, working_budget=budget)
        pending = output / "buckets/000000/COMPACTION_PENDING.json"
        self.assertTrue(pending.exists())
        published = read_receipt(output / "buckets/000000/CURRENT.json")["runs"][0]
        artifact = output / published["occurrences"]["path"]
        self.assertTrue(artifact.exists())
        compact_dedup(output, working_budget=budget)
        self.assertFalse(pending.exists())
        self.assertTrue(artifact.exists())
        self.assertEqual(dedup_status(output)["unique_winners"], 1)

    def test_scoped_signatures_are_real_and_cross_snapshot_comparison_is_separate(self):
        text = ("A sentence with eight carefully distinct informative useful words. "
                "Another sentence with eight carefully selected relevant helpful words. "
                "The final sentence includes eight quite different additional precise words.")
        path = self.shard("signatures", [self.row("web", text=text)])
        output = self.root / "signatures"
        receipts = []
        rows = []
        for snapshot in ("CC-MAIN-2026-30", "CC-MAIN-2026-38"):
            receipt = generate_signatures(
                [path], output, batch_id="signature-batch", snapshot=snapshot,
                semantic_namespace="english-web", language="en", batch_size=1,
            )
            receipts.append(receipt)
            rows.append(pq.read_table(output / receipt["artifacts"][0]["path"]).to_pylist()[0])
        self.assertEqual(rows[0]["minhash"], rows[1]["minhash"])
        self.assertEqual(len(rows[0]["minhash"]), 128 * 8)
        self.assertGreater(rows[0]["shingle_count"], 0)
        self.assertNotEqual(rows[0]["scope_id"], rows[1]["scope_id"])
        self.assertTrue(set(minhash_bands(rows[0]["minhash"], rows[0]["scope_id"])).isdisjoint(
            minhash_bands(rows[1]["minhash"], rows[1]["scope_id"])))
        self.assertTrue(all(receipt["span_signatures"] > 0 for receipt in receipts))
        self.assertTrue(all(receipt["signatures_complete"] for receipt in receipts))
        self.assertTrue(all(not receipt["decisions_complete"] for receipt in receipts))
        repeated = generate_signatures(
            [path], output, batch_id="signature-batch", snapshot="CC-MAIN-2026-30",
            semantic_namespace="english-web", language="en", batch_size=17,
        )
        self.assertEqual(repeated, receipts[0])
        status = signature_status(output)
        self.assertEqual((status["documents"], status["batches"], status["artifacts"]), (2, 2, 4))
        with self.assertRaisesRegex(ValueError, "language"):
            generate_signatures([path], output, batch_id="wrong-language",
                                snapshot="2026-38", semantic_namespace="web", language="fr")

    def test_code_and_math_signatures_preserve_semantics(self):
        texts = [
            "if x:\n    return Foo(x)\nreturn Bar(x)\n",
            "if x:\n    return Foo(x)\n    return Bar(x)\n",
            "if x:\n    return foo(x)\nreturn Bar(x)\n",
        ]
        rows = [self.row(str(index), text=text, category="code") for index, text in enumerate(texts)]
        rows.extend([
            self.row("math-upper", text="Let X = x + y. Then X is not x.", category="math"),
            self.row("math-lower", text="Let x = x + y. Then x is not x.", category="math"),
        ])
        path = self.shard("semantic", rows)
        output = self.root / "signature-semantic"
        receipt = generate_signatures([path], output, batch_id="semantic", snapshot="2026-09",
                                      semantic_namespace="raw-semantic", batch_size=2, raw_window=16)
        near = pq.read_table(output / receipt["artifacts"][0]["path"]).to_pylist()
        self.assertEqual(len({row["minhash"] for row in near}), 5)
        spans = pq.read_table(output / receipt["artifacts"][1]["path"]).to_pylist()
        raw_files = [row for row in spans if row["kind"].endswith(".file.raw")]
        self.assertEqual(len(raw_files), 5)
        self.assertEqual(len({row["signature"] for row in raw_files}), 5)
        self.assertEqual({row["token_policy"] for row in near}, {"raw-semantic"})
        self.assertTrue(path.exists())

    def test_minhash_matches_scalar_reference_and_bounded_chunking(self):
        text = "a b c d e f g h i j k l m n o"
        signature, count = minhash_signature(text, num_perm=12, shingle_batch_size=2)
        one_block, total = minhash_signature(text, num_perm=12, shingle_batch_size=1000)
        self.assertEqual((signature, count), (one_block, total))
        random = np.random.RandomState(16062026)
        prime = (1 << 61) - 1
        a = random.randint(1, prime, size=12, dtype=np.uint64).tolist()
        b = random.randint(0, prime, size=12, dtype=np.uint64).tolist()
        shingles = list(_shingles(text, semantic=False, n_grams=5))
        expected = [min((((value * x + y) & ((1 << 64) - 1)) % prime) for value in shingles)
                    for x, y in zip(a, b)]
        self.assertEqual(np.frombuffer(signature, dtype="<u8").tolist(), expected)
        self.assertEqual(count, 11)
        self.assertEqual(minhash_signature("too short"), (b"", 0))
        self.assertEqual(list(minhash_bands(b"", "scope")), [])

    def test_signature_redelivery_and_empty_artifacts(self):
        path = self.shard("sig-redelivery", [self.row("document")])
        output = self.root / "signature-redelivery"
        kwargs = {"snapshot": "2026-09", "semantic_namespace": "web"}
        first = generate_signatures([path], output, batch_id="first", **kwargs)
        repeated = generate_signatures([path], output, batch_id="other-worker", **kwargs)
        self.assertEqual((first["documents"], repeated["documents"]), (1, 0))
        for artifact in repeated["artifacts"]:
            self.assertEqual(pq.read_table(output / artifact["path"]).num_rows, 0)
        self.assertEqual(signature_status(output)["documents"], 1)
        different = self.shard("sig-different", [self.row("other")])
        with self.assertRaisesRegex(ValueError, "batch_id"):
            generate_signatures([different], output, batch_id="first", **kwargs)


if __name__ == "__main__":
    unittest.main()
