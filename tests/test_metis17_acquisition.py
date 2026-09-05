from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from metis_data17.acquisition import CapacityPending, DownloadFailure, IntakeBudget, download_object, file_lock, receipt_path
from metis_data17.catalogue import CatalogueWriter, catalogue_objects, resolve_source
from metis_data17.common import ObjectSpec, read_receipt, write_receipt


class Response:
    def __init__(self, blocks, *, status=200, headers=None):
        self.blocks = blocks
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size):
        for item in self.blocks:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.headers_seen = []

    def get(self, url, **kwargs):
        self.headers_seen.append(kwargs["headers"])
        return next(self.responses)


class Metis17AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.limits = {
            "max_raw_bytes": 100,
            "max_working_bytes": 1000,
            "working_reservation_factor": 4,
            "policy_and_metadata_reserve_bytes": 0,
            "filesystem_free_floor_bytes": 0,
            "max_unknown_object_bytes": 100,
        }
        self.content = b"abcde"
        self.spec = self.make_spec("first.parquet", self.content)

    def tearDown(self):
        self.tmp.cleanup()

    def make_spec(self, key, content, *, priority=90):
        return ObjectSpec.create(
            source_id="source",
            url=f"https://example.test/{key}",
            revision="pinned-revision",
            relative_key=key,
            wire_format="parquet",
            adapter="text",
            priority=priority,
            expected_bytes=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
            policy={"source_budget_bytes": 100},
        )

    def test_verified_receipt_and_idempotent_restart(self):
        session = Session([Response([self.content], headers={"Content-Length": "5"})])
        result = download_object(self.spec, self.root, self.limits, session=session)
        self.assertEqual((self.root / result.relative_path).read_bytes(), self.content)
        self.assertEqual(result.sha256, self.spec.expected_sha256)
        self.assertEqual(len(session.headers_seen), 1)
        again = download_object(self.spec, self.root, self.limits, session=session)
        self.assertEqual(result, again)
        budget = read_receipt(self.root / "state/intake-budget.json")
        self.assertEqual(budget["raw_bytes"], 5)
        self.assertEqual(budget["sources"]["source"]["objects"], 1)
        self.assertEqual(budget["inflight"], {})

    def test_interrupted_transfer_resumes_exact_offset(self):
        session = Session([
            Response([b"ab", requests.ConnectionError("interrupted")], headers={"Content-Length": "5", "ETag": "identity"}),
            Response([b"cde"], status=206, headers={"Content-Range": "bytes 2-4/5", "Content-Length": "3"}),
        ])
        with patch("metis_data17.acquisition.time.sleep"):
            receipt = download_object(self.spec, self.root, self.limits, session=session, attempts=2)
        self.assertEqual((self.root / receipt.relative_path).read_bytes(), b"abcde")
        self.assertEqual(session.headers_seen[1]["Range"], "bytes=2-")
        self.assertEqual(session.headers_seen[1]["If-Range"], "identity")
        self.assertEqual(read_receipt(self.root / "state/intake-budget.json")["network_payload_bytes"], 5)

    def test_same_size_corruption_is_not_a_completed_download(self):
        session = Session([Response([self.content], headers={"Content-Length": "5"})])
        receipt = download_object(self.spec, self.root, self.limits, session=session)
        (self.root / receipt.relative_path).write_bytes(b"wrong")
        with self.assertRaises(DownloadFailure):
            download_object(self.spec, self.root, self.limits, session=session)

    def test_range_ignored_restarts_instead_of_appending(self):
        session = Session([
            Response([b"ab", requests.ConnectionError("interrupted")], headers={"Content-Length": "5"}),
            Response([b"abcde"], headers={"Content-Length": "5"}),
        ])
        with patch("metis_data17.acquisition.time.sleep"):
            receipt = download_object(self.spec, self.root, self.limits, session=session, attempts=2)
        self.assertEqual((self.root / receipt.relative_path).read_bytes(), b"abcde")
        self.assertEqual(read_receipt(self.root / "state/intake-budget.json")["network_payload_bytes"], 7)

    def test_wrong_hash_never_publishes_readiness(self):
        session = Session([Response([b"wrong"], headers={"Content-Length": "5"})])
        with self.assertRaises(DownloadFailure):
            download_object(self.spec, self.root, self.limits, session=session, attempts=1)
        self.assertFalse(receipt_path(self.root, self.spec.object_id).exists())
        budget = read_receipt(self.root / "state/intake-budget.json")
        self.assertEqual(budget["raw_bytes"], 0)
        self.assertEqual(budget["inflight"], {})

    def test_capacity_prevents_body_consumption(self):
        response = Response([AssertionError("body must not be consumed")], headers={"Content-Length": "5"})
        limits = {**self.limits, "max_raw_bytes": 4}
        with self.assertRaises(CapacityPending):
            download_object(self.spec, self.root, limits, session=Session([response]))
        self.assertTrue(response.closed)
        self.assertFalse(receipt_path(self.root, self.spec.object_id).exists())

    def test_cross_instance_budget_reservations_do_not_overbook(self):
        limits = {**self.limits, "max_raw_bytes": 8}
        IntakeBudget(self.root, limits).reserve(self.spec, 5)
        second = self.make_spec("second.parquet", b"abcdef")
        with self.assertRaises(CapacityPending):
            IntakeBudget(self.root, limits).reserve(second, 6)
        self.assertEqual(len(read_receipt(self.root / "state/intake-budget.json")["inflight"]), 1)

    def test_lock_release_between_owner_probe_and_read_is_retried(self):
        lock = self.root / "racing.lock"
        lock.mkdir()
        owner = lock / "owner.json"
        owner.write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "nonce": "previous"}))
        original = Path.read_text
        raced = []

        def read(path, *args, **kwargs):
            if path == owner and not raced:
                raced.append(True)
                owner.unlink()
                lock.rmdir()
                raise FileNotFoundError("Previous holder released")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            with file_lock(lock):
                self.assertTrue(owner.exists())
        self.assertTrue(raced)
        self.assertFalse(lock.exists())

    def test_published_receipt_recovers_interrupted_budget_commit(self):
        budget = IntakeBudget(self.root, self.limits)
        budget.reserve(self.spec, 5)
        session = Session([Response([self.content], headers={"Content-Length": "5"})])
        receipt = download_object(self.spec, self.root, self.limits, session=session)
        state = read_receipt(budget.path)
        state["raw_bytes"] = 0
        state["sources"] = {}
        state["inflight"] = {self.spec.object_id: {"source_id": self.spec.source_id, "bytes": 5}}
        write_receipt(budget.path, state)
        second = self.make_spec("second.parquet", b"x")
        IntakeBudget(self.root, self.limits).reserve(second, 1)
        state = read_receipt(budget.path)
        self.assertEqual(state["raw_bytes"], receipt.byte_count)
        self.assertEqual(state["sources"]["source"]["objects"], 1)
        self.assertNotIn(self.spec.object_id, state["inflight"])

    def test_priority_does_not_change_raw_identity(self):
        better = self.make_spec("first.parquet", self.content, priority=100)
        self.assertEqual(better.object_id, self.spec.object_id)
        self.assertEqual(ObjectSpec.from_dict(better.to_dict()), better)

    def test_invalid_checksum_and_credentials_are_rejected(self):
        fields = dict(
            source_id="s", revision="r", relative_key="a", wire_format="parquet",
            adapter="text", priority=1,
        )
        with self.assertRaises(ValueError):
            ObjectSpec.create(url="https://user:password@example.test/a", **fields)
        with self.assertRaises(ValueError):
            ObjectSpec.create(url="https://example.test/a", expected_sha256="not-a-hash", **fields)

    def test_source_catalogue_pages_are_idempotent_and_complete(self):
        source = {"id": "source", "expected_inventory_bytes": 5, "expected_objects": 1}
        writer = CatalogueWriter(self.root, source, page_size=1)
        writer.add(self.spec)
        result = writer.seal()
        writer2 = CatalogueWriter(self.root, source, page_size=1)
        writer2.add(self.spec)
        self.assertEqual(result, writer2.seal())
        with self.assertRaises(RuntimeError):
            writer2.add(self.spec)

    def test_catalogue_known_inventory_shortfall_is_not_success(self):
        writer = CatalogueWriter(self.root, {"id": "source", "expected_inventory_bytes": 10})
        writer.add(self.spec)
        with self.assertRaises(RuntimeError):
            writer.seal()

    def test_reconstruction_driver_cannot_resolve(self):
        source = {"id": "forbidden", "kind": "github_repositories"}
        with self.assertRaises(ValueError):
            resolve_source(self.root, source)


if __name__ == "__main__":
    unittest.main()
