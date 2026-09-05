from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from metis_data17.acquisition import CapacityPending, DownloadFailure, DownloadPaused, IntakeBudget, download_object, file_lock, receipt_path
from metis_data17.catalogue import CatalogueWriter, catalogue_objects, resolve_source, select_bounded_objects
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

    def test_requested_stop_retains_partial_and_resumes_without_redownload(self):
        stop = threading.Event()

        def blocks():
            yield b"ab"
            stop.set()
            yield b"cde"

        first = Session([Response(blocks(), headers={"Content-Length": "5", "ETag": "identity"})])
        with self.assertRaises(DownloadPaused):
            download_object(self.spec, self.root, self.limits, session=first, stop_event=stop)
        self.assertFalse(receipt_path(self.root, self.spec.object_id).exists())
        self.assertIn(self.spec.object_id, read_receipt(self.root / "state/intake-budget.json")["inflight"])
        second = Session([Response([b"cde"], status=206, headers={"Content-Range": "bytes 2-4/5"})])
        receipt = download_object(self.spec, self.root, self.limits, session=second)
        self.assertEqual(second.headers_seen[0]["Range"], "bytes=2-")
        self.assertEqual((self.root / receipt.relative_path).read_bytes(), self.content)
        self.assertEqual(read_receipt(self.root / "state/intake-budget.json")["network_payload_bytes"], 8)

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

    def test_masked_publisher_digest_blocks_source_without_disabling_verification(self):
        source = {
            "id": "masked", "kind": "hf", "repo": "publisher/dataset", "revision": "a" * 40,
            "allow_patterns": ["*.jsonl.gz"], "budget_bytes": 100, "adapter": "text",
            "priority": 50, "policy": {},
        }
        item = SimpleNamespace(path="train.jsonl.gz", size=5, lfs={"sha256": "*" * 64})
        api = SimpleNamespace(
            dataset_info=lambda *args, **kwargs: SimpleNamespace(sha=source["revision"]),
            list_repo_tree=lambda *args, **kwargs: iter([item]),
        )
        with patch("metis_data17.catalogue.HfApi", return_value=api), patch(
            "metis_data17.catalogue.RepoFile", SimpleNamespace,
        ):
            with self.assertRaisesRegex(RuntimeError, "masked or invalid SHA-256"):
                resolve_source(self.root, source)
        self.assertFalse(any(self.root.glob("catalogue/*/SOURCE_COMPLETE.json")))

    def test_incomplete_catalogue_can_be_reconciled_without_changing_old_objects(self):
        source = {"id": "source", "kind": "hf", "expected_inventory_bytes": 10, "formats": ["parquet"]}
        corrected = {**source, "formats": ["parquet", "jsonl"]}
        second = self.make_spec("second.jsonl", b"12345")
        with patch("metis_data17.catalogue._resolve_hf",
                   side_effect=lambda root, source, writer: writer.add(self.spec)):
            with self.assertRaises(RuntimeError):
                resolve_source(self.root, source)
        active = self.root / "catalogue/active/source.json"
        previous = active.read_bytes()

        def resolve(root, source, writer):
            writer.add(self.spec)
            writer.add(second)

        with patch("metis_data17.catalogue._resolve_hf", side_effect=resolve):
            with self.assertRaises(RuntimeError):
                resolve_source(self.root, corrected)
            self.assertEqual(active.read_bytes(), previous)
            result = resolve_source(self.root, corrected, reconcile_incomplete=True)
        self.assertEqual(result["known_bytes"], 10)
        self.assertEqual({spec.object_id for spec in catalogue_objects(self.root)},
                         {self.spec.object_id, second.object_id})
        descriptor = read_receipt(active)
        proof = read_receipt(self.root / descriptor["directory"] / "RECONCILIATION.json")
        self.assertEqual((proof["preserved_objects"], proof["added_objects"]), (1, 1))
        self.assertEqual(resolve_source(self.root, corrected), result)
        with self.assertRaises(RuntimeError):
            resolve_source(self.root, {**corrected, "formats": ["different"]}, reconcile_incomplete=True)

    def test_inventory_correction_cannot_drop_previously_published_objects(self):
        source = {"id": "source", "kind": "hf", "expected_inventory_bytes": 10}
        with patch("metis_data17.catalogue._resolve_hf",
                   side_effect=lambda root, source, writer: writer.add(self.spec)):
            with self.assertRaises(RuntimeError):
                resolve_source(self.root, source)
        active = self.root / "catalogue/active/source.json"
        before = active.read_bytes()
        different = self.make_spec("different.parquet", b"12345")
        with patch("metis_data17.catalogue._resolve_hf",
                   side_effect=lambda root, source, writer: writer.add(different)):
            with self.assertRaisesRegex(RuntimeError, "preserve every existing object"):
                resolve_source(self.root, {**source, "expected_inventory_bytes": 5},
                               reconcile_incomplete=True)
        self.assertEqual(active.read_bytes(), before)

    def test_bounded_selection_accounts_for_every_candidate_exactly_once(self):
        objects = [self.make_spec(f"CC-MAIN-202{i}/part.parquet", b"x" * size)
                   for i, size in enumerate((5, 4, 8, 3))]
        selected, omitted = select_bounded_objects(objects, 9)
        self.assertEqual(sum(spec.expected_bytes for spec in selected), 7)
        self.assertEqual(selected, [objects[3], objects[1]])
        self.assertEqual({spec.object_id for spec in selected + omitted},
                         {spec.object_id for spec in objects})
        self.assertEqual(len(selected) + len(omitted), len(objects))
        with self.assertRaises(ValueError):
            select_bounded_objects(objects + objects[:1], 9)

    def test_reconstruction_driver_cannot_resolve(self):
        source = {"id": "forbidden", "kind": "github_repositories"}
        with self.assertRaises(ValueError):
            resolve_source(self.root, source)

    def test_unknown_hplt_selection_cannot_silently_enumerate_every_language(self):
        with patch("metis_data17.catalogue._cached_metadata") as metadata:
            with self.assertRaisesRegex(ValueError, "exact English WDS bucket"):
                resolve_source(self.root, {"id": "hplt", "kind": "hplt", "selection": "english-typo"})
        metadata.assert_not_called()

    def test_hplt_english_bucket_selection_is_disjoint_and_complete(self):
        import hashlib
        from metis_data17.catalogue import _resolve_hplt

        names = ["10_a.jsonl.zst", "9_a.jsonl.zst", "9_b.jsonl.zst", "8_a.jsonl.zst"]
        manifest = ("\n".join(json.dumps({
            "name": language, "md5": f"https://data.hplt-project.org/{language}.md5",
            "urls": [f"https://data.hplt-project.org/{language}/{name}" for name in names],
        }) for language in ("eng_Latn", "deu_Latn")) + "\n").encode()
        checksums = "".join(f"{'a' * 32}  {name}\n" for name in names).encode()
        base = {
            "id": "english", "kind": "hplt", "manifest_url": "https://data.hplt-project.org/manifest",
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "priority": 80, "budget_bytes": 1000, "policy": {},
        }
        selected = {}
        for bucket in (10, 9, 8):
            source = {**base, "id": f"english{bucket}", "selection": f"english-wds{bucket}"}
            writer = CatalogueWriter(self.root, source)
            with patch("metis_data17.catalogue._cached_metadata",
                       side_effect=lambda root, url: manifest if url.endswith("manifest") else checksums):
                _resolve_hplt(self.root, source, writer)
            writer.seal()
            selected[bucket] = {record["relative_key"] for page in writer.directory.glob("page-*.json")
                                for record in read_receipt(page)["objects"]}
        self.assertEqual(selected[9], {"eng_Latn/9_a.jsonl.zst", "eng_Latn/9_b.jsonl.zst"})
        self.assertEqual(sum(map(len, selected.values())), len(set.union(*selected.values())))
        self.assertEqual(set.union(*selected.values()), {f"eng_Latn/{name}" for name in names})


if __name__ == "__main__":
    unittest.main()
