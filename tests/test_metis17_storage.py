from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow.parquet as pq

from metis_data17 import prep, storage
from metis_data17.acquisition import CapacityPending
from metis_data17.common import read_receipt, sha256_file, write_receipt
from metis_data17.storage import WorkingBudget
from tests import test_metis17_prep as prep_fixtures


def _limits(root: Path, derived: int, *, raw: int = 100, metadata: int = 20, floor: int = 0) -> None:
    write_receipt(root / "limits.json", {
        "max_raw_bytes": raw, "max_working_bytes": raw + metadata + derived,
        "policy_and_metadata_reserve_bytes": metadata,
        "filesystem_free_floor_bytes": floor, "capacity_confirmation": "pending",
    })


def _tree_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


class Metis17StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = (Path.cwd() / ".metis17-storage-tests" / uuid.uuid4().hex).resolve()
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        _limits(self.root, 10)

    def _child(self, code: str, *args: str) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(self.root), *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        def cleanup() -> None:
            if process.poll() is None:
                process.kill()
            process.communicate()

        self.addCleanup(cleanup)
        return process

    def _canonical_record(self) -> dict:
        record = {name: "" for name in prep.CANONICAL_COLUMNS}
        record.update(
            priority=0, quality_score=-1.0, character_count=1, text="x",
            metadata_json=json.dumps({"row_index": 1, "component": "text"}),
        )
        return record

    def test_chunk_writer_never_retries_a_failed_native_row_group(self) -> None:
        failure = CapacityPending("injected quota refusal")
        native = Mock()
        native.write_table.side_effect = failure
        directory = self.root / "derived"
        directory.mkdir()
        writer = prep._ChunkWriter(directory, directory, self.root, 1_000_000, 1)
        with patch.object(pq, "ParquetWriter", return_value=native):
            with self.assertRaises(CapacityPending):
                writer.add(self._canonical_record(), source_row=1)
            self.assertEqual(writer.buffer, [])
            for operation in (writer._flush, writer.finish,
                              lambda: writer.add(self._canonical_record(), source_row=1)):
                with self.assertRaises(CapacityPending) as caught:
                    operation()
                self.assertIs(caught.exception, failure)
            self.assertEqual(native.write_table.call_count, 1)
            self.assertEqual(writer.total_records, 1)
            native.close.assert_not_called()
            writer.abort()
            writer.abort()
            native.close.assert_called_once()
            with self.assertRaises(CapacityPending):
                writer.finish()

    def test_chunk_writer_footer_failure_cannot_become_an_empty_success_on_retry(self) -> None:
        failure = CapacityPending("injected footer refusal")
        native = Mock()
        native.close.side_effect = failure
        published = Mock()
        directory = self.root / "derived"
        directory.mkdir()
        writer = prep._ChunkWriter(directory, directory, self.root, 1_000_000, 1, on_chunk=published)
        with patch.object(pq, "ParquetWriter", return_value=native):
            writer.add(self._canonical_record(), source_row=1)
            with self.assertRaises(CapacityPending):
                writer.finish()
            with self.assertRaises(CapacityPending):
                writer.finish()
            with self.assertRaises(CapacityPending):
                writer.add(self._canonical_record(), source_row=1)
            native.close.assert_called_once()
            self.assertEqual(native.write_table.call_count, 1)
            published.assert_not_called()
            writer.abort()

    def test_real_arrow_failure_is_latched_before_any_row_group_retry(self) -> None:
        _limits(self.root, 8)
        budget = WorkingBudget(self.root)
        with budget.quota("fixture", self.root / "derived") as quota:
            writer = prep._ChunkWriter(quota.directory, quota.directory, self.root, 1_000_000, 1, quota=quota)
            try:
                with self.assertRaises(CapacityPending):
                    writer.add(self._canonical_record(), source_row=1)
                self.assertEqual(writer.buffer, [])
                native = writer.writer
                self.assertIsNotNone(native)
                with self.assertRaises(CapacityPending):
                    writer.finish()
                self.assertIs(writer.writer, native)
                self.assertLessEqual((quota.directory / "part-000000.parquet").stat().st_size, 8)
            finally:
                writer.abort()
        self.assertEqual(budget.snapshot()["reserved_bytes"], 8)

    def test_raw_cap_and_metadata_are_reserved_before_any_derived_write(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=8)
        self.assertEqual(budget.snapshot()["derived_limit_bytes"], 10)
        self.assertEqual(budget.snapshot()["capacity_confirmation"], "pending")
        directory = self.root / "derived"
        with budget.quota("fixture", directory) as quota:
            with quota.open(directory / "data") as writer:
                writer.write(b"123456")
                self.assertEqual(quota.reserved_bytes, 8)
                with self.assertRaises(CapacityPending):
                    writer.write(b"78901")
                self.assertEqual((directory / "data").read_bytes(), b"123456")
        self.assertEqual(budget.snapshot()["reserved_bytes"], 8)
        with budget.quota("fixture", directory) as quota:
            self.assertEqual(quota.used_bytes, 6)
            self.assertEqual(quota.reserved_bytes, 6)
            with quota.open("data", "ab") as writer:
                writer.write(b"7890")
                self.assertEqual((directory / "data").stat().st_size, 10)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 10)
        self.assertEqual(budget.snapshot()["committed_bytes"], 10)

    def test_explicit_reservation_is_exact_and_enforces_its_own_hard_ceiling(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=8)
        with budget.quota("fixture", self.root / "derived") as quota:
            self.assertEqual(quota.reserve(6), 6)
            self.assertEqual(quota.byte_limit, 6)
            state = budget.snapshot()
            held = quota.reserved_bytes - quota.used_bytes
            used = state["max_raw_bytes"] + state["policy_and_metadata_reserve_bytes"] + state["committed_bytes"]
            other = state["reserved_bytes"] - state["committed_bytes"] - held
            self.assertEqual(used + other + held, 126)
            self.assertLessEqual(used + other + held, state["max_working_bytes"])
            with quota.open("data") as writer:
                with self.assertRaises(ValueError):
                    quota.reserve(6)
                writer.write(b"123456")
                with self.assertRaises(CapacityPending):
                    writer.write(b"7")
            self.assertEqual((quota.directory / "data").read_bytes(), b"123456")
            self.assertEqual(quota.reserve(6), 6)
            with self.assertRaises(ValueError):
                quota.reserve(7)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 6)

    def test_preallocation_blocks_competitors_before_output_and_releases_unused_credit_on_success(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=8)
        with budget.quota("first", self.root / "first") as first:
            self.assertEqual(first.reserve(9), 9)
            with budget.quota("second", self.root / "second") as second:
                with self.assertRaises(CapacityPending):
                    second.reserve(2)
                self.assertIsNone(second.byte_limit)
                self.assertEqual(second.reserve(1), 1)
                self.assertEqual(budget.snapshot()["reserved_bytes"], 10)
            self.assertEqual(list(first.directory.iterdir()), [])
            self.assertEqual(budget.snapshot()["reserved_bytes"], 9)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 0)
        with budget.quota("second", self.root / "second") as second:
            self.assertEqual(second.reserve(10), 10)
            second.write_bytes(Path("data"), b"1234567890")
        self.assertEqual(budget.snapshot()["reserved_bytes"], 10)

    def test_reservation_total_includes_existing_namespace_bytes_without_double_charging(self) -> None:
        directory = self.root / "derived"
        directory.mkdir()
        (directory / "existing").write_bytes(b"123456")
        budget = WorkingBudget(self.root, allocation_bytes=8)
        with budget.quota("fixture", directory) as quota:
            self.assertEqual(quota.used_bytes, 6)
            with self.assertRaises(CapacityPending):
                quota.reserve(5)
            self.assertEqual(quota.reserve(9), 9)
            self.assertEqual(quota.reserved_bytes - quota.used_bytes, 3)
            with quota.open("new") as writer:
                writer.write(b"789")
            self.assertEqual(quota.used_bytes, 9)
            self.assertEqual(budget.snapshot()["reserved_bytes"], 9)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 9)

    def test_one_oversized_write_and_sparse_growth_are_refused_before_growth(self) -> None:
        with WorkingBudget(self.root).quota("fixture", self.root / "derived") as quota:
            with quota.open("data") as writer:
                writer.seek(9)
                with self.assertRaises(CapacityPending):
                    writer.write(b"12")
                self.assertEqual((quota.directory / "data").stat().st_size, 0)
        with WorkingBudget(self.root).quota("fixture", self.root / "derived") as quota:
            with quota.open("data") as writer:
                writer.seek(7)
                writer.write(b"123")
                self.assertEqual(quota.used_bytes, 10)
                with self.assertRaises(CapacityPending):
                    writer.truncate(11)
                self.assertEqual((quota.directory / "data").stat().st_size, 10)

    def test_truncation_overwrites_and_unlink_only_refund_removed_bytes(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        directory = self.root / "derived"
        with budget.quota("fixture", directory) as quota:
            with quota.open("data") as writer:
                writer.write(b"1234567890")
                writer.seek(2)
                writer.write(b"xx")
                self.assertEqual(quota.used_bytes, 10)
                with self.assertRaises(ValueError):
                    quota.unlink(directory / "data")
                with self.assertRaises(ValueError):
                    quota.reconcile()
                writer.truncate(4)
                self.assertEqual(quota.used_bytes, 4)
                self.assertEqual(quota.reserved_bytes, 4)
            self.assertEqual((directory / "data").read_bytes(), b"12xx")
            with patch.object(storage.os, "unlink", side_effect=PermissionError("fixture")):
                with self.assertRaises(PermissionError):
                    quota.unlink(directory / "data")
            self.assertEqual(budget.snapshot()["reserved_bytes"], 4)
            self.assertEqual((directory / "data").stat().st_size, 4)
            with quota.open("data", "wb") as writer:
                self.assertEqual((directory / "data").stat().st_size, 0)
                self.assertEqual(quota.used_bytes, 0)
                self.assertEqual(quota.reserved_bytes, 0)
                writer.write(b"abc")
            quota.unlink(directory / "data")
            self.assertEqual(budget.snapshot()["reserved_bytes"], 0)

    def test_replacing_files_preserves_or_refunds_only_real_net_size(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        with budget.quota("fixture", self.root / "derived") as quota:
            with quota.open("first") as writer:
                writer.write(b"1234")
            with quota.open("second") as writer:
                writer.write(b"abc")
            quota.replace(quota.directory / "first", quota.directory / "renamed")
            self.assertEqual(quota.used_bytes, 7)
            quota.replace(quota.directory / "renamed", quota.directory / "second")
            self.assertEqual(quota.used_bytes, 4)
            self.assertEqual(quota.reserved_bytes, 4)
            self.assertEqual((quota.directory / "second").read_bytes(), b"1234")
            quota.unlink(quota.directory / "second")
            self.assertEqual(quota.used_bytes, 0)

    def test_atomic_bytes_and_receipts_match_common_serialization_and_refund_replaced_files(self) -> None:
        _limits(self.root, 4096)
        budget = WorkingBudget(self.root, allocation_bytes=1)
        receipt = {"schema": "fixture/v1", "description": "é漢字", "items": [1, 2, 3]}
        reference = self.root / "reference.json"
        write_receipt(reference, receipt)
        with budget.quota("fixture", self.root / "derived") as quota:
            quota.write_bytes(Path("snapshot"), b"frozen bytes")
            quota.write_receipt(Path("READY.json"), receipt)
            target = quota.directory / "READY.json"
            self.assertEqual(target.read_bytes(), reference.read_bytes())
            self.assertEqual(read_receipt(target), receipt)
            expected = len(b"frozen bytes") + target.stat().st_size
            self.assertEqual(quota.used_bytes, expected)
            self.assertEqual(quota.reserved_bytes, expected)
            updated = {**receipt, "items": [4]}
            quota.write_receipt(target, updated)
            self.assertEqual(read_receipt(target), updated)
            self.assertEqual(quota.used_bytes, _tree_bytes(quota.directory))
            self.assertEqual(quota.reserved_bytes, _tree_bytes(quota.directory))
            self.assertEqual(sorted(path.name for path in quota.directory.iterdir()),
                             ["READY.json", "snapshot"])

    def test_atomic_replacement_reserves_both_copies_and_preserves_old_file_on_exhaustion(self) -> None:
        _limits(self.root, 7)
        budget = WorkingBudget(self.root, allocation_bytes=1)
        with budget.quota("fixture", self.root / "derived") as quota:
            quota.write_bytes(Path("snapshot"), b"old!")
            with self.assertRaises(CapacityPending):
                quota.write_bytes(Path("snapshot"), b"new!")
            self.assertEqual((quota.directory / "snapshot").read_bytes(), b"old!")
            self.assertEqual(quota.used_bytes, 4)
            self.assertLessEqual(_tree_bytes(quota.directory), 7)
            quota.write_bytes(Path("snapshot"), b"ok")
            self.assertEqual((quota.directory / "snapshot").read_bytes(), b"ok")
            self.assertEqual(quota.reserved_bytes, 2)
            self.assertEqual([path.name for path in quota.directory.iterdir()], ["snapshot"])

    def test_atomic_byte_writer_handles_short_writes_without_publishing_truncation(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        original = storage.QuotaWriter.write

        def short_write(writer, data):
            return original(writer, data[:2])

        with budget.quota("fixture", self.root / "derived") as quota:
            with patch.object(storage.QuotaWriter, "write", short_write):
                quota.write_bytes(Path("snapshot"), b"1234567")
            self.assertEqual((quota.directory / "snapshot").read_bytes(), b"1234567")
            self.assertEqual(quota.used_bytes, 7)
            with patch.object(storage.QuotaWriter, "write", return_value=0), self.assertRaises(OSError):
                quota.write_bytes(Path("snapshot"), b"new")
            self.assertEqual((quota.directory / "snapshot").read_bytes(), b"1234567")

    def test_atomic_helpers_enforce_namespace_and_receipt_seal_contract(self) -> None:
        budget = WorkingBudget(self.root)
        with budget.quota("fixture", self.root / "derived") as quota:
            with self.assertRaises(ValueError):
                quota.write_bytes(self.root / "outside", b"data")
            with self.assertRaises(ValueError):
                quota.write_receipt(Path("READY.json"), {"receipt_sha256": "already sealed"})
            self.assertEqual(list(quota.directory.iterdir()), [])

    def test_atomic_staging_crash_reuses_one_sibling_without_double_charging(self) -> None:
        code = """
import os, sys
from pathlib import Path
from metis_data17.storage import WorkingBudget
root = Path(sys.argv[1])
budget = WorkingBudget(root, allocation_bytes=1)
with budget.quota("fixture", root / "derived") as quota:
    quota.write_bytes(Path("snapshot"), b"old")
with budget.quota("fixture", root / "derived") as quota:
    quota.replace = lambda source, destination: os._exit(23)
    quota.write_bytes(Path("snapshot"), b"new!")
"""
        process = self._child(code)
        _, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 23, stderr)
        self.assertEqual((self.root / "derived" / "snapshot").read_bytes(), b"old")
        budget = WorkingBudget(self.root, allocation_bytes=1)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 7)
        with budget.quota("fixture", self.root / "derived") as quota:
            self.assertEqual(quota.used_bytes, 7)
            quota.write_bytes(Path("snapshot"), b"new!")
            self.assertEqual(quota.used_bytes, 4)
            self.assertEqual((quota.directory / "snapshot").read_bytes(), b"new!")
            self.assertEqual([path.name for path in quota.directory.iterdir()], ["snapshot"])
        self.assertEqual(budget.snapshot()["reserved_bytes"], 4)

    def test_write_pins_mutable_buffer_size_until_reserved_io_finishes(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        payload = bytearray(b"1234")
        with budget.quota("fixture", self.root / "derived") as quota:
            original = quota._reserve

            def resizing_during_reservation(growth):
                with self.assertRaises(BufferError):
                    payload.extend(b"unreserved bytes")
                return original(growth)

            with quota.open("data") as writer:
                with patch.object(quota, "_reserve", side_effect=resizing_during_reservation):
                    self.assertEqual(writer.write(payload), 4)
                payload.extend(b"5678")
                with self.assertRaises(CapacityPending):
                    writer.write(payload)
                payload.extend(b"9")
                self.assertEqual((quota.directory / "data").read_bytes(), b"1234")
                self.assertEqual(quota.used_bytes, 4)

    def test_stream_completes_partial_os_writes_before_reporting_success(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        with budget.quota("fixture", self.root / "derived") as quota:
            with quota.open("data") as writer:
                original = writer._file
                short = Mock(wraps=original)
                short.write.side_effect = lambda data: original.write(data[:2])
                with patch.object(writer, "_file", short):
                    self.assertEqual(writer.write(b"1234567"), 7)
                self.assertEqual(short.write.call_count, 4)
            self.assertEqual((quota.directory / "data").read_bytes(), b"1234567")
            self.assertEqual(quota.used_bytes, 7)

    def test_external_cleanup_cannot_refund_still_open_files(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        with budget.quota("fixture", self.root / "derived") as quota:
            writer = quota.open("data")
            writer.write(b"123456")
            (quota.directory / "data").unlink()
            with self.assertRaises(ValueError):
                quota.reconcile()
            self.assertEqual(budget.snapshot()["reserved_bytes"], 6)
            writer.close()
            quota.reconcile()
            self.assertEqual(budget.snapshot()["reserved_bytes"], 0)

    def test_path_confinement_covers_symlinks_hardlinks_and_prefix_collisions(self) -> None:
        budget = WorkingBudget(self.root)
        outside = self.root / "other"
        outside.write_bytes(b"unchanged")
        directory = self.root / "derived"
        for bad in (self.root, self.root / "raw", self.root / "state", self.root / "../escape"):
            with self.subTest(directory=bad), self.assertRaises(ValueError):
                budget.quota("bad", bad)
        with budget.quota("fixture", directory) as quota:
            for bad in (outside, Path("../other"), self.root / "derived-sibling" / "data"):
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    quota.open(bad)
            link = directory / "link"
            link.symlink_to(outside)
            with self.assertRaises((OSError, ValueError)):
                quota.open(link)
            link.unlink()
            link.symlink_to(self.root, target_is_directory=True)
            with self.assertRaises((OSError, ValueError)):
                quota.open(link / "other")
            link.unlink()
            os.link(outside, link)
            with self.assertRaises(ValueError):
                quota.open(link)
            link.unlink()
            self.assertEqual(outside.read_bytes(), b"unchanged")
        (directory / "link").symlink_to(outside)
        with self.assertRaises(ValueError):
            with budget.quota("fixture", directory):
                pass

    def test_namespace_and_directory_claims_are_exclusive_and_permanent(self) -> None:
        budget = WorkingBudget(self.root)
        directory = self.root / "derived"
        with budget.quota("fixture", directory):
            with self.assertRaises(BlockingIOError):
                with budget.quota("fixture", directory):
                    pass
        for name, path in (
            ("fixture", self.root / "different"),
            ("different", directory),
            ("child", directory / "child"),
        ):
            with self.subTest(name=name, path=path), self.assertRaises(ValueError):
                with budget.quota(name, path):
                    pass
        with budget.quota("leaf", self.root / "parent" / "leaf"):
            pass
        with self.assertRaises(ValueError):
            with budget.quota("parent", self.root / "parent"):
                pass

    def test_existing_namespace_is_charged_once_without_scanning_unrelated_data(self) -> None:
        directory = self.root / "derived"
        directory.mkdir()
        (directory / "existing").write_bytes(b"123456")
        (self.root / "unrelated").write_bytes(b"x" * 1000)
        budget = WorkingBudget(self.root)
        for _ in range(3):
            with budget.quota("fixture", directory) as quota:
                self.assertEqual(quota.used_bytes, 6)
                self.assertEqual(budget.snapshot()["reserved_bytes"], 6)
        self.assertEqual(budget.snapshot()["namespaces"], 1)

    def test_preexisting_overage_is_recorded_before_refusing_other_writers(self) -> None:
        directory = self.root / "derived"
        directory.mkdir()
        (directory / "existing").write_bytes(b"x" * 11)
        budget = WorkingBudget(self.root)
        with self.assertRaises(CapacityPending):
            with budget.quota("fixture", directory):
                pass
        self.assertEqual(budget.snapshot()["reserved_bytes"], 11)
        with self.assertRaises(CapacityPending):
            with budget.quota("other", self.root / "other"):
                pass
        self.assertEqual((directory / "existing").stat().st_size, 11)

    def test_allocation_coordination_is_coarse_and_total_does_not_list_namespaces(self) -> None:
        _limits(self.root, 4096)
        budget = WorkingBudget(self.root, allocation_bytes=128)
        with budget.quota("fixture", self.root / "derived") as quota:
            before = budget.snapshot()["sequence"]
            with quota.open("data") as writer:
                for _ in range(1024):
                    writer.write(b"x")
            self.assertEqual(budget.snapshot()["sequence"] - before, 8)
        for index in range(20):
            with budget.quota(f"empty:{index}", self.root / f"empty-{index}"):
                pass
        state = read_receipt(budget.path)
        self.assertEqual(state["namespaces"], 21)
        self.assertEqual(state["reserved_bytes"], 1024)
        self.assertLess(budget.path.stat().st_size, 512)
        self.assertNotIn("fixture", json.dumps(state))

    def test_free_floor_includes_other_namespaces_unspent_growth(self) -> None:
        _limits(self.root, 100, floor=5)
        budget = WorkingBudget(self.root, allocation_bytes=8)
        usage = shutil.disk_usage(self.root)
        with budget.quota("first", self.root / "first") as first:
            with first.open("data") as writer:
                with patch.object(storage.shutil, "disk_usage", return_value=usage._replace(free=135)):
                    writer.write(b"x")
                with budget.quota("second", self.root / "second") as second:
                    with second.open("data") as other:
                        with patch.object(storage.shutil, "disk_usage", return_value=usage._replace(free=134)):
                            with self.assertRaises(CapacityPending):
                                other.write(b"1234")
            self.assertEqual(first.reserved_bytes, 8)
        with budget.quota("second", self.root / "second") as second:
            with second.open("data") as other:
                with patch.object(storage.shutil, "disk_usage", return_value=usage._replace(free=134)):
                    other.write(b"1234")

    def test_published_raw_bytes_do_not_reserve_the_same_physical_space_twice(self) -> None:
        _limits(self.root, 10, floor=5)
        budget = WorkingBudget(self.root, allocation_bytes=1)
        write_receipt(self.root / "state" / "intake-budget.json", {"raw_bytes": 100})
        usage = shutil.disk_usage(self.root)
        with budget.quota("fixture", self.root / "derived") as quota:
            with quota.open("data") as writer:
                with patch.object(storage.shutil, "disk_usage", return_value=usage._replace(free=30)):
                    writer.write(b"12345")
                with patch.object(storage.shutil, "disk_usage", return_value=usage._replace(free=25)):
                    with self.assertRaises(CapacityPending):
                        writer.write(b"6")
                self.assertEqual((quota.directory / "data").stat().st_size, 5)

    def test_limits_are_sealed_and_pending_approval_is_never_overridden(self) -> None:
        limits = read_receipt(self.root / "limits.json")
        write_receipt(self.root / "limits.json", {**limits, "max_working_bytes": 2_000_000_000_001})
        with self.assertRaises(CapacityPending):
            WorkingBudget(self.root)
        write_receipt(self.root / "limits.json", {**limits, "max_raw_bytes": 400_000_000_001})
        with self.assertRaises(CapacityPending):
            WorkingBudget(self.root)
        write_receipt(self.root / "limits.json", {**limits, "max_working_bytes": 119})
        with self.assertRaises(CapacityPending):
            WorkingBudget(self.root)
        write_receipt(self.root / "limits.json", {**limits, "max_working_bytes": True})
        with self.assertRaises(ValueError):
            WorkingBudget(self.root)
        (self.root / "limits.json").write_text(json.dumps(limits))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            WorkingBudget(self.root)

    def test_missing_or_corrupted_global_ledger_fails_closed(self) -> None:
        budget = WorkingBudget(self.root)
        with budget.quota("fixture", self.root / "derived") as quota:
            with quota.open("data") as writer:
                writer.write(b"123")
        original = budget.path.read_text()
        budget.path.write_text(original.replace('"reserved_bytes":3', '"reserved_bytes":0'))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            WorkingBudget(self.root)
        budget.path.unlink()
        with self.assertRaisesRegex(ValueError, "total is missing"):
            WorkingBudget(self.root)

    def test_two_processes_cannot_spend_the_same_remaining_allowance(self) -> None:
        _limits(self.root, 100)
        code = """
import sys
from pathlib import Path
from metis_data17.storage import WorkingBudget
from metis_data17.acquisition import CapacityPending
root, name = Path(sys.argv[1]), sys.argv[2]
try:
    with WorkingBudget(root, allocation_bytes=1).quota(name, root / name) as quota:
        print("ready", flush=True)
        sys.stdin.readline()
        with quota.open("data") as writer:
            writer.write(b"x" * 70)
    print("written", flush=True)
except CapacityPending:
    print("blocked", flush=True)
"""
        first, second = self._child(code, "first"), self._child(code, "second")
        self.assertEqual(first.stdout.readline().strip(), "ready")
        self.assertEqual(second.stdout.readline().strip(), "ready")
        first.stdin.write("\n")
        first.stdin.flush()
        second.stdin.write("\n")
        second.stdin.flush()
        outcomes = []
        for process in (first, second):
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
            outcomes.append(stdout.strip())
        self.assertCountEqual(outcomes, ["written", "blocked"])
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], 70)
        self.assertEqual(_tree_bytes(self.root / "first") + _tree_bytes(self.root / "second"), 70)

    def test_killed_writer_retains_then_reconciles_published_partial_bytes(self) -> None:
        code = """
import os, sys
from pathlib import Path
from metis_data17.storage import WorkingBudget
root = Path(sys.argv[1])
with WorkingBudget(root, allocation_bytes=8).quota("fixture", root / "derived") as quota:
    with quota.open("part") as writer:
        writer.write(b"123456")
    os._exit(23)
"""
        process = self._child(code)
        _, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 23, stderr)
        budget = WorkingBudget(self.root, allocation_bytes=8)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 8)
        with budget.quota("fixture", self.root / "derived") as quota:
            self.assertEqual(quota.reserved_bytes, 6)
            with quota.open("part", "ab") as writer:
                writer.write(b"7890")
        self.assertEqual(budget.snapshot()["reserved_bytes"], 10)
        self.assertEqual((self.root / "derived" / "part").read_bytes(), b"1234567890")

    def test_pending_growth_transaction_recovers_at_every_persistence_boundary(self) -> None:
        code = """
import os, sys
from pathlib import Path
from metis_data17 import storage
root, boundary = Path(sys.argv[1]), sys.argv[2]
original = storage._write_sealed
def interrupted(path, value):
    original(path, value)
    if boundary == "journal":
        stop = path.name == "pending.json" and value["namespace_after"]["reserved_bytes"] > 0
    elif boundary == "namespace":
        stop = value.get("schema") == storage._NAMESPACE_SCHEMA and value["reserved_bytes"] > 0
    else:
        stop = path.name == "total.json" and value["reserved_bytes"] > 0
    if stop:
        os._exit(23)
storage._write_sealed = interrupted
with storage.WorkingBudget(root, allocation_bytes=8).quota(boundary, root / boundary) as quota:
    with quota.open("data") as writer:
        writer.write(b"123456")
"""
        for boundary in ("journal", "namespace", "total"):
            with self.subTest(boundary=boundary):
                process = self._child(code, boundary)
                _, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 23, stderr)
                budget = WorkingBudget(self.root, allocation_bytes=8)
                self.assertEqual(budget.snapshot()["reserved_bytes"], 8)
                self.assertFalse(budget.pending_path.exists())
                self.assertEqual((self.root / boundary / "data").stat().st_size, 0)
                with budget.quota(boundary, self.root / boundary) as quota:
                    self.assertEqual(quota.reserved_bytes, 0)
                    with quota.open("data") as writer:
                        writer.write(b"123456")
                    quota.unlink(quota.directory / "data")
                self.assertEqual(budget.snapshot()["reserved_bytes"], 0)

    def test_refund_transaction_crash_only_releases_already_deleted_data(self) -> None:
        code = """
import os, sys
from pathlib import Path
from metis_data17 import storage
root = Path(sys.argv[1])
budget = storage.WorkingBudget(root, allocation_bytes=1)
with budget.quota("fixture", root / "derived") as quota:
    with quota.open("data") as writer:
        writer.write(b"123456")
original = storage._write_sealed
def interrupted(path, value):
    original(path, value)
    if path.name == "pending.json" and value["namespace_after"]["reserved_bytes"] == 0:
        os._exit(23)
with budget.quota("fixture", root / "derived") as quota:
    storage._write_sealed = interrupted
    quota.unlink(quota.directory / "data")
"""
        process = self._child(code)
        _, stderr = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 23, stderr)
        budget = WorkingBudget(self.root)
        self.assertFalse((self.root / "derived" / "data").exists())
        self.assertEqual(budget.snapshot()["reserved_bytes"], 0)
        with budget.quota("fixture", self.root / "derived") as quota:
            with quota.open("data") as writer:
                writer.write(b"x" * 10)
        self.assertEqual(budget.snapshot()["reserved_bytes"], 10)

    def test_binary_writer_supports_buffered_unicode_text(self) -> None:
        _limits(self.root, 32)
        with WorkingBudget(self.root, allocation_bytes=8).quota("fixture", self.root / "derived") as quota:
            with io.TextIOWrapper(quota.open("text"), encoding="utf-8") as stream:
                stream.write("é漢字\n")
            self.assertEqual(quota.used_bytes, len("é漢字\n".encode()))
            self.assertEqual((quota.directory / "text").read_text(), "é漢字\n")

    def test_two_threads_cannot_spend_the_same_local_reservation(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=10)
        started, release = threading.Event(), threading.Event()
        with budget.quota("fixture", self.root / "derived") as quota:
            first, second = quota.open("first"), quota.open("second")
            raw_write = first._file.write

            def delayed_write(data):
                started.set()
                if not release.wait(10):
                    raise AssertionError("fixture writer was not released")
                return raw_write(data)

            raw = Mock(wraps=first._file)
            raw.write.side_effect = delayed_write
            with patch.object(first, "_file", raw), ThreadPoolExecutor(max_workers=2) as pool:
                one = pool.submit(first.write, b"1234567")
                try:
                    self.assertTrue(started.wait(10))
                    two = pool.submit(second.write, b"1234567")
                    with self.assertRaises(TimeoutError):
                        two.result(timeout=0.1)
                finally:
                    release.set()
                self.assertEqual(one.result(timeout=10), 7)
                with self.assertRaises(CapacityPending):
                    two.result(timeout=10)
            self.assertEqual(_tree_bytes(quota.directory), 7)
            self.assertLessEqual(quota.used_bytes, quota.reserved_bytes)

    def test_partial_os_write_failure_prevents_reuse_of_unaccounted_local_credit(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=10)
        for error in (OSError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                with budget.quota("fixture", self.root / "derived") as quota:
                    first, second = quota.open("first"), quota.open("second")
                    original = first._file.write

                    def partial_write(data):
                        original(data[:2])
                        raise error("injected partial write")

                    raw = Mock(wraps=first._file)
                    raw.write.side_effect = partial_write
                    with patch.object(first, "_file", raw), self.assertRaises(error):
                        first.write(b"1234567")
                    self.assertEqual((quota.directory / "first").stat().st_size, 2)
                    with self.assertRaises(ValueError):
                        second.write(b"x" * 9)
                self.assertEqual(budget.snapshot()["reserved_bytes"], 10)
                with budget.quota("fixture", self.root / "derived") as quota:
                    self.assertEqual(quota.used_bytes, 2)
                    self.assertEqual(quota.reserved_bytes, 2)


class Metis17StoragePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = prep_fixtures.Metis17PreparationTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = {**self.fixture.config, "enforce_storage_budget": True}
        _limits(self.root, 2_000_000, metadata=20_000)

    def test_ready_chunks_and_eligibility_still_stream_with_shared_accounting(self) -> None:
        rows = [{"text": f"{self.fixture.SAFE} Ordinal {index}."} for index in range(4)]
        rows.append({"text": ""})
        _limits(self.root, 2 * storage.ALLOCATION_BYTES, metadata=20_000)
        spec, raw, _ = self.fixture._object(rows)
        output = self.root / "reblock" / spec.object_id[:2] / spec.object_id
        prepared = self.root / "prepared" / spec.object_id[:2] / spec.object_id
        config = {**self.config, "output_chunk_bytes": 1}
        reader, opening = prep.iter_source_rows, storage.WorkingQuota.open
        early, opened = [], []

        def observing_reader(*args, **kwargs):
            for index, item in enumerate(reader(*args, **kwargs)):
                yield item
                if index == 0:
                    ready, = output.glob("normalized/*/part-*.READY.json")
                    result = prep.prepare_chunk(ready, prepared, config)
                    self.assertEqual(result["status"], "ELIGIBLE_PENDING_OBJECT_COMPLETION")
                    self.assertFalse((output / "REBLOCK_COMPLETE.json").exists())
                    early.append(ready)

        def recording_open(quota, path, mode="wb"):
            opened.append(Path(path))
            return opening(quota, path, mode)

        with patch.object(prep, "iter_source_rows", side_effect=observing_reader), \
                patch.object(storage.WorkingQuota, "open", recording_open):
            manifest = prep.reblock_object(spec, raw, output, config)
            results = [prep.prepare_chunk(self.root / chunk["path"], prepared, config)
                       for chunk in manifest["chunks"]]
        self.assertEqual(len(early), 1)
        self.assertEqual(sum(result["eligible_documents"] for result in results), 4)
        self.assertEqual(manifest["rejected"], {"empty_text": 1})
        self.assertTrue(any(path.suffix == ".parquet" and path.is_relative_to(output) for path in opened))
        self.assertTrue(any(path.suffix == ".parquet" and path.is_relative_to(prepared) for path in opened))
        self.assertTrue(any(path.name == "decisions.jsonl" for path in opened))
        self.assertTrue((self.root / raw.relative_path).exists())
        expected = _tree_bytes(output / "normalized") + _tree_bytes(prepared / "chunks")
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], expected)
        for result in results:
            for chunk in result["chunks"]:
                self.assertEqual(pq.read_table(self.root / chunk["path"]).num_rows, chunk["records"])
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw replayed")):
            self.assertEqual(prep.reblock_object(spec, raw, output, config), manifest)
            for result, chunk in zip(results, manifest["chunks"]):
                self.assertEqual(prep.prepare_chunk(self.root / chunk["path"], prepared, config), result)
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], expected)

    def test_parquet_capacity_failure_is_explicit_before_physical_overshoot(self) -> None:
        spec, raw, output = self.fixture._object([{"text": self.fixture.SAFE}])
        _limits(self.root, 8, metadata=20_000)
        original = storage.QuotaWriter.write
        sizes = []

        def observed(writer, data):
            try:
                return original(writer, data)
            finally:
                sizes.append(os.fstat(writer.fileno()).st_size)

        with patch.object(storage.QuotaWriter, "write", observed), self.assertRaises(CapacityPending):
            prep.reblock_object(spec, raw, output, self.config)
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 8)
        self.assertFalse((output / "REBLOCK_COMPLETE.json").exists())
        self.assertEqual(list(output.rglob("*.READY.json")), [])
        self.assertTrue((self.root / raw.relative_path).is_file())
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], 8)

    def test_large_decisions_are_metered_not_silently_dropped(self) -> None:
        spec, raw, output = self.fixture._object([{"text": ""} for _ in range(300)])
        _limits(self.root, 64, metadata=20_000)
        with self.assertRaises(CapacityPending):
            prep.reblock_object(spec, raw, output, self.config)
        self.assertFalse((output / "REBLOCK_COMPLETE.json").exists())
        self.assertEqual(list(output.rglob("*.parquet")), [])
        self.assertTrue((self.root / raw.relative_path).is_file())
        self.assertLessEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], 64)

    def test_parquet_and_decisions_remain_complete_when_os_writes_are_short(self) -> None:
        spec, raw, output = self.fixture._object([{"text": self.fixture.SAFE}, {"text": ""}])
        opening = storage.WorkingQuota.open

        def short_open(quota, path, mode="wb"):
            stream = opening(quota, path, mode)
            original = stream._file
            stream._file = Mock(wraps=original)
            stream._file.write.side_effect = lambda data: original.write(data[:3])
            return stream

        with patch.object(storage.WorkingQuota, "open", short_open):
            result = prep.prepare_object(spec, raw, output, self.config)
        self.assertEqual(result["eligible_documents"], 1)
        self.assertEqual(result["rejected"], {"empty_text": 1})
        self.assertEqual(self.fixture._rows(result)[0]["text"], self.fixture.SAFE)
        normalized = read_receipt(output / "NORMALIZED.json")
        decisions = self.root / normalized["decisions"]["path"]
        rejected, = [json.loads(line) for line in decisions.read_text().splitlines()]
        self.assertEqual(rejected["reason"], "empty_text")
        expected = _tree_bytes(output / "normalized") + _tree_bytes(output / "eligible")
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], expected)

    def test_parquet_footer_exhaustion_never_publishes_a_ready_chunk(self) -> None:
        spec, raw, output = self.fixture._object([{"text": self.fixture.SAFE}])
        baseline = prep.reblock_object(spec, raw, output, self.fixture.config)
        size = baseline["chunks"][0]["byte_count"]
        shutil.rmtree(output)
        _limits(self.root, size - 1, metadata=20_000)
        original = storage.QuotaWriter.write
        written = []

        def observed(writer, data):
            try:
                return original(writer, data)
            finally:
                written.append(os.fstat(writer.fileno()).st_size)

        with patch.object(storage.QuotaWriter, "write", observed), self.assertRaises(CapacityPending):
            prep.reblock_object(spec, raw, output, self.config)
        self.assertGreater(max(written), size // 2)
        self.assertLessEqual(max(written), size - 1)
        self.assertEqual(list(output.rglob("*.READY.json")), [])
        self.assertFalse((output / "REBLOCK_COMPLETE.json").exists())
        self.assertTrue((self.root / raw.relative_path).exists())

    def test_partial_normalized_retry_needs_only_one_extra_chunk_not_a_second_object(self) -> None:
        spec, raw, output = self.fixture._object(
            [{"text": f"{self.fixture.SAFE} Record {index}."} for index in range(30)])
        config = {**self.config, "output_chunk_bytes": 1}
        baseline = prep.reblock_object(spec, raw, output, {**config, "enforce_storage_budget": False})
        expected_bytes = _tree_bytes(output / "normalized")
        largest = max(chunk["byte_count"] for chunk in baseline["chunks"])
        shutil.rmtree(output)
        _limits(self.root, expected_bytes + largest + 1024, metadata=20_000)
        reader = prep.iter_source_rows

        def interrupted(*args, **kwargs):
            for index, item in enumerate(reader(*args, **kwargs)):
                if index == 29:
                    raise prep.PreparationError(spec, "injected_interruption", item.index)
                yield item

        with patch.object(prep, "iter_source_rows", side_effect=interrupted), \
                self.assertRaises(prep.PreparationError):
            prep.reblock_object(spec, raw, output, config)
        ready = list(output.glob("normalized/*/part-*.READY.json"))
        self.assertEqual(len(ready), 29)
        before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in ready}
        manifest = prep.reblock_object(spec, raw, output, config)
        self.assertEqual(manifest["normalized_documents"], 30)
        self.assertEqual(before, {path: (sha256_file(path), path.stat().st_mtime_ns) for path in ready})
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"],
                         _tree_bytes(output / "normalized"))
        self.assertLessEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"],
                             expected_bytes + largest + 1024)

    def test_verified_legacy_outputs_are_adopted_without_changing_fingerprints(self) -> None:
        spec, raw, output = self.fixture._object([{"text": self.fixture.SAFE}, {"text": ""}])
        result = prep.prepare_object(spec, raw, output, self.fixture.config)
        expected = _tree_bytes(output / "normalized") + _tree_bytes(output / "eligible")
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw replayed")):
            adopted = prep.prepare_object(spec, raw, output, self.config)
        self.assertEqual(adopted, result)
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], expected)
        self.assertEqual(WorkingBudget(self.root).snapshot()["namespaces"], 2)
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw replayed")):
            self.assertEqual(prep.prepare_object(spec, raw, output, self.config), result)
        self.assertEqual(WorkingBudget(self.root).snapshot()["reserved_bytes"], expected)

    def test_abandoned_generation_cleanup_reconciles_only_after_removal(self) -> None:
        budget = WorkingBudget(self.root, allocation_bytes=1)
        output = self.root / "output"
        fingerprint = "a" * 64
        with budget.quota("fixture", output / "normalized") as quota:
            abandoned = quota.directory / f".{fingerprint}.{'b' * 32}.partial"
            abandoned.mkdir()
            with quota.open(abandoned / "data") as writer:
                writer.write(b"123456")
            self.assertEqual(quota.used_bytes, 6)
            _, staging = prep._generation(output, "normalized", fingerprint, quota)
            self.assertFalse(abandoned.exists())
            self.assertTrue(staging.is_dir())
            self.assertEqual(quota.used_bytes, 0)
            self.assertEqual(budget.snapshot()["reserved_bytes"], 0)

    def test_malformed_storage_opt_in_cannot_silently_disable_enforcement(self) -> None:
        spec, raw, output = self.fixture._object([{"text": self.fixture.SAFE}])
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            prep.reblock_object(spec, raw, output, {**self.config, "enforce_storage_budget": "true"})
        self.assertEqual(list(output.rglob("*.parquet")), [])

    def test_retained_extraction_and_eligibility_generations_are_charged_separately(self) -> None:
        spec, raw, _ = self.fixture._object([{"text": self.fixture.SAFE}])
        reblock = [
            self.root / "reblock" / (value * 64) / spec.object_id[:2] / spec.object_id
            for value in ("a", "b")
        ]
        prepared = [
            self.root / "prepared" / (value * 64) / spec.object_id[:2] / spec.object_id
            for value in ("c", "d")
        ]
        manifests = [prep.reblock_object(spec, raw, directory, self.config) for directory in reblock]
        self.assertEqual(manifests[0]["normalization_fingerprint"],
                         manifests[1]["normalization_fingerprint"])
        self.assertEqual(manifests[0]["chunks"][0]["chunk_id"], manifests[1]["chunks"][0]["chunk_id"])
        base = self.root / manifests[0]["chunks"][0]["path"]
        results = [prep.prepare_chunk(base, directory, self.config) for directory in prepared]
        self.assertEqual(results[0]["eligibility_fingerprint"], results[1]["eligibility_fingerprint"])
        self.assertEqual([result["eligible_documents"] for result in results], [1, 1])
        expected = (
            sum(_tree_bytes(directory / "normalized") for directory in reblock)
            + sum(_tree_bytes(directory / "chunks") for directory in prepared)
        )
        budget = WorkingBudget(self.root)
        self.assertEqual(budget.snapshot()["namespaces"], 4)
        self.assertEqual(budget.snapshot()["reserved_bytes"], expected)
        immutable = [
            path for directory in (*reblock, *prepared)
            for path in directory.rglob("*")
            if path.suffix == ".parquet" or path.name.endswith(".READY.json")
        ]
        before = {path: (sha256_file(path), path.stat().st_mtime_ns) for path in immutable}
        with patch.object(prep, "iter_source_rows", side_effect=AssertionError("raw replayed")):
            for manifest, directory in zip(manifests, reblock):
                self.assertEqual(prep.reblock_object(spec, raw, directory, self.config), manifest)
            for result, directory in zip(results, prepared):
                self.assertEqual(prep.prepare_chunk(base, directory, self.config), result)
        self.assertEqual(before, {path: (sha256_file(path), path.stat().st_mtime_ns) for path in immutable})
        self.assertEqual(budget.snapshot()["namespaces"], 4)
        self.assertEqual(budget.snapshot()["reserved_bytes"], expected)


if __name__ == "__main__":
    unittest.main()
