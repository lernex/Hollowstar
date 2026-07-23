from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis_data.datatrove_blocks import (
    cluster_priority_minhash_pairs,
    finalize_priority_minhash_rank_removals,
    require_verified_priority_minhash_rank,
    resolve_priority_minhash_bucket,
    verified_minhash_bucket_inventory,
    verify_priority_minhash_completion,
    write_minhash_bucket_output_manifest,
    write_priority_minhash_rank_candidates,
)


def _write_rank(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_removals(path: Path) -> list[int]:
    if not path.exists():
        return []
    payload = path.read_bytes()
    return [value[0] for value in struct.iter_unpack("<I", payload)]


class DiskPriorityMinhashTests(unittest.TestCase):
    def test_partitioned_pipeline_preserves_priority_and_tie_semantics(self) -> None:
        try:
            import datatrove  # noqa: F401
        except ImportError:
            self.skipTest("DataTrove is installed by the Metis-1.6 data runtime")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicates = root / "duplicates"
            documents = root / "documents"
            work = root / "work"
            removals = root / "removals"
            scratch = root / "scratch"
            duplicates.mkdir()
            documents.mkdir()
            _write_rank(
                documents / "00000.jsonl",
                [
                    {"id": "a-low", "text": "a", "metadata": {"priority": 3}},
                    {"id": "b-winner", "text": "b", "metadata": {"priority": 30}},
                ],
            )
            _write_rank(
                documents / "00001.jsonl",
                [
                    {"id": "a-tie-one", "text": "c", "metadata": {"priority": 20}},
                    {"id": "b-low", "text": "d", "metadata": {"priority": 2}},
                ],
            )
            _write_rank(
                documents / "00002.jsonl",
                [
                    {"id": "a-tie-two", "text": "e", "metadata": {"priority": 20}},
                ],
            )
            with (duplicates / "pairs.dups").open("wb") as handle:
                handle.write(struct.pack("<4I", 0, 0, 1, 0))
                handle.write(struct.pack("<4I", 1, 0, 2, 0))
                handle.write(struct.pack("<4I", 0, 1, 1, 1))

            # The clustering stage must stream pair files rather than loading
            # them through Path.read_bytes.
            with mock.patch.object(
                Path,
                "read_bytes",
                side_effect=AssertionError("corpus-sized read_bytes is forbidden"),
            ):
                cluster = cluster_priority_minhash_pairs(
                    duplicates,
                    work,
                    total_tasks=3,
                    bucket_count=3,
                    sqlite_cache_mb=8,
                    transaction_rows=1,
                    temporary_directory=scratch,
                )
            self.assertEqual(cluster["duplicate_pairs"], 3)
            self.assertEqual(cluster["component_members"], 5)
            self.assertEqual(cluster["components"], 2)
            self.assertEqual(
                cluster_priority_minhash_pairs(
                    duplicates,
                    work,
                    total_tasks=3,
                    bucket_count=3,
                    sqlite_cache_mb=8,
                    temporary_directory=scratch,
                )["manifest_sha256"],
                cluster["manifest_sha256"],
            )

            candidates = [
                write_priority_minhash_rank_candidates(
                    documents,
                    work,
                    rank=rank,
                    total_tasks=3,
                    max_open_files=1,
                )
                for rank in range(3)
            ]
            self.assertEqual(sum(row["candidate_count"] for row in candidates), 5)
            resolved = [
                resolve_priority_minhash_bucket(
                    work,
                    bucket=bucket,
                    total_tasks=3,
                    sqlite_cache_mb=8,
                    transaction_rows=1,
                    temporary_directory=scratch,
                )
                for bucket in range(3)
            ]
            self.assertEqual(sum(row["components"] for row in resolved), 2)
            self.assertEqual(sum(row["removed"] for row in resolved), 3)
            finalized = [
                finalize_priority_minhash_rank_removals(
                    work,
                    removals,
                    rank=rank,
                    total_tasks=3,
                    sqlite_cache_mb=8,
                    transaction_rows=1,
                    temporary_directory=scratch,
                )
                for rank in range(3)
            ]
            self.assertEqual(sum(row["removal_count"] for row in finalized), 3)
            complete = verify_priority_minhash_completion(
                work,
                removals,
                total_tasks=3,
            )
            self.assertEqual(complete["component_members"], 5)
            self.assertEqual(complete["components"], 2)
            self.assertEqual(complete["removed"], 3)
            self.assertEqual(
                [row["rank"] for row in complete["removal_files"]],
                [0, 1, 2],
            )
            rank_receipts = [
                require_verified_priority_minhash_rank(
                    work,
                    removals,
                    rank=rank,
                    total_tasks=3,
                )
                for rank in range(3)
            ]
            self.assertEqual(sum(row["removal_count"] for row in rank_receipts), 3)
            self.assertEqual(list(scratch.rglob("*.sqlite3")), [])

            finalizer_marker = work / "finalized" / "rank-000001.json"
            finalizer_bytes = finalizer_marker.read_bytes()
            finalizer_marker.unlink()
            with self.assertRaisesRegex(RuntimeError, "finalizer marker inventory"):
                verify_priority_minhash_completion(
                    work,
                    removals,
                    total_tasks=3,
                )
            finalizer_marker.write_bytes(finalizer_bytes)
            (removals / "999999.remove").touch()
            with self.assertRaisesRegex(RuntimeError, "final removal inventory"):
                verify_priority_minhash_completion(
                    work,
                    removals,
                    total_tasks=3,
                )
            (removals / "999999.remove").unlink()

            removed = {
                (rank, document_id)
                for rank in range(3)
                for document_id in _read_removals(removals / f"{rank:06d}.remove")
            }
            tie_winner_rank = max(
                (1, 2),
                key=lambda rank: hashlib.sha256(
                    ("a-tie-one" if rank == 1 else "a-tie-two").encode("utf-8")
                ).hexdigest(),
            )
            expected_kept = {(0, 1), (tie_winner_rank, 0)}
            all_members = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)}
            self.assertEqual(all_members - removed, expected_kept)
            for rank in range(3):
                values = _read_removals(removals / f"{rank:06d}.remove")
                self.assertEqual(values, sorted(set(values)))
                marker = work / "finalized" / f"rank-{rank:06d}.json"
                self.assertEqual(json.loads(marker.read_text())["status"], "complete")

            tampered_rank = next(
                rank for rank, row in enumerate(rank_receipts) if row["removal_count"]
            )
            with (removals / f"{tampered_rank:06d}.remove").open("ab") as handle:
                handle.write(struct.pack("<I", 99))
            with self.assertRaisesRegex(RuntimeError, "changed after finalization"):
                require_verified_priority_minhash_rank(
                    work,
                    removals,
                    rank=tampered_rank,
                    total_tasks=3,
                )

    def test_missing_rank_manifest_and_missing_document_fail_closed(self) -> None:
        try:
            import datatrove  # noqa: F401
        except ImportError:
            self.skipTest("DataTrove is installed by the Metis-1.6 data runtime")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicates = root / "duplicates"
            documents = root / "documents"
            work = root / "work"
            duplicates.mkdir()
            documents.mkdir()
            _write_rank(
                documents / "00000.jsonl",
                [{"id": "only", "text": "one", "metadata": {"priority": 1}}],
            )
            (duplicates / "pairs.dups").write_bytes(
                struct.pack("<4I", 0, 0, 0, 1)
            )
            cluster_priority_minhash_pairs(
                duplicates,
                work,
                total_tasks=1,
                bucket_count=2,
                sqlite_cache_mb=8,
            )
            with self.assertRaisesRegex(RuntimeError, "no corresponding document"):
                write_priority_minhash_rank_candidates(
                    documents,
                    work,
                    rank=0,
                    total_tasks=1,
                )
            self.assertFalse(
                (work / "candidates" / "rank-000000" / "CANDIDATES.json").exists()
            )

            _write_rank(
                documents / "00000.jsonl",
                [
                    {"id": "only", "text": "one", "metadata": {"priority": 1}},
                    {"id": "now-present", "text": "two", "metadata": {"priority": 2}},
                ],
            )
            marker = write_priority_minhash_rank_candidates(
                documents,
                work,
                rank=0,
                total_tasks=1,
            )
            self.assertEqual(marker["candidate_count"], 2)
            (work / "candidates" / "rank-000000" / "CANDIDATES.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Completeness manifest is missing"):
                resolve_priority_minhash_bucket(
                    work,
                    bucket=0,
                    total_tasks=1,
                    sqlite_cache_mb=8,
                )

    def test_datatrove_bucket_inventory_seals_empty_outputs_and_exact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicates = root / "duplicates"
            inventory = root / "inventory"
            work = root / "work"
            scratch = root / "scratch"
            duplicates.mkdir()
            (duplicates / "00000_00.dups").write_bytes(
                struct.pack("<4I", 0, 0, 0, 1)
            )
            (duplicates / "00001_00.dups").touch()

            first = write_minhash_bucket_output_manifest(
                duplicates,
                inventory,
                bucket=0,
                expected_buckets=2,
            )
            second = write_minhash_bucket_output_manifest(
                duplicates,
                inventory,
                bucket=1,
                expected_buckets=2,
            )
            self.assertEqual(first["output"]["records"], 1)
            self.assertFalse(first["output"]["empty"])
            self.assertEqual(second["output"]["records"], 0)
            self.assertTrue(second["output"]["empty"])
            paths, rows, inventory_sha256 = verified_minhash_bucket_inventory(
                duplicates,
                inventory,
                expected_buckets=2,
            )
            self.assertEqual([path.name for path in paths], ["00000_00.dups", "00001_00.dups"])
            self.assertEqual([row["records"] for row in rows], [1, 0])
            self.assertEqual(len(inventory_sha256), 64)

            cluster = cluster_priority_minhash_pairs(
                duplicates,
                work,
                total_tasks=1,
                bucket_count=2,
                bucket_inventory_folder=inventory,
                expected_duplicate_buckets=2,
                sqlite_cache_mb=8,
                temporary_directory=scratch,
            )
            self.assertEqual(
                cluster["input_contract"]["mode"],
                "sealed_datatrove_buckets",
            )
            self.assertEqual(cluster["duplicate_pairs"], 1)
            self.assertEqual(list(scratch.rglob("*.sqlite3")), [])

            (duplicates / "00001_00.dups").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "changed after sealing"):
                verified_minhash_bucket_inventory(
                    duplicates,
                    inventory,
                    expected_buckets=2,
                )

    def test_datatrove_bucket_inventory_rejects_missing_and_extra_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicates = root / "duplicates"
            inventory = root / "inventory"
            duplicates.mkdir()
            for bucket in range(2):
                (duplicates / f"{bucket:05d}_00.dups").touch()
                write_minhash_bucket_output_manifest(
                    duplicates,
                    inventory,
                    bucket=bucket,
                    expected_buckets=2,
                )

            (inventory / "bucket-000001.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "manifest inventory is incomplete"):
                verified_minhash_bucket_inventory(
                    duplicates,
                    inventory,
                    expected_buckets=2,
                )
            write_minhash_bucket_output_manifest(
                duplicates,
                inventory,
                bucket=1,
                expected_buckets=2,
            )
            (duplicates / "99999_00.dups").touch()
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                verified_minhash_bucket_inventory(
                    duplicates,
                    inventory,
                    expected_buckets=2,
                )

    def test_cluster_failure_removes_node_local_sqlite_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicates = root / "duplicates"
            work = root / "work"
            scratch = root / "scratch"
            duplicates.mkdir()
            (duplicates / "broken.dups").write_bytes(b"not-a-fixed-record")
            with self.assertRaisesRegex(RuntimeError, "Corrupt fixed-record"):
                cluster_priority_minhash_pairs(
                    duplicates,
                    work,
                    total_tasks=1,
                    bucket_count=1,
                    temporary_directory=scratch,
                    sqlite_cache_mb=8,
                )
            self.assertEqual(list(scratch.rglob("*.sqlite3")), [])


if __name__ == "__main__":
    unittest.main()
