from __future__ import annotations

import unittest

from metis_data.build_inputs import _split_oversized


def _records(*sizes: int) -> list[dict]:
    return [
        {"source_id": "s", "local_path": f"/lus/{index:03d}.jsonl.gz", "size": size}
        for index, size in enumerate(sizes)
    ]


def _assign(part_index: int, part_count: int, total_rows: int) -> list[int]:
    """Rows this part keeps, mirroring the normalize loop exactly."""

    return [row for row in range(total_rows) if row % part_count == part_index]


class OversizedInputSplitTests(unittest.TestCase):
    """Splitting must partition records, never duplicate or drop them.

    A part that silently skips records does not fail anything: the shard is
    smaller, every hash downstream still verifies, and the corpus is quietly
    missing documents. So the property under test is exact coverage.
    """

    def test_parts_cover_every_record_exactly_once(self) -> None:
        for total_rows in (0, 1, 2, 7, 1000, 779_046):
            for part_count in (1, 2, 3, 7, 8):
                with self.subTest(rows=total_rows, parts=part_count):
                    seen: list[int] = []
                    for part in range(part_count):
                        seen.extend(_assign(part, part_count, total_rows))
                    self.assertEqual(sorted(seen), list(range(total_rows)))
                    self.assertEqual(len(seen), len(set(seen)))

    def test_only_oversized_inputs_are_split(self) -> None:
        cap = 1_000_000_000
        records = _records(200_000_000, cap, cap + 1, 6_860_000_000)
        split = _split_oversized(records, cap)
        counts = {}
        for row in split:
            counts.setdefault(row["local_path"], []).append(row)
        self.assertEqual(len(counts["/lus/000.jsonl.gz"]), 1)
        self.assertEqual(len(counts["/lus/001.jsonl.gz"]), 1)
        self.assertEqual(len(counts["/lus/002.jsonl.gz"]), 2)
        # The 1.6 giant: 6.86GB at a 1GB cap becomes seven tasks.
        self.assertEqual(len(counts["/lus/003.jsonl.gz"]), 7)

    def test_parts_are_numbered_consistently(self) -> None:
        split = _split_oversized(_records(6_860_000_000), 1_000_000_000)
        self.assertEqual([row["part_index"] for row in split], list(range(7)))
        self.assertTrue(all(row["part_count"] == 7 for row in split))

    def test_disabled_cap_leaves_inputs_alone(self) -> None:
        records = _records(6_860_000_000, 10)
        self.assertEqual(_split_oversized(records, 0), records)

    def test_split_preserves_every_original_field(self) -> None:
        records = [
            {
                "source_id": "pes2o",
                "local_path": "/lus/02034.jsonl.gz",
                "size": 3_000_000_000,
                "sha256": "abc",
                "revision": "r1",
            }
        ]
        for row in _split_oversized(records, 1_000_000_000):
            self.assertEqual(row["sha256"], "abc")
            self.assertEqual(row["revision"], "r1")
            self.assertEqual(row["source_id"], "pes2o")
            self.assertEqual(row["size"], 3_000_000_000)

    def test_largest_task_shrinks_to_the_cap(self) -> None:
        """The point of the change: bound the biggest single unit of work."""

        cap = 1_000_000_000
        sizes = [200_000_000] * 50 + [6_860_000_000] * 10
        split = _split_oversized(_records(*sizes), cap)
        largest = max(
            int(row["size"]) / int(row["part_count"]) for row in split
        )
        self.assertLessEqual(largest, cap)
        self.assertEqual(len(split), 50 + 10 * 7)


if __name__ == "__main__":
    unittest.main()
