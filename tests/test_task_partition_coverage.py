from __future__ import annotations

import unittest

from metis_data.stage_runner import _task_indices


def _cover(total: int, tasks_per_job: int, *, stride: bool, offset: int = 0) -> list[int]:
    """Every index the whole array would run, in submission order."""

    entries = -(-total // tasks_per_job)
    covered: list[int] = []
    for group in range(entries):
        if stride:
            first = offset + group
            covered.extend(
                _task_indices(
                    first_index=first,
                    task_count=tasks_per_job,
                    task_limit=offset + total,
                    task_stride=entries,
                )
            )
        else:
            first = offset + group * tasks_per_job
            covered.extend(
                _task_indices(
                    first_index=first,
                    task_count=tasks_per_job,
                    task_limit=offset + total,
                    task_stride=0,
                )
            )
    return covered


class TaskPartitionCoverageTests(unittest.TestCase):
    """Coverage is the only property that matters here.

    A partition that drops an index does not fail: it silently omits a shard
    from the corpus and every downstream hash still verifies. So both layouts
    are checked for exact once-only coverage, not merely for plausible sizes.
    """

    def test_both_layouts_cover_every_index_exactly_once(self) -> None:
        for total, per_job in (
            (3274, 40),
            (3227, 40),
            (1851, 32),
            (100, 7),
            (41, 40),
            (40, 40),
            (39, 40),
            (1, 1),
            (7, 1),
        ):
            for stride in (False, True):
                with self.subTest(total=total, per_job=per_job, stride=stride):
                    covered = _cover(total, per_job, stride=stride)
                    self.assertEqual(sorted(covered), list(range(total)))
                    self.assertEqual(len(covered), len(set(covered)))

    def test_offset_chunks_stay_inside_their_range(self) -> None:
        offset, total, per_job = 3000, 500, 48
        for stride in (False, True):
            with self.subTest(stride=stride):
                covered = _cover(total, per_job, stride=stride, offset=offset)
                self.assertEqual(sorted(covered), list(range(offset, offset + total)))

    def test_contiguous_layout_is_unchanged(self) -> None:
        """Already-submitted arrays must keep the exact partition they were sized for."""

        self.assertEqual(
            _task_indices(first_index=2000, task_count=40, task_limit=3274, task_stride=0),
            list(range(2000, 2040)),
        )

    def test_striding_spreads_a_run_of_large_shards(self) -> None:
        """The 1.6 failure: ten adjacent giants landing in two array entries."""

        total, per_job = 3274, 40
        entries = -(-total // per_job)
        giants = set(range(2034, 2044))

        def owners(stride: bool) -> set[int]:
            found = set()
            for group in range(entries):
                first = group if stride else group * per_job
                got = _task_indices(
                    first_index=first,
                    task_count=per_job,
                    task_limit=total,
                    task_stride=entries if stride else 0,
                )
                if giants.intersection(got):
                    found.add(group)
            return found

        self.assertEqual(len(owners(stride=False)), 2)
        self.assertEqual(len(owners(stride=True)), len(giants))

    def test_stride_requires_a_limit(self) -> None:
        with self.assertRaises(ValueError):
            _task_indices(first_index=0, task_count=40, task_limit=0, task_stride=82)


if __name__ == "__main__":
    unittest.main()
