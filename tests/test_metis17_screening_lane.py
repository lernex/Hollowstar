from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from metis_data17 import prep
from metis_data17.acquisition import receipt_path
from metis_data17.cli import append_event
from metis_data17.common import ObjectSpec, read_receipt, utc_now, write_receipt
from metis_data17.worker import _raw_candidates, claim, prep_service, screen_chunk
from tests import test_metis17_worker as fixtures


class Metis17ScreeningLaneTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Metis17WorkerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = {
            **self.fixture.config,
            "raw_readers_per_node": 1,
            "source_minimum_acceptance": 0.1,
            "opt_out_snapshot": self.fixture.fixture._opt_out(),
        }

    def _publish(self, count=2):
        result = []
        for index in range(count):
            spec, raw, _ = self.fixture._object()
            event = {**raw.to_dict(), "spec": spec.to_dict()}
            write_receipt(receipt_path(self.root, spec.object_id), event)
            append_event(self.root, f"raw/uplink-{index % 2}", event)
            result.append((spec, raw))
        return result

    def _run(self, screening_only):
        real_sleep = time.sleep
        with patch("metis_data17.worker.worker_configuration", return_value=(
            dict(self.config), {"config": {"limits": {"capacity_confirmation": "pending"}}},
        )), patch("metis_data17.worker.code_commit", return_value="a" * 40), patch(
            "metis_data17.worker.ProcessPoolExecutor",
            side_effect=lambda max_workers, **_: ThreadPoolExecutor(max_workers=max_workers),
        ), patch("metis_data17.worker.time.sleep", side_effect=lambda _: real_sleep(0.005)):
            prep_service(
                self.root, workers=2, idle_seconds=0.1, maximum_seconds=20,
                screening_only=screening_only,
            )

    def _screened(self):
        return [
            read_receipt(path) for path in
            (self.root / "state/screened-objects" / self.config["generation"]).glob("*/*.json")
        ]

    def test_screening_covers_both_uplinks_without_claiming_index_completion(self):
        objects = self._publish()
        with patch("metis_data17.worker.index_chunk", side_effect=AssertionError("Producer attempted indexing")):
            self._run(True)
        screened = self._screened()
        self.assertEqual({row["object_id"] for row in screened}, {spec.object_id for spec, _ in objects})
        self.assertEqual(len(screened), 2)
        self.assertEqual(sum(row["eligible_documents"] for row in screened), 28)
        for row in screened:
            self.assertEqual(row["schema"], "metis17.screened-object/v1")
            self.assertFalse(row["indexing_complete"])
            self.assertEqual(row["indexed_chunk_count"], 0)
            self.assertEqual(len(row["chunk_ids"]), len(set(row["chunk_ids"])))
            self.assertNotIn("index_generation", row)
        self.assertFalse(list((self.root / "state/prepared-objects").glob("**/*.json")))
        self.assertFalse((self.root / "dedup").exists())

        with patch("metis_data17.worker._reblock_job", side_effect=AssertionError("Repeated completed screening")):
            self._run(True)
        self.assertEqual(self._screened(), screened)

        self._run(False)
        completed = [
            read_receipt(path) for path in
            (self.root / "state/prepared-objects" / self.config["index_generation"]).glob("*/*.json")
        ]
        self.assertEqual({row["object_id"] for row in completed}, {spec.object_id for spec, _ in objects})
        self.assertTrue(all(row["indexing_complete"] for row in completed))
        self.assertEqual(
            {row["object_id"]: row["chunk_ids"] for row in completed},
            {row["object_id"]: row["chunk_ids"] for row in screened},
        )
        self.assertFalse(list((self.root / "state/prep-failures").glob("**/*.json")))

    def test_producers_share_object_claims_with_existing_full_workers(self):
        objects = self._publish()
        first = objects[0][0].object_id
        lease = claim(self.root / "locks/prep-objects" / self.config["index_generation"] / f"{first}.flock")
        self.assertIsNotNone(lease)
        try:
            self._run(True)
            self.assertEqual([row["object_id"] for row in self._screened()], [objects[1][0].object_id])
        finally:
            lease.close()
        self._run(True)
        self.assertEqual({row["object_id"] for row in self._screened()}, {spec.object_id for spec, _ in objects})

    def test_screening_completion_waits_for_parent_eof(self):
        self._publish(1)
        pending = threading.Event()
        original_reader = prep.iter_source_rows

        def slow_reader(*args, **kwargs):
            for index, row in enumerate(original_reader(*args, **kwargs)):
                yield row
                if index == 7:
                    self.assertTrue(pending.wait(10), "No concurrent screening before EOF")
                    self.assertEqual(self._screened(), [])

        def observed_screen(*args):
            result = screen_chunk(*args)
            if result["status"] == "ELIGIBLE_PENDING_OBJECT_COMPLETION":
                pending.set()
            return result

        with patch("metis_data17.prep.iter_source_rows", side_effect=slow_reader), patch(
            "metis_data17.worker.screen_chunk", side_effect=observed_screen,
        ):
            self._run(True)
        self.assertTrue(pending.is_set())
        self.assertEqual(len(self._screened()), 1)
        self.assertEqual(self._screened()[0]["eligible_documents"], 14)

    def test_an_index_failure_does_not_block_or_get_overwritten_by_screening(self):
        spec, _ = self._publish(1)[0]
        failure = {
            "generation": self.config["generation"], "index_generation": self.config["index_generation"],
            "object_id": spec.object_id, "status": "failed", "stage": "index", "created_at": utc_now(),
        }
        path = self.root / "state/prep-failures" / self.config["index_generation"] / spec.object_id[:2] / f"{spec.object_id}.json"
        write_receipt(path, failure)
        append_event(self.root, "prep-errors/full-worker", failure)
        self._run(True)
        self.assertEqual([row["object_id"] for row in self._screened()], [spec.object_id])
        self.assertEqual(read_receipt(path), failure)

    def test_category_balancing_preserves_quality_order_and_exact_candidate_coverage(self):
        candidates = []
        categories = ("math", "code", "science", "web", "multilingual")
        for category_index, category in enumerate(categories):
            for index in range(3):
                spec = ObjectSpec.create(
                    source_id=f"{category}-{index}", url=f"https://example.test/{category}/{index}",
                    revision="r", relative_key=f"{category}-{index}", wire_format="raw_jsonl",
                    adapter="text", priority=100 - 10 * category_index - index,
                    policy={"category": category},
                )
                candidates.append(((True, -spec.priority, 1, spec.object_id), spec, SimpleNamespace()))
        active = {}
        ordered = []
        for candidate in _raw_candidates(candidates, active, balance_categories=True):
            spec = candidate[1]
            ordered.append(spec)
            active[spec.object_id] = SimpleNamespace(spec=spec)
        self.assertEqual([spec.policy["category"] for spec in ordered[:5]], list(categories))
        self.assertEqual(len({spec.object_id for spec in ordered}), len(candidates))
        self.assertEqual({spec.object_id for spec in ordered}, {row[1].object_id for row in candidates})
        for category in categories:
            priorities = [spec.priority for spec in ordered if spec.policy["category"] == category]
            self.assertEqual(priorities, sorted(priorities, reverse=True))
        default = list(_raw_candidates(candidates, {}, balance_categories=False))
        self.assertEqual(default, sorted(candidates, key=lambda row: row[0]))
        key, spec, raw = candidates[-1]
        canary = ((False, *key[1:]), spec, raw)
        self.assertEqual(
            next(_raw_candidates([*candidates[:-1], canary], {}, balance_categories=True))[1].object_id,
            spec.object_id,
        )

    def test_screening_cannot_accidentally_run_index_maintenance(self):
        with self.assertRaisesRegex(ValueError, "do not own"):
            prep_service(self.root, screening_only=True, defer_compaction=True)


if __name__ == "__main__":
    unittest.main()
