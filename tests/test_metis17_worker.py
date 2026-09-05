from __future__ import annotations

import json
import pickle
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from metis_data17.acquisition import CapacityPending, receipt_path
from metis_data17.cli import append_event, main
from metis_data17.common import digest_json, read_receipt, write_receipt
from metis_data17.worker import (
    EventTail, WorkerFailure, _claim_compaction, _compact_job, _execute, _job_result,
    _raw_event_metadata, _reblock_job, admit_source, claim,
    failure_blocks, index_chunk, observe_failure, prep_service, raw_event,
    screen_chunk, worker_configuration,
)
from tests import test_metis17_prep as fixtures


class Metis17WorkerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Metis17PreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = {
            **self.fixture.config, "generation": "fixture-generation",
            "extraction_generation": "fixture-extraction",
            "index_generation": "fixture-index",
            "dedup": {"bucket_count": 7, "max_fan_in": 16},
        }

    def _object(self):
        return self.fixture._object([
            {"text": self.fixture.SAFE + f" Source paragraph {i}."}
            for i in range(14)
        ])

    def test_raw_journal_must_agree_with_sealed_download(self):
        spec, raw, _ = self._object()
        event = {**raw.to_dict(), "spec": spec.to_dict()}
        write_receipt(receipt_path(self.root, spec.object_id), event)
        self.assertEqual(raw_event(self.root, event), (spec, raw))
        changed = {**event, "byte_count": raw.byte_count + 1}
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            raw_event(self.root, changed)

    def test_discovery_is_metadata_only_but_selected_work_rechecks_the_raw_receipt(self):
        spec, raw, _ = self._object()
        event = {**raw.to_dict(), "spec": spec.to_dict()}
        with patch("metis_data17.worker.read_receipt", side_effect=AssertionError("Unclaimed object I/O")):
            self.assertEqual(_raw_event_metadata(event), (spec, raw))
        write_receipt(receipt_path(self.root, spec.object_id), {**event, "byte_count": raw.byte_count + 1})
        with patch("metis_data17.worker.prep.reblock_object") as reader:
            with self.assertRaisesRegex(RuntimeError, "disagrees"):
                _reblock_job(spec, raw, self.config)
        reader.assert_not_called()

    def test_configuration_publishes_a_reusable_policy_generation(self):
        settings = {
            "prep": {
                "quality_profiles_path": str(self.fixture.quality),
                "output_chunk_bytes": 4096, "batch_size": 3,
            },
            "dedup": self.config["dedup"],
        }
        run = {"config": settings, "config_sha256": digest_json(settings)}
        write_receipt(self.root / "RUN.json", run)
        policy = {
            "policy_ready": True,
            "benchmark_registry": str(self.fixture.registry_path),
            "decontamination_index": str(self.fixture.index_path),
            "opt_out_snapshot": str(self.fixture._opt_out()),
        }
        with patch("metis_data17.worker.policy_config", return_value=policy):
            config, loaded = worker_configuration(self.root)
            again, _ = worker_configuration(self.root)
        self.assertEqual(loaded, run)
        self.assertEqual(config["generation"], again["generation"])
        self.assertEqual(config["index_generation"], again["index_generation"])
        self.assertIs(config["enforce_storage_budget"], True)
        descriptor = read_receipt(
            self.root / "preparation" / "generations" / f"{config['generation']}.json",
        )
        self.assertEqual(digest_json(descriptor), config["generation"])
        indexed = read_receipt(
            self.root / "preparation" / "indexes" / f"{config['index_generation']}.json",
        )
        self.assertEqual(indexed["eligibility_generation"], config["generation"])
        expected_modules = {
            path.name for path in (Path(__file__).parents[1] / "src" / "metis_data17").glob("dedup*.py")
        }
        self.assertEqual(set(indexed["code"]), expected_modules)

    def test_optout_parser_changes_invalidate_eligibility_not_raw_extraction(self):
        before = fixtures.prep._compute_stage_code("chunk_eligibility")
        extraction = fixtures.prep._compute_stage_code("normalization")
        with patch.dict(fixtures.prep._IMPORTED_CODE, {"metis_data17.optout17": "f" * 64}):
            changed = fixtures.prep._compute_stage_code("chunk_eligibility")
            self.assertNotEqual(before, changed)
            self.assertEqual(extraction, fixtures.prep._compute_stage_code("normalization"))

    def test_claim_is_exclusive_and_reusable_without_foreign_pid_guessing(self):
        path = self.root / "claims" / "object.flock"
        first = claim(path)
        self.assertIsNotNone(first)
        self.assertIsNone(claim(path))
        first.close()
        second = claim(path)
        self.assertIsNotNone(second)
        second.close()

    def test_worker_exception_payload_survives_process_boundary(self):
        spec, _, _ = self._object()

        def failed():
            raise fixtures.PreparationError(spec, "bad_schema")

        result = pickle.loads(pickle.dumps(_execute(failed)))
        self.assertFalse(result["ok"])
        self.assertIn("bad_schema", result["traceback"])
        with self.assertRaisesRegex(WorkerFailure, "bad_schema"):
            _job_result(SimpleNamespace(result=lambda: result))

        def full():
            raise CapacityPending("derived storage full")

        self.assertTrue(_execute(full)["capacity_pending"])

    def test_only_capacity_failures_retry_when_the_limits_change(self):
        full = {"status": "capacity_pending", "limits_sha256": "old"}
        self.assertTrue(failure_blocks(full, "old"))
        self.assertFalse(failure_blocks(full, "approved-expansion"))
        self.assertTrue(failure_blocks({"status": "failed"}, "approved-expansion"))
        self.assertFalse(failure_blocks(
            {"status": "failed", "worker_sha256": "broken-worker"},
            "unchanged-limits", implementation_sha256="fixed-worker",
        ))
        failures = {}
        newer = {**full, "object_id": "object", "created_at": "2026-09-05T02:00:00Z"}
        older = {**newer, "limits_sha256": "older", "created_at": "2026-09-05T01:00:00Z"}
        observe_failure(failures, newer)
        observe_failure(failures, older)
        self.assertEqual(failures["object"], newer)

    def test_real_producer_screen_dedup_signature_chain_is_restartable(self):
        spec, raw, output = self._object()
        normalized = fixtures.reblock_object(spec, raw, output, self.config)
        screened = {}
        seals = []
        for chunk in normalized["chunks"]:
            result = screen_chunk(self.root / chunk["ready_receipt"], spec, self.config)
            screened[result["chunk_id"]] = result
            stage_path = self.root / result["receipt_path"]
            self.assertEqual(stage_path.name, "ELIGIBLE.json")
            indexed = index_chunk(stage_path, spec, self.config)
            self.assertEqual(indexed["stage_receipt_sha256"], digest_json(read_receipt(stage_path)))
            self.assertEqual(index_chunk(stage_path, spec, self.config), indexed)
            seals.append(indexed["stage_receipt_sha256"])
        self.assertEqual(len(set(seals)), len(normalized["chunks"]))
        admission = admit_source(
            self.root, spec, normalized, screened,
            generation=self.config["generation"], minimum_acceptance=0.1,
        )
        self.assertEqual(admission["status"], "admitted")
        self.assertEqual(admission["eligible_documents"], 14)
        from metis_data17.dedup import iter_survivors
        survivors = list(iter_survivors(self.root / "dedup" / "exact" / self.config["index_generation"]))
        self.assertEqual(len(survivors), 14)
        self.assertTrue(all(row["stage_receipt_sha256"] in seals for row in survivors))

    def test_source_admission_detects_missing_duplicate_and_pending_chunks(self):
        spec, raw, output = self._object()
        normalized = fixtures.reblock_object(spec, raw, output, self.config)
        results = {
            chunk["chunk_id"]: screen_chunk(self.root / chunk["ready_receipt"], spec, self.config)
            for chunk in normalized["chunks"]
        }
        self.assertGreater(len(results), 1)
        missing = dict(results)
        missing.pop(next(iter(missing)))
        kwargs = {"generation": "fixture-generation", "minimum_acceptance": 0.1}
        with self.assertRaisesRegex(RuntimeError, "exact coverage"):
            admit_source(self.root, spec, normalized, missing, **kwargs)
        duplicated = {**normalized, "chunks": normalized["chunks"] + normalized["chunks"][:1]}
        with self.assertRaisesRegex(RuntimeError, "exact coverage"):
            admit_source(self.root, spec, duplicated, results, **kwargs)
        pending = {key: {**value, "status": "ELIGIBLE_PENDING_OBJECT_COMPLETION"}
                   for key, value in results.items()}
        with self.assertRaisesRegex(RuntimeError, "Pending"):
            admit_source(self.root, spec, normalized, pending, **kwargs)
        false_counts = {key: {**value, "input_documents": value["input_documents"] + 1}
                        for key, value in results.items()}
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            admit_source(self.root, spec, normalized, false_counts, **kwargs)
        self.assertFalse((self.root / "admissions" / f"{digest_json(spec.source_id)}.json").exists())

    def test_cli_reaches_actual_prep_service(self):
        with patch("metis_data17.worker.prep_service") as run:
            self.assertEqual(main(["prep", "--root", str(self.root), "--workers", "9", "--raw-readers", "2"]), 0)
        self.assertEqual(run.call_args.args, (self.root,))
        kwargs = dict(run.call_args.kwargs)
        self.assertIs(kwargs.pop("defer_compaction", False), False)
        self.assertEqual(kwargs, {
            "workers": 9, "raw_readers": 2, "idle_seconds": 600, "maximum_seconds": 42000,
        })

    def test_independent_compaction_visits_every_bucket_without_blocking_index_admission(self):
        spec, raw, output = self._object()
        config = {**self.config, "defer_compaction": True}
        normalized = fixtures.reblock_object(spec, raw, output, config)
        with patch("metis_data17.dedup.compact_dedup", side_effect=AssertionError("Synchronous compaction")):
            for chunk in normalized["chunks"]:
                screened = screen_chunk(self.root / chunk["ready_receipt"], spec, config)
                index_chunk(self.root / screened["receipt_path"], spec, config)
        write_receipt(self.root / "limits.json", {
            "capacity_confirmation": "unlimited", "max_raw_bytes": 100_000_000,
            "max_working_bytes": 200_000_000, "policy_and_metadata_reserve_bytes": 1_000_000,
            "filesystem_free_floor_bytes": 0,
        })
        from metis_data17.dedup import iter_survivors

        exact = self.root / "dedup" / "exact" / config["index_generation"]
        self.assertTrue((exact / "INDEX.json").is_file())
        before = list(iter_survivors(exact))
        buckets = config["dedup"]["bucket_count"]
        seen = []
        for _ in range(buckets):
            bucket, lease = _claim_compaction(config)
            try:
                progress = _compact_job({**config, "compaction_bucket": bucket})
            finally:
                lease.close()
            seen.append(bucket)
            self.assertLessEqual(progress["merges"], 1)
        self.assertEqual(seen, list(range(buckets)))
        self.assertEqual(list(iter_survivors(exact)), before)
        self.assertEqual(
            read_receipt(self.root / "state" / "compaction" / "fixture-index.json")["next_bucket"], 0,
        )

    def test_parallel_compactors_receive_disjoint_buckets_and_cover_the_whole_partition(self):
        leases = []
        try:
            for _ in range(self.config["dedup"]["bucket_count"]):
                leases.append(_claim_compaction(self.config))
            self.assertEqual([bucket for bucket, _ in leases], list(range(7)))
            self.assertIsNone(_claim_compaction(self.config))
        finally:
            for _, lease in leases:
                lease.close()
        bucket, lease = _claim_compaction(self.config)
        try:
            self.assertEqual(bucket, 0)
        finally:
            lease.close()

    def test_deferred_workers_reserve_a_process_that_can_relieve_backpressure(self):
        with patch("metis_data17.worker.worker_configuration", return_value=(
            self.config, {"config": {"limits": {"capacity_confirmation": "pending"}}},
        )), patch("metis_data17.worker.prep.prepare_runtime"), patch(
            "metis_data17.worker.code_commit", return_value="fixture",
        ), patch("metis_data17.worker._stop_event", return_value=threading.Event()), patch(
            "metis_data17.worker.ProcessPoolExecutor", side_effect=RuntimeError("pool sentinel"),
        ) as pool:
            with self.assertRaisesRegex(RuntimeError, "pool sentinel"):
                prep_service(self.root, workers=3, raw_readers=2, defer_compaction=True)
        self.assertEqual(pool.call_args.kwargs["max_workers"], 6)

    def test_dispatcher_covers_both_uplinks_and_screens_before_raw_eof(self):
        first, first_raw, _ = self._object()
        second, second_raw, _ = self._object()
        for host, spec, raw in (("uplink1", first, first_raw), ("uplink2", second, second_raw)):
            event = {**raw.to_dict(), "spec": spec.to_dict()}
            write_receipt(receipt_path(self.root, spec.object_id), event)
            append_event(self.root, f"raw/{host}", event)
        config = {
            **self.config, "raw_readers_per_node": 1,
            "source_minimum_acceptance": 0.1, "opt_out_snapshot": self.fixture._opt_out(),
        }
        run = {"config": {"limits": {"capacity_confirmation": "pending"}}}
        original_reader = fixtures.prep.iter_source_rows
        original_screen = screen_chunk
        seen_pending = threading.Event()
        real_sleep = time.sleep

        def slow_reader(spec, *args, **kwargs):
            for index, row in enumerate(original_reader(spec, *args, **kwargs)):
                yield row
                if spec.object_id == first.object_id and index == 7:
                    if not seen_pending.wait(10):
                        raise RuntimeError("The dispatcher did not screen a ready chunk before EOF")

        def observed_screen(*args):
            result = original_screen(*args)
            if result["status"] == "ELIGIBLE_PENDING_OBJECT_COMPLETION":
                seen_pending.set()
            return result

        with patch("metis_data17.worker.worker_configuration", return_value=(config, run)), patch(
            "metis_data17.worker.code_commit", return_value="a" * 40,
        ), patch("metis_data17.worker.ProcessPoolExecutor",
                 side_effect=lambda max_workers, **_: ThreadPoolExecutor(max_workers=max_workers)), patch(
            "metis_data17.worker.time.sleep", side_effect=lambda _: real_sleep(0.01),
        ), patch("metis_data17.prep.iter_source_rows", side_effect=slow_reader), patch(
            "metis_data17.worker.screen_chunk", side_effect=observed_screen,
        ):
            prep_service(self.root, workers=2, idle_seconds=0.1, maximum_seconds=20)
        self.assertTrue(seen_pending.is_set())
        completed = [
            read_receipt(path) for path in
            (self.root / "state" / "prepared-objects" / config["index_generation"]).glob("*/*.json")
        ]
        self.assertEqual({row["object_id"] for row in completed}, {first.object_id, second.object_id})
        self.assertEqual(sum(row["eligible_documents"] for row in completed), 28)
        self.assertFalse(list((self.root / "state" / "prep-failures").glob("**/*.json")))


class Metis17JournalTailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "events" / "raw" / "host.jsonl"

    def test_interrupted_append_does_not_skip_next_complete_record(self):
        cursor = EventTail()
        append_event(self.root, "raw/host", {"object_id": "first"})
        self.assertEqual([row["object_id"] for row in cursor.read(self.path)], ["first"])
        with self.path.open("ab") as stream:
            stream.write(b'{"partial":')
        self.assertEqual(cursor.read(self.path), [])
        append_event(self.root, "raw/host", {"object_id": "second"})
        self.assertEqual([row["object_id"] for row in cursor.read(self.path)], ["second"])
        self.assertEqual(cursor.read(self.path), [])

    def test_journal_truncation_and_corrupt_seals_are_errors_not_zero_work(self):
        append_event(self.root, "raw/host", {"object_id": "first"})
        cursor = EventTail()
        cursor.read(self.path)
        self.path.write_bytes(b"")
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            cursor.read(self.path)
        append_event(self.root, "raw/host", {"object_id": "second"})
        value = json.loads(self.path.read_text())
        value["object_id"] = "tampered"
        self.path.write_text(json.dumps(value) + "\n")
        with self.assertRaisesRegex(RuntimeError, "Corrupt"):
            EventTail().read(self.path)


if __name__ == "__main__":
    unittest.main()
