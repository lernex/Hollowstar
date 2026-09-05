from __future__ import annotations

import json
import itertools
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from metis_data17.cli import append_event
from metis_data17.common import digest_json, read_receipt, write_receipt
from metis_data17.tokenizer_service import tokenize_event, tokenizer_service
from metis_data17.worker import EventTail


class TokenizerServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.generation = "a" * 64
        self.scratch = self.root / "scratch"
        self.stage_path = self.root / "prepared" / "stage.json"
        self.stage = {
            "status": "ELIGIBLE", "eligible": True, "training_ready": True,
            "object_complete": True, "chunks": [{"path": "prepared/data.parquet"}],
        }
        self.event = self.publish(self.stage)

    def publish(self, stage):
        write_receipt(self.stage_path, stage)
        return {
            "generation": self.generation,
            "receipt_path": str(self.stage_path.relative_to(self.root)),
            "stage_receipt_sha256": digest_json(stage),
        }

    def test_explicit_event_partition_is_reused_without_tokenizing_twice(self):
        with patch("metis_data17.tokenizer_service.tokenize_ready_partition",
                   return_value={"partition_root": "tokenizer/partition"}) as cache:
            first = tokenize_event(self.root, self.event, generation=self.generation, scratch_dir=self.scratch)
            second = tokenize_event(self.root, self.event, generation=self.generation, scratch_dir=self.scratch)
        self.assertEqual(first, second)
        cache.assert_called_once()
        self.assertEqual(cache.call_args.kwargs["partition_id"], digest_json(self.stage))

    def test_superseded_events_do_not_open_old_artifacts(self):
        event = {**self.event, "generation": "b" * 64, "receipt_path": "missing/stage.json"}
        self.assertIsNone(tokenize_event(self.root, event, generation=self.generation, scratch_dir=self.scratch))

    def test_unsealed_pending_or_repeated_inputs_never_enter_cache(self):
        cases = [
            {**self.stage, "status": "ELIGIBLE_PENDING_OBJECT_COMPLETION", "object_complete": False},
            {**self.stage, "training_ready": False},
            {**self.stage, "chunks": self.stage["chunks"] * 2},
        ]
        for stage in cases:
            with self.subTest(stage=stage), patch("metis_data17.tokenizer_service.tokenize_ready_partition") as cache:
                event = self.publish(stage)
                with self.assertRaises(RuntimeError):
                    tokenize_event(self.root, event, generation=self.generation, scratch_dir=self.scratch)
                cache.assert_not_called()
        event = self.publish(self.stage)
        write_receipt(self.stage_path, {**self.stage, "changed": True})
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            tokenize_event(self.root, event, generation=self.generation, scratch_dir=self.scratch)

    def test_journal_batches_cover_every_committed_event_exactly_once(self):
        for index in range(7):
            append_event(self.root, "eligible/worker", {"sequence": index})
        path = self.root / "events" / "eligible" / "worker.jsonl"
        tail = EventTail()
        received = []
        while batch := tail.read(path, maximum_events=2):
            self.assertLessEqual(len(batch), 2)
            received.extend(row["sequence"] for row in batch)
        self.assertEqual(received, list(range(7)))
        self.assertEqual(tail.positions[path][1], path.stat().st_size)
        with self.assertRaises(ValueError):
            tail.read(path, maximum_events=0)

    def run_service(self, steps, stop, cache):
        with patch("metis_data17.tokenizer_service.worker_configuration",
                   return_value=({"generation": self.generation}, {})), patch(
            "metis_data17.tokenizer_service.code_commit", return_value="c" * 40,
        ), patch("metis_data17.tokenizer_service._stop_event", return_value=stop), patch(
            "metis_data17.tokenizer_service.run_tokenizer_step",
            side_effect=itertools.chain(steps, itertools.repeat(steps[-1])),
        ), patch("metis_data17.tokenizer_service.tokenize_ready_partition", cache), patch.object(
            stop, "wait", side_effect=lambda seconds: stop.set(),
        ):
            tokenizer_service(self.root, scratch_dir=self.scratch, poll_seconds=0.01)

    def test_waiting_tokenizer_cannot_start_id_caching(self):
        append_event(self.root, "eligible/worker", self.event)
        with patch("metis_data17.tokenizer_service.tokenize_ready_partition") as cache:
            self.run_service([{"status": "WAITING"}], threading.Event(), cache)
        cache.assert_not_called()
        self.assertFalse((self.root / "state" / "tokenizer-dispatch").exists())

    def test_training_gate_and_durable_dispatch_cursor_survive_restart(self):
        from unittest.mock import Mock

        append_event(self.root, "eligible/worker", self.event)
        cache = Mock(return_value={"partition_root": "tokenizer/partition"})
        self.run_service([{"status": "SAMPLE_READY"}, {"status": "TRAINED"}], threading.Event(), cache)
        cache.assert_called_once()
        cursor = read_receipt(self.root / "state" / "tokenizer-dispatch" / f"{self.generation}.json")
        self.assertEqual(cursor["processed_eligible_events"], 1)
        self.assertEqual(cursor["journals"][0]["offset"],
                         (self.root / "events" / "eligible" / "worker.jsonl").stat().st_size)
        self.run_service([{"status": "TRAINED"}], threading.Event(), cache)
        cache.assert_called_once()
        progress = json.loads((self.root / "status" / "tokenizer.json").read_text())
        self.assertEqual(progress["processed_eligible_events"], 1)


class TokenizerServiceIntegrationTests(unittest.TestCase):
    def test_real_post_freeze_preparation_receipts_from_two_journals_use_bounded_partition_proofs(self):
        from metis_data17 import tokenizer as tk
        from metis_data17 import tokenizer_pipeline as pipeline
        from metis_data17.prep import prepare_chunk, reblock_object
        from metis_data17.storage import WorkingBudget
        from tests.test_metis17_prep import Metis17PreparationTests
        from tests.test_metis17_tokenizer_pipeline import TokenizerPipelineTests

        fixture = TokenizerPipelineTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config["tokenizer"]["max_events_per_step"] = 1
        fixture.config["tokenizer"]["max_input_paths"] = 5
        fixture.write_run()
        write_receipt(fixture.root / "limits.json", {
            "max_raw_bytes": 1_000_000, "max_working_bytes": 2_000_000_000,
            "capacity_confirmation": "pending", "policy_and_metadata_reserve_bytes": 1_000_000,
            "filesystem_free_floor_bytes": 0,
        })
        budget = WorkingBudget(fixture.root)
        for category in pipeline.REQUIRED_CATEGORIES:
            event, _, _ = fixture.source(category, append=False)
            append_event(fixture.root, "eligible/training", event)
        for _ in range(10):
            state = fixture.step(working_budget=budget)
            self.assertNotEqual(state["status"], "BLOCKED", state.get("error"))
            if state["status"] == "TRAINED":
                break
        self.assertEqual(state["status"], "TRAINED")
        frozen_sample, frozen_inputs = state["sample"], state["chunks"]

        producer = Metis17PreparationTests()
        producer.setUp()
        self.addCleanup(producer.doCleanups)
        producer.root = fixture.root
        producer.config = {**producer.config, "root": fixture.root, "generation": fixture.generation}
        late = []
        for host in ("a-late", "b-late"):
            spec, raw, output = producer._object([
                {"text": producer.SAFE + f" New independent {host} paragraph."},
            ])
            normalized = reblock_object(spec, raw, output, producer.config)
            self.assertEqual(len(normalized["chunks"]), 1)
            result = prepare_chunk(
                fixture.root / normalized["chunks"][0]["ready_receipt"], output, producer.config,
            )
            self.assertEqual(result["status"], "ELIGIBLE")
            event = {
                "generation": fixture.generation, "source_id": spec.source_id, "object_id": spec.object_id,
                "receipt_path": result["receipt_path"],
                "stage_receipt_sha256": digest_json(read_receipt(fixture.root / result["receipt_path"])),
            }
            append_event(fixture.root, f"eligible/{host}", event)
            late.append(event)

        def service():
            stop = threading.Event()
            with patch("metis_data17.tokenizer_service.worker_configuration",
                       return_value=({"generation": fixture.generation}, {})), patch(
                "metis_data17.tokenizer_service.code_commit", return_value="fixture",
            ), patch("metis_data17.tokenizer_service._stop_event", return_value=stop), patch.object(
                stop, "wait", side_effect=lambda _seconds: stop.set(),
            ):
                tokenizer_service(
                    fixture.root, scratch_dir=fixture.scratch, poll_seconds=0.001,
                    test_mode=True, working_budget=budget,
                )

        service()
        for event in late:
            seal = event["stage_receipt_sha256"]
            marker = read_receipt(
                fixture.root / "state" / "tokenized-chunks" / fixture.generation / seal[:2] / f"{seal}.json",
            )
            self.assertEqual(marker["result"]["stage_receipt_sha256"], seal)
            self.assertTrue((fixture.root / marker["result"]["partition_root"] / "PARTITION_RECEIPT.json").is_file())
        after = pipeline._read(fixture.work / "STATE.json")
        self.assertEqual(after["chunks"], frozen_inputs)
        self.assertEqual(after["sample"], frozen_sample)
        self.assertLessEqual(len(after.get("recent_partition_inputs", {})), 1)
        with patch.object(tk, "_encode_batch", side_effect=AssertionError("Completed partitions were re-encoded")):
            service()


if __name__ == "__main__":
    unittest.main()
