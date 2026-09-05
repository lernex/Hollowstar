from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from metis_data17.acquisition import CapacityPending
from metis_data17.cli import (
    _limits, append_event, download_order_key, download_service, init_run,
    main, read_events, select_download_group, status,
)
from metis_data17.common import ObjectSpec, RawReceipt, digest_json, read_receipt, write_receipt


class Metis17CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "metis-1.7-test"
        self.config = Path(__file__).resolve().parents[1] / "configs/metis17/pipeline.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def test_initial_run_is_explicitly_capacity_bounded(self):
        with patch("metis_data17.cli.code_commit", return_value="a" * 40):
            run = init_run(self.root, self.config)
        self.assertFalse(run["full_capacity_approved"])
        self.assertEqual(run["config"]["tokenizer"]["vocabulary_size"], 131072)
        self.assertIs(run["config"]["tokenizer"]["split_digits"], True)
        self.assertEqual(_limits(self.root)["max_raw_bytes"], 400_000_000_000)
        self.assertEqual(status(self.root)["downloaders"], {})
        with patch("metis_data17.cli.code_commit", return_value="b" * 40):
            self.assertEqual(init_run(self.root, self.config), run)

    def test_unconfirmed_limit_cannot_be_silently_expanded(self):
        with patch("metis_data17.cli.code_commit", return_value="a" * 40):
            init_run(self.root, self.config)
        limits = read_receipt(self.root / "limits.json")
        limits["max_raw_bytes"] = 200_000_000_000_000
        write_receipt(self.root / "limits.json", limits)
        with self.assertRaises(CapacityPending):
            _limits(self.root)

    def test_download_and_derived_storage_share_capacity_confirmation_rules(self):
        from metis_data17.storage import WorkingBudget

        cases = [
            ("pending", 400_000_000_000, 2_000_000_000_000, True),
            ("administrator-confirmed", 200_000_000_000_000, 800_000_000_000_000, True),
            ("unlimited", 200_000_000_000_000, 800_000_000_000_000, True),
            ("pending", 400_000_000_001, 2_000_000_000_000, False),
            ("pending", 400_000_000_000, 2_000_000_000_001, False),
            ("administrator-confirmed", 200_000_000_000_001, 800_000_000_000_000, False),
            ("assumed-approved", 400_000_000_000, 2_000_000_000_000, False),
            ("pending", True, 2_000_000_000_000, False),
            ("pending", 0, 2_000_000_000_000, False),
        ]
        for confirmation, raw, working, allowed in cases:
            with self.subTest(confirmation=confirmation, raw=raw, working=working):
                write_receipt(self.root / "limits.json", {
                    "capacity_confirmation": confirmation, "max_raw_bytes": raw,
                    "max_working_bytes": working,
                })
                if allowed:
                    self.assertEqual(_limits(self.root)["max_raw_bytes"],
                                     WorkingBudget(self.root).snapshot()["max_raw_bytes"])
                else:
                    with self.assertRaises((ValueError, CapacityPending)):
                        _limits(self.root)
                    with self.assertRaises((ValueError, CapacityPending)):
                        WorkingBudget(self.root)

    def test_digit_splitting_cannot_be_disabled_by_configuration(self):
        value = yaml.safe_load(self.config.read_text())
        value["tokenizer"]["split_digits"] = False
        config = Path(self.tmp.name) / "bad.yaml"
        config.write_text(yaml.safe_dump(value))
        with self.assertRaises(ValueError):
            init_run(self.root, config)

    def test_supervision_preserves_the_virtualenv_launcher_symlink(self):
        launcher = Path(self.tmp.name) / "runtime" / "bin" / "python"
        launcher.parent.mkdir(parents=True)
        launcher.symlink_to(Path(sys.executable).resolve())
        self.assertNotEqual(launcher, launcher.resolve())
        with patch("metis_data17.runtime.supervise_prep") as supervise:
            self.assertEqual(main(["supervise-prep", "--root", str(self.root),
                                   "--python", str(launcher)]), 0)
        self.assertEqual(supervise.call_args.args[2], launcher)
        self.assertTrue(supervise.call_args.args[2].is_symlink())

    def test_no_reconstruction_source_kind(self):
        value = yaml.safe_load(self.config.read_text())
        value["sources"][0]["kind"] = "github_repositories"
        config = Path(self.tmp.name) / "bad.yaml"
        config.write_text(yaml.safe_dump(value))
        with self.assertRaises(ValueError):
            init_run(self.root, config)

    def test_journal_recovers_only_uncommitted_tail(self):
        append_event(self.root, "raw/host", {"object_id": "first"})
        path = self.root / "events/raw/host.jsonl"
        with path.open("ab") as stream:
            stream.write(b'{"interrupted":')
        self.assertEqual([r["object_id"] for r in read_events(path)], ["first"])
        append_event(self.root, "raw/host", {"object_id": "second"})
        self.assertEqual([r["object_id"] for r in read_events(path)], ["first", "second"])

    def test_corrupt_committed_journal_record_fails(self):
        append_event(self.root, "raw/host", {"object_id": "first"})
        path = self.root / "events/raw/host.jsonl"
        row = json.loads(path.read_text())
        row["object_id"] = "changed"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaises(RuntimeError):
            list(read_events(path))

    def test_partial_progress_precedes_new_work_without_changing_quality_order(self):
        def spec(name, priority):
            return ObjectSpec.create(
                source_id="source", url=f"https://example.test/{name}", revision="r",
                relative_key=name, wire_format="parquet", adapter="text", priority=priority,
                policy={"bootstrap": True},
            )
        partial = spec("partial", 10)
        high = spec("high", 100)
        low = spec("low", 20)
        resumable = {partial.object_id}
        ordered = sorted([low, high, partial], key=lambda item: download_order_key(item, resumable))
        self.assertEqual(ordered, [partial, high, low])

    def test_independent_origin_gets_a_slot_without_changing_per_origin_priority(self):
        def spec(origin, name, priority):
            return ObjectSpec.create(
                source_id="source", url=f"https://{origin}/{name}", revision="r",
                relative_key=name, wire_format="parquet", adapter="text", priority=priority,
            )
        busy = spec("slow.test", "busy", 99)
        high = spec("slow.test", "next", 98)
        other = spec("fast.test", "fresh", 50)
        specs = {s.object_id: s for s in (busy, high, other)}
        candidates = [(download_order_key(high, set()), "high"), (download_order_key(other, set()), "fresh")]
        self.assertEqual(select_download_group(candidates, specs, [busy])[1], "fresh")
        self.assertEqual(select_download_group(candidates, specs, [busy, other])[1], "high")

    def test_oversized_candidate_does_not_pause_other_downloads(self):
        from concurrent.futures import Future

        class ImmediatePool:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def submit(self, function, *args, **kwargs):
                future = Future()
                try:
                    future.set_result(function(*args, **kwargs))
                except CapacityPending as exc:
                    future.set_exception(exc)
                return future

        stop = threading.Event()
        config = {"download": {
            "hosts": {"fixture-host": ["example.test"]}, "workers_per_host": 1,
            "attempts": 1, "timeout_seconds": 1, "admission_objects_per_group": 2,
            "poll_seconds": 0.01,
        }}
        write_receipt(self.root / "RUN.json", {"config": config, "config_sha256": digest_json(config)})
        write_receipt(self.root / "limits.json", {
            "capacity_confirmation": "pending", "max_raw_bytes": 5, "max_working_bytes": 50,
        })
        specs = [
            ObjectSpec.create(
                source_id="source", url=f"https://example.test/{name}", revision="r",
                relative_key=name, wire_format="parquet", adapter="text", priority=priority,
            )
            for name, priority in (("large", 100), ("small", 90))
        ]
        write_receipt(self.root / "catalogue" / "active" / "source.json", {
            "directory": "catalogue/fixture", "source_hash": "fixture",
        })
        write_receipt(self.root / "catalogue" / "fixture" / "page-000000.json", {
            "source_hash": "fixture", "objects": [spec.to_dict() for spec in specs],
        })
        attempted = []
        ticks = []

        def download(spec, *_args, **_kwargs):
            attempted.append(spec.relative_key)
            if spec.relative_key == "large":
                raise CapacityPending("Does not fit remaining reservation")
            return RawReceipt(spec.object_id, spec.source_id, "raw/small", 5, "a" * 64,
                              "fixture-host", "2026-09-05T00:00:00+00:00")

        def tick(_seconds):
            ticks.append(1)
            if len(ticks) >= 5:
                stop.set()

        with patch("metis_data17.cli._stop_event", return_value=stop), patch(
            "metis_data17.cli.socket.gethostname", return_value="fixture-host",
        ), patch("metis_data17.cli.code_commit", return_value="a" * 40), patch(
            "metis_data17.cli.ThreadPoolExecutor", ImmediatePool,
        ), patch("metis_data17.cli.download_object", side_effect=download), patch(
            "metis_data17.cli.time.sleep", side_effect=tick,
        ):
            download_service(self.root)
        self.assertEqual(attempted, ["large", "small"])
        progress = json.loads((self.root / "status" / "download-fixture-host.json").read_text())
        self.assertEqual(progress["completed_objects"], 1)
        self.assertEqual(progress["capacity_blocked_objects"], 1)


if __name__ == "__main__":
    unittest.main()
