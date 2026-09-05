from __future__ import annotations

import json
import fnmatch
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from metis_data17.acquisition import CapacityPending, DownloadFailure
from metis_data17.cli import (
    _limits, activate_batch, append_event, download_order_key, download_owner, download_service, init_run,
    intake_candidate_fits, main, read_events, select_download_group, status,
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

    def test_recipe_corrects_legacy_training_target_without_mutating_run(self):
        value = yaml.safe_load(self.config.read_text())
        value["tokenizer"]["sample_target_bytes"] = 160_000_000_000
        path = Path(self.tmp.name) / "legacy.yaml"
        path.write_text(yaml.safe_dump(value))
        with patch("metis_data17.cli.code_commit", return_value="a" * 40):
            init_run(self.root, path)
        before = (self.root / "RUN.json").read_bytes()
        self.assertEqual(main(["freeze-tokenizer-recipe", "--root", str(self.root),
                               "--config", str(self.config.with_name("tokenizer-recipe.yaml"))]), 0)
        recipe = read_receipt(self.root / "tokenizer" / "RECIPE.json")
        self.assertEqual(recipe["tokenizer"]["sample_target_bytes"], 150_000_000_000)
        self.assertEqual(recipe["tokenizer"]["vocabulary_size"], 131072)
        self.assertEqual(recipe["tokenizer"]["min_sample_bytes_per_category"], 30_000_000_000)
        self.assertEqual((self.root / "RUN.json").read_bytes(), before)

    def test_status_uses_validated_active_generation_not_a_flat_artifact_filename(self):
        with patch("metis_data17.cli.code_commit", return_value="a" * 40):
            init_run(self.root, self.config)
        flat = self.root / "tokenizer" / "TOKENIZER_RELEASE.json"
        flat.parent.mkdir(exist_ok=True)
        flat.write_text("{}")
        expected = {"ready": False, "status": "WAITING"}
        with patch("metis_data17.policy.policy_config", return_value={"policy_ready": True}), patch(
            "metis_data17.worker.worker_configuration", return_value=({"generation": "b" * 64}, {}),
        ), patch("metis_data17.tokenizer_pipeline.tokenizer_status", return_value=expected) as current:
            result = status(self.root)
        current.assert_called_once_with(self.root, generation="b" * 64)
        self.assertFalse(result["tokenizer_ready"])
        self.assertEqual(result["tokenizer"], expected)

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

    def test_deferred_compaction_reaches_both_worker_and_supervisor(self):
        with patch("metis_data17.worker.prep_service") as worker:
            self.assertEqual(main(["prep", "--root", str(self.root), "--defer-compaction"]), 0)
        self.assertIs(worker.call_args.kwargs["defer_compaction"], True)
        with patch("metis_data17.runtime.supervise_prep") as supervise:
            self.assertEqual(main(["supervise-prep", "--root", str(self.root),
                                   "--python", sys.executable, "--defer-compaction"]), 0)
        self.assertIs(supervise.call_args.kwargs["defer_compaction"], True)

    def test_no_reconstruction_source_kind(self):
        value = yaml.safe_load(self.config.read_text())
        value["sources"][0]["kind"] = "github_repositories"
        config = Path(self.tmp.name) / "bad.yaml"
        config.write_text(yaml.safe_dump(value))
        with self.assertRaises(ValueError):
            init_run(self.root, config)

    def test_complement_selectors_match_real_hive_paths_and_exclude_token_indexes(self):
        path = self.config.with_name("expansion-complements.yaml")
        sources = {source["id"]: source for source in yaml.safe_load(path.read_text())["sources"]}
        essential = sources["essential_web_2023_2024"]["allow_patterns"]
        self.assertTrue(any(fnmatch.fnmatchcase("data/crawl=CC-MAIN-2024-38/part.parquet", pattern)
                            for pattern in essential))
        self.assertFalse(any(fnmatch.fnmatchcase("data/crawl=CC-MAIN-2022-49/part.parquet", pattern)
                             for pattern in essential))
        code = sources["dolma35_materialized_code"]
        self.assertTrue(any(fnmatch.fnmatchcase("swallow-code/allenai/dolma2-tokenizer/0000.csv.gz", pattern)
                            for pattern in code["deny_patterns"]))
        self.assertEqual(code["expected_inventory_bytes"] + 1_098_537_316, 2_458_368_343_174)

    def test_expansion_activates_a_separate_batch_without_mutating_frozen_run(self):
        with patch("metis_data17.cli.code_commit", return_value="a" * 40):
            init_run(self.root, self.config)
        frozen = (self.root / "RUN.json").read_bytes()
        path = Path(self.tmp.name) / "expansion.yaml"
        source = {"id": "additional", "kind": "hf"}
        path.write_text(yaml.safe_dump({"schema": "metis17.activation/v1", "sources": [source]}))
        with patch("metis_data17.cli.resolve_source", return_value={"objects": 4, "known_bytes": 128}) as resolve:
            result = activate_batch(self.root, path)
        resolve.assert_called_once_with(self.root, source)
        self.assertEqual((self.root / "RUN.json").read_bytes(), frozen)
        self.assertEqual(result["sources"][0]["objects"], 4)
        self.assertTrue((self.root / "activations" / result["batch_id"] / "BATCH.json").is_file())

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

    def test_fast_origin_is_not_limited_to_one_slot_behind_a_throttled_origin(self):
        def spec(origin, name, priority):
            return ObjectSpec.create(
                source_id="source", url=f"https://{origin}/{name}", revision="r",
                relative_key=name, wire_format="parquet", adapter="text", priority=priority,
            )
        high = spec("slow.test", "premium", 100)
        fast = spec("fast.test", "fresh", 50)
        specs = {value.object_id: value for value in (high, fast)}
        choices = [(download_order_key(high, set()), "premium"),
                   (download_order_key(fast, set()), "fresh")]
        self.assertEqual(select_download_group(choices, specs, [high] * 15 + [fast])[1], "fresh")

    def test_shared_origin_partition_covers_every_object_exactly_once(self):
        hosts = {"login1": ["hf.test"], "login2": ["cc.test"]}
        specs = [
            ObjectSpec.create(
                source_id="source", url=f"https://cc.test/{index}", revision="r",
                relative_key=str(index), wire_format="parquet", adapter="text", priority=50,
            )
            for index in range(257)
        ]
        partials = {specs[0].object_id: "login2.cluster"}
        assignments = {
            host: {spec.object_id for spec in specs
                   if download_owner(spec, hosts, {"cc.test"}, partials) == host}
            for host in hosts
        }
        self.assertTrue(all(assignments.values()))
        self.assertFalse(assignments["login1"] & assignments["login2"])
        self.assertEqual(set.union(*assignments.values()), {spec.object_id for spec in specs})
        self.assertIn(specs[0].object_id, assignments["login2"])
        self.assertTrue(all(download_owner(spec, hosts, set(), {}) == "login2" for spec in specs))
        bootstrap = ObjectSpec.from_dict({**specs[1].to_dict(), "policy": {"bootstrap": True}})
        self.assertEqual(download_owner(bootstrap, hosts, {"cc.test"}, {}), "login2")

    def test_failed_activation_continues_independent_sources_and_returns_failure(self):
        with patch("metis_data17.cli.code_commit", return_value="a" * 40):
            init_run(self.root, self.config)
        path = Path(self.tmp.name) / "expansion.yaml"
        sources = [{"id": "bad", "kind": "hf"}, {"id": "good", "kind": "hf"}]
        path.write_text(yaml.safe_dump({"schema": "metis17.activation/v1", "sources": sources}))
        with patch("metis_data17.cli.resolve_source",
                   side_effect=[RuntimeError("Incomplete publisher inventory"),
                                {"objects": 2, "known_bytes": 128}]):
            result = activate_batch(self.root, path)
        self.assertFalse(result["ok"])
        self.assertEqual([value["status"] for value in result["sources"]], ["failed", "complete"])

    def _download_with_failed_head(self, failure):
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
                except (CapacityPending, DownloadFailure) as exc:
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
            "max_unknown_object_bytes": 5,
        })
        write_receipt(self.root / "state" / "intake-budget.json", {
            "schema": "metis17.intake-budget/v1", "raw_bytes": 0, "network_payload_bytes": 0,
            "sources": {}, "inflight": {},
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
                raise failure
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
        progress = json.loads((self.root / "status" / "download-fixture-host.json").read_text())
        return attempted, progress

    def test_oversized_candidate_does_not_pause_other_downloads(self):
        attempted, progress = self._download_with_failed_head(CapacityPending("Does not fit remaining reservation"))
        self.assertEqual(attempted, ["large", "small"])
        self.assertEqual(progress["completed_objects"], 1)

    def test_budget_snapshot_does_not_depend_on_racy_file_timestamps(self):
        original_stat, original_exists = Path.stat, Path.exists

        def stat(path, *args, **kwargs):
            if path.name == "intake-budget.json":
                raise FileNotFoundError("Replacement raced an unlocked metadata probe")
            return original_stat(path, *args, **kwargs)

        def exists(path):
            return True if path.name == "intake-budget.json" else original_exists(path)

        with patch.object(Path, "stat", stat), patch.object(Path, "exists", exists):
            attempted, progress = self._download_with_failed_head(CapacityPending("Large object does not fit"))
        self.assertEqual(attempted, ["large", "small"])
        self.assertEqual(progress["completed_objects"], 1)
        self.assertEqual(progress["capacity_blocked_objects"], 1)

    def test_retry_backoff_does_not_stall_other_objects_in_the_same_source(self):
        attempted, progress = self._download_with_failed_head(DownloadFailure("Retry this object later"))
        self.assertEqual(attempted, ["large", "small"])
        self.assertEqual(progress["completed_objects"], 1)
    def test_capacity_hint_avoids_requests_without_losing_resumable_objects(self):
        spec = ObjectSpec.create(
            source_id="source", url="https://example.test/object", revision="r",
            relative_key="object", wire_format="parquet", adapter="text", priority=100,
            expected_bytes=100,
        )
        limits = {"max_raw_bytes": 100, "max_unknown_object_bytes": 50}
        self.assertFalse(intake_candidate_fits(spec, {"raw_bytes": 90, "inflight": {}}, limits))
        self.assertTrue(intake_candidate_fits(
            spec, {"raw_bytes": 0, "inflight": {spec.object_id: {"bytes": 100}}}, limits,
        ))
        unknown = ObjectSpec.create(
            source_id="source", url="https://example.test/unknown", revision="r",
            relative_key="unknown", wire_format="parquet", adapter="text", priority=100,
        )
        self.assertFalse(intake_candidate_fits(unknown, {"raw_bytes": 99, "inflight": {}}, limits))
        self.assertTrue(intake_candidate_fits(unknown, {"raw_bytes": 0, "inflight": {}}, limits))


if __name__ == "__main__":
    unittest.main()
