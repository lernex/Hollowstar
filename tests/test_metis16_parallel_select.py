"""The parallel selection must decide exactly what the serial one decides.

Selection has two implementations now: the readable one that routes and writes
in a single stream, and the one that routes text-free and materialises across a
node. The whole point of the second is that it is not a second set of rules, so
these pin the two together on real shard files rather than checking each in
isolation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metis_data.select_parallel import (  # noqa: E402
    _block,
    _stripe,
    build_selection_parallel,
)
from metis_data.selection import build_selection  # noqa: E402
from metis_data.stage_runner import _iter_rows  # noqa: E402


def _manifest() -> dict:
    return {
        "selection": {"seed": 7, "replay": {"maximum_document_exposures": 4}},
        "schedule": {
            "target_tokens": 1000,
            "phases": {
                "phase_a": {"target_tokens": 600, "replay_tokens": 0},
                "phase_b": {"target_tokens": 300, "replay_tokens": 0},
                "phase_c": {"target_tokens": 100, "replay_tokens": 100},
            },
        },
        "sources": [
            {"id": "alpha", "phase_tokens": {"phase_a": 360, "phase_b": 180, "phase_c": 60}},
            {"id": "beta", "phase_tokens": {"phase_a": 240, "phase_b": 120, "phase_c": 40}},
        ],
    }


def _write_shards(root: Path, shards: int = 4, beta_every: int = 3) -> list[Path]:
    """A corpus split across real zstd shards, interleaved by source."""

    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index in range(240):
        source = "beta" if index % beta_every == 0 else "alpha"
        rows.append(
            {
                "source_id": source,
                "doc_id": f"{source}-{index:04d}",
                "text": f"document {index} " + ("lorem ipsum " * (index % 7 + 1)),
                "token_count": 5 + index % 4,
                "generated": index % 11 == 0,
                "transformed": index % 13 == 0,
                "content_sha256": f"{index:064x}",
                "text_sha256": f"{index + 1:064x}",
                "license": "cc-by",
                "license_status": "resolved",
            }
        )
    paths = []
    for shard in range(shards):
        path = root / f"shard-{shard:05d}.jsonl.zst"
        payload = b"".join(
            json.dumps(row, sort_keys=True).encode() + b"\n"
            for row in rows[shard::shards]
        )
        path.write_bytes(zstd.ZstdCompressor(level=1).compress(payload))
        paths.append(path)
    return paths


def _shard_lines(path: Path) -> list[dict]:
    with path.open("rb") as raw:
        with zstd.ZstdDecompressor().stream_reader(raw, read_across_frames=True) as stream:
            with io.TextIOWrapper(stream, encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]


def _donor_manifest() -> dict:
    """A manifest where beta cannot fill its own quota and alpha donates."""

    manifest = _manifest()
    for source in manifest["sources"]:
        source["category"] = "web"
        source["provenance"] = {"generated": False, "transformed": False}
    manifest["replacement_policy"] = {
        "version": "test",
        "defaults": {
            "phase_resolution_order": ["phase_b", "phase_a", "phase_c"],
            "preserve_category": True,
            "preserve_freshness_bucket": True,
            "no_generated_increase": True,
            "no_transformed_increase": True,
        },
        "groups": [
            {"id": "web", "members": ["alpha", "beta"], "donor_order": ["alpha", "beta"]}
        ],
    }
    return manifest


class PartitionTests(unittest.TestCase):
    """A partition that quietly drops a task index verifies perfectly."""

    def test_stripe_covers_every_index_exactly_once(self) -> None:
        for count in (1, 7, 64, 3274):
            for tasks in (1, 3, 8, 97):
                seen: list[int] = []
                for task in range(tasks):
                    seen.extend(_stripe(count, tasks, task))
                self.assertEqual(sorted(seen), list(range(count)), (count, tasks))

    def test_block_covers_every_index_exactly_once_and_stays_contiguous(self) -> None:
        for count in (1, 7, 64, 3274):
            for groups in (1, 3, 8, 512):
                seen: list[int] = []
                for group in range(groups):
                    block = _block(count, groups, group)
                    if block:
                        self.assertEqual(block, list(range(block[0], block[-1] + 1)))
                    seen.extend(block)
                self.assertEqual(sorted(seen), list(range(count)), (count, groups))

    def test_partitions_reject_an_out_of_range_task(self) -> None:
        with self.assertRaises(ValueError):
            _stripe(10, 4, 4)
        with self.assertRaises(ValueError):
            _block(10, 4, -1)


class ParallelSelectionEquivalenceTests(unittest.TestCase):
    def _run_both(
        self,
        temporary: Path,
        *,
        manifest: dict | None = None,
        beta_every: int = 3,
    ) -> tuple[dict, dict]:
        shards = _write_shards(temporary / "token-counts", beta_every=beta_every)
        manifest = manifest or _manifest()
        eligible = {"alpha": 0, "beta": 0}
        for path in shards:
            for row in _iter_rows(path):
                eligible[row["source_id"]] += int(row["token_count"])

        def records():
            for path in shards:
                yield from _iter_rows(path)

        serial = build_selection(
            records(),
            manifest=manifest,
            eligible_tokens=eligible,
            output_root=temporary / "serial",
            shard_tokens=25,
            token_count_contract_sha256="c" * 64,
            tokenizer_contract={"vocabulary_size": 32},
        )
        parallel = build_selection_parallel(
            shards,
            manifest=manifest,
            eligible_tokens=eligible,
            output_root=temporary / "parallel",
            shard_tokens=25,
            token_count_contract_sha256="c" * 64,
            tokenizer_contract={"vocabulary_size": 32},
            workers=2,
            group_count=3,
        )
        return serial, parallel

    def test_both_paths_emit_identical_schedule_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            serial, parallel = self._run_both(temporary)

            self.assertEqual(len(serial["shards"]), len(parallel["shards"]))
            self.assertGreater(len(serial["shards"]), 1)
            compared = 0
            for left, right in zip(serial["shards"], parallel["shards"]):
                self.assertEqual(left["global_index"], right["global_index"])
                self.assertEqual(left["target_tokens"], right["target_tokens"])
                left_rows = _shard_lines(Path(left["path"]))
                right_rows = _shard_lines(Path(right["path"]))
                self.assertEqual(
                    left_rows,
                    right_rows,
                    f"shard {left['global_index']} differs between the two paths",
                )
                self.assertEqual(
                    sum(int(row["token_count"]) for row in right_rows),
                    int(right["target_tokens"]),
                )
                compared += len(right_rows)
            self.assertGreater(compared, 0)

    def test_both_paths_agree_on_every_quota(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            serial, parallel = self._run_both(Path(raw))
            for key in (
                "unique_quotas",
                "replay_quotas",
                "unique_written",
                "actual_source_unique_written",
                "replay_written",
                "unique_tokens",
                "replay_tokens",
                "replacement_allocation",
                "phase_tokens",
                "selection_seed",
            ):
                self.assertEqual(serial[key], parallel[key], key)
            self.assertEqual(
                parallel["unique_tokens"] + parallel["replay_tokens"],
                sum(int(shard["target_tokens"]) for shard in parallel["shards"]),
            )

    def test_parallel_path_seals_measured_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            _, parallel = self._run_both(Path(raw))
            self.assertIn("schedule_manifest_sha256", parallel)
            for shard in parallel["shards"]:
                path = Path(shard["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, int(shard["size"]))
                self.assertEqual(len(str(shard["sha256"])), 64)

    def test_both_paths_agree_when_a_donor_fills_a_short_source(self) -> None:
        """Replacement rows carry a quota source that is not their own.

        The materialiser rebuilds ``quota_source_id`` and
        ``replacement_for_source_id`` from an integer in the plan rather than
        from the record, so a corpus where nothing is replaced cannot tell
        whether it rebuilds them correctly.
        """

        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            serial, parallel = self._run_both(
                temporary, manifest=_donor_manifest(), beta_every=11
            )
            self.assertGreater(int(parallel["replacement_tokens"]), 0)
            self.assertEqual(
                serial["replacement_tokens"], parallel["replacement_tokens"]
            )

            replaced = 0
            for left, right in zip(serial["shards"], parallel["shards"]):
                left_rows = _shard_lines(Path(left["path"]))
                right_rows = _shard_lines(Path(right["path"]))
                self.assertEqual(left_rows, right_rows, left["global_index"])
                for row in right_rows:
                    if row["replacement"]:
                        replaced += 1
                        self.assertNotEqual(row["quota_source_id"], row["source_id"])
                        self.assertEqual(
                            row["replacement_for_source_id"], row["quota_source_id"]
                        )
                    else:
                        self.assertIsNone(row["replacement_for_source_id"])
            self.assertGreater(replaced, 0, "the donor fixture replaced nothing")


class ScheduleDigestTests(unittest.TestCase):
    """The schedule check must stay fail-closed now that it runs in a pool."""

    def _rows(self, root: Path, count: int = 5) -> list[tuple[str, int, str]]:
        rows = []
        for index in range(count):
            path = root / f"shard-{index:05d}.jsonl.zst"
            data = bytes((index + 7) % 251 for _ in range(4096)) * (index + 1)
            path.write_bytes(data)
            rows.append((str(path), len(data), hashlib.sha256(data).hexdigest()))
        return rows

    def test_matching_shards_pass_and_tampered_shards_do_not(self) -> None:
        from metis_data.stage_runner import _verify_schedule_digests

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = self._rows(root)
            self.assertTrue(all(ok for _, ok in _verify_schedule_digests(rows)))

            wrong_digest = list(rows)
            wrong_digest[2] = (rows[2][0], rows[2][1], "0" * 64)
            self.assertEqual(
                [ok for _, ok in _verify_schedule_digests(wrong_digest)],
                [True, True, False, True, True],
            )

            wrong_size = list(rows)
            wrong_size[1] = (rows[1][0], rows[1][1] + 1, rows[1][2])
            self.assertEqual(
                [ok for _, ok in _verify_schedule_digests(wrong_size)],
                [True, False, True, True, True],
            )

            absent = list(rows)
            absent[0] = (str(root / "gone.jsonl.zst"), 10, "a" * 64)
            self.assertEqual(
                [ok for _, ok in _verify_schedule_digests(absent)],
                [False, True, True, True, True],
            )

    def test_a_task_sharing_its_node_hashes_serially(self) -> None:
        from metis_data.stage_runner import _verify_schedule_digests

        with tempfile.TemporaryDirectory() as raw:
            rows = self._rows(Path(raw))
            tampered = list(rows)
            tampered[3] = (rows[3][0], rows[3][1], "f" * 64)
            previous = os.environ.get("METIS_TASKS_PER_JOB")
            os.environ["METIS_TASKS_PER_JOB"] = "34"
            try:
                self.assertEqual(
                    [ok for _, ok in _verify_schedule_digests(tampered)],
                    [True, True, True, False, True],
                )
            finally:
                if previous is None:
                    os.environ.pop("METIS_TASKS_PER_JOB", None)
                else:
                    os.environ["METIS_TASKS_PER_JOB"] = previous

    def test_an_empty_schedule_is_not_a_pool(self) -> None:
        from metis_data.stage_runner import _verify_schedule_digests

        self.assertEqual(_verify_schedule_digests([]), [])


if __name__ == "__main__":
    unittest.main()
