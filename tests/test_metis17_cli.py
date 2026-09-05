from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from metis_data17.acquisition import CapacityPending
from metis_data17.cli import _limits, append_event, init_run, read_events, status
from metis_data17.common import read_receipt, write_receipt


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

    def test_digit_splitting_cannot_be_disabled_by_configuration(self):
        value = yaml.safe_load(self.config.read_text())
        value["tokenizer"]["split_digits"] = False
        config = Path(self.tmp.name) / "bad.yaml"
        config.write_text(yaml.safe_dump(value))
        with self.assertRaises(ValueError):
            init_run(self.root, config)

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


if __name__ == "__main__":
    unittest.main()
