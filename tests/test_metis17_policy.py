from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from metis_data17.common import sha256_file, write_receipt
from metis_data17.policy import policy_config


class Metis17PolicySettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = {
            "minimum_matching_ngrams": 2,
            "minimum_short_matching_ngrams": 0,
            "minimum_code_matching_ngrams": 16,
            "minimum_code_skeleton_matching_ngrams": 0,
            "match_fraction": 0.002,
            "contiguous_run_minimum": 8,
        }
        self.registry = self.root / "registry.yaml"
        self.registry.write_text("fixtures: []\n")
        self.index = self.root / "index.json"
        self.index.write_text(json.dumps({**self.settings, "manifest_sha256": "a" * 64}))
        write_receipt(self.root / "policy" / "CURRENT.json", {
            "decontamination_index": str(self.index),
            "index_manifest_sha256": "a" * 64,
            "benchmark_registry": str(self.registry),
            "holdout_registry_sha256": sha256_file(self.registry),
            "opt_out_snapshot": "opt-out.json",
            "opt_out_sha256": "b" * 64,
        })
        self.write_run(self.settings)

    def write_run(self, settings):
        write_receipt(self.root / "RUN.json", {"config": {"decontamination": settings}})

    def load(self):
        with patch("metis_data17.policy._strict_opt_out_state", return_value={}):
            return policy_config(self.root)

    def test_every_declared_threshold_is_the_one_the_worker_will_load(self):
        self.assertEqual(self.load()["decontamination_effective"], self.settings)
        for key in self.settings:
            with self.subTest(key=key):
                self.write_run({**self.settings, key: self.settings[key] + 1})
                with self.assertRaisesRegex(RuntimeError, key):
                    self.load()
        self.write_run(self.settings)
        self.assertIs(self.load()["policy_ready"], True)

    def test_unknown_or_boolean_thresholds_cannot_be_ignored(self):
        for extra in ({"minimum_matching_ngrams": True}, {"undeclared_knob": 1}):
            with self.subTest(extra=extra):
                self.write_run({**self.settings, **extra})
                with self.assertRaisesRegex(RuntimeError, "disagree"):
                    self.load()

    def test_index_identity_is_pinned_even_when_thresholds_match(self):
        self.index.write_text(json.dumps({**self.settings, "manifest_sha256": "c" * 64}))
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.load()


if __name__ == "__main__":
    unittest.main()
