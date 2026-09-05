from __future__ import annotations

import unittest
from unittest.mock import patch

from metis_data17.acquisition import CapacityPending, receipt_path
from metis_data17.admission import admit_source
from metis_data17.canary import prepare_canaries
from metis_data17.common import digest_json, read_receipt, write_receipt
from tests import test_metis17_prep as fixtures


class Metis17CanaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Metis17PreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        settings = {
            "prep": {
                "quality_profiles_path": str(self.fixture.quality),
                "output_chunk_bytes": 4096, "batch_size": 3,
                "source_minimum_acceptance": 0.1,
            },
        }
        write_receipt(self.root / "RUN.json", {
            "config": settings, "config_sha256": digest_json(settings),
        })
        write_receipt(self.root / "limits.json", {
            "capacity_confirmation": "pending", "max_raw_bytes": 32_000_000,
            "max_working_bytes": 512_000_000, "policy_and_metadata_reserve_bytes": 1_000_000,
            "filesystem_free_floor_bytes": 0,
        })
        self.policy = {
            "policy_ready": True,
            "benchmark_registry": str(self.fixture.registry_path),
            "decontamination_index": str(self.fixture.index_path),
            "opt_out_snapshot": str(self.fixture._opt_out()),
        }
        self.spec, self.raw, self.output = self.fixture._object([
            {"text": self.fixture.SAFE + f" Source example {i}."} for i in range(7)
        ])
        write_receipt(receipt_path(self.root, self.spec.object_id), {
            **self.raw.to_dict(), "spec": self.spec.to_dict(),
        })

    def test_canary_performs_real_eligibility_before_opening_its_source(self):
        with patch("metis_data17.canary.policy_config", return_value=self.policy):
            results = prepare_canaries(self.root, [self.spec.object_id])
        self.assertEqual(results[0]["status"], "admitted")
        self.assertEqual(results[0]["eligible_documents"], 7)
        admission = read_receipt(self.root / "admissions" / f"{digest_json(self.spec.source_id)}.json")
        self.assertTrue(admission["eligible_receipts"])
        for item in admission["eligible_receipts"]:
            value = read_receipt(self.root / item["path"])
            self.assertEqual(digest_json(value), item["receipt_sha256"])
            self.assertIs(value["object_complete"], True)

    def test_canary_byte_ceiling_is_enforced_before_policy_preload(self):
        with patch("metis_data17.canary.policy_config", return_value=self.policy), patch(
            "metis_data17.canary.prepare_runtime",
        ) as preload:
            with self.assertRaises(CapacityPending):
                prepare_canaries(self.root, [self.spec.object_id], maximum_raw_bytes=1)
        preload.assert_not_called()
        self.assertFalse((self.root / "admissions").exists())

    def test_duplicate_objects_are_not_recounted_as_independent_canaries(self):
        with self.assertRaises(ValueError):
            prepare_canaries(self.root, [self.spec.object_id, self.spec.object_id])

    def test_a_supervised_worker_cannot_preload_missing_policies_as_ready(self):
        missing = {**self.fixture.config, "opt_out_snapshot": self.fixture._opt_out(),
                   "decontamination_index": self.root / "missing-index.json"}
        with self.assertRaisesRegex(RuntimeError, "preparation_policies_pending"):
            fixtures.prepare_runtime(missing, require_ready=True)

    def test_canary_rechecks_the_actual_eligibility_receipt_flags(self):
        normalized = fixtures.reblock_object(self.spec, self.raw, self.output, self.fixture.config)
        screened = {}
        for chunk in normalized["chunks"]:
            value = fixtures.prepare_chunk(self.root / chunk["ready_receipt"], self.output, self.fixture.config)
            screened[value["chunk_id"]] = value
        key = next(iter(screened))
        original = screened[key]
        forged = self.root / "false-eligibility.json"
        write_receipt(forged, {**original, "object_complete": False})
        screened[key] = {**original, "receipt_path": str(forged.relative_to(self.root))}
        with self.assertRaisesRegex(RuntimeError, "Invalid eligible"):
            admit_source(
                self.root, self.spec, normalized, screened,
                generation="fixture", minimum_acceptance=0.1,
            )
        self.assertFalse((self.root / "admissions").exists())


if __name__ == "__main__":
    unittest.main()
