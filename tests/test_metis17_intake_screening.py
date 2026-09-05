from __future__ import annotations

import unittest

from metis_data17.admission import admit_source
from metis_data17.common import digest_json, read_receipt
from metis_data17.intake_screening import screen_intake_chunk
from tests import test_metis17_prep as fixtures


class IntakeScreeningTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Metis17PreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = {**self.fixture.config, "opt_out_snapshot": self.fixture._opt_out()}

    def test_intake_runs_compliance_without_granting_training_eligibility(self):
        spec, raw, output = self.fixture._object([
            {"text": self.fixture.SAFE, "url": "https://allowed.example/first"},
            {"text": self.fixture.SAFE + " Another useful paragraph.", "url": "https://allowed.example/second"},
            {"text": self.fixture.HOLDOUT, "url": "https://allowed.example/benchmark"},
            {"text": self.fixture.SAFE, "url": "https://blocked.example/private"},
        ], policy={"common_crawl_derived": True, "metadata": {"quality_selection_pending": True}})
        normalized = fixtures.reblock_object(spec, raw, output / "base", self.config)
        screened = {}
        for chunk in normalized["chunks"]:
            receipt = screen_intake_chunk(self.root / chunk["ready_receipt"], output, self.config)
            self.assertEqual(receipt["status"], "SCREENED_FOR_INTAKE")
            self.assertIs(receipt["eligible"], False)
            self.assertIs(receipt["training_ready"], False)
            self.assertEqual(receipt["chunks"], [])
            self.assertEqual(receipt["eligible_documents"], 0)
            self.assertEqual(screen_intake_chunk(self.root / chunk["ready_receipt"], output, self.config), receipt)
            screened[receipt["chunk_id"]] = receipt
        self.assertEqual(sum(row["accepted_documents"] for row in screened.values()), 2)
        admission = admit_source(
            self.root, spec, normalized, screened, generation="intake-fixture", minimum_acceptance=0.1,
        )
        self.assertEqual(admission["status"], "admitted")
        self.assertEqual(admission["admission_basis"], "compliance_screening_quality_deferred")
        self.assertEqual(admission["eligible_documents"], 0)
        self.assertEqual(admission["screened_documents"], 2)
        self.assertIn("intake_receipts", admission)
        self.assertNotIn("eligible_receipts", admission)
        self.assertEqual(
            read_receipt(self.root / "admissions" / f"{digest_json(spec.source_id)}.json")["screened_documents"], 2,
        )

    def test_missing_benchmark_policy_cannot_be_waived_for_intake(self):
        spec, raw, output = self.fixture._object(
            [{"text": self.fixture.SAFE}], policy={"metadata": {"quality_selection_pending": True}},
        )
        normalized = fixtures.reblock_object(spec, raw, output / "base", self.config)
        config = {**self.config, "decontamination_index": None}
        with self.assertRaisesRegex(ValueError, "exactly the deferred"):
            screen_intake_chunk(self.root / normalized["chunks"][0]["ready_receipt"], output, config)

    def test_ordinary_eligible_source_cannot_use_the_intake_only_contract(self):
        spec, raw, output = self.fixture._object([{"text": self.fixture.SAFE}])
        normalized = fixtures.reblock_object(spec, raw, output / "base", self.config)
        with self.assertRaisesRegex(ValueError, "exactly the deferred"):
            screen_intake_chunk(self.root / normalized["chunks"][0]["ready_receipt"], output, self.config)


if __name__ == "__main__":
    unittest.main()
