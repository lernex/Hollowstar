from __future__ import annotations

import unittest

from metis_data.quality import evaluate_quality, load_quality_profiles, text_features


class PersonalDataPrecisionTests(unittest.TestCase):
    """The PII gate must survive documents made of numbers.

    The original phone pattern accepted a bare space between all three digit
    groups, so any three spaced numbers matched. Scientific PDFs and legal texts
    are the corpora densest in tabular numbers and lost the majority of their
    records to `personal_data` because of it.
    """

    REAL = [
        "Call us at (555) 123-4567 today.",
        "Reach me at 555.123.4567 any time.",
        "Support: 555-123-4567 during business hours.",
        "Telephone 555 123 4567 for enquiries.",
        "Dial +1 555-123-4567 from abroad.",
        "Write to jane.doe@example.edu for the dataset.",
        "The form lists SSN 123-45-6789 in the header.",
    ]
    NOT_PERSONAL = [
        "Sample  100 200 3000  yields 4.2 across the run.",
        "Values were 120 240 4800 respectively in each trial.",
        "x y z\n001 002 0003\n004 005 0006",
        "Peaks at 105 220 3340 cm-1 were observed in the spectrum.",
        "In 1999 400 5000 units shipped to the regional depots.",
        "See 42 USC 1983 405 6000 for the governing provision.",
        "The experiment ran for three weeks in total.",
    ]

    def test_contact_details_are_still_detected(self) -> None:
        for text in self.REAL:
            with self.subTest(text=text):
                self.assertTrue(text_features(text)["contains_personal_data"])

    def test_tabular_numbers_are_not_personal_data(self) -> None:
        for text in self.NOT_PERSONAL:
            with self.subTest(text=text):
                self.assertFalse(text_features(text)["contains_personal_data"])

    def test_a_numeric_results_table_survives_the_default_profile(self) -> None:
        # web_general_v1 inherits reject_personal_data from the defaults, which
        # is what made this a corpus-wide loss rather than one profile's problem.
        table = "Results\n\n" + "\n".join(
            f"Trial {n}: 100 200 3000 and 105 220 3340 cm-1 measured under load, "
            f"with consistent behaviour across the replicate series."
            for n in range(12)
        )
        decision = evaluate_quality(
            table,
            profile_name="web_general_v1",
            metadata={"quality_score": 0.9, "language_probability": 0.99},
            profiles=load_quality_profiles(),
        )
        self.assertNotEqual(decision.reason, "personal_data")


class NonProseLanguageGateTests(unittest.TestCase):
    """Formal mathematics is not English and must not be scored as if it were."""

    LEAN = (
        "theorem add_comm (a b : Nat) : a + b = b + a := by\n"
        "  induction a with\n"
        "  | zero => simp\n"
        "  | succ n ih => rw [Nat.succ_add, ih]\n"
    ) * 6

    def _evidence(self, source_id: str, category: str, profile: str) -> dict:
        from metis_data.normalization_evidence import derive_normalization_evidence

        source = {
            "id": source_id,
            "category": category,
            "license": {"status": "reviewed", "expression": "Apache-2.0"},
            "provenance": {},
            "processing": {"quality_profile": profile},
        }
        return derive_normalization_evidence({"text": self.LEAN}, source, {}, self.LEAN)

    def test_formal_proofs_are_exempt_from_the_english_gate(self) -> None:
        evidence = self._evidence("formal_theorem_corpora", "math", "formal_proof_v1")
        self.assertEqual(evidence.get("language_probability"), 1.0)

    def test_the_exemption_follows_the_profile_not_only_the_category(self) -> None:
        # The category == "code" branch already existed; formal mathematics is
        # filed under `math`, which is exactly how this went unnoticed.
        without = self._evidence("prose_source", "math", "math_score3_v1")
        self.assertNotEqual(without.get("language_probability"), 1.0)

    def test_english_prose_still_gets_a_measured_probability(self) -> None:
        from metis_data.normalization_evidence import derive_normalization_evidence

        prose = " ".join(
            "The theorem states that addition is commutative and the proof "
            "proceeds by induction on the first argument." for _ in range(8)
        )
        source = {
            "id": "web", "category": "web",
            "license": {"status": "reviewed", "expression": "CC-BY-4.0"},
            "provenance": {}, "processing": {"quality_profile": "web_general_v1"},
        }
        evidence = derive_normalization_evidence({"text": prose}, source, {}, prose)
        probability = evidence.get("language_probability")
        self.assertIsNotNone(probability)
        self.assertGreater(float(probability), 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
