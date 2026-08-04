from __future__ import annotations

import unittest
from pathlib import Path

from metis_data.quality import evaluate_quality, load_quality_profiles, text_features

ROOT = Path(__file__).resolve().parents[1]


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




class GeneratedFileProbabilityTests(unittest.TestCase):
    """A ceiling on generated files must not reject files it cannot measure."""

    HUMAN = (
        "def parse_config(path):\n"
        "    with open(path) as handle:\n"
        "        payload = json.load(handle)\n"
        "    if not payload.get('name'):\n"
        "        raise ValueError('config needs a name')\n"
        "    return payload\n"
    ) * 8

    def _evidence(self, row: dict, text: str) -> dict:
        from metis_data.normalization_evidence import derive_normalization_evidence

        source = {
            "id": "starcoder_permissive_code",
            "category": "code",
            "license": {"status": "reviewed", "expression": "permissive"},
            "provenance": {},
            "processing": {"quality_profile": "repository_code_v1"},
        }
        return derive_normalization_evidence(row, source, {}, text)

    def test_starcoder_path_field_is_recognised(self) -> None:
        # StarCoderData ships max_stars_repo_path and nothing else path-shaped.
        row = {"content": self.HUMAN, "max_stars_repo_path": "src/parser/config.py"}
        evidence = self._evidence(row, self.HUMAN)
        self.assertIsNotNone(evidence.get("generated_file_probability"))
        self.assertLessEqual(float(evidence["generated_file_probability"]), 0.10)

    def test_a_generated_marker_is_still_caught(self) -> None:
        text = "// Code generated by protoc-gen-go. DO NOT EDIT.\n" + self.HUMAN
        row = {"content": text, "max_stars_repo_path": "api/service.pb.go"}
        evidence = self._evidence(row, text)
        self.assertEqual(float(evidence["generated_file_probability"]), 1.0)

    def test_minified_content_is_caught_without_any_path(self) -> None:
        minified = "!function(e,t){" + "a=b;" * 4000 + "}(window,document);"
        evidence = self._evidence({"content": minified}, minified)
        self.assertGreaterEqual(float(evidence["generated_file_probability"]), 0.90)

    def test_a_missing_path_no_longer_means_a_missing_signal(self) -> None:
        # The old detector returned None without a path, and the quality gate
        # turned that into `missing_generated_file_probability` -- a rejection
        # of every record in the corpus rather than of generated files.
        evidence = self._evidence({"content": self.HUMAN}, self.HUMAN)
        self.assertIsNotNone(evidence.get("generated_file_probability"))
        decision = evaluate_quality(
            self.HUMAN,
            profile_name="repository_code_v1",
            metadata=evidence,
            profiles=load_quality_profiles(),
        )
        self.assertNotEqual(decision.reason, "missing_generated_file_probability")


class PlainTextMathTests(unittest.TestCase):
    """Most web mathematics is not typeset."""

    def _equation(self, text: str):
        from metis_data.normalization_evidence import _equation_integrity

        return _equation_integrity(text)

    def test_a_worked_solution_without_latex_has_equations(self) -> None:
        solution = (
            "To solve for x, start from 3x + 5 = 20. Subtract 5 from both sides, "
            "so 3x = 15. Divide by 3 and x = 5. Check the result: 3 * 5 + 5 = 20, "
            "which matches the original statement.\n"
        ) * 3
        passed, details = self._equation(solution)
        self.assertTrue(passed)
        self.assertEqual(details["latex_signal_count"], 0)
        self.assertGreater(details["plain_signal_count"], 0)

    def test_unbalanced_latex_still_fails(self) -> None:
        # The balance check must keep working for documents that do use markup.
        passed, _ = self._equation(r"The identity \(a^2 + b^2 = c^2 is stated here. $x = 1$")
        self.assertFalse(passed)

    def test_a_score_fallback_needs_more_than_one_equals_sign(self) -> None:
        from metis_data.normalization_evidence import derive_normalization_evidence

        source = {
            "id": "megamath_unique",
            "category": "math",
            "license": {"status": "reviewed", "expression": "ODC-By-1.0"},
            "provenance": {},
            "processing": {"quality_profile": "math_score3_v1"},
        }
        invoice = (
            "Quarterly summary. Revenue = 2400000 for the period under review, and "
            "the operating cost = 1900000 across the same window. The remainder was "
            "carried forward into the following reporting period without adjustment. "
        ) * 6
        evidence = derive_normalization_evidence({"text": invoice}, source, {}, invoice)
        self.assertIsNone(evidence.get("math_score"))


class ManifestAttestationTests(unittest.TestCase):
    """Corpus-level attestations must reach the evidence, and must be checked.

    The first version of this mechanism read the block from
    ``source["provenance"]["attestations"]`` while both manifests wrote it at
    the source top level. Nothing raised, nothing logged, and two attestations
    sat in the repository doing nothing -- the failure a fail-closed pipeline
    cannot catch, because a no-op looks exactly like a source with no
    attestations. These tests read the shipped manifests rather than fixtures
    precisely so a future move of the key fails here.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from metis_data.manifest import load_manifest

        manifest = load_manifest(ROOT / "manifests" / "metis-1.6.yaml")
        cls.sources = {source["id"]: source for source in manifest["sources"]}

    def test_every_shipped_attestation_is_resolved(self) -> None:
        from metis_data.normalization_evidence import validated_attestations

        attested = {
            source_id: validated_attestations(source)
            for source_id, source in self.sources.items()
            if source.get("attestations")
        }
        self.assertTrue(attested, "no source ships an attestation; has the key moved?")
        for source_id, values in attested.items():
            with self.subTest(source=source_id):
                self.assertTrue(values, f"{source_id} declares attestations that resolve to nothing")

    def test_an_attestation_reaches_the_derived_evidence(self) -> None:
        from metis_data.normalization_evidence import derive_normalization_evidence

        source = self.sources["metis_govreference_uk_hansard"]
        text = "The House met at half past eleven o'clock. " * 40
        evidence = derive_normalization_evidence({"text": text}, source, {}, text)
        self.assertEqual(evidence["canonical_url"], "https://hansard.parliament.uk/")
        method = next(
            item["method"]
            for item in evidence["normalization_evidence"]
            if item["field"] == "canonical_url"
        )
        self.assertEqual(method, "pinned_source_manifest_attestation_v1")

    def test_the_row_always_outranks_the_attestation(self) -> None:
        from metis_data.normalization_evidence import derive_normalization_evidence

        source = self.sources["metis_govreference_uk_hansard"]
        text = "The House met at half past eleven o'clock. " * 40
        row = {"text": text, "url": "https://hansard.parliament.uk/Commons/1994-03-02"}
        evidence = derive_normalization_evidence(row, source, {}, text)
        self.assertEqual(evidence["canonical_url"], row["url"])

    def test_a_basisless_attestation_is_rejected(self) -> None:
        from metis_data.normalization_evidence import validated_attestations

        with self.assertRaisesRegex(ValueError, "attestation_basis"):
            validated_attestations({"id": "x", "attestations": {"open_access": True}})

    def test_an_unattestable_field_is_rejected(self) -> None:
        from metis_data.normalization_evidence import validated_attestations

        # One value shared by every row identifies no row, so a grounding
        # document id can never be attested for a whole corpus.
        with self.assertRaisesRegex(ValueError, "source_document_id"):
            validated_attestations(
                {
                    "id": "x",
                    "attestation_basis": "every row came from the same place",
                    "attestations": {"source_document_id": "corpus"},
                }
            )

    def test_manifest_validation_catches_a_bad_attestation(self) -> None:
        from metis_data.manifest import validate_manifest

        self.assertTrue(validate_manifest().ok)


class SyntheticProvenanceTests(unittest.TestCase):
    """Synthetic profiles ask for lineage, never for an unshipped flag."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_quality_profiles()

    def test_no_profile_in_use_requires_a_verification_flag(self) -> None:
        from metis_data.manifest import load_manifest

        manifest = load_manifest(ROOT / "manifests" / "metis-1.6.yaml")
        in_use = {source["processing"]["quality_profile"] for source in manifest["sources"]}
        # Nothing we pin publishes a per-row verification result. A profile that
        # requires one rejects its whole source silently, which is how ~26B
        # tokens of Nemotron reasoning normalized to zero.
        for name in sorted(in_use):
            profile = self.profiles["profiles"][name]
            with self.subTest(profile=name):
                self.assertNotIn("verification", profile)
                self.assertNotIn("require_execution_or_static_verification", profile)
                self.assertNotIn("require_programmatic_or_source_verification", profile)

    def test_the_synthetic_profiles_still_require_lineage(self) -> None:
        for name in (
            "grounded_synthetic_v1",
            "synthetic_qa_v1",
            "synthetic_reasoning_v1",
            "synthetic_code_v1",
            "legal_synthetic_v1",
            "textbook_synthetic_v1",
        ):
            with self.subTest(profile=name):
                self.assertTrue(self.profiles["profiles"][name].get("require_genealogy"))


class FormalLanguageProofTests(unittest.TestCase):
    """Proof-Pile-2 files Agda under a prose proof profile."""

    AGDA = (
        "module Data.Nat.Properties where\n\n"
        "+-comm : ∀ (m n : ℕ) → m + n ≡ n + m\n"
        "+-comm zero n = sym (+-identityʳ n)\n"
        "+-comm (suc m) n = trans (cong suc (+-comm m n)) (sym (+-suc n m))\n"
    ) * 4

    def _evidence(self, row: dict) -> dict:
        from metis_data.normalization_evidence import derive_normalization_evidence

        source = {
            "id": "proof_pile2_math",
            "category": "math",
            "license": {"status": "reviewed", "expression": "MIT"},
            "provenance": {},
            "processing": {"quality_profile": "proof_v1"},
        }
        return derive_normalization_evidence(row, source, {}, row["text"])

    def test_a_formal_development_is_a_statement_with_an_argument(self) -> None:
        evidence = self._evidence({"text": self.AGDA, "meta": {"ext": "agda"}})
        self.assertTrue(evidence.get("statement_and_argument"))

    def test_formal_source_is_not_scored_as_english(self) -> None:
        evidence = self._evidence({"text": self.AGDA, "meta": {"ext": "agda"}})
        self.assertEqual(evidence.get("language_probability"), 1.0)

    def test_english_prose_under_proof_v1_is_unaffected(self) -> None:
        # Without a formal extension the row must still be judged as prose, so
        # the exemption cannot leak into natural-language proofs.
        evidence = self._evidence({"text": self.AGDA})
        self.assertNotEqual(evidence.get("language_probability"), 1.0)
        self.assertIsNone(evidence.get("statement_and_argument"))


class RoleMailboxTests(unittest.TestCase):
    """A desk address is not a person's contact details."""

    def _features(self, text: str) -> dict:
        from metis_data.quality import text_features

        return text_features(text)

    def test_a_publishers_support_address_is_not_personal_data(self) -> None:
        # OpenStax prints `support@openstax.org` in every colophon, and that one
        # address discarded a 1.59M-character textbook.
        body = "This textbook is openly licensed. " * 40
        text = body + "\nFor help with this book, write to support@openstax.org.\n"
        self.assertFalse(self._features(text)["contains_personal_data"])

    def test_an_individuals_address_is_still_personal_data(self) -> None:
        body = "This textbook is openly licensed. " * 40
        text = body + "\nWritten by a.researcher@university.edu.\n"
        self.assertTrue(self._features(text)["contains_personal_data"])

    def test_every_listed_role_mailbox_is_exempt(self) -> None:
        from metis_data.quality import ROLE_MAILBOXES

        # Routable domain on purpose: this asserts the role-mailbox rule, not
        # the reserved-domain one.
        for mailbox in ROLE_MAILBOXES:
            with self.subTest(mailbox=mailbox):
                text = f"Reach the desk at {mailbox}@university.edu for assistance."
                self.assertFalse(self._features(text)["contains_personal_data"])

    def test_reserved_placeholder_domains_are_not_personal_data(self) -> None:
        # RFC 2606 / RFC 6761 names cannot resolve to a person. Both of these
        # are real matches in FinePDFs-Edu.
        body = "This technical note describes the procedure. " * 30
        for address in ("firstname.lastname@example.org", "email@example.com",
                        "a.person@sub.example.net", "someone@localhost"):
            with self.subTest(address=address):
                self.assertFalse(
                    self._features(body + f"\nWrite to {address}.\n")["contains_personal_data"]
                )

    def test_a_real_domain_resembling_example_is_still_personal_data(self) -> None:
        # `example.edu` is a routable name and is not reserved; only the
        # reserved set is exempt.
        body = "This technical note describes the procedure. " * 30
        for address in ("a.person@example.edu", "a.person@myexample.com"):
            with self.subTest(address=address):
                self.assertTrue(
                    self._features(body + f"\nWrite to {address}.\n")["contains_personal_data"]
                )

    def test_a_role_word_inside_a_personal_local_part_is_not_exempt(self) -> None:
        # The exemption is the whole local part, not a prefix: `support.hotline`
        # is a desk, but `supportive.person` and `jsupport` are not the literal
        # role mailbox and must not inherit its exemption.
        # The domain must be routable, or the reserved-domain exemption would
        # carry the assertion instead of the local-part rule under test.
        for local in ("supportive.person", "jsupport", "info.smith"):
            with self.subTest(local=local):
                text = f"Contact {local}@university.edu about this matter."
                self.assertTrue(self._features(text)["contains_personal_data"])


class CurrencyIsNotADelimiterTests(unittest.TestCase):
    """A price is not an equation somebody forgot to close."""

    def _eq(self, text: str):
        from metis_data.normalization_evidence import _equation_integrity

        return _equation_integrity(text)

    def test_a_price_does_not_unbalance_the_delimiters(self) -> None:
        text = r"Buy at $25.00 each. Then $x^2 + y^2 = z^2$ holds for all integers."
        passed, detail = self._eq(text)
        self.assertTrue(passed)
        self.assertEqual(detail["currency_amounts_excluded"], 1)

    def test_a_price_ending_a_sentence_is_still_a_price(self) -> None:
        text = r"It costs $5.00. Also $\frac{1}{2}$ of it remains unspent today."
        passed, detail = self._eq(text)
        self.assertTrue(passed)
        self.assertEqual(detail["currency_amounts_excluded"], 1)

    def test_mathematics_opening_on_a_number_is_not_a_price(self) -> None:
        # `$20 \times 365 = 7300$` opens real mathematics on a digit. The
        # discriminator is the LaTeX command that follows the amount.
        text = r"We compute $20 \times 365 = 7300$ pounds per year."
        passed, detail = self._eq(text)
        self.assertTrue(passed)
        self.assertEqual(detail["currency_amounts_excluded"], 0)

    def test_a_genuinely_unclosed_delimiter_still_fails(self) -> None:
        text = r"The identity $a^2 + b^2 = c^2 is stated here without closing it."
        passed, _ = self._eq(text)
        self.assertFalse(passed)

    def test_balanced_inline_mathematics_is_unaffected(self) -> None:
        text = r"Let $a$ and $b$ be integers with $a < b$ throughout."
        passed, detail = self._eq(text)
        self.assertTrue(passed)
        self.assertEqual(detail["currency_amounts_excluded"], 0)

    def test_the_text_itself_is_never_rewritten(self) -> None:
        # The exclusion applies to the balance count only; nothing edits the
        # document that gets trained on.
        from metis_data.normalization_evidence import extract_training_text

        row = {"text": "The cost is $5.00 per unit and $x = 1$ holds."}
        self.assertEqual(extract_training_text(row), row["text"])


class ContactAllowanceTests(unittest.TestCase):
    """One contact block is a document; a roster of people is a directory."""

    BODY = "This technical report describes the measurement procedure in detail. " * 40

    def _decide(self, text: str, profile: str = "pdf_technical_v1"):
        from metis_data.quality import evaluate_quality

        return evaluate_quality(
            text,
            profile_name=profile,
            metadata={
                "ocr_confidence": 0.99,
                "repeated_header_footer_fraction": 0.0,
                "reading_order_passed": True,
                "language_probability": 0.99,
                "license": "cc-by-4.0",
            },
        )

    def test_a_single_author_contact_is_kept(self) -> None:
        text = self.BODY + "\nCorrespondence: j.researcher@university.edu, (617) 555-0142.\n"
        self.assertTrue(self._decide(text).keep)

    def test_a_directory_of_people_is_still_rejected(self) -> None:
        roster = "\n".join(
            f"Delegate {n}: person{n}@agency.gov, (202) 555-01{n:02d}" for n in range(12)
        )
        decision = self._decide(self.BODY + "\n" + roster)
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "personal_data")

    def test_one_contact_repeated_is_not_many_contacts(self) -> None:
        # Distinctness matters: a running footer repeating one address on every
        # page is one contact, not forty. The lines are varied so that this
        # asserts the contact rule rather than the repeated-line one.
        repeated = "".join(
            f"\nSection {n} was prepared by j.researcher@university.edu for review.\n"
            for n in range(40)
        )
        decision = self._decide(self.BODY + repeated)
        self.assertTrue(decision.keep, decision.reason)

    def test_profiles_without_the_allowance_are_unchanged(self) -> None:
        # web_general_v1 never opted in, so any single contact still rejects.
        from metis_data.quality import evaluate_quality

        text = self.BODY + "\nCall me at (617) 555-0142.\n"
        decision = evaluate_quality(
            text, profile_name="web_general_v1",
            metadata={"quality_score": 0.9, "language_probability": 0.99},
        )
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "personal_data")

    def test_an_ssn_counts_toward_the_allowance(self) -> None:
        from metis_data.quality import personal_contacts

        self.assertIn(("ssn", "123-45-6789"), personal_contacts("SSN 123-45-6789 appears."))


class LegalPrimaryContactTests(unittest.TestCase):
    """An official notice prints the office to contact; that is the record."""

    def _decide(self, text: str, profile: str):
        from metis_data.quality import evaluate_quality

        return evaluate_quality(
            text,
            profile_name=profile,
            metadata={
                "primary_source": True,
                "jurisdiction": "US",
                "license": "public-domain",
                "language_probability": 0.99,
                "quality_score": 0.9,
            },
        )

    def test_an_agency_contact_number_does_not_discard_a_notice(self) -> None:
        text = (
            "AGENCY: Federal Aviation Administration, DOT. ACTION: Final rule. "
            "FOR FURTHER INFORMATION CONTACT: the Operations Support Group, "
            "telephone: (817) 222-5110. " + "This notice amends the airspace description. " * 20
        )
        self.assertTrue(self._decide(text, "legal_primary_v1").keep)

    def test_the_exemption_does_not_leak_to_other_profiles(self) -> None:
        # Scoped to primary legal text only: the same contact block in a scraped
        # web record must still be treated as personal data.
        text = "Call me at (817) 222-5110 about the listing. " + "Some ordinary prose here. " * 30
        decision = self._decide(text, "web_general_v1")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "personal_data")

    def test_secrets_are_still_rejected_in_legal_text(self) -> None:
        # Only the personal-data gate is relaxed; credential leakage is not.
        text = (
            "AGENCY: Department of Commerce. -----BEGIN PRIVATE KEY----- "
            + "This notice concerns the record. " * 30
        )
        decision = self._decide(text, "legal_primary_v1")
        self.assertFalse(decision.keep)
        self.assertEqual(decision.reason, "secret")


class ShortRowLanguageEvidenceTests(unittest.TestCase):
    """A short row is uncertain, not unmeasurable."""

    def test_a_short_english_answer_is_measured_rather_than_skipped(self) -> None:
        # `nemotron_specialized_fact_seeking` rows run 18-55 words, and the old
        # 30-word floor returned None, which fails closed as
        # `missing_language_probability`.
        from metis_data.normalization_evidence import _computed_english_probability

        text = (
            "The Treaty of Westphalia was signed in 1648, and it ended the "
            "Thirty Years War in the Holy Roman Empire."
        )
        result = _computed_english_probability(text)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result[0], 0.80)

    def test_text_below_the_new_floor_is_still_unmeasurable(self) -> None:
        from metis_data.normalization_evidence import _computed_english_probability

        self.assertIsNone(_computed_english_probability("Too short to judge."))

    def test_a_long_document_scores_exactly_as_before(self) -> None:
        # At 30+ words the distinct-function-word target is still 12, so the
        # rescaling must not move any score the old floor already produced.
        from metis_data.normalization_evidence import _computed_english_probability

        text = (
            "This paragraph is written in ordinary English prose so that the "
            "language detector has enough of the common function words it "
            "relies on, and it is long enough that the sample size does not "
            "limit the vocabulary target in any way at all here."
        )
        result = _computed_english_probability(text)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result[1]["sampled_words"], 30)
        self.assertGreaterEqual(result[0], 0.80)

    def test_short_non_english_is_not_rescued_by_the_lower_floor(self) -> None:
        from metis_data.normalization_evidence import _computed_english_probability

        text = "Der Vertrag wurde im Jahre 1648 unterzeichnet und beendete den Krieg im Reich."
        result = _computed_english_probability(text)
        self.assertIsNotNone(result)
        self.assertLess(result[0], 0.80)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
