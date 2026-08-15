from __future__ import annotations

import unittest
from unittest import mock

from metis_data import stage_runner


class DecontaminationTuningScopeTests(unittest.TestCase):
    """Retuning detection must not invalidate stages that never read the tuning.

    The contract deliberately binds detection tuning so a retuned threshold
    cannot be silently ignored by a stage holding a completion marker. Bound to
    every stage, that guarantee became a trap: raising one threshold invalidated
    normalize, exact, span, minhash and code as well, so the documented ability
    to "retune without a new release" actually meant rebuilding the corpus from
    raw. Tuning cannot change what a dedup stage emits, so it must not be able
    to invalidate one.
    """

    def _contracts(self, profile, stages):
        artifact = mock.MagicMock()
        artifact.is_file.return_value = True
        artifact.read_text.return_value = "{}"
        state = mock.MagicMock()
        state.path.return_value = artifact
        with mock.patch.object(stage_runner, "_manifest", return_value={"release": "r"}), \
             mock.patch.object(stage_runner, "_manifest_contract_sha256", return_value="m"), \
             mock.patch.object(stage_runner, "_content_identity", side_effect=lambda p, k: p), \
             mock.patch.object(stage_runner, "stage_code_sha256", side_effect=lambda s: f"code-{s}"):
            return {
                stage: stage_runner._stage_execution_contract(profile, state, stage)
                for stage in stages
            }

    def setUp(self) -> None:
        self.profile = {
            "gates": {},
            "scheduler": {},
            "decontamination": {"minimum_short_matching_ngrams": 4},
        }
        self.unaffected = [
            "normalize",
            "exact_filter",
            "span_filter",
            "minhash_filter",
            "code_filter",
            "final_hash_filter",
            "tokenizer_sample",
            "pack",
        ]
        self.affected = ["decontam_index", "decontam_filter", "cleanup_decontam"]

    def test_retuning_leaves_unrelated_stages_valid(self) -> None:
        stages = self.unaffected + self.affected
        before = self._contracts(self.profile, stages)
        retuned = {
            **self.profile,
            "decontamination": {"minimum_short_matching_ngrams": 0},
        }
        after = self._contracts(retuned, stages)
        for stage in self.unaffected:
            with self.subTest(stage=stage):
                self.assertEqual(before[stage], after[stage])

    def test_retuning_does_invalidate_the_decontamination_stages(self) -> None:
        stages = self.affected
        before = self._contracts(self.profile, stages)
        retuned = {
            **self.profile,
            "decontamination": {"minimum_short_matching_ngrams": 0},
        }
        after = self._contracts(retuned, stages)
        for stage in self.affected:
            with self.subTest(stage=stage):
                self.assertNotEqual(before[stage], after[stage])


if __name__ == "__main__":
    unittest.main()
