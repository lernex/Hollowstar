from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from metis_data import stage_runner
from metis_data.stage_runner import _completion_inventory, _filter_chain_drift_allowed


class FilterChainDriftGateTests(unittest.TestCase):
    """The override must be explicit, justified, and recorded.

    A contract mismatch normally means filtering code changed and the stage did
    not re-run. Accepting one is a judgement an operator makes, not a default,
    and a release that was accepted over a mismatch has to say so -- otherwise
    the override is indistinguishable from the hazard it is meant to survive.
    """

    def test_refused_by_default(self) -> None:
        self.assertFalse(_filter_chain_drift_allowed({}))
        self.assertFalse(_filter_chain_drift_allowed({"gates": {}}))

    def test_refused_when_enabled_without_a_reason(self) -> None:
        for reason in (None, "", "   "):
            gates = {"allow_filter_chain_contract_drift": True}
            if reason is not None:
                gates["filter_chain_contract_drift_reason"] = reason
            with self.subTest(reason=reason), self.assertRaises(RuntimeError):
                _filter_chain_drift_allowed({"gates": gates})

    def test_allowed_only_with_both(self) -> None:
        self.assertTrue(
            _filter_chain_drift_allowed(
                {
                    "gates": {
                        "allow_filter_chain_contract_drift": True,
                        "filter_chain_contract_drift_reason": "checked the diff",
                    }
                }
            )
        )

    def test_reason_alone_does_not_enable_it(self) -> None:
        self.assertFalse(
            _filter_chain_drift_allowed(
                {"gates": {"filter_chain_contract_drift_reason": "checked the diff"}}
            )
        )


class CompletionInventoryDriftTests(unittest.TestCase):
    def _state(self, root: Path, contracts: list[str]):
        folder = root / "completed" / "demo"
        folder.mkdir(parents=True)
        for index, contract in enumerate(contracts):
            (folder / f"task-{index:06d}.json").write_text(
                json.dumps({"execution_contract_sha256": contract}), encoding="utf-8"
            )
        state = mock.MagicMock()
        state.path.side_effect = lambda *parts: root.joinpath(*parts)
        return state

    def test_mismatch_still_raises_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["old", "old"])
            with self.assertRaises(RuntimeError):
                _completion_inventory(
                    state, "demo", 2, expected_execution_contract_sha256="new"
                )

    def test_drift_is_recorded_when_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["old", "old"])
            receipt = _completion_inventory(
                state,
                "demo",
                2,
                expected_execution_contract_sha256="new",
                allow_contract_drift=True,
            )
            self.assertTrue(receipt["contract_drift"]["accepted"])
            self.assertEqual(
                receipt["contract_drift"]["observed_execution_contract_sha256"], ["old"]
            )

    def test_no_drift_key_when_contracts_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["same", "same"])
            receipt = _completion_inventory(
                state,
                "demo",
                2,
                expected_execution_contract_sha256="same",
                allow_contract_drift=True,
            )
            self.assertNotIn("contract_drift", receipt)

    def test_missing_markers_still_fail_even_with_drift_allowed(self) -> None:
        """The override forgives a changed contract, never absent work."""

        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(Path(tmp), ["old"])
            with self.assertRaises(RuntimeError) as caught:
                _completion_inventory(
                    state,
                    "demo",
                    4,
                    expected_execution_contract_sha256="new",
                    allow_contract_drift=True,
                )
            self.assertIn("incomplete", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
