from __future__ import annotations

import unittest


def _plan(source_targets, availability, *, target, tolerance):
    """The apportionment and its bound, in the shape stage_runner applies them."""

    quotas: dict[str, dict[str, int]] = {}
    short: dict[str, int] = {}
    for source_id in sorted(source_targets):
        wanted = int(source_targets[source_id])
        holders = availability.get(source_id, {})
        total_available = sum(holders.values())
        if total_available < wanted:
            short[source_id] = wanted - total_available
            wanted = total_available
        remaining = wanted
        for task in sorted(
            (t for t in holders if holders[t] > 0),
            key=lambda t: (-holders[t], int(t)),
        ):
            if remaining <= 0:
                break
            take = min(holders[task], remaining)
            quotas.setdefault(str(task), {})[source_id] = int(take)
            remaining -= take
        if remaining:
            raise RuntimeError(f"could not place {remaining} for {source_id}")
    shortfall = sum(short.values())
    if short and shortfall > tolerance * target:
        raise RuntimeError(f"shortfall {shortfall} exceeds tolerance: {short}")
    return quotas, short, shortfall


class TokenizerSampleShortfallTests(unittest.TestCase):
    """A thin source should cost coverage, not the whole sample -- up to a bound.

    Failing outright meant a source 1.24GB short of a 160GB sample, 0.77%, could
    not be sampled at all. Ignoring shortfalls entirely would let the sample
    quietly stop being stratified. The bound is the whole point, so it is what
    the tests pin.
    """

    TARGET = 160_000_000_000
    # The real 1.6 numbers.
    TARGETS = {
        "nemotron_math_proofs": 1_483_000_000,
        "common_pile_libretexts_reference": 97_000_000,
        "fineweb_edu": 20_000_000_000,
    }
    AVAIL = {
        "nemotron_math_proofs": {"1": 308_000_000},
        "common_pile_libretexts_reference": {"2": 35_000_000},
        "fineweb_edu": {"3": 50_000_000_000},
    }

    def test_real_shortfall_is_tolerated_at_two_percent(self) -> None:
        quotas, short, shortfall = _plan(
            self.TARGETS, self.AVAIL, target=self.TARGET, tolerance=0.02
        )
        self.assertEqual(set(short), {"nemotron_math_proofs", "common_pile_libretexts_reference"})
        self.assertAlmostEqual(shortfall / self.TARGET, 0.0077, places=4)
        # The thin sources still contribute everything they have.
        self.assertEqual(quotas["1"]["nemotron_math_proofs"], 308_000_000)
        self.assertEqual(quotas["2"]["common_pile_libretexts_reference"], 35_000_000)
        # A healthy source is still capped at its target, not drained.
        self.assertEqual(quotas["3"]["fineweb_edu"], 20_000_000_000)

    def test_zero_tolerance_reproduces_the_old_refusal(self) -> None:
        with self.assertRaises(RuntimeError):
            _plan(self.TARGETS, self.AVAIL, target=self.TARGET, tolerance=0.0)

    def test_a_large_shortfall_still_fails(self) -> None:
        """Losing a major source is not a rounding error and must not pass."""

        avail = dict(self.AVAIL)
        avail["fineweb_edu"] = {"3": 1_000_000}
        with self.assertRaises(RuntimeError):
            _plan(self.TARGETS, avail, target=self.TARGET, tolerance=0.02)

    def test_no_shortfall_needs_no_tolerance(self) -> None:
        avail = {
            "nemotron_math_proofs": {"1": 2_000_000_000},
            "common_pile_libretexts_reference": {"2": 200_000_000},
            "fineweb_edu": {"3": 50_000_000_000},
        }
        _quotas, short, shortfall = _plan(
            self.TARGETS, avail, target=self.TARGET, tolerance=0.0
        )
        self.assertEqual(short, {})
        self.assertEqual(shortfall, 0)

    def test_a_source_with_nothing_is_reported_not_skipped(self) -> None:
        avail = dict(self.AVAIL)
        avail["common_pile_libretexts_reference"] = {}
        _quotas, short, _ = _plan(
            self.TARGETS, avail, target=self.TARGET, tolerance=0.02
        )
        self.assertEqual(short["common_pile_libretexts_reference"], 97_000_000)


if __name__ == "__main__":
    unittest.main()
