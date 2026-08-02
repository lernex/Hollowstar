from __future__ import annotations

import unittest
from pathlib import Path

class MediaDecodingTests(unittest.TestCase):
    """Image and audio columns must never be decoded to build the registry.

    MMMU and MathVista are pinned in the holdout registry and carry image
    columns. Decoding them imports Pillow from inside `datasets` iteration,
    which is where a missing decoder killed a five-hour acquisition run --
    and the decoded object was then discarded, because decontamination reads
    only text.
    """

    def test_image_and_audio_columns_are_cast_to_undecoded(self) -> None:
        from datasets import Audio, Image, Value

        from metis_data.holdouts import _without_media_decoding

        class Recording:
            def __init__(self, features):
                self.features = features
                self.casts: list[tuple[str, object]] = []

            def cast_column(self, name, feature):
                self.casts.append((name, feature))
                remaining = dict(self.features)
                remaining.pop(name)
                cast = Recording(remaining)
                cast.casts = self.casts
                return cast

        dataset = Recording(
            {
                "question": Value("string"),
                "image_1": Image(),
                "audio": Audio(),
            }
        )
        result = _without_media_decoding(dataset)
        self.assertEqual(
            sorted(name for name, _ in result.casts), ["audio", "image_1"]
        )
        # Cast, not dropped: the column stays, only its decoding stops.
        self.assertTrue(all(f.decode is False for _, f in result.casts))

    def test_a_dataset_without_features_is_returned_untouched(self) -> None:
        from metis_data.holdouts import _without_media_decoding

        class NoFeatures:
            features = None

        dataset = NoFeatures()
        self.assertIs(_without_media_decoding(dataset), dataset)


class RemoteCodeTests(unittest.TestCase):
    def test_streaming_never_enables_a_dataset_loading_script(self) -> None:
        """Building a decontamination registry must not execute third-party code.

        maveriq/bigbenchhard ships a loading script and no data files, so
        datasets stopped to ask on stdin -- inside Screen that is an immediate
        failed run. The registry now pins a data-only mirror, and the refusal
        is explicit so a future entry fails deterministically instead of
        prompting or silently executing.
        """
        from unittest import mock

        from metis_data import holdouts

        captured: dict = {}

        def fake_load_dataset(repo_id, **kwargs):
            captured.update(kwargs)
            return []

        with mock.patch.object(holdouts, "load_dataset", fake_load_dataset):
            rows = list(
                holdouts._benchmark_source_rows(
                    {"repo_id": "example/benchmark", "revision": "abc"},
                    {"config": "default", "split": "test"},
                    Path("/tmp/cache-does-not-need-to-exist"),
                )
            )
        self.assertEqual(rows, [])
        self.assertIs(captured.get("trust_remote_code"), False)

    def test_the_registry_pins_no_script_backed_benchmark(self) -> None:
        # bigbenchhard is the one that bit us; keep it named so a revert is loud.
        from metis_data.config import load_yaml, repository_root

        registry = load_yaml(
            repository_root() / "manifests" / "contamination" / "eval-holdouts.yaml"
        )
        repos = {str(e.get("repo_id")) for e in registry["benchmarks"]}
        self.assertNotIn("maveriq/bigbenchhard", repos)
        self.assertIn("lukaemon/bbh", repos)


class FragmentExplosionTests(unittest.TestCase):
    def test_a_tokenized_parallel_encoding_does_not_become_one_fragment_per_word(
        self,
    ) -> None:
        """Natural Questions carries {"text": ..., "tokens": [...]} everywhere.

        Recursing into those token lists emitted one holdout fragment per word
        -- 'episode', 'celebrity', 'guests' -- and produced 1.8M records in
        minutes, heading for tens of gigabytes. The token list is a redundant
        encoding of text that is already in the record, so it carries no
        information the text form does not.
        """
        from metis_data.holdouts import _benchmark_fragments

        row = {
            "question": {
                "text": "who sang the theme to the grand tour series",
                "tokens": ["who", "sang", "the", "theme", "grand", "tour"],
            },
            "document": {
                "title": "The Grand Tour (TV series)",
                "tokens": {"token": ["episode", "celebrity", "guests"]},
            },
        }
        texts = [text for _kind, text in _benchmark_fragments(row)]
        self.assertEqual(len(texts), 2)
        joined = " ".join(texts)
        self.assertIn("who sang the theme to the grand tour series", joined)
        self.assertIn("The Grand Tour (TV series)", joined)
        for word in ("episode", "celebrity", "guests", "sang"):
            self.assertNotIn(word, [t.strip() for t in texts])

    def test_a_row_cannot_emit_unbounded_fragments(self) -> None:
        # Backstop for an expanding schema no key name anticipates.
        from metis_data.holdouts import (
            MAXIMUM_FRAGMENTS_PER_ROW,
            _benchmark_fragments,
        )

        row = {"context": [f"sentence number {i} of the passage" for i in range(5000)]}
        self.assertEqual(
            len(list(_benchmark_fragments(row))), MAXIMUM_FRAGMENTS_PER_ROW
        )

    def test_short_answers_and_code_are_still_extracted(self) -> None:
        # The cap must not become a length filter: real benchmark answers and
        # test assertions are short, and dropping them was the wrong fix.
        from metis_data.holdouts import _benchmark_fragments

        row = {
            "question": "What does the function return?",
            "choices": ["zero", "the input plus one"],
            "solution": "The input plus one.",
            "test_list": ["assert f(1) == 2"],
        }
        kinds = {kind for kind, _ in _benchmark_fragments(row)}
        self.assertTrue({"query", "choices", "answer", "code"} <= kinds)


class ExtractionBoundsTests(unittest.TestCase):
    """No benchmark schema may produce unbounded holdout text.

    Three separate bugs in one day came from the same shape: a benchmark row
    carrying something enormous that the extractor happily walked. Tokenized
    word lists gave 1.8M single-word fragments. Raw Wikipedia HTML took the
    registry to 7GB. Naming each offending key as it appears does not converge,
    so the row is bounded outright and every bound reports itself.
    """

    def _nq_row(self) -> dict:
        return {
            "question": {
                "text": "who sang the theme to the grand tour series",
                "tokens": ["who", "sang", "the", "theme"],
            },
            "document": {
                "title": "The Grand Tour (TV series)",
                "url": "https://en.wikipedia.org/wiki/The_Grand_Tour",
                "html": "<html>" + ("<p>filler</p>" * 100000) + "</html>",
                "text": "A" * 400_000,
                "tokens": {"token": ["episode", "celebrity", "guests"]},
            },
        }

    def test_markup_and_locators_never_become_fragments(self) -> None:
        from metis_data.holdouts import _benchmark_fragments

        texts = [text for _kind, text in _benchmark_fragments(self._nq_row())]
        joined = " ".join(texts)
        self.assertNotIn("<p>", joined)
        self.assertNotIn("wikipedia.org", joined)
        self.assertLess(sum(len(t) for t in texts), 1000)

    def test_an_oversized_fragment_is_dropped_and_counted(self) -> None:
        from metis_data.holdouts import (
            MAXIMUM_FRAGMENT_CHARACTERS,
            _benchmark_fragments,
        )

        stats: dict = {}
        row = {"context": "B" * (MAXIMUM_FRAGMENT_CHARACTERS + 1)}
        self.assertEqual(list(_benchmark_fragments(row, stats)), [])
        self.assertEqual(stats.get("fragments_dropped_oversized"), 1)

    def test_a_row_cannot_exceed_the_character_budget(self) -> None:
        from metis_data.holdouts import (
            MAXIMUM_CHARACTERS_PER_ROW,
            _benchmark_fragments,
        )

        # Each passage is under the per-fragment limit, so only the row budget
        # can stop this -- the bound that needs no knowledge of the schema.
        stats: dict = {}
        row = {"context": [("C" * 20_000) + f" {i}" for i in range(40)]}
        total = sum(len(t) for _kind, t in _benchmark_fragments(row, stats))
        self.assertLessEqual(total, MAXIMUM_CHARACTERS_PER_ROW + 20_001)
        self.assertEqual(stats.get("rows_truncated_character_budget"), 1)

    def test_bounds_are_reported_rather_than_silent(self) -> None:
        from metis_data.holdouts import _benchmark_fragments

        stats: dict = {}
        list(_benchmark_fragments(self._nq_row(), stats))
        self.assertTrue(stats, "a bound fired but reported nothing")
