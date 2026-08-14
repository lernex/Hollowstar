from __future__ import annotations

import random
import unittest

from metis_data.datatrove_blocks import DiskContaminationIndex, _postings_array
from metis_data.decontaminate import ContaminationIndex

import numpy as np


def _to_disk_index(index: ContaminationIndex) -> DiskContaminationIndex:
    """The ndarray twin of an in-memory index, built the way the loader does."""

    exact = np.array(sorted(digest.encode("ascii") for digest in index.exact), dtype="S64")
    return DiskContaminationIndex(
        exact=exact,
        ngram_postings=_postings_array(index.ngram_postings),
        short_ngram_postings=_postings_array(index.short_ngram_postings),
        code_ngram_postings=_postings_array(index.code_ngram_postings),
        ngram_size=index.ngram_size,
        minimum_matching_ngrams=index.minimum_matching_ngrams,
        short_ngram_size=index.short_ngram_size,
        minimum_short_matching_ngrams=index.minimum_short_matching_ngrams,
        code_ngram_size=index.code_ngram_size,
        minimum_code_matching_ngrams=index.minimum_code_matching_ngrams,
        code_skeleton_ngram_postings=_postings_array(index.code_skeleton_ngram_postings),
        code_skeleton_ngram_size=index.code_skeleton_ngram_size,
        minimum_code_skeleton_matching_ngrams=index.minimum_code_skeleton_matching_ngrams,
        maximum_shingle_rows=index.maximum_shingle_rows,
        match_fraction=index.match_fraction,
        contiguous_run_minimum=index.contiguous_run_minimum,
    )


WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
]

CODE_HOLDOUT = (
    "def solve(values):\n"
    "    total = 0\n"
    "    for value in values:\n"
    "        if value % 2 == 0:\n"
    "            total += value * 3\n"
    "    return total;\n"
)


class DiskIndexMatchesReferenceTests(unittest.TestCase):
    """The ndarray path must agree with the in-memory reference, document for document.

    The ndarray implementation is the one that actually runs on the corpus, and
    it is a hand-written twin of the mapping-based reference. Nothing pinned the
    two together, so a divergence in the ndarray path would not have failed any
    test -- it would simply have filtered the corpus differently and silently.
    """

    def _index_pair(self, holdouts, **tuning):
        settings = {
            "ngram_size": 5,
            "minimum_matching_ngrams": 2,
            "short_ngram_size": 3,
            "minimum_short_matching_ngrams": 4,
            "code_ngram_size": 4,
            "minimum_code_matching_ngrams": 3,
            "code_skeleton_ngram_size": 5,
            "minimum_code_skeleton_matching_ngrams": 3,
            "contiguous_run_minimum": 4,
            "match_fraction": 0.05,
        }
        settings.update(tuning)
        memory = ContaminationIndex.build(holdouts, **settings)
        return memory, _to_disk_index(memory)

    def test_agrees_on_randomised_documents(self) -> None:
        rng = random.Random(20260814)
        holdouts = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randrange(8, 60)))
            for _ in range(40
                           )
        ]
        holdouts.append(CODE_HOLDOUT)
        memory, disk = self._index_pair(holdouts)

        documents: list[str] = []
        # Clean documents, verbatim holdouts, and holdouts buried in filler --
        # the three shapes the run and count thresholds have to separate.
        for _ in range(120):
            documents.append(" ".join(rng.choice(WORDS) for _ in range(rng.randrange(0, 90))))
        documents.extend(holdouts)
        for holdout in holdouts[:20]:
            filler = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(5, 40)))
            documents.append(f"{filler} {holdout} {filler}")
        for holdout in holdouts[:20]:
            words = holdout.split()
            documents.append(" ".join(words[: max(1, len(words) // 2)]))
        documents.extend(["", "   ", "alpha", CODE_HOLDOUT.replace("total", "acc")])

        disagreements = [
            (document, memory.reason(document), disk.reason(document))
            for document in documents
            if memory.reason(document) != disk.reason(document)
        ]
        self.assertEqual(disagreements, [])

    def test_agrees_when_contiguous_run_is_disabled(self) -> None:
        rng = random.Random(99)
        holdouts = [
            " ".join(rng.choice(WORDS) for _ in range(rng.randrange(8, 40))) for _ in range(25)
        ]
        memory, disk = self._index_pair(holdouts, contiguous_run_minimum=0, match_fraction=0.0)
        documents = holdouts + [
            " ".join(rng.choice(WORDS) for _ in range(rng.randrange(0, 60))) for _ in range(80)
        ]
        for document in documents:
            self.assertEqual(memory.reason(document), disk.reason(document))

    def test_repeated_ngrams_do_not_inflate_the_match_count(self) -> None:
        """De-duplicating the probe must not change which documents are dropped.

        The reference counts distinct n-grams, because ngram_hashes returns a
        set. A document that repeats one short phrase supplies very few distinct
        n-grams but very many occurrences, so counting occurrences instead would
        drop it on repetition alone. Thresholds here are set so only the n-gram
        count can fire, which is what makes the two behaviours distinguishable.
        """

        phrase = " ".join(WORDS[:6])
        memory, disk = self._index_pair(
            [phrase],
            ngram_size=5,
            minimum_matching_ngrams=3,
            match_fraction=0.0,
            contiguous_run_minimum=4,
            short_ngram_size=40,
            minimum_short_matching_ngrams=50,
            code_ngram_size=40,
            minimum_code_matching_ngrams=50,
            code_skeleton_ngram_size=40,
            minimum_code_skeleton_matching_ngrams=50,
        )
        repeated = " ".join([phrase] * 30)
        # Two distinct n-grams overlap the holdout, below the threshold of three,
        # however many times they recur.
        self.assertIsNone(memory.reason(repeated))
        self.assertEqual(memory.reason(repeated), disk.reason(repeated))

    def test_repetition_still_cannot_reach_the_run_threshold(self) -> None:
        """A run is consecutive positions, not repeated occurrences."""

        phrase = " ".join(WORDS[:6])
        memory, disk = self._index_pair(
            [phrase],
            ngram_size=5,
            minimum_matching_ngrams=50,
            match_fraction=0.0,
            contiguous_run_minimum=4,
            short_ngram_size=40,
            minimum_short_matching_ngrams=50,
            code_ngram_size=40,
            minimum_code_matching_ngrams=50,
            code_skeleton_ngram_size=40,
            minimum_code_skeleton_matching_ngrams=50,
        )
        repeated = " ".join([phrase] * 30)
        self.assertIsNone(memory.reason(repeated))
        self.assertEqual(memory.reason(repeated), disk.reason(repeated))
        # The same words in one unbroken span do reach the run threshold.
        contiguous = " ".join(WORDS[:12])
        memory_run, disk_run = self._index_pair(
            [contiguous],
            ngram_size=5,
            minimum_matching_ngrams=50,
            match_fraction=0.0,
            contiguous_run_minimum=4,
            short_ngram_size=40,
            minimum_short_matching_ngrams=50,
            code_ngram_size=40,
            minimum_code_matching_ngrams=50,
            code_skeleton_ngram_size=40,
            minimum_code_skeleton_matching_ngrams=50,
        )
        # Buried in filler so the exact-digest test cannot fire first.
        embedded = f"{' '.join(WORDS[14:20])} {contiguous} {' '.join(WORDS[14:20])}"
        self.assertEqual(disk_run.reason(embedded), "benchmark_contiguous_run")
        self.assertEqual(memory_run.reason(embedded), disk_run.reason(embedded))

    def test_proportional_threshold_scales_on_distinct_ngrams(self) -> None:
        """match_fraction scales against the document's distinct n-grams.

        The reference derives the threshold from ngram_hashes, which is a set, so
        a repetitive document does not raise its own bar by repeating itself. A
        threshold scaled on total occurrences instead would let any document
        evade the filter simply by duplicating its content.
        """

        phrase = " ".join(WORDS[:10])
        memory, disk = self._index_pair(
            [phrase],
            ngram_size=5,
            minimum_matching_ngrams=1,
            match_fraction=0.5,
            contiguous_run_minimum=0,
            short_ngram_size=40,
            minimum_short_matching_ngrams=50,
            code_ngram_size=40,
            minimum_code_matching_ngrams=50,
            code_skeleton_ngram_size=40,
            minimum_code_skeleton_matching_ngrams=50,
        )
        repeated = " ".join([phrase] * 10)
        self.assertEqual(disk.reason(repeated), "benchmark_ngram")
        self.assertEqual(memory.reason(repeated), disk.reason(repeated))

    def test_empty_index_keeps_every_document(self) -> None:
        memory, disk = self._index_pair([])
        for document in ("alpha bravo charlie", "", CODE_HOLDOUT):
            self.assertIsNone(disk.reason(document))
            self.assertEqual(memory.reason(document), disk.reason(document))


if __name__ == "__main__":
    unittest.main()
