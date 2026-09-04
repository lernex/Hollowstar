from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from metis_data.tokenizer import (
    tokenizer_split_digits_setting,
    tokenizer_splits_digits,
    train_tokenizer,
)


CORPUS = [
    "The year 2026 cost 147832 dollars and 55 cents",
    "x = 1234567 + 89 * 4321",
    "invoice 90210 total 31415926 balance 271828",
    "port 8080 pid 12345 offset 65536 length 1024",
] * 60


def _train(tmp: str, *, split_digits: bool) -> tuple[Tokenizer, dict]:
    release = train_tokenizer(
        iter(CORPUS),
        output_dir=tmp,
        vocabulary_size=800,
        special_tokens=["<|endoftext|>"],
        minimum_frequency=1,
        split_digits=split_digits,
    )
    return Tokenizer.from_file(str(Path(tmp) / "tokenizer.json")), release


class DigitSplittingTests(unittest.TestCase):
    def test_digits_are_separate_tokens_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer, release = _train(tmp, split_digits=True)
            self.assertTrue(release["split_digits"])
            self.assertTrue(tokenizer_splits_digits(tokenizer))
            for number in ("147832", "1234567", "31415926"):
                with self.subTest(number=number):
                    self.assertEqual(
                        len(tokenizer.encode(number).tokens),
                        len(number),
                    )

    def test_default_still_merges_digit_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer, release = _train(tmp, split_digits=False)
            self.assertFalse(release["split_digits"])
            self.assertFalse(tokenizer_splits_digits(tokenizer))
            self.assertLess(
                len(tokenizer.encode("147832").tokens),
                len("147832"),
            )

    def test_round_trip_is_lossless_either_way(self) -> None:
        text = "invoice 90210 total 31415926 balance 271828"
        for split_digits in (True, False):
            with self.subTest(split_digits=split_digits):
                with tempfile.TemporaryDirectory() as tmp:
                    tokenizer, _release = _train(
                        tmp,
                        split_digits=split_digits,
                    )
                    self.assertEqual(
                        tokenizer.decode(tokenizer.encode(text).ids),
                        text,
                    )

    def test_letters_are_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer, _release = _train(tmp, split_digits=True)
            self.assertLess(
                len(tokenizer.encode("dollars").tokens),
                len("dollars"),
            )

    def test_non_boolean_manifest_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            tokenizer_split_digits_setting({"split_digits": "false"})


if __name__ == "__main__":
    unittest.main()
