from __future__ import annotations

import unittest

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
