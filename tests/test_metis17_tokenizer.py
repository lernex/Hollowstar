from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import struct
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import models, pre_tokenizers

from metis_data.tokenizer import train_tokenizer
from metis_data17 import tokenizer as implementation
from metis_data17.common import digest_json, sha256_file
from metis_data17.tokenizer import (
    DIGIT_POLICY,
    PREPARED_SCHEMA,
    PRODUCTION_VOCABULARY_SIZE,
    TOKENIZER_RELEASE,
    TokenCacheLimits17,
    TokenizationSession17,
    build_tokenizer_sample17,
    iter_tokenizer_sample17,
    load_tokenizer17,
    token_cache_key17,
    tokenization_policy17,
    tokenize_parquet17,
    train_tokenizer17,
)


CORPUS = [
    "The year 2026 cost 147832 dollars and 55 cents",
    "x = 1234567 + 89 * 4321",
    "invoice 90210 total 31415926 balance 271828",
    "port 8080 pid 12345 offset 65536 length 1024",
    "def square(x):\n\treturn x**2  # preserve code whitespace\n",
    r"\int_{0}^{10} x^2\,dx = \frac{1000}{3}",
    "中文 Ελληνικά العربية हिन्दी 日本語 🧮🙂",
] * 8
ROUNDTRIPS = [
    "",
    "012345678901234567890",
    "-1.2300e-123 + 4.567E+89; 0x123abc; 0b101010",
    r"\sum_{i=123}^{456} \frac{x_i^{2026}}{0.0001} \quad \alpha_{98}",
    "def f(x=123.456e-7):\r\n\treturn {'key': x ** 2}  \r\n",
    " \t\n\r\n  \u00a0\u2003\u2009 \x00 ",
    "É e\u0301 ﬃ 中文 日本語 한국어 Ελληνικά العربية हिन्दी 🙂👩🏽‍💻",
    "١٢٣٤٥ ۱۲۳۴۵ १२३४५ １２３４５ 𝟘𝟙𝟚 ²³⁴ Ⅷ",
    "<|endoftext|>hello<|endoftext|> 123",
]


def _row(text: str, doc_id: str, *, category: str = "web", language: str = "en") -> dict:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "doc_id": doc_id,
        "content_hash": content_hash,
        "dedup_hash": content_hash,
        "source_id": f"{category}-publisher",
        "object_id": hashlib.sha256(f"object-{doc_id}".encode()).hexdigest(),
        "text": text,
        "metadata_json": '{"license":"test-only"}',
        "priority": 30,
        "quality_score": -1.0,
        "language": language,
        "category": category,
        "character_count": len(text),
    }


def _write_prepared(path: Path, rows: list[dict], *, row_group_size: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=PREPARED_SCHEMA), path, row_group_size=row_group_size)


def _reseal(path: Path, changes: dict | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("receipt_sha256")
    payload.update(changes or {})
    return implementation._sealed(path, payload)


def _test_only_artifact(directory: Path, size: int, *, production: bool) -> None:
    """Synthetic vocabulary for ID-width validation, never a production training shortcut."""
    directory.mkdir(parents=True)
    tokenizer = implementation._new_tokenizer()
    vocabulary = {"<|endoftext|>": 0}
    vocabulary.update(
        {f"UNUSED_TEST_FIXTURE_SYMBOL_{index}": index for index in range(1, size - 256)}
    )
    vocabulary.update(
        {symbol: index for index, symbol in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()), size - 256)}
    )
    tokenizer.model = models.BPE(vocab=vocabulary, merges=[], unk_token=None)
    tokenizer.add_special_tokens(["<|endoftext|>"])
    tokenizer.save(str(directory / "tokenizer.json"))
    implementation._sealed(
        directory / TOKENIZER_RELEASE,
        {
            "schema": "metis17.tokenizer/v1",
            "test_fixture_only": True,
            "production": production,
            "algorithm": "byte_level_bpe",
            "split_digits": True,
            "digit_policy": DIGIT_POLICY,
            "digit_policy_sha256": digest_json(DIGIT_POLICY),
            "dtype": "<u4",
            "byte_order": "little",
            "vocabulary_size": size,
            "requested_vocabulary_size": PRODUCTION_VOCABULARY_SIZE if production else size,
            "maximum_token_id": size - 1,
            "special_tokens": {"<|endoftext|>": 0},
            "eos_token": "<|endoftext|>",
            "eos_token_id": 0,
            "tokenizer_sha256": sha256_file(directory / "tokenizer.json"),
        },
    )


class WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = (Path.cwd() / f".metis17-tokenizer-test-{uuid.uuid4().hex}").resolve()
        self.root.mkdir()
        self.scratch = self.root / "node-local"
        self.addCleanup(shutil.rmtree, self.root)

    def sample(self, paths: list[Path], output: Path, **kwargs: object) -> dict:
        return build_tokenizer_sample17(paths, output, scratch_dir=self.scratch, **kwargs)

    def cache_index(self, output: Path) -> Path:
        return self.scratch / "token-cache" / "index.sqlite3"

    def train_small(self, name: str = "tokenizer", texts: list[str] | None = None) -> Path:
        directory = self.root / name
        train_tokenizer17(
            iter(CORPUS if texts is None else texts), directory,
            vocabulary_size=600, minimum_frequency=1, production=False,
        )
        return directory

    def assert_valid_receipt(self, receipt: dict) -> None:
        self.assertEqual(
            receipt["receipt_sha256"],
            digest_json({key: value for key, value in receipt.items() if key != "receipt_sha256"}),
        )


class Tokenizer17ArtifactTests(WorkspaceTest):
    def test_production_target_is_exact_and_tiny_corpus_is_never_padded(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 131072"):
            train_tokenizer17(CORPUS, self.root / "wrong-size", vocabulary_size=65_536)
        with self.assertRaisesRegex(ValueError, "exactly 131072"):
            train_tokenizer17(["abc 123"] * 2, self.root / "shortfall")
        self.assertFalse((self.root / "shortfall" / TOKENIZER_RELEASE).exists())
        self.assertFalse((self.root / ".shortfall.training-incomplete").exists())

    def test_small_artifact_requires_explicit_nonproduction_mode(self) -> None:
        directory = self.train_small()
        release = json.loads((directory / TOKENIZER_RELEASE).read_text())
        self.assert_valid_receipt(release)
        self.assertFalse(release["production"])
        self.assertLess(release["vocabulary_size"], PRODUCTION_VOCABULARY_SIZE)
        with self.assertRaisesRegex(ValueError, "test tokenizer"):
            load_tokenizer17(directory)
        tokenizer = load_tokenizer17(directory, production=False)
        self.assertEqual(len(tokenizer.get_vocab()), release["vocabulary_size"])
        self.assertEqual(release["special_tokens"], {"<|endoftext|>": 0})

    def test_empty_corpus_and_digit_bearing_specials_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            train_tokenizer17([], self.root / "empty", vocabulary_size=300, production=False)
        with self.assertRaisesRegex(ValueError, "containing digits"):
            train_tokenizer17(CORPUS, self.root / "special", special_tokens=["<|number123|>"])
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                train_tokenizer17(CORPUS, self.root / "invalid", vocabulary_size=value)

    def test_special_tokens_cannot_introduce_lossy_unicode_decoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "Special token does not round-trip"):
            train_tokenizer17(
                CORPUS, self.root / "unicode-special", vocabulary_size=600,
                special_tokens=["<|éos|>"], production=False,
            )
        self.assertFalse((self.root / "unicode-special" / TOKENIZER_RELEASE).exists())

    def test_roundtrip_code_math_unicode_and_whitespace_without_boundaries(self) -> None:
        tokenizer = load_tokenizer17(self.train_small(), production=False)
        for text in ROUNDTRIPS:
            with self.subTest(text=text):
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)
        for number in ("0123456789", "147832", "1234567", "31415926"):
            self.assertEqual(tokenizer.encode(number, add_special_tokens=False).tokens, list(number))

    def test_digit_pretokenization_matches_legacy_and_false_remains_unchanged(self) -> None:
        tokenizer = load_tokenizer17(self.train_small(), production=False)
        for split_digits in (True, False):
            directory = self.root / f"legacy-{split_digits}"
            release = train_tokenizer(
                CORPUS, output_dir=directory, vocabulary_size=600,
                special_tokens=["<|endoftext|>"], minimum_frequency=1, split_digits=split_digits,
            )
            legacy = implementation.Tokenizer.from_file(str(directory / "tokenizer.json"))
            self.assertIs(release["split_digits"], split_digits)
            if split_digits:
                for text in ROUNDTRIPS:
                    with self.subTest(text=text):
                        self.assertEqual(
                            tokenizer.pre_tokenizer.pre_tokenize_str(text),
                            legacy.pre_tokenizer.pre_tokenize_str(text),
                        )
            else:
                self.assertLess(len(legacy.encode("147832").tokens), 6)
                self.assertTrue(release["uint16_safe"])

    def test_training_never_overwrites_an_existing_artifact(self) -> None:
        directory = self.train_small()
        before = (directory / "tokenizer.json").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "immutable"):
            train_tokenizer17(["different"], directory, vocabulary_size=600, production=False)
        self.assertEqual((directory / "tokenizer.json").read_bytes(), before)

    def test_existing_empty_output_and_multiple_literal_specials_are_supported(self) -> None:
        directory = self.root / "empty-output"
        directory.mkdir()
        specials = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
        release = train_tokenizer17(
            CORPUS, directory, vocabulary_size=600, special_tokens=specials,
            minimum_frequency=1, production=False,
        )
        tokenizer = load_tokenizer17(directory, production=False)
        self.assertEqual(release["special_tokens"], {token: index for index, token in enumerate(specials)})
        text = "\t<|fim_prefix|>x = 123\n<|fim_suffix|><|fim_middle|>"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)
        self.assertEqual(tokenizer.get_vocab_size(), release["vocabulary_size"])
        policy = tokenization_policy17()
        policy["digit_policy"]["individual_digits"] = False
        self.assertTrue(tokenization_policy17()["digit_policy"]["individual_digits"])
        self.assertTrue(release["digit_policy"]["individual_digits"])

    def test_artifact_hash_and_self_hash_are_verified(self) -> None:
        directory = self.train_small()
        path = directory / "tokenizer.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            load_tokenizer17(directory, production=False)
        _reseal(directory / TOKENIZER_RELEASE, {"tokenizer_sha256": sha256_file(path)})
        load_tokenizer17(directory, production=False)
        payload = json.loads((directory / TOKENIZER_RELEASE).read_text())
        payload["vocabulary_size"] += 1
        (directory / TOKENIZER_RELEASE).write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "Receipt hash mismatch"):
            load_tokenizer17(directory, production=False)

    def test_resealed_wrong_digit_policy_and_serialization_are_rejected(self) -> None:
        original = self.train_small()
        for mutation in ("manifest", "individual-digits", "order", "normalizer", "padding"):
            with self.subTest(mutation=mutation):
                directory = self.root / mutation
                shutil.copytree(original, directory)
                path = directory / "tokenizer.json"
                serialized = json.loads(path.read_text())
                changes = {}
                if mutation == "manifest":
                    changes["split_digits"] = False
                elif mutation == "individual-digits":
                    serialized["pre_tokenizer"]["pretokenizers"][0]["individual_digits"] = False
                elif mutation == "order":
                    serialized["pre_tokenizer"]["pretokenizers"].reverse()
                elif mutation == "normalizer":
                    serialized["normalizer"] = {"type": "NFKC"}
                else:
                    tokenizer = implementation.Tokenizer.from_file(str(path))
                    tokenizer.enable_padding(length=1000)
                    serialized = json.loads(tokenizer.to_str())
                path.write_text(json.dumps(serialized), encoding="utf-8")
                changes["tokenizer_sha256"] = sha256_file(path)
                _reseal(directory / TOKENIZER_RELEASE, changes)
                with self.assertRaises(ValueError):
                    load_tokenizer17(directory, production=False)

    def test_actual_131072_vocabulary_and_high_ids_use_little_endian_uint32(self) -> None:
        directory = self.root / "wide-id-test-fixture"
        _test_only_artifact(directory, PRODUCTION_VOCABULARY_SIZE, production=True)
        tokenizer = load_tokenizer17(directory)
        self.assertEqual(tokenizer.get_vocab_size(with_added_tokens=True), 131_072)
        text = "x = 65536 + 131071;\n\t🙂"
        expected = tokenizer.encode(text, add_special_tokens=False).ids
        self.assertGreater(max(expected), 65_535)
        source = self.root / "prepared.parquet"
        _write_prepared(source, [_row(text, "wide-id-record")])
        output = self.root / "ids"
        receipt = tokenize_parquet17([source], output, directory, scratch_dir=self.scratch, partition_id="wide-test")
        row = pq.read_table(output / receipt["metadata_path"]).to_pylist()[0]
        data = (output / row["ids_path"]).read_bytes()
        self.assertEqual(row["token_offset"], 0)
        self.assertEqual(row["token_count"], len(expected))
        self.assertEqual(len(data), len(expected) * 4)
        self.assertEqual(list(struct.unpack(f"<{len(expected)}I", data)), expected)
        self.assertEqual(tokenizer.decode(expected, skip_special_tokens=False), text)

    def test_production_contract_uses_actual_vocabulary_not_claimed_target(self) -> None:
        directory = self.root / "short-test-fixture"
        _test_only_artifact(directory, 300, production=True)
        _reseal(directory / TOKENIZER_RELEASE, {"vocabulary_size": 131_072, "maximum_token_id": 131_071})
        with self.assertRaisesRegex(ValueError, "exactly 131072"):
            load_tokenizer17(directory)
        _reseal(directory / TOKENIZER_RELEASE, {"production": False})
        with self.assertRaisesRegex(ValueError, "vocabulary does not match"):
            load_tokenizer17(directory, production=False)

    def test_sparse_ids_and_mismatched_specials_are_rejected(self) -> None:
        directory = self.train_small()
        release = directory / TOKENIZER_RELEASE
        _reseal(release, {"special_tokens": {"<|endoftext|>": 1}})
        with self.assertRaisesRegex(ValueError, "special tokens"):
            load_tokenizer17(directory, production=False)
        _reseal(release, {"special_tokens": {"<|endoftext|>": 0}})
        path = directory / "tokenizer.json"
        payload = json.loads(path.read_text())
        payload["model"]["vocab"]["a"] = 65_536
        path.write_text(json.dumps(payload))
        _reseal(release, {"tokenizer_sha256": sha256_file(path)})
        with self.assertRaises(ValueError):
            load_tokenizer17(directory, production=False)

    def test_manifest_integer_fields_are_not_coerced_from_booleans_or_floats(self) -> None:
        directory = self.train_small()
        path = directory / TOKENIZER_RELEASE
        initial = path.read_bytes()
        manifest = json.loads(initial)
        for field, value in (
            ("eos_token_id", False),
            ("vocabulary_size", float(manifest["vocabulary_size"])),
            ("maximum_token_id", float(manifest["maximum_token_id"])),
        ):
            with self.subTest(field=field):
                path.write_bytes(initial)
                _reseal(path, {field: value})
                with self.assertRaises(ValueError):
                    load_tokenizer17(directory, production=False)


class Tokenizer17CacheTests(WorkspaceTest):
    def setUp(self) -> None:
        super().setUp()
        self.tokenizer_dir = self.train_small()
        self.output = self.root / "cache-output"
        self.source = self.root / "prepared.parquet"

    def tokenize(self, paths: list[Path] | None = None, **kwargs: object) -> dict:
        return tokenize_parquet17(
            [self.source] if paths is None else paths, self.output, self.tokenizer_dir,
            scratch_dir=self.scratch, partition_id="worker-000", production=False, **kwargs,
        )

    def test_offsets_counts_and_exactly_once_record_coverage_with_duplicate_text(self) -> None:
        sources = {
            self.source: [_row(text, f"a-{index}") for index, text in enumerate(["abc", "123", "abc", "", ROUNDTRIPS[4]])],
            self.root / "second.parquet": [_row(text, f"b-{index}") for index, text in enumerate(["new", " \n", "xyz"])],
            self.root / "empty.parquet": [],
        }
        for path, rows in sources.items():
            _write_prepared(path, rows)
        with mock.patch.object(implementation, "_encode_batch", wraps=implementation._encode_batch) as encode:
            receipt = self.tokenize(list(reversed(sources)), batch_size=2)
        self.assertTrue(all(len(call.args[1]) <= 2 for call in encode.call_args_list))
        self.assert_valid_receipt(receipt)
        table = pq.read_table(self.output / receipt["metadata_path"])
        self.assertNotIn("text", table.schema.names)
        metadata = table.to_pylist()
        expected_refs = [
            (str(path), index) for path in sorted(sources) for index in range(len(sources[path]))
        ]
        self.assertEqual([(row["source_shard"], row["source_row"]) for row in metadata], expected_refs)
        self.assertEqual(len(set(expected_refs)), len(metadata))
        tokenizer = load_tokenizer17(self.tokenizer_dir, production=False)
        total = 0
        for row in metadata:
            original = sources[Path(row["source_shard"])][row["source_row"]]
            self.assertEqual(row["doc_id"], original["doc_id"])
            self.assertEqual(row["content_hash"], original["content_hash"])
            expected = tokenizer.encode(original["text"], add_special_tokens=False).ids
            binary = (self.output / row["ids_path"]).read_bytes()
            actual = struct.unpack_from(f"<{row['token_count']}I", binary, row["token_offset"] * 4)
            self.assertEqual(list(actual), expected)
            self.assertEqual(row["token_count"], len(expected))
            total += len(expected)
        self.assertEqual(receipt["token_count"], total)
        self.assertEqual(receipt["records"], len(metadata))
        self.assertEqual(receipt["encoded_documents"], len({row["text"] for rows in sources.values() for row in rows}))
        self.assertEqual(receipt["encoded_documents"] + receipt["reused_documents"], len(metadata))
        for binary in (self.output / "cache").glob("*/*/partitions/*/shards/*/ids.bin"):
            position = 0
            for row in pq.read_table(binary.with_name("offsets.parquet")).to_pylist():
                self.assertEqual(row["token_offset"], position)
                position += row["token_count"]
            self.assertEqual(position * 4, binary.stat().st_size)

    def test_replay_and_selection_never_reencode_unchanged_text(self) -> None:
        rows = [_row(text, f"doc-{index}") for index, text in enumerate(["abc", "123", "unchanged"])]
        _write_prepared(self.source, rows)
        initial = self.tokenize(batch_size=2)
        before = sorted((self.output / "cache").glob("*/*/partitions/*/shards/*/ids.bin"))
        selection = self.root / "selection.parquet"
        _write_prepared(selection, [rows[2], rows[0], rows[2]])
        with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("unexpected re-encoding")):
            replay = self.tokenize(batch_size=1)
            selected = self.tokenize([selection], batch_size=1)
        self.assertEqual(replay, initial)
        self.assertEqual(selected["encoded_documents"], 0)
        self.assertEqual(selected["reused_documents"], 3)
        self.assertEqual(sorted((self.output / "cache").glob("*/*/partitions/*/shards/*/ids.bin")), before)

    def test_changed_text_and_changed_tokenizer_invalidate_cache_keys(self) -> None:
        _write_prepared(self.source, [_row("same text", "same-id")])
        first = self.tokenize()
        first_row = pq.read_table(self.output / first["metadata_path"]).to_pylist()[0]
        _write_prepared(self.source, [_row("changed exact text", "same-id")])
        changed = self.tokenize()
        changed_row = pq.read_table(self.output / changed["metadata_path"]).to_pylist()[0]
        self.assertEqual(changed["encoded_documents"], 1)
        self.assertNotEqual(first["run_id"], changed["run_id"])
        self.assertNotEqual(first_row["cache_key"], changed_row["cache_key"])
        self.assertTrue((self.output / first["receipt_path"]).exists())
        other = self.train_small("other-tokenizer", ["changed changed completely different tokenizer"] * 10)
        with mock.patch.object(implementation, "_encode_batch", wraps=implementation._encode_batch) as encode:
            second_tokenizer = tokenize_parquet17(
                [self.source], self.output, other, production=False,
                scratch_dir=self.scratch, partition_id="worker-000",
            )
        self.assertEqual(encode.call_count, 1)
        other_row = pq.read_table(self.output / second_tokenizer["metadata_path"]).to_pylist()[0]
        self.assertNotEqual(changed_row["cache_key"], other_row["cache_key"])
        self.assertNotEqual(changed_row["ids_path"], other_row["ids_path"])

    def test_encoding_policy_is_part_of_durable_cache_identity(self) -> None:
        _write_prepared(self.source, [_row("123456", "one")])
        first = self.tokenize()
        changed_policy = {**tokenization_policy17(), "implementation_revision": "explicit-test-revision"}
        with mock.patch.object(implementation, "tokenization_policy17", return_value=changed_policy):
            second = self.tokenize()
        self.assertEqual(second["encoded_documents"], 1)
        self.assertNotEqual(first["run_id"], second["run_id"])
        content_hash = hashlib.sha256(b"123456").hexdigest()
        tokenizer_hash = first["identity"]["tokenizer_sha256"]
        self.assertNotEqual(
            token_cache_key17(content_hash, tokenizer_hash),
            token_cache_key17(content_hash, tokenizer_hash, changed_policy),
        )

    def test_wrong_content_hash_never_publishes_success(self) -> None:
        row = _row("original", "one")
        row["text"] = "tampered"
        _write_prepared(self.source, [row])
        with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("must validate first")):
            with self.assertRaisesRegex(ValueError, "content_hash"):
                self.tokenize()
        self.assertFalse(list((self.output / "runs").glob("*/receipt.json")))
        _write_prepared(self.source, [_row("corrected", "one")])
        self.assertEqual(self.tokenize()["records"], 1)

    def test_corrupt_committed_artifacts_are_rejected_without_reencoding(self) -> None:
        _write_prepared(self.source, [_row("cached exact bytes 123", "one")])
        for corruption in ("binary", "cache-offsets", "cache-receipt", "run-offsets", "run-receipt", "index"):
            with self.subTest(corruption=corruption):
                self.output = self.root / f"output-{corruption}"
                receipt = self.tokenize()
                cache_shard = next((self.output / "cache").glob("*/*/partitions/*/shards/*"))
                if corruption == "index":
                    with sqlite3.connect(self.cache_index(self.output)) as database:
                        database.execute("UPDATE entries SET token_offset=token_offset+1")
                else:
                    path = {
                        "binary": cache_shard / "ids.bin",
                        "cache-offsets": cache_shard / "offsets.parquet",
                        "cache-receipt": cache_shard / "receipt.json",
                        "run-offsets": self.output / receipt["metadata_path"],
                        "run-receipt": self.output / receipt["receipt_path"],
                    }[corruption]
                    data = bytearray(path.read_bytes())
                    data[len(data) // 2] ^= 1
                    path.write_bytes(data)
                with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("corruption cannot retrain")):
                    with self.assertRaises(ValueError):
                        self.tokenize()

    def test_deleted_cache_index_is_rebuilt_from_committed_shards(self) -> None:
        _write_prepared(self.source, [_row("one", "1"), _row("two", "2"), _row("three", "3")])
        receipt = self.tokenize(batch_size=2)
        index = self.cache_index(self.output)
        index.unlink()
        with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("index recovery must not encode")):
            self.assertEqual(self.tokenize(batch_size=1), receipt)
        self.assertTrue(index.exists())

    def test_crash_after_shard_commit_recovers_without_encoding_committed_text(self) -> None:
        rows = [_row(text, str(index)) for index, text in enumerate(["one", "two", "three", "four"])]
        _write_prepared(self.source, rows)
        publish = implementation._publish

        def crash_after_publication(stage: Path, destination: Path) -> None:
            publish(stage, destination)
            if destination.parent.name == "shards":
                raise RuntimeError("simulated crash after atomic directory publication")

        with mock.patch.object(implementation, "_publish", side_effect=crash_after_publication):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                self.tokenize(batch_size=2)
        with mock.patch.object(implementation, "_encode_batch", wraps=implementation._encode_batch) as encode:
            receipt = self.tokenize(batch_size=2)
        encoded = [text for call in encode.call_args_list for text in call.args[1]]
        self.assertEqual(encoded, ["three", "four"])
        self.assertEqual(receipt["records"], 4)
        self.assertEqual(receipt["encoded_documents"], 2)

    def test_crash_before_run_commit_reuses_all_atomic_shards(self) -> None:
        _write_prepared(self.source, [_row("one", "1"), _row("two", "2")])
        publish = implementation._publish

        def crash_before_run(stage: Path, destination: Path) -> None:
            if destination.parent.parent.name == "runs":
                raise RuntimeError("simulated final metadata interruption")
            publish(stage, destination)

        with mock.patch.object(implementation, "_publish", side_effect=crash_before_run):
            with self.assertRaisesRegex(RuntimeError, "simulated final"):
                self.tokenize(batch_size=1)
        with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("cached IDs must survive")):
            receipt = self.tokenize(batch_size=2)
        self.assertEqual(receipt["records"], 2)
        self.assertEqual(receipt["encoded_documents"], 0)

    def test_orphan_staging_is_recovered_but_missing_committed_receipt_is_rejected(self) -> None:
        _write_prepared(self.source, [_row("one", "1")])
        receipt = self.tokenize()
        shard = next((self.output / "cache").glob("*/*/partitions/*/shards/*"))
        incomplete = shard.parent.parent / ".incomplete-shard"
        incomplete.mkdir()
        (incomplete / "ids.bin").write_bytes(b"\xff")
        incomplete_run = (self.output / receipt["receipt_path"]).parent.parent / ".incomplete-run"
        incomplete_run.mkdir()
        (incomplete_run / "offsets.parquet").write_bytes(b"interrupted")
        self.tokenize()
        self.assertFalse(incomplete.exists())
        self.assertFalse(incomplete_run.exists())
        (shard / "receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.tokenize()

    def test_duplicate_input_paths_and_invalid_prepared_schema_are_rejected(self) -> None:
        _write_prepared(self.source, [_row("one", "1")])
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.tokenize([self.source, self.source])
        pq.write_table(pa.table({"text": ["raw records are not prepared"]}), self.source)
        with self.assertRaisesRegex(ValueError, "Prepared Parquet"):
            self.tokenize()

    def test_resealed_missing_record_and_wrong_offset_are_rejected(self) -> None:
        _write_prepared(self.source, [_row("one", "1"), _row("two", "2")])
        receipt = self.tokenize()
        path = self.output / receipt["metadata_path"]
        table = pq.read_table(path)
        pq.write_table(table.slice(0, 1), path)
        _reseal(self.output / receipt["receipt_path"], {"metadata_sha256": sha256_file(path), "records": 1})
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.tokenize()
        rows = table.to_pylist()
        rows[1]["token_offset"] += 1
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        _reseal(self.output / receipt["receipt_path"], {"metadata_sha256": sha256_file(path), "records": 2})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.tokenize()

    def test_resealed_source_identity_swaps_do_not_hide_coverage_errors(self) -> None:
        _write_prepared(self.source, [_row("one", "1"), _row("two", "2")])
        receipt = self.tokenize()
        path = self.output / receipt["metadata_path"]
        table = pq.read_table(path)
        rows = table.to_pylist()
        rows[0]["doc_id"], rows[1]["doc_id"] = rows[1]["doc_id"], rows[0]["doc_id"]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
        _reseal(self.output / receipt["receipt_path"], {"metadata_sha256": sha256_file(path)})
        with self.assertRaisesRegex(ValueError, "record identity"):
            self.tokenize()

    def test_record_id_hashes_are_verified_when_rebuilding_an_index(self) -> None:
        _write_prepared(self.source, [_row("one", "1"), _row("two", "2")])
        self.tokenize()
        shard = next((self.output / "cache").glob("*/*/partitions/*/shards/*"))
        binary = shard / "ids.bin"
        data = bytearray(binary.read_bytes())
        data[0] ^= 1
        binary.write_bytes(data)
        _reseal(shard / "receipt.json", {"ids_sha256": sha256_file(binary)})
        self.cache_index(self.output).unlink()
        with self.assertRaisesRegex(ValueError, "receipt mismatch|record ID checksum"):
            self.tokenize()


class Tokenizer17SampleTests(WorkspaceTest):
    def test_stratified_selection_matches_independent_hash_rank_reference(self) -> None:
        caps = {"math": 70, "code": 70, "science": 70, "web": 70, "multilingual": 70}
        sources: dict[Path, list[dict]] = {self.root / "a.parquet": [], self.root / "b.parquet": []}
        for category in caps:
            for index in range(12):
                text = f"{category} example {index}: " + "α" * (index % 4)
                path = list(sources)[index % 2]
                sources[path].append(_row(text, f"{category}-{index}", category=category))
        sources[self.root / "b.parquet"].append(dict(sources[self.root / "a.parquet"][0], doc_id="duplicate"))
        sources[self.root / "b.parquet"].append(_row("x" * 71, "oversized", category="math"))
        sources[self.root / "a.parquet"].append(_row("not selected", "uncapped", category="other"))
        for path, rows in sources.items():
            _write_prepared(path, rows, row_group_size=2)
        output = self.root / "sample"
        receipt = self.sample(list(reversed(sources)), output, stratum_byte_caps=caps, seed="fixed-seed", batch_size=3)
        self.assertTrue(receipt["ready"])
        self.assert_valid_receipt(receipt)
        self.assertEqual(receipt["records_scanned"], sum(map(len, sources.values())))
        self.assertEqual(receipt["uncapped_documents"], 1)
        self.assertEqual(receipt["coverage"]["math"]["duplicate_documents"], 1)
        self.assertEqual(receipt["coverage"]["math"]["oversized_documents"], 1)
        expected_refs = []
        for category, cap in caps.items():
            unique = {}
            for path in sorted(sources):
                for index, row in enumerate(sources[path]):
                    if row["category"] == category and len(row["text"].encode("utf-8")) <= cap:
                        unique.setdefault(row["content_hash"], (path, index, row))
            ranked = sorted(
                unique.values(),
                key=lambda item: (
                    digest_json({"seed": "fixed-seed", "stratum": category, "content_hash": item[2]["content_hash"]}),
                    item[2]["content_hash"],
                ),
            )
            used = 0
            for path, index, row in ranked:
                size = len(row["text"].encode("utf-8"))
                if used + size <= cap:
                    expected_refs.append((str(path), index))
                    used += size
            self.assertEqual(receipt["coverage"][category]["selected_bytes"], used)
            self.assertLessEqual(used, cap)
        selected = pq.read_table(output / "samples.parquet").to_pylist()
        self.assertEqual(
            [(row["source_shard"], row["source_row"]) for row in selected],
            sorted(expected_refs),
        )
        self.assertNotIn("text", pq.read_schema(output / "samples.parquet").names)
        self.assertEqual(
            list(iter_tokenizer_sample17(output, batch_size=1)),
            [sources[Path(path)][row]["text"] for path, row in sorted(expected_refs)],
        )
        other = self.root / "sample-other-batch"
        self.sample(list(sources), other, stratum_byte_caps=caps, seed="fixed-seed", batch_size=7)
        self.assertEqual(pq.read_table(other / "samples.parquet").to_pylist(), selected)

    def test_missing_required_strata_and_minimum_bytes_block_production_iteration(self) -> None:
        source = self.root / "english.parquet"
        _write_prepared(source, [_row("english only", "one")])
        output = self.root / "sample"
        receipt = self.sample(
            [source], output, stratum_byte_caps={"web": 100, "math": 100},
            minimum_bytes_per_stratum={"web": 50},
        )
        self.assertFalse(receipt["ready"])
        self.assertEqual(receipt["missing_strata"], ["math", "web"])
        with self.assertRaisesRegex(ValueError, "not ready"):
            list(iter_tokenizer_sample17(output))
        self.assertEqual(list(iter_tokenizer_sample17(output, production=False)), ["english only"])
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.sample(
                [source], self.root / "invalid", stratum_byte_caps={"web": 10},
                minimum_bytes_per_stratum={"web": 11},
            )

    def test_utf8_caps_count_bytes_and_never_truncate_documents(self) -> None:
        source = self.root / "unicode.parquet"
        _write_prepared(source, [
            _row("漢字", "fits", category="multilingual"),
            _row("🙂🙂🙂", "too-big", category="multilingual"),
        ])
        output = self.root / "sample"
        receipt = self.sample([source], output, stratum_byte_caps={"multilingual": 9})
        self.assertEqual(receipt["selected_bytes"], 6)
        self.assertEqual(receipt["coverage"]["multilingual"]["oversized_documents"], 1)
        self.assertEqual(list(iter_tokenizer_sample17(output)), ["漢字"])

    def test_category_language_strata_are_explicit(self) -> None:
        source = self.root / "multilingual.parquet"
        _write_prepared(source, [
            _row("English", "en", language="en"),
            _row("中文", "zh", language="zh"),
            _row("العربية", "ar", language="ar"),
        ])
        output = self.root / "sample"
        receipt = self.sample(
            [source], output, stratum_byte_caps={"web/en": 100, "web/zh": 100},
            stratum_columns=("category", "language"),
        )
        self.assertTrue(receipt["ready"])
        self.assertEqual(receipt["uncapped_documents"], 1)
        self.assertEqual(list(iter_tokenizer_sample17(output)), ["English", "中文"])

    def test_immutable_sample_replay_and_changed_policy_rejection(self) -> None:
        source = self.root / "prepared.parquet"
        _write_prepared(source, [_row("sample text", "one")])
        output = self.root / "sample"
        first = self.sample([source], output, stratum_byte_caps={"web": 100})
        self.assertEqual(self.sample([source], output, stratum_byte_caps={"web": 100}, batch_size=1), first)
        with self.assertRaisesRegex(ValueError, "policy changed"):
            self.sample([source], output, stratum_byte_caps={"web": 99})

    def test_changed_source_and_corrupt_sample_metadata_are_rejected(self) -> None:
        source = self.root / "prepared.parquet"
        rows = [_row("sample text", "one")]
        _write_prepared(source, rows)
        output = self.root / "sample"
        self.sample([source], output, stratum_byte_caps={"web": 100})
        _write_prepared(source, [_row("changed text", "one")])
        with self.assertRaisesRegex(ValueError, "source changed"):
            list(iter_tokenizer_sample17(output))
        _write_prepared(source, rows)
        path = output / "samples.parquet"
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "Corrupt tokenizer sample"):
            list(iter_tokenizer_sample17(output))

    def test_quarantine_and_raw_records_cannot_enter_tokenizer_sample(self) -> None:
        for name in ("quarantine", "raw"):
            source = self.root / name / "object.parquet"
            _write_prepared(source, [_row("even a schema-shaped quarantined record must not train", "one")])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "eligible prepared"):
                self.sample([source], self.root / f"sample-{name}", stratum_byte_caps={"web": 100})
        source = self.root / "raw-shaped.parquet"
        pq.write_table(pa.table({"text": ["raw schema"]}), source)
        with self.assertRaisesRegex(ValueError, "Prepared Parquet"):
            self.sample([source], self.root / "sample-schema", stratum_byte_caps={"web": 100})

    def test_sample_restart_publishes_only_complete_manifest(self) -> None:
        source = self.root / "prepared.parquet"
        _write_prepared(source, [_row("sample text", "one")])
        output = self.root / "sample"
        with mock.patch.object(implementation, "_publish", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.sample([source], output, stratum_byte_caps={"web": 100})
        self.assertFalse((output / "SAMPLE_RECEIPT.json").exists())
        receipt = self.sample([source], output, stratum_byte_caps={"web": 100})
        self.assertTrue(receipt["ready"])
        self.assertEqual(list(iter_tokenizer_sample17(output)), ["sample text"])


class Tokenizer17ScratchArchitectureTests(WorkspaceTest):
    def setUp(self) -> None:
        super().setUp()
        self.tokenizer_dir = self.train_small()
        self.output = self.root / "durable"
        self.source = self.root / "prepared.parquet"
        _write_prepared(self.source, [_row("one", "1"), _row("two", "2")])

    def run_partition(self, partition: str = "p000", *, scratch: Path | None = None, paths: list[Path] | None = None, **kwargs: object) -> dict:
        return tokenize_parquet17(
            [self.source] if paths is None else paths, self.output, self.tokenizer_dir,
            scratch_dir=self.scratch if scratch is None else scratch,
            partition_id=partition, production=False, **kwargs,
        )

    def test_explicit_local_scratch_and_partition_are_required(self) -> None:
        with self.assertRaises(TypeError):
            tokenize_parquet17([self.source], self.output, self.tokenizer_dir, production=False)
        with self.assertRaises(TypeError):
            build_tokenizer_sample17([self.source], self.root / "sample", stratum_byte_caps={"web": 10})
        for scratch in (self.output, self.output / "scratch", self.root):
            with self.subTest(scratch=scratch), self.assertRaisesRegex(ValueError, "disjoint"):
                self.run_partition(scratch=scratch)
        with self.assertRaisesRegex(ValueError, "partition_id"):
            self.run_partition("../not-a-worker")

    def test_network_and_unknown_filesystems_fail_before_sqlite(self) -> None:
        for filesystem in ("lustre", "nfs", "nfs4", "cifs", "unknown", "overlay"):
            with self.subTest(filesystem=filesystem):
                with mock.patch.object(implementation, "_filesystem_type17", return_value=filesystem):
                    with mock.patch.object(implementation.sqlite3, "connect", side_effect=AssertionError("must not open SQLite")):
                        with self.assertRaisesRegex(ValueError, "verified node-local"):
                            self.run_partition()
                        with self.assertRaisesRegex(ValueError, "verified node-local"):
                            self.sample([self.source], self.root / "sample", stratum_byte_caps={"web": 10})

    def test_symlinks_cannot_redirect_a_local_database_into_durable_output(self) -> None:
        self.run_partition()
        index = self.cache_index(self.output)
        misplaced = self.output / "forbidden.sqlite3"
        index.rename(misplaced)
        index.symlink_to(misplaced)
        with mock.patch.object(implementation.sqlite3, "connect", side_effect=AssertionError("symlink must fail first")):
            with self.assertRaisesRegex(ValueError, "verified node-local scratch"):
                self.run_partition()

    def test_all_sqlite_is_local_and_durable_artifacts_survive_scratch_eviction(self) -> None:
        connect = sqlite3.connect
        connections = []

        def guarded_connect(path: Path, *args: object, **kwargs: object) -> sqlite3.Connection:
            resolved = Path(path).resolve()
            self.assertTrue(resolved.is_relative_to(self.scratch))
            self.assertFalse(resolved.is_relative_to(self.output))
            connections.append(resolved)
            return connect(path, *args, **kwargs)

        with mock.patch.object(implementation.sqlite3, "connect", side_effect=guarded_connect):
            receipt = self.run_partition()
            self.sample([self.source], self.root / "sample", stratum_byte_caps={"web": 10})
            with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("retained IDs must survive eviction")):
                self.assertEqual(self.run_partition(), receipt)
        self.assertEqual(len(connections), 3)
        self.assertFalse(list(self.output.rglob("*.sqlite*")))
        self.assertFalse(list((self.root / "sample").rglob("*.sqlite*")))
        self.assertEqual(len(list(self.scratch.rglob("*.sqlite3"))), 1)

    def test_reusable_session_opens_one_database_and_verifies_each_used_shard_once(self) -> None:
        second = self.root / "second.parquet"
        _write_prepared(second, [_row("one", "again"), _row("three", "3")])
        validated = implementation._TokenCache._validated_entries
        calls = []

        def track(cache: object, shard_id: str, receipt: dict, *, verify_ids: bool = True):
            calls.append((shard_id, verify_ids))
            yield from validated(cache, shard_id, receipt, verify_ids=verify_ids)

        with mock.patch.object(implementation.sqlite3, "connect", wraps=sqlite3.connect) as connect:
            with mock.patch.object(implementation._TokenCache, "_validated_entries", track):
                with TokenizationSession17(
                    self.output, self.tokenizer_dir, scratch_dir=self.scratch,
                    partition_id="p000", production=False,
                ) as session:
                    initial = session.tokenize_parquet([self.source])
                    following = session.tokenize_parquet([second])
                    with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("replay cannot encode")):
                        self.assertEqual(session.tokenize_parquet([self.source]), initial)
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(following["encoded_documents"], 1)
        for shard in {shard for shard, _checked in calls}:
            self.assertEqual(calls.count((shard, True)), 1)

    def test_reusable_session_detects_changed_artifacts_between_calls(self) -> None:
        with TokenizationSession17(
            self.output, self.tokenizer_dir, scratch_dir=self.scratch,
            partition_id="p000", production=False,
        ) as session:
            receipt = session.tokenize_parquet([self.source])
            row = pq.read_table(self.output / receipt["metadata_path"]).to_pylist()[0]
            path = self.output / row["ids_path"]
            data = bytearray(path.read_bytes())
            data[0] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "checksum"):
                session.tokenize_parquet([self.source])

    def test_reusable_session_recovers_an_interrupted_call_without_reencoding(self) -> None:
        publish = implementation._publish

        def interrupted(stage: Path, destination: Path) -> None:
            publish(stage, destination)
            if destination.parent.name == "shards":
                raise RuntimeError("interrupted publication")

        with TokenizationSession17(
            self.output, self.tokenizer_dir, scratch_dir=self.scratch,
            partition_id="p000", production=False,
        ) as session:
            with mock.patch.object(implementation, "_publish", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    session.tokenize_parquet([self.source])
            with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("committed IDs cannot be encoded twice")):
                receipt = session.tokenize_parquet([self.source])
        self.assertEqual(receipt["records"], 2)
        self.assertEqual(receipt["encoded_documents"], 0)

    def test_warm_calls_do_not_recover_or_hash_unrelated_shards_or_partitions(self) -> None:
        scratch_a, scratch_b = self.root / "scratch-a", self.root / "scratch-b"
        self.run_partition("a", scratch=scratch_a)
        self.run_partition("b", scratch=scratch_b)
        old_shards = set((self.output / "cache").glob("*/*/partitions/*/shards/*"))
        partition_b = next(path for path in old_shards if path.parent.parent.name == "b")
        (partition_b / "ids.bin").write_bytes(b"corrupt but unrelated")
        fresh = self.root / "fresh.parquet"
        _write_prepared(fresh, [_row("never encoded before", "fresh")])
        read_receipt = implementation._TokenCache._receipt
        scanned = implementation.os.scandir
        sha = implementation.sha256_file

        def guard_receipt(cache: object, path: Path) -> dict:
            self.assertNotIn(path, old_shards)
            return read_receipt(cache, path)

        def guard_scan(path: object):
            if isinstance(path, (str, Path)):
                self.assertNotIn(Path(path).name, {"shards", "commits", "partitions"})
            return scanned(path)

        def guard_hash(path: Path) -> str:
            self.assertFalse(any(Path(path).is_relative_to(shard) for shard in old_shards))
            return sha(path)

        with mock.patch.object(implementation._TokenCache, "_receipt", guard_receipt):
            with mock.patch.object(implementation.os, "scandir", side_effect=guard_scan):
                with mock.patch.object(implementation, "sha256_file", side_effect=guard_hash):
                    result = self.run_partition("a", scratch=scratch_a, paths=[fresh])
        self.assertEqual(result["encoded_documents"], 1)

    def test_cold_rebuild_never_reads_another_partition(self) -> None:
        scratch_a, scratch_b = self.root / "scratch-a", self.root / "scratch-b"
        first = self.run_partition("a", scratch=scratch_a)
        self.run_partition("b", scratch=scratch_b)
        shutil.rmtree(scratch_a / "token-cache")
        receipt_reader = implementation._TokenCache._receipt

        def guard(cache: object, path: Path) -> dict:
            self.assertEqual(path.parent.parent.name, "a")
            return receipt_reader(cache, path)

        with mock.patch.object(implementation._TokenCache, "_receipt", guard):
            with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("local loss cannot retrain retained IDs")):
                self.assertEqual(self.run_partition("a", scratch=scratch_a), first)

    def test_cross_partition_duplicates_are_explicit_and_local_index_count_is_bounded(self) -> None:
        first = self.run_partition("a")
        second = self.run_partition("b")
        self.assertEqual(first["encoded_documents"], 2)
        self.assertEqual(second["encoded_documents"], 2)
        self.assertNotEqual(first["identity"]["cache_partition"], second["identity"]["cache_partition"])
        self.assertEqual(len(list(self.scratch.rglob("*.sqlite3"))), 1)
        with mock.patch.object(implementation, "_encode_batch", side_effect=AssertionError("returning to a partition must reuse durable IDs")):
            self.assertEqual(self.run_partition("a"), first)
        self.assertEqual(len(list(self.scratch.rglob("*.sqlite3"))), 1)

    def test_worker_and_partition_leases_are_scoped_and_fail_fast(self) -> None:
        with TokenizationSession17(
            self.output, self.tokenizer_dir, scratch_dir=self.scratch, partition_id="a", production=False,
        ):
            with self.assertRaisesRegex(ValueError, "already held"):
                self.run_partition("b")
            with self.assertRaisesRegex(ValueError, "already held"):
                self.run_partition("a", scratch=self.root / "other-scratch")
            self.run_partition("b", scratch=self.root / "independent-scratch")
        self.run_partition("a", scratch=self.root / "other-scratch")

    def test_document_shard_byte_and_input_admission_limits_are_enforced(self) -> None:
        for limits, batch_size, message in (
            (TokenCacheLimits17(max_documents=1), 256, "admission limit"),
            (TokenCacheLimits17(max_shards=1), 1, "admission limit"),
            (TokenCacheLimits17(max_token_bytes=1), 256, "byte limit"),
        ):
            with self.subTest(message=message, limits=limits):
                partition = f"limit-{uuid.uuid4().hex}"
                with self.assertRaisesRegex(ValueError, message):
                    self.run_partition(partition, limits=limits, batch_size=batch_size)
        with self.assertRaisesRegex(ValueError, "Input shard count"):
            self.run_partition("paths", paths=[self.source, self.source], limits=TokenCacheLimits17(max_input_paths=1))
        self.run_partition("fixed-limits")
        with self.assertRaisesRegex(ValueError, "identity or admission limits"):
            self.run_partition("fixed-limits", limits=TokenCacheLimits17(max_documents=500_000))

    def test_sqlite_byte_limit_failure_keeps_published_ids_recoverable(self) -> None:
        _write_prepared(self.source, [_row(f"unique document {index}", str(index)) for index in range(800)])
        small = TokenCacheLimits17(max_scratch_bytes=256 * 1024)
        with self.assertRaisesRegex(ValueError, "node-local.*SQLite"):
            self.run_partition(limits=small, batch_size=128)
        retained = {
            row["content_hash"]
            for path in (self.output / "cache").glob("*/*/partitions/*/shards/*/offsets.parquet")
            for row in pq.read_table(path).to_pylist()
        }
        self.assertTrue(retained)
        self.assertLessEqual(sum(path.stat().st_size for path in self.scratch.rglob("*") if path.is_file()), small.max_scratch_bytes)
        with mock.patch.object(implementation, "_encode_batch", wraps=implementation._encode_batch) as encode:
            result = self.run_partition(limits=TokenCacheLimits17(max_scratch_bytes=4 * 1024**2), batch_size=128)
        self.assertEqual(result["records"], 800)
        for call in encode.call_args_list:
            self.assertTrue(all(hashlib.sha256(text.encode()).hexdigest() not in retained for text in call.args[1]))

    def test_sampling_limits_fail_without_publishing_or_leaking_sqlite(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate count"):
            self.sample(
                [self.source], self.root / "candidate-limit", stratum_byte_caps={"web": 10},
                max_candidate_documents=1,
            )
        large_ids = [_row(f"sample text {index}", "x" * 10_000 + str(index)) for index in range(40)]
        _write_prepared(self.source, large_ids)
        with self.assertRaisesRegex(ValueError, "node-local sampling SQLite"):
            self.sample(
                [self.source], self.root / "byte-limit", stratum_byte_caps={"web": 10_000},
                max_scratch_bytes=256 * 1024,
            )
        self.assertFalse(list(self.scratch.rglob("*.sqlite*")))
        self.assertFalse((self.root / "candidate-limit" / "SAMPLE_RECEIPT.json").exists())
        self.assertFalse((self.root / "byte-limit" / "SAMPLE_RECEIPT.json").exists())

    def test_sampling_decodes_only_potential_candidates_not_the_entire_corpus(self) -> None:
        _write_prepared(self.source, [
            _row("x" * 10_000, "uncapped", category="not-requested"),
            _row("y" * 1_000, "oversized"),
            _row("ok", "fits"),
        ], row_group_size=1)
        iterate = pq.ParquetFile.iter_batches
        text_groups = []

        def track(parquet: pq.ParquetFile, *args: object, **kwargs: object):
            if "text" in parquet.schema_arrow.names and "text" in kwargs.get("columns", []):
                text_groups.extend(kwargs["row_groups"])
            return iterate(parquet, *args, **kwargs)

        with mock.patch.object(pq.ParquetFile, "iter_batches", track):
            sample = self.sample([self.source], self.root / "sample", stratum_byte_caps={"web": 10})
        self.assertEqual(text_groups, [2])
        self.assertEqual(sample["records_scanned"], 3)
        self.assertEqual(sample["selected_bytes"], 2)
        self.assertIsNone(sample["coverage"]["web"]["eligible_bytes"])
        self.assertEqual(sample["coverage"]["web"]["decoded_documents"], 1)


if __name__ == "__main__":
    unittest.main()
