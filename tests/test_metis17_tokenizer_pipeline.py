from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import unittest
import uuid
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from metis_data17 import tokenizer as tk
from metis_data17 import tokenizer_pipeline as pipeline
from metis_data17.common import digest_json, read_receipt, sha256_file


SPECIALS = [
    "<|endoftext|>", "<|bos|>", "<|user|>", "<|assistant|>",
    "<|system|>", "<|fim_prefix|>", "<|fim_suffix|>",
]


def _budget_snapshot() -> dict:
    return {
        "max_raw_bytes": 400_000_000_000, "max_working_bytes": 2_000_000_000_000,
        "policy_and_metadata_reserve_bytes": 20_000_000_000, "derived_limit_bytes": 1_580_000_000_000,
    }


def _quota(open_function):
    return SimpleNamespace(
        open=open_function, reserve=lambda total: total,
        unlink=lambda path: Path(path).unlink(missing_ok=True),
        replace=lambda source, destination: Path(source).replace(destination),
    )


class TokenizerPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = (Path.cwd() / f".metis17-pipeline-test-{uuid.uuid4().hex}").resolve()
        self.root = self.workspace / "release"
        self.root.mkdir(parents=True)
        self.scratch = self.workspace / "node-local"
        self.generation = hashlib.sha256(b"test-eligibility-policy").hexdigest()
        self.work = self.root / "tokenizer" / "test" / "generations" / self.generation
        self.addCleanup(shutil.rmtree, self.workspace)
        self.config = {
            "tokenizer": {
                "production": False, "vocabulary_size": 600, "split_digits": True,
                "special_tokens": SPECIALS, "dtype": "<u4", "byte_order": "little",
                "sample_target_bytes": 3000, "minimum_category_bytes": 100,
                "minimum_frequency": 1, "max_sample_document_bytes": 1024,
                "max_scratch_bytes": 4 * 1024**2, "max_sample_output_bytes": 4 * 1024**2,
                "max_model_output_bytes": 2 * 1024**2,
                "max_id_partition_output_bytes": 128 * 1024**2,
                "max_candidate_documents": 10_000, "max_input_paths": 100,
                "max_events_per_step": 16, "max_sample_attempts": 8, "batch_size": 8,
            },
            "limits": {"max_working_bytes": 2_000_000_000_000, "max_raw_bytes": 400_000_000_000},
        }
        self.write_run()
        self.write_generation(self.generation)
        self.requests: list[dict] = []
        self.released: list[str] = []
        self.sources: list[Path] = []

    def write_run(self) -> None:
        (self.root / "RUN.json").write_text(json.dumps(self.config, sort_keys=True), encoding="utf-8")

    def write_generation(self, generation: str) -> dict:
        return pipeline._seal(
            self.root / "preparation" / "generations" / f"{generation}.json",
            {"schema": "metis17.eligibility-generation/v1", "generation": generation, "policies": []},
        )

    @contextlib.contextmanager
    def reserve(self, request):
        self.requests.append(dict(request))
        try:
            yield {
                "reservation_id": request["reservation_id"], "reserved_bytes": request["requested_bytes"],
                "used_working_bytes": 0, "other_reserved_bytes": 0,
            }
        finally:
            self.released.append(request["reservation_id"])

    def step(self, **kwargs) -> dict:
        kwargs.setdefault("reserve_output", None if kwargs.get("working_budget") is not None else self.reserve)
        kwargs.setdefault("generation", self.generation)
        return pipeline.run_tokenizer_step(self.root, scratch_dir=self.scratch, test_mode=True, **kwargs)

    def append(self, event: dict, host: str = "host-a", *, newline: bool = True) -> Path:
        path = self.root / "events" / "eligible" / f"{host}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(json.dumps(event, sort_keys=True).encode() + (b"\n" if newline else b""))
        return path

    def source(
        self, category: str, texts: list[str] | None = None, *, suffix: str = "base",
        append: bool = True, host: str = "host-a", schema: str = "metis17.prepared-chunk/v1",
        source_id: str | None = None, language: str | None = None,
    ) -> tuple[dict, Path, dict]:
        object_id = hashlib.sha256(f"{category}-{suffix}".encode()).hexdigest()
        source_id = source_id or f"{category}-publisher"
        language = language or ("en" if category != "multilingual" else "zh")
        if texts is None:
            texts = [
                f"{category} example {index}: def f(x=123.456e-7):\n\treturn x ** 2  # exact whitespace 中文 αβ \n" * 2
                for index in range(10)
            ]
        rows = [
            {
                "doc_id": f"{category}-{suffix}-{index}", "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "dedup_hash": hashlib.sha256(text.encode()).hexdigest(),
                "source_id": source_id, "object_id": object_id, "text": text,
                "metadata_json": '{"normalization":"line-endings-only"}',
                "priority": 10, "quality_score": -1.0, "language": language,
                "category": category, "character_count": len(text),
            }
            for index, text in enumerate(texts)
        ]
        directory = self.root / "prepared" / object_id
        directory.mkdir(parents=True)
        parquet = directory / "documents.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=tk.PREPARED_SCHEMA), parquet, row_group_size=3)
        fingerprint = hashlib.sha256(b"test-normalization-policy").hexdigest()
        ready_path = directory / "BASE_CHUNK_READY.json"
        base_chunk = {
            "chunk_id": f"chunk-{object_id}", "path": str(parquet.relative_to(self.root)),
            "ready_receipt": str(ready_path.relative_to(self.root)), "records": len(rows),
            "byte_count": parquet.stat().st_size, "sha256": sha256_file(parquet),
        }
        ready = pipeline._seal(ready_path, {
            "schema": "metis17.base-chunk/v1", "status": "NORMALIZED_CHUNK_READY",
            "source_id": source_id, "object_id": object_id, "normalization_fingerprint": fingerprint,
            "chunk_id": base_chunk["chunk_id"], "chunk_index": 0, "chunk": base_chunk,
        })
        completion = directory / "OBJECT_COMPLETE.json"
        complete = pipeline._seal(completion, {
            "schema": "metis17.normalized-object/v1", "object_id": object_id,
            "object_complete": True, "complete": True, "records": len(rows),
            "source_id": source_id, "normalization_fingerprint": fingerprint,
            "status": "NORMALIZED", "reblock_complete": True,
            "chunks": [base_chunk], "chunk_receipts": [base_chunk["ready_receipt"]],
        })
        body = {
            "schema": schema, "status": "ELIGIBLE", "eligible": True, "training_ready": True,
            "generation": self.generation,
            "object_id": object_id, "source_id": source_id, "pending_reasons": [],
            "normalization_fingerprint": fingerprint, "chunk_id": base_chunk["chunk_id"], "chunk_index": 0,
            "base_chunk": base_chunk["path"], "base_chunk_receipt": base_chunk["ready_receipt"],
            "inputs": {"base_chunk_receipt_sha256": ready["receipt_sha256"]},
            "object_complete": True,
            "object_completion": {
                "path": str(completion.relative_to(self.root)), "receipt_sha256": complete["receipt_sha256"],
            },
            "eligible_documents": len(rows),
            "chunks": [{
                "path": str(parquet.relative_to(self.root)), "sha256": sha256_file(parquet),
                "byte_count": parquet.stat().st_size, "records": len(rows),
            }],
        }
        stage_hash = digest_json(body)
        receipt_path = self.root / "accepted-receipts" / f"{stage_hash}.json"
        pipeline._seal(receipt_path, body)
        event = {
            "object_id": object_id, "source_id": source_id,
            "receipt_path": str(receipt_path.relative_to(self.root)), "stage_receipt_sha256": stage_hash,
            "generation": self.generation, "chunk_id": f"chunk-{object_id}",
            "input_documents": len(rows), "eligible_documents": len(rows),
        }
        if append:
            self.append(event, host)
        self.sources.append(parquet)
        return event, parquet, body

    def full_population(self) -> None:
        for index, category in enumerate(pipeline.REQUIRED_CATEGORIES):
            self.source(category, host="host-a" if index % 2 else "host-b")

    def assert_sealed(self, value: dict) -> None:
        self.assertEqual(value["receipt_sha256"], digest_json({
            key: item for key, item in value.items() if key != "receipt_sha256"
        }))

    def test_automatic_wait_ready_train_and_test_artifact_separation(self) -> None:
        waiting = self.step()
        self.assertEqual(waiting["status"], "WAITING")
        self.assertFalse(self.requests)
        self.full_population()
        ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        self.assertGreaterEqual(ready["sample"]["selected_bytes"], 3000)
        self.assertLessEqual(ready["sample"]["overshoot_bytes"], 5 * 1024)
        sample = read_receipt(self.work / ready["sample"]["path"] / "SAMPLE_RECEIPT.json")
        for category in pipeline.REQUIRED_CATEGORIES:
            self.assertGreaterEqual(sample["coverage"][category]["selected_bytes"], 100)
        self.assertFalse((self.work / "artifact").exists())
        trained = self.step()
        self.assertEqual(trained["status"], "TRAINED", trained.get("error"))
        self.assert_sealed(trained)
        self.assertFalse(trained["production"])
        self.assertEqual(trained["id_cache"]["status"], "READY_FOR_PARTITIONS")
        artifact = self.work / "artifact"
        release = read_receipt(artifact / tk.TOKENIZER_RELEASE)
        self.assertEqual(release["special_tokens"], {token: index for index, token in enumerate(SPECIALS)})
        self.assertEqual(release["training"]["utf8_bytes"], ready["sample"]["selected_bytes"])
        with self.assertRaisesRegex(ValueError, "test tokenizer"):
            tk.load_tokenizer17(artifact)
        self.assertFalse((self.root / "tokenizer" / "artifact").exists())
        self.assertEqual([request["kind"] for request in self.requests], ["tokenizer_sample", "tokenizer_model"])
        self.assertEqual(len(self.released), 2)
        with mock.patch.object(tk, "_validate_sample", side_effect=AssertionError("trained polls cannot rescan sample metadata")):
            with mock.patch.object(tk, "train_tokenizer17", side_effect=AssertionError("must not retrain")):
                self.assertEqual(self.step()["status"], "TRAINED")

    def test_production_gate_requires_full_target_not_only_five_minimums(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["tokenizer"].update({
            "production": True, "vocabulary_size": 131_072,
            "sample_target_bytes": 150_000_000_000, "minimum_category_bytes": 1_000_000_000,
        })
        resolved = pipeline._config(config, False)
        sample = {
            "ready": True, "target_met": True, "target_bytes": 150_000_000_000,
            "selected_bytes": 5_000_000_000,
            "coverage": {category: {"selected_bytes": 1_000_000_000} for category in pipeline.REQUIRED_CATEGORIES},
        }
        self.assertFalse(pipeline._sample_gate(sample, resolved))
        sample["selected_bytes"] = 149_999_999_999
        self.assertFalse(pipeline._sample_gate(sample, resolved))
        sample["selected_bytes"] = 150_000_000_000
        self.assertTrue(pipeline._sample_gate(sample, resolved))
        sample["selected_bytes"] += resolved.overshoot_limit
        self.assertTrue(pipeline._sample_gate(sample, resolved))
        sample["selected_bytes"] += 1
        self.assertFalse(pipeline._sample_gate(sample, resolved))
        sample["selected_bytes"] = 150_000_000_000
        sample.update({"training_bytes": 149_999_999_999, "heldout_bytes": 1})
        self.assertFalse(pipeline._sample_gate(sample, resolved))
        sample.update({"training_bytes": 150_000_000_000, "heldout_bytes": 0})
        sample["coverage"]["math"]["selected_bytes"] = 999_999_999
        self.assertFalse(pipeline._sample_gate(sample, resolved))

    def test_production_requirements_cannot_be_weakened_by_run_or_test_defaults(self) -> None:
        for patch in (
            {"production": True, "sample_target_bytes": 3000, "vocabulary_size": 131_072},
            {"production": True, "sample_target_bytes": 150_000_000_000, "vocabulary_size": 600},
            {"production": True, "sample_target_bytes": 150_000_000_000, "vocabulary_size": 131_072, "minimum_category_bytes": 1},
            {"production": True, "sample_target_bytes": 160_000_000_000, "vocabulary_size": 131_072,
             "minimum_category_bytes": 1_000_000_000},
            {"production": False},
        ):
            with self.subTest(patch=patch):
                config = json.loads(json.dumps(self.config))
                config["tokenizer"].update(patch)
                with self.assertRaises(ValueError):
                    pipeline._config(config, False)
        with self.assertRaises(ValueError):
            pipeline._config({**self.config, "tokenizer": {**self.config["tokenizer"], "production": True}}, True)
        for patch in ({"split_digits": False}, {"dtype": "uint16"}, {"special_tokens": SPECIALS[:6]}, {"special_tokens": ["<|x1|>"] + SPECIALS[1:]}):
            config = {**self.config, "tokenizer": {**self.config["tokenizer"], **patch}}
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                pipeline._config(config, True)

    def test_recipe_freezes_150gb_contract_without_rewriting_legacy_run(self) -> None:
        self.config["tokenizer"].update({
            "production": True, "vocabulary_size": 131_072, "sample_target_bytes": 160_000_000_000,
            "minimum_category_bytes": 30_000_000_000,
        })
        self.write_run()
        original_run = (self.root / "RUN.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "explicit 150GB tokenizer recipe"):
            pipeline._config(self.config, False)
        blocked = pipeline.run_tokenizer_step(
            self.root, scratch_dir=self.scratch, generation=self.generation, reserve_output=self.reserve,
        )
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertFalse(blocked["recipe_bound"])
        settings = {**self.config["tokenizer"], "sample_target_bytes": 150_000_000_000}
        recipe = pipeline.freeze_tokenizer_recipe(self.root, settings)
        self.assert_sealed(recipe)
        self.assertEqual(recipe["schema"], "metis17.tokenizer-recipe/v1")
        self.assertEqual(recipe["run_sha256"], hashlib.sha256(original_run).hexdigest())
        self.assertEqual(pipeline.freeze_tokenizer_recipe(self.root, settings), recipe)
        self.assertEqual((self.root / "RUN.json").read_bytes(), original_run)
        with self.assertRaisesRegex(ValueError, "Immutable tokenizer recipe changed"):
            pipeline.freeze_tokenizer_recipe(self.root, {**settings, "sample_seed": "changed"})
        with self.assertRaises(ValueError):
            pipeline.freeze_tokenizer_recipe(self.root, {**settings, "vocabulary_size": 600})
        waiting = pipeline.run_tokenizer_step(
            self.root, scratch_dir=self.scratch, generation=self.generation, reserve_output=self.reserve,
        )
        self.assertEqual(waiting["status"], "WAITING", waiting.get("error"))
        self.assertEqual(waiting["recipe_sha256"], recipe["receipt_sha256"])
        self.assertEqual(waiting["target_bytes"], 150_000_000_000)
        self.assertEqual((self.root / "RUN.json").read_bytes(), original_run)
        (self.root / "RUN.json").write_bytes(original_run + b"\n")
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            pipeline.freeze_tokenizer_recipe(self.root, settings)

    def test_recipe_is_bound_to_sample_training_and_both_public_entrypoints(self) -> None:
        recipe = pipeline.freeze_tokenizer_recipe(self.root, self.config["tokenizer"])
        self.full_population()
        ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        sample = read_receipt(self.work / ready["sample"]["path"] / "SAMPLE_RECEIPT.json")
        self.assertEqual(sample["identity"]["recipe_sha256"], recipe["receipt_sha256"])
        trained = self.step()
        self.assertEqual(trained["status"], "TRAINED", trained.get("error"))
        provenance = read_receipt(self.work / "TRAINING_PROVENANCE.json")
        self.assertEqual(provenance["recipe_sha256"], recipe["receipt_sha256"])
        self.assertEqual(provenance["training_bytes"], sample["selected_bytes"])
        self.assertEqual(provenance["heldout_bytes"], 0)
        path = self.root / "tokenizer" / "RECIPE.json"
        changed = read_receipt(path)
        changed["tokenizer"]["sample_seed"] = "another sample"
        pipeline._seal(path, changed)
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            pipeline.tokenize_ready_partition(
                self.root, [self.sources[0]], scratch_dir=self.scratch, partition_id="changed-recipe",
                generation=self.generation, reserve_output=self.reserve, test_mode=True,
            )
        blocked = self.step()
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn("recipe changed", blocked["error"]["message"])

    def test_missing_recipe_or_changed_recipe_seal_is_rejected(self) -> None:
        recipe = pipeline.freeze_tokenizer_recipe(self.root, self.config["tokenizer"])
        self.full_population()
        self.step()
        self.assertEqual(self.step()["status"], "TRAINED")
        path = self.root / "tokenizer" / "RECIPE.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "recipe changed or disappeared"):
            pipeline.tokenize_ready_partition(
                self.root, [self.sources[0]], scratch_dir=self.scratch, partition_id="missing-recipe",
                generation=self.generation, reserve_output=self.reserve, test_mode=True,
            )
        self.assertIn("recipe changed or disappeared", self.step()["error"]["message"])
        path.write_bytes(original)
        self.assertEqual(self.step()["status"], "TRAINED")
        recipe["tokenizer"]["sample_seed"] = "unsealed mutation"
        path.write_text(json.dumps(recipe))
        with self.assertRaises(ValueError):
            pipeline.tokenize_ready_partition(
                self.root, [self.sources[0]], scratch_dir=self.scratch, partition_id="unsealed-recipe",
                generation=self.generation, reserve_output=self.reserve, test_mode=True,
            )
        self.assertEqual(self.step()["status"], "BLOCKED")

    def test_required_sources_and_languages_wait_for_validated_usable_text(self) -> None:
        settings = {
            **self.config["tokenizer"], "required_source_minimum_bytes": {"delayed-native-source": 1500},
            "required_language_minimum_bytes": {"de": 1500},
        }
        recipe = pipeline.freeze_tokenizer_recipe(self.root, settings)
        self.full_population()
        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("required source is absent")):
            waiting = self.step()
        self.assertEqual(waiting["status"], "WAITING", waiting.get("error"))
        self.assertEqual(waiting["sample_attempts"], 0)
        self.assertFalse((self.work / "artifact").exists())
        self.source(
            "multilingual", ["Ausschließlich ein zu langes Dokument. " * 100],
            source_id="delayed-native-source", language="de", suffix="oversized-native",
        )
        waiting = self.step()
        self.assertEqual(waiting["status"], "WAITING", waiting.get("error"))
        self.assertEqual(waiting["last_sample_attempt"]["missing_sources"], ["delayed-native-source"])
        self.assertEqual(waiting["last_sample_attempt"]["missing_languages"], ["de"])
        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("unchanged inventory must not rescan")):
            self.assertEqual(self.step()["status"], "WAITING")
        self.source(
            "multilingual", [f"Deutscher Originaltext {index} über Wissenschaft und Technik. " * 10 for index in range(3)],
            source_id="delayed-native-source", language="de", suffix="usable-native",
        )
        ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        sample = read_receipt(self.work / ready["sample"]["path"] / "SAMPLE_RECEIPT.json")
        self.assertEqual(sample["identity"]["recipe_sha256"], recipe["receipt_sha256"])
        self.assertGreaterEqual(sample["source_coverage"]["delayed-native-source"]["selected_bytes"], 1500)
        self.assertGreaterEqual(sample["language_coverage"]["de"]["selected_bytes"], 1500)
        selected = pq.read_table(self.work / ready["sample"]["path"] / "samples.parquet").to_pylist()
        actual = {"source": 0, "language": 0}
        for item in selected:
            row = pq.read_table(item["source_shard"]).to_pylist()[item["source_row"]]
            if row["source_id"] == "delayed-native-source":
                actual["source"] += len(row["text"].encode())
            if row["language"] == "de":
                actual["language"] += len(row["text"].encode())
        self.assertEqual(actual["source"], sample["source_coverage"]["delayed-native-source"]["selected_bytes"])
        self.assertEqual(actual["language"], sample["language_coverage"]["de"]["selected_bytes"])
        self.assertEqual({item["path"] for item in sample["identity"]["inputs"]}, {str(path) for path in self.sources})

    def test_required_coverage_settings_are_bounded_and_explicit(self) -> None:
        for key in ("required_source_minimum_bytes", "required_language_minimum_bytes"):
            for value in ([], {"": 1}, {"required": True}, {"required": 0}, {"required": 3001}):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    pipeline.freeze_tokenizer_recipe(self.root, {**self.config["tokenizer"], key: value})
        self.assertFalse((self.root / "tokenizer" / "RECIPE.json").exists())

    def test_required_language_gate_is_independent_of_source_coverage(self) -> None:
        pipeline.freeze_tokenizer_recipe(self.root, {
            **self.config["tokenizer"], "required_source_minimum_bytes": {"web-publisher": 100},
            "required_language_minimum_bytes": {"de": 100},
        })
        self.full_population()
        waiting = self.step()
        self.assertEqual(waiting["status"], "WAITING", waiting.get("error"))
        self.assertGreater(waiting["admitted_source_characters"]["web-publisher"], 100)
        self.assertEqual(waiting["admitted_language_characters"]["de"], 0)
        self.source("multilingual", ["Deutscher Originaltext. " * 10], suffix="native-german", language="de")
        ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        sample = read_receipt(self.work / ready["sample"]["path"] / "SAMPLE_RECEIPT.json")
        self.assertGreaterEqual(sample["language_coverage"]["de"]["selected_bytes"], 100)

    def test_minimums_only_do_not_train_and_unchanged_polls_do_not_rescan(self) -> None:
        self.config["tokenizer"].update({"sample_target_bytes": 1500, "minimum_category_bytes": 100})
        self.write_run()
        for category in pipeline.REQUIRED_CATEGORIES:
            self.source(category, [(category + " " + "x" * 120)])
        with mock.patch.object(tk, "train_tokenizer17", side_effect=AssertionError("small minimums cannot trigger training")):
            result = self.step()
        self.assertEqual(result["status"], "WAITING", result.get("error"))
        self.assertEqual(result["sample_attempts"], 1)
        self.assertEqual(result["last_sample_attempt"]["missing_categories"], [])
        self.assertLess(result["last_sample_attempt"]["available_unique_bytes"], 1500)
        original_sha = pipeline.sha256_file

        def no_parquet_hash(path):
            self.assertNotEqual(Path(path).suffix, ".parquet")
            return original_sha(path)

        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("no new sampling attempt")):
            with mock.patch.object(pipeline, "_chunk_inventory", side_effect=AssertionError("no corpus metadata rescan")):
                with mock.patch.object(pipeline, "sha256_file", side_effect=no_parquet_hash):
                    again = self.step()
                    unchanged = self.step()
        self.assertEqual(again["sample_attempts"], 1)
        self.assertEqual(again, unchanged)

    def test_measured_growth_retries_at_target_without_waiting_for_a_blind_doubling(self) -> None:
        self.config["tokenizer"].update({"sample_target_bytes": 1500, "minimum_category_bytes": 100})
        self.write_run()
        for category in pipeline.REQUIRED_CATEGORIES:
            self.source(category, [category + " " + "x" * 220])
        first = self.step()
        self.assertEqual(first["status"], "WAITING")
        previous = sum(first["admitted_characters"].values())
        self.assertLess(first["last_sample_attempt"]["next_total_characters"], previous * 2)
        self.source("web", ["unique-extra " + "y" * 410], suffix="extra")
        second = self.step()
        self.assertEqual(second["status"], "SAMPLE_READY", second.get("error"))
        self.assertEqual(second["sample_attempts"], 2)

    def test_polling_checkpoints_partial_lines_and_fair_host_progress(self) -> None:
        self.config["tokenizer"]["max_events_per_step"] = 1
        self.write_run()
        first, _path, _receipt = self.source("web", ["short"], append=False)
        second, _path, _receipt = self.source("math", ["short"], append=False)
        third, _path, _receipt = self.source("code", ["short"], append=False)
        a = self.append(first, "a", newline=False)
        self.assertEqual(self.step()["admitted_rows"], 0)
        with a.open("ab") as stream:
            stream.write(b"\n")
        self.append(third, "a")
        self.append(second, "b")
        one, two, three = self.step(), self.step(), self.step()
        self.assertEqual([one["admitted_rows"], two["admitted_rows"], three["admitted_rows"]], [1, 2, 3])
        self.assertIn(second["stage_receipt_sha256"], two["stage_receipts"])
        self.assertEqual(three["event_cursors"]["events/eligible/a.jsonl"]["offset"], a.stat().st_size)
        a.write_bytes(b"")
        self.assertEqual(self.step()["status"], "BLOCKED")

    def test_duplicate_events_and_aliases_do_not_double_count_or_rescan(self) -> None:
        event, _path, _body = self.source("web", ["small sample"])
        first = self.step()
        alias = self.root / "accepted-receipts" / "alias.json"
        shutil.copyfile(self.root / event["receipt_path"], alias)
        self.append({**event, "receipt_path": str(alias.relative_to(self.root))}, "other-host")
        with mock.patch.object(pipeline, "_chunk_inventory", side_effect=AssertionError("duplicate admission must not rescan")):
            second = self.step()
        self.assertEqual(first["admitted_rows"], second["admitted_rows"])
        self.assertEqual(len(second["chunks"]), 1)

    def test_only_canonical_stage_hash_and_complete_eligible_receipts_are_admitted(self) -> None:
        event, _path, body = self.source("web", ["small"], append=False)
        self.append({**event, "stage_receipt_sha256": sha256_file(self.root / event["receipt_path"])})
        result = self.step()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("canonical stage seal", result["error"]["message"])
        self.assertEqual(result["admitted_rows"], 0)
        for status in ("NORMALIZED_PENDING_DECONTAMINATION", "ELIGIBLE_PENDING_OBJECT_COMPLETION", "FILTERED"):
            changed = {**body, "status": status, "chunks": [], "screened_chunks": body["chunks"]}
            path = self.root / "accepted-receipts" / f"{status}.json"
            receipt = pipeline._seal(path, changed)
            changed_event = {**event, "receipt_path": str(path.relative_to(self.root)), "stage_receipt_sha256": receipt["receipt_sha256"]}
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "Pending"):
                pipeline._stage_event(self.root, changed_event, True)
        changed = {**body, "object_complete": False}
        path = self.root / "accepted-receipts" / "incomplete.json"
        receipt = pipeline._seal(path, changed)
        with self.assertRaisesRegex(ValueError, "object_complete"):
            pipeline._stage_event(self.root, {**event, "receipt_path": str(path.relative_to(self.root)), "stage_receipt_sha256": receipt["receipt_sha256"]}, True)

    def test_production_object_completion_proof_is_hash_and_identity_bound(self) -> None:
        event, _path, body = self.source("web", ["small"], append=False)
        pipeline._stage_event(self.root, event, True)
        proof_path = self.root / body["object_completion"]["path"]
        proof_path.write_bytes(proof_path.read_bytes() + b" ")
        pipeline._stage_event(self.root, event, True)
        changed = read_receipt(proof_path)
        changed["object_id"] = "another-object"
        pipeline._seal(proof_path, changed)
        with self.assertRaisesRegex(ValueError, "seal mismatch"):
            pipeline._stage_event(self.root, event, True)
        incomplete = {
            "schema": "metis17.normalized-chunk/v1", "object_id": event["object_id"],
            "status": "NORMALIZED_CHUNK_READY",
        }
        pending_proof = pipeline._seal(proof_path, incomplete)
        body["object_completion"]["receipt_sha256"] = pending_proof["receipt_sha256"]
        changed = pipeline._seal(self.root / event["receipt_path"], body)
        with self.assertRaisesRegex(ValueError, "positively prove EOF"):
            pipeline._stage_event(self.root, {**event, "stage_receipt_sha256": changed["receipt_sha256"]}, True)

    def test_reservations_are_required_and_working_bound_cannot_expand(self) -> None:
        self.full_population()
        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("must reserve before allocation")):
            missing = self.step(reserve_output=None)
        self.assertEqual(missing["status"], "BLOCKED")
        self.assertIn("reservation", missing["error"]["message"])

        @contextlib.contextmanager
        def over_budget(request):
            yield {
                "reserved_bytes": request["requested_bytes"],
                "used_working_bytes": request["max_working_bytes"], "other_reserved_bytes": 0,
            }

        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("cannot allocate over cap")):
            rejected = self.step(reserve_output=over_budget)
        self.assertEqual(rejected["status"], "BLOCKED")
        self.assertIn("max_working_bytes", rejected["error"]["message"])
        self.assertFalse((self.work / "samples").exists())
        config = json.loads(json.dumps(self.config))
        config["limits"]["max_working_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "may not expand"):
            pipeline._config(config, True)

    def test_full_raw_reservation_is_inside_total_cap_not_an_extra_allowance(self) -> None:
        config = pipeline._config(self.config, True)
        state = pipeline._initial("0" * 64, False, self.generation)
        request = pipeline._reservation_request(self.work, state, config, "tokenizer_sample", "test", 1024)
        self.assertEqual(request["raw_reservation_bytes"], 400_000_000_000)

        @contextlib.contextmanager
        def exact_fit(_request):
            yield {
                "reserved_bytes": 1024, "other_reserved_bytes": 0,
                "used_working_bytes": 2_000_000_000_000 - 400_000_000_000 - 1024,
            }

        with pipeline._reserved(request, exact_fit) as grant:
            self.assertEqual(grant["raw_reservation_bytes"], 400_000_000_000)

        @contextlib.contextmanager
        def one_byte_over(_request):
            yield {
                "reserved_bytes": 1024, "other_reserved_bytes": 0,
                "used_working_bytes": 2_000_000_000_000 - 400_000_000_000 - 1023,
            }

        with self.assertRaisesRegex(ValueError, "full raw reservation"):
            with pipeline._reserved(request, one_byte_over):
                self.fail("Reservation must not grant an extra 400GB")

    def test_confirmed_capacity_expansion_preserves_the_full_raw_reservation(self) -> None:
        limits = {
            "capacity_confirmation": "unlimited", "max_raw_bytes": 200_000_000_000_000,
            "max_working_bytes": 1_000_000_000_000_000,
        }
        config = pipeline._config(self.config, True, live_limits=limits)
        request = pipeline._reservation_request(
            self.work, pipeline._initial("0" * 64, False, self.generation),
            config, "tokenizer_sample", "expanded", 1024,
        )
        self.assertEqual(request["raw_reservation_bytes"], limits["max_raw_bytes"])
        self.assertEqual(request["max_total_bytes"], limits["max_working_bytes"])

    def test_frozen_run_changes_are_blocked(self) -> None:
        self.step()
        self.config["tokenizer"]["sample_seed"] = "a changed recipe"
        self.write_run()
        result = self.step()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("Frozen RUN.json changed", result["error"]["message"])
        self.assertFalse((self.work / "artifact").exists())

    def test_sample_publication_and_model_publication_resume_without_repeating_work(self) -> None:
        self.full_population()
        with mock.patch.object(pipeline, "_sample_state", side_effect=RuntimeError("interrupted after sample publication")):
            interrupted = self.step()
        self.assertEqual(interrupted["status"], "BLOCKED")
        self.source("web", ["Arrived after the closed sample was published."], suffix="after-sample-publication")
        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("committed sample must be reused")):
            ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        sample = read_receipt(self.work / ready["sample"]["path"] / "SAMPLE_RECEIPT.json")
        self.assertEqual(len(sample["identity"]["inputs"]), 5)
        publish = pipeline._publish_model

        def interrupted_model(*args, **kwargs):
            publish(*args, **kwargs)
            raise RuntimeError("interrupted after model publication")

        with mock.patch.object(pipeline, "_publish_model", side_effect=interrupted_model):
            failed = self.step()
        self.assertEqual(failed["status"], "BLOCKED")
        with mock.patch.object(tk, "train_tokenizer17", side_effect=AssertionError("published artifact must not retrain")):
            trained = self.step()
        self.assertEqual(trained["status"], "TRAINED", trained.get("error"))

    def test_sample_failures_do_not_rescan_same_inventory_every_poll(self) -> None:
        self.full_population()
        with mock.patch.object(pipeline, "_build_sample", side_effect=ValueError("bounded local database is full")) as build:
            first = self.step()
            second = self.step()
        self.assertEqual(first["status"], "BLOCKED")
        self.assertEqual(second["status"], "BLOCKED")
        self.assertEqual(build.call_count, 1)

    def test_failed_training_never_publishes_success_or_retries_identical_sample(self) -> None:
        self.full_population()
        self.assertEqual(self.step()["status"], "SAMPLE_READY")
        with mock.patch.object(tk, "train_tokenizer17", side_effect=ValueError("vocabulary shortfall")) as train:
            failed = self.step()
            repeated = self.step()
        self.assertEqual(failed["status"], "BLOCKED")
        self.assertEqual(repeated["status"], "BLOCKED")
        self.assertEqual(train.call_count, 1)
        self.assertFalse((self.work / "artifact").exists())

    def test_bounded_sampler_uses_only_local_sqlite_and_reports_exact_overshoot(self) -> None:
        self.full_population()
        connect = pipeline.sqlite3.connect

        def local_connect(path, *args, **kwargs):
            self.assertTrue(Path(path).resolve().is_relative_to(self.scratch))
            self.assertFalse(Path(path).resolve().is_relative_to(self.root))
            return connect(path, *args, **kwargs)

        with mock.patch.object(pipeline.sqlite3, "connect", side_effect=local_connect):
            ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        self.assertFalse(list(self.root.rglob("*.sqlite*")))
        self.assertFalse(list(self.scratch.rglob("*.sqlite*")))
        receipt = read_receipt(self.work / ready["sample"]["path"] / "SAMPLE_RECEIPT.json")
        metadata = pq.read_table(self.work / ready["sample"]["path"] / "samples.parquet").to_pylist()
        self.assertEqual(sum(row["utf8_bytes"] for row in metadata), receipt["selected_bytes"])
        self.assertEqual(receipt["overshoot_bytes"], receipt["selected_bytes"] - 3000)
        self.assertEqual(len({(row["source_shard"], row["source_row"]) for row in metadata}), len(metadata))
        texts = list(tk.iter_tokenizer_sample17(self.work / ready["sample"]["path"], production=False))
        self.assertEqual(sum(len(text.encode()) for text in texts), receipt["selected_bytes"])

    def test_selection_matches_independent_minimum_then_balanced_hash_reference(self) -> None:
        self.full_population()
        self.source("math", ["too long " + "x" * 2048], suffix="oversized")
        ready = self.step()
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        pools = {category: {} for category in pipeline.REQUIRED_CATEGORIES}
        for path in sorted(self.sources):
            for index, row in enumerate(pq.read_table(path).to_pylist()):
                size = len(row["text"].encode("utf-8"))
                if 0 < size <= 1024:
                    pools[row["category"]].setdefault(row["content_hash"], (path, index, size))
        ordered = {
            category: sorted(
                entries.items(),
                key=lambda item: digest_json({
                    "seed": "metis1.7-tokenizer-v1", "stratum": category, "content_hash": item[0],
                }),
            )
            for category, entries in pools.items()
        }
        positions = {category: 0 for category in pools}
        byte_totals = {category: 0 for category in pools}
        chosen = []

        def take(category):
            _hash, (path, index, size) = ordered[category][positions[category]]
            positions[category] += 1
            byte_totals[category] += size
            chosen.append((str(path), index))

        for category in pipeline.REQUIRED_CATEGORIES:
            while byte_totals[category] < 100:
                take(category)
        while sum(byte_totals.values()) < 3000:
            category = min(
                (name for name in pools if positions[name] < len(ordered[name])),
                key=lambda name: (byte_totals[name], name),
            )
            take(category)
        rows = pq.read_table(self.work / ready["sample"]["path"] / "samples.parquet").to_pylist()
        self.assertEqual([(row["source_shard"], row["source_row"]) for row in rows], sorted(chosen))
        self.assertEqual(sum(byte_totals.values()), ready["sample"]["selected_bytes"])

    def test_sampler_byte_cap_and_explicit_event_batch_bound_fail_closed(self) -> None:
        target = self.workspace / "bounded.bin"
        sink = pipeline._BoundedFile(target, 4)
        sink.write(b"1234")
        with self.assertRaisesRegex(ValueError, "byte limit"):
            sink.write(b"5")
        sink.close()
        self.assertEqual(target.read_bytes(), b"1234")
        event, _path, _body = self.source("web", ["small"], append=False)
        result = self.step(eligible_events=[event] * 17)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["admitted_rows"], 0)

    def test_corrupt_model_and_unadmitted_token_partitions_are_rejected(self) -> None:
        self.full_population()
        self.step()
        trained = self.step()
        self.assertEqual(trained["status"], "TRAINED", trained.get("error"))
        limits = tk.TokenCacheLimits17(max_token_bytes=16_384, max_scratch_bytes=4 * 1024**2)
        first = pipeline.tokenize_ready_partition(
            self.root, [self.sources[0]], scratch_dir=self.scratch, partition_id="stable-web-000",
            generation=self.generation, reserve_output=self.reserve, limits=limits, test_mode=True,
        )
        with mock.patch.object(tk, "_encode_batch", side_effect=AssertionError("retained partition must not re-encode")):
            second = pipeline.tokenize_ready_partition(
                self.root, [self.sources[0]], scratch_dir=self.scratch, partition_id="stable-web-000",
                generation=self.generation, reserve_output=self.reserve, limits=limits, test_mode=True,
            )
        self.assertEqual(first, second)
        extra = self.root / "unadmitted.parquet"
        shutil.copyfile(self.sources[0], extra)
        with self.assertRaisesRegex(ValueError, "not admitted"):
            pipeline.tokenize_ready_partition(
                self.root, [extra], scratch_dir=self.scratch, partition_id="extra",
                generation=self.generation, reserve_output=self.reserve, limits=limits, test_mode=True,
            )
        artifact = self.work / "artifact" / "tokenizer.json"
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        self.assertEqual(self.step()["status"], "BLOCKED")

    def test_release_and_training_provenance_are_frozen_not_just_tokenizer_bytes(self) -> None:
        self.full_population()
        self.step()
        self.assertEqual(self.step()["status"], "TRAINED")
        release_path = self.work / "artifact" / tk.TOKENIZER_RELEASE
        release = read_receipt(release_path)
        release["created_at"] = "changed"
        pipeline._seal(release_path, release)
        result = self.step()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("provenance changed", result["error"]["message"])

    def test_corrupt_checkpoint_is_preserved_and_reported_in_a_sealed_error(self) -> None:
        self.step()
        state_path = self.work / "STATE.json"
        state_path.write_text('{"broken":')
        result = self.step()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(state_path.read_text(), '{"broken":')
        self.assertTrue((self.work / "BLOCKED.json").exists())
        self.assert_sealed(result)

    def test_status_without_state_waits_without_creating_files(self) -> None:
        before = {path.relative_to(self.root) for path in self.root.rglob("*")}
        for test_mode in (False, True):
            with self.subTest(test_mode=test_mode):
                status = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=test_mode)
                self.assertEqual(status["status"], "WAITING")
                self.assertFalse(status["ready"])
                self.assertEqual(status["generation"], self.generation)
                self.assertEqual(status["selected_utf8_bytes"], 0)
                self.assertIsNone(status["vocabulary_size"])
        self.assertEqual({path.relative_to(self.root) for path in self.root.rglob("*")}, before)
        self.assertFalse((self.root / "tokenizer").exists())

    def test_status_checks_actual_artifact_and_sample_bytes_without_writes_or_rescans(self) -> None:
        self.config["tokenizer"]["vocabulary_size"] = 4096
        self.write_run()
        recipe = pipeline.freeze_tokenizer_recipe(self.root, self.config["tokenizer"])
        self.step()
        waiting = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=True)
        self.assertEqual(waiting["status"], "WAITING")
        self.assertFalse(waiting["ready"])
        self.assertEqual(waiting["selected_utf8_bytes"], 0)
        self.assertIsNone(waiting["vocabulary_size"])
        self.full_population()
        sample_state = self.step()
        self.assertEqual(sample_state["status"], "SAMPLE_READY", sample_state.get("error"))
        sample_status = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=True)
        self.assertEqual(sample_status["status"], "SAMPLE_READY")
        self.assertFalse(sample_status["ready"])
        self.assertEqual(sample_status["selected_utf8_bytes"], sample_state["sample"]["selected_bytes"])
        self.assertIsNone(sample_status["vocabulary_size"])
        trained = self.step()
        self.assertEqual(trained["status"], "TRAINED", trained.get("error"))
        actual_vocab = tk.load_tokenizer17(self.work / "artifact", production=False).get_vocab_size(with_added_tokens=True)
        before = {
            path.relative_to(self.root): (sha256_file(path), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file()
        }
        with (
            mock.patch.object(tk, "load_tokenizer17", wraps=tk.load_tokenizer17) as load,
            mock.patch.object(tk, "_validate_sample", side_effect=AssertionError("status cannot rescan sample metadata")),
            mock.patch.object(pipeline, "_seal", side_effect=AssertionError("status must be read-only")),
        ):
            status = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=True)
        self.assertEqual(status["status"], "TRAINED", status.get("error"))
        self.assertTrue(status["ready"])
        self.assertEqual(status["selected_utf8_bytes"], sample_state["sample"]["selected_bytes"])
        self.assertEqual(status["overshoot_bytes"], status["selected_utf8_bytes"] - 3000)
        self.assertEqual(status["vocabulary_size"], actual_vocab)
        self.assertLess(actual_vocab, 4096)
        self.assertEqual(status["recipe_sha256"], recipe["receipt_sha256"])
        self.assertEqual(status["state_sha256"], trained["receipt_sha256"])
        self.assertEqual(status["sample_sha256"], sample_state["sample"]["receipt_sha256"])
        self.assertEqual(status["generation"], self.generation)
        for field in (
            "run_sha256", "generation_descriptor_sha256", "tokenizer_sha256",
            "tokenizer_release_sha256", "training_provenance_sha256",
        ):
            self.assertEqual(status[field], trained[field])
        load.assert_called_once_with(self.work / "artifact", production=False)
        self.assertEqual({
            path.relative_to(self.root): (sha256_file(path), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file()
        }, before)

    def test_status_and_partition_dispatch_reject_the_same_integrity_changes(self) -> None:
        pipeline.freeze_tokenizer_recipe(self.root, self.config["tokenizer"])
        self.full_population()
        self.step()
        trained = self.step()
        self.assertEqual(trained["status"], "TRAINED")
        cases = (
            (self.root / "RUN.json", None),
            (self.root / "tokenizer" / "RECIPE.json", {"tokenizer": {**self.config["tokenizer"], "sample_seed": "changed"}}),
            (self.work / "STATE.json", {"recipe_sha256": "f" * 64}),
            (self.work / "STATE.json", {"production": True}),
            (self.root / "preparation" / "generations" / f"{self.generation}.json", {"policies": ["changed"]}),
            (self.work / trained["sample"]["path"] / "SAMPLE_RECEIPT.json", {"selected_bytes": 1}),
            (self.work / "artifact" / "tokenizer.json", None),
            (self.work / "artifact" / tk.TOKENIZER_RELEASE, {"created_at": "changed"}),
            (self.work / "TRAINING_PROVENANCE.json", {"target_bytes": 1}),
        )
        for path, patch in cases:
            with self.subTest(path=path, patch=patch):
                original = path.read_bytes()
                mode = path.stat().st_mode
                if patch is None:
                    path.write_bytes(original + b"\n")
                else:
                    pipeline._seal(path, {**read_receipt(path), **patch})
                try:
                    before = {
                        item.relative_to(self.root): (sha256_file(item), item.stat().st_mtime_ns)
                        for item in self.root.rglob("*") if item.is_file()
                    }
                    status = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=True)
                    self.assertEqual(status["status"], "BLOCKED", status)
                    self.assertFalse(status["ready"])
                    self.assertTrue(status["error"]["message"])
                    with self.assertRaises(ValueError):
                        pipeline.tokenize_ready_partition(
                            self.root, [self.sources[0]], scratch_dir=self.scratch, partition_id="must-not-dispatch",
                            generation=self.generation, test_mode=True, reserve_output=self.reserve,
                        )
                    self.assertEqual({
                        item.relative_to(self.root): (sha256_file(item), item.stat().st_mtime_ns)
                        for item in self.root.rglob("*") if item.is_file()
                    }, before)
                finally:
                    path.write_bytes(original)
                    path.chmod(mode)
        state_path = self.work / "STATE.json"
        state_path.write_bytes(b'{"corrupt":')
        status = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=True)
        self.assertEqual(status["status"], "BLOCKED")
        self.assertFalse(status["ready"])
        self.assertEqual(state_path.read_bytes(), b'{"corrupt":')
        self.assertFalse((self.work / "BLOCKED.json").exists())
        self.assertFalse((self.work / "ids").exists())

    def test_status_preserves_blocked_error_instead_of_promoting_existing_model(self) -> None:
        self.full_population()
        self.step()
        self.assertEqual(self.step()["status"], "TRAINED")
        state_path = self.work / "STATE.json"
        error = {"type": "ValueError", "message": "A later eligibility receipt was invalid"}
        pipeline._seal(state_path, {**read_receipt(state_path), "status": "BLOCKED", "error": error})
        before = state_path.read_bytes()
        status = pipeline.tokenizer_status(self.root, generation=self.generation, test_mode=True)
        self.assertEqual(status["status"], "BLOCKED")
        self.assertFalse(status["ready"])
        self.assertEqual(status["error"], error)
        self.assertGreater(status["selected_utf8_bytes"], 0)
        self.assertIsNotNone(status["vocabulary_size"])
        self.assertEqual(state_path.read_bytes(), before)

    def test_new_eligible_events_after_training_do_not_change_the_frozen_tokenizer(self) -> None:
        self.full_population()
        self.step()
        trained = self.step()
        original = trained["tokenizer_sha256"]
        _event, path, _body = self.source(
            "science", ["newly admitted science document with numbers 2026"], suffix="new-after-freeze",
        )
        with mock.patch.object(pipeline, "_build_sample", side_effect=AssertionError("frozen model cannot resample")):
            with mock.patch.object(tk, "train_tokenizer17", side_effect=AssertionError("frozen model cannot retrain")):
                following = self.step()
        self.assertEqual(following["status"], "TRAINED", following.get("error"))
        self.assertEqual(following["tokenizer_sha256"], original)
        self.assertEqual(following["admitted_rows"], trained["admitted_rows"])
        self.assertEqual(following["chunks"], trained["chunks"])
        self.assertEqual(following["stage_receipts"], trained["stage_receipts"])
        self.assertIn(str(path.relative_to(self.root)), following["recent_partition_inputs"])
        partition = pipeline.tokenize_ready_partition(
            self.root, [path], scratch_dir=self.scratch, partition_id="latest-admitted-science",
            generation=self.generation, reserve_output=self.reserve, test_mode=True,
            limits=tk.TokenCacheLimits17(max_token_bytes=16_384, max_scratch_bytes=4 * 1024**2),
        )
        self.assertEqual(partition["tokenizer_sha256"], original)

    def test_post_freeze_batches_are_bounded_and_older_partitions_use_explicit_proofs(self) -> None:
        recipe = pipeline.freeze_tokenizer_recipe(self.root, self.config["tokenizer"])
        self.full_population()
        self.step()
        frozen = self.step()
        self.assertEqual(frozen["status"], "TRAINED", frozen.get("error"))
        events = []
        for index in range(20):
            event, path, _body = self.source(
                "math", [f"Additional mathematics input {index}: 123.45"], suffix=f"after-freeze-{index}", append=False,
            )
            events.append((event, path))
            admitted = self.step(eligible_events=[event])
            self.assertEqual(admitted["status"], "TRAINED", admitted.get("error"))
            self.assertEqual(admitted["chunks"], frozen["chunks"])
            self.assertEqual(admitted["stage_receipts"], frozen["stage_receipts"])
            self.assertEqual(set(admitted["recent_partition_inputs"]), {str(path.relative_to(self.root))})
            self.assertLess(len(json.dumps(admitted)), len(json.dumps(frozen)) + 2048)
        event, path = events[0]
        before = (self.work / "STATE.json").read_bytes()
        options = {
            "scratch_dir": self.scratch, "partition_id": "old-explicit-proof", "generation": self.generation,
            "reserve_output": self.reserve, "test_mode": True,
            "limits": tk.TokenCacheLimits17(max_token_bytes=16_384, max_scratch_bytes=4 * 1024**2),
        }
        with self.assertRaisesRegex(ValueError, "not admitted"):
            pipeline.tokenize_ready_partition(self.root, [path], **options)
        options.update({
            "stage_receipt_path": event["receipt_path"], "stage_receipt_sha256": event["stage_receipt_sha256"],
        })
        first = pipeline.tokenize_ready_partition(self.root, [path], **options)
        self.assertEqual(first["recipe_sha256"], recipe["receipt_sha256"])
        self.assertEqual(first["stage_receipt_sha256"], event["stage_receipt_sha256"])
        with mock.patch.object(tk, "_encode_batch", side_effect=AssertionError("retained IDs must be reused")):
            self.assertEqual(pipeline.tokenize_ready_partition(self.root, [path], **options), first)
        self.assertEqual((self.work / "STATE.json").read_bytes(), before)

    def test_direct_partition_proofs_require_exact_ready_complete_generation_inventory(self) -> None:
        self.full_population()
        self.step()
        self.assertEqual(self.step()["status"], "TRAINED")
        event, path, body = self.source("web", ["Later eligible input."], suffix="direct", append=False)
        options = {
            "scratch_dir": self.scratch, "partition_id": "direct-proofs", "generation": self.generation,
            "reserve_output": self.reserve, "test_mode": True, "stage_receipt_path": event["receipt_path"],
        }
        with self.assertRaisesRegex(ValueError, "both stage receipt"):
            pipeline.tokenize_ready_partition(self.root, [path], **options)
        options["stage_receipt_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            pipeline.tokenize_ready_partition(self.root, [path], **options)
        options["stage_receipt_sha256"] = event["stage_receipt_sha256"]
        with self.assertRaisesRegex(ValueError, "outside its explicit eligibility receipt"):
            pipeline.tokenize_ready_partition(self.root, [self.sources[0]], **options)
        for patch in (
            {"training_ready": False}, {"object_complete": False},
            {"generation": "f" * 64}, {"chunks": body["chunks"] * 2},
            {"normalization_fingerprint": "e" * 64}, {"chunk_index": 1},
            {"base_chunk_receipt": "unrelated-ready-receipt.json"},
            {"inputs": {"base_chunk_receipt_sha256": "e" * 64}},
        ):
            with self.subTest(patch=patch):
                changed = pipeline._seal(self.root / event["receipt_path"], {**body, **patch})
                with self.assertRaises(ValueError):
                    pipeline.tokenize_ready_partition(
                        self.root, [path], **{**options, "stage_receipt_sha256": changed["receipt_sha256"]},
                    )
        self.assertFalse((self.work / "ids").exists())

    def test_superseded_and_unscoped_events_are_skipped_without_opening_artifacts(self) -> None:
        old = hashlib.sha256(b"old-policy").hexdigest()
        self.append({"generation": old, "receipt_path": "../must-not-open", "stage_receipt_sha256": "invalid"})
        self.append({"receipt_path": "../also-must-not-open"})
        with mock.patch.object(pipeline, "_stage_event", side_effect=AssertionError("foreign generations cannot open receipts")):
            result = self.step()
        self.assertEqual(result["status"], "WAITING", result.get("error"))
        self.assertEqual(result["admitted_rows"], 0)
        self.assertEqual(result["ignored_other_generation_events"], 1)
        self.assertEqual(result["ignored_unscoped_events"], 1)
        self.assertEqual(len(result["stage_receipts"]), 0)

    def test_generation_namespaces_and_sample_provenance_never_mix(self) -> None:
        self.full_population()
        first = self.step()
        self.assertEqual(first["status"], "SAMPLE_READY", first.get("error"))
        new_generation = hashlib.sha256(b"replacement-policy").hexdigest()
        self.write_generation(new_generation)
        with mock.patch.object(pipeline, "_stage_event", side_effect=AssertionError("old receipts cannot enter the new generation")):
            second = self.step(generation=new_generation)
        self.assertEqual(second["status"], "WAITING")
        self.assertEqual(second["generation"], new_generation)
        self.assertEqual(second["admitted_rows"], 0)
        self.assertEqual(second["ignored_other_generation_events"], 5)
        sample = read_receipt(self.work / first["sample"]["path"] / "SAMPLE_RECEIPT.json")
        self.assertEqual(sample["identity"]["generation"], self.generation)
        self.assertEqual(sample["identity"]["generation_descriptor_sha256"], first["generation_descriptor_sha256"])
        original_state = read_receipt(self.work / "STATE.json")
        self.assertEqual(original_state["generation"], self.generation)
        self.assertEqual(original_state["status"], "SAMPLE_READY")

    def test_generation_descriptor_is_frozen_and_receipt_generation_must_agree(self) -> None:
        self.step()
        descriptor = self.root / "preparation" / "generations" / f"{self.generation}.json"
        payload = read_receipt(descriptor)
        payload["policies"] = [{"path": "changed-policy.json", "sha256": "0" * 64}]
        pipeline._seal(descriptor, payload)
        changed = self.step()
        self.assertEqual(changed["status"], "BLOCKED")
        self.assertIn("generation descriptor changed", changed["error"]["message"])
        event, _path, body = self.source("web", ["small"], append=False)
        body["generation"] = "f" * 64
        receipt = pipeline._seal(self.root / event["receipt_path"], body)
        with self.assertRaisesRegex(ValueError, "eligibility generation"):
            pipeline._stage_event(self.root, {**event, "stage_receipt_sha256": receipt["receipt_sha256"]}, False)

    def test_working_budget_streams_guard_sample_model_and_receipt_writes(self) -> None:
        self.full_population()
        opened = []
        namespaces = {}
        case = self

        class NoFilenoStream(io.RawIOBase):
            def __init__(self, path, mode):
                self.stream = path.open(mode)

            def writable(self):
                return True

            def write(self, data):
                return self.stream.write(data)

            def tell(self):
                return self.stream.tell()

            def flush(self):
                if not self.stream.closed:
                    self.stream.flush()

            def close(self):
                if not self.stream.closed:
                    self.stream.close()
                super().close()

        class Budget:
            def snapshot(self):
                return _budget_snapshot()

            @contextlib.contextmanager
            def quota(self, namespace, directory):
                directory = Path(directory).resolve()
                if namespace not in namespaces:
                    for other in namespaces.values():
                        case.assertFalse(directory.is_relative_to(other) or other.is_relative_to(directory))
                namespaces[namespace] = directory
                directory.mkdir(parents=True, exist_ok=True)

                def guarded_open(path, mode="wb"):
                    path = Path(path).resolve()
                    case.assertTrue(path.is_relative_to(directory))
                    opened.append(path)
                    return NoFilenoStream(path, mode)

                yield _quota(guarded_open)

        budget = Budget()
        ready = self.step(working_budget=budget)
        self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
        trained = self.step(working_budget=budget)
        self.assertEqual(trained["status"], "TRAINED", trained.get("error"))
        self.assertEqual(trained["reservation"]["provider"], "WorkingBudget")
        self.assertEqual(len(namespaces), 2)
        self.assertTrue(any(path.name == "samples.parquet" for path in opened))
        self.assertTrue(any("SAMPLE_RECEIPT.json" in path.name for path in opened))
        self.assertTrue(any(path.name == "tokenizer.json" for path in opened))
        self.assertTrue(any(path.name == tk.TOKENIZER_RELEASE for path in opened))
        self.assertFalse(self.requests)

    def test_working_budget_capacity_failure_is_not_reported_as_sample_ready(self) -> None:
        self.full_population()

        class CapacityPending(RuntimeError):
            pass

        class Denied(io.RawIOBase):
            def __init__(self, path):
                self.stream = Path(path).open("wb")

            def writable(self):
                return True

            def tell(self):
                return self.stream.tell()

            def write(self, data):
                raise CapacityPending("global raw+working quota exhausted before growth")

            def flush(self):
                if not self.stream.closed:
                    self.stream.flush()

            def close(self):
                self.stream.close()
                super().close()

        class Budget:
            def snapshot(self):
                return _budget_snapshot()

            @contextlib.contextmanager
            def quota(self, namespace, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                yield _quota(lambda path, mode="wb": Denied(path))

        result = self.step(working_budget=Budget())
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("quota exhausted", result["error"]["message"])
        self.assertFalse((self.work / "artifact" / tk.TOKENIZER_RELEASE).exists())
        self.assertFalse(list((self.work / "samples").glob("*/SAMPLE_RECEIPT.json")))

    def test_default_production_allocation_uses_working_budget_without_cli_import(self) -> None:
        config = pipeline._config(self.config, True)
        state = pipeline._initial("0" * 64, False, self.generation)
        request = pipeline._reservation_request(self.work, state, config, "tokenizer_sample", "default", 1024)
        request["production"] = True
        directory = self.work / "samples"
        calls = []

        class Budget:
            def __init__(self, root):
                calls.append(("root", Path(root)))

            def snapshot(self):
                return _budget_snapshot()

            @contextlib.contextmanager
            def quota(self, namespace, output):
                calls.append((namespace, output))
                Path(output).mkdir(parents=True, exist_ok=True)
                yield _quota(lambda path, mode="wb": Path(path).open(mode))

        with mock.patch.dict(sys.modules, {"metis_data17.storage": SimpleNamespace(WorkingBudget=Budget)}):
            with pipeline._allocation(self.root, directory, request, None, None) as (grant, opener):
                with opener(directory / "bounded-data", mode="wb") as stream:
                    stream.write(b"123")
        self.assertEqual(calls[0], ("root", self.root))
        self.assertEqual(grant["provider"], "WorkingBudget")
        self.assertEqual(grant["raw_reservation_bytes"], 400_000_000_000)

    def test_real_working_budget_accounts_sample_model_and_bulk_reserved_ids(self) -> None:
        from metis_data17.storage import WorkingBudget

        pipeline._seal(self.root / "limits.json", {
            **self.config["limits"], "capacity_confirmation": "pending",
        })
        self.full_population()
        disk = SimpleNamespace(total=4_000_000_000_000, used=0, free=4_000_000_000_000)
        with mock.patch("shutil.disk_usage", return_value=disk):
            budget = WorkingBudget(self.root)
            ready = self.step(working_budget=budget)
            self.assertEqual(ready["status"], "SAMPLE_READY", ready.get("error"))
            trained = self.step(working_budget=budget)
            self.assertEqual(trained["status"], "TRAINED", trained.get("error"))
            limits = tk.TokenCacheLimits17(max_token_bytes=16_384, max_scratch_bytes=4 * 1024**2)
            first = pipeline.tokenize_ready_partition(
                self.root, [self.sources[0]], scratch_dir=self.scratch, generation=self.generation,
                partition_id="actual-quota-partition", working_budget=budget, limits=limits, test_mode=True,
            )
            with mock.patch.object(tk, "_encode_batch", side_effect=AssertionError("retained IDs cannot re-encode")):
                second = pipeline.tokenize_ready_partition(
                    self.root, [self.sources[0]], scratch_dir=self.scratch, generation=self.generation,
                    partition_id="actual-quota-partition", working_budget=budget, limits=limits, test_mode=True,
                )
            self.assertEqual(first, second)
            self.assertEqual(first["tokenization"]["records"], 10)
            snapshot = budget.snapshot()
            self.assertEqual(snapshot["max_raw_bytes"], 400_000_000_000)
            self.assertEqual(snapshot["derived_limit_bytes"], 1_580_000_000_000)
            self.assertEqual(snapshot["outstanding_bytes"], 0)
            actual = sum(
                path.stat().st_size
                for directory in (self.work / "samples", self.work / "artifact", self.work / "ids")
                for path in directory.rglob("*") if path.is_file()
            )
            self.assertEqual(snapshot["committed_bytes"], actual)
        self.assertFalse(list(self.root.rglob("*.sqlite*")))

    def test_working_budget_must_not_omit_raw_or_add_metadata_outside_total(self) -> None:
        request = pipeline._reservation_request(
            self.work, pipeline._initial("0" * 64, False, self.generation),
            pipeline._config(self.config, True), "tokenizer_sample", "wrong-budget", 1024,
        )
        for changes in (
            {"max_raw_bytes": 0},
            {"max_working_bytes": 2_400_000_000_000},
            {"derived_limit_bytes": 2_000_000_000_000},
        ):
            budget = SimpleNamespace(
                snapshot=lambda: {**_budget_snapshot(), **changes},
                quota=mock.Mock(side_effect=AssertionError("invalid budget must fail before namespace admission")),
            )
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                with pipeline._allocation(self.root, self.work / "samples", request, None, budget):
                    self.fail("Quota must enforce the inclusive total")

    def test_bulk_id_reservation_precedes_the_opaque_token_engine(self) -> None:
        self.full_population()
        self.step()
        self.assertEqual(self.step()["status"], "TRAINED")
        reserved = []

        class Budget:
            def snapshot(self):
                return _budget_snapshot()

            @contextlib.contextmanager
            def quota(self, namespace, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)
                quota = _quota(lambda path, mode="wb": Path(path).open(mode))
                quota.reserve = lambda amount: reserved.append(amount) or amount
                quota.reconcile = lambda: None
                yield quota

        enter = tk.TokenizationSession17.__enter__

        def guarded_enter(session):
            self.assertTrue(reserved, "Opaque engine must not allocate before the full bound is reserved")
            return enter(session)

        with mock.patch.object(tk.TokenizationSession17, "__enter__", guarded_enter):
            result = pipeline.tokenize_ready_partition(
                self.root, [self.sources[0]], scratch_dir=self.scratch, generation=self.generation,
                partition_id="bulk-order", working_budget=Budget(), test_mode=True,
                limits=tk.TokenCacheLimits17(max_token_bytes=16_384, max_scratch_bytes=4 * 1024**2),
            )
        self.assertEqual(result["tokenization"]["records"], 10)
        self.assertTrue((self.root / result["partition_root"] / "PARTITION_RECEIPT.json").exists())


if __name__ == "__main__":
    unittest.main()
