from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metis_training.distributed import ParallelTopology, Runtime
from metis_training.optimizers import OptimizerBundle
from metis_training.posttraining import (
    gspo_loss,
    gspo_token_loss,
    masked_causal_cross_entropy,
)
from metis_training.stage_backend import (
    BATCH_MIGRATION_SCHEMA,
    MMAP_BUNDLE_SCHEMA,
    MMapStageBundle,
    PostTrainingRequeue,
    SealedRequirement,
    StageBackendError,
    _align_supervised_labels,
    _apply_optimizer_state_transition,
    _context_gate_decision,
    _load_release_index,
    _load_canonical_lookup,
    _load_stage_batch_migration,
    _reconcile_active_policy_checkpoint_state,
    _resume_global_batch,
    _rlvr_rewards,
    _rlvr_prompt_fingerprints,
    _selected_token_log_probs_from_hidden,
    _validate_runtime_working_set,
    _write_stage_oom_request,
    _run_supervised_stage,
)


def _canonical_hash(value: object, *, omit: str) -> str:
    assert isinstance(value, dict)
    cooked = {key: item for key, item in value.items() if key != omit}
    return hashlib.sha256(
        json.dumps(
            cooked,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_lookup(vocabulary_size: int) -> np.ndarray:
    return np.arange(vocabulary_size, dtype="<u2")


def _canonical_ids_sha256(vocabulary_size: int) -> str:
    return hashlib.sha256(_canonical_lookup(vocabulary_size).tobytes()).hexdigest()


def _training(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "epochs": 1,
        "micro_batch_size": 2,
        "gradient_accumulation": 1,
        "shuffle_seed": 7,
        "checkpoint_interval_steps": 0,
        "learning_rate": 1.0e-2,
        "minimum_learning_rate_ratio": 0.1,
        "warmup_steps": 0,
        "gradient_clip": 1.0,
    }
    result.update(overrides)
    return result


def _sealed_bundle(
    root: Path,
    *,
    stage: str,
    arrays: dict[str, np.ndarray],
    sequence_length: int,
    vocabulary_size: int,
    tokenizer_sha256: str,
    parent_checkpoint_sha256: str,
    training: dict[str, object] | None = None,
    bundle_metadata: dict[str, object] | None = None,
) -> SealedRequirement:
    root.mkdir(parents=True)
    arrays = dict(arrays)
    if stage in {"context_extension", "cold_start_sft", "overall_sft"}:
        input_ids = arrays["input_ids"]
        records, sequence_length_from_array = input_ids.shape
        arrays.setdefault(
            "attention_mask",
            np.ones((records, sequence_length_from_array), dtype=np.bool_),
        )
        arrays.setdefault(
            "document_ids",
            np.repeat(
                np.arange(records, dtype=np.int32)[:, None],
                sequence_length_from_array,
                axis=1,
            ),
        )
        reset = np.zeros(
            (records, sequence_length_from_array), dtype=np.bool_
        )
        reset[:, 0] = True
        arrays.setdefault("reset_mask", reset)
        arrays.setdefault("canonical_ids", np.array(input_ids, copy=True))
    is_gspo = stage == "hybrid_mode_gspo" or stage.startswith("specialist_")
    if is_gspo:
        arrays.setdefault(
            "candidate_attention_mask",
            np.ones_like(arrays["candidate_input_ids"], dtype=np.bool_),
        )
        records = int(arrays["candidate_input_ids"].shape[0])
        if records % 3:
            raise AssertionError("GSPO test fixtures must contain full three-mode groups")
        modes = np.tile(np.arange(3, dtype=np.int8), records // 3)
        arrays["candidate_input_ids"] = np.array(
            arrays["candidate_input_ids"], copy=True
        )
        arrays["candidate_input_ids"][:, :, 0] = modes[:, None]
        arrays.setdefault("reasoning_mode", modes)
        arrays.setdefault(
            "mode_overlap_id",
            np.repeat(np.arange(records // 3, dtype=np.int32), 3),
        )
        base_fingerprints = np.zeros((records, 32), dtype=np.uint8)
        for group_index in range(records // 3):
            base_fingerprints[group_index * 3 : group_index * 3 + 3, 0] = (
                group_index + 1
            )
        arrays.setdefault("base_prompt_fingerprint", base_fingerprints)
        arrays.setdefault(
            "mode_compliance",
            np.ones_like(arrays["correctness"], dtype=np.float32),
        )
        arrays.setdefault(
            "split_fingerprint",
            _rlvr_prompt_fingerprints(arrays),
        )
    records = next(iter(arrays.values())).shape[0]
    array_specs: dict[str, object] = {}
    payload_files: list[Path] = []
    for name, array in arrays.items():
        path = root / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        payload_files.append(path)
        array_specs[name] = {
            "path": path.name,
            "dtype": array.dtype.name,
            "shape": list(array.shape),
        }
    bundle: dict[str, object] = {
        "schema": MMAP_BUNDLE_SCHEMA,
        "stage": stage,
        "family": "praxis",
        "tokenizer_sha256": tokenizer_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "vocabulary_size": vocabulary_size,
        "records": records,
        "sequence_length": sequence_length,
        "ngram_canonical_map_self_sha256": "d" * 64,
        "ngram_canonical_ids_sha256": _canonical_ids_sha256(vocabulary_size),
        "training": training or _training(),
        "arrays": array_specs,
        "bundle_sha256": "",
    }
    if stage == "context_extension":
        unique_active_tokens = int(
            np.count_nonzero(arrays["attention_mask"])
        )
        bundle["unique_active_tokens"] = unique_active_tokens
        bundle["training_tokens"] = unique_active_tokens * int(
            bundle["training"]["epochs"]  # type: ignore[index]
        )
    if is_gspo:
        bundle["working_set"] = {
            "token_chunk_size": 2,
            "candidate_micro_group_size": 4,
            "maximum_device_bytes": 1_000_000,
            "maximum_host_bytes": 1_000_000,
            "headroom_fraction": 0.5,
        }
        response_tokens = np.asarray(
            arrays["candidate_response_mask"]
        ).sum(axis=(1, 2), dtype=np.int64)
        total_tokens = int(response_tokens.sum())
        bundle.update(
            {
                "reasoning_mode_ids": {
                    "direct": 0,
                    "think": 1,
                    "think_max": 2,
                },
                "response_token_share_by_mode": {
                    mode: float(response_tokens[modes == mode_id].sum())
                    / total_tokens
                    for mode_id, mode in enumerate(
                        ("direct", "think", "think_max")
                    )
                },
                "target_token_share_by_mode_audited": True,
                "on_policy": True,
                "single_use_rollouts": True,
                "rollout_policy": "parent_checkpoint",
                "policy_updates_before_generation": 0,
                "samples_per_prompt": 16,
                "mode_overlap_validated": True,
            }
        )
        bundle["document_layout"] = "single_prompt_response_per_record"
    if bundle_metadata:
        bundle.update(bundle_metadata)
    bundle["bundle_sha256"] = _canonical_hash(bundle, omit="bundle_sha256")
    bundle_path = root / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    payload_files.append(bundle_path)
    schema = {
        "context_extension": "metis.context-extension-data/v1",
        "cold_start_sft": "metis.sft-data/v2",
        "overall_sft": "metis.sft-data/v2",
        "hybrid_mode_gspo": "metis.rlvr-data/v2",
        "specialist_reasoning": "metis.rlvr-data/v2",
    }[stage]
    envelope: dict[str, object] = {
        "envelope_schema": "metis.sealed-artifact/v1",
        "schema": schema,
        "complete": True,
        "tokenizer_sha256": tokenizer_sha256,
        "metadata": {
            "backend_contract": MMAP_BUNDLE_SCHEMA,
            "bundle_manifest": bundle_path.name,
        },
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _file_hash(path),
            }
            for path in payload_files
        ],
        "manifest_sha256": "",
    }
    envelope["manifest_sha256"] = _canonical_hash(
        envelope, omit="manifest_sha256"
    )
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    return SealedRequirement(
        name="data",
        schema=schema,
        environment_variable="TEST_DATA",
        manifest_path=manifest_path,
        manifest_sha256=str(envelope["manifest_sha256"]),
        payload=envelope,
    )


def _topology() -> ParallelTopology:
    return ParallelTopology(
        family="praxis",
        world_size=1,
        rank=0,
        local_rank=0,
        expert_parallel_size=1,
        expert_replica_count=1,
        expert_group=None,
        expert_group_ranks=(0,),
        expert_data_group=None,
        expert_data_group_ranks=(0,),
        dense_data_group=None,
    )


class TinyCausalLM(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)

    @property
    def output(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_logits: bool | None = None,
        **_kwargs: object,
    ) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        zero = logits.float().sum() * 0.0
        loss = (
            F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            if labels is not None
            else None
        )
        return SimpleNamespace(
            logits=logits if return_logits is not False else None,
            loss=loss,
            auxiliary_loss=zero,
            final_hidden_state=hidden,
        )


class MMapContractTests(unittest.TestCase):
    def test_context_gate_rejects_underthinking_regression(self) -> None:
        baseline = {
            "base_window_nll": 2.0,
            "full_context_tail_nll": 2.5,
            "needle_target_log_probability": -5.0,
        }
        policy = {
            "base_validation_relative_tolerance": 0.01,
            "long_validation_relative_tolerance": 0.005,
            "require_needle_gain_after_first_gate": True,
        }
        passing = _context_gate_decision(
            metrics={
                "base_window_nll": 2.01,
                "full_context_tail_nll": 2.4,
                "needle_target_log_probability": -4.9,
            },
            baseline=baseline,
            gate_target_tokens=6_000_000_000,
            gate_policy=policy,
        )
        self.assertTrue(passing["passed"])
        failing = _context_gate_decision(
            metrics={
                "base_window_nll": 2.01,
                "full_context_tail_nll": 2.4,
                "needle_target_log_probability": -5.1,
            },
            baseline=baseline,
            gate_target_tokens=6_000_000_000,
            gate_policy=policy,
        )
        self.assertFalse(failing["passed"])
        self.assertFalse(failing["needle_validation_passed"])

    def test_posttraining_tokenizer_binds_actual_base_release_bytes_and_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            base_root = root / "base"
            base_root.mkdir()
            base_tokenizer = base_root / "tokenizer.json"
            base_tokenizer.write_bytes(b'{"model":"exact-base-tokenizer"}')
            canonical_map = base_root / "NGRAM_CANONICAL_IDS.json"
            canonical_map.write_text("{}", encoding="utf-8")
            canonical_ids = base_root / "NGRAM_CANONICAL_IDS.uint16"
            lookup = np.arange(65_536, dtype="<u2")
            canonical_ids.write_bytes(lookup.tobytes())
            base_tokenizer_sha = _file_hash(base_tokenizer)
            canonical_ids_sha = _file_hash(canonical_ids)
            release_descriptor: dict[str, object] = {
                "schema": "metis.data-release/v2",
                "artifacts": {
                    "tokenizer": "tokenizer.json",
                    "ngram_canonical_map": "NGRAM_CANONICAL_IDS.json",
                    "ngram_canonical_ids": "NGRAM_CANONICAL_IDS.uint16",
                },
                "tokenizer_sha256": base_tokenizer_sha,
                "ngram_canonical_map_self_sha256": "b" * 64,
                "ngram_canonical_ids_sha256": canonical_ids_sha,
                "release_sha256": "",
            }
            release_descriptor["release_sha256"] = _canonical_hash(
                release_descriptor, omit="release_sha256"
            )
            (base_root / "RELEASE.json").write_text(
                json.dumps(release_descriptor), encoding="utf-8"
            )
            release_sha = str(release_descriptor["release_sha256"])

            post_root = root / "post-tokenizer"
            post_root.mkdir()
            sealed_tokenizer = post_root / "tokenizer.json"
            sealed_tokenizer.write_bytes(base_tokenizer.read_bytes())
            manifest_path = post_root / "MANIFEST.json"
            manifest_path.write_text("{}", encoding="utf-8")
            payload = {
                "metadata": {
                    "vocabulary_size": 65_536,
                    "tokenizer_file": "tokenizer.json",
                    "tokenizer_sha256": base_tokenizer_sha,
                    "base_release_sha256": release_sha,
                    "base_release_tokenizer_path": "tokenizer.json",
                    "base_release_tokenizer_sha256": base_tokenizer_sha,
                    "base_release_canonical_map_path": (
                        "NGRAM_CANONICAL_IDS.json"
                    ),
                    "base_release_canonical_ids_path": (
                        "NGRAM_CANONICAL_IDS.uint16"
                    ),
                    "ngram_canonical_map_self_sha256": "b" * 64,
                    "ngram_canonical_ids_sha256": canonical_ids_sha,
                },
                "files": [
                    {
                        "path": "tokenizer.json",
                        "bytes": sealed_tokenizer.stat().st_size,
                        "sha256": base_tokenizer_sha,
                    }
                ],
            }
            inventory = SimpleNamespace(
                root=base_root,
                tokenizer=base_tokenizer,
                release_sha256=release_sha,
                ngram_canonical_map=canonical_map,
                ngram_canonical_ids=canonical_ids,
                ngram_canonical_map_self_sha256="b" * 64,
                ngram_canonical_ids_sha256=canonical_ids_sha,
            )
            patches = (
                mock.patch(
                    "metis_training.stage_backend."
                    "ReleaseInventory.from_release_root",
                    return_value=inventory,
                ),
                mock.patch(
                    "metis_training.stage_backend."
                    "validate_canonical_id_sidecar"
                ),
            )
            with patches[0], patches[1] as validate_sidecar:
                observed, map_sha, ids_sha = _load_canonical_lookup(
                    data_release=base_root,
                    tokenizer_payload=payload,
                    tokenizer_manifest_path=manifest_path,
                    topology=_topology(),
                )
            self.assertTrue(np.array_equal(observed, lookup))
            self.assertEqual(map_sha, "b" * 64)
            self.assertEqual(ids_sha, canonical_ids_sha)
            validate_sidecar.assert_called_once()

            sealed_tokenizer.write_bytes(b'{"model":"claimed-only-drift"}')
            with patches[0], patches[1]:
                with self.assertRaisesRegex(
                    StageBackendError, "bytes differ"
                ):
                    _load_canonical_lookup(
                        data_release=base_root,
                        tokenizer_payload=payload,
                        tokenizer_manifest_path=manifest_path,
                        topology=_topology(),
                    )

    def test_release_index_is_self_hashed_and_pipeline_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            index: dict[str, object] = {
                "schema": "metis.posttraining-release-index/v1",
                "family": "praxis",
                "pipeline_sha256": "c" * 64,
                "tokenizer_manifest": "tokenizer/MANIFEST.json",
                "requirements": {},
                "index_sha256": "",
            }
            index["index_sha256"] = _canonical_hash(
                index, omit="index_sha256"
            )
            path = root / "INDEX.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            with mock.patch.dict(
                "os.environ",
                {"METIS_POSTTRAINING_RELEASE_INDEX": str(path)},
                clear=False,
            ):
                loaded = _load_release_index(
                    family="praxis",
                    pipeline_sha256="c" * 64,
                    topology=_topology(),
                )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[0]["index_sha256"], index["index_sha256"])

    def test_release_umbrella_selects_and_pins_family_index(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            family_root = root / "families" / "praxis"
            family_root.mkdir(parents=True)
            family_index: dict[str, object] = {
                "schema": "metis.posttraining-release-index/v1",
                "family": "praxis",
                "pipeline_sha256": "c" * 64,
                "tokenizer_manifest": {
                    "path": "tokenizer/MANIFEST.json",
                    "sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                },
                "requirements": {},
                "index_sha256": "",
            }
            family_index["index_sha256"] = _canonical_hash(
                family_index, omit="index_sha256"
            )
            family_path = family_root / "RELEASE_INDEX.json"
            family_path.write_text(json.dumps(family_index), encoding="utf-8")
            umbrella: dict[str, object] = {
                "schema": "metis.posttraining-release-umbrella/v1",
                "posttraining_contract_sha256": "c" * 64,
                "families": {
                    "praxis": {
                        "path": "families/praxis/RELEASE_INDEX.json",
                        "sha256": _file_hash(family_path),
                        "index_sha256": family_index["index_sha256"],
                    }
                },
                "umbrella_sha256": "",
            }
            umbrella["umbrella_sha256"] = _canonical_hash(
                umbrella, omit="umbrella_sha256"
            )
            umbrella_path = root / "POSTTRAINING_RELEASE.json"
            umbrella_path.write_text(json.dumps(umbrella), encoding="utf-8")
            with mock.patch.dict(
                "os.environ",
                {"METIS_POSTTRAINING_RELEASE_INDEX": str(umbrella_path)},
                clear=False,
            ):
                loaded = _load_release_index(
                    family="praxis",
                    pipeline_sha256="c" * 64,
                    topology=_topology(),
                )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[1], family_path.resolve())
            self.assertEqual(loaded[0]["family"], "praxis")

            umbrella["families"]["praxis"]["sha256"] = "d" * 64  # type: ignore[index]
            umbrella["umbrella_sha256"] = _canonical_hash(
                umbrella, omit="umbrella_sha256"
            )
            umbrella_path.write_text(json.dumps(umbrella), encoding="utf-8")
            with mock.patch.dict(
                "os.environ",
                {"METIS_POSTTRAINING_RELEASE_INDEX": str(umbrella_path)},
                clear=False,
            ):
                with self.assertRaisesRegex(StageBackendError, "file hash changed"):
                    _load_release_index(
                        family="praxis",
                        pipeline_sha256="c" * 64,
                        topology=_topology(),
                    )

    def test_supervised_bundle_is_memory_mapped_and_sealed(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        arrays = {
            "input_ids": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "labels": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = _sealed_bundle(
                Path(raw) / "sft",
                stage="cold_start_sft",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
            )
            bundle = MMapStageBundle.load(
                requirement,
                stage_id="cold_start_sft",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )
            self.assertIsInstance(bundle.arrays["input_ids"], np.memmap)
            self.assertEqual(bundle.records, 2)

    def test_supervised_bundle_rejects_canonical_id_or_sidecar_drift(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        input_ids = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32
        )
        canonical_ids = np.array(input_ids, copy=True)
        canonical_ids[0, 1] = 6
        arrays = {
            "input_ids": input_ids,
            "labels": np.array(input_ids, copy=True),
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
            "canonical_ids": canonical_ids,
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = _sealed_bundle(
                Path(raw) / "bad-canonical",
                stage="cold_start_sft",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
            )
            with self.assertRaisesRegex(
                StageBackendError, "canonical_ids differ"
            ):
                MMapStageBundle.load(
                    requirement,
                    stage_id="cold_start_sft",
                    family="praxis",
                    tokenizer_sha256=tokenizer,
                    parent_checkpoint_sha256=parent,
                    vocabulary_size=7,
                    canonical_id_lookup=_canonical_lookup(7),
                    canonical_map_self_sha256="d" * 64,
                    canonical_ids_sha256=_canonical_ids_sha256(7),
                )

            requirement = _sealed_bundle(
                Path(raw) / "bad-sidecar",
                stage="cold_start_sft",
                arrays={
                    "input_ids": input_ids,
                    "labels": np.array(input_ids, copy=True),
                    "loss_mask": np.ones((2, 4), dtype=np.bool_),
                },
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
            )
            with self.assertRaisesRegex(
                StageBackendError, "sidecar lineage"
            ):
                MMapStageBundle.load(
                    requirement,
                    stage_id="cold_start_sft",
                    family="praxis",
                    tokenizer_sha256=tokenizer,
                    parent_checkpoint_sha256=parent,
                    vocabulary_size=7,
                    canonical_id_lookup=_canonical_lookup(7),
                    canonical_map_self_sha256="e" * 64,
                    canonical_ids_sha256=_canonical_ids_sha256(7),
                )

    def test_supervised_boundaries_padding_and_token_count_fail_closed(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64

        def arrays() -> dict[str, np.ndarray]:
            return {
                "input_ids": np.array(
                    [[1, 2, 3, 0], [4, 5, 0, 0]], dtype=np.int32
                ),
                "labels": np.array(
                    [[1, 2, 3, -100], [4, 5, -100, -100]],
                    dtype=np.int32,
                ),
                "loss_mask": np.array(
                    [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.bool_
                ),
                "attention_mask": np.array(
                    [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.bool_
                ),
                "document_ids": np.array(
                    [[0, 0, 1, -1], [2, 2, -1, -1]], dtype=np.int32
                ),
                "reset_mask": np.array(
                    [[1, 0, 1, 0], [1, 0, 0, 0]], dtype=np.bool_
                ),
                "canonical_ids": np.array(
                    [[1, 2, 3, 0], [4, 5, 0, 0]], dtype=np.int32
                ),
            }

        def load(requirement: SealedRequirement) -> MMapStageBundle:
            return MMapStageBundle.load(
                requirement,
                stage_id="context_extension",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            valid = _sealed_bundle(
                root / "valid",
                stage="context_extension",
                arrays=arrays(),
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                bundle_metadata={"training_tokens": 5},
            )
            self.assertEqual(load(valid).manifest["training_tokens"], 5)

            bad_reset_arrays = arrays()
            bad_reset_arrays["reset_mask"][0, 1] = True
            bad_reset = _sealed_bundle(
                root / "bad-reset",
                stage="context_extension",
                arrays=bad_reset_arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                bundle_metadata={"training_tokens": 5},
            )
            with self.assertRaisesRegex(StageBackendError, "reset_mask"):
                load(bad_reset)

            bad_padding_arrays = arrays()
            bad_padding_arrays["labels"][0, 3] = 0
            bad_padding = _sealed_bundle(
                root / "bad-padding",
                stage="context_extension",
                arrays=bad_padding_arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                bundle_metadata={"training_tokens": 5},
            )
            with self.assertRaisesRegex(StageBackendError, "padding"):
                load(bad_padding)

            bad_count = _sealed_bundle(
                root / "bad-count",
                stage="context_extension",
                arrays=arrays(),
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                bundle_metadata={"training_tokens": 6},
            )
            with self.assertRaisesRegex(
                StageBackendError, "exactly equal"
            ):
                load(bad_count)

    def test_context_extension_counts_total_exposure_across_epochs(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        arrays = {
            "input_ids": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "labels": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
        }

        def load(requirement: SealedRequirement) -> MMapStageBundle:
            return MMapStageBundle.load(
                requirement,
                stage_id="context_extension",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            valid = _sealed_bundle(
                root / "valid",
                stage="context_extension",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                training=_training(epochs=2),
                bundle_metadata={
                    "unique_active_tokens": 8,
                    "training_tokens": 16,
                },
            )
            self.assertEqual(load(valid).manifest["training_tokens"], 16)

            one_epoch_claim = _sealed_bundle(
                root / "one-epoch-claim",
                stage="context_extension",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                training=_training(epochs=2),
                bundle_metadata={
                    "unique_active_tokens": 8,
                    "training_tokens": 8,
                },
            )
            with self.assertRaisesRegex(StageBackendError, "times epochs"):
                load(one_epoch_claim)

            false_unique_count = _sealed_bundle(
                root / "false-unique-count",
                stage="context_extension",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                training=_training(epochs=2),
                bundle_metadata={
                    "unique_active_tokens": 16,
                    "training_tokens": 32,
                },
            )
            with self.assertRaisesRegex(StageBackendError, "unique_active_tokens"):
                load(false_unique_count)

    def test_bound_bundle_rejects_family_or_checkpoint_mismatch(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        arrays = {
            "input_ids": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "labels": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = dataclasses.replace(
                _sealed_bundle(
                    Path(raw) / "bound",
                    stage="cold_start_sft",
                    arrays=arrays,
                    sequence_length=4,
                    vocabulary_size=7,
                    tokenizer_sha256=tokenizer,
                    parent_checkpoint_sha256=parent,
                ),
                family_bound=True,
                checkpoint_bound=True,
            )
            common = {
                "requirement": requirement,
                "stage_id": "cold_start_sft",
                "tokenizer_sha256": tokenizer,
                "vocabulary_size": 7,
                "canonical_id_lookup": _canonical_lookup(7),
                "canonical_map_self_sha256": "d" * 64,
                "canonical_ids_sha256": _canonical_ids_sha256(7),
            }
            with self.assertRaisesRegex(StageBackendError, "family mismatch"):
                MMapStageBundle.load(
                    **common,
                    family="logos",
                    parent_checkpoint_sha256=parent,
                )
            with self.assertRaisesRegex(
                StageBackendError, "live parent"
            ):
                MMapStageBundle.load(
                    **common,
                    family="praxis",
                    parent_checkpoint_sha256="c" * 64,
                )

    def test_measured_oom_migration_preserves_effective_batch_and_resume(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        arrays = {
            "input_ids": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "labels": np.arange(8, dtype=np.int32).reshape(2, 4) % 7,
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            requirement = _sealed_bundle(
                root / "bundle",
                stage="cold_start_sft",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                training=_training(
                    micro_batch_size=2,
                    gradient_accumulation=1,
                ),
            )
            bundle = MMapStageBundle.load(
                requirement,
                stage_id="cold_start_sft",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )
            output_root = root / "output"
            request_path = _write_stage_oom_request(
                output_root=output_root,
                family="praxis",
                stage_id="cold_start_sft",
                parent_checkpoint_sha256=parent,
                precision_role_plan_sha256="e" * 64,
                bundle=bundle,
                topology=_topology(),
                phase="kernel_canary",
                resume={
                    "epoch": 0,
                    "next_global_batch": 3,
                    "optimizer_step": 3,
                    "campaign_token_cursor": 100,
                },
                exception=RuntimeError("CUDA out of memory"),
            )
            request = json.loads(request_path.read_text(encoding="utf-8"))
            revision: dict[str, object] = {
                "reason": "measured_stage_oom",
                "old": request["current"],
                "new": request["proposed"],
                "prior_batch_migration_sha256": None,
                "oom_request_path": str(request_path),
                "oom_request_file_sha256": _file_hash(request_path),
                "oom_request_sha256": request["request_sha256"],
                "revision_sha256": "",
            }
            revision["revision_sha256"] = _canonical_hash(
                revision, omit="revision_sha256"
            )
            receipt: dict[str, object] = {
                "schema": BATCH_MIGRATION_SCHEMA,
                "family": "praxis",
                "stage": "cold_start_sft",
                "parent_checkpoint_sha256": parent,
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "precision_role_plan_sha256": "e" * 64,
                "sealed_training": {
                    "micro_batch_size": 2,
                    "gradient_accumulation": 1,
                },
                "effective_local_batch_records": 2,
                "revisions": [revision],
                "receipt_sha256": "",
            }
            receipt["receipt_sha256"] = _canonical_hash(
                receipt, omit="receipt_sha256"
            )
            migration_root = root / "migrations"
            migration_path = (
                migration_root / "praxis" / "cold_start_sft.json"
            )
            migration_path.parent.mkdir(parents=True)
            migration_path.write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with mock.patch.dict(
                "os.environ",
                {
                    "METIS_POSTTRAINING_BATCH_MIGRATION_ROOT": str(
                        migration_root
                    )
                },
                clear=False,
            ):
                migrated = _load_stage_batch_migration(
                    bundle,
                    family="praxis",
                    parent_checkpoint_sha256=parent,
                    precision_role_plan_sha256="e" * 64,
                    output_root=output_root,
                    topology=_topology(),
                )
            self.assertEqual(migrated.training["micro_batch_size"], 1)
            self.assertEqual(migrated.training["gradient_accumulation"], 2)
            self.assertEqual(
                int(migrated.training["micro_batch_size"])
                * int(migrated.training["gradient_accumulation"]),
                2,
            )
            self.assertEqual(
                _resume_global_batch(
                    {
                        "runtime_batch": {
                            "micro_batch_size": 2,
                            "gradient_accumulation": 1,
                        },
                        "next_global_batch": 3,
                    },
                    migrated,
                ),
                6,
            )

    def test_rlvr_refuses_stale_or_reused_rollouts(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        records, group, sequence = 3, 16, 4
        arrays = {
            "candidate_input_ids": np.zeros(
                (records, group, sequence), dtype=np.int32
            ),
            "candidate_response_mask": np.ones(
                (records, group, sequence - 1), dtype=np.bool_
            ),
            "old_token_log_probs": np.zeros(
                (records, group, sequence - 1), dtype=np.float32
            ),
            "correctness": np.tile(
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                         dtype=np.float32),
                (records, 1),
            ),
            "truncated": np.zeros((records, group), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = _sealed_bundle(
                Path(raw) / "rlvr",
                stage="specialist_reasoning",
                arrays=arrays,
                sequence_length=sequence,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                training=_training(epochs=2),
                bundle_metadata={
                    "on_policy": True,
                    "single_use_rollouts": True,
                    "rollout_policy": "parent_checkpoint",
                    "policy_updates_before_generation": 0,
                    "samples_per_prompt": 16,
                },
            )
            with self.assertRaisesRegex(StageBackendError, "single-use"):
                MMapStageBundle.load(
                    requirement,
                    stage_id="specialist_reasoning",
                    family="praxis",
                    tokenizer_sha256=tokenizer,
                    parent_checkpoint_sha256=parent,
                    vocabulary_size=7,
                    canonical_id_lookup=_canonical_lookup(7),
                    canonical_map_self_sha256="d" * 64,
                    canonical_ids_sha256=_canonical_ids_sha256(7),
                )

    def test_working_set_cap_fails_before_training(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        records, group, sequence = 3, 16, 4
        arrays = {
            "candidate_input_ids": np.zeros(
                (records, group, sequence), dtype=np.int32
            ),
            "candidate_response_mask": np.ones(
                (records, group, sequence - 1), dtype=np.bool_
            ),
            "old_token_log_probs": np.zeros(
                (records, group, sequence - 1), dtype=np.float32
            ),
            "correctness": np.tile(
                np.array([0] * 8 + [1] * 8, dtype=np.float32), (records, 1)
            ),
            "truncated": np.zeros((records, group), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = _sealed_bundle(
                Path(raw) / "rlvr",
                stage="specialist_reasoning",
                arrays=arrays,
                sequence_length=sequence,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                bundle_metadata={
                    "on_policy": True,
                    "single_use_rollouts": True,
                    "rollout_policy": "parent_checkpoint",
                    "policy_updates_before_generation": 0,
                    "samples_per_prompt": 16,
                    "working_set": {
                        "token_chunk_size": 2,
                        "candidate_micro_group_size": 4,
                        "maximum_device_bytes": 1,
                        "maximum_host_bytes": 1,
                        "headroom_fraction": 0.5,
                    },
                },
            )
            bundle = MMapStageBundle.load(
                requirement,
                stage_id="specialist_reasoning",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )
            with self.assertRaisesRegex(StageBackendError, "working set"):
                _validate_runtime_working_set(
                    bundle,
                    runtime=Runtime(
                        device=torch.device("cpu"),
                        rank=0,
                        local_rank=0,
                        world_size=1,
                        distributed=False,
                    ),
                    topology=_topology(),
                    vocabulary_size=7,
                )


class InProcessLoopTests(unittest.TestCase):
    def test_length_shaping_applies_only_to_think_mode(self) -> None:
        correctness = torch.tensor(
            [[0.0] * 8 + [1.0] * 8] * 3,
            dtype=torch.float32,
        )
        response_mask = torch.ones(3, 16, 4, dtype=torch.bool)
        response_mask[1, 8:, 1:] = False
        rewards, _rates, _lengths, _diagnostics = _rlvr_rewards(
            "specialist_reasoning",
            {
                "correctness": correctness,
                "candidate_response_mask": response_mask,
                "reasoning_mode": torch.tensor([0, 1, 2]),
                "mode_compliance": torch.ones_like(correctness),
            },
            length_coefficient=0.05,
            mode_compliance_weight=1.0,
        )
        torch.testing.assert_close(rewards[0], correctness[0])
        torch.testing.assert_close(rewards[2], correctness[2])
        self.assertFalse(torch.equal(rewards[1], correctness[1]))

    def test_active_state_reconciles_same_cursor_final_promotion_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = root / "checkpoints" / "tokens-0000000000042"
            checkpoint.mkdir(parents=True)
            runtime_batch = {
                "micro_batch_size": 1,
                "gradient_accumulation": 1,
            }
            metrics = {"loss": 0.25, "grad_norm": 0.5}
            extra = {
                "posttraining_stage": "context_extension",
                "parent_checkpoint_sha256": "a" * 64,
                "stage_config_sha256": "b" * 64,
                "bundle_sha256": "c" * 64,
                "optimizer_state_policy": "preserve",
                "runtime_batch": runtime_batch,
                "stage_epoch": 1,
                "stage_next_global_batch": 0,
                "stage_optimizer_step": 9,
                "campaign_token_cursor": 42,
                "last_loss": metrics["loss"],
                "last_grad_norm": metrics["grad_norm"],
                "last_metrics": metrics,
                "stage_complete": True,
            }
            manifest = {
                "schema": "metis.distributed-checkpoint/v1",
                "phase": "context_extension",
                "optimizer_step": 9,
                "global_token_cursor": 42,
                "phase_boundary": True,
                "extra_state": extra,
            }
            manifest["checkpoint_sha256"] = _canonical_hash(
                manifest,
                omit="checkpoint_sha256",
            )
            (checkpoint / "MANIFEST.json").write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            state = {
                "active": {
                    "kind": "policy",
                    "stage_id": "context_extension",
                    "parent_checkpoint_sha256": "a" * 64,
                    "stage_config_sha256": "b" * 64,
                    "bundle_sha256": "c" * 64,
                    "optimizer_state_policy": "preserve",
                    "runtime_batch": runtime_batch,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": "d" * 64,
                    "epoch": 1,
                    "next_global_batch": 0,
                    "optimizer_step": 9,
                    "campaign_token_cursor": 42,
                    "last_loss": metrics["loss"],
                    "last_grad_norm": metrics["grad_norm"],
                    "last_metrics": metrics,
                },
                "state_sha256": "",
            }
            state_path = root / "STATE.json"
            reconciled = _reconcile_active_policy_checkpoint_state(
                state_path=state_path,
                state=state,
                topology=_topology(),
            )
            self.assertEqual(
                reconciled["active"]["checkpoint_sha256"],
                manifest["checkpoint_sha256"],
            )
            self.assertTrue(reconciled["active"]["checkpoint_stage_complete"])
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["active"]["checkpoint_sha256"],
                manifest["checkpoint_sha256"],
            )
            self.assertEqual(
                persisted["state_sha256"],
                _canonical_hash(persisted, omit="state_sha256"),
            )

    def test_candidate_micro_chunks_equal_dense_sequence_gspo(self) -> None:
        torch.manual_seed(11)
        current = torch.randn(2, 16, 5) * 0.01
        old = torch.randn(2, 16, 5) * 0.01
        rewards = torch.randn(2, 16)
        mask = torch.ones(2, 16, 5, dtype=torch.bool)
        truncated = torch.zeros(2, 16, dtype=torch.bool)
        dense = gspo_loss(
            current_token_log_probs=current,
            old_token_log_probs=old,
            rewards=rewards,
            response_mask=mask,
            truncated=truncated,
        )
        normalized = (
            rewards - rewards.mean(dim=1, keepdim=True)
        ) / rewards.std(dim=1, keepdim=True, unbiased=False)
        weighted_loss = torch.zeros(())
        valid = 0
        for start in range(0, 16, 4):
            result = gspo_token_loss(
                current_token_log_probs=current[:, start : start + 4],
                old_token_log_probs=old[:, start : start + 4],
                token_advantages=normalized[:, start : start + 4, None].expand(
                    -1, -1, 5
                ),
                response_mask=mask[:, start : start + 4],
                truncated=truncated[:, start : start + 4],
            )
            count = int(result["valid_sequences"].item())
            weighted_loss = weighted_loss + result["loss"] * count
            valid += count
        torch.testing.assert_close(weighted_loss / valid, dense["loss"])

    def test_optimizer_reset_is_fresh_stage_only(self) -> None:
        model = TinyCausalLM(7)
        optimizer = OptimizerBundle(
            torch.optim.AdamW(model.parameters(), lr=1.0e-2), None
        )
        loss = model.lm_head(model.embedding(torch.tensor([[1, 2]]))).sum()
        loss.backward()
        optimizer.step()
        self.assertTrue(optimizer.dense.state)
        first_parameter = next(iter(optimizer.dense.state))
        optimizer.dense.state[first_parameter]["master_param"] = (
            first_parameter.detach().float().clone()
        )
        master_before = optimizer.dense.state[first_parameter][
            "master_param"
        ].clone()
        parameter_before = first_parameter.detach().clone()
        _apply_optimizer_state_transition(
            optimizer, policy="reset", active_resume=True
        )
        self.assertTrue(optimizer.dense.state)
        _apply_optimizer_state_transition(
            optimizer, policy="reset", active_resume=False
        )
        self.assertTrue(optimizer.dense.state)
        for state in optimizer.dense.state.values():
            self.assertEqual(float(state["step"].item()), 0.0)
            self.assertEqual(float(state["exp_avg"].abs().sum().item()), 0.0)
            self.assertEqual(float(state["exp_avg_sq"].abs().sum().item()), 0.0)
        torch.testing.assert_close(
            optimizer.dense.state[first_parameter]["master_param"],
            master_before,
        )
        torch.testing.assert_close(first_parameter.detach(), parameter_before)

    def test_chunked_selected_log_probs_match_dense_logits(self) -> None:
        torch.manual_seed(3)
        model = TinyCausalLM(7)
        hidden = torch.randn(2, 5, 8, requires_grad=True)
        targets = torch.randint(0, 7, (2, 5))
        observed = _selected_token_log_probs_from_hidden(
            model,
            hidden,
            targets,
            token_chunk_size=2,
        )
        dense_logits = model.lm_head(hidden).float()
        expected = torch.log_softmax(dense_logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        torch.testing.assert_close(observed, expected)

    def test_next_token_alignment_matches_masked_objective(self) -> None:
        labels = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
        loss_mask = torch.tensor(
            [[False, True, True, True], [False, True, False, True]]
        )
        attention = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        document_ids = torch.tensor([[0, 0, 0, 0], [1, 1, 2, -1]])
        reset_mask = torch.tensor(
            [[True, False, False, False], [True, False, True, False]]
        )
        logits = torch.randn(2, 4, 7)
        aligned = _align_supervised_labels(
            labels,
            loss_mask,
            attention,
            document_ids,
            reset_mask,
        )
        chunk_contract_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            aligned.reshape(-1),
            ignore_index=-100,
        )
        expected = masked_causal_cross_entropy(
            logits[:, :-1],
            labels[:, 1:],
            loss_mask[:, 1:]
            & attention[:, :-1]
            & attention[:, 1:]
            & document_ids[:, :-1].eq(document_ids[:, 1:])
            & ~reset_mask[:, 1:],
        )
        torch.testing.assert_close(chunk_contract_loss, expected)

    def test_supervised_loop_updates_live_model(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        arrays = {
            "input_ids": np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32),
            "labels": np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32),
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = _sealed_bundle(
                Path(raw) / "sft",
                stage="cold_start_sft",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
            )
            bundle = MMapStageBundle.load(
                requirement,
                stage_id="cold_start_sft",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )
            model = TinyCausalLM(7)
            before = model.output.weight.detach().clone()
            optimizer = OptimizerBundle(
                torch.optim.AdamW(model.parameters(), lr=1.0e-2), None
            )
            metrics = _run_supervised_stage(
                stage={
                    "id": "cold_start_sft",
                    "sequence_length": 4,
                    "objective": {"name": "response_only_cross_entropy"},
                },
                bundle=bundle,
                model=model,
                optimizer=optimizer,
                runtime=Runtime(
                    device=torch.device("cpu"),
                    rank=0,
                    local_rank=0,
                    world_size=1,
                    distributed=False,
                ),
                topology=_topology(),
                start_epoch=0,
                start_global_batch=0,
                start_optimizer_step=0,
                start_cursor=1_000,
                checkpoint_callback=lambda *_args: None,
            )
            self.assertEqual(metrics["optimizer_steps"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertFalse(torch.equal(before, model.output.weight.detach()))

    def test_signal_requeues_only_after_safe_checkpoint_callback(self) -> None:
        tokenizer = "a" * 64
        parent = "b" * 64
        arrays = {
            "input_ids": np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32),
            "labels": np.array([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32),
            "loss_mask": np.ones((2, 4), dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as raw:
            requirement = _sealed_bundle(
                Path(raw) / "sft",
                stage="cold_start_sft",
                arrays=arrays,
                sequence_length=4,
                vocabulary_size=7,
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
            )
            bundle = MMapStageBundle.load(
                requirement,
                stage_id="cold_start_sft",
                family="praxis",
                tokenizer_sha256=tokenizer,
                parent_checkpoint_sha256=parent,
                vocabulary_size=7,
                canonical_id_lookup=_canonical_lookup(7),
                canonical_map_self_sha256="d" * 64,
                canonical_ids_sha256=_canonical_ids_sha256(7),
            )
            model = TinyCausalLM(7)
            optimizer = OptimizerBundle(
                torch.optim.AdamW(model.parameters(), lr=1.0e-2), None
            )
            checkpoints: list[object] = []
            coordinator = SimpleNamespace(requested=True, reason="SIGUSR1")
            with self.assertRaises(PostTrainingRequeue) as raised:
                _run_supervised_stage(
                    stage={
                        "id": "cold_start_sft",
                        "sequence_length": 4,
                        "objective": {"name": "response_only_cross_entropy"},
                    },
                    bundle=bundle,
                    model=model,
                    optimizer=optimizer,
                    runtime=Runtime(
                        device=torch.device("cpu"),
                        rank=0,
                        local_rank=0,
                        world_size=1,
                        distributed=False,
                    ),
                    topology=_topology(),
                    start_epoch=0,
                    start_global_batch=0,
                    start_optimizer_step=0,
                    start_cursor=1_000,
                    checkpoint_callback=lambda progress, complete: checkpoints.append(
                        (progress, complete)
                    ),
                    signal_coordinator=coordinator,
                )
            self.assertEqual(raised.exception.code, 75)
            self.assertEqual(len(checkpoints), 1)
            self.assertFalse(checkpoints[0][1])

if __name__ == "__main__":
    unittest.main()
