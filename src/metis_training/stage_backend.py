from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import nullcontext
from inspect import signature
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from metis_data.ngram_canonical import validate_canonical_id_sidecar

from .checkpointing import CheckpointManager, SignalCoordinator
from .contracts import canonical_json_sha256, load_autotune_selection, sha256_file
from .data import ReleaseInventory
from .distributed import (
    ParallelTopology,
    Runtime,
    all_reduce_sum,
    barrier,
    global_any,
    normalize_summed_gradients,
    synchronize_gradients,
)
from .optimizers import OptimizerBundle, clip_grad_norm_
from .posttraining import (
    CHECKPOINT_RECEIPT_SCHEMA,
    EXPECTED_STAGE_IDS,
    PipelineContractError,
    REASONING_MODES,
    RLVR_GSPO_STAGE_IDS,
    SPECIALIST_STAGE_IDS,
    avg_at_k,
    difficulty_adaptive_length_reward,
    evaluate_metric_gate,
    gated_code_efficiency_reward,
    gspo_token_loss,
    load_pipeline,
    strict_on_policy_filter,
    thinking_length_diagnostics,
)


MMAP_BUNDLE_SCHEMA = "metis.posttraining-mmap/v1"
EVALUATION_RESULTS_SCHEMA = "metis.evaluation-results/v1"
CAMPAIGN_STATE_SCHEMA = "metis.inprocess-posttraining-state/v1"
STAGE_RECEIPT_SCHEMA = "metis.inprocess-stage-receipt/v1"
RELEASE_CANDIDATE_SCHEMA = "metis.release-candidate/v1"
RELEASE_INDEX_SCHEMA = "metis.posttraining-release-index/v1"
RELEASE_UMBRELLA_SCHEMA = "metis.posttraining-release-umbrella/v1"
CONTEXT_GATE_RECEIPT_SCHEMA = "metis.context-gate-checkpoint/v1"
CONTEXT_GATE_BASELINE_SCHEMA = "metis.context-gate-baseline/v1"
CONTEXT_GATE_EVALUATION_SCHEMA = "metis.context-gate-evaluation/v1"
CONTEXT_GATE_PROMOTION_SCHEMA = "metis.context-gate-promotion/v1"
PROFILE_SELECTION_SCHEMA = "metis.posttraining-profile-selection/v1"
LIVE_PROFILE_AUTOTUNE_SCHEMA = "metis.posttraining-live-profile-autotune/v1"
LIVE_PROFILE_AUTOTUNE_RECEIPT_SCHEMA = (
    "metis.posttraining-live-profile-autotune-receipt/v1"
)
LIVE_PROFILE_EVALUATOR_SCHEMA = "metis.posttraining-offline-policy-evaluator/v1"
BATCH_MIGRATION_SCHEMA = "metis.posttraining-batch-migration/v1"
WORKING_SET_AUTOTUNE_SCHEMA = "metis.posttraining-working-set-autotune/v1"
WORKING_SET_AUTOTUNE_RECEIPT_SCHEMA = (
    "metis.posttraining-working-set-autotune-receipt/v1"
)
OOM_REVISION_REQUEST_SCHEMA = "metis.posttraining-oom-revision-request/v1"
DEFERRED_MATERIALIZATION_REQUEST_SCHEMA = (
    "metis.deferred-materialization-request/v1"
)
DEFERRED_MATERIALIZATION_EXIT_CODE = 252
BASE_TOKEN_CURSOR = 1_000_000_000_000

_DATA_SCHEMAS = {
    "context_extension": "metis.context-extension-data/v1",
    "cold_start_sft": "metis.sft-data/v2",
    "overall_sft": "metis.sft-data/v2",
    **{stage_id: "metis.rlvr-data/v2" for stage_id in RLVR_GSPO_STAGE_IDS},
    "opd_consolidation": "metis.opd-data/v1",
    "evaluation": EVALUATION_RESULTS_SCHEMA,
}


def _is_specialist_stage(stage_id: str) -> bool:
    return stage_id in SPECIALIST_STAGE_IDS


def _is_gspo_stage(stage_id: str) -> bool:
    return stage_id in RLVR_GSPO_STAGE_IDS

_SUPERVISED_ARRAYS = {
    "input_ids",
    "labels",
    "loss_mask",
    "attention_mask",
    "document_ids",
    "reset_mask",
    "canonical_ids",
}
_COMPACT_SUPERVISED_ARRAYS = {
    "input_ids",
    "document_start",
    "sequence_lengths",
    "gate_ids",
}
_CONTEXT_EVALUATION_ARRAYS = {
    "context_evaluation_input_ids",
    "context_evaluation_probe_target_ids",
    "context_evaluation_probe_positions",
    "context_evaluation_split_fingerprint",
}
_COMPACT_CAUSAL_LAYOUT = "metis.compact-causal/v1"
_RLVR_ARRAYS = {
    "split_fingerprint",
    "base_prompt_fingerprint",
    "reasoning_mode",
    "mode_overlap_id",
    "candidate_input_ids",
    "candidate_attention_mask",
    "candidate_response_mask",
    "old_token_log_probs",
    "correctness",
    "mode_compliance",
    "truncated",
}
_OPD_ARRAYS = {
    "split_fingerprint",
    "reasoning_mode",
    "input_ids",
    "attention_mask",
    "response_mask",
    "teacher_union_token_ids",
    "teacher_union_logits",
    "teacher_union_count",
    "teacher_route",
}
def _content_fingerprint(
    domain: str,
    fields: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    for name, value in fields:
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                list(contiguous.shape),
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
    return np.frombuffer(digest.digest(), dtype=np.uint8).copy()


def _rlvr_prompt_fingerprints(
    arrays: Mapping[str, np.ndarray],
    *,
    prefix: str = "",
) -> np.ndarray:
    ids = np.asarray(arrays[f"{prefix}candidate_input_ids"])
    attention = np.asarray(
        arrays[f"{prefix}candidate_attention_mask"]
    ).astype(np.bool_, copy=False)
    response = np.asarray(
        arrays[f"{prefix}candidate_response_mask"]
    ).astype(np.bool_, copy=False)
    if ids.ndim != 3 or attention.shape != ids.shape or response.shape != (
        ids.shape[0],
        ids.shape[1],
        ids.shape[2] - 1,
    ):
        raise StageBackendError(
            "RLVR split fingerprints require aligned candidate arrays"
        )
    rows: list[np.ndarray] = []
    for record in range(ids.shape[0]):
        starts: list[int] = []
        for candidate in range(ids.shape[1]):
            positions = np.flatnonzero(response[record, candidate])
            if positions.size == 0:
                raise StageBackendError(
                    "RLVR split fingerprint cannot derive an empty response"
                )
            starts.append(int(positions[0]))
        if len(set(starts)) != 1:
            raise StageBackendError(
                "RLVR candidates disagree on the prompt/response boundary"
            )
        prompt_tokens = starts[0] + 1
        prompt_ids = ids[record, :, :prompt_tokens]
        prompt_attention = attention[record, :, :prompt_tokens]
        if (
            not np.all(prompt_ids == prompt_ids[0:1])
            or not np.all(prompt_attention == prompt_attention[0:1])
        ):
            raise StageBackendError(
                "RLVR candidates do not share one canonical prompt prefix"
            )
        rows.append(
            _content_fingerprint(
                "metis/rlvr-prompt/v1",
                [
                    (
                        "prompt_input_ids",
                        np.asarray(prompt_ids[0], dtype="<i8"),
                    ),
                    (
                        "prompt_attention_mask",
                        np.asarray(prompt_attention[0], dtype=np.uint8),
                    ),
                ],
            )
        )
    return np.stack(rows)


class StageBackendError(PipelineContractError):
    """Raised when the in-process stage backend must fail closed."""


class PostTrainingRequeue(SystemExit):
    """Typed, checkpoint-safe Slurm requeue request (exit status 75)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(75)


class PostTrainingOOM(SystemExit):
    """Typed supervisor handoff after a stage-local measured OOM (status 253)."""

    def __init__(self, request_path: Path) -> None:
        self.request_path = request_path
        super().__init__(253)


class DeferredMaterialization(SystemExit):
    """Supervisor handoff for a checkpoint-bound deferred input (status 252)."""

    def __init__(self, request_path: Path) -> None:
        self.request_path = request_path
        super().__init__(DEFERRED_MATERIALIZATION_EXIT_CODE)


def _canonical_hash(value: Any, *, omit: Iterable[str] = ()) -> str:
    omitted = set(omit)
    if isinstance(value, Mapping):
        value = {key: item for key, item in value.items() if key not in omitted}
    return canonical_json_sha256(value)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_relative(root: Path, raw: str, *, label: str) -> Path:
    relative = Path(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise StageBackendError(f"{label} must be a safe relative path")
    unresolved = root / relative
    if unresolved.is_symlink():
        raise StageBackendError(f"{label} may not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise StageBackendError(f"{label} escapes the sealed artifact") from exc
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise StageBackendError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBackendError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise StageBackendError(f"{label} must contain a JSON object")
    return value


def _collective_errors(
    topology: ParallelTopology,
    operation: Any,
    *,
    label: str,
) -> Any:
    result: Any = None
    failure: str | None = None
    try:
        result = operation()
    except BaseException as exc:
        failure = f"rank {topology.rank}: {type(exc).__name__}: {exc}"
    if topology.distributed:
        failures: list[str | None] = [None] * topology.world_size
        dist.all_gather_object(failures, failure, group=topology.dense_data_group)
    else:
        failures = [failure]
    observed = [item for item in failures if item]
    if observed:
        raise StageBackendError(f"{label} failed collectively: {'; '.join(observed)}")
    return result


def _broadcast_object(value: Any, topology: ParallelTopology) -> Any:
    if not topology.distributed:
        return value
    objects = [value if topology.rank == 0 else None]
    dist.broadcast_object_list(objects, src=0, group=topology.dense_data_group)
    return objects[0]


def _all_reduce_int(value: int, topology: ParallelTopology, device: torch.device) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    all_reduce_sum(tensor, topology)
    return int(tensor.item())


def _signal_requested(
    signal_coordinator: SignalCoordinator | None,
    *,
    topology: ParallelTopology,
    runtime: Runtime,
) -> bool:
    if signal_coordinator is None:
        return False
    return global_any(
        bool(signal_coordinator.requested),
        topology,
        runtime.device,
    )


def _output_value(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


@dataclasses.dataclass(frozen=True)
class SealedRequirement:
    name: str
    schema: str
    environment_variable: str
    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]
    family_bound: bool = False
    checkpoint_bound: bool = False


@dataclasses.dataclass(frozen=True)
class ArraySpec:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class VocabChunk:
    path: Path
    vocab_start: int
    vocab_end: int
    values: np.ndarray


@dataclasses.dataclass(frozen=True)
class ChunkedTeacherDistribution:
    name: str
    records: int
    tokens: int
    vocabulary_size: int
    chunks: tuple[VocabChunk, ...]

    def token_slice(
        self,
        record_indices: np.ndarray,
        token_start: int,
        token_end: int,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        for chunk in self.chunks:
            yield (
                chunk.vocab_start,
                chunk.vocab_end,
                np.array(
                    chunk.values[record_indices, token_start:token_end],
                    copy=True,
                ),
            )


@dataclasses.dataclass
class MMapStageBundle:
    stage_id: str
    family: str
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    arrays: dict[str, np.ndarray]
    specs: dict[str, ArraySpec]
    teacher_distributions: dict[str, ChunkedTeacherDistribution]
    training: Mapping[str, Any]
    sealed_training: Mapping[str, Any]
    batch_migration_path: Path | None
    batch_migration_sha256: str | None
    batch_migration_chain: tuple[tuple[int, int], ...]
    working_set_autotune_sha256: str | None
    canonical_id_lookup: np.ndarray
    canonical_map_self_sha256: str
    canonical_ids_sha256: str
    canonical_lookup_tensor: Tensor | None
    working_set: Mapping[str, Any]
    records: int
    sequence_length: int

    @classmethod
    def load(
        cls,
        requirement: SealedRequirement,
        *,
        stage_id: str,
        family: str,
        tokenizer_sha256: str,
        parent_checkpoint_sha256: str,
        vocabulary_size: int,
        canonical_id_lookup: np.ndarray,
        canonical_map_self_sha256: str,
        canonical_ids_sha256: str,
    ) -> "MMapStageBundle":
        envelope = requirement.payload
        metadata = _require_mapping(envelope.get("metadata"), "sealed data metadata")
        if metadata.get("backend_contract") != MMAP_BUNDLE_SCHEMA:
            raise StageBackendError(
                f"{requirement.environment_variable} must declare "
                f"metadata.backend_contract={MMAP_BUNDLE_SCHEMA}"
            )
        raw_bundle = metadata.get("bundle_manifest")
        if not isinstance(raw_bundle, str):
            raise StageBackendError("sealed data metadata lacks bundle_manifest")
        root = requirement.manifest_path.parent.resolve()
        bundle_path = _safe_relative(root, raw_bundle, label="bundle_manifest")
        sealed_files = {
            str(record["path"]): record
            for record in _require_list(envelope.get("files"), "sealed files")
            if isinstance(record, Mapping) and isinstance(record.get("path"), str)
        }
        try:
            relative_bundle = str(bundle_path.relative_to(root))
        except ValueError as exc:
            raise StageBackendError("bundle_manifest escaped its sealed root") from exc
        if relative_bundle not in sealed_files:
            raise StageBackendError("bundle_manifest is not sealed by the outer manifest")
        bundle = _read_json(bundle_path, label="post-training mmap bundle")
        if bundle.get("schema") != MMAP_BUNDLE_SCHEMA:
            raise StageBackendError(f"bundle schema must be {MMAP_BUNDLE_SCHEMA}")
        observed_hash = _canonical_hash(bundle, omit={"bundle_sha256"})
        if bundle.get("bundle_sha256") != observed_hash:
            raise StageBackendError("mmap bundle failed its self-hash")
        expected = {
            "stage": stage_id,
            "tokenizer_sha256": tokenizer_sha256,
        }
        for field, value in expected.items():
            if bundle.get(field) != value:
                raise StageBackendError(
                    f"mmap bundle {field} mismatch: {bundle.get(field)!r} != {value!r}"
                )
        bundle_family = bundle.get("family")
        if requirement.family_bound:
            if bundle_family != family:
                raise StageBackendError(
                    f"mmap bundle family mismatch: {bundle_family!r} != {family!r}"
                )
        elif bundle_family not in {family, "shared"}:
            raise StageBackendError(
                "unbound mmap bundle must declare the live family or family=shared"
            )
        bundle_parent = bundle.get("parent_checkpoint_sha256")
        if requirement.checkpoint_bound:
            if bundle_parent != parent_checkpoint_sha256:
                raise StageBackendError(
                    "checkpoint-bound mmap bundle does not match the live parent"
                )
        elif bundle_parent not in {None, "unbound", parent_checkpoint_sha256}:
            raise StageBackendError(
                "unbound mmap bundle carries an unrelated parent checkpoint"
            )
        if int(bundle.get("vocabulary_size", -1)) != vocabulary_size:
            raise StageBackendError("mmap bundle vocabulary size does not match Metis")
        if (
            canonical_id_lookup.shape != (vocabulary_size,)
            or canonical_id_lookup.dtype != np.dtype("<u2")
            or bundle.get("ngram_canonical_map_self_sha256")
            != canonical_map_self_sha256
            or bundle.get("ngram_canonical_ids_sha256")
            != canonical_ids_sha256
        ):
            raise StageBackendError(
                "mmap bundle canonical-ID sidecar lineage is invalid"
            )
        records = int(bundle.get("records", 0))
        sequence_length = int(bundle.get("sequence_length", 0))
        if records <= 0 or sequence_length <= 1:
            raise StageBackendError("mmap bundle records/sequence_length must be positive")
        if "records" in metadata and int(metadata["records"]) != records:
            raise StageBackendError(
                "outer sealed metadata.records differs from the mmap bundle"
            )
        if "tokens" in metadata:
            if int(bundle.get("training_tokens", -1)) != int(metadata["tokens"]):
                raise StageBackendError(
                    "outer sealed metadata.tokens differs from bundle.training_tokens"
                )
        training = _require_mapping(bundle.get("training"), "mmap bundle training")
        _validate_training_contract(training)
        profile_selection = bundle.get("profile_selection")
        live_autotune: Mapping[str, Any] | None = None
        live_evaluation_records: int | None = None
        if isinstance(profile_selection, Mapping) and isinstance(
            profile_selection.get("live_autotune"), Mapping
        ):
            live_autotune = _require_mapping(
                profile_selection["live_autotune"],
                "profile_selection.live_autotune",
            )
            evaluator = _require_mapping(
                live_autotune.get("evaluator"),
                "profile_selection.live_autotune.evaluator",
            )
            raw_evaluation_records = evaluator.get("records")
            if (
                isinstance(raw_evaluation_records, bool)
                or not isinstance(raw_evaluation_records, int)
                or raw_evaluation_records <= 0
            ):
                raise StageBackendError(
                    "live profile evaluator records must be a positive integer"
                )
            live_evaluation_records = raw_evaluation_records
        raw_arrays = _require_mapping(bundle.get("arrays"), "mmap bundle arrays")
        specs: dict[str, ArraySpec] = {}
        arrays: dict[str, np.ndarray] = {}
        for name, raw_spec in raw_arrays.items():
            spec = _require_mapping(raw_spec, f"arrays.{name}")
            raw_path = spec.get("path")
            raw_dtype = spec.get("dtype")
            raw_shape = spec.get("shape")
            if (
                not isinstance(raw_path, str)
                or not isinstance(raw_dtype, str)
                or not isinstance(raw_shape, list)
                or not raw_shape
            ):
                raise StageBackendError(f"arrays.{name} has an invalid specification")
            path = _safe_relative(root, raw_path, label=f"arrays.{name}.path")
            try:
                relative = str(path.relative_to(root))
            except ValueError as exc:
                raise StageBackendError(f"arrays.{name} escaped its sealed root") from exc
            if relative not in sealed_files:
                raise StageBackendError(f"arrays.{name} is not sealed by the outer manifest")
            if path.suffix != ".npy":
                raise StageBackendError(
                    f"arrays.{name} must be uncompressed .npy for true memory mapping"
                )
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(item) for item in raw_shape)
            dtype = np.dtype(raw_dtype)
            if array.shape != shape or array.dtype != dtype:
                raise StageBackendError(
                    f"arrays.{name} header differs from its sealed shape/dtype"
                )
            if array.dtype.hasobject:
                raise StageBackendError(f"arrays.{name} may not use object dtype")
            if str(name).startswith("autotune_evaluation_"):
                expected_records = live_evaluation_records
            elif str(name).startswith("context_evaluation_"):
                expected_records = int(
                    _require_mapping(
                        bundle.get("long_range_calibration"),
                        "context long_range_calibration",
                    ).get("sample_records", 0)
                )
            else:
                expected_records = records
            if expected_records is None or shape[0] != expected_records:
                expectation = (
                    "the sealed live-evaluation record count"
                    if str(name).startswith("autotune_evaluation_")
                    else "records"
                )
                raise StageBackendError(
                    f"arrays.{name} first dimension must equal {expectation}"
                )
            specs[str(name)] = ArraySpec(str(name), path, dtype.name, shape)
            arrays[str(name)] = array
        working_set = _require_mapping(
            bundle.get("working_set", {}),
            "mmap bundle working_set",
        )
        if (
            _is_gspo_stage(stage_id)
            or stage_id == "opd_consolidation"
        ):
            _validate_working_set_contract(working_set, stage_id=stage_id)
        teacher_distributions: dict[str, ChunkedTeacherDistribution] = {}
        raw_distributions = _require_mapping(
            bundle.get("teacher_distributions", {}),
            "mmap bundle teacher_distributions",
        )
        for distribution_name, raw_distribution in raw_distributions.items():
            distribution = _require_mapping(
                raw_distribution,
                f"teacher_distributions.{distribution_name}",
            )
            if (
                int(distribution.get("records", -1)) != records
                or int(distribution.get("tokens", -1)) != sequence_length - 1
                or int(distribution.get("vocabulary_size", -1)) != vocabulary_size
            ):
                raise StageBackendError(
                    f"teacher_distributions.{distribution_name} dimensions changed"
                )
            raw_chunks = _require_list(
                distribution.get("chunks"),
                f"teacher_distributions.{distribution_name}.chunks",
            )
            chunks: list[VocabChunk] = []
            next_vocab = 0
            for index, raw_chunk in enumerate(raw_chunks):
                chunk = _require_mapping(
                    raw_chunk,
                    f"teacher_distributions.{distribution_name}.chunks[{index}]",
                )
                start = int(chunk.get("vocab_start", -1))
                end = int(chunk.get("vocab_end", -1))
                if start != next_vocab or not start < end <= vocabulary_size:
                    raise StageBackendError(
                        f"{distribution_name} teacher vocab chunks must be contiguous"
                    )
                raw_path = chunk.get("path")
                if not isinstance(raw_path, str):
                    raise StageBackendError("teacher vocab chunk path must be a string")
                path = _safe_relative(
                    root,
                    raw_path,
                    label=f"teacher_distributions.{distribution_name}.chunk",
                )
                relative = str(path.relative_to(root))
                if relative not in sealed_files or path.suffix != ".npy":
                    raise StageBackendError(
                        f"{distribution_name} teacher chunk is not a sealed .npy file"
                    )
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                expected_shape = (records, sequence_length - 1, end - start)
                if (
                    values.shape != expected_shape
                    or not np.issubdtype(values.dtype, np.floating)
                ):
                    raise StageBackendError(
                        f"{distribution_name} teacher chunk has the wrong shape/dtype"
                    )
                chunks.append(VocabChunk(path, start, end, values))
                next_vocab = end
            if next_vocab != vocabulary_size:
                raise StageBackendError(
                    f"{distribution_name} teacher chunks do not cover the full vocabulary"
                )
            teacher_distributions[str(distribution_name)] = (
                ChunkedTeacherDistribution(
                    name=str(distribution_name),
                    records=records,
                    tokens=sequence_length - 1,
                    vocabulary_size=vocabulary_size,
                    chunks=tuple(chunks),
                )
            )
        loaded = cls(
            stage_id=stage_id,
            family=family,
            root=root,
            manifest=bundle,
            manifest_sha256=observed_hash,
            arrays=arrays,
            specs=specs,
            teacher_distributions=teacher_distributions,
            training=training,
            sealed_training=training,
            batch_migration_path=None,
            batch_migration_sha256=None,
            batch_migration_chain=(
                (
                    int(training["micro_batch_size"]),
                    int(training["gradient_accumulation"]),
                ),
            ),
            working_set_autotune_sha256=None,
            canonical_id_lookup=canonical_id_lookup,
            canonical_map_self_sha256=canonical_map_self_sha256,
            canonical_ids_sha256=canonical_ids_sha256,
            canonical_lookup_tensor=None,
            working_set=working_set,
            records=records,
            sequence_length=sequence_length,
        )
        loaded.validate_layout(vocabulary_size=vocabulary_size)
        loaded.validate_live_profile_autotune()
        if stage_id == "context_extension":
            unique_active_tokens = int(
                bundle.get("unique_active_tokens", -1)
            )
            epochs = int(training["epochs"])
            total_exposure = int(bundle.get("training_tokens", -1))
            if (
                unique_active_tokens <= 0
                or total_exposure <= 0
                or unique_active_tokens * epochs != total_exposure
            ):
                raise StageBackendError(
                    "context-extension bundle must seal unique_active_tokens "
                    "and exact total training exposure"
                )
        return loaded

    def validate_layout(self, *, vocabulary_size: int) -> None:
        compact_supervised = bool(
            self.stage_id == "context_extension"
            and self.manifest.get("compact_layout") == _COMPACT_CAUSAL_LAYOUT
        )
        if self.stage_id in {"context_extension", "cold_start_sft", "overall_sft"}:
            required = (
                _COMPACT_SUPERVISED_ARRAYS
                if compact_supervised
                else _SUPERVISED_ARRAYS
            )
        elif _is_gspo_stage(self.stage_id):
            required = _RLVR_ARRAYS
        elif self.stage_id == "opd_consolidation":
            required = _OPD_ARRAYS
        else:
            required = set()
        missing = sorted(required - set(self.arrays))
        if missing:
            raise StageBackendError(
                f"{self.stage_id} mmap bundle is missing arrays: {', '.join(missing)}"
            )
        if compact_supervised:
            self._require_shapes(
                {
                    "input_ids": (self.records, self.sequence_length),
                    "document_start": (self.records, self.sequence_length),
                    "sequence_lengths": (self.records,),
                    "gate_ids": (self.records,),
                }
            )
            self._require_integer("input_ids")
            self._require_mask("document_start")
            self._require_integer("sequence_lengths")
            self._require_integer("gate_ids")
            self._validate_compact_supervised_boundaries(
                vocabulary_size=vocabulary_size
            )
            self._validate_context_evaluation(
                vocabulary_size=vocabulary_size
            )
        elif self.stage_id in {"context_extension", "cold_start_sft", "overall_sft"}:
            shape = (self.records, self.sequence_length)
            self._require_shapes({name: shape for name in required})
            self._require_integer("input_ids")
            self._require_integer("labels")
            self._require_integer("document_ids")
            self._require_integer("canonical_ids")
            self._require_mask("loss_mask")
            self._require_mask("attention_mask")
            self._require_mask("reset_mask")
            self._validate_supervised_boundaries(vocabulary_size=vocabulary_size)
        elif _is_gspo_stage(self.stage_id):
            ids = self.arrays["candidate_input_ids"]
            if ids.ndim != 3 or ids.shape[0] != self.records or ids.shape[2] != self.sequence_length:
                raise StageBackendError("candidate_input_ids must have shape [N,G,T]")
            group = ids.shape[1]
            if group != 16:
                raise StageBackendError("GSPO bundles must contain exactly 16 candidates per prompt")
            token_shape = (self.records, group, self.sequence_length - 1)
            attention = self.arrays["candidate_attention_mask"]
            if attention.shape != ids.shape:
                raise StageBackendError("candidate_attention_mask has the wrong shape")
            if self.arrays["candidate_response_mask"].shape != token_shape:
                raise StageBackendError("candidate_response_mask has the wrong shape")
            if self.arrays["old_token_log_probs"].shape != token_shape:
                raise StageBackendError("old_token_log_probs has the wrong shape")
            if self.arrays["truncated"].shape != (self.records, group):
                raise StageBackendError("truncated has the wrong shape")
            self._require_integer("candidate_input_ids")
            self._require_mask("candidate_attention_mask")
            self._require_mask("candidate_response_mask")
            self._require_mask("truncated")
            self._require_floating("old_token_log_probs")
            if self.manifest.get("document_layout") != (
                "single_prompt_response_per_record"
            ):
                raise StageBackendError(
                    "GSPO bundles must prove one prompt-response document per candidate"
                )
            active = np.asarray(attention).astype(np.bool_, copy=False)
            response = np.asarray(
                self.arrays["candidate_response_mask"]
            ).astype(np.bool_, copy=False)
            if np.any(active[:, :, 1:] & ~active[:, :, :-1]) or np.any(
                response & ~(active[:, :, :-1] & active[:, :, 1:])
            ):
                raise StageBackendError(
                    "GSPO attention/response masks violate single-document padding"
                )
            self._validated_split_fingerprints(
                "split_fingerprint",
                records=self.records,
                expected=_rlvr_prompt_fingerprints(self.arrays),
            )
            if self.arrays["correctness"].shape != (self.records, group):
                raise StageBackendError("correctness has the wrong shape")
            if self.arrays["mode_compliance"].shape != (self.records, group):
                raise StageBackendError("mode_compliance has the wrong shape")
            self._require_floating("correctness")
            self._require_floating("mode_compliance")
            compliance = np.asarray(self.arrays["mode_compliance"])
            if (
                not np.isfinite(compliance).all()
                or np.any(compliance < 0)
                or np.any(compliance > 1)
            ):
                raise StageBackendError(
                    "mode_compliance must contain finite values in [0,1]"
                )
            modes = np.asarray(self.arrays["reasoning_mode"])
            overlap_ids = np.asarray(self.arrays["mode_overlap_id"])
            base_fingerprints = np.asarray(
                self.arrays["base_prompt_fingerprint"]
            )
            if (
                modes.shape != (self.records,)
                or overlap_ids.shape != (self.records,)
                or base_fingerprints.shape != (self.records, 32)
            ):
                raise StageBackendError(
                    "reasoning modes and mode-overlap fingerprints have invalid shapes"
                )
            self._require_integer("reasoning_mode")
            self._require_integer("mode_overlap_id")
            self._require_integer("base_prompt_fingerprint")
            if (
                set(np.unique(modes).tolist()) != {0, 1, 2}
                or np.any(overlap_ids < -1)
                or self.manifest.get("reasoning_mode_ids")
                != {"direct": 0, "think": 1, "think_max": 2}
            ):
                raise StageBackendError(
                    "GSPO bundles must explicitly contain direct, think, and think_max"
                )
            overlap_groups = [
                int(value) for value in np.unique(overlap_ids) if int(value) >= 0
            ]
            if not overlap_groups:
                raise StageBackendError(
                    "GSPO bundles must contain same-prompt three-mode overlap groups"
                )
            for overlap_id in overlap_groups:
                indices = np.flatnonzero(overlap_ids == overlap_id)
                if (
                    indices.size != len(REASONING_MODES)
                    or set(modes[indices].tolist()) != {0, 1, 2}
                    or not np.all(
                        base_fingerprints[indices]
                        == base_fingerprints[indices[0]]
                    )
                ):
                    raise StageBackendError(
                        "each mode-overlap group must bind one shared base prompt "
                        "to direct, think, and think_max exactly once"
                    )
            response_tokens = response.sum(axis=(1, 2), dtype=np.int64)
            total_response_tokens = int(response_tokens.sum())
            if total_response_tokens <= 0:
                raise StageBackendError("GSPO bundle contains no response tokens")
            observed_token_share = {
                mode: float(response_tokens[modes == index].sum())
                / total_response_tokens
                for index, mode in enumerate(REASONING_MODES)
            }
            declared_token_share = self.manifest.get(
                "response_token_share_by_mode"
            )
            if (
                not isinstance(declared_token_share, Mapping)
                or set(declared_token_share) != set(REASONING_MODES)
                or any(
                    not math.isclose(
                        float(declared_token_share[mode]),
                        observed_token_share[mode],
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                    for mode in REASONING_MODES
                )
                or self.manifest.get("target_token_share_by_mode_audited")
                is not True
            ):
                raise StageBackendError(
                    "GSPO bundle reasoning-mode token shares are not sealed exactly"
                )
            required_metadata = {
                "on_policy": True,
                "single_use_rollouts": True,
                "rollout_policy": "parent_checkpoint",
                "policy_updates_before_generation": 0,
                "samples_per_prompt": 16,
                "mode_overlap_validated": True,
            }
            for field, expected in required_metadata.items():
                if self.manifest.get(field) != expected:
                    raise StageBackendError(
                        f"true GSPO requires bundle {field}={expected!r}"
                    )
            if int(self.training.get("epochs", 0)) != 1:
                raise StageBackendError("on-policy rollout bundles are single-use")
            if self.stage_id == "specialist_code":
                efficiency = self.arrays.get("efficiency_reward")
                if (
                    efficiency is None
                    or efficiency.shape != (self.records, group)
                    or not np.issubdtype(efficiency.dtype, np.floating)
                ):
                    raise StageBackendError(
                        "code RLVR requires floating efficiency_reward [N,G]"
                    )
            if self.stage_id == "specialist_agentic":
                advantages = self.arrays.get("token_advantages")
                if (
                    advantages is None
                    or advantages.shape != token_shape
                    or not np.issubdtype(advantages.dtype, np.floating)
                ):
                    raise StageBackendError(
                        "agentic RLVR requires floating token_advantages [N,G,T-1]"
                    )
        elif self.stage_id == "opd_consolidation":
            self._validate_opd_layout(vocabulary_size=vocabulary_size)
    def _validate_opd_layout(self, *, vocabulary_size: int) -> None:
        ids = self.arrays["input_ids"]
        attention = self.arrays["attention_mask"]
        response = self.arrays["response_mask"]
        union_ids = self.arrays["teacher_union_token_ids"]
        union_logits = self.arrays["teacher_union_logits"]
        union_count = self.arrays["teacher_union_count"]
        routes = self.arrays["teacher_route"]
        reasoning_modes = self.arrays["reasoning_mode"]
        if (
            ids.shape != (self.records, self.sequence_length)
            or attention.shape != ids.shape
            or response.shape != (self.records, self.sequence_length - 1)
            or union_ids.ndim != 3
            or union_ids.shape[:2] != response.shape
            or union_logits.shape != union_ids.shape
            or union_count.shape != response.shape
            or routes.shape != (self.records,)
            or reasoning_modes.shape != (self.records,)
            or union_ids.shape[-1] != 64
        ):
            raise StageBackendError("OPD top-k-union arrays have invalid shapes")
        self._require_integer("input_ids")
        self._require_mask("attention_mask")
        self._require_mask("response_mask")
        self._require_integer("teacher_union_token_ids")
        self._require_floating("teacher_union_logits")
        self._require_integer("teacher_union_count")
        self._require_integer("teacher_route")
        self._require_integer("reasoning_mode")
        active = np.asarray(attention).astype(np.bool_, copy=False)
        response_bool = np.asarray(response).astype(np.bool_, copy=False)
        counts = np.asarray(union_count)
        routes_array = np.asarray(routes)
        reasoning_mode_array = np.asarray(reasoning_modes)
        if (
            np.any(active[:, 1:] & ~active[:, :-1])
            or np.any(response_bool & ~(active[:, :-1] & active[:, 1:]))
            or np.any(response_bool & ((counts < 1) | (counts > 64)))
            or np.any(~response_bool & (counts != 0))
            or np.any(routes_array < 0)
            or np.any(routes_array >= len(SPECIALIST_STAGE_IDS))
            or set(np.unique(routes_array).tolist())
            != set(range(len(SPECIALIST_STAGE_IDS)))
            or set(np.unique(reasoning_mode_array).tolist()) != {0, 1, 2}
            or self.manifest.get("reasoning_mode_ids")
            != {"direct": 0, "think": 1, "think_max": 2}
        ):
            raise StageBackendError(
                "OPD masks, union counts, specialist routing, or reasoning modes are invalid"
            )
        rows_per_chunk = max(1, 1_000_000 // self.sequence_length)
        for start in range(0, self.records, rows_per_chunk):
            end = min(self.records, start + rows_per_chunk)
            local_ids = np.asarray(ids[start:end])
            local_active = active[start:end]
            local_union = np.asarray(union_ids[start:end])
            local_logits = np.asarray(union_logits[start:end])
            local_counts = counts[start:end]
            if (
                np.any(local_ids[local_active] < 0)
                or np.any(local_ids[local_active] >= vocabulary_size)
                or not np.isfinite(local_logits).all()
            ):
                raise StageBackendError(
                    "OPD input IDs or teacher union logits are invalid"
                )
            if np.any(local_union < 0) or np.any(
                local_union >= vocabulary_size
            ):
                raise StageBackendError(
                    "OPD teacher-union token IDs escape the vocabulary"
                )
        specialist_checkpoints = self.manifest.get("specialist_checkpoints")
        if (
            self.manifest.get("on_policy") is not True
            or self.manifest.get("single_use_rollouts") is not True
            or self.manifest.get("rollout_policy") != "parent_checkpoint"
            or self.manifest.get("policy_updates_before_generation") != 0
            or int(self.training["epochs"]) != 1
            or int(self.manifest.get("top_k_per_model", -1)) != 32
            or self.manifest.get("union_student_and_teacher_top_k") is not True
            or self.manifest.get("prompt_mode_preserved_for_teacher_prefill")
            is not True
            or self.manifest.get("domain_mode_target_token_share_audited")
            is not True
            or not isinstance(specialist_checkpoints, Mapping)
            or set(specialist_checkpoints) != set(SPECIALIST_STAGE_IDS)
            or not all(
                _is_sha256(value) for value in specialist_checkpoints.values()
            )
        ):
            raise StageBackendError(
                "OPD bundle is not a single-use, checkpoint-bound top-k union"
            )
        self._validated_split_fingerprints(
            "split_fingerprint", records=self.records
        )

    def _require_shapes(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape:
                raise StageBackendError(f"{name} has shape {self.arrays[name].shape}, expected {shape}")

    def _validate_supervised_boundaries(self, *, vocabulary_size: int) -> None:
        rows_per_chunk = max(1, 1_000_000 // self.sequence_length)
        active_tokens = 0
        for start in range(0, self.records, rows_per_chunk):
            end = min(self.records, start + rows_per_chunk)
            input_ids = np.asarray(self.arrays["input_ids"][start:end])
            labels = np.asarray(self.arrays["labels"][start:end])
            loss_mask = np.asarray(self.arrays["loss_mask"][start:end]).astype(
                np.bool_, copy=False
            )
            attention = np.asarray(
                self.arrays["attention_mask"][start:end]
            ).astype(np.bool_, copy=False)
            document_ids = np.asarray(self.arrays["document_ids"][start:end])
            reset = np.asarray(self.arrays["reset_mask"][start:end]).astype(
                np.bool_, copy=False
            )
            canonical = np.asarray(self.arrays["canonical_ids"][start:end])
            if np.any(attention[:, 1:] & ~attention[:, :-1]):
                raise StageBackendError(
                    "supervised attention_mask must be a contiguous active prefix"
                )
            if np.any(loss_mask & ~attention) or np.any(labels[~attention] != -100):
                raise StageBackendError(
                    "supervised padding must have loss_mask=false and labels=-100"
                )
            if np.any(reset & ~attention):
                raise StageBackendError("reset_mask may not reset a padding token")
            if np.any(document_ids[~attention] != -1) or np.any(
                document_ids[attention] < 0
            ):
                raise StageBackendError(
                    "document_ids must be non-negative on tokens and -1 on padding"
                )
            expected_reset = np.zeros_like(attention)
            expected_reset[:, 0] = attention[:, 0]
            expected_reset[:, 1:] = attention[:, 1:] & (
                document_ids[:, 1:] != document_ids[:, :-1]
            )
            if not np.array_equal(reset, expected_reset):
                raise StageBackendError(
                    "reset_mask must exactly mark every packed-document boundary"
                )
            if np.any(input_ids[attention] < 0) or np.any(
                input_ids[attention] >= vocabulary_size
            ):
                raise StageBackendError("supervised input_ids escape the vocabulary")
            if np.any(canonical[attention] < 0) or np.any(
                canonical[attention] >= vocabulary_size
            ):
                raise StageBackendError(
                    "supervised canonical_ids escape the vocabulary"
                )
            if np.any(canonical[~attention] != 0):
                raise StageBackendError(
                    "supervised canonical padding must use canonical ID zero"
                )
            if not np.array_equal(
                canonical[attention],
                self.canonical_id_lookup[input_ids[attention]],
            ):
                raise StageBackendError(
                    "supervised canonical_ids differ from the verified tokenizer map"
                )
            supervised_labels = labels[(labels != -100) & attention]
            if np.any(supervised_labels < 0) or np.any(
                supervised_labels >= vocabulary_size
            ):
                raise StageBackendError("supervised labels escape the vocabulary")
            active_tokens += int(np.count_nonzero(attention))
        unique_active_tokens = self.manifest.get("unique_active_tokens")
        if self.stage_id == "context_extension" and unique_active_tokens is None:
            raise StageBackendError(
                "context extension must seal unique_active_tokens"
            )
        if (
            unique_active_tokens is not None
            and int(unique_active_tokens) != active_tokens
        ):
            raise StageBackendError(
                "unique_active_tokens must exactly equal attention_mask active tokens"
            )
        total_exposure = active_tokens * int(self.training["epochs"])
        declared_tokens = self.manifest.get("training_tokens")
        if declared_tokens is not None and int(declared_tokens) != total_exposure:
            raise StageBackendError(
                "training_tokens must exactly equal unique active tokens times epochs"
            )

    def _validate_compact_supervised_boundaries(
        self,
        *,
        vocabulary_size: int,
    ) -> None:
        lengths = np.asarray(self.arrays["sequence_lengths"], dtype=np.int64)
        gate_ids = np.asarray(self.arrays["gate_ids"], dtype=np.int64)
        if (
            np.any(lengths <= 1)
            or np.any(lengths > self.sequence_length)
            or np.any(gate_ids < 0)
            or np.any(gate_ids > 2)
        ):
            raise StageBackendError(
                "compact context lengths or checkpoint-gate IDs are invalid"
            )
        active_tokens = int(lengths.sum())
        declared_unique = int(self.manifest.get("unique_active_tokens", -1))
        declared_training = int(self.manifest.get("training_tokens", -1))
        epochs = int(self.training["epochs"])
        if (
            declared_unique != active_tokens
            or declared_training != active_tokens * epochs
        ):
            raise StageBackendError(
                "compact context active-token accounting is not exact"
            )
        rows_per_chunk = max(1, 1_000_000 // self.sequence_length)
        for start in range(0, self.records, rows_per_chunk):
            end = min(self.records, start + rows_per_chunk)
            ids = np.asarray(self.arrays["input_ids"][start:end])
            starts = np.asarray(
                self.arrays["document_start"][start:end]
            ).astype(np.bool_, copy=False)
            local_lengths = lengths[start:end]
            positions = np.arange(self.sequence_length)[None, :]
            active = positions < local_lengths[:, None]
            if (
                np.any(ids[active] < 0)
                or np.any(ids[active] >= vocabulary_size)
                or np.any(starts & ~active)
                or not np.all(starts[:, 0])
            ):
                raise StageBackendError(
                    "compact context IDs or document boundaries are invalid"
                )
        observed_gate_tokens = [
            int(lengths[gate_ids == gate].sum()) for gate in range(3)
        ]
        checkpoints = tuple(
            int(value) for value in self.manifest.get("checkpoint_gates", ())
        )
        if checkpoints:
            expected_tranches = [
                checkpoints[index] - (checkpoints[index - 1] if index else 0)
                for index in range(len(checkpoints))
            ]
            if observed_gate_tokens != expected_tranches:
                raise StageBackendError(
                    "compact context records do not exactly match checkpoint tranches"
                )

    def _validate_context_evaluation(
        self,
        *,
        vocabulary_size: int,
    ) -> None:
        calibration = _require_mapping(
            self.manifest.get("long_range_calibration"),
            "context long_range_calibration",
        )
        records = int(calibration.get("sample_records", 0))
        context = int(calibration.get("context", 0))
        if (
            calibration.get("required") is not True
            or calibration.get("implementation")
            != "metis.long-range-information/v1"
            or calibration.get("score")
            != "full_context_nll_gain_over_4096"
            or calibration.get("training_disjoint") is not True
            or records != 384
            or context != 131_072
            or int(calibration.get("tail_tokens", 0)) != 4_096
            or not _is_sha256(
                calibration.get("evaluation_pack_receipt_sha256")
            )
            or not _CONTEXT_EVALUATION_ARRAYS.issubset(self.arrays)
        ):
            raise StageBackendError(
                "context bundle omits its disjoint model-calibration set"
            )
        inputs = self.arrays["context_evaluation_input_ids"]
        targets = self.arrays[
            "context_evaluation_probe_target_ids"
        ]
        positions = self.arrays[
            "context_evaluation_probe_positions"
        ]
        fingerprints = self.arrays[
            "context_evaluation_split_fingerprint"
        ]
        if (
            inputs.shape != (records, context)
            or targets.shape != (records,)
            or positions.shape != (records,)
            or fingerprints.shape != (records, 4)
            or not np.issubdtype(inputs.dtype, np.integer)
            or not np.issubdtype(targets.dtype, np.integer)
            or not np.issubdtype(positions.dtype, np.integer)
            or not np.issubdtype(fingerprints.dtype, np.integer)
            or np.any(inputs < 0)
            or np.any(inputs >= vocabulary_size)
            or np.any(targets < 0)
            or np.any(targets >= vocabulary_size)
            or np.any(positions != context - 1)
            or len(
                {
                    tuple(int(value) for value in row)
                    for row in np.asarray(fingerprints)
                }
            )
            != records
        ):
            raise StageBackendError(
                "context model-calibration arrays are invalid"
            )

    def _require_integer(self, name: str) -> None:
        if not np.issubdtype(self.arrays[name].dtype, np.integer):
            raise StageBackendError(f"{name} must use an integer dtype")

    def _require_floating(self, name: str) -> None:
        if not np.issubdtype(self.arrays[name].dtype, np.floating):
            raise StageBackendError(f"{name} must use a floating dtype")

    def _require_mask(self, name: str) -> None:
        if self.arrays[name].dtype != np.bool_ and not np.issubdtype(
            self.arrays[name].dtype, np.integer
        ):
            raise StageBackendError(f"{name} must use bool or an integer mask dtype")

    def _validated_split_fingerprints(
        self,
        name: str,
        *,
        records: int,
        expected: np.ndarray | None = None,
    ) -> np.ndarray:
        array = np.asarray(self.arrays.get(name))
        if (
            array.shape != (records, 32)
            or array.dtype != np.uint8
            or np.unique(array, axis=0).shape[0] != records
        ):
            raise StageBackendError(
                f"{name} must contain {records} unique SHA-256 fingerprint rows"
            )
        if expected is not None and not np.array_equal(array, expected):
            raise StageBackendError(
                f"{name} differs from canonical prompt/sample content hashes"
            )
        return np.ascontiguousarray(array)

    def validate_live_profile_autotune(self) -> None:
        """Validate disjoint held-out replay data for every GSPO stage."""

        if not _is_gspo_stage(self.stage_id):
            return
        if self.manifest.get("profile_selection") is None:
            # Low-level layout diagnostics may load before materialization has
            # selected a live profile. The production campaign fails closed.
            return
        evidence = _require_mapping(
            self.manifest.get("profile_selection"),
            f"{self.stage_id} profile_selection",
        )
        live = _require_mapping(
            evidence.get("live_autotune"),
            f"{self.stage_id} profile_selection.live_autotune",
        )
        evaluator = _require_mapping(
            live.get("evaluator"),
            f"{self.stage_id} live evaluator",
        )
        steps = live.get("training_optimizer_steps")
        tolerance = evaluator.get("reproduction_tolerance")
        if (
            live.get("schema") != LIVE_PROFILE_AUTOTUNE_SCHEMA
            or live.get("stage") != self.stage_id
            or live.get("live_autotune_sha256")
            != _canonical_hash(live, omit={"live_autotune_sha256"})
            or evaluator.get("schema") != LIVE_PROFILE_EVALUATOR_SCHEMA
            or evaluator.get("implementation")
            != "metis.rlvr-offline-policy-replay/v1"
            or evaluator.get("evaluator_sha256")
            != _canonical_hash(evaluator, omit={"evaluator_sha256"})
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not 0.0 <= float(tolerance) <= 0.1
        ):
            raise StageBackendError(
                f"{self.stage_id} live profile-autotune contract is invalid"
            )
        records = int(evaluator.get("records", -1))
        if records <= 0:
            raise StageBackendError(
                f"{self.stage_id} live evaluator has no held-out records"
            )
        if evaluator.get("dataset_sha256") != _canonical_hash(
            {
                "stage": self.stage_id,
                "records": records,
                "arrays": evaluator.get("array_sha256"),
            }
        ):
            raise StageBackendError(
                f"{self.stage_id} live evaluation dataset binding is invalid"
            )

        prefix = "autotune_evaluation_"
        evaluation_fingerprint = f"{prefix}split_fingerprint"
        required = {
            evaluation_fingerprint,
            f"{prefix}candidate_input_ids",
            f"{prefix}candidate_attention_mask",
            f"{prefix}candidate_response_mask",
            f"{prefix}correctness",
            f"{prefix}mode_compliance",
            f"{prefix}reasoning_mode",
            f"{prefix}truncated",
        }
        if self.stage_id == "specialist_code":
            required.add(f"{prefix}efficiency_reward")
        hashes = _require_mapping(
            evaluator.get("array_sha256"),
            f"{self.stage_id} live evaluator array_sha256",
        )
        if set(hashes) != required or not required.issubset(self.arrays):
            raise StageBackendError(
                f"{self.stage_id} live evaluator must seal its exact array set"
            )
        for name in sorted(required):
            expected = hashes.get(name)
            if (
                not _is_sha256(expected)
                or sha256_file(self.specs[name].path) != expected
            ):
                raise StageBackendError(
                    f"{self.stage_id} live evaluator array hash changed: {name}"
                )
        training_fingerprints = self._validated_split_fingerprints(
            "split_fingerprint",
            records=self.records,
            expected=_rlvr_prompt_fingerprints(self.arrays),
        )
        evaluation_fingerprints = self._validated_split_fingerprints(
            evaluation_fingerprint,
            records=records,
            expected=_rlvr_prompt_fingerprints(self.arrays, prefix=prefix),
        )
        overlap = {
            row.tobytes() for row in training_fingerprints
        } & {
            row.tobytes() for row in evaluation_fingerprints
        }
        if overlap:
            raise StageBackendError(
                f"{self.stage_id} live evaluator overlaps the training split "
                f"on {len(overlap)} sealed prompt fingerprints"
            )

        ids = np.asarray(self.arrays[f"{prefix}candidate_input_ids"])
        attention = np.asarray(
            self.arrays[f"{prefix}candidate_attention_mask"]
        )
        response = np.asarray(
            self.arrays[f"{prefix}candidate_response_mask"]
        )
        correctness = np.asarray(self.arrays[f"{prefix}correctness"])
        compliance = np.asarray(self.arrays[f"{prefix}mode_compliance"])
        modes = np.asarray(self.arrays[f"{prefix}reasoning_mode"])
        truncated = np.asarray(self.arrays[f"{prefix}truncated"])
        expected_tokens = (records, 16, self.sequence_length)
        if (
            ids.shape != expected_tokens
            or attention.shape != expected_tokens
            or response.shape != (records, 16, self.sequence_length - 1)
            or correctness.shape != (records, 16)
            or compliance.shape != (records, 16)
            or modes.shape != (records,)
            or truncated.shape != (records, 16)
            or set(np.unique(modes).tolist()) != {0, 1, 2}
            or not np.isfinite(correctness).all()
            or not np.isfinite(compliance).all()
            or np.any(correctness < 0)
            or np.any(correctness > 1)
            or np.any(compliance < 0)
            or np.any(compliance > 1)
            or np.any(
                response.astype(bool)
                & ~(
                    attention[:, :, :-1].astype(bool)
                    & attention[:, :, 1:].astype(bool)
                )
            )
            or np.any(response.sum(axis=-1) <= 0)
        ):
            raise StageBackendError(
                f"{self.stage_id} live evaluator arrays are invalid"
            )
        pass_rates = correctness.astype(np.float64).mean(axis=1)
        if np.any(pass_rates < 0.10) or np.any(pass_rates > 0.90):
            raise StageBackendError(
                f"{self.stage_id} live evaluator violates strict avg@16 filtering"
            )
        if self.stage_id == "specialist_code":
            efficiency = np.asarray(
                self.arrays[f"{prefix}efficiency_reward"]
            )
            if efficiency.shape != (records, 16) or not np.isfinite(
                efficiency
            ).all():
                raise StageBackendError(
                    "RLVR-code live efficiency rewards are invalid"
                )
    def iter_rank_batches(
        self,
        topology: ParallelTopology,
        *,
        start_epoch: int = 0,
        start_global_batch: int = 0,
    ) -> Iterator[tuple[int, int, dict[str, np.ndarray]]]:
        for epoch, global_batch_index, indices in self.iter_rank_indices(
            topology,
            start_epoch=start_epoch,
            start_global_batch=start_global_batch,
        ):
            batch = self.materialize_batch(indices)
            yield epoch, global_batch_index, batch

    def materialize_batch(
        self,
        indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not (
            self.stage_id == "context_extension"
            and self.manifest.get("compact_layout") == _COMPACT_CAUSAL_LAYOUT
        ):
            return {
                name: np.array(array[indices], copy=True)
                for name, array in self.arrays.items()
            }
        input_ids = np.array(self.arrays["input_ids"][indices], copy=True)
        starts = np.array(
            self.arrays["document_start"][indices], dtype=np.bool_, copy=True
        )
        lengths = np.array(
            self.arrays["sequence_lengths"][indices],
            dtype=np.int64,
            copy=True,
        )
        positions = np.arange(self.sequence_length)[None, :]
        attention = positions < lengths[:, None]
        starts &= attention
        if not np.all(starts[:, 0]):
            raise StageBackendError(
                "compact context batch lost its leading document boundary"
            )
        document_ids = np.cumsum(starts, axis=1, dtype=np.int32) - 1
        document_ids[~attention] = -1
        labels = input_ids.astype(np.int32)
        labels[~attention] = -100
        canonical = self.canonical_id_lookup[input_ids]
        canonical = np.asarray(canonical, dtype=np.dtype("<u2"))
        canonical[~attention] = 0
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": attention.copy(),
            "attention_mask": attention,
            "document_ids": document_ids,
            "reset_mask": starts,
            "canonical_ids": canonical,
            "sequence_lengths": lengths,
            "gate_ids": np.array(
                self.arrays["gate_ids"][indices], copy=True
            ),
        }

    def iter_rank_indices(
        self,
        topology: ParallelTopology,
        *,
        start_epoch: int = 0,
        start_global_batch: int = 0,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        micro_batch = int(self.training["micro_batch_size"])
        global_batch = micro_batch * topology.world_size
        if self.records % global_batch:
            raise StageBackendError(
                f"{self.stage_id} records must be divisible by world_size*micro_batch "
                "so no distributed rank can run short"
            )
        batches_per_epoch = self.records // global_batch
        epochs = int(self.training["epochs"])
        seed = int(self.training["shuffle_seed"])
        for epoch in range(start_epoch, epochs):
            generator = np.random.Generator(np.random.PCG64(seed + epoch))
            order = generator.permutation(self.records)
            first_batch = start_global_batch if epoch == start_epoch else 0
            if not 0 <= first_batch <= batches_per_epoch:
                raise StageBackendError("resume global batch is outside this mmap bundle")
            for global_batch_index in range(first_batch, batches_per_epoch):
                offset = (
                    global_batch_index * global_batch
                    + topology.rank * micro_batch
                )
                indices = order[offset : offset + micro_batch]
                yield epoch, global_batch_index, indices


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBackendError(f"{label} must be a mapping")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise StageBackendError(f"{label} must be a list")
    return value


def _validate_training_contract(training: Mapping[str, Any]) -> None:
    integer_fields = (
        "epochs",
        "micro_batch_size",
        "gradient_accumulation",
        "shuffle_seed",
        "checkpoint_interval_steps",
    )
    for field in integer_fields:
        value = training.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise StageBackendError(f"training.{field} must be an integer")
        if field != "checkpoint_interval_steps" and value <= 0:
            raise StageBackendError(f"training.{field} must be positive")
        if field == "checkpoint_interval_steps" and value < 0:
            raise StageBackendError("training.checkpoint_interval_steps cannot be negative")
    for field in ("learning_rate", "minimum_learning_rate_ratio", "gradient_clip"):
        value = training.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StageBackendError(f"training.{field} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise StageBackendError(f"training.{field} must be finite and positive")
    warmup = training.get("warmup_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise StageBackendError("training.warmup_steps must be a non-negative integer")


def _validate_working_set_contract(
    working_set: Mapping[str, Any],
    *,
    stage_id: str,
) -> None:
    for field in ("token_chunk_size", "maximum_device_bytes", "maximum_host_bytes"):
        value = working_set.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageBackendError(
                f"{stage_id} working_set.{field} must be a positive integer"
            )
    fraction = working_set.get("headroom_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0.05 <= float(fraction) <= 0.5
    ):
        raise StageBackendError(
            f"{stage_id} working_set.headroom_fraction must be in [0.05,0.5]"
        )
    if _is_gspo_stage(stage_id):
        micro_group = working_set.get("candidate_micro_group_size")
        if (
            isinstance(micro_group, bool)
            or not isinstance(micro_group, int)
            or not 1 <= micro_group <= 16
            or 16 % micro_group
        ):
            raise StageBackendError(
                f"{stage_id} candidate_micro_group_size must divide 16"
            )
    raw_autotune = working_set.get("autotune")
    if raw_autotune is not None:
        autotune = _require_mapping(
            raw_autotune, f"{stage_id} working_set.autotune"
        )
        warmups = autotune.get("warmup_trials")
        measurements = autotune.get("measurement_trials")
        candidates = _require_list(
            autotune.get("candidates"),
            f"{stage_id} working_set.autotune.candidates",
        )
        if (
            autotune.get("schema") != WORKING_SET_AUTOTUNE_SCHEMA
            or autotune.get("contract_sha256")
            != _canonical_hash(autotune, omit={"contract_sha256"})
            or isinstance(warmups, bool)
            or not isinstance(warmups, int)
            or not 1 <= warmups <= 3
            or isinstance(measurements, bool)
            or not isinstance(measurements, int)
            or not 2 <= measurements <= 7
            or not candidates
            or len(candidates) > 12
        ):
            raise StageBackendError(
                f"{stage_id} working-set autotune contract is invalid"
            )
        require_group = _is_gspo_stage(stage_id)
        expected_fields = {
            "micro_batch_size",
            "token_chunk_size",
        } | ({"candidate_micro_group_size"} if require_group else set())
        observed: set[str] = set()
        for index, raw_candidate in enumerate(candidates):
            candidate = _require_mapping(
                raw_candidate,
                f"{stage_id} working-set candidate {index}",
            )
            if set(candidate) != expected_fields:
                raise StageBackendError(
                    f"{stage_id} working-set candidate fields are invalid"
                )
            for name, value in candidate.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    raise StageBackendError(
                        f"{stage_id} working-set candidate {name} must be positive"
                    )
            if require_group and 16 % int(
                candidate["candidate_micro_group_size"]
            ):
                raise StageBackendError(
                    f"{stage_id} working-set candidate group must divide 16"
                )
            fingerprint = _canonical_hash(candidate)
            if fingerprint in observed:
                raise StageBackendError(
                    f"{stage_id} working-set candidates must be unique"
                )
            observed.add(fingerprint)


def _runtime_batch_payload(bundle: MMapStageBundle) -> dict[str, Any]:
    return {
        "micro_batch_size": int(bundle.training["micro_batch_size"]),
        "gradient_accumulation": int(bundle.training["gradient_accumulation"]),
        "effective_local_batch_records": (
            int(bundle.training["micro_batch_size"])
            * int(bundle.training["gradient_accumulation"])
        ),
        "migration_receipt_sha256": bundle.batch_migration_sha256,
        "working_set_autotune_receipt_sha256": (
            bundle.working_set_autotune_sha256
        ),
        "token_chunk_size": (
            int(bundle.working_set["token_chunk_size"])
            if bundle.working_set
            else None
        ),
        "candidate_micro_group_size": (
            int(bundle.working_set["candidate_micro_group_size"])
            if "candidate_micro_group_size" in bundle.working_set
            else None
        ),
    }


def _resume_global_batch(
    active: Mapping[str, Any],
    bundle: MMapStageBundle,
) -> int:
    runtime_batch = _require_mapping(
        active.get("runtime_batch"), "active stage runtime_batch"
    )
    old_micro = int(runtime_batch.get("micro_batch_size", -1))
    old_accumulation = int(runtime_batch.get("gradient_accumulation", -1))
    new_micro = int(bundle.training["micro_batch_size"])
    new_accumulation = int(bundle.training["gradient_accumulation"])
    old_batch = int(active["next_global_batch"])
    if (
        old_micro <= 0
        or old_accumulation <= 0
        or old_micro * old_accumulation != new_micro * new_accumulation
        or new_micro > old_micro
        or (old_micro, old_accumulation) not in bundle.batch_migration_chain
        or bundle.batch_migration_chain[-1] != (new_micro, new_accumulation)
    ):
        raise StageBackendError(
            "active checkpoint runtime batch is not migration-compatible"
        )
    numerator = old_batch * old_micro
    if numerator % new_micro:
        raise StageBackendError(
            "active checkpoint record cursor cannot map to the revised micro-batch"
        )
    return numerator // new_micro


def _load_stage_batch_migration(
    bundle: MMapStageBundle,
    *,
    family: str,
    parent_checkpoint_sha256: str,
    precision_role_plan_sha256: str,
    output_root: Path,
    topology: ParallelTopology,
) -> MMapStageBundle:
    raw_root = os.environ.get(
        f"METIS_POSTTRAINING_BATCH_MIGRATION_ROOT_{family.upper()}"
    ) or os.environ.get("METIS_POSTTRAINING_BATCH_MIGRATION_ROOT")
    if not raw_root:
        return bundle
    root = Path(raw_root).expanduser().resolve()
    path = root / family / f"{bundle.stage_id}.json"
    if not path.exists():
        return bundle

    def validate() -> tuple[dict[str, Any], str, tuple[tuple[int, int], ...]]:
        receipt = _read_json(path, label="post-training batch migration")
        if (
            receipt.get("schema") != BATCH_MIGRATION_SCHEMA
            or receipt.get("family") != family
            or receipt.get("stage") != bundle.stage_id
            or receipt.get("parent_checkpoint_sha256")
            != parent_checkpoint_sha256
            or receipt.get("precision_role_plan_sha256")
            != precision_role_plan_sha256
            or receipt.get("bundle_manifest_sha256")
            != bundle.manifest_sha256
            or receipt.get("receipt_sha256")
            != _canonical_hash(receipt, omit={"receipt_sha256"})
        ):
            raise StageBackendError(
                f"{bundle.stage_id} batch migration lineage is invalid"
            )
        sealed = _require_mapping(
            receipt.get("sealed_training"),
            f"{bundle.stage_id} batch migration sealed_training",
        )
        sealed_pair = (
            int(bundle.sealed_training["micro_batch_size"]),
            int(bundle.sealed_training["gradient_accumulation"]),
        )
        if (
            set(sealed) != {"micro_batch_size", "gradient_accumulation"}
            or int(sealed.get("micro_batch_size", -1)) != sealed_pair[0]
            or int(sealed.get("gradient_accumulation", -1)) != sealed_pair[1]
        ):
            raise StageBackendError(
                f"{bundle.stage_id} batch migration changed sealed training"
            )
        effective_batch = sealed_pair[0] * sealed_pair[1]
        prior = sealed_pair
        chain = [sealed_pair]
        revisions = _require_list(
            receipt.get("revisions"),
            f"{bundle.stage_id} batch migration revisions",
        )
        if not revisions:
            raise StageBackendError("batch migration must contain a measured OOM revision")
        oom_root = (output_root / "oom").resolve()
        for index, raw_revision in enumerate(revisions):
            revision = _require_mapping(
                raw_revision,
                f"{bundle.stage_id} batch migration revisions[{index}]",
            )
            if (
                revision.get("revision_sha256")
                != _canonical_hash(revision, omit={"revision_sha256"})
                or revision.get("reason") != "measured_stage_oom"
            ):
                raise StageBackendError("batch migration revision failed its self-hash")
            old = _require_mapping(revision.get("old"), "batch migration old")
            new = _require_mapping(revision.get("new"), "batch migration new")
            old_pair = (
                int(old.get("micro_batch_size", -1)),
                int(old.get("gradient_accumulation", -1)),
            )
            new_pair = (
                int(new.get("micro_batch_size", -1)),
                int(new.get("gradient_accumulation", -1)),
            )
            if (
                old_pair != prior
                or new_pair[0] <= 0
                or new_pair[1] <= 0
                or new_pair[0] >= old_pair[0]
                or new_pair[1] <= old_pair[1]
                or new_pair[0] * new_pair[1] != effective_batch
            ):
                raise StageBackendError(
                    "batch migration must strictly lower micro-batch and preserve "
                    "the exact effective local batch"
                )
            raw_request = revision.get("oom_request_path")
            if not isinstance(raw_request, str):
                raise StageBackendError("batch migration omits its OOM request path")
            request_path = Path(raw_request).expanduser().resolve()
            try:
                request_path.relative_to(oom_root)
            except ValueError as exc:
                raise StageBackendError(
                    "batch migration OOM request escapes the campaign output"
                ) from exc
            request = _read_json(
                request_path,
                label=f"{bundle.stage_id} measured OOM request",
            )
            if (
                sha256_file(request_path) != revision.get("oom_request_file_sha256")
                or request.get("request_sha256")
                != revision.get("oom_request_sha256")
                or request.get("request_sha256")
                != _canonical_hash(request, omit={"request_sha256"})
                or request.get("schema") != OOM_REVISION_REQUEST_SCHEMA
                or request.get("family") != family
                or request.get("stage") != bundle.stage_id
                or request.get("parent_checkpoint_sha256")
                != parent_checkpoint_sha256
                or request.get("bundle_manifest_sha256")
                != bundle.manifest_sha256
                or not isinstance(request.get("slurm_job_id"), str)
                or not request.get("slurm_job_id")
                or isinstance(request.get("slurm_restart_count"), bool)
                or not isinstance(request.get("slurm_restart_count"), int)
                or int(request.get("slurm_restart_count", -1)) < 0
                or not isinstance(request.get("resume"), Mapping)
                or request.get("prior_batch_migration_sha256")
                != revision.get("prior_batch_migration_sha256")
                or request.get("current") != dict(old)
                or request.get("proposed") != dict(new)
                or request.get("revision_available") is not True
            ):
                raise StageBackendError(
                    "batch migration is not bound to its exact measured OOM request"
                )
            prior = new_pair
            chain.append(new_pair)
        if int(receipt.get("effective_local_batch_records", -1)) != effective_batch:
            raise StageBackendError("batch migration effective batch declaration changed")
        training = dict(bundle.sealed_training)
        training["micro_batch_size"] = prior[0]
        training["gradient_accumulation"] = prior[1]
        _validate_training_contract(training)
        return training, str(receipt["receipt_sha256"]), tuple(chain)

    training, receipt_sha, chain = _collective_errors(
        topology,
        validate,
        label=f"{bundle.stage_id} batch-migration validation",
    )
    return dataclasses.replace(
        bundle,
        training=training,
        batch_migration_path=path,
        batch_migration_sha256=receipt_sha,
        batch_migration_chain=chain,
    )


def _proposed_batch_revision(bundle: MMapStageBundle) -> dict[str, int] | None:
    current_micro = int(bundle.training["micro_batch_size"])
    current_accumulation = int(bundle.training["gradient_accumulation"])
    effective = current_micro * current_accumulation
    for micro_batch in range(current_micro - 1, 0, -1):
        if effective % micro_batch == 0:
            return {
                "micro_batch_size": micro_batch,
                "gradient_accumulation": effective // micro_batch,
            }
    return None


def _write_stage_oom_request(
    *,
    output_root: Path,
    family: str,
    stage_id: str,
    parent_checkpoint_sha256: str,
    precision_role_plan_sha256: str,
    bundle: MMapStageBundle,
    topology: ParallelTopology,
    phase: str,
    resume: Mapping[str, Any],
    exception: BaseException,
) -> Path:
    current = {
        "micro_batch_size": int(bundle.training["micro_batch_size"]),
        "gradient_accumulation": int(bundle.training["gradient_accumulation"]),
    }
    proposed = _proposed_batch_revision(bundle)
    payload: dict[str, Any] = {
        "schema": OOM_REVISION_REQUEST_SCHEMA,
        "family": family,
        "stage": stage_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "precision_role_plan_sha256": precision_role_plan_sha256,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "prior_batch_migration_sha256": bundle.batch_migration_sha256,
        "rank": topology.rank,
        "world_size": topology.world_size,
        "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "local")),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "sequence_length": bundle.sequence_length,
        "phase": phase,
        "resume": dict(resume),
        "current": current,
        "proposed": proposed,
        "revision_available": proposed is not None,
        "exception_type": type(exception).__name__,
        "created_unix": int(time.time()),
        "request_sha256": "",
    }
    payload["request_sha256"] = _canonical_hash(
        payload, omit={"request_sha256"}
    )
    path = (
        output_root
        / "oom"
        / (
            f"{stage_id}-rank{topology.rank:05d}-"
            f"{payload['request_sha256'][:16]}.json"
        )
    )
    _atomic_json(path, payload)
    return path


def _is_device_oom(exception: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", ())
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
    if (
        isinstance(oom_type, type)
        and isinstance(exception, oom_type)
    ) or (
        isinstance(cuda_oom_type, type)
        and isinstance(exception, cuda_oom_type)
    ):
        return True
    message = str(exception).lower()
    return isinstance(exception, RuntimeError) and any(
        marker in message
        for marker in ("out of memory", "memory allocation", "hip error out of memory")
    )


def _raise_stage_oom(
    *,
    output_root: Path,
    family: str,
    stage_id: str,
    parent_checkpoint_sha256: str,
    precision_role_plan_sha256: str,
    bundle: MMapStageBundle,
    topology: ParallelTopology,
    phase: str,
    resume: Mapping[str, Any],
    exception: BaseException,
) -> None:
    if not _is_device_oom(exception):
        raise exception
    request_path = _write_stage_oom_request(
        output_root=output_root,
        family=family,
        stage_id=stage_id,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        precision_role_plan_sha256=precision_role_plan_sha256,
        bundle=bundle,
        topology=topology,
        phase=phase,
        resume=resume,
        exception=exception,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "post-training out of memory; status=253; "
        f"revision_request={request_path}",
        file=sys.stderr,
        flush=True,
    )
    raise PostTrainingOOM(request_path)


def _available_host_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def _validate_runtime_working_set(
    bundle: MMapStageBundle,
    *,
    runtime: Runtime,
    topology: ParallelTopology,
    vocabulary_size: int,
) -> None:
    if not bundle.working_set:
        return
    token_chunk = int(bundle.working_set["token_chunk_size"])
    micro_batch = int(bundle.training["micro_batch_size"])
    group = int(bundle.working_set.get("candidate_micro_group_size", 1))
    sequences = micro_batch * group
    # Student logits are accumulated in FP32 for stable logsumexp.
    multiplier = 1
    estimated_device = (
        sequences * token_chunk * vocabulary_size * 4 * multiplier
        + sequences * token_chunk * 16
    )
    largest_teacher_chunk = 0
    for distribution in bundle.teacher_distributions.values():
        for chunk in distribution.chunks:
            largest_teacher_chunk = max(
                largest_teacher_chunk,
                sequences
                * token_chunk
                * (chunk.vocab_end - chunk.vocab_start)
                * chunk.values.dtype.itemsize,
            )
    estimated_host = largest_teacher_chunk or (
        sequences * token_chunk * vocabulary_size * 4
    )
    if estimated_device > int(bundle.working_set["maximum_device_bytes"]):
        raise StageBackendError(
            f"{bundle.stage_id} recomputed device working set {estimated_device} "
            "exceeds its sealed maximum_device_bytes"
        )
    if estimated_host > int(bundle.working_set["maximum_host_bytes"]):
        raise StageBackendError(
            f"{bundle.stage_id} recomputed host working set {estimated_host} "
            "exceeds its sealed maximum_host_bytes"
        )
    fraction = float(bundle.working_set["headroom_fraction"])
    if runtime.device.type == "cuda":
        free_device, _total_device = torch.cuda.mem_get_info(runtime.device)
        if estimated_device > int(free_device * fraction):
            raise StageBackendError(
                f"{bundle.stage_id} chunk working set lacks sealed HBM headroom"
            )
    free_host = _available_host_bytes()
    if free_host is not None and estimated_host > int(free_host * fraction):
        raise StageBackendError(
            f"{bundle.stage_id} chunk working set lacks sealed host-memory headroom"
        )
    # Every rank performs the same deterministic byte calculation.
    observed = _all_reduce_int(estimated_device, topology, runtime.device)
    if observed != estimated_device * topology.world_size:
        raise StageBackendError("working-set feasibility differs across ranks")


def _working_set_candidate_bundle(
    bundle: MMapStageBundle,
    candidate: Mapping[str, Any],
    *,
    topology: ParallelTopology,
) -> MMapStageBundle:
    micro_batch = int(candidate["micro_batch_size"])
    effective = (
        int(bundle.sealed_training["micro_batch_size"])
        * int(bundle.sealed_training["gradient_accumulation"])
    )
    if effective % micro_batch:
        raise StageBackendError(
            f"{bundle.stage_id} working-set candidate changes effective batch"
        )
    training = dict(bundle.training)
    training["micro_batch_size"] = micro_batch
    training["gradient_accumulation"] = effective // micro_batch
    _validate_training_contract(training)
    if bundle.records % (micro_batch * topology.world_size):
        raise StageBackendError(
            f"{bundle.stage_id} working-set candidate cannot shard every record"
        )
    global_batches = bundle.records // (micro_batch * topology.world_size)
    if global_batches % int(training["gradient_accumulation"]):
        raise StageBackendError(
            f"{bundle.stage_id} working-set candidate leaves partial accumulation"
        )
    working_set = dict(bundle.working_set)
    working_set["token_chunk_size"] = int(candidate["token_chunk_size"])
    if "candidate_micro_group_size" in candidate:
        working_set["candidate_micro_group_size"] = int(
            candidate["candidate_micro_group_size"]
        )
    return dataclasses.replace(
        bundle,
        training=training,
        working_set=working_set,
        working_set_autotune_sha256=None,
    )


def _run_live_working_set_autotune(
    *,
    stage: Mapping[str, Any],
    stage_config_sha256: str,
    bundle: MMapStageBundle,
    model: nn.Module,
    forward_model: nn.Module,
    runtime: Runtime,
    topology: ParallelTopology,
    output_root: Path,
    parent_checkpoint_sha256: str,
    precision_role_plan_sha256: str,
    autotune_profile_sha256: str,
    compile_mode: str,
    vocabulary_size: int,
    signal_coordinator: SignalCoordinator | None,
    active_resume: bool = False,
) -> MMapStageBundle:
    raw_contract = bundle.working_set.get("autotune")
    raw_live_policy = (
        _require_mapping(
            _require_mapping(
                stage.get("autotune"), f"{bundle.stage_id}.autotune"
            ).get("live_canary"),
            f"{bundle.stage_id}.autotune.live_canary",
        )
        if _is_gspo_stage(bundle.stage_id)
        else {}
    )
    require_autotune = raw_live_policy.get(
        "require_working_set_autotune"
    ) is True
    if raw_contract is None:
        if require_autotune:
            raise StageBackendError(
                f"{bundle.stage_id} requires sealed working-set candidates"
            )
        return bundle
    contract = _require_mapping(
        raw_contract, f"{bundle.stage_id} working_set.autotune"
    )
    candidates = [
        dict(
            _require_mapping(
                item, f"{bundle.stage_id} working-set candidate"
            )
        )
        for item in _require_list(
            contract.get("candidates"),
            f"{bundle.stage_id} working_set.autotune.candidates",
        )
    ]
    if (
        len(candidates)
        > int(raw_live_policy.get("maximum_working_set_candidates", 12))
        or int(contract["warmup_trials"])
        != int(
            raw_live_policy.get(
                "working_set_warmup_trials", contract["warmup_trials"]
            )
        )
        or int(contract["measurement_trials"])
        != int(
            raw_live_policy.get(
                "working_set_measurement_trials",
                contract["measurement_trials"],
            )
        )
    ):
        raise StageBackendError(
            f"{bundle.stage_id} working-set contract exceeds its pipeline bounds"
        )
    current = {
        "micro_batch_size": int(bundle.training["micro_batch_size"]),
        "token_chunk_size": int(bundle.working_set["token_chunk_size"]),
    }
    if "candidate_micro_group_size" in bundle.working_set and (
        _is_gspo_stage(bundle.stage_id)
    ):
        current["candidate_micro_group_size"] = int(
            bundle.working_set["candidate_micro_group_size"]
        )
    if current not in candidates:
        raise StageBackendError(
            f"{bundle.stage_id} working-set candidates must include the sealed "
            "or OOM-migrated fallback"
        )
    runtime_inventory, runtime_inventory_sha = _runtime_rank_inventory(
        runtime=runtime,
        topology=topology,
        compile_mode=compile_mode,
    )
    candidate_set_sha = _canonical_hash(candidates)
    bindings = {
        "family": bundle.family,
        "stage": bundle.stage_id,
        "stage_config_sha256": stage_config_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "batch_migration_sha256": bundle.batch_migration_sha256,
        "working_set_contract_sha256": str(contract["contract_sha256"]),
        "candidate_set_sha256": candidate_set_sha,
        "precision_role_plan_sha256": precision_role_plan_sha256,
        "base_autotune_profile_sha256": autotune_profile_sha256,
        "compile_mode": compile_mode,
        "world_size": topology.world_size,
        "expert_parallel_size": topology.expert_parallel_size,
        "expert_replica_count": topology.expert_replica_count,
        "runtime_inventory_sha256": runtime_inventory_sha,
    }
    path = output_root / "autotune" / f"{bundle.stage_id}-working-set.json"

    def load_receipt() -> dict[str, Any]:
        if topology.rank != 0 or not path.is_file():
            return {}
        payload = _read_json(
            path, label=f"{bundle.stage_id} working-set autotune"
        )
        if (
            payload.get("schema") != WORKING_SET_AUTOTUNE_RECEIPT_SCHEMA
            or any(payload.get(name) != value for name, value in bindings.items())
            or payload.get("runtime_inventory") != runtime_inventory
            or payload.get("receipt_sha256")
            != _canonical_hash(payload, omit={"receipt_sha256"})
        ):
            raise StageBackendError(
                f"{bundle.stage_id} working-set autotune resume lineage is invalid"
            )
        return dict(payload)

    loaded = _collective_errors(
        topology,
        load_receipt,
        label=f"{bundle.stage_id} working-set receipt load",
    )
    receipt = dict(
        _broadcast_object(loaded if topology.rank == 0 else None, topology)
    )
    if receipt.get("complete") is True:
        selected = _require_mapping(
            receipt.get("selected_candidate"),
            f"{bundle.stage_id} selected working set",
        )
        selected_sha = _canonical_hash(selected)
        if (
            selected not in candidates
            or receipt.get("selected_candidate_sha256") != selected_sha
        ):
            raise StageBackendError(
                f"{bundle.stage_id} selected working set is corrupt"
            )
        return dataclasses.replace(
            _working_set_candidate_bundle(
                bundle, selected, topology=topology
            ),
            working_set_autotune_sha256=str(receipt["receipt_sha256"]),
        )
    if receipt:
        raise StageBackendError(
            f"{bundle.stage_id} incomplete working-set receipt cannot be "
            "promoted; remove only after operator inspection"
        )
    if active_resume:
        raise StageBackendError(
            f"{bundle.stage_id} active checkpoint exists without a complete "
            "working-set autotune receipt"
        )

    warmups = int(contract["warmup_trials"])
    measurements = int(contract["measurement_trials"])
    trial_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if _signal_requested(
            signal_coordinator, topology=topology, runtime=runtime
        ):
            raise PostTrainingRequeue(
                signal_coordinator.reason or "remote-rank-signal"
            )
        candidate_sha = _canonical_hash(candidate)
        candidate_bundle = _working_set_candidate_bundle(
            bundle, candidate, topology=topology
        )
        timings: list[float] = []
        canary_receipts: list[str] = []
        safe = True
        rejection_reason: str | None = None
        try:
            _validate_runtime_working_set(
                candidate_bundle,
                runtime=runtime,
                topology=topology,
                vocabulary_size=vocabulary_size,
            )
            for trial_index in range(warmups + measurements):
                if runtime.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                payload = _run_stage_kernel_canary(
                    stage=stage,
                    bundle=candidate_bundle,
                    model=model,
                    forward_model=forward_model,
                    runtime=runtime,
                    topology=topology,
                    output_root=output_root,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    compile_mode=compile_mode,
                    receipt_name=(
                        f"{bundle.stage_id}-working-set-{candidate_sha[:16]}-"
                        f"{trial_index:02d}.json"
                    ),
                    force=True,
                )
                elapsed = _all_reduce_float_vector(
                    [float(payload["elapsed_seconds_local_rank"])],
                    topology=topology,
                    runtime=runtime,
                    operation="max",
                )[0]
                canary_receipts.append(str(payload["receipt_sha256"]))
                if trial_index >= warmups:
                    timings.append(elapsed)
        except Exception as exception:
            expected_capacity_rejection = _is_device_oom(exception) or (
                isinstance(exception, StageBackendError)
                and any(
                    marker in str(exception).lower()
                    for marker in (
                        "working set",
                        "headroom",
                        "exceeds its sealed maximum",
                    )
                )
            )
            if not expected_capacity_rejection:
                raise
            safe = False
            rejection_reason = type(exception).__name__ + ":" + str(exception)
            model.zero_grad(set_to_none=True)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
        if safe and (
            len(timings) != measurements
            or any(not math.isfinite(item) or item <= 0 for item in timings)
        ):
            raise StageBackendError(
                f"{bundle.stage_id} working-set timing is incomplete"
            )
        ordered = sorted(timings)
        median = (
            ordered[len(ordered) // 2]
            if ordered
            else float("inf")
        )
        p95 = (
            ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
            if ordered
            else float("inf")
        )
        group = int(candidate.get("candidate_micro_group_size", 1))
        tokens = (
            int(candidate["micro_batch_size"])
            * topology.world_size
            * group
            * bundle.sequence_length
        )
        row: dict[str, Any] = {
            "candidate": candidate,
            "candidate_sha256": candidate_sha,
            "safe": safe,
            "rejection_reason": rejection_reason,
            "measurement_seconds_max_rank": timings,
            "median_seconds_max_rank": median if safe else None,
            "p95_seconds_max_rank": p95 if safe else None,
            "tokens_per_second": tokens / median if safe else 0.0,
            "canary_receipt_sha256": canary_receipts,
            "trial_sha256": "",
        }
        row["trial_sha256"] = _canonical_hash(row, omit={"trial_sha256"})
        trial_rows.append(row)
    safe_rows = [row for row in trial_rows if row["safe"] is True]
    if not safe_rows:
        raise StageBackendError(
            f"{bundle.stage_id} has no safe working-set candidate"
        )
    winner = sorted(
        safe_rows,
        key=lambda row: (
            -float(row["tokens_per_second"]),
            float(row["p95_seconds_max_rank"]),
            str(row["candidate_sha256"]),
        ),
    )[0]
    receipt = {
        "schema": WORKING_SET_AUTOTUNE_RECEIPT_SCHEMA,
        **bindings,
        "runtime_inventory": runtime_inventory,
        "trials": trial_rows,
        "selected_candidate": dict(winner["candidate"]),
        "selected_candidate_sha256": str(winner["candidate_sha256"]),
        "complete": True,
        "receipt_sha256": "",
    }
    receipt["receipt_sha256"] = _canonical_hash(
        receipt, omit={"receipt_sha256"}
    )
    if topology.rank == 0:
        _atomic_json(path, receipt)
    barrier(topology)
    return dataclasses.replace(
        _working_set_candidate_bundle(
            bundle, winner["candidate"], topology=topology
        ),
        working_set_autotune_sha256=str(receipt["receipt_sha256"]),
    )


def _validate_sealed_envelope(
    path: Path,
    *,
    expected_schema: str,
    tokenizer_sha256: str | None,
    verify_hashes: bool,
) -> tuple[dict[str, Any], str]:
    payload = _read_json(path.resolve(), label="sealed requirement")
    if payload.get("envelope_schema") != "metis.sealed-artifact/v1":
        raise StageBackendError(f"{path} is not a sealed Metis artifact")
    if payload.get("schema") != expected_schema or payload.get("complete") is not True:
        raise StageBackendError(f"{path} has the wrong schema or is incomplete")
    manifest_sha = _canonical_hash(payload, omit={"manifest_sha256"})
    if payload.get("manifest_sha256") != manifest_sha:
        raise StageBackendError(f"{path} failed its self-hash")
    if tokenizer_sha256 is not None and payload.get("tokenizer_sha256") != tokenizer_sha256:
        raise StageBackendError(f"{path} tokenizer lineage mismatch")
    files = _require_list(payload.get("files"), f"{path}.files")
    if not files:
        raise StageBackendError(f"{path} seals no payloads")
    root = path.parent.resolve()
    seen: set[str] = set()
    for index, raw_record in enumerate(files):
        record = _require_mapping(raw_record, f"{path}.files[{index}]")
        raw_file = record.get("path")
        if not isinstance(raw_file, str) or raw_file in seen:
            raise StageBackendError(f"{path} contains an invalid or duplicate payload path")
        seen.add(raw_file)
        payload_path = _safe_relative(root, raw_file, label="sealed payload")
        if not payload_path.is_file():
            raise StageBackendError(f"sealed payload is missing: {payload_path}")
        if payload_path.stat().st_size != int(record.get("bytes", -1)):
            raise StageBackendError(f"sealed payload size changed: {payload_path}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise StageBackendError(f"sealed payload hash is invalid: {payload_path}")
        if verify_hashes and sha256_file(payload_path) != expected_hash:
            raise StageBackendError(f"sealed payload hash changed: {payload_path}")
    return payload, manifest_sha


def _load_release_index(
    *,
    family: str,
    pipeline_sha256: str,
    topology: ParallelTopology,
) -> tuple[dict[str, Any], Path] | None:
    raw = os.environ.get("METIS_POSTTRAINING_RELEASE_INDEX")
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()

    def validate_family_index(
        payload: Mapping[str, Any],
        *,
        require_pins: bool,
    ) -> dict[str, Any]:
        if (
            payload.get("schema") != RELEASE_INDEX_SCHEMA
            or payload.get("family") != family
            or payload.get("pipeline_sha256") != pipeline_sha256
            or payload.get("index_sha256")
            != _canonical_hash(payload, omit={"index_sha256"})
        ):
            raise StageBackendError("post-training release index lineage is invalid")
        requirements = _require_mapping(
            payload.get("requirements"), "release-index requirements"
        )
        tokenizer = payload.get("tokenizer_manifest")
        if isinstance(tokenizer, str):
            if require_pins:
                raise StageBackendError(
                    "umbrella-selected tokenizer manifest must be hash pinned"
                )
        elif isinstance(tokenizer, Mapping):
            if (
                not isinstance(tokenizer.get("path"), str)
                or not _is_sha256(tokenizer.get("sha256"))
                or not _is_sha256(tokenizer.get("manifest_sha256"))
            ):
                raise StageBackendError(
                    "release-index tokenizer pin is incomplete"
                )
        else:
            raise StageBackendError("release index omits tokenizer_manifest")
        if require_pins:
            for stage_id, raw_stage in requirements.items():
                stage_records = _require_mapping(
                    raw_stage, f"release-index requirements.{stage_id}"
                )
                for requirement_name, raw_record in stage_records.items():
                    record = _require_mapping(
                        raw_record,
                        f"release-index {stage_id}.{requirement_name}",
                    )
                    state = record.get("state")
                    if not isinstance(record.get("schema"), str):
                        raise StageBackendError(
                            f"release-index {stage_id}.{requirement_name} omits schema"
                        )
                    if state == "sealed":
                        if (
                            not isinstance(record.get("manifest"), str)
                            or not _is_sha256(record.get("sha256"))
                            or not _is_sha256(record.get("manifest_sha256"))
                        ):
                            raise StageBackendError(
                                f"sealed release-index {stage_id}.{requirement_name} "
                                "is not immutably pinned"
                            )
                    elif state == "deferred":
                        hook = _require_mapping(
                            record.get("generation_hook"),
                            f"release-index {stage_id}.{requirement_name} hook",
                        )
                        execution = _require_mapping(
                            hook.get("execution"),
                            f"release-index {stage_id}.{requirement_name} "
                            "hook.execution",
                        )
                        protocol = execution.get("protocol")
                        valid_execution = (
                            protocol == "distributed_family_v1"
                            and set(execution) == {"protocol"}
                        ) or (
                            protocol == "rank0_only_v1"
                            and set(execution)
                            == {
                                "protocol",
                                "nodes",
                                "tasks",
                                "gpus_per_task",
                            }
                            and int(execution.get("nodes", 0)) == 1
                            and int(execution.get("tasks", 0)) == 1
                            and int(execution.get("gpus_per_task", -1)) in {0, 1}
                        )
                        if (
                            not isinstance(record.get("manifest"), str)
                            or not isinstance(hook.get("executable"), str)
                            or not _is_sha256(hook.get("executable_sha256"))
                            or not isinstance(hook.get("receipt"), str)
                            or not isinstance(hook.get("rank_receipts"), str)
                            or not valid_execution
                        ):
                            raise StageBackendError(
                                f"deferred release-index {stage_id}.{requirement_name} "
                                "does not pin its generation hook"
                            )
                    else:
                        raise StageBackendError(
                            f"release-index {stage_id}.{requirement_name} has "
                            "an unsupported state"
                        )
        return dict(payload)

    def load_local() -> dict[str, Any]:
        payload = _read_json(path, label="post-training release index")
        if payload.get("schema") == RELEASE_UMBRELLA_SCHEMA:
            if (
                payload.get("posttraining_contract_sha256") != pipeline_sha256
                or payload.get("umbrella_sha256")
                != _canonical_hash(payload, omit={"umbrella_sha256"})
            ):
                raise StageBackendError(
                    "post-training release umbrella lineage is invalid"
                )
            families = _require_mapping(
                payload.get("families"), "release umbrella families"
            )
            pointer = _require_mapping(
                families.get(family), f"release umbrella families.{family}"
            )
            raw_family_path = pointer.get("path")
            if (
                not isinstance(raw_family_path, str)
                or not _is_sha256(pointer.get("sha256"))
                or not _is_sha256(pointer.get("index_sha256"))
            ):
                raise StageBackendError(
                    f"release umbrella {family} pointer is incomplete"
                )
            family_path = _safe_relative(
                path.parent.resolve(),
                raw_family_path,
                label=f"release umbrella {family} index",
            )
            if (
                not family_path.is_file()
                or sha256_file(family_path) != pointer["sha256"]
            ):
                raise StageBackendError(
                    f"release umbrella {family} index file hash changed"
                )
            selected = validate_family_index(
                _read_json(
                    family_path,
                    label=f"{family} post-training release index",
                ),
                require_pins=True,
            )
            if selected["index_sha256"] != pointer["index_sha256"]:
                raise StageBackendError(
                    f"release umbrella {family} index self-hash changed"
                )
            return {"payload": selected, "path": str(family_path)}
        if (
            payload.get("schema") == RELEASE_INDEX_SCHEMA
            and isinstance(payload.get("family"), str)
        ):
            selected = validate_family_index(payload, require_pins=False)
            return {"payload": selected, "path": str(path)}
        # Compatibility with the first Portage global index. It remains fully
        # hash-pinned, but is normalized to the backend-native family view.
        if (
            payload.get("schema") != RELEASE_INDEX_SCHEMA
            or payload.get("index_self_sha256")
            != _canonical_hash(payload, omit={"index_self_sha256"})
            or payload.get("posttraining_contract_sha256") != pipeline_sha256
        ):
            raise StageBackendError("post-training release index lineage is invalid")
        shared = _require_mapping(payload.get("shared"), "legacy release-index shared")
        families = _require_mapping(
            payload.get("families"), "legacy release-index families"
        )
        tokenizer = _require_mapping(
            shared.get("METIS_TOKENIZER_MANIFEST"),
            "legacy release-index tokenizer",
        )
        family_bindings = _require_mapping(
            families.get(family), f"legacy release-index families.{family}"
        )
        normalized = {
            "schema": RELEASE_INDEX_SCHEMA,
            "family": family,
            "pipeline_sha256": pipeline_sha256,
            "tokenizer_manifest": dict(tokenizer),
            "requirements": {},
            "legacy_bindings": {
                str(name): dict(
                    _require_mapping(
                        record, f"legacy release-index {family}.{name}"
                    )
                )
                for name, record in family_bindings.items()
            },
            "_allow_absolute_paths": True,
            "_legacy_index_self_sha256": payload["index_self_sha256"],
        }
        return {"payload": normalized, "path": str(path)}

    loaded = _collective_errors(
        topology, load_local, label="post-training release-index load"
    )
    return dict(loaded["payload"]), Path(str(loaded["path"])).resolve()


def _indexed_path(
    release_index: tuple[Mapping[str, Any], Path],
    raw: str,
    *,
    label: str,
) -> Path:
    payload, index_path = release_index
    del payload
    return _safe_relative(index_path.parent.resolve(), raw, label=label)


def _indexed_record_path(
    release_index: tuple[Mapping[str, Any], Path],
    record: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> Path:
    raw = record.get(field)
    if not isinstance(raw, str):
        raise StageBackendError(f"{label} omits {field}")
    candidate = Path(raw).expanduser()
    payload, _index_path = release_index
    if candidate.is_absolute():
        if payload.get("_allow_absolute_paths") is not True:
            raise StageBackendError(f"{label}.{field} must be a safe relative path")
        if candidate.is_symlink():
            raise StageBackendError(f"{label}.{field} may not be a symlink")
        return candidate.resolve()
    return _indexed_path(release_index, raw, label=f"{label}.{field}")


def _validate_indexed_manifest_pin(
    *,
    path: Path,
    record: Mapping[str, Any],
    manifest_sha256: str,
    verify_file_hash: bool,
    label: str,
) -> None:
    expected_file_hash = record.get("sha256")
    expected_manifest_hash = record.get("manifest_sha256")
    if expected_file_hash is not None:
        if not _is_sha256(expected_file_hash):
            raise StageBackendError(f"{label} file SHA-256 is invalid")
        if verify_file_hash and sha256_file(path) != expected_file_hash:
            raise StageBackendError(f"{label} file SHA-256 changed")
    if expected_manifest_hash is not None:
        if (
            not _is_sha256(expected_manifest_hash)
            or expected_manifest_hash != manifest_sha256
        ):
            raise StageBackendError(f"{label} sealed manifest SHA-256 changed")


def _materialize_generation_hook(
    *,
    release_index: tuple[Mapping[str, Any], Path],
    record: Mapping[str, Any],
    family: str,
    stage_id: str,
    requirement_name: str,
    parent_checkpoint_sha256: str,
    topology: ParallelTopology,
    output_root: Path,
    stage_bindings: Mapping[str, Any],
) -> Path:
    """Resolve a generated requirement or hand it to the allocation supervisor.

    This function runs inside the live trainer ``srun``.  It must never launch
    another process: all family APUs are occupied and a nested generator can
    deadlock the Slurm allocation.  Rank zero instead seals one collective
    request, all ranks leave with status 252, and ``FamilySupervisor`` runs the
    pinned hook only after this trainer step has released the family's nodes.
    """

    hook = _require_mapping(
        record.get("generation_hook"),
        f"generation hook {stage_id}.{requirement_name}",
    )
    raw_manifest = record.get("manifest")
    raw_receipt = hook.get("receipt")
    raw_executable = hook.get("executable")
    if not all(
        isinstance(value, str)
        for value in (raw_manifest, raw_receipt, raw_executable)
    ):
        raise StageBackendError("generation hook paths must be relative strings")
    output_manifest = _indexed_path(
        release_index,
        str(raw_manifest),
        label=f"generated manifest {stage_id}.{requirement_name}",
    )
    receipt_path = _indexed_path(
        release_index,
        str(raw_receipt),
        label=f"generation receipt {stage_id}.{requirement_name}",
    )
    executable = _indexed_path(
        release_index,
        str(raw_executable),
        label=f"generation executable {stage_id}.{requirement_name}",
    )
    expected_executable_sha = hook.get("executable_sha256")
    if (
        not executable.is_file()
        or executable.is_symlink()
        or not isinstance(expected_executable_sha, str)
        or sha256_file(executable) != expected_executable_sha
    ):
        raise StageBackendError("generation hook executable failed its pinned hash")
    raw_args = _require_list(hook.get("args", []), "generation hook args")
    if not all(isinstance(item, str) for item in raw_args):
        raise StageBackendError("generation hook args must be strings")
    timeout_seconds = int(hook.get("timeout_seconds", 0))
    if not 1 <= timeout_seconds <= 7 * 24 * 60 * 60:
        raise StageBackendError("generation hook timeout must be in [1s,7d]")
    execution = _require_mapping(
        hook.get("execution"),
        f"generation hook {stage_id}.{requirement_name}.execution",
    )
    protocol = execution.get("protocol")
    if protocol == "distributed_family_v1":
        if set(execution) != {"protocol"}:
            raise StageBackendError(
                "distributed generation execution shape is derived from the family"
            )
        task_count = topology.world_size
    elif protocol == "rank0_only_v1":
        if (
            set(execution)
            != {"protocol", "nodes", "tasks", "gpus_per_task"}
            or int(execution.get("nodes", 0)) != 1
            or int(execution.get("tasks", 0)) != 1
            or int(execution.get("gpus_per_task", -1)) not in {0, 1}
        ):
            raise StageBackendError(
                "rank0-only generation must explicitly request one node/task "
                "and zero or one GPU"
            )
        task_count = 1
    else:
        raise StageBackendError("generation hook has no supported execution protocol")
    raw_rank_receipts = hook.get("rank_receipts")
    if not isinstance(raw_rank_receipts, str):
        raise StageBackendError("generation hook omits rank_receipts")
    rank_receipt_root = _indexed_path(
        release_index,
        raw_rank_receipts,
        label=f"generation rank receipts {stage_id}.{requirement_name}",
    )
    if rank_receipt_root in {output_manifest, receipt_path, executable}:
        raise StageBackendError("generation hook output paths must be distinct")
    index_payload, index_path = release_index
    index_file_sha256 = sha256_file(index_path)
    index_self_sha256 = str(index_payload.get("index_sha256", ""))
    record_sha256 = _canonical_hash(record)
    deep_path_raw = os.environ.get("METIS_POSTTRAINING_DEEP_VERIFICATION", "")
    deep_file_sha256 = os.environ.get(
        "METIS_POSTTRAINING_DEEP_VERIFICATION_FILE_SHA256", ""
    )
    deep_receipt_sha256 = os.environ.get(
        "METIS_POSTTRAINING_DEEP_VERIFICATION_RECEIPT_SHA256", ""
    )
    deep_path = Path(deep_path_raw).expanduser().resolve()
    if (
        not deep_path_raw
        or not deep_path.is_file()
        or deep_path.is_symlink()
        or not _is_sha256(deep_file_sha256)
        or not _is_sha256(deep_receipt_sha256)
        or sha256_file(deep_path) != deep_file_sha256
    ):
        raise StageBackendError(
            "deferred materialization requires the validated deep-verification receipt"
        )
    deep_receipt = _read_json(
        deep_path, label="post-training deep-verification receipt"
    )
    if (
        deep_receipt.get("schema")
        != "metis.posttraining-release-deep-verification/v1"
        or deep_receipt.get("receipt_sha256")
        != _canonical_hash(deep_receipt, omit={"receipt_sha256"})
        or deep_receipt.get("receipt_sha256") != deep_receipt_sha256
        or deep_receipt.get("complete") is not True
    ):
        raise StageBackendError("post-training deep-verification receipt changed")

    def validate_completed() -> bool:
        present = (
            output_manifest.exists(),
            receipt_path.exists(),
            rank_receipt_root.exists(),
        )
        if not any(present):
            return False
        if not all(present):
            raise StageBackendError(
                "deferred generation has a partial output/receipt set"
            )
        if (
            output_manifest.is_symlink()
            or receipt_path.is_symlink()
            or rank_receipt_root.is_symlink()
            or not output_manifest.is_file()
            or not receipt_path.is_file()
            or not rank_receipt_root.is_dir()
        ):
            raise StageBackendError("deferred generation outputs are unsafe")
        receipt = _read_json(receipt_path, label="generation hook reducer receipt")
        request_sha = receipt.get("request_sha256")
        if not _is_sha256(request_sha):
            raise StageBackendError("generation hook reducer omits its request binding")
        request_path = (
            output_root
            / "materialization"
            / "requests"
            / f"{stage_id}--{requirement_name}--{parent_checkpoint_sha256[:16]}.json"
        ).resolve()
        if not request_path.is_file() or request_path.is_symlink():
            raise StageBackendError("generation receipt has no immutable request")
        request = _read_json(request_path, label="deferred materialization request")
        if (
            request.get("schema") != DEFERRED_MATERIALIZATION_REQUEST_SCHEMA
            or request.get("request_sha256")
            != _canonical_hash(request, omit={"request_sha256"})
            or request.get("request_sha256") != request_sha
        ):
            raise StageBackendError("generation request/receipt lineage is invalid")
        rank_rows = _require_list(
            receipt.get("rank_receipts"), "generation reducer rank_receipts"
        )
        if len(rank_rows) != task_count:
            raise StageBackendError("generation reducer does not cover every hook rank")
        seen: set[int] = set()
        for raw_row in rank_rows:
            row = _require_mapping(raw_row, "generation reducer rank receipt")
            rank = int(row.get("rank", -1))
            if rank in seen or not 0 <= rank < task_count:
                raise StageBackendError("generation reducer rank coverage is invalid")
            seen.add(rank)
            rank_path = (
                rank_receipt_root / f"rank-{rank:05d}.json"
            ).resolve()
            try:
                rank_path.relative_to(rank_receipt_root.resolve())
            except ValueError as exc:
                raise StageBackendError("generation rank receipt escaped its root") from exc
            if (
                not rank_path.is_file()
                or rank_path.is_symlink()
                or sha256_file(rank_path) != row.get("file_sha256")
            ):
                raise StageBackendError("generation rank receipt bytes changed")
            rank_receipt = _read_json(
                rank_path, label=f"generation rank {rank} receipt"
            )
            if (
                rank_receipt.get("schema")
                != "metis.generation-hook-rank-receipt/v1"
                or rank_receipt.get("request_sha256") != request_sha
                or rank_receipt.get("family") != family
                or rank_receipt.get("stage") != stage_id
                or rank_receipt.get("requirement") != requirement_name
                or rank_receipt.get("parent_checkpoint_sha256")
                != parent_checkpoint_sha256
                or rank_receipt.get("stage_bindings") != dict(stage_bindings)
                or rank_receipt.get("rank") != rank
                or rank_receipt.get("world_size") != task_count
                or rank_receipt.get("success") is not True
                or rank_receipt.get("receipt_sha256")
                != _canonical_hash(rank_receipt, omit={"receipt_sha256"})
                or rank_receipt.get("receipt_sha256")
                != row.get("receipt_sha256")
            ):
                raise StageBackendError("generation rank receipt lineage is invalid")
        output_payload = _read_json(
            output_manifest, label="generated sealed requirement"
        )
        if (
            output_payload.get("envelope_schema") != "metis.sealed-artifact/v1"
            or output_payload.get("schema") != record.get("schema")
            or output_payload.get("complete") is not True
            or output_payload.get("manifest_sha256")
            != _canonical_hash(output_payload, omit={"manifest_sha256"})
            or receipt.get("schema") != "metis.generation-hook-receipt/v2"
            or receipt.get("request_sha256") != request_sha
            or receipt.get("family") != family
            or receipt.get("stage") != stage_id
            or receipt.get("requirement") != requirement_name
            or receipt.get("parent_checkpoint_sha256")
            != parent_checkpoint_sha256
            or receipt.get("stage_bindings") != dict(stage_bindings)
            or receipt.get("release_index_file_sha256") != index_file_sha256
            or receipt.get("release_index_sha256") != index_self_sha256
            or receipt.get("record_sha256") != record_sha256
            or receipt.get("deep_verification_file_sha256")
            != deep_file_sha256
            or receipt.get("deep_verification_receipt_sha256")
            != deep_receipt_sha256
            or receipt.get("executable_sha256") != expected_executable_sha
            or receipt.get("execution_protocol") != protocol
            or receipt.get("world_size") != task_count
            or receipt.get("output_manifest_sha256")
            != sha256_file(output_manifest)
            or receipt.get("output_manifest_self_sha256")
            != output_payload.get("manifest_sha256")
            or receipt.get("success") is not True
            or receipt.get("receipt_sha256")
            != _canonical_hash(receipt, omit={"receipt_sha256"})
        ):
            raise StageBackendError("generation reducer/output lineage is invalid")
        return True

    completed = _collective_errors(
        topology,
        validate_completed,
        label=f"generation receipt {stage_id}.{requirement_name}",
    )
    if completed:
        barrier(topology)
        return output_manifest

    request_path = (
        output_root
        / "materialization"
        / "requests"
        / f"{stage_id}--{requirement_name}--{parent_checkpoint_sha256[:16]}.json"
    ).resolve()

    def write_request() -> str | None:
        if topology.rank != 0:
            return None
        request: dict[str, Any] = {
            "schema": DEFERRED_MATERIALIZATION_REQUEST_SCHEMA,
            "family": family,
            "stage": stage_id,
            "requirement": requirement_name,
            "requirement_schema": str(record.get("schema", "")),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "stage_bindings": dict(stage_bindings),
            "release_index_path": str(index_path),
            "release_index_file_sha256": index_file_sha256,
            "release_index_sha256": index_self_sha256,
            "record_sha256": record_sha256,
            "deep_verification": {
                "path": str(deep_path),
                "file_sha256": deep_file_sha256,
                "receipt_sha256": deep_receipt_sha256,
            },
            "hook": {
                "executable": str(executable),
                "executable_sha256": expected_executable_sha,
                "args": list(raw_args),
                "timeout_seconds": timeout_seconds,
                "output_manifest": str(output_manifest),
                "reducer_receipt": str(receipt_path),
                "rank_receipts": str(rank_receipt_root),
                "execution": dict(execution),
                "world_size": task_count,
            },
            "trainer_world_size": topology.world_size,
            "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "local")),
            "slurm_restart_count": int(
                os.environ.get("SLURM_RESTART_COUNT", "0")
            ),
            "created_unix": int(time.time()),
        }
        request["request_sha256"] = _canonical_hash(
            request, omit={"request_sha256"}
        )
        if request_path.exists():
            existing = _read_json(
                request_path, label="existing deferred materialization request"
            )
            immutable_fields = (
                "schema",
                "family",
                "stage",
                "requirement",
                "requirement_schema",
                "parent_checkpoint_sha256",
                "stage_bindings",
                "release_index_path",
                "release_index_file_sha256",
                "release_index_sha256",
                "record_sha256",
                "deep_verification",
                "hook",
            )
            if (
                existing.get("request_sha256")
                != _canonical_hash(existing, omit={"request_sha256"})
                or any(existing.get(field) != request.get(field) for field in immutable_fields)
            ):
                raise StageBackendError(
                    "existing deferred materialization request was tampered"
                )
            # A prior allocation may have stopped while the external hook was
            # pending. Rebind the otherwise identical request to this exact
            # Slurm execution so the supervisor cannot consume a stale handoff.
            _atomic_json(request_path, request)
        else:
            _atomic_json(request_path, request)
        return str(request["request_sha256"])

    request_sha = _collective_errors(
        topology,
        write_request,
        label=f"deferred request {stage_id}.{requirement_name}",
    )
    request_sha = _broadcast_object(
        request_sha if topology.rank == 0 else None, topology
    )
    if not _is_sha256(request_sha):
        raise StageBackendError("deferred materialization request was not sealed")
    barrier(topology)
    raise DeferredMaterialization(request_path)


def _load_tokenizer(
    topology: ParallelTopology,
    release_index: tuple[Mapping[str, Any], Path] | None = None,
) -> tuple[dict[str, Any], str, Path]:
    indexed_record: Mapping[str, Any] | None = None
    if release_index is not None:
        payload, _index_path = release_index
        raw_tokenizer = payload["tokenizer_manifest"]
        if isinstance(raw_tokenizer, Mapping):
            indexed_record = raw_tokenizer
            path = _indexed_record_path(
                release_index,
                indexed_record,
                field="path",
                label="release-index tokenizer_manifest",
            )
        else:
            path = _indexed_path(
                release_index,
                str(raw_tokenizer),
                label="release-index tokenizer_manifest",
            )
    else:
        raw_path = os.environ.get("METIS_TOKENIZER_MANIFEST")
        if not raw_path:
            raise StageBackendError(
                "METIS_TOKENIZER_MANIFEST or METIS_POSTTRAINING_RELEASE_INDEX is required"
            )
        path = Path(raw_path).expanduser().resolve()
    def validate() -> tuple[dict[str, Any], str]:
        tokenizer_payload, manifest_sha = _validate_sealed_envelope(
            path,
            expected_schema="metis.tokenizer/v1",
            tokenizer_sha256=None,
            verify_hashes=topology.rank == 0,
        )
        if indexed_record is not None:
            _validate_indexed_manifest_pin(
                path=path,
                record=indexed_record,
                manifest_sha256=manifest_sha,
                verify_file_hash=topology.rank == 0,
                label="release-index tokenizer_manifest",
            )
        return tokenizer_payload, manifest_sha

    payload, manifest_sha = _collective_errors(
        topology,
        validate,
        label="tokenizer validation",
    )
    metadata = _require_mapping(payload.get("metadata"), "tokenizer metadata")
    if int(metadata.get("vocabulary_size", payload.get("vocabulary_size", -1))) != 65_536:
        raise StageBackendError("Metis post-training requires the sealed 65,536-token tokenizer")
    return payload, manifest_sha, path


def _load_canonical_lookup(
    *,
    data_release: str | Path,
    tokenizer_payload: Mapping[str, Any],
    tokenizer_manifest_path: Path,
    topology: ParallelTopology,
) -> tuple[np.ndarray, str, str]:
    metadata = _require_mapping(
        tokenizer_payload.get("metadata"), "tokenizer metadata"
    )

    def load_on_rank() -> dict[str, Any] | None:
        if topology.rank != 0:
            return None
        inventory = ReleaseInventory.from_release_root(data_release)
        if (
            inventory.ngram_canonical_map is None
            or inventory.ngram_canonical_ids is None
        ):
            raise StageBackendError(
                "base release omits the canonical-ID sidecar"
            )
        release_descriptor = _read_json(
            inventory.root / "RELEASE.json",
            label="base data-release descriptor",
        )
        release_artifacts = _require_mapping(
            release_descriptor.get("artifacts"),
            "base data-release artifacts",
        )
        if (
            release_descriptor.get("release_sha256")
            != _canonical_hash(
                release_descriptor, omit={"release_sha256"}
            )
            or release_descriptor.get("release_sha256")
            != inventory.release_sha256
            or release_artifacts.get("tokenizer")
            != str(inventory.tokenizer.relative_to(inventory.root))
            or release_artifacts.get("ngram_canonical_map")
            != str(inventory.ngram_canonical_map.relative_to(inventory.root))
            or release_artifacts.get("ngram_canonical_ids")
            != str(inventory.ngram_canonical_ids.relative_to(inventory.root))
        ):
            raise StageBackendError(
                "base data release does not self-hash its exact tokenizer "
                "and canonical-sidecar paths"
            )
        tokenizer_file = metadata.get("tokenizer_file")
        if not isinstance(tokenizer_file, str):
            raise StageBackendError(
                "post-training tokenizer metadata must identify tokenizer_file"
            )
        sealed_tokenizer = _safe_relative(
            tokenizer_manifest_path.parent.resolve(),
            tokenizer_file,
            label="post-training tokenizer bytes",
        )
        sealed_files = {
            str(record.get("path")): record
            for record in _require_list(
                tokenizer_payload.get("files"),
                "post-training tokenizer files",
            )
            if isinstance(record, Mapping)
        }
        base_tokenizer_sha256 = sha256_file(inventory.tokenizer)
        base_tokenizer_relative = str(
            inventory.tokenizer.relative_to(inventory.root)
        )
        canonical_map_relative = str(
            inventory.ngram_canonical_map.relative_to(inventory.root)
        )
        canonical_ids_relative = str(
            inventory.ngram_canonical_ids.relative_to(inventory.root)
        )
        if (
            release_descriptor.get("tokenizer_sha256")
            != base_tokenizer_sha256
            or release_descriptor.get("ngram_canonical_map_self_sha256")
            != inventory.ngram_canonical_map_self_sha256
            or release_descriptor.get("ngram_canonical_ids_sha256")
            != inventory.ngram_canonical_ids_sha256
        ):
            raise StageBackendError(
                "base data release claimed tokenizer or canonical hashes "
                "differ from verified live artifacts"
            )
        expected_metadata = {
            "tokenizer_sha256": base_tokenizer_sha256,
            "base_release_sha256": inventory.release_sha256,
            "base_release_tokenizer_path": base_tokenizer_relative,
            "base_release_tokenizer_sha256": base_tokenizer_sha256,
            "base_release_canonical_map_path": canonical_map_relative,
            "base_release_canonical_ids_path": canonical_ids_relative,
            "ngram_canonical_map_self_sha256": (
                inventory.ngram_canonical_map_self_sha256
            ),
            "ngram_canonical_ids_sha256": (
                inventory.ngram_canonical_ids_sha256
            ),
        }
        if any(
            metadata.get(field) != value
            for field, value in expected_metadata.items()
        ):
            raise StageBackendError(
                "post-training tokenizer is not directly bound to the live "
                "base-release tokenizer and canonical sidecar"
            )
        tokenizer_record = sealed_files.get(tokenizer_file)
        if (
            not sealed_tokenizer.is_file()
            or sealed_tokenizer.is_symlink()
            or not isinstance(tokenizer_record, Mapping)
            or tokenizer_record.get("sha256") != base_tokenizer_sha256
            or sha256_file(sealed_tokenizer) != base_tokenizer_sha256
        ):
            raise StageBackendError(
                "post-training tokenizer bytes differ from the base release"
            )
        validate_canonical_id_sidecar(
            manifest_path=inventory.ngram_canonical_map,
            binary_path=inventory.ngram_canonical_ids,
            tokenizer_path=inventory.tokenizer,
            expected_vocabulary_size=65_536,
            expected_manifest_sha256=(
                inventory.ngram_canonical_map_self_sha256
            ),
            expected_binary_sha256=inventory.ngram_canonical_ids_sha256,
            recompute_from_tokenizer=True,
        )
        if (
            metadata.get("ngram_canonical_map_self_sha256")
            != inventory.ngram_canonical_map_self_sha256
            or metadata.get("ngram_canonical_ids_sha256")
            != inventory.ngram_canonical_ids_sha256
        ):
            raise StageBackendError(
                "post-training tokenizer disagrees with the verified canonical-ID sidecar"
            )
        lookup = np.fromfile(inventory.ngram_canonical_ids, dtype="<u2")
        if lookup.shape != (65_536,):
            raise StageBackendError(
                "verified canonical-ID lookup has the wrong vocabulary shape"
            )
        return {
            "lookup": lookup,
            "map_self_sha256": inventory.ngram_canonical_map_self_sha256,
            "ids_sha256": inventory.ngram_canonical_ids_sha256,
        }

    local = _collective_errors(
        topology,
        load_on_rank,
        label="canonical-ID sidecar load",
    )
    payload = _broadcast_object(local if topology.rank == 0 else None, topology)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("lookup"), np.ndarray
    ):
        raise StageBackendError("canonical-ID sidecar broadcast failed")
    lookup = payload["lookup"]
    if lookup.dtype != np.dtype("<u2") or lookup.shape != (65_536,):
        raise StageBackendError("canonical-ID sidecar broadcast changed dtype or shape")
    return (
        lookup,
        str(payload["map_self_sha256"]),
        str(payload["ids_sha256"]),
    )


def _resolve_requirements(
    stage: Mapping[str, Any],
    *,
    family: str,
    parent_stage: str,
    parent_checkpoint_sha256: str,
    tokenizer_sha256: str,
    topology: ParallelTopology,
    output_root: Path,
    stage_bindings: Mapping[str, Any],
    release_index: tuple[Mapping[str, Any], Path] | None = None,
) -> list[SealedRequirement]:
    result: list[SealedRequirement] = []
    for raw_requirement in _require_list(stage.get("requirements"), f"{stage['id']}.requirements"):
        requirement = _require_mapping(raw_requirement, "stage requirement")
        base_env = str(requirement.get("env", ""))
        family_env = f"{base_env}_{family.upper()}"
        if requirement.get("family_bound") is True:
            environment_variable = family_env
        elif family_env in os.environ:
            environment_variable = family_env
        else:
            environment_variable = base_env
        indexed: Mapping[str, Any] | None = None
        if release_index is not None:
            index_payload, _index_path = release_index
            legacy_bindings = index_payload.get("legacy_bindings")
            if isinstance(legacy_bindings, Mapping):
                indexed = _require_mapping(
                    legacy_bindings.get(base_env),
                    f"legacy release-index {family}.{base_env}",
                )
                path = _indexed_record_path(
                    release_index,
                    indexed,
                    field="path",
                    label=f"legacy release-index {family}.{base_env}",
                )
            else:
                stage_records = _require_mapping(
                    _require_mapping(
                        index_payload.get("requirements"),
                        "release-index requirements",
                    ).get(str(stage["id"])),
                    f"release-index requirements.{stage['id']}",
                )
                indexed = _require_mapping(
                    stage_records.get(str(requirement.get("name", ""))),
                    f"release-index {stage['id']}.{requirement.get('name')}",
                )
                if indexed.get("schema") != requirement.get("schema"):
                    raise StageBackendError(
                        f"release-index schema mismatch for "
                        f"{stage['id']}.{requirement.get('name')}"
                    )
                state = indexed.get("state")
                if state == "sealed":
                    path = _indexed_record_path(
                        release_index,
                        indexed,
                        field="manifest",
                        label=(
                            f"release-index {stage['id']}."
                            f"{requirement.get('name')}"
                        ),
                    )
                elif (
                    state == "deferred"
                    and indexed.get("generation_hook") is not None
                ):
                    path = _materialize_generation_hook(
                        release_index=release_index,
                        record=indexed,
                        family=family,
                        stage_id=str(stage["id"]),
                        requirement_name=str(requirement.get("name")),
                        parent_checkpoint_sha256=parent_checkpoint_sha256,
                        topology=topology,
                        output_root=output_root,
                        stage_bindings=stage_bindings,
                    )
                else:
                    raise StageBackendError(
                        f"release-index requirement {stage['id']}."
                        f"{requirement.get('name')} is neither sealed nor backed "
                        "by a pinned generation hook"
                    )
            environment_variable = (
                f"release-index:{stage['id']}:{requirement.get('name')}"
            )
        else:
            raw_path = os.environ.get(environment_variable)
            if not raw_path:
                raise StageBackendError(
                    f"{stage['id']} requires {environment_variable}; no data is synthesized"
                )
            path = Path(raw_path).expanduser().resolve()
        def validate_requirement(
            p: Path = path,
            r: Mapping[str, Any] = requirement,
            pin: Mapping[str, Any] | None = indexed,
        ) -> tuple[dict[str, Any], str]:
            requirement_payload, requirement_sha = _validate_sealed_envelope(
                p,
                expected_schema=str(r["schema"]),
                tokenizer_sha256=(
                    tokenizer_sha256 if r.get("tokenizer_bound", True) else None
                ),
                verify_hashes=topology.rank == 0,
            )
            if pin is not None:
                _validate_indexed_manifest_pin(
                    path=p,
                    record=pin,
                    manifest_sha256=requirement_sha,
                    verify_file_hash=topology.rank == 0,
                    label=(
                        f"release-index {stage['id']}."
                        f"{requirement.get('name')}"
                    ),
                )
            return requirement_payload, requirement_sha

        payload, manifest_sha = _collective_errors(
            topology,
            validate_requirement,
            label=f"{stage['id']} requirement {environment_variable}",
        )
        metadata = _require_mapping(payload.get("metadata"), f"{environment_variable}.metadata")
        if requirement.get("family_bound") is True and metadata.get("family") != family:
            raise StageBackendError(f"{environment_variable} is not bound to {family}")
        generated_from = requirement.get("generated_from_stage")
        if generated_from is not None and metadata.get("generated_from_stage") != parent_stage:
            raise StageBackendError(
                f"{environment_variable} was not generated from {parent_stage}"
            )
        generated_from_stages = requirement.get("generated_from_stages")
        if generated_from_stages is not None:
            declared_stages = tuple(
                str(value)
                for value in _require_list(
                    generated_from_stages,
                    f"{environment_variable}.generated_from_stages",
                )
            )
            specialist_bindings = _require_mapping(
                stage_bindings.get("specialist_checkpoints"),
                f"{environment_variable} specialist checkpoint bindings",
            )
            expected_stage_checkpoints = {
                parent_stage: parent_checkpoint_sha256,
                **{
                    specialist_id: str(
                        _require_mapping(
                            specialist_bindings[specialist_id],
                            f"{environment_variable}.{specialist_id}",
                        )["checkpoint_sha256"]
                    )
                    for specialist_id in SPECIALIST_STAGE_IDS
                },
            }
            if (
                set(declared_stages) != set(expected_stage_checkpoints)
                or metadata.get("generated_from_stage_checkpoints")
                != expected_stage_checkpoints
                or metadata.get("stage_bindings_sha256")
                != _canonical_hash(stage_bindings)
            ):
                raise StageBackendError(
                    f"{environment_variable} is not bound to the exact "
                    "multi-checkpoint generation lineage"
                )
        if (
            requirement.get("checkpoint_bound") is True
            and metadata.get("generated_from_checkpoint_sha256")
            != parent_checkpoint_sha256
        ):
            raise StageBackendError(
                f"{environment_variable} is not bound to the live parent checkpoint"
            )
        if "minimum_records" in requirement and int(metadata.get("records", -1)) < int(
            requirement["minimum_records"]
        ):
            raise StageBackendError(f"{environment_variable} has too few records")
        if "minimum_source_instructions" in requirement and int(
            metadata.get("source_instruction_count", -1)
        ) < int(requirement["minimum_source_instructions"]):
            raise StageBackendError(
                f"{environment_variable} has too few source instructions"
            )
        if "minimum_tokens" in requirement and int(metadata.get("tokens", -1)) < int(
            requirement["minimum_tokens"]
        ):
            raise StageBackendError(f"{environment_variable} has too few tokens")
        if "maximum_tokens" in requirement and int(metadata.get("tokens", -1)) > int(
            requirement["maximum_tokens"]
        ):
            raise StageBackendError(f"{environment_variable} exceeds the locked token budget")
        for field, expected in _require_mapping(
            requirement.get("required_metadata", {}), "required_metadata"
        ).items():
            if metadata.get(field) != expected:
                raise StageBackendError(
                    f"{environment_variable} metadata {field} must be "
                    f"{expected!r}, got {metadata.get(field)!r}"
                )
        result.append(
            SealedRequirement(
                name=str(requirement.get("name", environment_variable)),
                schema=str(requirement["schema"]),
                environment_variable=environment_variable,
                manifest_path=path,
                manifest_sha256=manifest_sha,
                payload=payload,
                family_bound=requirement.get("family_bound") is True,
                checkpoint_bound=requirement.get("checkpoint_bound") is True,
            )
        )
    return result


def _data_requirement(
    requirements: Sequence[SealedRequirement],
    *,
    stage_id: str,
) -> SealedRequirement:
    expected = _DATA_SCHEMAS[stage_id]
    matches = [item for item in requirements if item.schema == expected]
    if len(matches) != 1:
        raise StageBackendError(
            f"{stage_id} must resolve exactly one {expected} mmap data artifact"
        )
    return matches[0]


def _to_device(batch: Mapping[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    converted: dict[str, Tensor] = {}
    for name, array in batch.items():
        tensor = torch.from_numpy(array)
        if np.issubdtype(array.dtype, np.integer):
            tensor = tensor.long() if "mask" not in name and name != "truncated" else tensor.bool()
        elif array.dtype == np.bool_:
            tensor = tensor.bool()
        converted[name] = tensor.to(device=device, non_blocking=device.type == "cuda")
    return converted


def _canonicalize_ids(bundle: MMapStageBundle, input_ids: Tensor) -> Tensor:
    lookup = bundle.canonical_lookup_tensor
    if lookup is None or lookup.device != input_ids.device:
        lookup = torch.from_numpy(
            bundle.canonical_id_lookup.astype(np.int64, copy=False)
        ).to(device=input_ids.device, non_blocking=input_ids.device.type == "cuda")
        bundle.canonical_lookup_tensor = lookup
    if torch.any(input_ids < 0) or torch.any(input_ids >= lookup.numel()):
        raise StageBackendError("post-training input IDs escape the tokenizer vocabulary")
    return lookup[input_ids.long()]


def _align_supervised_labels(
    labels: Tensor,
    loss_mask: Tensor,
    attention_mask: Tensor,
    document_ids: Tensor,
    reset_mask: Tensor,
) -> Tensor:
    """Convert token-aligned shard labels to the model's next-token convention."""

    if any(
        tensor.shape != labels.shape
        for tensor in (loss_mask, attention_mask, document_ids, reset_mask)
    ):
        raise StageBackendError(
            "supervised labels and boundary tensors must have identical shapes"
        )
    aligned = torch.full_like(labels.long(), -100)
    valid = (
        loss_mask[:, 1:].bool()
        & attention_mask[:, :-1].bool()
        & attention_mask[:, 1:].bool()
        & document_ids[:, :-1].eq(document_ids[:, 1:])
        & ~reset_mask[:, 1:].bool()
        & labels[:, 1:].ne(-100)
    )
    aligned[:, :-1] = torch.where(
        valid,
        labels[:, 1:].long(),
        torch.full_like(labels[:, 1:].long(), -100),
    )
    return aligned


def _model_forward(
    model: nn.Module,
    input_ids: Tensor,
    *,
    canonical_ids: Tensor,
    attention_mask: Tensor | None = None,
    labels: Tensor | None = None,
    deterministic: bool = False,
    return_logits: bool = False,
) -> Any:
    curriculum = {"stochastic_routing": False} if deterministic else None
    return model(
        input_ids,
        labels=labels,
        attention_mask=attention_mask,
        canonical_ids=canonical_ids,
        curriculum=curriculum,
        return_logits=return_logits,
    )


def _compile_posttraining_forward_model(
    model: nn.Module,
    *,
    compile_mode: str,
) -> nn.Module:
    if compile_mode in {"eager", "none"}:
        return model
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise StageBackendError(
            f"unsupported measured post-training compile mode: {compile_mode}"
        )
    compiler = getattr(torch, "compile", None)
    if not callable(compiler):
        raise StageBackendError("measured profile requires torch.compile")
    # The optimized module shares the raw module's Parameter objects. Checkpoint
    # and optimizer ownership therefore remain on ``model`` while all trunk
    # forwards use the measured dynamic-shape execution policy.
    return compiler(
        model,
        mode=compile_mode,
        dynamic=True,
        fullgraph=False,
    )


def _lm_head(model: nn.Module) -> nn.Module:
    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Module):
        raise StageBackendError("Metis model does not expose its chunkable lm_head")
    return head


def _head_execution_context(model: nn.Module) -> Any:
    precision_policy = getattr(model, "precision_policy", None)
    factory = getattr(precision_policy, "execution_context", None)
    if not callable(factory):
        return nullcontext()
    if "module" in signature(factory).parameters:
        return factory(module=_lm_head(model))
    return factory()


def _head_checkpoint_context_fn(model: nn.Module) -> Any:
    precision_policy = getattr(model, "precision_policy", None)
    factory = getattr(
        precision_policy,
        "activation_checkpoint_context_fn",
        None,
    )
    if callable(factory):
        return factory

    def contexts() -> tuple[Any, Any]:
        return nullcontext(), nullcontext()

    return contexts


def _selected_token_log_probs_from_hidden(
    model: nn.Module,
    hidden_states: Tensor,
    target_ids: Tensor,
    *,
    token_chunk_size: int,
) -> Tensor:
    """Exact selected-token log-probs without retaining a full-vocabulary tensor."""

    if hidden_states.shape[:2] != target_ids.shape:
        raise StageBackendError("hidden states and selected-token targets do not align")
    head = _lm_head(model)
    chunks: list[Tensor] = []
    for start in range(0, hidden_states.shape[1], token_chunk_size):
        end = min(hidden_states.shape[1], start + token_chunk_size)
        fixed_targets = target_ids[:, start:end]

        def selected(chunk_hidden: Tensor, targets: Tensor = fixed_targets) -> Tensor:
            with _head_execution_context(model):
                logits = head(chunk_hidden)
            logits_fp32 = logits.float()
            return (
                logits_fp32.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(logits_fp32, dim=-1)
            )

        with set_checkpoint_early_stop(False):
            chunks.append(
                checkpoint(
                    selected,
                    hidden_states[:, start:end],
                    use_reentrant=False,
                    context_fn=_head_checkpoint_context_fn(model),
                )
            )
    return torch.cat(chunks, dim=1)


def _topk_union_reverse_kl(
    *,
    model: nn.Module,
    hidden_states: Tensor,
    teacher_union_token_ids: Tensor,
    teacher_union_logits: Tensor,
    teacher_union_count: Tensor,
    response_mask: Tensor,
    token_chunk_size: int,
    temperature: float,
) -> Tensor:
    """Reverse KL on the sealed union of student and specialist top-k tokens.

    The union is produced from the on-policy student checkpoint and the routed
    same-tokenizer Metis specialist. We normalize both distributions on that
    exact union, matching MiniCPM5's efficient OPD approximation without
    pretending that a cross-tokenizer external teacher has aligned logits.
    """

    if (
        hidden_states.shape[:2] != teacher_union_token_ids.shape[:2]
        or teacher_union_token_ids.shape != teacher_union_logits.shape
        or teacher_union_count.shape != hidden_states.shape[:2]
        or response_mask.shape != hidden_states.shape[:2]
        or teacher_union_token_ids.shape[-1] != 64
        or temperature <= 0
    ):
        raise StageBackendError("OPD top-k-union tensors do not align")
    total = hidden_states.float().sum() * 0.0
    count = response_mask.bool().sum()
    if not bool(count > 0):
        raise StageBackendError("OPD batch has no response tokens")
    head = _lm_head(model)
    for start in range(0, hidden_states.shape[1], token_chunk_size):
        end = min(hidden_states.shape[1], start + token_chunk_size)
        fixed_ids = teacher_union_token_ids[:, start:end].long()
        fixed_teacher = teacher_union_logits[:, start:end].float()
        fixed_counts = teacher_union_count[:, start:end].long()
        fixed_response = response_mask[:, start:end].bool()

        def chunk_loss(chunk_hidden: Tensor) -> Tensor:
            with _head_execution_context(model):
                full_logits = head(chunk_hidden)
            selected = full_logits.float().gather(-1, fixed_ids)
            positions = torch.arange(
                selected.shape[-1], device=selected.device
            ).view(1, 1, -1)
            union_mask = positions < fixed_counts.unsqueeze(-1)
            negative = torch.finfo(torch.float32).min
            student_scaled = (selected / temperature).masked_fill(
                ~union_mask, negative
            )
            teacher_scaled = (fixed_teacher / temperature).masked_fill(
                ~union_mask, negative
            )
            student_log_probs = F.log_softmax(student_scaled, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_scaled, dim=-1)
            student_probs = student_log_probs.exp()
            per_token = (
                student_probs
                * (student_log_probs - teacher_log_probs.detach())
                * union_mask
            ).sum(dim=-1)
            return per_token.masked_select(fixed_response).sum()

        with set_checkpoint_early_stop(False):
            total = total + checkpoint(
                chunk_loss,
                hidden_states[:, start:end],
                use_reentrant=False,
                context_fn=_head_checkpoint_context_fn(model),
            )
    return total / count.to(dtype=torch.float32)


def _set_learning_rate(optimizer: OptimizerBundle, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        scale = float(group.get("_metis_lr_scale", 1.0))
        group["lr"] = learning_rate * scale


def _optimizer_state_policy(stage: Mapping[str, Any]) -> str:
    policy = stage.get("optimizer_state")
    expected = (
        {"none"}
        if stage["id"] in {"evaluation", "publish_gate"}
        else {"preserve", "reset"}
    )
    if policy not in expected:
        raise StageBackendError(
            f"{stage['id']}.optimizer_state must be one of {sorted(expected)}"
        )
    return str(policy)


def _apply_optimizer_state_transition(
    optimizer: OptimizerBundle,
    *,
    policy: str,
    active_resume: bool,
) -> None:
    if active_resume:
        # The distributed checkpoint already restored the exact optimizer
        # state. Clearing it here would corrupt crash-resume equivalence.
        return
    if policy == "preserve":
        return
    if policy != "reset":
        raise StageBackendError(f"invalid policy optimizer transition: {policy}")
    for state in optimizer.dense.state.values():
        for key in list(state):
            value = state[key]
            if key == "master_param":
                continue
            if key in {"step", "exp_avg", "exp_avg_sq", "momentum_buffer"}:
                if isinstance(value, Tensor):
                    value.zero_()
                elif key == "step":
                    state[key] = 0
                else:
                    del state[key]
            else:
                del state[key]
    if optimizer.sparse is not None:
        for state in optimizer.sparse.state.values():
            for key in list(state):
                value = state[key]
                if key == "master_param":
                    continue
                if key in {"step", "exp_avg", "exp_avg_sq"}:
                    if isinstance(value, Tensor):
                        value.zero_()
                    elif key == "step":
                        state[key] = 0
                    else:
                        del state[key]
                else:
                    del state[key]
    optimizer.zero_grad(set_to_none=True)


def _scheduled_learning_rate(
    training: Mapping[str, Any],
    *,
    optimizer_step: int,
    total_steps: int,
) -> float:
    peak = float(training["learning_rate"])
    floor = peak * float(training["minimum_learning_rate_ratio"])
    warmup = int(training["warmup_steps"])
    if warmup > 0 and optimizer_step < warmup:
        return peak * float(optimizer_step + 1) / float(warmup)
    progress = (optimizer_step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def _main_optimizer_step(
    *,
    model: nn.Module,
    optimizer: OptimizerBundle,
    topology: ParallelTopology,
    runtime: Runtime,
    local_weight: int,
    gradient_clip: float,
    expert_selection_counts: Tensor | None = None,
) -> tuple[int, float]:
    global_weight = _all_reduce_int(local_weight, topology, runtime.device)
    if global_weight <= 0:
        raise StageBackendError("optimizer step has no globally valid training units")
    synchronize_gradients(model, topology)
    normalize_summed_gradients(
        model,
        topology,
        global_supervised_tokens=global_weight,
    )
    grad_norm = clip_grad_norm_(
        model, gradient_clip, topology=topology
    )
    value = float(grad_norm.detach().float().item())
    if not math.isfinite(value):
        raise StageBackendError("post-training produced a non-finite gradient norm")
    optimizer.step()
    selection_bias_updater = getattr(
        model, "update_expert_selection_biases", None
    )
    if callable(selection_bias_updater):
        if expert_selection_counts is None:
            raise StageBackendError(
                "policy update omitted per-layer expert selection counts"
            )
        selection_bias_updater(expert_selection_counts)
    optimizer.zero_grad(set_to_none=True)
    return global_weight, value


def _accumulate_expert_selection_counts(
    current: Tensor | None,
    output: Any,
    *,
    include: bool = True,
) -> Tensor | None:
    telemetry = _output_value(output, "telemetry", {})
    counts = (
        telemetry.get("expert_selection_counts")
        if isinstance(telemetry, Mapping)
        else None
    )
    if counts is None:
        return current
    if (
        not isinstance(counts, Tensor)
        or counts.ndim != 2
        or not torch.isfinite(counts).all()
        or torch.any(counts < 0)
    ):
        raise StageBackendError(
            "expert_selection_counts telemetry must be a finite non-negative "
            "[layers,experts] tensor"
        )
    detached = counts.detach()
    if not include:
        detached = torch.zeros_like(detached)
    if current is not None and current.shape != detached.shape:
        raise StageBackendError(
            "expert_selection_counts shape changed within an optimizer batch"
        )
    return (
        detached.clone()
        if current is None
        else current + detached
    )


def _module_extra_state_snapshot(model: nn.Module) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for name, module in model.named_modules():
        if type(module).get_extra_state is nn.Module.get_extra_state:
            continue
        value = module.get_extra_state()
        snapshot[name] = (
            value.detach().clone()
            if isinstance(value, Tensor)
            else copy.deepcopy(value)
        )
    return snapshot


def _restore_module_extra_state(
    model: nn.Module,
    snapshot: Mapping[str, Any],
) -> None:
    modules = dict(model.named_modules())
    if not set(snapshot).issubset(modules):
        raise StageBackendError(
            "model module inventory changed during the kernel canary"
        )
    for name, value in snapshot.items():
        modules[name].set_extra_state(value)
    for name, expected in snapshot.items():
        observed = modules[name].get_extra_state()
        if isinstance(expected, Tensor):
            if not isinstance(observed, Tensor) or not torch.equal(
                observed.cpu(),
                expected.cpu(),
            ):
                raise StageBackendError(
                    f"kernel canary failed to restore module extra state: {name}"
                )
        elif observed != expected:
            raise StageBackendError(
                f"kernel canary failed to restore module extra state: {name}"
            )


def _run_stage_kernel_canary(
    *,
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
    model: nn.Module,
    forward_model: nn.Module,
    runtime: Runtime,
    topology: ParallelTopology,
    output_root: Path,
    parent_checkpoint_sha256: str,
    compile_mode: str,
    receipt_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a destructive F/B probe while preserving TE delayed-scaling state."""

    extra_state = _module_extra_state_snapshot(model)
    try:
        return _run_stage_kernel_canary_impl(
            stage=stage,
            bundle=bundle,
            model=model,
            forward_model=forward_model,
            runtime=runtime,
            topology=topology,
            output_root=output_root,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            compile_mode=compile_mode,
            receipt_name=receipt_name,
            force=force,
        )
    finally:
        model.zero_grad(set_to_none=True)
        _restore_module_extra_state(model, extra_state)


def _run_stage_kernel_canary_impl(
    *,
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
    model: nn.Module,
    forward_model: nn.Module,
    runtime: Runtime,
    topology: ParallelTopology,
    output_root: Path,
    parent_checkpoint_sha256: str,
    compile_mode: str,
    receipt_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    path = (
        output_root
        / "canaries"
        / (receipt_name or f"{bundle.stage_id}.json")
    )

    def existing_on_rank() -> dict[str, Any] | None:
        if force or topology.rank != 0 or not path.is_file():
            return None
        receipt = _read_json(path, label=f"{bundle.stage_id} kernel canary")
        expected = {
            "schema": "metis.posttraining-kernel-canary/v1",
            "family": bundle.family,
            "stage": bundle.stage_id,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "batch_migration_sha256": bundle.batch_migration_sha256,
            "canonical_ids_sha256": bundle.canonical_ids_sha256,
            "compile_mode": compile_mode,
            "sequence_length": bundle.sequence_length,
            "runtime_batch": _runtime_batch_payload(bundle),
            "passed": True,
        }
        if (
            any(receipt.get(field) != value for field, value in expected.items())
            or receipt.get("receipt_sha256")
            != _canonical_hash(receipt, omit={"receipt_sha256"})
        ):
            return None
        return receipt

    existing = _collective_errors(
        topology,
        existing_on_rank,
        label=f"{bundle.stage_id} kernel-canary receipt load",
    )
    existing = _broadcast_object(existing if topology.rank == 0 else None, topology)
    if isinstance(existing, Mapping):
        return dict(existing)

    indices = next(
        bundle.iter_rank_indices(topology, start_epoch=0, start_global_batch=0)
    )[2]
    cpu_batch = bundle.materialize_batch(indices)
    batch = _to_device(cpu_batch, runtime.device)
    model.zero_grad(set_to_none=True)
    forward_model.train()
    fork_devices: list[int] = []
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    started = time.perf_counter()
    if runtime.device.type == "cuda":
        fork_devices = [
            runtime.device.index
            if runtime.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=fork_devices):
        if bundle.stage_id in {
            "context_extension",
            "cold_start_sft",
            "overall_sft",
        }:
            input_ids = batch["input_ids"].long()
            attention = batch["attention_mask"].bool()
            labels = _align_supervised_labels(
                batch["labels"].long(),
                batch["loss_mask"].bool(),
                attention,
                batch["document_ids"].long(),
                batch["reset_mask"].bool(),
            )
            output = forward_model(
                input_ids,
                labels=labels,
                attention_mask=attention,
                document_ids=batch["document_ids"].long(),
                reset_mask=batch["reset_mask"].bool(),
                canonical_ids=batch["canonical_ids"].long(),
                return_logits=False,
            )
            objective = _output_value(output, "loss")
        elif _is_gspo_stage(bundle.stage_id):
            micro_group = int(bundle.working_set["candidate_micro_group_size"])
            ids = batch["candidate_input_ids"][:, :micro_group].reshape(
                -1, bundle.sequence_length
            ).long()
            attention = batch["candidate_attention_mask"][
                :, :micro_group
            ].reshape(-1, bundle.sequence_length).bool()
            output = _model_forward(
                forward_model,
                ids,
                attention_mask=attention,
                canonical_ids=_canonicalize_ids(bundle, ids),
                return_logits=False,
            )
            hidden = _output_value(output, "final_hidden_state")[:, :-1]
            token_chunk = min(
                int(bundle.working_set["token_chunk_size"]),
                hidden.shape[1],
            )
            objective = -_selected_token_log_probs_from_hidden(
                model,
                hidden[:, :token_chunk],
                ids[:, 1 : token_chunk + 1],
                token_chunk_size=token_chunk,
            ).mean()
        elif bundle.stage_id == "opd_consolidation":
            input_ids = batch["input_ids"].long()
            output = _model_forward(
                forward_model,
                input_ids,
                attention_mask=batch["attention_mask"].bool(),
                canonical_ids=_canonicalize_ids(bundle, input_ids),
                return_logits=False,
            )
            hidden = _output_value(output, "final_hidden_state")[:, :-1]
            token_chunk = min(
                int(bundle.working_set["token_chunk_size"]),
                hidden.shape[1],
            )
            objective = _topk_union_reverse_kl(
                model=model,
                hidden_states=hidden[:, :token_chunk],
                teacher_union_token_ids=batch["teacher_union_token_ids"][
                    :, :token_chunk
                ].long(),
                teacher_union_logits=batch["teacher_union_logits"][
                    :, :token_chunk
                ].float(),
                teacher_union_count=batch["teacher_union_count"][
                    :, :token_chunk
                ].long(),
                response_mask=batch["response_mask"][:, :token_chunk].bool(),
                token_chunk_size=token_chunk,
                temperature=1.0,
            )
        else:
            raise StageBackendError(
                f"no kernel canary exists for {bundle.stage_id}"
            )
        if not isinstance(objective, Tensor):
            raise StageBackendError("kernel canary did not produce a tensor objective")
        auxiliary = _output_value(output, "auxiliary_loss", objective.detach() * 0.0)
        canary_loss = objective + auxiliary
        if not torch.isfinite(canary_loss):
            raise StageBackendError("kernel canary produced a non-finite loss")
        canary_loss.backward()
    if any(
        parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise StageBackendError("kernel canary produced a non-finite gradient")
    model.zero_grad(set_to_none=True)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    elapsed_seconds = time.perf_counter() - started
    payload: dict[str, Any] = {
        "schema": "metis.posttraining-kernel-canary/v1",
        "family": bundle.family,
        "stage": bundle.stage_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "batch_migration_sha256": bundle.batch_migration_sha256,
        "canonical_ids_sha256": bundle.canonical_ids_sha256,
        "compile_mode": compile_mode,
        "sequence_length": bundle.sequence_length,
        "runtime_batch": _runtime_batch_payload(bundle),
        "passed": True,
        "elapsed_seconds_local_rank": elapsed_seconds,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(
        payload, omit={"receipt_sha256"}
    )

    def write_on_rank() -> None:
        if topology.rank == 0:
            _atomic_json(path, payload)

    _collective_errors(
        topology,
        write_on_rank,
        label=f"{bundle.stage_id} kernel-canary receipt write",
    )
    barrier(topology)
    return payload


@dataclasses.dataclass(frozen=True)
class StageProgress:
    epoch: int
    next_global_batch: int
    optimizer_step: int
    campaign_token_cursor: int
    loss: float
    grad_norm: float
    metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)


def _batch_position_after(
    bundle: MMapStageBundle,
    topology: ParallelTopology,
    *,
    epoch: int,
    global_batch: int,
) -> tuple[int, int]:
    batches_per_epoch = bundle.records // (
        int(bundle.training["micro_batch_size"]) * topology.world_size
    )
    next_batch = global_batch + 1
    next_epoch = epoch
    if next_batch == batches_per_epoch:
        next_epoch += 1
        next_batch = 0
    return next_epoch, next_batch


def _total_optimizer_steps(
    bundle: MMapStageBundle,
    topology: ParallelTopology,
) -> int:
    global_micro_batch = int(bundle.training["micro_batch_size"]) * topology.world_size
    global_batches = bundle.records // global_micro_batch
    accumulation = int(bundle.training["gradient_accumulation"])
    if global_batches % accumulation:
        raise StageBackendError(
            f"{bundle.stage_id} batches per epoch must be divisible by gradient_accumulation"
        )
    return int(bundle.training["epochs"]) * global_batches // accumulation


def _optional_tensor(
    batch: Mapping[str, Tensor],
    name: str,
    default: Tensor | None = None,
) -> Tensor | None:
    value = batch.get(name)
    return value if value is not None else default


def _run_supervised_stage(
    *,
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
    model: nn.Module,
    optimizer: OptimizerBundle,
    runtime: Runtime,
    topology: ParallelTopology,
    start_epoch: int,
    start_global_batch: int,
    start_optimizer_step: int,
    start_cursor: int,
    checkpoint_callback: Any,
    signal_coordinator: SignalCoordinator | None = None,
    forward_model: nn.Module | None = None,
    resume_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    expected_sequence = int(stage["sequence_length"])
    if bundle.sequence_length != expected_sequence:
        raise StageBackendError(
            f"{stage['id']} requires sequence length {expected_sequence}, "
            f"bundle has {bundle.sequence_length}"
        )
    if stage["id"] == "context_extension":
        expected_tokens = int(stage["token_budget"])
        unique_active_tokens = int(
            bundle.manifest.get("unique_active_tokens", -1)
        )
        epochs = int(bundle.training["epochs"])
        total_exposure = int(bundle.manifest.get("training_tokens", -1))
        if (
            unique_active_tokens <= 0
            or unique_active_tokens * epochs != expected_tokens
            or total_exposure != expected_tokens
        ):
            raise StageBackendError(
                "context-extension bundle does not seal exact locked exposure: "
                f"unique_active_tokens * epochs must equal {expected_tokens}"
            )
    total_steps = _total_optimizer_steps(bundle, topology)
    accumulation = int(bundle.training["gradient_accumulation"])
    checkpoint_interval = int(bundle.training["checkpoint_interval_steps"])
    optimizer_step = start_optimizer_step
    campaign_cursor = start_cursor
    accumulated_weight = 0
    accumulated_tokens = 0
    accumulation_index = 0
    expert_selection_counts: Tensor | None = None
    prior_metrics = resume_metrics or {}
    last_loss = float(prior_metrics.get("loss", 0.0))
    last_grad_norm = float(prior_metrics.get("grad_norm", 0.0))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    execution_model = forward_model if forward_model is not None else model
    execution_model.train()
    context_gates: tuple[int, ...] = ()
    pending_context_gates: list[int] = []
    maximum_gate_overshoot = 0
    if stage["id"] == "context_extension":
        context_gates = tuple(int(value) for value in stage["checkpoint_gates"])
        maximum_gate_overshoot = int(
            _require_mapping(
                stage["gate_policy"], "context_extension.gate_policy"
            )["maximum_gate_overshoot_tokens"]
        )
        if (
            not context_gates
            or tuple(sorted(set(context_gates))) != context_gates
            or context_gates[-1] != int(stage["token_budget"])
            or start_cursor < BASE_TOKEN_CURSOR
        ):
            raise StageBackendError(
                "context-extension checkpoint gates are invalid"
            )
        completed_context_tokens = start_cursor - BASE_TOKEN_CURSOR
        pending_context_gates = [
            target
            for target in context_gates[:-1]
            if target > completed_context_tokens
        ]

    for epoch, global_batch, cpu_batch in bundle.iter_rank_batches(
        topology,
        start_epoch=start_epoch,
        start_global_batch=start_global_batch,
    ):
        batch = _to_device(cpu_batch, runtime.device)
        input_ids = batch["input_ids"].long()
        labels = batch["labels"].long()
        loss_mask = batch["loss_mask"].bool()
        attention = batch["attention_mask"].bool()
        document_ids = batch["document_ids"].long()
        reset_mask = batch["reset_mask"].bool()
        kwargs: dict[str, Any] = {
            "labels": _align_supervised_labels(
                labels,
                loss_mask,
                attention,
                document_ids,
                reset_mask,
            ),
            "attention_mask": attention,
            "document_ids": document_ids,
            "reset_mask": reset_mask,
            "canonical_ids": batch["canonical_ids"].long(),
            "return_logits": False,
        }
        output = execution_model(input_ids, **kwargs)
        expert_selection_counts = _accumulate_expert_selection_counts(
            expert_selection_counts, output
        )
        objective = _output_value(output, "loss")
        if not isinstance(objective, Tensor):
            raise StageBackendError(
                "Metis model did not return its chunked supervised loss"
            )
        auxiliary = _output_value(
            output,
            "auxiliary_loss",
            objective.detach() * 0.0,
        )
        supervised = int((kwargs["labels"] != -100).sum().item())
        if supervised <= 0:
            raise StageBackendError(f"{stage['id']} batch has no supervised tokens")
        loss = objective + auxiliary
        if not torch.isfinite(loss):
            raise StageBackendError(f"{stage['id']} produced a non-finite loss")
        (loss * supervised).backward()
        accumulated_weight += supervised
        accumulated_tokens += int(attention.sum().item())
        accumulation_index += 1
        last_loss = float(loss.detach().float().item())
        next_epoch, next_batch = _batch_position_after(
            bundle, topology, epoch=epoch, global_batch=global_batch
        )
        if accumulation_index < accumulation:
            continue
        learning_rate = _scheduled_learning_rate(
            bundle.training,
            optimizer_step=optimizer_step,
            total_steps=total_steps,
        )
        _set_learning_rate(optimizer, learning_rate)
        _, last_grad_norm = _main_optimizer_step(
            model=model,
            optimizer=optimizer,
            topology=topology,
            runtime=runtime,
            local_weight=accumulated_weight,
            gradient_clip=float(bundle.training["gradient_clip"]),
            expert_selection_counts=expert_selection_counts,
        )
        emitted = _all_reduce_int(accumulated_tokens, topology, runtime.device)
        campaign_cursor += emitted
        optimizer_step += 1
        accumulation_index = 0
        accumulated_weight = 0
        accumulated_tokens = 0
        expert_selection_counts = None
        progress = StageProgress(
            epoch=next_epoch,
            next_global_batch=next_batch,
            optimizer_step=optimizer_step,
            campaign_token_cursor=campaign_cursor,
            loss=last_loss,
            grad_norm=last_grad_norm,
            metrics={
                "loss": last_loss,
                "grad_norm": last_grad_norm,
            },
        )
        checkpoint_written = False
        if context_gates:
            context_tokens = campaign_cursor - BASE_TOKEN_CURSOR
            crossed = [
                target
                for target in pending_context_gates
                if context_tokens >= target
            ]
            if len(crossed) > 1:
                raise StageBackendError(
                    "one context optimizer boundary crossed multiple gates"
                )
            if crossed:
                target = crossed[0]
                if context_tokens - target > maximum_gate_overshoot:
                    raise StageBackendError(
                        "context gate exceeded its maximum optimizer-boundary "
                        "overshoot"
                    )
                checkpoint_callback(
                    progress,
                    False,
                    target,
                )
                pending_context_gates.remove(target)
                checkpoint_written = True
        if _signal_requested(
            signal_coordinator, topology=topology, runtime=runtime
        ):
            if not checkpoint_written:
                checkpoint_callback(progress, False)
            raise PostTrainingRequeue(
                signal_coordinator.reason or "remote-rank-signal"
            )
        if (
            not checkpoint_written
            and
            checkpoint_interval > 0
            and optimizer_step % checkpoint_interval == 0
            and optimizer_step < total_steps
        ):
            checkpoint_callback(progress, False)

    if accumulation_index:
        raise StageBackendError("stage ended with an incomplete gradient accumulation")
    if optimizer_step != total_steps:
        raise StageBackendError(
            f"{stage['id']} ended at optimizer step {optimizer_step}, expected {total_steps}"
        )
    result = {
        "optimizer_steps": optimizer_step,
        "campaign_token_cursor": campaign_cursor,
        "loss": last_loss,
        "grad_norm": last_grad_norm,
        "objective": str(_require_mapping(stage["objective"], "objective")["name"]),
    }
    if context_gates:
        final_context_tokens = campaign_cursor - BASE_TOKEN_CURSOR
        if final_context_tokens != int(stage["token_budget"]):
            raise StageBackendError(
                "context extension did not emit its exact locked token budget"
            )
        result.update(
            {
                "context_tokens": final_context_tokens,
                "checkpoint_gate_count": len(context_gates),
                "final_checkpoint_gate": context_gates[-1],
            }
        )
    return result


def _evaluate_context_checkpoint(
    *,
    bundle: MMapStageBundle,
    model: nn.Module,
    forward_model: nn.Module,
    runtime: Runtime,
    topology: ParallelTopology,
) -> dict[str, float]:
    calibration = _require_mapping(
        bundle.manifest.get("long_range_calibration"),
        "context long_range_calibration",
    )
    records = int(calibration["sample_records"])
    context = int(calibration["context"])
    tail_tokens = int(calibration["tail_tokens"])
    prefix_tokens = int(calibration["probe_prefix_tokens"])
    if (
        records % topology.world_size
        or context != 131_072
        or tail_tokens != 4_096
        or not 2 <= prefix_tokens <= 32
    ):
        raise StageBackendError(
            "context gate evaluation cannot divide exactly over this family"
        )
    local_indices = np.arange(
        topology.rank,
        records,
        topology.world_size,
        dtype=np.int64,
    )
    if local_indices.size != records // topology.world_size:
        raise StageBackendError(
            "context gate evaluation rank partition is imbalanced"
        )
    prior_model_training = model.training
    prior_forward_training = forward_model.training
    model.eval()
    forward_model.eval()
    base_loss_sum = 0.0
    base_targets = 0
    long_loss_sum = 0.0
    long_targets = 0
    short_loss_sum = 0.0
    short_targets = 0
    probe_log_probability_sum = 0.0
    probe_top1 = 0
    try:
        with torch.inference_mode():
            for raw_index in local_indices:
                index = int(raw_index)
                cpu_ids = np.array(
                    bundle.arrays["context_evaluation_input_ids"][
                        index : index + 1
                    ],
                    dtype=np.int64,
                    copy=True,
                )
                ids = torch.from_numpy(cpu_ids).to(
                    device=runtime.device,
                    dtype=torch.long,
                )
                attention = torch.ones_like(ids, dtype=torch.bool)
                document_ids = torch.zeros_like(ids, dtype=torch.int32)
                reset = torch.zeros_like(ids, dtype=torch.bool)
                reset[:, 0] = True
                long_labels = torch.full_like(ids, -100)
                tail_start = context - tail_tokens
                target_end = context - prefix_tokens - 1
                long_labels[:, tail_start:target_end] = ids[
                    :, tail_start + 1 : target_end + 1
                ]
                full_output = forward_model(
                    ids,
                    labels=long_labels,
                    attention_mask=attention,
                    document_ids=document_ids,
                    reset_mask=reset,
                    canonical_ids=_canonicalize_ids(bundle, ids),
                    curriculum={"stochastic_routing": False},
                    return_logits=False,
                )
                long_loss = _output_value(full_output, "loss")
                hidden = _output_value(
                    full_output, "final_hidden_state"
                )
                local_long_targets = int(
                    (long_labels != -100).sum().item()
                )
                if (
                    not isinstance(long_loss, Tensor)
                    or not isinstance(hidden, Tensor)
                    or local_long_targets <= 0
                    or not torch.isfinite(long_loss)
                ):
                    raise StageBackendError(
                        "context long evaluation produced invalid outputs"
                    )
                long_loss_sum += (
                    float(long_loss.float().item())
                    * local_long_targets
                )
                long_targets += local_long_targets

                probe_position = int(
                    bundle.arrays[
                        "context_evaluation_probe_positions"
                    ][index]
                )
                probe_target = int(
                    bundle.arrays[
                        "context_evaluation_probe_target_ids"
                    ][index]
                )
                if probe_position != context - 1:
                    raise StageBackendError(
                        "context associative probe position changed"
                    )
                with _head_execution_context(model):
                    probe_logits = _lm_head(model)(
                        hidden[:, probe_position : probe_position + 1]
                    ).float()[:, 0]
                probe_log_probs = F.log_softmax(
                    probe_logits, dim=-1
                )
                probe_log_probability_sum += float(
                    probe_log_probs[0, probe_target].item()
                )
                probe_top1 += int(
                    int(probe_logits.argmax(dim=-1).item())
                    == probe_target
                )

                short_ids = ids[:, tail_start : context - prefix_tokens]
                short_attention = torch.ones_like(
                    short_ids, dtype=torch.bool
                )
                short_labels = torch.full_like(short_ids, -100)
                short_labels[:, :-1] = short_ids[:, 1:]
                short_output = forward_model(
                    short_ids,
                    labels=short_labels,
                    attention_mask=short_attention,
                    document_ids=torch.zeros_like(
                        short_ids, dtype=torch.int32
                    ),
                    reset_mask=torch.nn.functional.pad(
                        torch.ones(
                            (1, 1),
                            device=runtime.device,
                            dtype=torch.bool,
                        ),
                        (0, short_ids.shape[1] - 1),
                        value=False,
                    ),
                    canonical_ids=_canonicalize_ids(
                        bundle, short_ids
                    ),
                    curriculum={"stochastic_routing": False},
                    return_logits=False,
                )
                short_loss = _output_value(short_output, "loss")
                local_short_targets = int(
                    (short_labels != -100).sum().item()
                )
                if (
                    not isinstance(short_loss, Tensor)
                    or local_short_targets != local_long_targets
                    or not torch.isfinite(short_loss)
                ):
                    raise StageBackendError(
                        "context short-tail evaluation is invalid"
                    )
                short_loss_sum += (
                    float(short_loss.float().item())
                    * local_short_targets
                )
                short_targets += local_short_targets

                base_ids = ids[:, :tail_tokens]
                base_labels = torch.full_like(base_ids, -100)
                base_labels[:, :-1] = base_ids[:, 1:]
                base_output = forward_model(
                    base_ids,
                    labels=base_labels,
                    attention_mask=torch.ones_like(
                        base_ids, dtype=torch.bool
                    ),
                    document_ids=torch.zeros_like(
                        base_ids, dtype=torch.int32
                    ),
                    reset_mask=torch.nn.functional.pad(
                        torch.ones(
                            (1, 1),
                            device=runtime.device,
                            dtype=torch.bool,
                        ),
                        (0, base_ids.shape[1] - 1),
                        value=False,
                    ),
                    canonical_ids=_canonicalize_ids(bundle, base_ids),
                    curriculum={"stochastic_routing": False},
                    return_logits=False,
                )
                base_loss = _output_value(base_output, "loss")
                local_base_targets = int(
                    (base_labels != -100).sum().item()
                )
                if (
                    not isinstance(base_loss, Tensor)
                    or local_base_targets <= 0
                    or not torch.isfinite(base_loss)
                ):
                    raise StageBackendError(
                        "context base-window evaluation is invalid"
                    )
                base_loss_sum += (
                    float(base_loss.float().item())
                    * local_base_targets
                )
                base_targets += local_base_targets
    finally:
        model.train(prior_model_training)
        forward_model.train(prior_forward_training)
    reduced = _all_reduce_float_vector(
        [
            base_loss_sum,
            float(base_targets),
            long_loss_sum,
            float(long_targets),
            short_loss_sum,
            float(short_targets),
            probe_log_probability_sum,
            float(probe_top1),
            float(local_indices.size),
        ],
        topology=topology,
        runtime=runtime,
    )
    if (
        reduced[1] <= 0
        or reduced[3] <= 0
        or reduced[5] <= 0
        or int(reduced[8]) != records
    ):
        raise StageBackendError(
            "context gate evaluation did not cover its exact sealed sample"
        )
    metrics = {
        "base_window_nll": reduced[0] / reduced[1],
        "full_context_tail_nll": reduced[2] / reduced[3],
        "short_context_tail_nll": reduced[4] / reduced[5],
        "long_context_nll_gain": (
            reduced[4] / reduced[5] - reduced[2] / reduced[3]
        ),
        "needle_target_log_probability": reduced[6] / reduced[8],
        "needle_top1_accuracy": reduced[7] / reduced[8],
        "evaluation_records": reduced[8],
        "evaluation_context": float(context),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise StageBackendError(
            "context gate evaluation produced non-finite metrics"
        )
    return metrics


def _context_gate_decision(
    *,
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    gate_target_tokens: int,
    gate_policy: Mapping[str, Any],
) -> dict[str, Any]:
    base_tolerance = float(
        gate_policy["base_validation_relative_tolerance"]
    )
    long_tolerance = float(
        gate_policy["long_validation_relative_tolerance"]
    )
    base_pass = metrics["base_window_nll"] <= (
        baseline["base_window_nll"] * (1.0 + base_tolerance)
    )
    long_pass = metrics["full_context_tail_nll"] <= (
        baseline["full_context_tail_nll"] * (1.0 + long_tolerance)
    )
    needle_delta = (
        metrics["needle_target_log_probability"]
        - baseline["needle_target_log_probability"]
    )
    needle_pass = (
        needle_delta > 0.0
        if gate_target_tokens == 6_000_000_000
        and gate_policy.get("require_needle_gain_after_first_gate") is True
        else needle_delta >= 0.0
    )
    long_quality = math.exp(
        max(
            -20.0,
            min(
                20.0,
                (
                    baseline["full_context_tail_nll"]
                    - metrics["full_context_tail_nll"]
                )
                / max(baseline["full_context_tail_nll"], 1.0e-12),
            ),
        )
    )
    needle_quality = math.exp(max(-20.0, min(20.0, needle_delta)))
    promotion_score = (
        2.0
        * long_quality
        * needle_quality
        / max(long_quality + needle_quality, 1.0e-12)
    )
    return {
        "base_retention_passed": base_pass,
        "long_validation_passed": long_pass,
        "needle_validation_passed": needle_pass,
        "passed": bool(base_pass and long_pass and needle_pass),
        "needle_log_probability_delta": needle_delta,
        "promotion_score": promotion_score,
    }


@dataclasses.dataclass(frozen=True)
class ProfileSelection:
    profile: Mapping[str, float]
    profile_sha256: str
    evidence_sha256: str
    selected_metrics: Mapping[str, float]
    gate_name: str


def _normalized_profile(
    raw: Any,
    *,
    fields: Sequence[str],
    label: str,
) -> dict[str, float]:
    profile = _require_mapping(raw, label)
    if set(profile) != set(fields):
        raise StageBackendError(
            f"{label} must contain exactly {', '.join(fields)}"
        )
    result: dict[str, float] = {}
    for field in fields:
        value = profile[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StageBackendError(f"{label}.{field} must be numeric")
        cooked = float(value)
        if not math.isfinite(cooked) or cooked < 0:
            raise StageBackendError(f"{label}.{field} must be finite and non-negative")
        result[field] = cooked
    return result


def _metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    integral: bool = False,
    positive: bool = False,
) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageBackendError(f"profile trial metric {name} must be numeric")
    cooked = float(value)
    if not math.isfinite(cooked):
        raise StageBackendError(f"profile trial metric {name} must be finite")
    if integral and (cooked < 0 or not cooked.is_integer()):
        raise StageBackendError(
            f"profile trial metric {name} must be a non-negative integer"
        )
    if positive and cooked <= 0:
        raise StageBackendError(f"profile trial metric {name} must be positive")
    return cooked


def _validated_profile_selection(
    *,
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
    candidates: Sequence[Mapping[str, float]],
    fields: Sequence[str],
    gate_name: str,
    eligible: Any,
    ordering: Any,
) -> ProfileSelection:
    evidence = _require_mapping(
        bundle.manifest.get("profile_selection"),
        f"{stage['id']} profile_selection",
    )
    candidate_rows = [dict(row) for row in candidates]
    if (
        evidence.get("schema") != PROFILE_SELECTION_SCHEMA
        or evidence.get("stage") != stage["id"]
        or evidence.get("parent_checkpoint_sha256")
        != bundle.manifest.get("parent_checkpoint_sha256")
        or evidence.get("candidate_set_sha256")
        != _canonical_hash(candidate_rows)
        or not _is_sha256(evidence.get("evaluator_sha256"))
        or not _is_sha256(evidence.get("evaluation_dataset_sha256"))
        or evidence.get("selection_sha256")
        != _canonical_hash(evidence, omit={"selection_sha256"})
    ):
        raise StageBackendError(
            f"{stage['id']} profile-selection evidence lineage is invalid"
        )
    expected = {
        _canonical_hash(dict(candidate)): dict(candidate)
        for candidate in candidate_rows
    }
    observed: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for index, raw_trial in enumerate(
        _require_list(evidence.get("trials"), f"{stage['id']} profile trials")
    ):
        trial = _require_mapping(raw_trial, f"{stage['id']} profile trial {index}")
        profile = _normalized_profile(
            trial.get("profile"),
            fields=fields,
            label=f"{stage['id']} profile trial {index}.profile",
        )
        profile_sha = _canonical_hash(profile)
        metrics = _require_mapping(
            trial.get("metrics"),
            f"{stage['id']} profile trial {index}.metrics",
        )
        if (
            trial.get("profile_sha256") != profile_sha
            or profile_sha not in expected
            or profile_sha in observed
            or not _is_sha256(trial.get("candidate_checkpoint_sha256"))
            or not _is_sha256(trial.get("evaluation_receipt_sha256"))
            or trial.get("metrics_sha256") != _canonical_hash(metrics)
            or trial.get("trial_sha256")
            != _canonical_hash(trial, omit={"trial_sha256"})
        ):
            raise StageBackendError(
                f"{stage['id']} profile trial {index} is not cryptographically bound"
            )
        parsed_metrics = {
            str(name): _metric(metrics, str(name))
            for name in metrics
        }
        observed[profile_sha] = profile, parsed_metrics
    if set(observed) != set(expected):
        raise StageBackendError(
            f"{stage['id']} must evaluate its exact deterministic candidate set"
        )
    passing = [
        (profile_sha, profile, metrics)
        for profile_sha, (profile, metrics) in observed.items()
        if eligible(metrics)
    ]
    if not passing:
        raise StageBackendError(f"{stage['id']} has no candidate that passes {gate_name}")
    winner_sha, winner_profile, winner_metrics = sorted(
        passing,
        key=lambda row: (*ordering(row[2]), row[0]),
    )[0]
    if evidence.get("selected_profile_sha256") != winner_sha:
        raise StageBackendError(
            f"{stage['id']} selected profile is not the derived gate winner"
        )
    raw_selected = bundle.manifest.get("selected_profile")
    if raw_selected is not None and _normalized_profile(
        raw_selected,
        fields=fields,
        label=f"{stage['id']} selected_profile",
    ) != winner_profile:
        raise StageBackendError(
            f"{stage['id']} selected_profile differs from its trial winner"
        )
    if bundle.manifest.get("selected_profile_sha256") not in {None, winner_sha}:
        raise StageBackendError(
            f"{stage['id']} selected_profile_sha256 differs from its trial winner"
        )
    return ProfileSelection(
        profile=winner_profile,
        profile_sha256=winner_sha,
        evidence_sha256=str(evidence["selection_sha256"]),
        selected_metrics=winner_metrics,
        gate_name=gate_name,
    )


def _run_opd_stage(
    *,
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
    model: nn.Module,
    optimizer: OptimizerBundle,
    runtime: Runtime,
    topology: ParallelTopology,
    start_epoch: int,
    start_global_batch: int,
    start_optimizer_step: int,
    start_cursor: int,
    checkpoint_callback: Any,
    signal_coordinator: SignalCoordinator | None = None,
    forward_model: nn.Module | None = None,
    resume_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if (
        stage["id"] != "opd_consolidation"
        or bundle.stage_id != "opd_consolidation"
        or bundle.sequence_length != int(stage["sequence_length"])
    ):
        raise StageBackendError("OPD stage/bundle contract changed")
    objective_config = _require_mapping(
        stage.get("objective"), "opd_consolidation.objective"
    )
    if (
        objective_config.get("divergence") != "reverse_kl"
        or int(objective_config.get("top_k_per_model", -1)) != 32
        or objective_config.get("union_student_and_teacher_top_k") is not True
    ):
        raise StageBackendError("OPD objective is not the locked top-k-union RKL")
    temperature = float(objective_config.get("temperature", 1.0))
    total_steps = _total_optimizer_steps(bundle, topology)
    accumulation = int(bundle.training["gradient_accumulation"])
    checkpoint_interval = int(bundle.training["checkpoint_interval_steps"])
    optimizer_step = start_optimizer_step
    campaign_cursor = start_cursor
    accumulated_weight = 0
    accumulated_tokens = 0
    accumulation_index = 0
    expert_selection_counts: Tensor | None = None
    last_loss = float((resume_metrics or {}).get("loss", 0.0))
    last_grad_norm = float((resume_metrics or {}).get("grad_norm", 0.0))
    route_counts = torch.zeros(
        len(SPECIALIST_STAGE_IDS),
        dtype=torch.float64,
        device=runtime.device,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    execution_model = forward_model if forward_model is not None else model
    execution_model.train()
    token_chunk = int(bundle.working_set["token_chunk_size"])

    for epoch, global_batch, cpu_batch in bundle.iter_rank_batches(
        topology,
        start_epoch=start_epoch,
        start_global_batch=start_global_batch,
    ):
        batch = _to_device(cpu_batch, runtime.device)
        input_ids = batch["input_ids"].long()
        attention = batch["attention_mask"].bool()
        response = batch["response_mask"].bool()
        output = _model_forward(
            execution_model,
            input_ids,
            attention_mask=attention,
            canonical_ids=_canonicalize_ids(bundle, input_ids),
            return_logits=False,
        )
        expert_selection_counts = _accumulate_expert_selection_counts(
            expert_selection_counts, output
        )
        hidden = _output_value(output, "final_hidden_state")[:, :-1]
        rkl = _topk_union_reverse_kl(
            model=model,
            hidden_states=hidden,
            teacher_union_token_ids=batch["teacher_union_token_ids"].long(),
            teacher_union_logits=batch["teacher_union_logits"].float(),
            teacher_union_count=batch["teacher_union_count"].long(),
            response_mask=response,
            token_chunk_size=token_chunk,
            temperature=temperature,
        )
        auxiliary = _output_value(output, "auxiliary_loss", rkl.detach() * 0.0)
        loss = rkl + auxiliary
        if not torch.isfinite(loss):
            raise StageBackendError("OPD consolidation produced a non-finite loss")
        supervised = int(response.sum().item())
        if supervised <= 0:
            raise StageBackendError("OPD consolidation batch has no response tokens")
        (loss * supervised).backward()
        accumulated_weight += supervised
        accumulated_tokens += int(attention.sum().item())
        accumulation_index += 1
        last_loss = float(loss.detach().float().item())
        route_counts.add_(
            torch.bincount(
                batch["teacher_route"].long(),
                minlength=len(SPECIALIST_STAGE_IDS),
            ).to(dtype=torch.float64)
        )
        next_epoch, next_batch = _batch_position_after(
            bundle, topology, epoch=epoch, global_batch=global_batch
        )
        if accumulation_index < accumulation:
            continue
        _set_learning_rate(
            optimizer,
            _scheduled_learning_rate(
                bundle.training,
                optimizer_step=optimizer_step,
                total_steps=total_steps,
            ),
        )
        _, last_grad_norm = _main_optimizer_step(
            model=model,
            optimizer=optimizer,
            topology=topology,
            runtime=runtime,
            local_weight=accumulated_weight,
            gradient_clip=float(bundle.training["gradient_clip"]),
            expert_selection_counts=expert_selection_counts,
        )
        campaign_cursor += _all_reduce_int(
            accumulated_tokens, topology, runtime.device
        )
        optimizer_step += 1
        accumulation_index = 0
        accumulated_weight = 0
        accumulated_tokens = 0
        expert_selection_counts = None
        progress = StageProgress(
            epoch=next_epoch,
            next_global_batch=next_batch,
            optimizer_step=optimizer_step,
            campaign_token_cursor=campaign_cursor,
            loss=last_loss,
            grad_norm=last_grad_norm,
            metrics={
                "loss": last_loss,
                "reverse_kl": last_loss,
                "grad_norm": last_grad_norm,
            },
        )
        if _signal_requested(
            signal_coordinator, topology=topology, runtime=runtime
        ):
            checkpoint_callback(progress, False)
            raise PostTrainingRequeue(
                signal_coordinator.reason or "remote-rank-signal"
            )
        if (
            checkpoint_interval > 0
            and optimizer_step % checkpoint_interval == 0
            and optimizer_step < total_steps
        ):
            checkpoint_callback(progress, False)
    if accumulation_index or optimizer_step != total_steps:
        raise StageBackendError("OPD did not consume its exact single-use bundle")
    if topology.initialized:
        dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
    route_total = float(route_counts.sum().item())
    if route_total <= 0:
        raise StageBackendError("OPD observed no specialist routes")
    route_probabilities = route_counts / route_total
    route_entropy = float(
        (
            -route_probabilities.clamp_min(1.0e-12)
            * route_probabilities.clamp_min(1.0e-12).log()
        ).sum().item()
        / math.log(len(SPECIALIST_STAGE_IDS))
    )
    return {
        "optimizer_steps": optimizer_step,
        "campaign_token_cursor": campaign_cursor,
        "loss": last_loss,
        "reverse_kl": last_loss,
        "grad_norm": last_grad_norm,
        "specialist_route_entropy_ratio": route_entropy,
        "specialist_routes": {
            stage_id: int(route_counts[index].item())
            for index, stage_id in enumerate(SPECIALIST_STAGE_IDS)
        },
    }


def _rlvr_rewards(
    stage_id: str,
    batch: Mapping[str, Tensor],
    *,
    length_coefficient: float,
    mode_compliance_weight: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    correctness = batch["correctness"].float()
    if not torch.isfinite(correctness).all() or torch.any(
        (correctness < 0) | (correctness > 1)
    ):
        raise StageBackendError("RLVR correctness must be finite in [0,1]")
    pass_rates = avg_at_k(correctness, k=16)
    in_band = strict_on_policy_filter(pass_rates, minimum=0.10, maximum=0.90)
    if not bool(in_band.all()):
        raise StageBackendError("RLVR bundle violates the strict 10%-90% on-policy filter")
    response_mask = batch["candidate_response_mask"].bool()
    lengths = response_mask.sum(dim=-1).float().clamp_min(1.0)
    correct_weight = correctness.sum(dim=-1).clamp_min(1.0)
    mean_correct_length = (lengths * correctness).sum(dim=-1) / correct_weight
    maximum_length = response_mask.shape[-1]
    think_shaped = difficulty_adaptive_length_reward(
        correctness=correctness,
        response_lengths=lengths,
        pass_rate=pass_rates[:, None],
        mean_correct_length=mean_correct_length[:, None],
        maximum_length=maximum_length,
        coefficient=length_coefficient,
        deadband_fraction=0.10,
    )
    reasoning_mode = batch["reasoning_mode"].long()
    if (
        reasoning_mode.shape != (correctness.shape[0],)
        or torch.any((reasoning_mode < 0) | (reasoning_mode > 2))
    ):
        raise StageBackendError("RLVR reasoning_mode must be [N] with IDs 0..2")
    think_prompt = reasoning_mode.eq(1).unsqueeze(-1)
    # Efficiency shaping is intentionally exclusive to ordinary think mode.
    # Direct responses have no visible reasoning trace to compress, while
    # think_max receives a larger generation budget without a verbosity reward.
    shaped = correctness + torch.where(
        think_prompt,
        think_shaped - correctness,
        torch.zeros_like(correctness),
    )
    diagnostics = thinking_length_diagnostics(
        correctness=correctness * think_prompt,
        response_lengths=lengths,
        pass_rate=pass_rates[:, None],
        mean_correct_length=mean_correct_length[:, None],
        maximum_length=maximum_length,
        deadband_fraction=0.10,
    )
    if stage_id == "specialist_code":
        if "efficiency_reward" not in batch:
            raise StageBackendError("code RLVR requires efficiency_reward")
        code = gated_code_efficiency_reward(
            correctness,
            batch["efficiency_reward"].float(),
        )
        shaped = code + (shaped - correctness)
    compliance = batch["mode_compliance"].float()
    if (
        compliance.shape != correctness.shape
        or not torch.isfinite(compliance).all()
        or torch.any((compliance < 0) | (compliance > 1))
        or not math.isfinite(mode_compliance_weight)
        or mode_compliance_weight <= 0
    ):
        raise StageBackendError("RLVR mode-compliance reward contract is invalid")
    shaped = shaped - mode_compliance_weight * (1.0 - compliance)
    return shaped, pass_rates, lengths, diagnostics


def _selected_gspo_profile(
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
) -> ProfileSelection:
    default_profile, candidates, gate = _gspo_profile_candidates(stage)
    del default_profile

    def eligible(metrics: Mapping[str, float]) -> bool:
        nonfinite = _metric(metrics, "nonfinite_steps", integral=True)
        _metric(metrics, "evaluation_records", integral=True, positive=True)
        _metric(metrics, "rollout_prompts", integral=True, positive=True)
        reward_gain = _metric(metrics, "reward_gain")
        entropy_delta = _metric(metrics, "entropy_delta")
        evaluation_regression = _metric(metrics, "evaluation_regression")
        return bool(
            reward_gain >= float(gate["minimum_reward_gain"])
            and abs(entropy_delta) <= float(gate["maximum_entropy_delta"])
            and evaluation_regression
            <= float(gate["maximum_evaluation_regression"])
            and nonfinite <= int(gate["maximum_nonfinite_steps"])
        )

    def ordering(metrics: Mapping[str, float]) -> tuple[float, ...]:
        reward_gain = _metric(metrics, "reward_gain")
        evaluation_regression = _metric(metrics, "evaluation_regression")
        entropy_delta = _metric(metrics, "entropy_delta")
        return (
            -reward_gain,
            evaluation_regression,
            abs(entropy_delta),
        )

    return _validated_profile_selection(
        stage=stage,
        bundle=bundle,
        candidates=candidates,
        fields=("clip_low", "clip_high", "length_coefficient"),
        gate_name=str(gate["name"]),
        eligible=eligible,
        ordering=ordering,
    )


def _gspo_profile_candidates(
    stage: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, float]], Mapping[str, Any]]:
    objective = _require_mapping(stage.get("objective"), f"{stage['id']}.objective")
    default_profile = {
        "clip_low": float(objective["clip_low"]),
        "clip_high": float(objective["clip_high"]),
        "length_coefficient": float(
            _require_mapping(stage.get("reward", {}), f"{stage['id']}.reward").get(
                "length_coefficient", 0.0
            )
        ),
    }
    bounds = _require_mapping(stage.get("autotune"), f"{stage['id']}.autotune")
    clip_pairs: list[tuple[float, float]] = []
    for index, raw_pair in enumerate(
        _require_list(bounds.get("clip_pairs"), f"{stage['id']}.autotune.clip_pairs")
    ):
        pair = _require_list(raw_pair, f"{stage['id']}.autotune.clip_pairs[{index}]")
        if len(pair) != 2:
            raise StageBackendError("GSPO clip candidate must contain low/high")
        clip_pairs.append((float(pair[0]), float(pair[1])))
    coefficients = [
        float(item)
        for item in _require_list(
            bounds.get("length_coefficients"),
            f"{stage['id']}.autotune.length_coefficients",
        )
    ]
    candidates = [
        {
            "clip_low": clip_low,
            "clip_high": clip_high,
            "length_coefficient": coefficient,
        }
        for (clip_low, clip_high), coefficient in product(
            clip_pairs, coefficients
        )
    ]
    if default_profile not in candidates:
        raise StageBackendError(
            f"{stage['id']} default profile is outside its candidate grid"
        )
    gate = _require_mapping(
        bounds.get("gate"), f"{stage['id']}.autotune.gate"
    )
    if gate.get("name") != "stable_reward_and_entropy_with_no_eval_regression":
        raise StageBackendError(f"{stage['id']} has an unsupported autotune gate")
    return default_profile, candidates, gate


@dataclasses.dataclass(frozen=True)
class _RNGSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: Tensor
    device_state: Tensor | None


def _capture_rng_state(runtime: Runtime) -> _RNGSnapshot:
    return _RNGSnapshot(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),
        torch_state=torch.random.get_rng_state().clone(),
        device_state=(
            torch.cuda.get_rng_state(runtime.device).clone()
            if runtime.device.type == "cuda"
            else None
        ),
    )


def _restore_rng_state(snapshot: _RNGSnapshot, runtime: Runtime) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.random.set_rng_state(snapshot.torch_state)
    if snapshot.device_state is not None:
        torch.cuda.set_rng_state(snapshot.device_state, runtime.device)


def _seed_live_trial(seed: int, runtime: Runtime) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed(seed)


def _all_reduce_float_vector(
    values: Sequence[float],
    *,
    topology: ParallelTopology,
    runtime: Runtime,
    operation: str = "sum",
) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=runtime.device)
    if topology.distributed:
        reduce_op = {
            "sum": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }.get(operation)
        if reduce_op is None:
            raise StageBackendError(f"unsupported live-autotune reduction {operation}")
        dist.all_reduce(tensor, op=reduce_op, group=topology.dense_data_group)
    return [float(item) for item in tensor.cpu().tolist()]


def _live_evaluation_indices(
    *,
    records: int,
    micro_batch_size: int,
    topology: ParallelTopology,
) -> Iterator[np.ndarray]:
    global_batch = micro_batch_size * topology.world_size
    if records <= 0 or records % global_batch:
        raise StageBackendError(
            "live evaluator records must be divisible by "
            "world_size*evaluation_micro_batch_size"
        )
    for offset in range(0, records, global_batch):
        local_start = offset + topology.rank * micro_batch_size
        yield np.arange(
            local_start,
            local_start + micro_batch_size,
            dtype=np.int64,
        )


def _eval_selected_token_log_probs(
    model: nn.Module,
    hidden_states: Tensor,
    target_ids: Tensor,
    *,
    token_chunk_size: int,
) -> Tensor:
    if hidden_states.shape[:2] != target_ids.shape:
        raise StageBackendError("live evaluator hidden states and targets do not align")
    head = _lm_head(model)
    chunks: list[Tensor] = []
    for start in range(0, hidden_states.shape[1], token_chunk_size):
        end = min(hidden_states.shape[1], start + token_chunk_size)
        with _head_execution_context(model):
            logits = head(hidden_states[:, start:end])
        logits = logits.float()
        chunks.append(
            logits.gather(
                -1, target_ids[:, start:end].unsqueeze(-1)
            ).squeeze(-1)
            - torch.logsumexp(logits, dim=-1)
        )
    return torch.cat(chunks, dim=1)


def _evaluate_live_gspo_profile(
    *,
    stage: Mapping[str, Any],
    profile: Mapping[str, float],
    bundle: MMapStageBundle,
    model: nn.Module,
    forward_model: nn.Module,
    runtime: Runtime,
    topology: ParallelTopology,
    evaluator: Mapping[str, Any],
) -> dict[str, float]:
    prefix = "autotune_evaluation_"
    records = int(evaluator["records"])
    micro_batch = int(evaluator.get("micro_batch_size", 1))
    if micro_batch <= 0:
        raise StageBackendError("live evaluator micro_batch_size must be positive")
    token_chunk = int(bundle.working_set["token_chunk_size"])
    micro_group = int(
        evaluator.get(
            "candidate_micro_group_size",
            bundle.working_set["candidate_micro_group_size"],
        )
    )
    if micro_group <= 0 or 16 % micro_group:
        raise StageBackendError(
            "live evaluator candidate_micro_group_size must divide 16"
        )
    local_reward = 0.0
    local_entropy = 0.0
    local_correct_nll = 0.0
    local_correct = 0.0
    local_prompts = 0.0
    was_training = model.training
    model.eval()
    forward_model.eval()
    try:
        with torch.no_grad():
            for indices in _live_evaluation_indices(
                records=records,
                micro_batch_size=micro_batch,
                topology=topology,
            ):
                cpu_batch = {
                    name: np.array(array[indices], copy=True)
                    for name, array in bundle.arrays.items()
                    if name.startswith(prefix)
                }
                batch = _to_device(cpu_batch, runtime.device)
                ids = batch[f"{prefix}candidate_input_ids"].long()
                attention = batch[f"{prefix}candidate_attention_mask"].bool()
                response = batch[f"{prefix}candidate_response_mask"].bool()
                correctness = batch[f"{prefix}correctness"].float()
                truncated = batch[f"{prefix}truncated"].bool()
                reward_batch: dict[str, Tensor] = {
                    "correctness": correctness,
                    "candidate_response_mask": response,
                    "reasoning_mode": batch[
                        f"{prefix}reasoning_mode"
                    ].long(),
                    "mode_compliance": batch[
                        f"{prefix}mode_compliance"
                    ].float(),
                }
                if stage["id"] == "specialist_code":
                    reward_batch["efficiency_reward"] = batch[
                        f"{prefix}efficiency_reward"
                    ].float()
                rewards, _pass_rates, _lengths, _thinking = _rlvr_rewards(
                    str(stage["id"]),
                    reward_batch,
                    length_coefficient=float(profile["length_coefficient"]),
                    mode_compliance_weight=float(
                        _require_mapping(
                            stage.get("reward"), f"{stage['id']}.reward"
                        )["mode_compliance_weight"]
                    ),
                )
                sequence_log_probs = torch.empty(
                    (ids.shape[0], 16),
                    dtype=torch.float32,
                    device=runtime.device,
                )
                for group_start in range(0, 16, micro_group):
                    group_end = group_start + micro_group
                    flat_ids = ids[:, group_start:group_end].reshape(
                        -1, bundle.sequence_length
                    )
                    flat_attention = attention[
                        :, group_start:group_end
                    ].reshape(-1, bundle.sequence_length)
                    output = _model_forward(
                        forward_model,
                        flat_ids,
                        attention_mask=flat_attention,
                        canonical_ids=_canonicalize_ids(bundle, flat_ids),
                        deterministic=True,
                        return_logits=False,
                    )
                    hidden = _output_value(
                        output, "final_hidden_state"
                    )[:, :-1]
                    token_log_probs = _eval_selected_token_log_probs(
                        model,
                        hidden,
                        flat_ids[:, 1:],
                        token_chunk_size=token_chunk,
                    ).reshape(
                        ids.shape[0],
                        micro_group,
                        bundle.sequence_length - 1,
                    )
                    mask = response[:, group_start:group_end]
                    sequence_log_probs[:, group_start:group_end] = (
                        (token_log_probs * mask).sum(dim=-1)
                        / mask.sum(dim=-1).clamp_min(1)
                    )
                valid_candidates = ~truncated
                if not bool(valid_candidates.any(dim=-1).all()):
                    raise StageBackendError(
                        f"{stage['id']} live evaluator prompt has no "
                        "non-truncated candidate"
                    )
                policy_logits = sequence_log_probs.masked_fill(
                    ~valid_candidates, float("-inf")
                )
                probabilities = torch.softmax(policy_logits, dim=-1)
                entropy = -(
                    probabilities * torch.log(probabilities.clamp_min(1.0e-12))
                ).sum(dim=-1)
                expected_reward = (probabilities * rewards.float()).sum(dim=-1)
                correct = (correctness > 0.5) & valid_candidates
                correct_count = int(correct.sum().item())
                if correct_count <= 0:
                    raise StageBackendError(
                        "RLVR live evaluator batch has no verified-correct response"
                    )
                if (
                    not torch.isfinite(expected_reward).all()
                    or not torch.isfinite(entropy).all()
                    or not torch.isfinite(sequence_log_probs).all()
                ):
                    raise StageBackendError(
                        f"{stage['id']} live evaluator produced non-finite metrics"
                    )
                local_reward += float(expected_reward.sum().item())
                local_entropy += float(entropy.sum().item())
                local_correct_nll += float(
                    (-sequence_log_probs[correct]).sum().item()
                )
                local_correct += float(correct_count)
                local_prompts += float(ids.shape[0])
    finally:
        model.train(was_training)
        forward_model.train(was_training)
    values = _all_reduce_float_vector(
        [
            local_reward,
            local_entropy,
            local_correct_nll,
            local_correct,
            local_prompts,
        ],
        topology=topology,
        runtime=runtime,
    )
    if int(values[4]) != records or values[3] <= 0:
        raise StageBackendError(
            f"{stage['id']} live evaluator did not reproduce its sealed records"
        )
    return {
        "expected_reward": values[0] / values[4],
        "entropy": values[1] / values[4],
        "correct_response_nll": values[2] / values[3],
        "evaluation_records": values[3],
        "rollout_prompts": values[4],
    }


def _live_metrics_from_evaluations(
    *,
    stage_id: str,
    parent: Mapping[str, float],
    candidate: Mapping[str, float],
    nonfinite_steps: int,
) -> dict[str, float]:
    if int(parent["rollout_prompts"]) != int(candidate["rollout_prompts"]):
        raise StageBackendError("GSPO live evaluator prompt count changed")
    return {
        "reward_gain": (
            float(candidate["expected_reward"]) - float(parent["expected_reward"])
        ),
        "entropy_delta": float(candidate["entropy"]) - float(parent["entropy"]),
        "evaluation_regression": (
            float(candidate["correct_response_nll"])
            - float(parent["correct_response_nll"])
        ),
        "nonfinite_steps": float(nonfinite_steps),
        "evaluation_records": float(candidate["evaluation_records"]),
        "rollout_prompts": float(candidate["rollout_prompts"]),
    }


def _live_profile_gate(
    stage: Mapping[str, Any],
    metrics: Mapping[str, float],
) -> tuple[bool, tuple[float, ...], str]:
    _default, _candidates, gate = _gspo_profile_candidates(stage)
    reward = _metric(metrics, "reward_gain")
    entropy = _metric(metrics, "entropy_delta")
    regression = _metric(metrics, "evaluation_regression")
    nonfinite = _metric(metrics, "nonfinite_steps", integral=True)
    _metric(metrics, "evaluation_records", integral=True, positive=True)
    _metric(metrics, "rollout_prompts", integral=True, positive=True)
    passed = bool(
        reward >= float(gate["minimum_reward_gain"])
        and abs(entropy) <= float(gate["maximum_entropy_delta"])
        and regression <= float(gate["maximum_evaluation_regression"])
        and nonfinite <= int(gate["maximum_nonfinite_steps"])
    )
    return (
        passed,
        (-reward, regression, abs(entropy)),
        str(gate["name"]),
    )


def _runtime_rank_inventory(
    *,
    runtime: Runtime,
    topology: ParallelTopology,
    compile_mode: str,
) -> tuple[list[dict[str, Any]], str]:
    device_name = str(runtime.device)
    total_memory = 0
    if runtime.device.type == "cuda":
        properties = torch.cuda.get_device_properties(runtime.device)
        device_name = str(properties.name)
        total_memory = int(properties.total_memory)
    local = {
        "rank": topology.rank,
        "local_rank": topology.local_rank,
        "device": device_name,
        "device_total_bytes": total_memory,
        "torch_version": str(torch.__version__),
        "hip_version": str(getattr(torch.version, "hip", None)),
        "compile_mode": compile_mode,
    }
    if topology.distributed:
        gathered: list[Any] = [None for _ in range(topology.world_size)]
        dist.all_gather_object(
            gathered, local, group=topology.dense_data_group
        )
        rows = [
            dict(_require_mapping(item, "live-autotune runtime rank"))
            for item in gathered
        ]
    else:
        rows = [local]
    rows.sort(key=lambda item: int(item["rank"]))
    return rows, _canonical_hash(rows)


def _profile_trial_state_fingerprint(
    model: nn.Module,
    *,
    topology: ParallelTopology,
) -> str:
    """Cheap execution fingerprint; the durable parent remains the full checkpoint.

    Hashing every 12B-parameter trial would turn the canary into an I/O
    benchmark.  This fingerprint samples deterministic endpoints/midpoints
    from every local tensor and is used only to bind the observed execution;
    it is never represented as a checkpoint hash.
    """

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            flat = parameter.detach().reshape(-1)
            if flat.numel() == 0:
                sample: list[float] = []
            else:
                positions = sorted(
                    {
                        0,
                        flat.numel() // 4,
                        flat.numel() // 2,
                        (3 * flat.numel()) // 4,
                        flat.numel() - 1,
                    }
                )
                position_tensor = torch.tensor(
                    positions, dtype=torch.long, device=flat.device
                )
                sample = [
                    float(value)
                    for value in flat.index_select(
                        0, position_tensor
                    ).float().cpu().tolist()
                ]
                if not all(math.isfinite(value) for value in sample):
                    raise StageBackendError(
                        f"live trial parameter sample is non-finite: {name}"
                    )
            rows.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "sample": sample,
                }
            )
    local = {
        "rank": topology.rank,
        "parameters_sha256": _canonical_hash(rows),
    }
    if topology.distributed:
        gathered: list[Any] = [None for _ in range(topology.world_size)]
        dist.all_gather_object(
            gathered, local, group=topology.dense_data_group
        )
        inventory = sorted(
            (
                dict(_require_mapping(item, "live trial state fingerprint"))
                for item in gathered
            ),
            key=lambda item: int(item["rank"]),
        )
    else:
        inventory = [local]
    return _canonical_hash(inventory)


def _expected_profile_trials(
    evidence: Mapping[str, Any],
    *,
    fields: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _require_list(evidence.get("trials"), "profile-selection trials")
    ):
        trial = _require_mapping(raw, f"profile-selection trial {index}")
        profile = _normalized_profile(
            trial.get("profile"),
            fields=fields,
            label=f"profile-selection trial {index}.profile",
        )
        profile_sha = _canonical_hash(profile)
        metrics = _require_mapping(
            trial.get("metrics"),
            f"profile-selection trial {index}.metrics",
        )
        if (
            trial.get("profile_sha256") != profile_sha
            or trial.get("metrics_sha256") != _canonical_hash(metrics)
            or trial.get("trial_sha256")
            != _canonical_hash(trial, omit={"trial_sha256"})
            or profile_sha in result
        ):
            raise StageBackendError(
                "profile-selection expected trial lineage is invalid"
            )
        result[profile_sha] = trial
    return result


def _metrics_reproduce(
    live: Mapping[str, float],
    expected: Mapping[str, Any],
    *,
    tolerance: float,
) -> None:
    if set(live) != set(expected):
        raise StageBackendError(
            "live evaluator metrics differ from the sealed evaluator contract"
        )
    for name, value in live.items():
        target = _metric(expected, name)
        allowed = tolerance * max(1.0, abs(target))
        if abs(float(value) - target) > allowed:
            raise StageBackendError(
                f"live evaluator could not reproduce sealed metric {name}: "
                f"{value} vs {target} (tolerance {allowed})"
            )


def _run_live_profile_autotune(
    *,
    stage: Mapping[str, Any],
    stage_config_sha256: str,
    bundle: MMapStageBundle,
    model: nn.Module,
    optimizer: OptimizerBundle,
    forward_model: nn.Module,
    runtime: Runtime,
    topology: ParallelTopology,
    output_root: Path,
    parent_checkpoint_sha256: str,
    precision_role_plan_sha256: str,
    autotune_profile_sha256: str,
    compile_mode: str,
    optimizer_state_policy: str,
    restore_parent: Any,
    signal_coordinator: SignalCoordinator | None,
    active_resume: bool = False,
) -> ProfileSelection:
    if not _is_gspo_stage(str(stage["id"])):
        raise StageBackendError("live profile autotune called for an unsupported stage")
    # This first validation keeps the independently generated evidence useful
    # as a second implementation/oracle, but its selected row is not promoted.
    external_selection = _selected_gspo_profile(stage, bundle)
    del external_selection
    evidence = _require_mapping(
        bundle.manifest.get("profile_selection"),
        f"{stage['id']} profile_selection",
    )
    live_contract = _require_mapping(
        evidence.get("live_autotune"),
        f"{stage['id']} profile_selection.live_autotune",
    )
    evaluator = _require_mapping(
        live_contract.get("evaluator"),
        f"{stage['id']} live evaluator",
    )
    pipeline_policy = _require_mapping(
        _require_mapping(
            stage.get("autotune"), f"{stage['id']}.autotune"
        ).get("live_canary"),
        f"{stage['id']}.autotune.live_canary",
    )
    steps = int(live_contract.get("training_optimizer_steps", 0))
    expected_evaluator = "metis.rlvr-offline-policy-replay/v1"
    if (
        pipeline_policy.get("schema")
        != "metis.posttraining-live-canary-policy/v1"
        or pipeline_policy.get("evaluator_implementation")
        != expected_evaluator
        or steps != int(pipeline_policy.get("training_optimizer_steps", -1))
        or int(evaluator["records"])
        < int(pipeline_policy.get("minimum_evaluation_records", 0))
        or float(evaluator["reproduction_tolerance"])
        > float(
            pipeline_policy.get(
                "maximum_reproduction_tolerance", -1.0
            )
        )
        or pipeline_policy.get("restore_parent_between_trials") is not True
        or pipeline_policy.get("restore_rng_between_trials") is not True
        or steps <= 0
        or steps > _total_optimizer_steps(bundle, topology)
    ):
        raise StageBackendError(
            f"{stage['id']} live canary exceeds its pipeline policy"
        )
    seed = live_contract.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise StageBackendError("live profile autotune seed is invalid")
    tolerance = float(evaluator["reproduction_tolerance"])
    _default, candidates, _gate = _gspo_profile_candidates(stage)
    fields = ("clip_low", "clip_high", "length_coefficient")
    expected_trials = _expected_profile_trials(evidence, fields=fields)
    candidate_rows = [dict(row) for row in candidates]
    if set(expected_trials) != {
        _canonical_hash(candidate) for candidate in candidate_rows
    }:
        raise StageBackendError(
            f"{stage['id']} live tuner candidate set differs from sealed evidence"
        )
    runtime_inventory, runtime_inventory_sha = _runtime_rank_inventory(
        runtime=runtime,
        topology=topology,
        compile_mode=compile_mode,
    )
    bindings = {
        "family": bundle.family,
        "stage": str(stage["id"]),
        "stage_config_sha256": stage_config_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "working_set_autotune_sha256": (
            bundle.working_set_autotune_sha256
        ),
        "runtime_batch": _runtime_batch_payload(bundle),
        "profile_selection_sha256": str(evidence["selection_sha256"]),
        "live_autotune_sha256": str(live_contract["live_autotune_sha256"]),
        "candidate_set_sha256": _canonical_hash(candidate_rows),
        "evaluator_sha256": str(evaluator["evaluator_sha256"]),
        "evaluation_dataset_sha256": str(evaluator["dataset_sha256"]),
        "precision_role_plan_sha256": precision_role_plan_sha256,
        "base_autotune_profile_sha256": autotune_profile_sha256,
        "world_size": topology.world_size,
        "expert_parallel_size": topology.expert_parallel_size,
        "expert_replica_count": topology.expert_replica_count,
        "runtime_inventory_sha256": runtime_inventory_sha,
        "optimizer_state_policy": optimizer_state_policy,
        "training_optimizer_steps": steps,
        "seed": seed,
    }
    receipt_path = output_root / "autotune" / f"{stage['id']}.json"

    def load_receipt() -> dict[str, Any]:
        if topology.rank != 0 or not receipt_path.is_file():
            return {}
        receipt = _read_json(
            receipt_path, label=f"{stage['id']} live-autotune receipt"
        )
        if (
            receipt.get("schema") != LIVE_PROFILE_AUTOTUNE_RECEIPT_SCHEMA
            or any(receipt.get(name) != value for name, value in bindings.items())
            or receipt.get("runtime_inventory") != runtime_inventory
            or receipt.get("receipt_sha256")
            != _canonical_hash(receipt, omit={"receipt_sha256"})
        ):
            raise StageBackendError(
                f"{stage['id']} live-autotune resume lineage is invalid"
            )
        for index, raw_trial in enumerate(
            _require_list(receipt.get("trials"), "live-autotune trials")
        ):
            trial = _require_mapping(raw_trial, f"live-autotune trial {index}")
            if trial.get("trial_sha256") != _canonical_hash(
                trial, omit={"trial_sha256"}
            ):
                raise StageBackendError(
                    f"{stage['id']} live-autotune trial receipt is corrupt"
                )
        return dict(receipt)

    loaded = _collective_errors(
        topology,
        load_receipt,
        label=f"{stage['id']} live-autotune receipt load",
    )
    receipt = dict(
        _broadcast_object(loaded if topology.rank == 0 else None, topology)
    )
    if not receipt:
        receipt = {
            "schema": LIVE_PROFILE_AUTOTUNE_RECEIPT_SCHEMA,
            **bindings,
            "runtime_inventory": runtime_inventory,
            "baselines": {},
            "trials": [],
            "selected_profile": None,
            "selected_profile_sha256": None,
            "complete": False,
            "receipt_sha256": "",
        }

    def write_receipt() -> None:
        receipt["receipt_sha256"] = _canonical_hash(
            receipt, omit={"receipt_sha256"}
        )
        if topology.rank == 0:
            _atomic_json(receipt_path, receipt)
        barrier(topology)

    if receipt.get("complete") is True:
        profile = _normalized_profile(
            receipt.get("selected_profile"),
            fields=fields,
            label=f"{stage['id']} live selected_profile",
        )
        profile_sha = _canonical_hash(profile)
        if receipt.get("selected_profile_sha256") != profile_sha:
            raise StageBackendError(
                f"{stage['id']} live-autotune winner hash changed"
            )
        selected_trial = next(
            (
                item
                for item in receipt["trials"]
                if item["profile_sha256"] == profile_sha
            ),
            None,
        )
        if not isinstance(selected_trial, Mapping):
            raise StageBackendError(
                f"{stage['id']} live-autotune winner trial is missing"
            )
        return ProfileSelection(
            profile=profile,
            profile_sha256=profile_sha,
            evidence_sha256=str(receipt["receipt_sha256"]),
            selected_metrics=dict(
                _require_mapping(
                    selected_trial.get("metrics"),
                    f"{stage['id']} selected live metrics",
                )
            ),
            gate_name=str(selected_trial["gate_name"]),
        )
    if active_resume:
        raise StageBackendError(
            f"{stage['id']} active checkpoint exists without a complete live-"
            "autotune receipt"
        )

    original_rng = _capture_rng_state(runtime)
    try:
        baselines = _require_mapping(
            receipt.get("baselines"), "live-autotune baselines"
        )
        required_baseline_keys = {
            f"{candidate['length_coefficient']:.17g}"
            for candidate in candidate_rows
        }
        for key in sorted(required_baseline_keys):
            if key in baselines:
                continue
            restore_parent()
            _apply_optimizer_state_transition(
                optimizer,
                policy=optimizer_state_policy,
                active_resume=False,
            )
            _seed_live_trial(seed, runtime)
            profile = next(
                item
                for item in candidate_rows
                if f"{item['length_coefficient']:.17g}" == key
            )
            baseline = _evaluate_live_gspo_profile(
                stage=stage,
                profile=profile,
                bundle=bundle,
                model=model,
                forward_model=forward_model,
                runtime=runtime,
                topology=topology,
                evaluator=evaluator,
            )
            mutable_baselines = dict(baselines)
            baseline_row = {
                "metrics": baseline,
                "metrics_sha256": _canonical_hash(baseline),
                "baseline_sha256": "",
            }
            baseline_row["baseline_sha256"] = _canonical_hash(
                baseline_row, omit={"baseline_sha256"}
            )
            mutable_baselines[key] = baseline_row
            receipt["baselines"] = mutable_baselines
            baselines = mutable_baselines
            write_receipt()

        completed = {
            str(item["profile_sha256"])
            for item in _require_list(receipt.get("trials"), "live trials")
        }
        for profile in candidate_rows:
            profile_sha = _canonical_hash(profile)
            if profile_sha in completed:
                continue
            if _signal_requested(
                signal_coordinator, topology=topology, runtime=runtime
            ):
                write_receipt()
                raise PostTrainingRequeue(
                    signal_coordinator.reason or "remote-rank-signal"
                )
            restore_parent()
            _apply_optimizer_state_transition(
                optimizer,
                policy=optimizer_state_policy,
                active_resume=False,
            )
            _seed_live_trial(seed, runtime)
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(runtime.device)
                torch.cuda.synchronize(runtime.device)
            started = time.perf_counter()
            selection = ProfileSelection(
                profile=profile,
                profile_sha256=profile_sha,
                evidence_sha256=str(evidence["selection_sha256"]),
                selected_metrics={},
                gate_name="live_trial",
            )
            nonfinite_steps = 0
            candidate_failure: str | None = None
            try:
                _run_gspo_stage(
                    stage=stage,
                    bundle=bundle,
                    model=model,
                    optimizer=optimizer,
                    runtime=runtime,
                    topology=topology,
                    start_epoch=0,
                    start_global_batch=0,
                    start_optimizer_step=0,
                    start_cursor=0,
                    checkpoint_callback=lambda *_args: None,
                    forward_model=forward_model,
                    selection_override=selection,
                    maximum_optimizer_steps=steps,
                )
            except Exception as exception:
                if _is_device_oom(exception):
                    candidate_failure = "measured_candidate_oom"
                    nonfinite_steps = 1
                    model.zero_grad(set_to_none=True)
                    if runtime.device.type == "cuda":
                        torch.cuda.empty_cache()
                elif isinstance(exception, StageBackendError) and (
                    "non-finite" in str(exception).lower()
                ):
                    candidate_failure = "nonfinite_candidate"
                    nonfinite_steps = 1
                else:
                    raise
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            elapsed = time.perf_counter() - started
            elapsed_max = _all_reduce_float_vector(
                [elapsed],
                topology=topology,
                runtime=runtime,
                operation="max",
            )[0]
            if nonfinite_steps:
                live_metrics = {
                    "reward_gain": 0.0,
                    "entropy_delta": 0.0,
                    "evaluation_regression": 0.0,
                    "nonfinite_steps": 1.0,
                    "evaluation_records": float(evaluator["records"]),
                    "rollout_prompts": float(evaluator["records"]),
                }
            else:
                candidate_evaluation = _evaluate_live_gspo_profile(
                    stage=stage,
                    profile=profile,
                    bundle=bundle,
                    model=model,
                    forward_model=forward_model,
                    runtime=runtime,
                    topology=topology,
                    evaluator=evaluator,
                )
                baseline_key = f"{profile['length_coefficient']:.17g}"
                baseline = _require_mapping(
                    _require_mapping(
                        receipt["baselines"][baseline_key],
                        "live baseline row",
                    ).get("metrics"),
                    "live baseline metrics",
                )
                live_metrics = _live_metrics_from_evaluations(
                    stage_id=str(stage["id"]),
                    parent={
                        name: float(value)
                        for name, value in baseline.items()
                    },
                    candidate=candidate_evaluation,
                    nonfinite_steps=0,
                )
                _metrics_reproduce(
                    live_metrics,
                    _require_mapping(
                        expected_trials[profile_sha].get("metrics"),
                        "sealed expected live metrics",
                    ),
                    tolerance=tolerance,
                )
            passed, ordering, gate_name = _live_profile_gate(stage, live_metrics)
            if candidate_failure is not None:
                passed = False
            peak_hbm = (
                float(torch.cuda.max_memory_allocated(runtime.device))
                if runtime.device.type == "cuda"
                else 0.0
            )
            peak_hbm = _all_reduce_float_vector(
                [peak_hbm],
                topology=topology,
                runtime=runtime,
                operation="max",
            )[0]
            global_records = (
                steps
                * int(bundle.training["micro_batch_size"])
                * int(bundle.training["gradient_accumulation"])
                * topology.world_size
            )
            trial: dict[str, Any] = {
                "profile": profile,
                "profile_sha256": profile_sha,
                "metrics": live_metrics,
                "metrics_sha256": _canonical_hash(live_metrics),
                "expected_trial_sha256": str(
                    expected_trials[profile_sha]["trial_sha256"]
                ),
                "passed": passed,
                "candidate_failure": candidate_failure,
                "gate_name": gate_name,
                "ordering": list(ordering),
                "elapsed_seconds_max_rank": elapsed_max,
                "global_records_per_second": (
                    global_records / elapsed_max if elapsed_max > 0 else 0.0
                ),
                "peak_hbm_bytes_max_rank": int(peak_hbm),
                "state_sample_sha256": _profile_trial_state_fingerprint(
                    model, topology=topology
                ),
                "trial_sha256": "",
            }
            trial["trial_sha256"] = _canonical_hash(
                trial, omit={"trial_sha256"}
            )
            receipt["trials"].append(trial)
            write_receipt()
            completed.add(profile_sha)

        passing = [
            trial
            for trial in receipt["trials"]
            if trial.get("passed") is True
        ]
        if not passing:
            raise StageBackendError(
                f"{stage['id']} live trials produced no gate-safe profile"
            )
        winner = sorted(
            passing,
            key=lambda item: (
                *tuple(float(value) for value in item["ordering"]),
                -float(item["global_records_per_second"]),
                str(item["profile_sha256"]),
            ),
        )[0]
        receipt["selected_profile"] = dict(winner["profile"])
        receipt["selected_profile_sha256"] = str(winner["profile_sha256"])
        receipt["complete"] = True
        write_receipt()
        return ProfileSelection(
            profile=dict(winner["profile"]),
            profile_sha256=str(winner["profile_sha256"]),
            evidence_sha256=str(receipt["receipt_sha256"]),
            selected_metrics=dict(winner["metrics"]),
            gate_name=str(winner["gate_name"]),
        )
    finally:
        # The production stage starts from the exact parent and from the RNG
        # state it had before tuning.  No trial optimizer moment or sampled
        # random number can leak into the promoted run.
        restore_parent()
        _apply_optimizer_state_transition(
            optimizer,
            policy=optimizer_state_policy,
            active_resume=False,
        )
        _restore_rng_state(original_rng, runtime)
        optimizer.zero_grad(set_to_none=True)
        barrier(topology)


def _run_gspo_stage(
    *,
    stage: Mapping[str, Any],
    bundle: MMapStageBundle,
    model: nn.Module,
    optimizer: OptimizerBundle,
    runtime: Runtime,
    topology: ParallelTopology,
    start_epoch: int,
    start_global_batch: int,
    start_optimizer_step: int,
    start_cursor: int,
    checkpoint_callback: Any,
    signal_coordinator: SignalCoordinator | None = None,
    forward_model: nn.Module | None = None,
    selection_override: ProfileSelection | None = None,
    maximum_optimizer_steps: int | None = None,
    resume_metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if bundle.sequence_length != int(stage["sequence_length"]):
        raise StageBackendError(f"{stage['id']} sequence-length contract changed")
    objective_config = _require_mapping(stage.get("objective"), f"{stage['id']}.objective")
    selection = (
        selection_override or _selected_gspo_profile(stage, bundle)
        if _is_gspo_stage(str(stage["id"]))
        else None
    )
    profile = (
        selection.profile
        if selection is not None
        else {
            "clip_low": float(objective_config["clip_low"]),
            "clip_high": float(objective_config["clip_high"]),
            "length_coefficient": 0.0,
        }
    )
    clip_low = profile["clip_low"]
    clip_high = profile["clip_high"]
    total_steps = _total_optimizer_steps(bundle, topology)
    raw_length_schedule = objective_config.get("length_schedule")
    if raw_length_schedule is not None:
        length_schedule = _require_mapping(
            raw_length_schedule, f"{stage['id']}.objective.length_schedule"
        )
        correctness_only_fraction = float(
            length_schedule["correctness_only_fraction"]
        )
        if not 0.5 <= correctness_only_fraction < 1.0:
            raise StageBackendError(
                f"{stage['id']} length bootstrap fraction is invalid"
            )
        adaptive_length_start_step = max(
            1,
            min(
                total_steps - 1,
                int(math.floor(total_steps * correctness_only_fraction)),
            ),
        )
    else:
        adaptive_length_start_step = 0
    accumulation = int(bundle.training["gradient_accumulation"])
    checkpoint_interval = int(bundle.training["checkpoint_interval_steps"])
    optimizer_step = start_optimizer_step
    campaign_cursor = start_cursor
    accumulated_weight = 0
    accumulated_tokens = 0
    accumulation_index = 0
    expert_selection_counts: Tensor | None = None
    last = {
        str(name): float(value)
        for name, value in (resume_metrics or {}).items()
        if name != "grad_norm"
    }
    last_grad_norm = float((resume_metrics or {}).get("grad_norm", 0.0))
    optimizer.zero_grad(set_to_none=True)
    model.train()
    execution_model = forward_model if forward_model is not None else model
    execution_model.train()

    for epoch, global_batch, cpu_batch in bundle.iter_rank_batches(
        topology,
        start_epoch=start_epoch,
        start_global_batch=start_global_batch,
    ):
        batch = _to_device(cpu_batch, runtime.device)
        candidate_ids = batch["candidate_input_ids"].long()
        batch_size, group_size, sequence_length = candidate_ids.shape
        candidate_attention = batch["candidate_attention_mask"].bool()
        response_mask = batch["candidate_response_mask"].bool()
        truncated = batch["truncated"].bool()
        thinking_diagnostics: dict[str, Tensor] | None = None
        adaptive_length_active = bool(
            raw_length_schedule is not None
            and (
                maximum_optimizer_steps is not None
                or optimizer_step >= adaptive_length_start_step
            )
        )
        rewards, pass_rates, _lengths, thinking_diagnostics = _rlvr_rewards(
            str(stage["id"]),
            batch,
            length_coefficient=(
                profile["length_coefficient"]
                if adaptive_length_active
                else 0.0
            ),
            mode_compliance_weight=float(
                _require_mapping(
                    stage.get("reward"), f"{stage['id']}.reward"
                )["mode_compliance_weight"]
            ),
        )
        for name, value in thinking_diagnostics.items():
            if not torch.isfinite(value):
                raise StageBackendError(
                    f"{stage['id']} produced non-finite {name}"
                )
        reward_mean = rewards.float().mean(dim=1, keepdim=True)
        reward_std = rewards.float().std(dim=1, keepdim=True, unbiased=False)
        if torch.any(reward_std <= 1.0e-6):
            raise StageBackendError(
                f"{stage['id']} contains an uninformative GSPO reward group"
            )
        normalized_advantages = (rewards.float() - reward_mean) / reward_std
        if stage["id"] == "specialist_agentic":
            if "token_advantages" not in batch:
                raise StageBackendError(
                    "agentic RLVR requires verified turn-level token_advantages"
                )
            all_token_advantages = (
                batch["token_advantages"].float()
                + normalized_advantages.unsqueeze(-1)
            )
            if all_token_advantages.shape != response_mask.shape:
                raise StageBackendError("agentic token_advantages has the wrong shape")
        else:
            all_token_advantages = normalized_advantages.unsqueeze(-1).expand_as(
                response_mask
            )

        micro_group = int(bundle.working_set["candidate_micro_group_size"])
        token_chunk = int(bundle.working_set["token_chunk_size"])
        batch_valid_sequences = 0
        metric_names = (
            "loss",
            "objective",
            "mean_sequence_ratio",
            "clipped_fraction",
            "valid_sequences",
            "valid_tokens",
        )
        metric_sums: dict[str, float] = {
            name: 0.0 for name in metric_names
        }
        for group_start in range(0, group_size, micro_group):
            group_end = group_start + micro_group
            micro_response_mask = response_mask[:, group_start:group_end]
            micro_truncated = truncated[:, group_start:group_end]
            micro_valid = (
                micro_response_mask.sum(dim=-1) > 0
            ) & ~micro_truncated
            local_has_valid = bool(micro_valid.any())
            if not global_any(local_has_valid, topology, runtime.device):
                continue
            ids = candidate_ids[:, group_start:group_end].reshape(
                batch_size * micro_group, sequence_length
            )
            attention = candidate_attention[
                :, group_start:group_end
            ].reshape(batch_size * micro_group, sequence_length)
            output = _model_forward(
                execution_model,
                ids,
                attention_mask=attention,
                canonical_ids=_canonicalize_ids(bundle, ids),
                return_logits=False,
            )
            expert_selection_counts = _accumulate_expert_selection_counts(
                expert_selection_counts,
                output,
                include=local_has_valid,
            )
            hidden = _output_value(output, "final_hidden_state")[:, :-1]
            current_log_probs = _selected_token_log_probs_from_hidden(
                model,
                hidden,
                ids[:, 1:],
                token_chunk_size=token_chunk,
            ).reshape(batch_size, micro_group, sequence_length - 1)
            if local_has_valid:
                objective = gspo_token_loss(
                    current_token_log_probs=current_log_probs,
                    old_token_log_probs=batch["old_token_log_probs"][
                        :, group_start:group_end
                    ],
                    token_advantages=all_token_advantages[
                        :, group_start:group_end
                    ],
                    response_mask=micro_response_mask,
                    truncated=micro_truncated,
                    clip_low=clip_low,
                    clip_high=clip_high,
                )
                valid_sequences = int(
                    objective["valid_sequences"].detach().item()
                )
                auxiliary = _output_value(
                    output, "auxiliary_loss", objective["loss"] * 0.0
                )
                loss = objective["loss"] + auxiliary
            else:
                # A rank with no valid local candidate still executes the
                # trunk, EP collectives, FP8 head chunks, and backward in the
                # same order as its peers. Its sentinel graph contributes
                # exactly zero to gradients and routing-bias statistics.
                zero = current_log_probs.float().sum() * 0.0
                auxiliary = _output_value(output, "auxiliary_loss", zero)
                loss = zero + auxiliary * 0.0
                valid_sequences = 0
                objective = {}
            if not torch.isfinite(loss):
                raise StageBackendError(
                    f"{stage['id']} produced a non-finite GSPO loss"
                )
            if local_has_valid and valid_sequences <= 0:
                raise StageBackendError(
                    f"{stage['id']} candidate micro-chunk has no valid sequences"
                )
            (loss * max(valid_sequences, 1)).backward()
            batch_valid_sequences += valid_sequences
            for name, value in objective.items():
                if isinstance(value, Tensor) and value.numel() == 1:
                    metric_sums[name] = metric_sums.get(name, 0.0) + (
                        float(value.detach().float().item()) * valid_sequences
                    )
        accumulated_weight += batch_valid_sequences
        accumulated_tokens += int(response_mask.sum().item())
        accumulation_index += 1
        reduced_metrics = _all_reduce_float_vector(
            [
                float(batch_valid_sequences),
                *(metric_sums[name] for name in metric_names),
                float(rewards.float().sum().item()),
                float(rewards.numel()),
                (
                    float(pass_rates.float().sum().item())
                    if pass_rates is not None
                    else 0.0
                ),
                float(pass_rates.numel()) if pass_rates is not None else 0.0,
                (
                    float(
                        thinking_diagnostics[
                            "correct_response_count"
                        ].item()
                    )
                    if thinking_diagnostics is not None
                    else 0.0
                ),
                (
                    float(
                        thinking_diagnostics[
                            "thinking_target_token_sum"
                        ].item()
                    )
                    if thinking_diagnostics is not None
                    else 0.0
                ),
                (
                    float(
                        thinking_diagnostics["underthinking_count"].item()
                    )
                    if thinking_diagnostics is not None
                    else 0.0
                ),
                (
                    float(
                        thinking_diagnostics["overthinking_count"].item()
                    )
                    if thinking_diagnostics is not None
                    else 0.0
                ),
                (
                    float(
                        thinking_diagnostics["on_budget_count"].item()
                    )
                    if thinking_diagnostics is not None
                    else 0.0
                ),
            ],
            topology=topology,
            runtime=runtime,
        )
        global_batch_valid = int(reduced_metrics[0])
        if global_batch_valid <= 0:
            raise StageBackendError(
                f"{stage['id']} batch has no globally valid sequences"
            )
        last = {
            name: reduced_metrics[index + 1] / global_batch_valid
            for index, name in enumerate(metric_names)
        }
        summary_offset = 1 + len(metric_names)
        global_reward_count = reduced_metrics[summary_offset + 1]
        if global_reward_count <= 0:
            raise StageBackendError(
                f"{stage['id']} batch has no global reward observations"
            )
        last["mean_reward"] = (
            reduced_metrics[summary_offset] / global_reward_count
        )
        last["adaptive_length_active"] = float(adaptive_length_active)
        last["adaptive_length_start_step"] = float(
            adaptive_length_start_step
        )
        global_pass_rate_count = reduced_metrics[summary_offset + 3]
        if pass_rates is not None:
            if global_pass_rate_count <= 0:
                raise StageBackendError(
                    f"{stage['id']} batch has no global pass-rate observations"
                )
            last["mean_pass_rate"] = (
                reduced_metrics[summary_offset + 2] / global_pass_rate_count
            )
        correct_thinking_count = reduced_metrics[summary_offset + 4]
        if thinking_diagnostics is not None and correct_thinking_count > 0:
            last["thinking_target_tokens"] = (
                reduced_metrics[summary_offset + 5]
                / correct_thinking_count
            )
            last["underthinking_rate"] = (
                reduced_metrics[summary_offset + 6]
                / correct_thinking_count
            )
            last["overthinking_rate"] = (
                reduced_metrics[summary_offset + 7]
                / correct_thinking_count
            )
            last["on_budget_rate"] = (
                reduced_metrics[summary_offset + 8]
                / correct_thinking_count
            )
        next_epoch, next_batch = _batch_position_after(
            bundle, topology, epoch=epoch, global_batch=global_batch
        )
        if accumulation_index < accumulation:
            continue
        _set_learning_rate(
            optimizer,
            _scheduled_learning_rate(
                bundle.training,
                optimizer_step=optimizer_step,
                total_steps=total_steps,
            ),
        )
        _, last_grad_norm = _main_optimizer_step(
            model=model,
            optimizer=optimizer,
            topology=topology,
            runtime=runtime,
            local_weight=accumulated_weight,
            gradient_clip=float(bundle.training["gradient_clip"]),
            expert_selection_counts=expert_selection_counts,
        )
        campaign_cursor += _all_reduce_int(
            accumulated_tokens, topology, runtime.device
        )
        optimizer_step += 1
        accumulation_index = 0
        accumulated_weight = 0
        accumulated_tokens = 0
        expert_selection_counts = None
        progress = StageProgress(
            epoch=next_epoch,
            next_global_batch=next_batch,
            optimizer_step=optimizer_step,
            campaign_token_cursor=campaign_cursor,
            loss=last.get("loss", 0.0),
            grad_norm=last_grad_norm,
            metrics={**last, "grad_norm": last_grad_norm},
        )
        if _signal_requested(
            signal_coordinator, topology=topology, runtime=runtime
        ):
            checkpoint_callback(progress, False)
            raise PostTrainingRequeue(
                signal_coordinator.reason or "remote-rank-signal"
            )
        if (
            checkpoint_interval > 0
            and optimizer_step % checkpoint_interval == 0
            and optimizer_step < total_steps
        ):
            checkpoint_callback(progress, False)
        if (
            maximum_optimizer_steps is not None
            and optimizer_step >= maximum_optimizer_steps
        ):
            break
    if maximum_optimizer_steps is not None:
        if maximum_optimizer_steps <= 0 or optimizer_step != maximum_optimizer_steps:
            raise StageBackendError(
                f"{stage['id']} live canary could not execute its exact step budget"
            )
        if accumulation_index:
            raise StageBackendError(
                f"{stage['id']} live canary ended mid accumulation"
            )
    elif accumulation_index or optimizer_step != total_steps:
        raise StageBackendError(f"{stage['id']} did not consume its exact mmap bundle")
    result: dict[str, Any] = {
        "optimizer_steps": optimizer_step,
        "campaign_token_cursor": campaign_cursor,
        "grad_norm": last_grad_norm,
        "selected_profile": dict(profile),
        **last,
    }
    if selection is not None:
        result.update(
            {
                "selected_profile_sha256": selection.profile_sha256,
                "selection_evidence_sha256": selection.evidence_sha256,
                "autotune_gate_passed": True,
                "selection_gate": selection.gate_name,
            }
        )
    return result


def _policy_audit(policy: Any) -> dict[str, Any]:
    audit = getattr(policy, "audit", None)
    if audit is None:
        return {"effective_profile": str(getattr(policy, "effective_profile", "unknown"))}
    converter = getattr(audit, "to_dict", None)
    if callable(converter):
        value = converter()
    elif dataclasses.is_dataclass(audit):
        value = dataclasses.asdict(audit)
    elif isinstance(audit, Mapping):
        value = dict(audit)
    else:
        value = {"description": str(audit)}
    return dict(value)


def _checkpoint_contract(
    *,
    requirements: Sequence[SealedRequirement],
    bundle: MMapStageBundle,
    family_manifest_sha256: str,
    pipeline_sha256: str,
    autotune_profile_sha256: str,
    precision_role_plan_sha256: str,
) -> dict[str, str]:
    return {
        "release_sha256": _canonical_hash(
            sorted(item.manifest_sha256 for item in requirements)
        ),
        "shard_manifest_sha256": bundle.manifest_sha256,
        "family_manifest_sha256": family_manifest_sha256,
        "runtime_manifest_sha256": pipeline_sha256,
        "autotune_profile_sha256": autotune_profile_sha256,
        "precision_role_plan_sha256": precision_role_plan_sha256,
    }


def _save_policy_checkpoint(
    *,
    manager: CheckpointManager,
    model: nn.Module,
    optimizer: OptimizerBundle,
    policy: Any,
    stage_id: str,
    stage_config_sha256: str,
    parent_checkpoint_sha256: str,
    bundle: MMapStageBundle,
    requirements: Sequence[SealedRequirement],
    family_manifest_sha256: str,
    pipeline_sha256: str,
    autotune_profile_sha256: str,
    precision_role_plan_sha256: str,
    progress: StageProgress,
    complete: bool,
    optimizer_state_policy: str,
    milestone: str | None = None,
) -> tuple[Path, str, dict[str, str]]:
    contract = _checkpoint_contract(
        requirements=requirements,
        bundle=bundle,
        family_manifest_sha256=family_manifest_sha256,
        pipeline_sha256=pipeline_sha256,
        autotune_profile_sha256=autotune_profile_sha256,
        precision_role_plan_sha256=precision_role_plan_sha256,
    )
    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        global_token_cursor=progress.campaign_token_cursor,
        optimizer_step=progress.optimizer_step,
        phase=stage_id,
        shard_order_seed=int(bundle.training["shuffle_seed"]),
        precision_audit=_policy_audit(policy),
        # Context-extension gates are durable promotion candidates even
        # though training continues after the first two. Marking them as
        # boundaries keeps CheckpointManager's safe pruning from removing
        # them while preserving stage_complete=False for exact resume.
        phase_boundary=complete or milestone is not None,
        extra_state={
            "posttraining_stage": stage_id,
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "stage_config_sha256": stage_config_sha256,
            "bundle_sha256": bundle.manifest_sha256,
            "stage_epoch": progress.epoch,
            "stage_next_global_batch": progress.next_global_batch,
            "stage_optimizer_step": progress.optimizer_step,
            "campaign_token_cursor": progress.campaign_token_cursor,
            "last_loss": progress.loss,
            "last_grad_norm": progress.grad_norm,
            "last_metrics": dict(progress.metrics),
            "stage_complete": complete,
            "checkpoint_milestone": milestone,
            "optimizer_state_policy": optimizer_state_policy,
            "runtime_batch": _runtime_batch_payload(bundle),
        },
        **contract,
    )
    manifest = _read_json(
        checkpoint / "MANIFEST.json", label="post-training checkpoint manifest"
    )
    checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
    if len(checkpoint_sha) != 64:
        raise StageBackendError("distributed checkpoint did not seal a checkpoint hash")
    return checkpoint, checkpoint_sha, contract


def _write_checkpoint_receipt(
    *,
    output_root: Path,
    family: str,
    stage_id: str,
    stage_config_sha256: str,
    parent_checkpoint_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    precision_role_plan_sha256: str,
) -> Path:
    path = output_root / "receipts" / f"{stage_id}-checkpoint.json"
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "family": family,
        "stage": stage_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "config_sha256": stage_config_sha256,
        "checkpoint_manifest": str(checkpoint / "MANIFEST.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "precision_role_plan_sha256": precision_role_plan_sha256,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(payload, omit={"receipt_sha256"})
    _atomic_json(path, payload)
    return path


def _write_context_gate_receipt(
    *,
    output_root: Path,
    family: str,
    stage_config_sha256: str,
    parent_checkpoint_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_contract: Mapping[str, str],
    progress: StageProgress,
    gate_target_tokens: int,
    maximum_overshoot_tokens: int,
    complete: bool,
    evaluation_receipt: Path,
    evaluation_receipt_sha256: str,
) -> Path:
    context_tokens = progress.campaign_token_cursor - BASE_TOKEN_CURSOR
    overshoot = context_tokens - gate_target_tokens
    if (
        gate_target_tokens <= 0
        or context_tokens < gate_target_tokens
        or overshoot > maximum_overshoot_tokens
    ):
        raise StageBackendError(
            "context gate receipt would record an invalid token boundary"
        )
    path = (
        output_root
        / "context_gates"
        / f"tokens-{gate_target_tokens:011d}.json"
    )
    payload: dict[str, Any] = {
        "schema": CONTEXT_GATE_RECEIPT_SCHEMA,
        "family": family,
        "stage": "context_extension",
        "stage_config_sha256": stage_config_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "gate_target_tokens": gate_target_tokens,
        "actual_context_tokens": context_tokens,
        "overshoot_tokens": overshoot,
        "maximum_overshoot_tokens": maximum_overshoot_tokens,
        "campaign_token_cursor": progress.campaign_token_cursor,
        "optimizer_step": progress.optimizer_step,
        "stage_complete": complete,
        "checkpoint_manifest": str(checkpoint / "MANIFEST.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_contract": dict(checkpoint_contract),
        "evaluation_receipt": str(evaluation_receipt),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(
        payload, omit={"receipt_sha256"}
    )
    _atomic_json(path, payload)
    return path


def _write_context_gate_baseline(
    *,
    output_root: Path,
    family: str,
    base_checkpoint_sha256: str,
    bundle_sha256: str,
    metrics: Mapping[str, float],
) -> Path:
    path = output_root / "context_gates" / "BASELINE.json"
    payload: dict[str, Any] = {
        "schema": CONTEXT_GATE_BASELINE_SCHEMA,
        "family": family,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "bundle_sha256": bundle_sha256,
        "metrics": dict(metrics),
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(
        payload, omit={"receipt_sha256"}
    )
    _atomic_json(path, payload)
    return path


def _write_context_gate_evaluation_receipt(
    *,
    output_root: Path,
    family: str,
    base_checkpoint_sha256: str,
    checkpoint_sha256: str,
    bundle_sha256: str,
    gate_target_tokens: int,
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    gate_policy: Mapping[str, Any],
) -> Path:
    decision = _context_gate_decision(
        metrics=metrics,
        baseline=baseline,
        gate_target_tokens=gate_target_tokens,
        gate_policy=gate_policy,
    )
    path = (
        output_root
        / "context_gates"
        / f"evaluation-tokens-{gate_target_tokens:011d}.json"
    )
    payload: dict[str, Any] = {
        "schema": CONTEXT_GATE_EVALUATION_SCHEMA,
        "family": family,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "bundle_sha256": bundle_sha256,
        "gate_target_tokens": gate_target_tokens,
        "metrics": dict(metrics),
        "baseline": dict(baseline),
        "decision": decision,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(
        payload, omit={"receipt_sha256"}
    )
    _atomic_json(path, payload)
    return path


def _write_context_gate_promotion(
    *,
    output_root: Path,
    family: str,
    gate_receipts: Mapping[str, Any],
) -> Path:
    expected_targets = (
        6_000_000_000,
        12_000_000_000,
        18_000_000_000,
    )
    candidates: list[dict[str, Any]] = []
    for target in expected_targets:
        raw_path = gate_receipts.get(str(target))
        if not isinstance(raw_path, str):
            raise StageBackendError(
                f"context promotion is missing gate {target}"
            )
        gate_path = Path(raw_path).expanduser().resolve()
        gate = _read_json(
            gate_path, label=f"context promotion gate {target}"
        )
        evaluation_path = Path(
            str(gate.get("evaluation_receipt", ""))
        ).expanduser().resolve()
        evaluation = _read_json(
            evaluation_path,
            label=f"context promotion evaluation {target}",
        )
        decision = _require_mapping(
            evaluation.get("decision"),
            f"context promotion decision {target}",
        )
        if (
            gate.get("schema") != CONTEXT_GATE_RECEIPT_SCHEMA
            or int(gate.get("gate_target_tokens", -1)) != target
            or gate.get("receipt_sha256")
            != _canonical_hash(gate, omit={"receipt_sha256"})
            or evaluation.get("schema")
            != CONTEXT_GATE_EVALUATION_SCHEMA
            or evaluation.get("receipt_sha256")
            != gate.get("evaluation_receipt_sha256")
            or evaluation.get("receipt_sha256")
            != _canonical_hash(
                evaluation, omit={"receipt_sha256"}
            )
            or evaluation.get("checkpoint_sha256")
            != gate.get("checkpoint_sha256")
            or not math.isfinite(
                float(decision.get("promotion_score", float("nan")))
            )
        ):
            raise StageBackendError(
                f"context promotion gate {target} is corrupt"
            )
        candidates.append(
            {
                "gate_target_tokens": target,
                "gate_receipt": str(gate_path),
                "gate_receipt_sha256": gate["receipt_sha256"],
                "evaluation_receipt": str(evaluation_path),
                "evaluation_receipt_sha256": evaluation[
                    "receipt_sha256"
                ],
                "checkpoint_path": str(
                    Path(str(gate["checkpoint_manifest"])).parent
                ),
                "checkpoint_sha256": gate["checkpoint_sha256"],
                "checkpoint_contract": dict(
                    _require_mapping(
                        gate["checkpoint_contract"],
                        f"context gate {target} checkpoint contract",
                    )
                ),
                "passed": decision.get("passed") is True,
                "promotion_score": float(decision["promotion_score"]),
            }
        )
    passing = [row for row in candidates if row["passed"]]
    if not passing:
        raise StageBackendError(
            "all 6B/12B/18B context checkpoints failed the autonomous gate"
        )
    selected = max(
        passing,
        key=lambda row: (
            float(row["promotion_score"]),
            int(row["gate_target_tokens"]),
        ),
    )
    path = output_root / "context_gates" / "PROMOTION.json"
    payload: dict[str, Any] = {
        "schema": CONTEXT_GATE_PROMOTION_SCHEMA,
        "family": family,
        "candidates": candidates,
        "selected": selected,
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(
        payload, omit={"receipt_sha256"}
    )
    _atomic_json(path, payload)
    return path


def _write_stage_receipt(
    *,
    output_root: Path,
    family: str,
    stage_id: str,
    stage_config_sha256: str,
    parent_checkpoint_sha256: str,
    policy_checkpoint_sha256: str,
    precision_role_plan_sha256: str,
    requirements: Sequence[SealedRequirement],
    metrics: Mapping[str, Any],
    optimizer_state_policy: str,
    checkpoint_receipt: Path | None = None,
) -> Path:
    path = output_root / "receipts" / f"{stage_id}-output.json"
    payload: dict[str, Any] = {
        "schema": STAGE_RECEIPT_SCHEMA,
        "family": family,
        "stage": stage_id,
        "stage_config_sha256": stage_config_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "policy_checkpoint_sha256": policy_checkpoint_sha256,
        "precision_role_plan_sha256": precision_role_plan_sha256,
        "optimizer_state_policy": optimizer_state_policy,
        "requirements": [
            {
                "name": item.name,
                "schema": item.schema,
                "environment_variable": item.environment_variable,
                "manifest_path": str(item.manifest_path),
                "manifest_sha256": item.manifest_sha256,
            }
            for item in requirements
        ],
        "metrics": dict(metrics),
        "checkpoint_receipt": str(checkpoint_receipt) if checkpoint_receipt else None,
        "completed_unix": int(time.time()),
        "receipt_sha256": "",
    }
    payload["receipt_sha256"] = _canonical_hash(payload, omit={"receipt_sha256"})
    _atomic_json(path, payload)
    return path


def _initial_campaign_state(
    *,
    family: str,
    pipeline_sha256: str,
    family_manifest_sha256: str,
    base_checkpoint_sha256: str,
    base_checkpoint: Path,
    base_receipt: Path,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema": CAMPAIGN_STATE_SCHEMA,
        "family": family,
        "pipeline_sha256": pipeline_sha256,
        "family_manifest_sha256": family_manifest_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "policy_checkpoint_sha256": base_checkpoint_sha256,
        "policy_checkpoint_path": str(base_checkpoint),
        "policy_checkpoint_receipt": str(base_receipt),
        "policy_checkpoint_contract": None,
        "campaign_token_cursor": BASE_TOKEN_CURSOR,
        "evaluation_receipt": None,
        "context_gate_receipts": {},
        "context_gate_promotion": None,
        "completed": [],
        "active": None,
        "state_sha256": "",
    }
    state["state_sha256"] = _canonical_hash(state, omit={"state_sha256"})
    return state


def _validate_campaign_state(
    state: Mapping[str, Any],
    *,
    family: str,
    pipeline_sha256: str,
    family_manifest_sha256: str,
    base_checkpoint_sha256: str,
) -> dict[str, Any]:
    if state.get("schema") != CAMPAIGN_STATE_SCHEMA:
        raise StageBackendError("post-training campaign state has the wrong schema")
    if state.get("state_sha256") != _canonical_hash(state, omit={"state_sha256"}):
        raise StageBackendError("post-training campaign state failed its self-hash")
    expected = {
        "family": family,
        "pipeline_sha256": pipeline_sha256,
        "family_manifest_sha256": family_manifest_sha256,
        "base_checkpoint_sha256": base_checkpoint_sha256,
    }
    for field, value in expected.items():
        if state.get(field) != value:
            raise StageBackendError(
                f"post-training resume refused because {field} changed"
            )
    completed = _require_list(state.get("completed"), "campaign completed")
    observed_ids = [str(item.get("stage_id", "")) for item in completed if isinstance(item, Mapping)]
    if observed_ids != list(EXPECTED_STAGE_IDS[: len(observed_ids)]):
        raise StageBackendError("post-training completed-stage order is corrupt")
    for raw_record in completed:
        record = _require_mapping(raw_record, "completed stage")
        receipt_path = Path(str(record.get("output_receipt", ""))).expanduser().resolve()
        receipt = _read_json(receipt_path, label="completed stage receipt")
        if receipt.get("receipt_sha256") != record.get("output_receipt_sha256"):
            raise StageBackendError("completed stage receipt hash differs from campaign state")
        if receipt.get("receipt_sha256") != _canonical_hash(
            receipt, omit={"receipt_sha256"}
        ):
            raise StageBackendError("completed stage receipt failed its self-hash")
    gate_receipts = _require_mapping(
        state.get("context_gate_receipts", {}),
        "context gate receipts",
    )
    for target, raw_path in gate_receipts.items():
        receipt = _read_json(
            Path(str(raw_path)).expanduser().resolve(),
            label=f"context gate {target} receipt",
        )
        if (
            receipt.get("schema") != CONTEXT_GATE_RECEIPT_SCHEMA
            or str(receipt.get("gate_target_tokens")) != str(target)
            or receipt.get("receipt_sha256")
            != _canonical_hash(receipt, omit={"receipt_sha256"})
        ):
            raise StageBackendError(
                f"context gate {target} receipt failed validation"
            )
        evaluation_path = Path(
            str(receipt.get("evaluation_receipt", ""))
        ).expanduser().resolve()
        evaluation = _read_json(
            evaluation_path,
            label=f"context gate {target} evaluation",
        )
        checkpoint_manifest = _read_json(
            Path(str(receipt.get("checkpoint_manifest", ""))),
            label=f"context gate {target} checkpoint",
        )
        if (
            evaluation.get("schema")
            != CONTEXT_GATE_EVALUATION_SCHEMA
            or evaluation.get("receipt_sha256")
            != receipt.get("evaluation_receipt_sha256")
            or evaluation.get("receipt_sha256")
            != _canonical_hash(
                evaluation, omit={"receipt_sha256"}
            )
            or evaluation.get("checkpoint_sha256")
            != receipt.get("checkpoint_sha256")
            or checkpoint_manifest.get("checkpoint_sha256")
            != receipt.get("checkpoint_sha256")
            or checkpoint_manifest.get("checkpoint_sha256")
            != _canonical_hash(
                checkpoint_manifest, omit={"checkpoint_sha256"}
            )
        ):
            raise StageBackendError(
                f"context gate {target} evaluation/checkpoint lineage failed"
            )
    raw_promotion = state.get("context_gate_promotion")
    if raw_promotion is not None:
        promotion_path = Path(str(raw_promotion)).expanduser().resolve()
        promotion = _read_json(
            promotion_path, label="context gate promotion"
        )
        selected = _require_mapping(
            promotion.get("selected"),
            "context gate promotion selected",
        )
        if (
            promotion.get("schema") != CONTEXT_GATE_PROMOTION_SCHEMA
            or promotion.get("family") != family
            or promotion.get("receipt_sha256")
            != _canonical_hash(
                promotion, omit={"receipt_sha256"}
            )
            or selected.get("passed") is not True
            or selected.get("checkpoint_sha256")
            != state.get("policy_checkpoint_sha256")
        ):
            raise StageBackendError(
                "context gate promotion is stale or corrupt"
            )
    return dict(state)


def _read_campaign_state(
    *,
    path: Path,
    topology: ParallelTopology,
    family: str,
    pipeline_sha256: str,
    family_manifest_sha256: str,
    base_checkpoint_sha256: str,
    base_checkpoint: Path,
    base_receipt: Path,
) -> dict[str, Any]:
    def read_on_rank() -> dict[str, Any]:
        if topology.rank != 0:
            return {}
        if path.exists():
            return _validate_campaign_state(
                _read_json(path, label="post-training campaign state"),
                family=family,
                pipeline_sha256=pipeline_sha256,
                family_manifest_sha256=family_manifest_sha256,
                base_checkpoint_sha256=base_checkpoint_sha256,
            )
        return _initial_campaign_state(
            family=family,
            pipeline_sha256=pipeline_sha256,
            family_manifest_sha256=family_manifest_sha256,
            base_checkpoint_sha256=base_checkpoint_sha256,
            base_checkpoint=base_checkpoint,
            base_receipt=base_receipt,
        )

    state = _collective_errors(
        topology,
        read_on_rank,
        label="campaign-state load",
    )
    return dict(_broadcast_object(state if topology.rank == 0 else None, topology))


def _write_campaign_state(
    path: Path,
    state: dict[str, Any],
    topology: ParallelTopology,
) -> None:
    state["state_sha256"] = _canonical_hash(state, omit={"state_sha256"})

    def write_on_rank() -> None:
        if topology.rank == 0:
            _atomic_json(path, state)

    _collective_errors(topology, write_on_rank, label="campaign-state write")
    barrier(topology)


def _load_distributed_checkpoint(
    *,
    manager: CheckpointManager,
    checkpoint: str | Path,
    contract: Mapping[str, str],
    model: nn.Module,
    optimizer: OptimizerBundle,
) -> None:
    manager.load(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_release_sha256=str(contract["release_sha256"]),
        expected_shard_manifest_sha256=str(contract["shard_manifest_sha256"]),
        expected_family_manifest_sha256=str(contract["family_manifest_sha256"]),
        expected_runtime_manifest_sha256=str(contract["runtime_manifest_sha256"]),
        expected_autotune_profile_sha256=str(contract["autotune_profile_sha256"]),
        expected_precision_role_plan_sha256=str(
            contract["precision_role_plan_sha256"]
        ),
    )


def _reconcile_active_policy_checkpoint_state(
    *,
    state_path: Path,
    state: dict[str, Any],
    topology: ParallelTopology,
) -> dict[str, Any]:
    """Bind campaign STATE to the currently verified checkpoint manifest.

    Checkpoint finalization can monotonically promote a same-cursor manifest
    from non-final to final. If the process dies after that atomic promotion
    but before ``STATE.json`` is rewritten, the state's old manifest hash must
    not leak into an OOM/requeue receipt.
    """

    active = _require_mapping(state.get("active"), "campaign active")
    if str(active.get("kind", "policy")) != "policy":
        return state
    checkpoint = Path(str(active["checkpoint_path"])).expanduser().resolve()

    def read_on_rank() -> dict[str, Any]:
        if topology.rank != 0:
            return {}
        manifest = _read_json(
            checkpoint / "MANIFEST.json",
            label="active policy checkpoint manifest",
        )
        if (
            manifest.get("schema") != "metis.distributed-checkpoint/v1"
            or manifest.get("checkpoint_sha256")
            != _canonical_hash(manifest, omit={"checkpoint_sha256"})
        ):
            raise StageBackendError(
                "active policy checkpoint manifest failed its integrity check"
            )
        return manifest

    manifest = _collective_errors(
        topology,
        read_on_rank,
        label="active policy checkpoint reconciliation",
    )
    manifest = dict(
        _broadcast_object(manifest if topology.rank == 0 else None, topology)
    )
    extra = _require_mapping(
        manifest.get("extra_state"),
        "active policy checkpoint extra_state",
    )
    expected = {
        "posttraining_stage": active.get("stage_id"),
        "parent_checkpoint_sha256": active.get("parent_checkpoint_sha256"),
        "stage_config_sha256": active.get("stage_config_sha256"),
        "bundle_sha256": active.get("bundle_sha256"),
        "optimizer_state_policy": active.get("optimizer_state_policy"),
        "runtime_batch": active.get("runtime_batch"),
    }
    checkpoint_milestone = extra.get("checkpoint_milestone")
    if checkpoint_milestone is not None and (
        not isinstance(checkpoint_milestone, str)
        or not checkpoint_milestone
    ):
        raise StageBackendError(
            "active checkpoint milestone metadata is invalid"
        )
    if (
        manifest.get("phase") != active.get("stage_id")
        or any(extra.get(name) != value for name, value in expected.items())
        or bool(manifest.get("phase_boundary"))
        != bool(extra.get("stage_complete") or checkpoint_milestone is not None)
    ):
        raise StageBackendError(
            "active campaign state and checkpoint manifest disagree"
        )
    last_metrics = _require_mapping(
        extra.get("last_metrics"),
        "active checkpoint last_metrics",
    )
    reconciled_active = dict(active)
    reconciled_active.update(
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
            "epoch": int(extra["stage_epoch"]),
            "next_global_batch": int(extra["stage_next_global_batch"]),
            "optimizer_step": int(extra["stage_optimizer_step"]),
            "campaign_token_cursor": int(extra["campaign_token_cursor"]),
            "last_loss": float(extra["last_loss"]),
            "last_grad_norm": float(extra["last_grad_norm"]),
            "last_metrics": dict(last_metrics),
            "checkpoint_phase_boundary": bool(
                manifest.get("phase_boundary")
            ),
            "checkpoint_stage_complete": bool(extra.get("stage_complete")),
        }
    )
    if (
        int(manifest.get("optimizer_step", -1))
        != reconciled_active["optimizer_step"]
        or int(manifest.get("global_token_cursor", -1))
        != reconciled_active["campaign_token_cursor"]
        or reconciled_active["last_metrics"].get("loss")
        != reconciled_active["last_loss"]
        or reconciled_active["last_metrics"].get("grad_norm")
        != reconciled_active["last_grad_norm"]
    ):
        raise StageBackendError(
            "active checkpoint progress metadata is internally inconsistent"
        )
    if reconciled_active != dict(active):
        state["active"] = reconciled_active
        _write_campaign_state(state_path, state, topology)
    return state


def _validate_base_checkpoint(
    *,
    family: str,
    base_checkpoint: Path,
    base_receipt: Path,
    expected_precision_role_plan_sha256: str,
) -> str:
    manifest = _read_json(
        base_checkpoint / "MANIFEST.json", label="base checkpoint manifest"
    )
    if (
        manifest.get("schema") != "metis.distributed-checkpoint/v1"
        or manifest.get("family") != family
        or manifest.get("precision_role_plan_sha256")
        != expected_precision_role_plan_sha256
    ):
        raise StageBackendError("base checkpoint has the wrong schema or family")
    observed = _canonical_hash(manifest, omit={"checkpoint_sha256"})
    if manifest.get("checkpoint_sha256") != observed:
        raise StageBackendError("base checkpoint manifest failed its self-hash")
    receipt = _read_json(base_receipt, label="base checkpoint receipt")
    if (
        receipt.get("schema") != CHECKPOINT_RECEIPT_SCHEMA
        or receipt.get("family") != family
        or receipt.get("stage") != "base_pretraining"
        or receipt.get("checkpoint_sha256") != observed
        or receipt.get("precision_role_plan_sha256")
        != expected_precision_role_plan_sha256
        or receipt.get("receipt_sha256")
        != _canonical_hash(receipt, omit={"receipt_sha256"})
    ):
        raise StageBackendError("base checkpoint receipt lineage is invalid")
    receipt_manifest = Path(str(receipt.get("checkpoint_manifest", ""))).resolve()
    if receipt_manifest != (base_checkpoint / "MANIFEST.json").resolve():
        raise StageBackendError("base receipt does not point at the supplied checkpoint")
    return observed


def _evaluation_payload(
    requirement: SealedRequirement,
    *,
    family: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    metadata = _require_mapping(requirement.payload.get("metadata"), "evaluation metadata")
    if metadata.get("backend_contract") != EVALUATION_RESULTS_SCHEMA:
        raise StageBackendError(
            f"evaluation metadata.backend_contract must be {EVALUATION_RESULTS_SCHEMA}"
        )
    raw_results = metadata.get("results_file")
    if not isinstance(raw_results, str):
        raise StageBackendError("evaluation metadata lacks results_file")
    results_path = _safe_relative(
        requirement.manifest_path.parent.resolve(),
        raw_results,
        label="evaluation results",
    )
    sealed_paths = {
        str(item.get("path"))
        for item in _require_list(requirement.payload.get("files"), "evaluation files")
        if isinstance(item, Mapping)
    }
    if str(results_path.relative_to(requirement.manifest_path.parent.resolve())) not in sealed_paths:
        raise StageBackendError("evaluation results file is not sealed")
    payload = _read_json(results_path, label="evaluation results")
    if (
        payload.get("schema") != EVALUATION_RESULTS_SCHEMA
        or payload.get("family") != family
        or payload.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise StageBackendError("evaluation results are not bound to the live policy")
    if (
        payload.get("contamination_recheck_passed") is not True
        or payload.get("tool_augmented_grounding_passed") is not True
    ):
        raise StageBackendError("evaluation contamination/grounding checks did not pass")
    return payload


def _run_evaluation(
    *,
    stage: Mapping[str, Any],
    requirement: SealedRequirement,
    family: str,
    checkpoint_sha256: str,
    topology: ParallelTopology,
) -> dict[str, Any]:
    payload = _collective_errors(
        topology,
        lambda: _evaluation_payload(
            requirement,
            family=family,
            checkpoint_sha256=checkpoint_sha256,
        ),
        label="evaluation results",
    )
    raw_metrics = _require_mapping(payload.get("metric_accumulators"), "metric_accumulators")
    metrics: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for name, raw_value in raw_metrics.items():
        accumulator = _require_mapping(raw_value, f"metric_accumulators.{name}")
        numerator = float(accumulator.get("sum", float("nan")))
        count = int(accumulator.get("count", 0))
        if not math.isfinite(numerator) or count <= 0:
            raise StageBackendError(f"evaluation metric {name} has an invalid accumulator")
        metrics[str(name)] = numerator / count
        sample_counts[str(name)] = count
    gate = evaluate_metric_gate(
        metrics,
        _require_mapping(stage.get("gate"), "evaluation.gate"),
        baselines=_require_mapping(payload.get("baselines"), "evaluation baselines"),
        suite_thresholds=_require_mapping(
            payload.get("suite_thresholds"), "evaluation suite_thresholds"
        ),
    )
    result = {
        "metrics": metrics,
        "sample_counts": sample_counts,
        "gate": gate,
    }
    if gate["passed"] is not True:
        raise StageBackendError(
            "evaluation publish gate failed: "
            + json.dumps(
                {
                    "missing": gate["missing_metrics"],
                    "failures": gate["failed_metrics"],
                },
                sort_keys=True,
            )
        )
    return result


def _write_release_candidate(
    *,
    output_root: Path,
    family: str,
    policy_checkpoint_sha256: str,
    policy_checkpoint_receipt: str,
    evaluation_receipt: str,
    topology: ParallelTopology,
) -> Path:
    path = output_root / "release" / "RELEASE_CANDIDATE.json"

    def write_on_rank() -> None:
        if topology.rank != 0:
            return
        payload: dict[str, Any] = {
            "schema": RELEASE_CANDIDATE_SCHEMA,
            "family": family,
            "policy_checkpoint_sha256": policy_checkpoint_sha256,
            "policy_checkpoint_receipt": policy_checkpoint_receipt,
            "evaluation_receipt": evaluation_receipt,
            "external_upload": False,
            "complete": True,
            "created_unix": int(time.time()),
            "candidate_sha256": "",
        }
        payload["candidate_sha256"] = _canonical_hash(
            payload, omit={"candidate_sha256"}
        )
        _atomic_json(path, payload)

    _collective_errors(topology, write_on_rank, label="release-candidate seal")
    barrier(topology)
    return path


def _rank_zero_path(
    topology: ParallelTopology,
    operation: Any,
    *,
    label: str,
) -> Path:
    def execute() -> str | None:
        if topology.rank == 0:
            return str(operation())
        return None

    raw = _collective_errors(topology, execute, label=label)
    raw = _broadcast_object(raw if topology.rank == 0 else None, topology)
    if not isinstance(raw, str):
        raise StageBackendError(f"{label} did not return a path")
    barrier(topology)
    return Path(raw).resolve()


def _completed_record(
    *,
    stage_id: str,
    stage_config_sha256: str,
    parent_checkpoint_sha256: str,
    state: Mapping[str, Any],
    output_receipt: Path,
    metrics: Mapping[str, Any],
    checkpoint_path: Path | None = None,
    checkpoint_contract: Mapping[str, str] | None = None,
    checkpoint_receipt: Path | None = None,
    release_candidate: Path | None = None,
    policy_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    receipt = _read_json(output_receipt, label=f"{stage_id} output receipt")
    return {
        "stage_id": stage_id,
        "stage_config_sha256": stage_config_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "policy_checkpoint_sha256": (
            policy_checkpoint_sha256
            if policy_checkpoint_sha256 is not None
            else state["policy_checkpoint_sha256"]
        ),
        "output_receipt": str(output_receipt),
        "output_receipt_sha256": receipt["receipt_sha256"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_contract": dict(checkpoint_contract) if checkpoint_contract else None,
        "checkpoint_receipt": str(checkpoint_receipt) if checkpoint_receipt else None,
        "release_candidate": str(release_candidate) if release_candidate else None,
        "metrics_sha256": _canonical_hash(metrics),
        "completed_unix": int(time.time()),
    }


def _stage_by_id(pipeline: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    stages = _require_list(pipeline.get("stages"), "pipeline.stages")
    result = {
        str(stage["id"]): stage
        for stage in stages
        if isinstance(stage, Mapping) and isinstance(stage.get("id"), str)
    }
    if tuple(result) != EXPECTED_STAGE_IDS:
        raise StageBackendError("post-training pipeline stage order changed")
    return result


def _run_posttraining_campaign_impl(
    args: Any,
    config: Any,
    model: nn.Module,
    optimizer: OptimizerBundle,
    policy: Any,
    runtime: Runtime,
    topology: ParallelTopology,
    family_manifest: Mapping[str, Any],
    base_checkpoint: str | Path,
    base_receipt: str | Path,
    posttraining_manifest: str | Path,
    signal_coordinator: SignalCoordinator,
) -> dict[str, Any]:
    """Execute the entire Metis context-extension/post-training campaign in process.

    This function is intentionally called by every already-initialized rank.
    There is no external trainer command and no rank-zero-only model update:
    all policy forwards, backwards, EP collectives, gradient synchronization,
    and distributed checkpoints use the live objects built by ``train.py``.
    """

    family = str(getattr(args, "family", topology.family)).lower()
    if family != topology.family or family not in {"praxis", "logos"}:
        raise StageBackendError("post-training family/topology mismatch")
    if runtime.rank != topology.rank or runtime.world_size != topology.world_size:
        raise StageBackendError("runtime and parallel topology disagree")
    pipeline_path = Path(posttraining_manifest).expanduser().resolve()
    pipeline = load_pipeline(pipeline_path)
    stages = _stage_by_id(pipeline)
    pipeline_sha256 = sha256_file(pipeline_path)
    family_manifest_path = Path(str(getattr(args, "manifest"))).expanduser().resolve()
    family_manifest_sha256 = sha256_file(family_manifest_path)
    if str(family_manifest.get("family", "")).lower() != family:
        raise StageBackendError("supplied family manifest is not for the live family")
    autotune_raw = getattr(args, "autotune_profile", None)
    if not autotune_raw:
        raise StageBackendError("post-training requires the measured base autotune profile")
    autotune_path = Path(str(autotune_raw)).expanduser().resolve()
    if not autotune_path.is_file():
        raise StageBackendError("base autotune profile is missing")
    autotune_profile_sha256 = sha256_file(autotune_path)
    autotune_selection = load_autotune_selection(
        autotune_path,
        family_manifest=family_manifest,
    )
    base_checkpoint_path = Path(base_checkpoint).expanduser().resolve()
    base_receipt_path = Path(base_receipt).expanduser().resolve()
    base_checkpoint_sha256 = _collective_errors(
        topology,
        lambda: _validate_base_checkpoint(
            family=family,
            base_checkpoint=base_checkpoint_path,
            base_receipt=base_receipt_path,
            expected_precision_role_plan_sha256=(
                autotune_selection.precision_role_plan_sha256
            ),
        ),
        label="base-checkpoint validation",
    )
    release_index = _load_release_index(
        family=family,
        pipeline_sha256=pipeline_sha256,
        topology=topology,
    )
    tokenizer_payload, tokenizer_sha256, tokenizer_path = _load_tokenizer(
        topology, release_index
    )
    (
        canonical_id_lookup,
        canonical_map_self_sha256,
        canonical_ids_sha256,
    ) = _load_canonical_lookup(
        data_release=getattr(args, "data_release"),
        tokenizer_payload=tokenizer_payload,
        tokenizer_manifest_path=tokenizer_path,
        topology=topology,
    )

    output_root = (
        Path(str(getattr(args, "output"))).expanduser().resolve()
        / "posttraining"
        / family
    )

    def create_output() -> None:
        if topology.rank == 0:
            output_root.mkdir(parents=True, exist_ok=True)

    _collective_errors(topology, create_output, label="post-training output creation")
    barrier(topology)
    manager = CheckpointManager(output_root, topology=topology, keep_last=3)
    state_path = output_root / "STATE.json"
    state = _read_campaign_state(
        path=state_path,
        topology=topology,
        family=family,
        pipeline_sha256=pipeline_sha256,
        family_manifest_sha256=family_manifest_sha256,
        base_checkpoint_sha256=base_checkpoint_sha256,
        base_checkpoint=base_checkpoint_path,
        base_receipt=base_receipt_path,
    )
    completed = _require_list(state["completed"], "campaign completed")
    if len(completed) == len(EXPECTED_STAGE_IDS):
        return {
            "family": family,
            "complete": True,
            "policy_checkpoint_sha256": state["policy_checkpoint_sha256"],
            "policy_checkpoint_receipt": state["policy_checkpoint_receipt"],
            "evaluation_receipt": state["evaluation_receipt"],
            "release_candidate": completed[-1].get("release_candidate"),
        }

    active = state.get("active")
    if active is not None:
        active = _require_mapping(active, "campaign active")
        expected_active = EXPECTED_STAGE_IDS[len(completed)]
        if active.get("stage_id") != expected_active:
            raise StageBackendError("active-stage resume order is corrupt")
        active_kind = str(active.get("kind", "policy"))
        if active_kind == "policy":
            _load_distributed_checkpoint(
                manager=manager,
                checkpoint=str(active["checkpoint_path"]),
                contract=_require_mapping(
                    active["checkpoint_contract"], "active checkpoint contract"
                ),
                model=model,
                optimizer=optimizer,
            )
            state = _reconcile_active_policy_checkpoint_state(
                state_path=state_path,
                state=state,
                topology=topology,
            )
            active = _require_mapping(state["active"], "campaign active")
        else:
            raise StageBackendError(f"unknown active checkpoint kind: {active_kind}")
    elif state.get("policy_checkpoint_contract") is not None:
        _load_distributed_checkpoint(
            manager=manager,
            checkpoint=str(state["policy_checkpoint_path"]),
            contract=_require_mapping(
                state["policy_checkpoint_contract"], "policy checkpoint contract"
            ),
            model=model,
            optimizer=optimizer,
        )
    barrier(topology)
    forward_model = _compile_posttraining_forward_model(
        model,
        compile_mode=autotune_selection.compile_mode,
    )

    for stage_index in range(len(completed), len(EXPECTED_STAGE_IDS)):
        if _signal_requested(
            signal_coordinator, topology=topology, runtime=runtime
        ):
            # The previous completed-stage receipt is already durable. No
            # optimizer work for the next stage has begun.
            raise PostTrainingRequeue(
                signal_coordinator.reason or "remote-rank-signal"
            )
        stage_id = EXPECTED_STAGE_IDS[stage_index]
        stage = stages[stage_id]
        stage_config_sha256 = _canonical_hash(stage)
        optimizer_state_policy = _optimizer_state_policy(stage)
        parent_stage = str(stage.get("input_stage", ""))
        if not parent_stage:
            raise StageBackendError(f"{stage_id} omits input_stage")
        parent_checkpoint_sha256 = str(state["policy_checkpoint_sha256"])
        if parent_stage == "base_pretraining":
            expected_parent_checkpoint_sha256 = base_checkpoint_sha256
        else:
            parent_records = [
                item
                for item in state["completed"]
                if isinstance(item, Mapping)
                and item.get("stage_id") == parent_stage
            ]
            if len(parent_records) != 1:
                raise StageBackendError(
                    f"{stage_id} cannot locate its unique parent stage "
                    f"{parent_stage}"
                )
            expected_parent_checkpoint_sha256 = str(
                parent_records[0]["policy_checkpoint_sha256"]
            )
        if parent_checkpoint_sha256 != expected_parent_checkpoint_sha256:
            raise StageBackendError(
                f"{stage_id} live policy is not its declared {parent_stage} "
                "parent checkpoint"
            )
        branch_unified_state: dict[str, Any] | None = None
        if _is_specialist_stage(stage_id):
            branch_unified_state = {
                "policy_checkpoint_sha256": state[
                    "policy_checkpoint_sha256"
                ],
                "policy_checkpoint_path": state["policy_checkpoint_path"],
                "policy_checkpoint_receipt": state[
                    "policy_checkpoint_receipt"
                ],
                "policy_checkpoint_contract": dict(
                    _require_mapping(
                        state["policy_checkpoint_contract"],
                        f"{stage_id} unified parent checkpoint contract",
                    )
                ),
            }
        stage_bindings: dict[str, Any] = {}
        raw_parent_contract = state.get("policy_checkpoint_contract")
        if isinstance(raw_parent_contract, Mapping):
            stage_bindings["parent_policy_checkpoint"] = {
                "stage_id": parent_stage,
                "checkpoint_path": str(state["policy_checkpoint_path"]),
                "checkpoint_sha256": parent_checkpoint_sha256,
                "checkpoint_receipt": str(
                    state["policy_checkpoint_receipt"]
                ),
                "checkpoint_contract": dict(raw_parent_contract),
            }
        if stage_id == "opd_consolidation":
            specialist_records = {
                str(item.get("stage_id")): item
                for item in state["completed"]
                if isinstance(item, Mapping)
                and item.get("stage_id") in SPECIALIST_STAGE_IDS
            }
            if set(specialist_records) != set(SPECIALIST_STAGE_IDS):
                raise StageBackendError(
                    "OPD cannot bind every completed Metis specialist"
                )
            stage_bindings["specialist_checkpoints"] = {
                specialist_id: {
                    "checkpoint_path": str(
                        specialist_records[specialist_id]["checkpoint_path"]
                    ),
                    "checkpoint_sha256": str(
                        specialist_records[specialist_id][
                            "policy_checkpoint_sha256"
                        ]
                    ),
                    "checkpoint_receipt": str(
                        specialist_records[specialist_id][
                            "checkpoint_receipt"
                        ]
                    ),
                    "checkpoint_contract": dict(
                        _require_mapping(
                            specialist_records[specialist_id][
                                "checkpoint_contract"
                            ],
                            f"{specialist_id} checkpoint contract",
                        )
                    ),
                }
                for specialist_id in SPECIALIST_STAGE_IDS
            }
            stage_bindings["unified_student_checkpoint"] = {
                "stage_id": "hybrid_mode_gspo",
                "checkpoint_path": str(state["policy_checkpoint_path"]),
                "checkpoint_sha256": parent_checkpoint_sha256,
                "checkpoint_receipt": str(
                    state["policy_checkpoint_receipt"]
                ),
                "checkpoint_contract": dict(
                    _require_mapping(
                        state["policy_checkpoint_contract"],
                        "OPD unified student checkpoint contract",
                    )
                ),
            }
        requirements = _resolve_requirements(
            stage,
            family=family,
            parent_stage=parent_stage,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            tokenizer_sha256=tokenizer_sha256,
            topology=topology,
            output_root=output_root,
            stage_bindings=stage_bindings,
            release_index=release_index,
        )
        metrics: dict[str, Any]
        checkpoint_path: Path | None = None
        checkpoint_contract: dict[str, str] | None = None
        checkpoint_receipt: Path | None = None
        release_candidate: Path | None = None
        stage_policy_checkpoint_sha256: str | None = None
        context_baseline_metrics: dict[str, float] | None = None

        if stage_id in {
            "context_extension",
            "cold_start_sft",
            "overall_sft",
            *RLVR_GSPO_STAGE_IDS,
            "opd_consolidation",
        }:
            requirement = _data_requirement(requirements, stage_id=stage_id)
            bundle = _collective_errors(
                topology,
                lambda: MMapStageBundle.load(
                    requirement,
                    stage_id=stage_id,
                    family=family,
                    tokenizer_sha256=tokenizer_sha256,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    vocabulary_size=int(config.vocab_size),
                    canonical_id_lookup=canonical_id_lookup,
                    canonical_map_self_sha256=canonical_map_self_sha256,
                    canonical_ids_sha256=canonical_ids_sha256,
                ),
                label=f"{stage_id} mmap-bundle load",
            )
            bundle = _load_stage_batch_migration(
                bundle,
                family=family,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                precision_role_plan_sha256=(
                    autotune_selection.precision_role_plan_sha256
                ),
                output_root=output_root,
                topology=topology,
            )
            bundle = _run_live_working_set_autotune(
                stage=stage,
                stage_config_sha256=stage_config_sha256,
                bundle=bundle,
                model=model,
                forward_model=forward_model,
                runtime=runtime,
                topology=topology,
                output_root=output_root,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                precision_role_plan_sha256=(
                    autotune_selection.precision_role_plan_sha256
                ),
                autotune_profile_sha256=autotune_profile_sha256,
                compile_mode=autotune_selection.compile_mode,
                vocabulary_size=int(config.vocab_size),
                signal_coordinator=signal_coordinator,
                active_resume=state.get("active") is not None,
            )
            _validate_runtime_working_set(
                bundle,
                runtime=runtime,
                topology=topology,
                vocabulary_size=int(config.vocab_size),
            )
            if stage_id == "context_extension":
                baseline_path = (
                    output_root / "context_gates" / "BASELINE.json"
                )

                def read_context_baseline() -> dict[str, Any] | None:
                    if topology.rank != 0 or not baseline_path.is_file():
                        return None
                    payload = _read_json(
                        baseline_path,
                        label="context gate baseline",
                    )
                    if (
                        payload.get("schema")
                        != CONTEXT_GATE_BASELINE_SCHEMA
                        or payload.get("family") != family
                        or payload.get("base_checkpoint_sha256")
                        != parent_checkpoint_sha256
                        or payload.get("bundle_sha256")
                        != bundle.manifest_sha256
                        or payload.get("receipt_sha256")
                        != _canonical_hash(
                            payload, omit={"receipt_sha256"}
                        )
                    ):
                        raise StageBackendError(
                            "context gate baseline is stale or corrupt"
                        )
                    return payload

                raw_baseline = _collective_errors(
                    topology,
                    read_context_baseline,
                    label="context gate baseline load",
                )
                raw_baseline = _broadcast_object(
                    raw_baseline if topology.rank == 0 else None,
                    topology,
                )
                if raw_baseline is None:
                    if state.get("active") is not None:
                        raise StageBackendError(
                            "context resume lost its pre-update gate baseline"
                        )
                    measured_baseline = _evaluate_context_checkpoint(
                        bundle=bundle,
                        model=model,
                        forward_model=forward_model,
                        runtime=runtime,
                        topology=topology,
                    )
                    baseline_path = _rank_zero_path(
                        topology,
                        lambda: _write_context_gate_baseline(
                            output_root=output_root,
                            family=family,
                            base_checkpoint_sha256=(
                                parent_checkpoint_sha256
                            ),
                            bundle_sha256=bundle.manifest_sha256,
                            metrics=measured_baseline,
                        ),
                        label="context gate baseline write",
                    )
                    raw_baseline = _read_json(
                        baseline_path,
                        label="context gate baseline",
                    )
                baseline_values = _require_mapping(
                    _require_mapping(
                        raw_baseline, "context gate baseline"
                    ).get("metrics"),
                    "context gate baseline metrics",
                )
                context_baseline_metrics = {
                    str(name): float(value)
                    for name, value in baseline_values.items()
                    if (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                    )
                }
                if len(context_baseline_metrics) != len(
                    baseline_values
                ):
                    raise StageBackendError(
                        "context gate baseline metrics are invalid"
                    )
            if stage_id == "opd_consolidation":
                expected_specialist_hashes = {
                    specialist_id: str(
                        _require_mapping(
                            _require_mapping(
                                stage_bindings["specialist_checkpoints"],
                                "OPD specialist checkpoint bindings",
                            )[specialist_id],
                            f"OPD {specialist_id} checkpoint binding",
                        )["checkpoint_sha256"]
                    )
                    for specialist_id in SPECIALIST_STAGE_IDS
                }
                if (
                    bundle.manifest.get("specialist_checkpoints")
                    != expected_specialist_hashes
                    or bundle.manifest.get(
                        "student_checkpoint_sha256"
                    )
                    != parent_checkpoint_sha256
                    or bundle.manifest.get("stage_bindings_sha256")
                    != _canonical_hash(stage_bindings)
                ):
                    raise StageBackendError(
                        "OPD rollouts are not bound to the exact unified "
                        "student and all five completed specialist checkpoints"
                    )
            active_for_stage = (
                _require_mapping(state["active"], "active stage")
                if state.get("active") is not None
                else None
            )
            if active_for_stage is not None:
                if (
                    active_for_stage.get("stage_id") != stage_id
                    or active_for_stage.get("bundle_sha256") != bundle.manifest_sha256
                    or active_for_stage.get("parent_checkpoint_sha256")
                    != parent_checkpoint_sha256
                    or active_for_stage.get("optimizer_state_policy")
                    != optimizer_state_policy
                ):
                    raise StageBackendError("active checkpoint does not match this stage/bundle")
                start_epoch = int(active_for_stage["epoch"])
                start_global_batch = _resume_global_batch(
                    active_for_stage,
                    bundle,
                )
                start_optimizer_step = int(active_for_stage["optimizer_step"])
                start_cursor = int(active_for_stage["campaign_token_cursor"])
                raw_resume_metrics = _require_mapping(
                    active_for_stage.get("last_metrics"),
                    "active stage last_metrics",
                )
                resume_metrics = {
                    str(name): float(value)
                    for name, value in raw_resume_metrics.items()
                    if (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                    )
                }
                if (
                    len(resume_metrics) != len(raw_resume_metrics)
                    or float(active_for_stage.get("last_loss", float("nan")))
                    != float(resume_metrics.get("loss", float("nan")))
                    or float(active_for_stage.get("last_grad_norm", float("nan")))
                    != float(resume_metrics.get("grad_norm", float("nan")))
                ):
                    raise StageBackendError(
                        "active checkpoint last metrics are invalid"
                    )
            else:
                start_epoch = 0
                start_global_batch = 0
                start_optimizer_step = 0
                start_cursor = int(state["campaign_token_cursor"])
                resume_metrics = None
            _apply_optimizer_state_transition(
                optimizer,
                policy=optimizer_state_policy,
                active_resume=active_for_stage is not None,
            )
            live_profile_selection: ProfileSelection | None = None
            if _is_gspo_stage(stage_id):
                parent_contract = state.get("policy_checkpoint_contract")
                if not isinstance(parent_contract, Mapping):
                    raise StageBackendError(
                        f"{stage_id} live tuning cannot restore its exact parent "
                        "checkpoint contract"
                    )

                def restore_profile_parent() -> None:
                    _load_distributed_checkpoint(
                        manager=manager,
                        checkpoint=str(state["policy_checkpoint_path"]),
                        contract=_require_mapping(
                            parent_contract,
                            f"{stage_id} parent checkpoint contract",
                        ),
                        model=model,
                        optimizer=optimizer,
                    )
                    barrier(topology)

                live_profile_selection = _run_live_profile_autotune(
                    stage=stage,
                    stage_config_sha256=stage_config_sha256,
                    bundle=bundle,
                    model=model,
                    optimizer=optimizer,
                    forward_model=forward_model,
                    runtime=runtime,
                    topology=topology,
                    output_root=output_root,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    precision_role_plan_sha256=(
                        autotune_selection.precision_role_plan_sha256
                    ),
                    autotune_profile_sha256=autotune_profile_sha256,
                    compile_mode=autotune_selection.compile_mode,
                    optimizer_state_policy=optimizer_state_policy,
                    restore_parent=restore_profile_parent,
                    signal_coordinator=signal_coordinator,
                    active_resume=active_for_stage is not None,
                )

            def checkpoint_callback(
                progress: StageProgress,
                complete_checkpoint: bool,
                gate_target_tokens: int | None = None,
            ) -> None:
                nonlocal checkpoint_path, checkpoint_contract
                if gate_target_tokens is not None and stage_id != "context_extension":
                    raise StageBackendError(
                        "only context extension may emit a token gate"
                    )
                (
                    checkpoint_path,
                    checkpoint_sha,
                    checkpoint_contract,
                ) = _save_policy_checkpoint(
                    manager=manager,
                    model=model,
                    optimizer=optimizer,
                    policy=policy,
                    stage_id=stage_id,
                    stage_config_sha256=stage_config_sha256,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    bundle=bundle,
                    requirements=requirements,
                    family_manifest_sha256=family_manifest_sha256,
                    pipeline_sha256=pipeline_sha256,
                    autotune_profile_sha256=autotune_profile_sha256,
                    precision_role_plan_sha256=(
                        autotune_selection.precision_role_plan_sha256
                    ),
                    progress=progress,
                    complete=complete_checkpoint,
                    optimizer_state_policy=optimizer_state_policy,
                    milestone=(
                        f"context_gate_{gate_target_tokens}"
                        if gate_target_tokens is not None
                        else None
                    ),
                )
                if gate_target_tokens is not None:
                    if context_baseline_metrics is None:
                        raise StageBackendError(
                            "context gate has no immutable pre-update baseline"
                        )
                    maximum_overshoot = int(
                        _require_mapping(
                            stage["gate_policy"],
                            "context_extension.gate_policy",
                        )["maximum_gate_overshoot_tokens"]
                    )
                    evaluation_metrics = _evaluate_context_checkpoint(
                        bundle=bundle,
                        model=model,
                        forward_model=forward_model,
                        runtime=runtime,
                        topology=topology,
                    )
                    evaluation_receipt = _rank_zero_path(
                        topology,
                        lambda: _write_context_gate_evaluation_receipt(
                            output_root=output_root,
                            family=family,
                            base_checkpoint_sha256=(
                                parent_checkpoint_sha256
                            ),
                            checkpoint_sha256=checkpoint_sha,
                            bundle_sha256=bundle.manifest_sha256,
                            gate_target_tokens=gate_target_tokens,
                            metrics=evaluation_metrics,
                            baseline=context_baseline_metrics,
                            gate_policy=_require_mapping(
                                bundle.manifest["gate_policy"],
                                "context bundle gate_policy",
                            ),
                        ),
                        label=(
                            f"context gate {gate_target_tokens} "
                            "evaluation receipt"
                        ),
                    )
                    evaluation_payload = _read_json(
                        evaluation_receipt,
                        label=(
                            f"context gate {gate_target_tokens} "
                            "evaluation receipt"
                        ),
                    )
                    gate_receipt = _rank_zero_path(
                        topology,
                        lambda: _write_context_gate_receipt(
                            output_root=output_root,
                            family=family,
                            stage_config_sha256=stage_config_sha256,
                            parent_checkpoint_sha256=parent_checkpoint_sha256,
                            checkpoint=checkpoint_path,
                            checkpoint_sha256=checkpoint_sha,
                            checkpoint_contract=checkpoint_contract,
                            progress=progress,
                            gate_target_tokens=gate_target_tokens,
                            maximum_overshoot_tokens=maximum_overshoot,
                            complete=complete_checkpoint,
                            evaluation_receipt=evaluation_receipt,
                            evaluation_receipt_sha256=str(
                                evaluation_payload["receipt_sha256"]
                            ),
                        ),
                        label=f"context gate {gate_target_tokens} receipt",
                    )
                    raw_gate_receipts = state.setdefault(
                        "context_gate_receipts", {}
                    )
                    if not isinstance(raw_gate_receipts, dict):
                        raise StageBackendError(
                            "campaign context gate receipt index is invalid"
                        )
                    raw_gate_receipts[str(gate_target_tokens)] = str(
                        gate_receipt
                    )
                if not complete_checkpoint:
                    state["active"] = {
                        "kind": "policy",
                        "stage_id": stage_id,
                        "stage_config_sha256": stage_config_sha256,
                        "parent_checkpoint_sha256": parent_checkpoint_sha256,
                        "bundle_sha256": bundle.manifest_sha256,
                        "optimizer_state_policy": optimizer_state_policy,
                        "runtime_batch": _runtime_batch_payload(bundle),
                        "epoch": progress.epoch,
                        "next_global_batch": progress.next_global_batch,
                        "optimizer_step": progress.optimizer_step,
                        "campaign_token_cursor": progress.campaign_token_cursor,
                        "last_loss": progress.loss,
                        "last_grad_norm": progress.grad_norm,
                        "last_metrics": dict(progress.metrics),
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_contract": checkpoint_contract,
                    }
                    _write_campaign_state(state_path, state, topology)

            def policy_oom_resume() -> dict[str, Any]:
                current_active = state.get("active")
                if isinstance(current_active, Mapping) and (
                    current_active.get("kind", "policy") == "policy"
                    and current_active.get("stage_id") == stage_id
                ):
                    resume_epoch = int(current_active["epoch"])
                    resume_batch = _resume_global_batch(
                        current_active,
                        bundle,
                    )
                    resume_step = int(current_active["optimizer_step"])
                    resume_cursor = int(
                        current_active["campaign_token_cursor"]
                    )
                    safe_path = str(current_active["checkpoint_path"])
                    safe_sha256 = str(
                        current_active["checkpoint_sha256"]
                    )
                else:
                    resume_epoch = start_epoch
                    resume_batch = start_global_batch
                    resume_step = start_optimizer_step
                    resume_cursor = start_cursor
                    safe_path = str(state["policy_checkpoint_path"])
                    safe_sha256 = str(
                        state["policy_checkpoint_sha256"]
                    )
                return {
                    "checkpoint_kind": "policy",
                    "checkpoint_path": safe_path,
                    "checkpoint_sha256": safe_sha256,
                    "epoch": resume_epoch,
                    "next_global_batch": resume_batch,
                    "optimizer_step": resume_step,
                    "campaign_token_cursor": resume_cursor,
                    "last_loss": (
                        float(current_active.get("last_loss", 0.0))
                        if isinstance(current_active, Mapping)
                        else float((resume_metrics or {}).get("loss", 0.0))
                    ),
                    "last_grad_norm": (
                        float(current_active.get("last_grad_norm", 0.0))
                        if isinstance(current_active, Mapping)
                        else float(
                            (resume_metrics or {}).get("grad_norm", 0.0)
                        )
                    ),
                    "last_metrics": (
                        dict(
                            _require_mapping(
                                current_active.get("last_metrics"),
                                "OOM resume last_metrics",
                            )
                        )
                        if isinstance(current_active, Mapping)
                        and current_active.get("last_metrics") is not None
                        else dict(resume_metrics or {})
                    ),
                    "runtime_batch": _runtime_batch_payload(bundle),
                    "checkpoint_safe": True,
                    "rollback_required": True,
                }

            try:
                kernel_canary = _run_stage_kernel_canary(
                    stage=stage,
                    bundle=bundle,
                    model=model,
                    forward_model=forward_model,
                    runtime=runtime,
                    topology=topology,
                    output_root=output_root,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    compile_mode=autotune_selection.compile_mode,
                )
            except Exception as exception:
                _raise_stage_oom(
                    output_root=output_root,
                    family=family,
                    stage_id=stage_id,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    precision_role_plan_sha256=(
                        autotune_selection.precision_role_plan_sha256
                    ),
                    bundle=bundle,
                    topology=topology,
                    phase="kernel_canary",
                    resume=policy_oom_resume(),
                    exception=exception,
                )
                raise AssertionError("unreachable after stage OOM")

            try:
                if stage_id in {
                    "context_extension",
                    "cold_start_sft",
                    "overall_sft",
                }:
                    metrics = _run_supervised_stage(
                        stage=stage,
                        bundle=bundle,
                        model=model,
                        optimizer=optimizer,
                        runtime=runtime,
                        topology=topology,
                        start_epoch=start_epoch,
                        start_global_batch=start_global_batch,
                        start_optimizer_step=start_optimizer_step,
                        start_cursor=start_cursor,
                        checkpoint_callback=checkpoint_callback,
                        signal_coordinator=signal_coordinator,
                        forward_model=forward_model,
                        resume_metrics=resume_metrics,
                    )
                elif stage_id == "opd_consolidation":
                    metrics = _run_opd_stage(
                        stage=stage,
                        bundle=bundle,
                        model=model,
                        optimizer=optimizer,
                        runtime=runtime,
                        topology=topology,
                        start_epoch=start_epoch,
                        start_global_batch=start_global_batch,
                        start_optimizer_step=start_optimizer_step,
                        start_cursor=start_cursor,
                        checkpoint_callback=checkpoint_callback,
                        signal_coordinator=signal_coordinator,
                        forward_model=forward_model,
                        resume_metrics=resume_metrics,
                    )
                else:
                    if not _is_gspo_stage(stage_id):
                        raise StageBackendError(
                            f"{stage_id} has no policy-stage runner"
                        )
                    metrics = _run_gspo_stage(
                        stage=stage,
                        bundle=bundle,
                        model=model,
                        optimizer=optimizer,
                        runtime=runtime,
                        topology=topology,
                        start_epoch=start_epoch,
                        start_global_batch=start_global_batch,
                        start_optimizer_step=start_optimizer_step,
                        start_cursor=start_cursor,
                        checkpoint_callback=checkpoint_callback,
                        signal_coordinator=signal_coordinator,
                        forward_model=forward_model,
                        selection_override=live_profile_selection,
                        resume_metrics=resume_metrics,
                    )
            except Exception as exception:
                _raise_stage_oom(
                    output_root=output_root,
                    family=family,
                    stage_id=stage_id,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    precision_role_plan_sha256=(
                        autotune_selection.precision_role_plan_sha256
                    ),
                    bundle=bundle,
                    topology=topology,
                    phase="forward_backward",
                    resume=policy_oom_resume(),
                    exception=exception,
                )
                raise AssertionError("unreachable after stage OOM")
            metrics.update(
                {
                    "runtime_batch": _runtime_batch_payload(bundle),
                    "kernel_canary_receipt_sha256": kernel_canary[
                        "receipt_sha256"
                    ],
                    "compile_mode": autotune_selection.compile_mode,
                    "compiled_trunk": (
                        autotune_selection.compile_mode
                        not in {"eager", "none"}
                    ),
                    "head_execution": (
                        "eager_exact_chunked"
                        if stage_id in {
                            *RLVR_GSPO_STAGE_IDS,
                            "opd_consolidation",
                        }
                        else "model_native_chunked"
                    ),
                }
            )
            final_progress = StageProgress(
                epoch=int(bundle.training["epochs"]),
                next_global_batch=0,
                optimizer_step=int(metrics["optimizer_steps"]),
                campaign_token_cursor=int(metrics["campaign_token_cursor"]),
                loss=float(metrics.get("loss", 0.0)),
                grad_norm=float(metrics.get("grad_norm", 0.0)),
                metrics={
                    str(name): float(value)
                    for name, value in metrics.items()
                    if (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                    )
                },
            )
            checkpoint_callback(
                final_progress,
                True,
                (
                    int(stage["checkpoint_gates"][-1])
                    if stage_id == "context_extension"
                    else None
                ),
            )
            assert checkpoint_path is not None and checkpoint_contract is not None
            if stage_id == "context_extension":
                promotion_path = _rank_zero_path(
                    topology,
                    lambda: _write_context_gate_promotion(
                        output_root=output_root,
                        family=family,
                        gate_receipts=_require_mapping(
                            state["context_gate_receipts"],
                            "context gate receipt index",
                        ),
                    ),
                    label="context gate promotion",
                )
                promotion = _read_json(
                    promotion_path,
                    label="context gate promotion",
                )
                selected_gate = _require_mapping(
                    promotion["selected"],
                    "context gate promotion selected",
                )
                selected_checkpoint = Path(
                    str(selected_gate["checkpoint_path"])
                ).expanduser().resolve()
                selected_contract = dict(
                    _require_mapping(
                        selected_gate["checkpoint_contract"],
                        "promoted context checkpoint contract",
                    )
                )
                if (
                    str(selected_gate["checkpoint_sha256"])
                    != _read_json(
                        selected_checkpoint / "MANIFEST.json",
                        label="promoted context checkpoint",
                    ).get("checkpoint_sha256")
                ):
                    raise StageBackendError(
                        "promoted context checkpoint hash changed"
                    )
                if selected_checkpoint != checkpoint_path:
                    _load_distributed_checkpoint(
                        manager=manager,
                        checkpoint=selected_checkpoint,
                        contract=selected_contract,
                        model=model,
                        optimizer=optimizer,
                    )
                    barrier(topology)
                checkpoint_path = selected_checkpoint
                checkpoint_contract = selected_contract
                metrics.update(
                    {
                        "promoted_context_gate_tokens": int(
                            selected_gate["gate_target_tokens"]
                        ),
                        "context_gate_promotion_score": float(
                            selected_gate["promotion_score"]
                        ),
                        "context_gate_promotion_receipt_sha256": str(
                            promotion["receipt_sha256"]
                        ),
                    }
                )
                state["context_gate_promotion"] = str(promotion_path)
            checkpoint_manifest = _read_json(
                checkpoint_path / "MANIFEST.json",
                label=f"{stage_id} checkpoint manifest",
            )
            checkpoint_sha = str(checkpoint_manifest["checkpoint_sha256"])
            stage_policy_checkpoint_sha256 = checkpoint_sha
            checkpoint_receipt = _rank_zero_path(
                topology,
                lambda: _write_checkpoint_receipt(
                    output_root=output_root,
                    family=family,
                    stage_id=stage_id,
                    stage_config_sha256=stage_config_sha256,
                    parent_checkpoint_sha256=parent_checkpoint_sha256,
                    checkpoint=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                    precision_role_plan_sha256=(
                        autotune_selection.precision_role_plan_sha256
                    ),
                ),
                label=f"{stage_id} checkpoint receipt",
            )
            state["policy_checkpoint_sha256"] = checkpoint_sha
            state["policy_checkpoint_path"] = str(checkpoint_path)
            state["policy_checkpoint_receipt"] = str(checkpoint_receipt)
            state["policy_checkpoint_contract"] = checkpoint_contract
            state["campaign_token_cursor"] = int(metrics["campaign_token_cursor"])

        elif stage_id == "evaluation":
            requirement = _data_requirement(requirements, stage_id=stage_id)
            metrics = _run_evaluation(
                stage=stage,
                requirement=requirement,
                family=family,
                checkpoint_sha256=parent_checkpoint_sha256,
                topology=topology,
            )

        elif stage_id == "publish_gate":
            if not state.get("evaluation_receipt"):
                raise StageBackendError("publish gate cannot locate a passed evaluation receipt")
            release_candidate = _write_release_candidate(
                output_root=output_root,
                family=family,
                policy_checkpoint_sha256=parent_checkpoint_sha256,
                policy_checkpoint_receipt=str(state["policy_checkpoint_receipt"]),
                evaluation_receipt=str(state["evaluation_receipt"]),
                topology=topology,
            )
            metrics = {
                "published_locally": True,
                "external_upload": False,
                "release_candidate": str(release_candidate),
            }
        else:
            raise StageBackendError(f"unsupported post-training stage {stage_id}")

        output_receipt = _rank_zero_path(
            topology,
            lambda: _write_stage_receipt(
                output_root=output_root,
                family=family,
                stage_id=stage_id,
                stage_config_sha256=stage_config_sha256,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                policy_checkpoint_sha256=str(state["policy_checkpoint_sha256"]),
                precision_role_plan_sha256=(
                    autotune_selection.precision_role_plan_sha256
                ),
                requirements=requirements,
                metrics=metrics,
                checkpoint_receipt=checkpoint_receipt,
                optimizer_state_policy=optimizer_state_policy,
            ),
            label=f"{stage_id} output receipt",
        )
        if stage_id == "evaluation":
            state["evaluation_receipt"] = str(output_receipt)
        state["active"] = None
        record = _completed_record(
            stage_id=stage_id,
            stage_config_sha256=stage_config_sha256,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            state=state,
            output_receipt=output_receipt,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            checkpoint_contract=checkpoint_contract,
            checkpoint_receipt=checkpoint_receipt,
            release_candidate=release_candidate,
            policy_checkpoint_sha256=stage_policy_checkpoint_sha256,
        )
        state["completed"].append(record)
        if branch_unified_state is not None:
            _load_distributed_checkpoint(
                manager=manager,
                checkpoint=str(
                    branch_unified_state["policy_checkpoint_path"]
                ),
                contract=_require_mapping(
                    branch_unified_state["policy_checkpoint_contract"],
                    f"{stage_id} unified checkpoint contract",
                ),
                model=model,
                optimizer=optimizer,
            )
            barrier(topology)
            # The specialist checkpoint remains sealed in its completed
            # record. The live campaign policy returns to the untouched
            # shared hybrid-mode student so the next specialist is an
            # independent branch and OPD starts from that same unified policy.
            state.update(branch_unified_state)
        _write_campaign_state(state_path, state, topology)

    return {
        "family": family,
        "complete": True,
        "policy_checkpoint_sha256": state["policy_checkpoint_sha256"],
        "policy_checkpoint_receipt": state["policy_checkpoint_receipt"],
        "evaluation_receipt": state["evaluation_receipt"],
        "release_candidate": state["completed"][-1]["release_candidate"],
    }


def run_posttraining_campaign(
    args: Any,
    config: Any,
    model: nn.Module,
    optimizer: OptimizerBundle,
    policy: Any,
    runtime: Runtime,
    topology: ParallelTopology,
    family_manifest: Mapping[str, Any],
    base_checkpoint: str | Path,
    base_receipt: str | Path,
    posttraining_manifest: str | Path,
    signal_coordinator: SignalCoordinator | None = None,
) -> dict[str, Any]:
    """Run post-training with checkpoint-safe Slurm signal ownership.

    ``train.py`` may pass a coordinator whose context spans base and
    post-training. When it does not, this backend installs its own coordinator
    for the entire campaign and restores the prior handlers on return.
    """

    if signal_coordinator is not None:
        return _run_posttraining_campaign_impl(
            args=args,
            config=config,
            model=model,
            optimizer=optimizer,
            policy=policy,
            runtime=runtime,
            topology=topology,
            family_manifest=family_manifest,
            base_checkpoint=base_checkpoint,
            base_receipt=base_receipt,
            posttraining_manifest=posttraining_manifest,
            signal_coordinator=signal_coordinator,
        )
    with SignalCoordinator() as owned_coordinator:
        return _run_posttraining_campaign_impl(
            args=args,
            config=config,
            model=model,
            optimizer=optimizer,
            policy=policy,
            runtime=runtime,
            topology=topology,
            family_manifest=family_manifest,
            base_checkpoint=base_checkpoint,
            base_receipt=base_receipt,
            posttraining_manifest=posttraining_manifest,
            signal_coordinator=owned_coordinator,
        )


__all__ = [
    "ArraySpec",
    "DeferredMaterialization",
    "MMapStageBundle",
    "PostTrainingRequeue",
    "SealedRequirement",
    "StageBackendError",
    "StageProgress",
    "run_posttraining_campaign",
]
