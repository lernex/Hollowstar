from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

from .autotune import (
    Candidate,
    inventory_fingerprint,
    load_tuning_bounds,
    tune_family,
    validate_performance_report,
    validate_profile,
)
from .config import FamilyTopology, PortageConfig, load_portage_config
from .release import validate_release_marker
from .posttraining_release import environment_for_family
from .runtime import validate_compute_runtime
from .util import (
    atomic_write_json,
    file_sha256,
    json_sha256,
    read_json,
    safe_environment,
    utc_now,
)
from metis_training.model_config import load_family_config
from metis_training.precision_plan import (
    measured_role_dtype_map,
    validate_precision_role_inventory,
    validate_precision_role_plan,
)


REQUEUE_EXIT_CODE = 75
DEFERRED_MATERIALIZATION_EXIT_CODE = 252
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE_BATCH_MIGRATION_SCHEMA = "metis.posttraining-batch-migration/v1"
_STAGE_OOM_REQUEST_SCHEMA = "metis.posttraining-oom-revision-request/v1"
_DEFERRED_REQUEST_SCHEMA = "metis.deferred-materialization-request/v1"
_POSTTRAINING_STATE_SCHEMA = "metis.inprocess-posttraining-state/v1"
_POSTTRAINING_STAGE_IDS = (
    "context_extension",
    "cold_start_sft",
    "overall_sft",
    "hybrid_mode_gspo",
    "specialist_reasoning",
    "specialist_code",
    "specialist_knowledge",
    "specialist_writing",
    "specialist_agentic",
    "opd_consolidation",
    "evaluation",
    "publish_gate",
)
_SPECIALIST_STAGE_IDS = (
    "specialist_reasoning",
    "specialist_code",
    "specialist_knowledge",
    "specialist_writing",
    "specialist_agentic",
)
_POSTTRAINING_PARENT_STAGE = {
    "hybrid_mode_gspo": "overall_sft",
    **{stage_id: "hybrid_mode_gspo" for stage_id in _SPECIALIST_STAGE_IDS},
    "opd_consolidation": "hybrid_mode_gspo",
    "evaluation": "opd_consolidation",
    "publish_gate": "evaluation",
}


def derive_oom_candidate(
    *,
    profile: dict[str, Any],
    family: FamilyTopology,
    config: PortageConfig,
) -> Candidate:
    """Return the next checkpoint-compatible memory reduction.

    Only micro-batch and accumulation may change, and their product is held
    exactly constant. Precision, kernels, table placement, learning rate, and
    optimizer semantics therefore remain unchanged across the checkpoint.
    """

    selected = profile.get("selected", {})
    bounds = load_tuning_bounds(
        family.manifest,
        default_maximum_hbm_fraction=float(
            config.raw["autotune"]["default_maximum_hbm_fraction"]
        ),
    )
    old_micro = int(selected["micro_batch_size"])
    old_accum = int(selected["grad_accum_steps"])
    local_batch = old_micro * old_accum
    candidates = [
        (micro, accumulation)
        for micro in bounds.micro_batch_sizes
        for accumulation in bounds.grad_accum_steps
        if micro < old_micro
        and accumulation > old_accum
        and micro * accumulation == local_batch
    ]
    if not candidates:
        raise RuntimeError(
            f"{family.name} has no smaller checkpoint-compatible "
            "micro-batch/accumulation candidate"
        )
    micro, accumulation = max(candidates, key=lambda row: row[0])
    return Candidate(
        micro_batch_size=micro,
        grad_accum_steps=accumulation,
        precision_profile=str(selected["precision_profile"]),
        compile_mode=str(selected["compile_mode"]),
        dispatch_overlap=bool(selected["dispatch_overlap"]),
        ngram_table_mode=str(selected["ngram_table_mode"]),
        learning_rate=None,
    )


def validate_checkpoint_for_requeue(
    output: str | Path,
    *,
    family: FamilyTopology,
    require_checkpoint: bool,
) -> dict[str, Any]:
    root = Path(output).expanduser().resolve() / "checkpoints"
    partial = sorted(path.name for path in root.glob(".incomplete-*")) if root.is_dir() else []
    if partial:
        raise RuntimeError(
            f"{family.name} has incomplete checkpoint directories: {partial}"
        )
    pointer = root / "LATEST.json"
    if not pointer.is_file():
        if require_checkpoint:
            raise RuntimeError(
                f"{family.name} requested checkpoint requeue without LATEST.json"
            )
        return {
            "family": family.name,
            "status": "restart_from_origin",
            "checkpoint": None,
        }
    latest = read_json(pointer)
    if latest.get("schema") != "metis.checkpoint-latest/v1":
        raise RuntimeError(f"{family.name} LATEST.json has the wrong schema")
    raw_name = str(latest.get("checkpoint", ""))
    if not re.fullmatch(r"tokens-\d{13}", raw_name):
        raise RuntimeError(f"{family.name} LATEST checkpoint name is unsafe")
    checkpoint = (root / raw_name).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{family.name} LATEST escapes its checkpoint root") from exc
    manifest_path = checkpoint / "MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"{family.name} latest checkpoint manifest is missing")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "metis.distributed-checkpoint/v1"
        or manifest.get("family") != family.name
        or manifest.get("checkpoint_sha256")
        != json_sha256(manifest, omit=("checkpoint_sha256",))
        or int(manifest.get("world_size", 0)) != family.world_size
        or int(manifest.get("expert_parallel_size", 0))
        != family.expert_parallel_size
        or int(manifest.get("expert_replica_count", 0))
        != family.expert_replicas
        or not _SHA256.fullmatch(
            str(manifest.get("precision_role_plan_sha256", ""))
        )
    ):
        raise RuntimeError(f"{family.name} latest checkpoint lineage is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(f"{family.name} checkpoint has no artifacts")
    seen: set[str] = set()
    for record in artifacts:
        if not isinstance(record, dict):
            raise RuntimeError(f"{family.name} checkpoint artifact record is invalid")
        relative = str(record.get("path", ""))
        if (
            not relative
            or Path(relative).is_absolute()
            or relative in seen
            or not _SHA256.fullmatch(str(record.get("sha256", "")))
        ):
            raise RuntimeError(f"{family.name} checkpoint artifact record is unsafe")
        seen.add(relative)
        artifact = (checkpoint / relative).resolve()
        try:
            artifact.relative_to(checkpoint)
        except ValueError as exc:
            raise RuntimeError(
                f"{family.name} checkpoint artifact escapes its root"
            ) from exc
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != int(record.get("bytes", -1))
        ):
            raise RuntimeError(
                f"{family.name} checkpoint artifact is missing or size-drifted: {relative}"
            )
        if file_sha256(artifact) != record["sha256"]:
            raise RuntimeError(
                f"{family.name} checkpoint artifact hash drifted: {relative}"
            )
    return {
        "family": family.name,
        "status": "durable_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "autotune_profile_sha256": manifest["autotune_profile_sha256"],
        "precision_role_plan_sha256": manifest.get(
            "precision_role_plan_sha256"
        ),
        "global_token_cursor": int(manifest["global_token_cursor"]),
        "optimizer_step": int(manifest["optimizer_step"]),
        "artifact_count": len(artifacts),
        "artifacts_total_bytes": sum(int(row["bytes"]) for row in artifacts),
    }


def _validate_posttraining_distributed_checkpoint(
    checkpoint: Path,
    *,
    checkpoint_root: Path,
    family: FamilyTopology,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    checkpoint_root = checkpoint_root.expanduser().resolve()
    try:
        checkpoint.relative_to(checkpoint_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{family.name} post-training checkpoint escapes its root"
        ) from exc
    if (
        not re.fullmatch(r"tokens-\d{13}", checkpoint.name)
        or checkpoint.is_symlink()
        or not checkpoint.is_dir()
    ):
        raise RuntimeError(f"{family.name} post-training checkpoint path is unsafe")
    manifest_path = checkpoint / "MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(
            f"{family.name} post-training checkpoint manifest is missing"
        )
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "metis.distributed-checkpoint/v1"
        or manifest.get("family") != family.name
        or manifest.get("checkpoint_sha256")
        != json_sha256(manifest, omit=("checkpoint_sha256",))
        or int(manifest.get("world_size", 0)) != family.world_size
        or int(manifest.get("expert_parallel_size", 0))
        != family.expert_parallel_size
        or int(manifest.get("expert_replica_count", 0))
        != family.expert_replicas
        or not _SHA256.fullmatch(
            str(manifest.get("precision_role_plan_sha256", ""))
        )
    ):
        raise RuntimeError(
            f"{family.name} post-training checkpoint lineage is invalid"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError(
            f"{family.name} post-training checkpoint has no artifacts"
        )
    seen: set[str] = set()
    for raw_record in artifacts:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("Post-training checkpoint artifact record is invalid")
        record = dict(raw_record)
        relative = str(record.get("path", ""))
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen
            or not _SHA256.fullmatch(str(record.get("sha256", "")))
        ):
            raise RuntimeError("Post-training checkpoint artifact record is unsafe")
        seen.add(relative)
        artifact = (checkpoint / relative).resolve()
        try:
            artifact.relative_to(checkpoint)
        except ValueError as exc:
            raise RuntimeError(
                "Post-training checkpoint artifact escapes its checkpoint"
            ) from exc
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != int(record.get("bytes", -1))
            or file_sha256(artifact) != record["sha256"]
        ):
            raise RuntimeError(
                f"Post-training checkpoint artifact failed validation: {relative}"
            )
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "phase": str(manifest.get("phase", "")),
        "global_token_cursor": int(manifest.get("global_token_cursor", -1)),
        "optimizer_step": int(manifest.get("optimizer_step", -1)),
        "artifact_count": len(artifacts),
        "artifacts_total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "manifest": manifest,
    }


def _validate_posttraining_checkpoint_receipt(
    path: Path,
    *,
    family: FamilyTopology,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"{family.name} post-training checkpoint receipt is missing"
        )
    receipt = read_json(path)
    raw_checkpoint_manifest = receipt.get("checkpoint_manifest")
    receipt_checkpoint_manifest = (
        Path(raw_checkpoint_manifest).expanduser().resolve()
        if isinstance(raw_checkpoint_manifest, str)
        and Path(raw_checkpoint_manifest).is_absolute()
        else None
    )
    expected_checkpoint_manifest = (
        Path(str(checkpoint["checkpoint"])).expanduser().resolve()
        / "MANIFEST.json"
    )
    if (
        receipt.get("schema") != "metis.checkpoint-receipt/v1"
        or receipt.get("family") != family.name
        or receipt.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
        or receipt_checkpoint_manifest != expected_checkpoint_manifest
        or not expected_checkpoint_manifest.is_file()
        or expected_checkpoint_manifest.is_symlink()
        or receipt.get("precision_role_plan_sha256")
        != checkpoint["manifest"].get("precision_role_plan_sha256")
        or receipt.get("receipt_sha256")
        != json_sha256(receipt, omit=("receipt_sha256",))
    ):
        raise RuntimeError(
            f"{family.name} post-training checkpoint receipt is invalid"
        )
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def validate_posttraining_state_for_requeue(
    output: str | Path,
    *,
    family: FamilyTopology,
) -> dict[str, Any]:
    """Deep-validate the active post-training resume point, when present."""

    post_root = (
        Path(output).expanduser().resolve() / "posttraining" / family.name
    ).resolve()
    state_path = post_root / "STATE.json"
    if not state_path.exists():
        return {
            "family": family.name,
            "status": "not_started",
            "state": None,
            "active": None,
            "policy_checkpoint": None,
            "checkpoint_receipt": None,
        }
    if not state_path.is_file() or state_path.is_symlink():
        raise RuntimeError(f"{family.name} post-training STATE.json is unsafe")
    state = read_json(state_path)
    if (
        state.get("schema") != _POSTTRAINING_STATE_SCHEMA
        or state.get("family") != family.name
        or state.get("state_sha256")
        != json_sha256(state, omit=("state_sha256",))
        or not _SHA256.fullmatch(str(state.get("base_checkpoint_sha256", "")))
        or not _SHA256.fullmatch(str(state.get("policy_checkpoint_sha256", "")))
        or not isinstance(state.get("completed"), list)
    ):
        raise RuntimeError(f"{family.name} post-training STATE.json is invalid")
    expected_stage_prefix: list[str] = []
    completed_receipts: list[dict[str, Any]] = []
    active = state.get("active")
    state_checkpoint_contract = state.get("policy_checkpoint_contract")
    if state["completed"] and (
        not isinstance(state_checkpoint_contract, Mapping)
        or not _SHA256.fullmatch(
            str(
                state_checkpoint_contract.get(
                    "precision_role_plan_sha256", ""
                )
            )
        )
    ):
        raise RuntimeError(
            "Completed post-training state has no precision role plan contract"
        )
    for raw_record in state["completed"]:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("Post-training completed-stage record is invalid")
        record = dict(raw_record)
        stage = str(record.get("stage_id", ""))
        expected_stage_prefix.append(stage)
        receipt_path = Path(str(record.get("output_receipt", ""))).expanduser().resolve()
        try:
            receipt_path.relative_to(post_root)
        except ValueError as exc:
            raise RuntimeError(
                "Post-training completed-stage receipt escapes output"
            ) from exc
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise RuntimeError("Post-training completed-stage receipt is missing")
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema") != "metis.inprocess-stage-receipt/v1"
            or receipt.get("family") != family.name
            or receipt.get("stage") != stage
            or receipt.get("receipt_sha256")
            != json_sha256(receipt, omit=("receipt_sha256",))
            or receipt.get("receipt_sha256")
            != record.get("output_receipt_sha256")
            or json_sha256(receipt.get("metrics", {}))
            != record.get("metrics_sha256")
            or (
                isinstance(state_checkpoint_contract, Mapping)
                and receipt.get("precision_role_plan_sha256")
                != state_checkpoint_contract.get(
                    "precision_role_plan_sha256"
                )
            )
        ):
            raise RuntimeError("Post-training completed-stage receipt is invalid")
        completed_receipts.append(
            {
                "stage": stage,
                "path": str(receipt_path),
                "file_sha256": file_sha256(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
    # The state validator in the trainer enforces the exact stage order.  The
    # supervisor additionally rejects duplicates or unsafe names without
    # importing the full GPU stage backend.
    if (
        tuple(expected_stage_prefix)
        != _POSTTRAINING_STAGE_IDS[: len(expected_stage_prefix)]
    ):
        raise RuntimeError("Post-training completed-stage order is invalid")
    if active is not None:
        # Assigned below after structural validation; this pre-check binds the
        # next stage before any checkpoint payload is trusted.
        if (
            not isinstance(active, Mapping)
            or len(expected_stage_prefix) >= len(_POSTTRAINING_STAGE_IDS)
            or active.get("stage_id")
            != _POSTTRAINING_STAGE_IDS[len(expected_stage_prefix)]
        ):
            raise RuntimeError("Post-training active-stage order is invalid")

    checkpoint_root = (post_root / "checkpoints").resolve()
    partial = (
        sorted(path.name for path in checkpoint_root.glob(".incomplete-*"))
        if checkpoint_root.is_dir()
        else []
    )
    if partial:
        raise RuntimeError(
            f"{family.name} has incomplete post-training checkpoints: {partial}"
        )
    active_summary: dict[str, Any] | None = None
    policy_summary: dict[str, Any] | None = None
    checkpoint_receipt: dict[str, Any] | None = None
    if active is not None:
        if not isinstance(active, Mapping):
            raise RuntimeError("Post-training active-stage state is invalid")
        active = dict(active)
        kind = str(active.get("kind", "policy"))
        if kind == "policy":
            policy_summary = _validate_posttraining_distributed_checkpoint(
                Path(str(active.get("checkpoint_path", ""))),
                checkpoint_root=checkpoint_root,
                family=family,
            )
            manifest = policy_summary.pop("manifest")
            extra = manifest.get("extra_state")
            contract = active.get("checkpoint_contract")
            if (
                active.get("checkpoint_sha256")
                != policy_summary["checkpoint_sha256"]
                or manifest.get("phase") != active.get("stage_id")
                or not isinstance(extra, Mapping)
                or not isinstance(contract, Mapping)
                or extra.get("posttraining_stage") != active.get("stage_id")
                or extra.get("parent_checkpoint_sha256")
                != active.get("parent_checkpoint_sha256")
                or extra.get("bundle_sha256") != active.get("bundle_sha256")
                or extra.get("stage_epoch") != active.get("epoch")
                or extra.get("stage_next_global_batch")
                != active.get("next_global_batch")
                or extra.get("stage_optimizer_step")
                != active.get("optimizer_step")
                or extra.get("campaign_token_cursor")
                != active.get("campaign_token_cursor")
                or extra.get("runtime_batch") != active.get("runtime_batch")
            ):
                raise RuntimeError(
                    f"{family.name} active post-training checkpoint/state diverged"
                )
            for contract_field, manifest_field in (
                ("release_sha256", "release_sha256"),
                ("shard_manifest_sha256", "shard_manifest_sha256"),
                ("family_manifest_sha256", "family_manifest_sha256"),
                ("runtime_manifest_sha256", "runtime_manifest_sha256"),
                ("autotune_profile_sha256", "autotune_profile_sha256"),
                ("precision_role_plan_sha256", "precision_role_plan_sha256"),
            ):
                if contract.get(contract_field) != manifest.get(manifest_field):
                    raise RuntimeError(
                        f"{family.name} active checkpoint contract changed: "
                        f"{contract_field}"
                    )
            active_summary = {
                "kind": kind,
                "stage": active.get("stage_id"),
                "checkpoint": policy_summary,
            }
        else:
            raise RuntimeError("Post-training active-stage kind is unsupported")
    elif state.get("policy_checkpoint_contract") is not None:
        policy_summary = _validate_posttraining_distributed_checkpoint(
            Path(str(state.get("policy_checkpoint_path", ""))),
            checkpoint_root=checkpoint_root,
            family=family,
        )
        if policy_summary["checkpoint_sha256"] != state.get(
            "policy_checkpoint_sha256"
        ):
            raise RuntimeError("Post-training policy checkpoint changed")
        checkpoint_receipt = _validate_posttraining_checkpoint_receipt(
            Path(str(state.get("policy_checkpoint_receipt", ""))),
            family=family,
            checkpoint=policy_summary,
        )
        policy_summary.pop("manifest")
    status = "active" if active_summary is not None else "stage_boundary"
    return {
        "family": family.name,
        "status": status,
        "state": {
            "path": str(state_path),
            "file_sha256": file_sha256(state_path),
            "state_sha256": state["state_sha256"],
            "completed_stages": expected_stage_prefix,
            "completed_receipts": completed_receipts,
        },
        "active": active_summary,
        "policy_checkpoint": policy_summary,
        "checkpoint_receipt": checkpoint_receipt,
    }


def validate_deferred_materialization_request(
    path: str | Path,
    *,
    output: str | Path,
    family: FamilyTopology,
    posttraining_preflight: Mapping[str, Any],
    expected_job_id: str,
    expected_restart_count: int,
) -> dict[str, Any]:
    request_path = Path(path).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    request_root = (
        output_root
        / "posttraining"
        / family.name
        / "materialization"
        / "requests"
    ).resolve()
    try:
        request_path.relative_to(request_root)
    except ValueError as exc:
        raise RuntimeError("Deferred materialization request escapes output") from exc
    if not request_path.is_file() or request_path.is_symlink():
        raise RuntimeError("Deferred materialization request is missing or unsafe")
    request = read_json(request_path)
    hook = request.get("hook")
    deep = request.get("deep_verification")
    stage_bindings = request.get("stage_bindings")
    if (
        request.get("schema") != _DEFERRED_REQUEST_SCHEMA
        or request.get("request_sha256")
        != json_sha256(request, omit=("request_sha256",))
        or request.get("family") != family.name
        or not re.fullmatch(r"[a-z][a-z0-9_]*", str(request.get("stage", "")))
        or not re.fullmatch(
            r"[a-z][a-z0-9_]*", str(request.get("requirement", ""))
        )
        or not isinstance(request.get("requirement_schema"), str)
        or not _SHA256.fullmatch(
            str(request.get("parent_checkpoint_sha256", ""))
        )
        or not _SHA256.fullmatch(
            str(request.get("release_index_file_sha256", ""))
        )
        or not _SHA256.fullmatch(
            str(request.get("release_index_sha256", ""))
        )
        or not _SHA256.fullmatch(str(request.get("record_sha256", "")))
        or request.get("trainer_world_size") != family.world_size
        or request.get("slurm_job_id") != expected_job_id
        or request.get("slurm_restart_count") != expected_restart_count
        or not isinstance(hook, Mapping)
        or not isinstance(deep, Mapping)
        or not isinstance(stage_bindings, Mapping)
    ):
        raise RuntimeError("Deferred materialization request lineage is invalid")

    posttraining_root = (
        output_root / "posttraining" / family.name
    ).resolve()
    checkpoint_root = (posttraining_root / "checkpoints").resolve()

    def validate_checkpoint_binding(
        raw: Any,
        *,
        label: str,
        expected_checkpoint_sha256: str | None = None,
        expected_phase: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_receipt",
            "checkpoint_contract",
        } | ({"stage_id"} if "stage_id" in raw else set()):
            raise RuntimeError(f"{label} checkpoint binding has invalid fields")
        path = Path(str(raw.get("checkpoint_path", ""))).expanduser().resolve()
        checkpoint = _validate_posttraining_distributed_checkpoint(
            path,
            checkpoint_root=checkpoint_root,
            family=family,
        )
        manifest = checkpoint["manifest"]
        checkpoint_sha256 = str(raw.get("checkpoint_sha256", ""))
        if (
            checkpoint_sha256 != checkpoint["checkpoint_sha256"]
            or (
                expected_checkpoint_sha256 is not None
                and checkpoint_sha256 != expected_checkpoint_sha256
            )
            or (
                expected_phase is not None
                and manifest.get("phase") != expected_phase
            )
        ):
            raise RuntimeError(f"{label} checkpoint binding changed")
        receipt_path = Path(
            str(raw.get("checkpoint_receipt", ""))
        ).expanduser().resolve()
        try:
            receipt_path.relative_to(posttraining_root)
        except ValueError as exc:
            raise RuntimeError(
                f"{label} checkpoint receipt escapes post-training output"
            ) from exc
        _validate_posttraining_checkpoint_receipt(
            receipt_path,
            family=family,
            checkpoint=checkpoint,
        )
        contract = raw.get("checkpoint_contract")
        if not isinstance(contract, Mapping):
            raise RuntimeError(f"{label} checkpoint contract is missing")
        for contract_field, manifest_field in (
            ("release_sha256", "release_sha256"),
            ("shard_manifest_sha256", "shard_manifest_sha256"),
            ("family_manifest_sha256", "family_manifest_sha256"),
            ("runtime_manifest_sha256", "runtime_manifest_sha256"),
            ("autotune_profile_sha256", "autotune_profile_sha256"),
            ("precision_role_plan_sha256", "precision_role_plan_sha256"),
        ):
            if contract.get(contract_field) != manifest.get(manifest_field):
                raise RuntimeError(
                    f"{label} checkpoint contract changed: {contract_field}"
                )
        return checkpoint

    expected_binding_names = {"parent_policy_checkpoint"}
    if request.get("stage") == "opd_consolidation":
        expected_binding_names.update(
            {"specialist_checkpoints", "unified_student_checkpoint"}
        )
    if set(stage_bindings) != expected_binding_names:
        raise RuntimeError(
            "Deferred materialization stage bindings do not match the stage"
        )

    parent_binding = stage_bindings.get("parent_policy_checkpoint")
    if (
        not isinstance(parent_binding, Mapping)
        or parent_binding.get("stage_id")
        != _POSTTRAINING_PARENT_STAGE.get(str(request.get("stage", "")))
    ):
        raise RuntimeError(
            "Deferred materialization omits its parent policy checkpoint"
        )
    validate_checkpoint_binding(
        parent_binding,
        label="parent policy",
        expected_checkpoint_sha256=str(
            request["parent_checkpoint_sha256"]
        ),
    )

    if request.get("stage") == "opd_consolidation":
        specialists = stage_bindings.get("specialist_checkpoints")
        unified = stage_bindings.get("unified_student_checkpoint")
        if (
            not isinstance(specialists, Mapping)
            or set(specialists) != set(_SPECIALIST_STAGE_IDS)
            or not isinstance(unified, Mapping)
            or unified.get("stage_id") != "hybrid_mode_gspo"
        ):
            raise RuntimeError(
                "OPD materialization omits the unified student or a specialist"
            )
        validate_checkpoint_binding(
            unified,
            label="OPD unified student",
            expected_checkpoint_sha256=str(
                request["parent_checkpoint_sha256"]
            ),
            expected_phase="hybrid_mode_gspo",
        )
        if (
            unified.get("checkpoint_path")
            != parent_binding.get("checkpoint_path")
            or unified.get("checkpoint_receipt")
            != parent_binding.get("checkpoint_receipt")
            or unified.get("checkpoint_contract")
            != parent_binding.get("checkpoint_contract")
        ):
            raise RuntimeError(
                "OPD unified-student binding differs from its live parent"
            )
        for specialist_id in _SPECIALIST_STAGE_IDS:
            validate_checkpoint_binding(
                specialists[specialist_id],
                label=f"OPD {specialist_id}",
                expected_phase=specialist_id,
            )

    observed_deep = posttraining_preflight.get("deep_verification")
    if not isinstance(observed_deep, Mapping) or dict(deep) != {
        "path": str(observed_deep.get("path", "")),
        "file_sha256": str(observed_deep.get("file_sha256", "")),
        "receipt_sha256": str(observed_deep.get("receipt_sha256", "")),
    }:
        raise RuntimeError(
            "Deferred materialization request changed deep verification"
        )
    deep_path = Path(str(deep["path"])).expanduser().resolve()
    if (
        not deep_path.is_file()
        or deep_path.is_symlink()
        or file_sha256(deep_path) != deep["file_sha256"]
    ):
        raise RuntimeError("Deferred materialization deep receipt changed")
    deep_receipt = read_json(deep_path)
    if (
        deep_receipt.get("schema")
        != "metis.posttraining-release-deep-verification/v1"
        or deep_receipt.get("receipt_sha256")
        != json_sha256(deep_receipt, omit=("receipt_sha256",))
        or deep_receipt.get("receipt_sha256") != deep["receipt_sha256"]
        or deep_receipt.get("complete") is not True
    ):
        raise RuntimeError("Deferred materialization deep receipt is invalid")

    index_path = Path(str(request.get("release_index_path", ""))).expanduser().resolve()
    family_row = posttraining_preflight.get("family_indexes", {}).get(
        family.name
    )
    if (
        not isinstance(family_row, Mapping)
        or index_path != Path(str(family_row.get("path", ""))).expanduser().resolve()
        or not index_path.is_file()
        or index_path.is_symlink()
        or file_sha256(index_path) != request["release_index_file_sha256"]
        or file_sha256(index_path) != family_row.get("file_sha256")
    ):
        raise RuntimeError("Deferred materialization family index changed")
    index = read_json(index_path)
    if (
        index.get("schema") != "metis.posttraining-release-index/v1"
        or index.get("family") != family.name
        or index.get("index_sha256")
        != json_sha256(index, omit=("index_sha256",))
        or index.get("index_sha256") != request["release_index_sha256"]
        or index.get("index_sha256") != family_row.get("index_sha256")
    ):
        raise RuntimeError("Deferred materialization index lineage is invalid")
    try:
        record = index["requirements"][request["stage"]][request["requirement"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Deferred requirement disappeared from its index") from exc
    if (
        not isinstance(record, Mapping)
        or record.get("state") != "deferred"
        or record.get("schema") != request["requirement_schema"]
        or json_sha256(record) != request["record_sha256"]
    ):
        raise RuntimeError("Deferred requirement record changed")
    indexed_hook = record.get("generation_hook")
    if not isinstance(indexed_hook, Mapping):
        raise RuntimeError("Deferred requirement hook is invalid")
    index_root = index_path.parent.resolve()

    def indexed_path(raw: Any, *, label: str) -> Path:
        if (
            not isinstance(raw, str)
            or not raw
            or Path(raw).is_absolute()
            or ".." in Path(raw).parts
        ):
            raise RuntimeError(f"{label} is not a safe relative path")
        resolved = (index_root / raw).resolve()
        try:
            resolved.relative_to(index_root)
        except ValueError as exc:
            raise RuntimeError(f"{label} escapes the family release") from exc
        if resolved.is_symlink():
            raise RuntimeError(f"{label} may not be a symlink")
        return resolved

    executable = indexed_path(indexed_hook.get("executable"), label="hook executable")
    output_manifest = indexed_path(record.get("manifest"), label="generated manifest")
    reducer_receipt = indexed_path(
        indexed_hook.get("receipt"), label="generation reducer receipt"
    )
    rank_receipts = indexed_path(
        indexed_hook.get("rank_receipts"), label="generation rank receipts"
    )
    execution = indexed_hook.get("execution")
    protocol = (
        execution.get("protocol") if isinstance(execution, Mapping) else None
    )
    if protocol == "distributed_family_v1":
        valid_execution = (
            set(execution) == {"protocol"}
            and hook.get("world_size") == family.world_size
        )
    elif protocol == "rank0_only_v1":
        valid_execution = (
            set(execution)
            == {"protocol", "nodes", "tasks", "gpus_per_task"}
            and int(execution.get("nodes", 0)) == 1
            and int(execution.get("tasks", 0)) == 1
            and int(execution.get("gpus_per_task", -1)) in {0, 1}
            and hook.get("world_size") == 1
        )
    else:
        valid_execution = False
    expected_hook = {
        "executable": str(executable),
        "executable_sha256": str(indexed_hook.get("executable_sha256", "")),
        "args": list(indexed_hook.get("args", [])),
        "timeout_seconds": int(indexed_hook.get("timeout_seconds", 0)),
        "output_manifest": str(output_manifest),
        "reducer_receipt": str(reducer_receipt),
        "rank_receipts": str(rank_receipts),
        "execution": dict(execution) if isinstance(execution, Mapping) else {},
        "world_size": family.world_size if protocol == "distributed_family_v1" else 1,
    }
    if (
        dict(hook) != expected_hook
        or not valid_execution
        or not executable.is_file()
        or executable.is_symlink()
        or file_sha256(executable) != hook["executable_sha256"]
        or not _SHA256.fullmatch(str(hook.get("executable_sha256", "")))
        or not isinstance(hook.get("args"), list)
        or not all(isinstance(item, str) for item in hook["args"])
        or not 1 <= int(hook.get("timeout_seconds", 0)) <= 7 * 24 * 60 * 60
        or (rank_receipts.exists() and not rank_receipts.is_dir())
        or len({executable, output_manifest, reducer_receipt, rank_receipts}) != 4
    ):
        raise RuntimeError("Deferred materialization hook/request changed")
    identity = {
        "family": family.name,
        "stage": request["stage"],
        "requirement": request["requirement"],
        "parent_checkpoint_sha256": request["parent_checkpoint_sha256"],
        "release_index_sha256": request["release_index_sha256"],
        "record_sha256": request["record_sha256"],
        "stage_bindings": dict(stage_bindings),
        "hook": expected_hook,
    }
    return {
        **request,
        "_request_path": str(request_path),
        "_request_file_sha256": file_sha256(request_path),
        "_identity_sha256": json_sha256(identity),
    }


def validate_deferred_materialization_result(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    hook = request["hook"]
    output_manifest = Path(str(hook["output_manifest"])).resolve()
    reducer_path = Path(str(hook["reducer_receipt"])).resolve()
    rank_root = Path(str(hook["rank_receipts"])).resolve()
    if (
        not output_manifest.is_file()
        or output_manifest.is_symlink()
        or not reducer_path.is_file()
        or reducer_path.is_symlink()
        or not rank_root.is_dir()
        or rank_root.is_symlink()
    ):
        raise RuntimeError("Generation hook left incomplete or unsafe outputs")
    output = read_json(output_manifest)
    files = output.get("files") if isinstance(output, Mapping) else None
    if (
        output.get("envelope_schema") != "metis.sealed-artifact/v1"
        or output.get("schema") != request["requirement_schema"]
        or output.get("complete") is not True
        or output.get("manifest_sha256")
        != json_sha256(output, omit=("manifest_sha256",))
        or not isinstance(files, list)
        or not files
    ):
        raise RuntimeError("Generated requirement manifest is not deeply sealed")
    manifest_root = output_manifest.parent.resolve()
    seen_files: set[str] = set()
    for raw_record in files:
        if not isinstance(raw_record, Mapping):
            raise RuntimeError("Generated payload record is invalid")
        relative = str(raw_record.get("path", ""))
        payload_path = (manifest_root / relative).resolve()
        try:
            payload_path.relative_to(manifest_root)
        except ValueError as exc:
            raise RuntimeError("Generated payload escapes its manifest") from exc
        if (
            not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_files
            or not payload_path.is_file()
            or payload_path.is_symlink()
            or payload_path.stat().st_size != int(raw_record.get("bytes", -1))
            or not _SHA256.fullmatch(str(raw_record.get("sha256", "")))
            or file_sha256(payload_path) != raw_record["sha256"]
        ):
            raise RuntimeError(f"Generated payload failed validation: {relative}")
        seen_files.add(relative)
    reducer = read_json(reducer_path)
    rank_rows = reducer.get("rank_receipts")
    world_size = int(hook["world_size"])
    if (
        reducer.get("schema") != "metis.generation-hook-receipt/v2"
        or reducer.get("request_sha256") != request["request_sha256"]
        or reducer.get("family") != request["family"]
        or reducer.get("stage") != request["stage"]
        or reducer.get("requirement") != request["requirement"]
        or reducer.get("parent_checkpoint_sha256")
        != request["parent_checkpoint_sha256"]
        or reducer.get("stage_bindings") != request["stage_bindings"]
        or reducer.get("release_index_file_sha256")
        != request["release_index_file_sha256"]
        or reducer.get("release_index_sha256")
        != request["release_index_sha256"]
        or reducer.get("record_sha256") != request["record_sha256"]
        or reducer.get("deep_verification_file_sha256")
        != request["deep_verification"]["file_sha256"]
        or reducer.get("deep_verification_receipt_sha256")
        != request["deep_verification"]["receipt_sha256"]
        or reducer.get("executable_sha256") != hook["executable_sha256"]
        or reducer.get("execution_protocol")
        != hook["execution"]["protocol"]
        or reducer.get("world_size") != world_size
        or reducer.get("output_manifest_sha256")
        != file_sha256(output_manifest)
        or reducer.get("output_manifest_self_sha256")
        != output["manifest_sha256"]
        or reducer.get("success") is not True
        or reducer.get("receipt_sha256")
        != json_sha256(reducer, omit=("receipt_sha256",))
        or not isinstance(rank_rows, list)
        or len(rank_rows) != world_size
    ):
        raise RuntimeError("Generation reducer receipt is invalid")
    seen_ranks: set[int] = set()
    rank_summaries: list[dict[str, Any]] = []
    for raw_row in rank_rows:
        if not isinstance(raw_row, Mapping):
            raise RuntimeError("Generation reducer rank record is invalid")
        rank = int(raw_row.get("rank", -1))
        rank_path = (rank_root / f"rank-{rank:05d}.json").resolve()
        if (
            rank in seen_ranks
            or not 0 <= rank < world_size
            or not rank_path.is_file()
            or rank_path.is_symlink()
            or file_sha256(rank_path) != raw_row.get("file_sha256")
        ):
            raise RuntimeError("Generation rank receipt coverage is invalid")
        seen_ranks.add(rank)
        rank_receipt = read_json(rank_path)
        if (
            rank_receipt.get("schema")
            != "metis.generation-hook-rank-receipt/v1"
            or rank_receipt.get("request_sha256") != request["request_sha256"]
            or rank_receipt.get("family") != request["family"]
            or rank_receipt.get("stage") != request["stage"]
            or rank_receipt.get("requirement") != request["requirement"]
            or rank_receipt.get("parent_checkpoint_sha256")
            != request["parent_checkpoint_sha256"]
            or rank_receipt.get("stage_bindings")
            != request["stage_bindings"]
            or rank_receipt.get("rank") != rank
            or rank_receipt.get("world_size") != world_size
            or rank_receipt.get("success") is not True
            or rank_receipt.get("receipt_sha256")
            != json_sha256(rank_receipt, omit=("receipt_sha256",))
            or rank_receipt.get("receipt_sha256")
            != raw_row.get("receipt_sha256")
        ):
            raise RuntimeError("Generation rank receipt lineage is invalid")
        rank_summaries.append(
            {
                "rank": rank,
                "path": str(rank_path),
                "file_sha256": file_sha256(rank_path),
                "receipt_sha256": rank_receipt["receipt_sha256"],
            }
        )
    return {
        "output_manifest": str(output_manifest),
        "output_manifest_file_sha256": file_sha256(output_manifest),
        "output_manifest_self_sha256": output["manifest_sha256"],
        "reducer_receipt": str(reducer_path),
        "reducer_receipt_file_sha256": file_sha256(reducer_path),
        "reducer_receipt_sha256": reducer["receipt_sha256"],
        "rank_receipts": sorted(rank_summaries, key=lambda row: row["rank"]),
    }


def validate_posttraining_batch_migration_for_requeue(
    summary: Mapping[str, Any],
    *,
    campaign_root: Path,
    output_root: Path,
    family: FamilyTopology,
) -> dict[str, Any]:
    raw_path = summary.get("path")
    if not isinstance(raw_path, str):
        raise RuntimeError("Stage batch migration summary omits its receipt path")
    path = Path(raw_path).expanduser().resolve()
    root = (
        campaign_root.resolve()
        / "posttraining-batch-migrations"
        / family.name
    ).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Stage batch migration receipt escapes campaign state") from exc
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Stage batch migration receipt is missing or unsafe")
    receipt = read_json(path)
    revisions = receipt.get("revisions")
    sealed = receipt.get("sealed_training")
    if (
        receipt.get("schema") != _STAGE_BATCH_MIGRATION_SCHEMA
        or receipt.get("receipt_sha256")
        != json_sha256(receipt, omit=("receipt_sha256",))
        or receipt.get("receipt_sha256") != summary.get("receipt_sha256")
        or receipt.get("family") != family.name
        or receipt.get("stage") != summary.get("stage")
        or not _SHA256.fullmatch(
            str(receipt.get("parent_checkpoint_sha256", ""))
        )
        or not _SHA256.fullmatch(
            str(receipt.get("bundle_manifest_sha256", ""))
        )
        or not _SHA256.fullmatch(
            str(receipt.get("precision_role_plan_sha256", ""))
        )
        or summary.get("precision_role_plan_sha256")
        != receipt.get("precision_role_plan_sha256")
        or not isinstance(sealed, Mapping)
        or set(sealed) != {"micro_batch_size", "gradient_accumulation"}
        or not isinstance(revisions, list)
        or not revisions
    ):
        raise RuntimeError("Stage batch migration receipt is invalid")
    effective = int(sealed["micro_batch_size"]) * int(
        sealed["gradient_accumulation"]
    )
    prior = dict(sealed)
    output_oom_root = (
        output_root.resolve()
        / "posttraining"
        / family.name
        / "oom"
    ).resolve()
    for revision in revisions:
        if (
            not isinstance(revision, Mapping)
            or revision.get("revision_sha256")
            != json_sha256(revision, omit=("revision_sha256",))
            or revision.get("reason") != "measured_stage_oom"
            or revision.get("old") != prior
            or not isinstance(revision.get("new"), Mapping)
        ):
            raise RuntimeError("Stage batch migration revision chain is invalid")
        new = dict(revision["new"])
        if (
            int(new.get("micro_batch_size", -1))
            >= int(prior.get("micro_batch_size", -1))
            or int(new.get("gradient_accumulation", -1))
            <= int(prior.get("gradient_accumulation", -1))
            or int(new.get("micro_batch_size", -1))
            * int(new.get("gradient_accumulation", -1))
            != effective
        ):
            raise RuntimeError("Stage batch migration changed effective batch")
        request_path = Path(
            str(revision.get("oom_request_path", ""))
        ).expanduser().resolve()
        try:
            request_path.relative_to(output_oom_root)
        except ValueError as exc:
            raise RuntimeError("Stage OOM request escapes training output") from exc
        if (
            not request_path.is_file()
            or request_path.is_symlink()
            or file_sha256(request_path)
            != revision.get("oom_request_file_sha256")
        ):
            raise RuntimeError("Stage OOM request bytes changed before requeue")
        request = read_json(request_path)
        if (
            request.get("schema") != _STAGE_OOM_REQUEST_SCHEMA
            or request.get("request_sha256")
            != json_sha256(request, omit=("request_sha256",))
            or request.get("request_sha256")
            != revision.get("oom_request_sha256")
            or request.get("family") != family.name
            or request.get("stage") != receipt["stage"]
            or request.get("parent_checkpoint_sha256")
            != receipt["parent_checkpoint_sha256"]
            or request.get("bundle_manifest_sha256")
            != receipt["bundle_manifest_sha256"]
            or request.get("precision_role_plan_sha256")
            != receipt["precision_role_plan_sha256"]
            or request.get("current") != dict(prior)
            or request.get("proposed") != new
            or request.get("prior_batch_migration_sha256")
            != revision.get("prior_batch_migration_sha256")
        ):
            raise RuntimeError("Stage OOM request lineage changed before requeue")
        prior = new
    if (
        receipt.get("effective_local_batch_records") != effective
        or summary.get("effective_local_batch_records") != effective
        or summary.get("old") != revisions[-1]["old"]
        or summary.get("new") != revisions[-1]["new"]
    ):
        raise RuntimeError("Stage batch migration summary is stale")
    return receipt


class FamilySupervisor:
    def __init__(self, config: PortageConfig, campaign_root: Path) -> None:
        self.config = config
        self.campaign_root = campaign_root
        self.processes: list[subprocess.Popen[str]] = []
        self.process_lock = threading.Lock()
        self.signal_requested = threading.Event()
        self.compute_inventory = read_json(campaign_root / "gates" / "compute_inventory.json")
        self.compute_runtime = validate_compute_runtime(
            campaign_root / "gates" / "runtime_compute.json",
            config=config,
        )
        if (
            self.compute_inventory.get("facts", {}).get(
                "runtime_compute_sha256"
            )
            != self.compute_runtime["runtime_compute_sha256"]
        ):
            raise RuntimeError(
                "Compute inventory is not bound to the validated runtime-kernel report"
            )
        single_apu = read_json(campaign_root / "gates" / "single_apu.json")
        if (
            single_apu.get("schema") != "metis.portage-probe/v1"
            or single_apu.get("stage") != "single_apu"
            or single_apu.get("ok") is not True
            or single_apu.get("report_sha256")
            != json_sha256(single_apu, omit=("report_sha256",))
        ):
            raise RuntimeError("Single-APU exact-role precision report is invalid")
        precision_families = single_apu.get("family_precision")
        if not isinstance(precision_families, Mapping):
            raise RuntimeError("Single-APU report has no exact-role precision plans")
        self.precision_role_plans: dict[str, dict[str, Any]] = {}
        self.precision_role_plan_paths: dict[str, Path] = {}
        for family in config.families:
            family_row = precision_families.get(family.name)
            raw_plan = (
                family_row.get("precision_role_plan")
                if isinstance(family_row, Mapping)
                else None
            )
            if not isinstance(raw_plan, Mapping):
                raise RuntimeError(
                    f"Compute runtime omits the {family.name} precision role plan"
                )
            model_config = load_family_config(family.manifest)
            raw_inventory = (
                family_row.get("precision_role_inventory")
                if isinstance(family_row, Mapping)
                else None
            )
            if not isinstance(raw_inventory, Mapping):
                raise RuntimeError(
                    f"Single-APU report omits the {family.name} role inventory"
                )
            validated_inventory = validate_precision_role_inventory(
                raw_inventory,
                config=model_config,
            )
            validated_plan = validate_precision_role_plan(
                raw_plan,
                config=model_config,
            )
            if (
                validated_plan["inventory_sha256"]
                != validated_inventory["inventory_sha256"]
            ):
                raise RuntimeError(
                    f"{family.name} precision plan/inventory binding changed"
                )
            path = (
                campaign_root
                / "gates"
                / f"precision-role-plan-{family.name}.json"
            )
            atomic_write_json(path, validated_plan)
            self.precision_role_plans[family.name] = validated_plan
            self.precision_role_plan_paths[family.name] = path
        self.release_marker = validate_release_marker(
            campaign_root / "gates" / "release_verification.json",
            expected_release_root=config.release_root,
            expected_contract_path=config.training_contract,
            expected_posttraining_preflight_path=(
                campaign_root / "posttraining-release-preflight.json"
            ),
        )
        self.posttraining_release = read_json(
            campaign_root / "posttraining-release-preflight.json"
        )
        campaign = read_json(campaign_root / "campaign.json")
        self.git_commit = str(campaign["git_commit"])
        self.fingerprint = inventory_fingerprint(
            compute_inventory=self.compute_inventory,
            git_commit=self.git_commit,
            config=config,
        )
        self.fingerprint = json_sha256(
            {
                "base_inventory_fingerprint": self.fingerprint,
                "precision_role_plans": {
                    family: plan["plan_sha256"]
                    for family, plan in sorted(
                        self.precision_role_plans.items()
                    )
                },
            }
        )
        self.hbm_bytes = int(
            self.compute_inventory["facts"]["torch"]["total_memory"]
        )
        self.allocated_nodes = self._allocated_nodes()

    def _allocated_nodes(self) -> list[str]:
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if not nodelist:
            # Unit-level construction outside Slurm may still inspect methods;
            # the production run() gate rejects a missing 128-node allocation.
            return []
        completed = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        nodes = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0 or len(nodes) != 128 or len(set(nodes)) != 128:
            raise RuntimeError(
                "Unable to resolve exactly 128 unique nodes from the family allocation"
            )
        return nodes

    def _nodes_for(self, family: FamilyTopology, placement: str) -> list[str]:
        if len(self.allocated_nodes) != 128:
            return []
        if placement == "contiguous":
            return self.allocated_nodes[
                family.relative_node : family.relative_node + family.nodes
            ]
        if placement == "interleaved":
            praxis = [
                node for index, node in enumerate(self.allocated_nodes) if index % 4 == 0
            ]
            logos = [
                node for index, node in enumerate(self.allocated_nodes) if index % 4 != 0
            ]
            return praxis if family.name == "praxis" else logos
        raise RuntimeError(f"Unsupported topology placement: {placement}")

    def _family_srun_prefix(
        self,
        family: FamilyTopology,
        *,
        placement: str = "contiguous",
    ) -> list[str]:
        # Leave one core for the node-local ROCm/CXI telemetry sampler and
        # reserve the remaining cores for filesystem/RCCL progress.
        cpus_per_task = max(
            1,
            int(self.config.raw["site"]["cpu_cores_per_node"])
            // self.config.accelerators_per_node
            - 1,
        )
        argv = [
            "srun",
            "--nodes",
            str(family.nodes),
            "--ntasks",
            str(family.world_size),
            "--ntasks-per-node",
            str(self.config.accelerators_per_node),
            "--gpus-per-task",
            "1",
            "--cpus-per-task",
            str(cpus_per_task),
            "--exact",
            "--kill-on-bad-exit=1",
            "--distribution=block:block",
            "--gpu-bind=closest",
            "--cpu-bind=cores",
            "--mem-bind=local",
        ]
        nodes = self._nodes_for(family, placement)
        if nodes:
            if len(nodes) != family.nodes or len(set(nodes)) != family.nodes:
                raise RuntimeError(
                    f"{placement} placement does not map {family.nodes} unique {family.name} nodes"
                )
            argv.extend(("--nodelist", ",".join(nodes)))
        else:
            argv.extend(("--relative", str(family.relative_node)))
        return argv

    def _base_trainer_args(
        self,
        family: FamilyTopology,
        *,
        output: Path,
        profile: Path | None,
        stage: str,
    ) -> list[str]:
        command = [
            *self.config.trainer_argv,
            "--manifest",
            str(family.manifest),
            "--data-release",
            str(self.config.release_root),
            "--output",
            str(output),
            "--resume",
            str(self.config.raw["training"]["resume"]),
            "--family",
            family.name,
            "--stage",
            stage,
            "--runtime-manifest",
            str(self.config.runtime_policy),
            "--posttraining-manifest",
            str(self.config.posttraining_contract),
        ]
        if profile is not None:
            command.extend(("--autotune-profile", str(profile)))
        return command

    def _environment(self, family: FamilyTopology) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "METIS_RELEASE_VERIFICATION_MARKER": str(
                    self.campaign_root / "gates" / "release_verification.json"
                ),
                "METIS_PORTAGE_CAMPAIGN_ROOT": str(self.campaign_root),
                "METIS_FAMILY": family.name,
                "METIS_EXPERT_PARALLEL_SIZE": str(family.expert_parallel_size),
                "METIS_EXPERT_REPLICAS": str(family.expert_replicas),
                "METIS_EXPECTED_WORLD_SIZE": str(family.world_size),
                "METIS_POSTTRAINING_BATCH_MIGRATION_ROOT": str(
                    self.campaign_root / "posttraining-batch-migrations"
                ),
                "TORCH_BLAS_PREFER_HIPBLASLT": "1",
                "AMD_COMGR_CACHE": "1",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(self.config.repository / "src"),
                        environment.get("PYTHONPATH", ""),
                    )
                ).rstrip(os.pathsep),
            }
        )
        environment.update(
            environment_for_family(
                self.posttraining_release,
                family.name,
                config=self.config,
            )
        )
        deep_verification = self.posttraining_release.get("deep_verification")
        if not isinstance(deep_verification, Mapping):
            raise RuntimeError(
                "Validated post-training preflight omits deep verification"
            )
        environment.update(
            {
                "METIS_POSTTRAINING_DEEP_VERIFICATION": str(
                    deep_verification["path"]
                ),
                "METIS_POSTTRAINING_DEEP_VERIFICATION_FILE_SHA256": str(
                    deep_verification["file_sha256"]
                ),
                "METIS_POSTTRAINING_DEEP_VERIFICATION_RECEIPT_SHA256": str(
                    deep_verification["receipt_sha256"]
                ),
            }
        )
        migration = (
            self.campaign_root
            / "autotune"
            / family.name
            / "profile-migration.json"
        )
        if migration.is_file():
            environment["METIS_AUTOTUNE_PROFILE_MIGRATION"] = str(migration)
        return environment

    def _output_for(self, family: FamilyTopology) -> Path:
        return (
            self.campaign_root
            / self.config.raw["training"]["output_subdirectory"]
            / family.name
        )

    def _training_log_path(self, family_name: str) -> Path:
        """Return the log owned by this exact Slurm execution attempt.

        Requeued allocations reuse the campaign directory.  Keeping attempts in
        separate files prevents an OOM or transient signature from an earlier
        execution from influencing classification of a later, unrelated
        trainer failure.
        """

        raw_restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
        try:
            restart_count = int(raw_restart_count)
        except ValueError as exc:
            raise RuntimeError("SLURM_RESTART_COUNT must be an integer") from exc
        if restart_count < 0:
            raise RuntimeError("SLURM_RESTART_COUNT may not be negative")
        if family_name not in {"praxis", "logos"}:
            raise RuntimeError(f"Unsafe family name for training log: {family_name!r}")
        return (
            self.campaign_root
            / "logs"
            / "training-attempts"
            / f"restart-{restart_count:03d}"
            / f"{family_name}.log"
        )

    def checkpoint_states(
        self,
        *,
        require_checkpoint: bool,
    ) -> dict[str, dict[str, Any]]:
        return {
            family.name: validate_checkpoint_for_requeue(
                self._output_for(family),
                family=family,
                require_checkpoint=require_checkpoint,
            )
            for family in self.config.families
        }

    def posttraining_states(self) -> dict[str, dict[str, Any]]:
        return {
            family.name: validate_posttraining_state_for_requeue(
                self._output_for(family),
                family=family,
            )
            for family in self.config.families
        }

    def write_requeue_marker(
        self,
        *,
        returncodes: Mapping[str, int],
        require_checkpoint: bool,
        classification: str,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        marker: dict[str, Any] = {
            "schema": "metis.portage-requeue/v1",
            "created_at": utc_now(),
            "returncodes": dict(returncodes),
            "classification": classification,
            "checkpoints": self.checkpoint_states(
                require_checkpoint=require_checkpoint
            ),
            "posttraining": self.posttraining_states(),
            "resume_safe": True,
        }
        marker.update(dict(extra or {}))
        marker["marker_sha256"] = json_sha256(
            marker, omit=("marker_sha256",)
        )
        atomic_write_json(self.campaign_root / "requeue.json", marker)
        return marker

    def _materialization_srun_prefix(
        self,
        family: FamilyTopology,
        *,
        placement: str,
        execution: Mapping[str, Any],
    ) -> list[str]:
        protocol = execution.get("protocol")
        if protocol == "distributed_family_v1":
            return self._family_srun_prefix(family, placement=placement)
        if protocol != "rank0_only_v1":
            raise RuntimeError("Unsupported materialization execution protocol")
        if (
            set(execution)
            != {"protocol", "nodes", "tasks", "gpus_per_task"}
            or int(execution["nodes"]) != 1
            or int(execution["tasks"]) != 1
            or int(execution["gpus_per_task"]) not in {0, 1}
        ):
            raise RuntimeError("Rank-zero hook execution shape is invalid")
        cpus_per_task = max(
            1,
            int(self.config.raw["site"]["cpu_cores_per_node"])
            // self.config.accelerators_per_node
            - 1,
        )
        argv = [
            "srun",
            "--nodes",
            "1",
            "--ntasks",
            "1",
            "--ntasks-per-node",
            "1",
            "--cpus-per-task",
            str(cpus_per_task),
            "--exact",
            "--kill-on-bad-exit=1",
            "--distribution=block:block",
            "--cpu-bind=cores",
            "--mem-bind=local",
        ]
        if int(execution["gpus_per_task"]) == 1:
            argv.extend(("--gpus-per-task", "1", "--gpu-bind=closest"))
        nodes = self._nodes_for(family, placement)
        if nodes:
            argv.extend(("--nodelist", nodes[0]))
        else:
            argv.extend(("--relative", str(family.relative_node)))
        return argv

    def _current_materialization_request(
        self,
        *,
        family: FamilyTopology,
        restart_count: int,
    ) -> dict[str, Any]:
        request_root = (
            self._output_for(family)
            / "posttraining"
            / family.name
            / "materialization"
            / "requests"
        )
        job_id = str(os.environ.get("SLURM_JOB_ID", "local"))
        matches: list[dict[str, Any]] = []
        for path in sorted(request_root.glob("*.json")):
            try:
                raw = read_json(path)
            except Exception:
                continue
            if (
                raw.get("schema") == _DEFERRED_REQUEST_SCHEMA
                and raw.get("family") == family.name
                and raw.get("slurm_job_id") == job_id
                and raw.get("slurm_restart_count") == restart_count
            ):
                matches.append(
                    validate_deferred_materialization_request(
                        path,
                        output=self._output_for(family),
                        family=family,
                        posttraining_preflight=self.posttraining_release,
                        expected_job_id=job_id,
                        expected_restart_count=restart_count,
                    )
                )
        if len(matches) != 1:
            raise RuntimeError(
                f"{family.name} deferred exit produced {len(matches)} current "
                "materialization requests; expected exactly one"
            )
        return matches[0]

    def materialize_deferred_requirement(
        self,
        *,
        family: FamilyTopology,
        profile: Mapping[str, Any],
        restart_count: int,
    ) -> dict[str, Any]:
        step_id = os.environ.get("SLURM_STEP_ID") or os.environ.get("SLURM_STEPID")
        if step_id not in {None, "", "batch", "extern"}:
            raise RuntimeError(
                "Refusing to launch a generation srun from inside another Slurm step"
            )
        request = self._current_materialization_request(
            family=family,
            restart_count=restart_count,
        )
        hook = request["hook"]
        identity = str(request["_identity_sha256"])
        materialization_root = (
            self._output_for(family)
            / "posttraining"
            / family.name
            / "materialization"
        )
        attempts_path = materialization_root / "attempts" / f"{identity}.json"
        if attempts_path.is_file():
            attempts = read_json(attempts_path)
            if (
                attempts.get("schema")
                != "metis.deferred-materialization-attempts/v1"
                or attempts.get("family") != family.name
                or attempts.get("identity_sha256") != identity
                or attempts.get("attempts_sha256")
                != json_sha256(attempts, omit=("attempts_sha256",))
                or not isinstance(attempts.get("attempts"), list)
            ):
                raise RuntimeError("Deferred materialization attempt ledger is invalid")
        else:
            attempts = {
                "schema": "metis.deferred-materialization-attempts/v1",
                "family": family.name,
                "identity_sha256": identity,
                "request_sha256": request["request_sha256"],
                "attempts": [],
            }
        maximum_attempts = int(
            self.config.raw["autonomy"].get(
                "maximum_deferred_materialization_attempts", 3
            )
        )
        if not 1 <= maximum_attempts <= 10:
            raise RuntimeError("Deferred materialization retry cap must be in [1,10]")
        placement = str(profile.get("topology_placement", "contiguous"))
        environment = self._environment(family)
        autotune_profile_path = (
            self.campaign_root / "autotune" / family.name / "profile.json"
        ).resolve()
        precision_role_plan_path = self.precision_role_plan_paths[
            family.name
        ].resolve()
        if (
            not autotune_profile_path.is_file()
            or autotune_profile_path.is_symlink()
            or not precision_role_plan_path.is_file()
            or precision_role_plan_path.is_symlink()
        ):
            raise RuntimeError(
                "Deferred materialization lost its measured runtime contracts"
            )
        environment.update(
            {
                "METIS_FAMILY": family.name,
                "METIS_REPOSITORY_ROOT": str(self.config.repository),
                "METIS_FAMILY_MANIFEST": str(family.manifest),
                "METIS_DATA_RELEASE": str(self.config.release_root),
                "METIS_TRAINING_RUNTIME": str(self.config.runtime_policy),
                "METIS_POSTTRAINING_CONTRACT": str(
                    self.config.posttraining_contract
                ),
                "METIS_AUTOTUNE_PROFILE": str(autotune_profile_path),
                "METIS_PRECISION_ROLE_PLAN": str(
                    precision_role_plan_path
                ),
                "METIS_FAMILY_OUTPUT": str(self._output_for(family)),
                "METIS_STAGE_ID": str(request["stage"]),
                "METIS_REQUIREMENT_NAME": str(request["requirement"]),
                "METIS_PARENT_CHECKPOINT_SHA256": str(
                    request["parent_checkpoint_sha256"]
                ),
                "METIS_GENERATION_STAGE_BINDINGS_JSON": json.dumps(
                    request["stage_bindings"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "METIS_OUTPUT_MANIFEST": str(hook["output_manifest"]),
                "METIS_GENERATION_RECEIPT": str(hook["reducer_receipt"]),
                "METIS_GENERATION_RANK_RECEIPT_DIRECTORY": str(
                    hook["rank_receipts"]
                ),
                "METIS_GENERATION_RANK_RECEIPT_TEMPLATE": str(
                    Path(str(hook["rank_receipts"])) / "rank-%05d.json"
                ),
                "METIS_GENERATION_REQUEST": str(request["_request_path"]),
                "METIS_GENERATION_REQUEST_SHA256": str(
                    request["request_sha256"]
                ),
                "METIS_GENERATION_REQUEST_FILE_SHA256": str(
                    request["_request_file_sha256"]
                ),
                "METIS_GENERATION_PROTOCOL": str(
                    hook["execution"]["protocol"]
                ),
                "METIS_GENERATION_WORLD_SIZE": str(hook["world_size"]),
                "METIS_RELEASE_INDEX_SHA256": str(
                    request["release_index_sha256"]
                ),
                "METIS_RELEASE_INDEX_FILE_SHA256": str(
                    request["release_index_file_sha256"]
                ),
                "METIS_DEFERRED_RECORD_SHA256": str(request["record_sha256"]),
            }
        )
        last_error = "not attempted"
        while len(attempts["attempts"]) < maximum_attempts:
            if self.signal_requested.is_set():
                raise RuntimeError(
                    "Checkpoint signal arrived before deferred materialization"
                )
            attempt_number = len(attempts["attempts"]) + 1
            argv = [
                *self._materialization_srun_prefix(
                    family,
                    placement=placement,
                    execution=hook["execution"],
                ),
                str(hook["executable"]),
                *list(hook["args"]),
            ]
            returncode = self._run(
                argv,
                log_path=(
                    self.campaign_root
                    / "logs"
                    / "materialization"
                    / family.name
                    / f"{identity[:16]}-attempt-{attempt_number:02d}.log"
                ),
                environment=environment,
                timeout=float(hook["timeout_seconds"]) + 120.0,
            )
            result: dict[str, Any] | None = None
            try:
                if returncode != 0:
                    raise RuntimeError(
                        f"generation srun exited with status {returncode}"
                    )
                result = validate_deferred_materialization_result(request)
                last_error = ""
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            attempt = {
                "attempt": attempt_number,
                "created_at": utc_now(),
                "request_sha256": request["request_sha256"],
                "returncode": returncode,
                "success": result is not None,
                "error": last_error or None,
                "result": result,
            }
            attempt["attempt_sha256"] = json_sha256(
                attempt, omit=("attempt_sha256",)
            )
            attempts["attempts"].append(attempt)
            attempts["updated_at"] = utc_now()
            attempts["attempts_sha256"] = json_sha256(
                attempts, omit=("attempts_sha256",)
            )
            atomic_write_json(attempts_path, attempts)
            if result is not None:
                completion = {
                    "schema": "metis.deferred-materialization-complete/v1",
                    "created_at": utc_now(),
                    "family": family.name,
                    "stage": request["stage"],
                    "requirement": request["requirement"],
                    "parent_checkpoint_sha256": request[
                        "parent_checkpoint_sha256"
                    ],
                    "identity_sha256": identity,
                    "request_path": request["_request_path"],
                    "request_file_sha256": request["_request_file_sha256"],
                    "request_sha256": request["request_sha256"],
                    "attempts_path": str(attempts_path),
                    "attempts_sha256": attempts["attempts_sha256"],
                    "result": result,
                    "resume_authorized": True,
                }
                completion["completion_sha256"] = json_sha256(
                    completion, omit=("completion_sha256",)
                )
                atomic_write_json(
                    materialization_root
                    / "completions"
                    / f"{identity}.json",
                    completion,
                )
                return completion
        raise RuntimeError(
            f"{family.name} deferred materialization exhausted "
            f"{maximum_attempts} attempts: {last_error}"
        )

    def train_with_deferred_materialization(
        self,
        family: FamilyTopology,
        profile: dict[str, Any],
    ) -> int:
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        maximum_handoffs = int(
            self.config.raw["autonomy"].get(
                "maximum_deferred_materializations_per_family", 32
            )
        )
        if not 1 <= maximum_handoffs <= 128:
            raise RuntimeError("Deferred materialization handoff cap is invalid")
        for _handoff in range(maximum_handoffs + 1):
            returncode = self.train(family, profile)
            if returncode != DEFERRED_MATERIALIZATION_EXIT_CODE:
                return returncode
            if self.signal_requested.is_set():
                return REQUEUE_EXIT_CODE
            try:
                self.materialize_deferred_requirement(
                    family=family,
                    profile=profile,
                    restart_count=restart_count,
                )
            except Exception as exc:
                if self.signal_requested.is_set():
                    return REQUEUE_EXIT_CODE
                failure = {
                    "schema": "metis.deferred-materialization-failure/v1",
                    "created_at": utc_now(),
                    "family": family.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "automatic_resume_safe": False,
                }
                failure["failure_sha256"] = json_sha256(
                    failure, omit=("failure_sha256",)
                )
                atomic_write_json(
                    self._output_for(family)
                    / "posttraining"
                    / family.name
                    / "materialization"
                    / "FAILURE.json",
                    failure,
                )
                return DEFERRED_MATERIALIZATION_EXIT_CODE
        raise RuntimeError(
            f"{family.name} exceeded its deferred materialization handoff cap"
        )

    def revise_posttraining_batch_after_oom(
        self,
        *,
        family: FamilyTopology,
        restart_count: int,
    ) -> dict[str, Any] | None:
        """Append one checkpoint-bound stage batch migration.

        The stage backend owns immutable OOM requests because it knows the
        exact live stage, sealed bundle, parent checkpoint, and resume
        coordinates.  The supervisor accepts only requests from this Slurm
        execution attempt, then writes a chained receipt outside the sealed
        data release.  No model, optimizer, learning-rate, or dataset field can
        change through this path.
        """

        job_id = str(os.environ.get("SLURM_JOB_ID", "local"))
        expected_precision_role_plan_sha256 = self.precision_role_plans[
            family.name
        ]["plan_sha256"]
        output_root = self._output_for(family) / "posttraining" / family.name
        oom_root = (output_root / "oom").resolve()
        if not oom_root.is_dir():
            return None
        matching: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(oom_root.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                request = read_json(path)
            except Exception:
                continue
            if (
                request.get("schema") == _STAGE_OOM_REQUEST_SCHEMA
                and request.get("family") == family.name
                and request.get("slurm_job_id") == job_id
                and request.get("slurm_restart_count") == restart_count
            ):
                matching.append((path.resolve(), request))
        if not matching:
            return None

        identity: dict[str, Any] | None = None
        ranks: set[int] = set()
        for path, request in matching:
            current = request.get("current")
            proposed = request.get("proposed")
            resume = request.get("resume")
            raw_rank = request.get("rank", -1)
            rank = int(raw_rank)
            if (
                request.get("request_sha256")
                != json_sha256(request, omit=("request_sha256",))
                or not _SHA256.fullmatch(
                    str(request.get("parent_checkpoint_sha256", ""))
                )
                or not _SHA256.fullmatch(
                    str(request.get("bundle_manifest_sha256", ""))
                )
                or request.get("precision_role_plan_sha256")
                != expected_precision_role_plan_sha256
                or request.get("world_size") != family.world_size
                or isinstance(raw_rank, bool)
                or not 0 <= rank < family.world_size
                or rank in ranks
                or not isinstance(current, Mapping)
                or not isinstance(proposed, Mapping)
                or not isinstance(resume, Mapping)
                or set(current)
                != {"micro_batch_size", "gradient_accumulation"}
                or set(proposed)
                != {"micro_batch_size", "gradient_accumulation"}
                or (
                    request.get("prior_batch_migration_sha256") is not None
                    and not _SHA256.fullmatch(
                        str(request.get("prior_batch_migration_sha256"))
                    )
                )
                or request.get("revision_available") is not True
            ):
                raise RuntimeError(
                    f"{family.name} stage OOM request is invalid: {path}"
                )
            old_micro = int(current.get("micro_batch_size", -1))
            old_accum = int(current.get("gradient_accumulation", -1))
            new_micro = int(proposed.get("micro_batch_size", -1))
            new_accum = int(proposed.get("gradient_accumulation", -1))
            if (
                old_micro <= 0
                or old_accum <= 0
                or new_micro <= 0
                or new_accum <= 0
                or new_micro >= old_micro
                or new_accum <= old_accum
                or old_micro * old_accum != new_micro * new_accum
            ):
                raise RuntimeError(
                    f"{family.name} stage OOM request changes effective batch"
                )
            try:
                path.relative_to(oom_root)
            except ValueError as exc:
                raise RuntimeError("Stage OOM request escapes its output root") from exc
            candidate_identity = {
                "stage": str(request.get("stage", "")),
                "parent_checkpoint_sha256": request["parent_checkpoint_sha256"],
                "bundle_manifest_sha256": request["bundle_manifest_sha256"],
                "precision_role_plan_sha256": request[
                    "precision_role_plan_sha256"
                ],
                "prior_batch_migration_sha256": request.get(
                    "prior_batch_migration_sha256"
                ),
                "sequence_length": int(request.get("sequence_length", 0)),
                "phase": str(request.get("phase", "")),
                "resume": dict(resume),
                "current": dict(current),
                "proposed": dict(proposed),
            }
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]*", candidate_identity["stage"])
                or candidate_identity["sequence_length"] <= 1
                or not candidate_identity["phase"]
                or (
                    identity is not None
                    and candidate_identity != identity
                )
            ):
                raise RuntimeError(
                    f"{family.name} ranks reported inconsistent stage OOMs"
                )
            identity = candidate_identity
            ranks.add(rank)
        assert identity is not None
        selected_path, selected_request = min(
            matching,
            key=lambda row: int(row[1]["rank"]),
        )
        migration_root = (
            self.campaign_root / "posttraining-batch-migrations"
        ).resolve()
        receipt_path = (
            migration_root / family.name / f"{identity['stage']}.json"
        )
        existing: dict[str, Any] | None = None
        if receipt_path.is_file():
            if receipt_path.is_symlink():
                raise RuntimeError("Post-training batch migration may not be a symlink")
            existing = read_json(receipt_path)
            if (
                existing.get("schema") != _STAGE_BATCH_MIGRATION_SCHEMA
                or existing.get("receipt_sha256")
                != json_sha256(existing, omit=("receipt_sha256",))
                or existing.get("family") != family.name
                or existing.get("stage") != identity["stage"]
                or existing.get("parent_checkpoint_sha256")
                != identity["parent_checkpoint_sha256"]
                or existing.get("bundle_manifest_sha256")
                != identity["bundle_manifest_sha256"]
                or existing.get("precision_role_plan_sha256")
                != identity["precision_role_plan_sha256"]
            ):
                raise RuntimeError(
                    f"{family.name} prior stage batch migration is invalid"
                )
            if (
                identity["prior_batch_migration_sha256"]
                != existing["receipt_sha256"]
            ):
                raise RuntimeError(
                    f"{family.name} stage OOM request has stale migration lineage"
                )
            if not isinstance(existing.get("sealed_training"), Mapping) or not isinstance(
                existing.get("revisions"), list
            ):
                raise RuntimeError(
                    f"{family.name} prior stage batch migration is malformed"
                )
            sealed_training = dict(existing["sealed_training"])
            revisions = list(existing["revisions"])
            if not revisions or revisions[-1].get("new") != identity["current"]:
                raise RuntimeError(
                    f"{family.name} stage migration chain does not reach current batch"
                )
        else:
            if identity["prior_batch_migration_sha256"] is not None:
                raise RuntimeError(
                    f"{family.name} stage OOM request references a missing migration"
                )
            sealed_training = dict(identity["current"])
            revisions = []
        if set(sealed_training) != {
            "micro_batch_size",
            "gradient_accumulation",
        }:
            raise RuntimeError("Stage migration sealed_training is invalid")
        effective_batch = int(sealed_training["micro_batch_size"]) * int(
            sealed_training["gradient_accumulation"]
        )
        if effective_batch <= 0:
            raise RuntimeError("Stage migration effective batch must be positive")

        # Re-validate the prior chain and every immutable OOM request before
        # authorizing a new receipt.
        prior = dict(sealed_training)
        for revision in revisions:
            if (
                not isinstance(revision, Mapping)
                or revision.get("revision_sha256")
                != json_sha256(revision, omit=("revision_sha256",))
                or revision.get("reason") != "measured_stage_oom"
                or revision.get("old") != prior
                or not isinstance(revision.get("new"), Mapping)
            ):
                raise RuntimeError("Prior stage batch migration chain is invalid")
            old = dict(revision["old"])
            new = dict(revision["new"])
            if (
                int(new.get("micro_batch_size", -1))
                >= int(old.get("micro_batch_size", -1))
                or int(new.get("gradient_accumulation", -1))
                <= int(old.get("gradient_accumulation", -1))
                or int(new.get("micro_batch_size", -1))
                * int(new.get("gradient_accumulation", -1))
                != effective_batch
            ):
                raise RuntimeError("Prior stage batch migration changed semantics")
            prior_request_path = Path(
                str(revision.get("oom_request_path", ""))
            ).expanduser().resolve()
            try:
                prior_request_path.relative_to(oom_root)
            except ValueError as exc:
                raise RuntimeError(
                    "Prior stage OOM request escaped the output root"
                ) from exc
            if (
                not prior_request_path.is_file()
                or prior_request_path.is_symlink()
                or file_sha256(prior_request_path)
                != revision.get("oom_request_file_sha256")
            ):
                raise RuntimeError("Prior stage OOM request bytes changed")
            prior_request = read_json(prior_request_path)
            if (
                prior_request.get("request_sha256")
                != revision.get("oom_request_sha256")
                or prior_request.get("request_sha256")
                != json_sha256(prior_request, omit=("request_sha256",))
                or prior_request.get("current") != old
                or prior_request.get("proposed") != new
                or prior_request.get("precision_role_plan_sha256")
                != identity["precision_role_plan_sha256"]
            ):
                raise RuntimeError("Prior stage OOM request lineage changed")
            prior = new
        if prior != identity["current"]:
            raise RuntimeError("Stage migration current batch is not the chain tip")

        revision: dict[str, Any] = {
            "reason": "measured_stage_oom",
            "old": identity["current"],
            "new": identity["proposed"],
            "prior_batch_migration_sha256": identity[
                "prior_batch_migration_sha256"
            ],
            "oom_request_path": str(selected_path),
            "oom_request_file_sha256": file_sha256(selected_path),
            "oom_request_sha256": selected_request["request_sha256"],
            "slurm_job_id": job_id,
            "slurm_restart_count": restart_count,
            "observed_ranks": sorted(ranks),
            "resume": identity["resume"],
        }
        revision["revision_sha256"] = json_sha256(revision)
        receipt: dict[str, Any] = {
            "schema": _STAGE_BATCH_MIGRATION_SCHEMA,
            "created_at": (
                existing.get("created_at") if existing is not None else utc_now()
            ),
            "updated_at": utc_now(),
            "family": family.name,
            "stage": identity["stage"],
            "parent_checkpoint_sha256": identity["parent_checkpoint_sha256"],
            "bundle_manifest_sha256": identity["bundle_manifest_sha256"],
            "precision_role_plan_sha256": identity[
                "precision_role_plan_sha256"
            ],
            "sealed_training": sealed_training,
            "effective_local_batch_records": effective_batch,
            "revisions": [*revisions, revision],
        }
        receipt["receipt_sha256"] = json_sha256(receipt)
        atomic_write_json(receipt_path, receipt)
        return {
            "path": str(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "stage": identity["stage"],
            "old": identity["current"],
            "new": identity["proposed"],
            "effective_local_batch_records": effective_batch,
            "precision_role_plan_sha256": identity[
                "precision_role_plan_sha256"
            ],
            "observed_ranks": sorted(ranks),
        }

    def revise_profile_after_oom(
        self,
        *,
        family: FamilyTopology,
        profile: dict[str, Any],
        checkpoint: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        candidate = derive_oom_candidate(
            profile=profile,
            family=family,
            config=self.config,
        )
        directory = self.campaign_root / "autotune" / family.name
        revision_index = int(profile.get("revision", {}).get("index", 0)) + 1
        report_path = (
            directory
            / "trials"
            / f"production-oom-revision-{revision_index:03d}.json"
        )
        placement = str(profile.get("topology_placement", "contiguous"))
        report = self.trial_runner(
            family,
            placement=placement,
        )(candidate, report_path, False)
        bounds = load_tuning_bounds(
            family.manifest,
            default_maximum_hbm_fraction=float(
                self.config.raw["autotune"]["default_maximum_hbm_fraction"]
            ),
        )
        parity_reference: dict[str, Any] | None = None
        if candidate.ngram_table_mode == "row_sharded":
            replicated = Candidate(
                **{
                    **candidate.__dict__,
                    "ngram_table_mode": "replicated",
                }
            )
            reference_path = report_path.with_name(
                report_path.stem + "-ngram-replicated.json"
            )
            reference_report = self.trial_runner(
                family,
                placement=placement,
            )(
                replicated,
                reference_path,
                False,
            )
            reference_passed, reference_reasons, _ = validate_performance_report(
                reference_report,
                candidate=replicated,
                bounds=bounds,
                hbm_bytes=self.hbm_bytes,
            )
            if not reference_passed:
                raise RuntimeError(
                    f"{family.name} OOM revision replicated parity lane failed: "
                    + "; ".join(reference_reasons)
                )
            row_loss = float(report["final_loss"])
            reference_loss = float(reference_report["final_loss"])
            relative_error = abs(row_loss - reference_loss) / max(
                abs(reference_loss),
                1.0e-12,
            )
            parity_reference = {
                "path": str(reference_path),
                "sha256": file_sha256(reference_path),
            }
            report = {
                **report,
                "ngram_layout_reference_ok": True,
                "ngram_layout_loss_relative_error": relative_error,
                "ngram_replicated_reference_report": str(reference_path),
                "ngram_replicated_reference_report_sha256": file_sha256(
                    reference_path
                ),
            }
            atomic_write_json(report_path, report)
        passed, reasons, throughput = validate_performance_report(
            report,
            candidate=candidate,
            bounds=bounds,
            hbm_bytes=self.hbm_bytes,
        )
        if (
            report.get("precision_role_plan_sha256")
            != self.precision_role_plans[family.name]["plan_sha256"]
        ):
            passed = False
            reasons = [
                *reasons,
                "OOM canary changed the precision role plan",
            ]
        if not passed:
            raise RuntimeError(
                f"{family.name} smaller OOM revision failed its canary: "
                + "; ".join(reasons)
            )
        old_profile_sha = str(profile["profile_sha256"])
        old_selected = dict(profile["selected"])
        new_selected = {
            **old_selected,
            "micro_batch_size": candidate.micro_batch_size,
            "grad_accum_steps": candidate.grad_accum_steps,
        }
        old_local_batch = int(old_selected["micro_batch_size"]) * int(
            old_selected["grad_accum_steps"]
        )
        new_local_batch = int(new_selected["micro_batch_size"]) * int(
            new_selected["grad_accum_steps"]
        )
        if old_local_batch != new_local_batch:
            raise RuntimeError("OOM profile revision changed the global token batch")
        expected_role_plan_sha = self.precision_role_plans[family.name][
            "plan_sha256"
        ]
        if (
            profile.get("precision_role_plan_sha256")
            != expected_role_plan_sha
            or profile.get("precision_role_plan")
            != self.precision_role_plans[family.name]
        ):
            raise RuntimeError(
                f"{family.name} OOM revision has a stale precision role plan"
            )
        revision = {
            "kind": "production_oom_microbatch_reduction",
            "index": revision_index,
            "created_at": utc_now(),
            "parent_profile_sha256": old_profile_sha,
            "changed_fields": [
                "micro_batch_size",
                "grad_accum_steps",
            ],
            "state_conversion": "none_parameter_and_optimizer_layout_unchanged",
            "canary_report": str(report_path),
            "canary_report_sha256": file_sha256(report_path),
            "parity_reference": parity_reference,
        }
        new_profile = {
            **profile,
            "created_at": utc_now(),
            "selected": new_selected,
            "measured_tokens_per_second": throughput,
            "measured_estimated_train_flops": float(
                report["estimated_train_flops"]
            ),
            "peak_hbm_bytes": int(report["peak_hbm_bytes"]),
            "revision": revision,
            "trials": [
                *list(profile.get("trials", [])),
                {
                    "kind": "production_oom_revision",
                    "candidate": candidate.__dict__,
                    "report": str(report_path),
                    "report_sha256": file_sha256(report_path),
                    "passed": True,
                    "reasons": [],
                    "tokens_per_second": throughput,
                },
            ],
        }
        migration: dict[str, Any] | None = None
        if checkpoint["status"] == "durable_checkpoint":
            checkpoint_profile_sha = str(
                checkpoint["autotune_profile_sha256"]
            )
            compatibility = profile.get("checkpoint_compatibility", {})
            if checkpoint_profile_sha == old_profile_sha:
                ancestor_selected = old_selected
            elif (
                isinstance(compatibility, dict)
                and compatibility.get("checkpoint_profile_sha256")
                == checkpoint_profile_sha
                and isinstance(compatibility.get("checkpoint_selected"), dict)
            ):
                ancestor_selected = dict(
                    compatibility["checkpoint_selected"]
                )
            else:
                raise RuntimeError(
                    f"{family.name} OOM revision cannot prove checkpoint/profile ancestry"
                )
            if checkpoint.get("precision_role_plan_sha256") != expected_role_plan_sha:
                raise RuntimeError(
                    f"{family.name} OOM revision would cross precision role plans"
                )
            immutable_fields = (
                "learning_rate",
                "precision_profile",
                "compile_mode",
                "dispatch_overlap",
                "ngram_table_mode",
            )
            if any(
                ancestor_selected.get(field) != new_selected.get(field)
                for field in immutable_fields
            ) or (
                int(ancestor_selected["micro_batch_size"])
                * int(ancestor_selected["grad_accum_steps"])
                != new_local_batch
            ):
                raise RuntimeError(
                    f"{family.name} OOM revision changed checkpoint state semantics"
                )
            new_profile["checkpoint_compatibility"] = {
                "checkpoint_profile_sha256": checkpoint_profile_sha,
                "checkpoint_selected": ancestor_selected,
                "allowed_changes": [
                    "micro_batch_size",
                    "grad_accum_steps",
                ],
            }
        else:
            new_profile.pop("checkpoint_compatibility", None)
        new_profile["profile_sha256"] = json_sha256(
            new_profile,
            omit=("profile_sha256",),
        )
        profile_path = directory / "profile.json"
        archive = directory / f"profile.revision-{old_profile_sha[:16]}.json"
        if profile_path.is_file():
            profile_path.replace(archive)
        atomic_write_json(profile_path, new_profile)
        migration_path = directory / "profile-migration.json"
        if checkpoint["status"] == "durable_checkpoint":
            migration = {
                "schema": "metis.autotune-profile-migration/v1",
                "created_at": utc_now(),
                "family": family.name,
                "checkpoint": checkpoint["checkpoint"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_manifest_sha256": checkpoint[
                    "checkpoint_manifest_sha256"
                ],
                "old_profile_sha256": checkpoint[
                    "autotune_profile_sha256"
                ],
                "new_profile_path": str(profile_path),
                "new_profile_sha256": new_profile["profile_sha256"],
                "precision_role_plan_sha256": expected_role_plan_sha,
                "old_selected": new_profile["checkpoint_compatibility"][
                    "checkpoint_selected"
                ],
                "new_selected": new_selected,
                "allowed_changes": [
                    "micro_batch_size",
                    "grad_accum_steps",
                ],
                "global_token_batch_unchanged": True,
                "state_conversion": "none_parameter_and_optimizer_layout_unchanged",
            }
            migration["receipt_sha256"] = json_sha256(migration)
            atomic_write_json(migration_path, migration)
        else:
            migration_path.unlink(missing_ok=True)
        return new_profile, migration

    def _run(
        self,
        argv: Sequence[str],
        *,
        log_path: Path,
        environment: dict[str, str],
        timeout: float | None = None,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "created_at": utc_now(),
                        "argv": list(argv),
                        "environment_keys": sorted(safe_environment(environment)),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            log.flush()
            process = subprocess.Popen(
                list(argv),
                cwd=self.config.repository,
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with self.process_lock:
                self.processes.append(process)
            try:
                return int(process.wait(timeout=timeout))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124
            finally:
                with self.process_lock:
                    if process in self.processes:
                        self.processes.remove(process)

    def request_checkpoint(self, _signum: int, _frame: Any) -> None:
        self.signal_requested.set()
        self.checkpoint_running()

    def checkpoint_running(self) -> None:
        with self.process_lock:
            processes = list(self.processes)
        for process in processes:
            if process.poll() is None:
                try:
                    # SchedMD documents that synchronous srun forwards signals
                    # it receives to every task in the controlled step.
                    os.kill(process.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass

    def audit(self, family: FamilyTopology) -> dict[str, Any]:
        output = self.campaign_root / "audit" / f"{family.name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        argv = [
            *self._base_trainer_args(
                family,
                output=self.campaign_root / self.config.raw["training"]["output_subdirectory"] / family.name,
                profile=None,
                stage="pretrain",
            ),
            "--audit-config",
            "--json-output",
            str(output),
        ]
        returncode = self._run(
            argv,
            log_path=self.campaign_root / "logs" / f"audit-{family.name}.log",
            environment=self._environment(family),
            timeout=600,
        )
        if returncode != 0 or not output.is_file():
            raise RuntimeError(f"Trainer config audit failed for {family.name}")
        report = read_json(output)
        if (
            report.get("ok") is not True
            or report.get("family") not in {None, family.name}
            or int(report.get("world_size", family.world_size)) != family.world_size
            or report.get("release", {}).get("marker_verified") is not True
        ):
            raise RuntimeError(f"Trainer rejected the {family.name} production manifest")
        return report

    def trial_runner(
        self,
        family: FamilyTopology,
        *,
        placement: str = "contiguous",
    ):
        tunable_directory = self.campaign_root / "autotune" / family.name / "kernels"

        def run(candidate: Candidate, report_path: Path, optimizer: bool) -> dict[str, Any]:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            command = self._base_trainer_args(
                family,
                output=self.campaign_root / "autotune" / family.name / "scratch",
                profile=None,
                stage="pretrain",
            )
            command.extend(
                (
                    "--probe",
                    "--probe-steps",
                    str(
                        self.config.raw["autotune"][
                            "optimizer_probe_steps" if optimizer else "probe_steps"
                        ]
                    ),
                    "--micro-batch",
                    str(candidate.micro_batch_size),
                    "--grad-accum",
                    str(candidate.grad_accum_steps),
                    "--precision-profile",
                    candidate.precision_profile,
                    "--compile-mode",
                    candidate.compile_mode,
                    "--overlap-dispatch",
                    "on" if candidate.dispatch_overlap else "off",
                    "--ngram-table-mode",
                    candidate.ngram_table_mode,
                    "--json-output",
                    str(report_path),
                    "--precision-role-plan",
                    str(self.precision_role_plan_paths[family.name]),
                )
            )
            if candidate.learning_rate is not None:
                command.extend(("--lr-candidate", repr(candidate.learning_rate)))
            argv = [
                *self._family_srun_prefix(family, placement=placement),
                "python3",
                "-m",
                "metis_portage.exec_train",
                "--tunable-directory",
                str(tunable_directory),
                "--tuning",
                "1",
                "--",
                *command,
            ]
            trial_name = report_path.stem
            returncode = self._run(
                argv,
                log_path=self.campaign_root
                / "logs"
                / "autotune"
                / family.name
                / f"{trial_name}.log",
                environment=self._environment(family),
                timeout=None,
            )
            if not report_path.is_file():
                failure = {
                    "schema": "metis.trainer-probe/v1",
                    "ok": False,
                    "finite_loss": False,
                    "returncode": returncode,
                    "failure": "trainer did not emit a probe report",
                    "created_at": utc_now(),
                }
                atomic_write_json(report_path, failure)
            report = read_json(report_path)
            if returncode != 0:
                report = {
                    **report,
                    "ok": False,
                    "returncode": returncode,
                }
                atomic_write_json(report_path, report)
            return report

        return run

    def tune(self, family: FamilyTopology) -> dict[str, Any]:
        profile_path = self.campaign_root / "autotune" / family.name / "profile.json"
        if profile_path.is_file():
            try:
                return validate_profile(
                    profile_path,
                    family=family,
                    inventory_fingerprint=self.fingerprint,
                    release_marker_sha256=self.release_marker["marker_sha256"],
                    maximum_age_days=int(
                        self.config.raw["autotune"]["maximum_profile_age_days"]
                    ),
                )
            except RuntimeError:
                # Stale profiles are retained as evidence and never reused.
                stale = profile_path.with_name(
                    f"profile.stale-{int(time.time())}.json"
                )
                profile_path.replace(stale)
        rejected_path = (
            self.campaign_root / "autotune" / family.name / "rejected-candidates.json"
        )
        rejected = (
            read_json(rejected_path).get("candidates", [])
            if rejected_path.is_file()
            else []
        )
        return tune_family(
            config=self.config,
            family=family,
            inventory_fingerprint=self.fingerprint,
            release_marker=self.release_marker,
            hbm_bytes=self.hbm_bytes,
            output_directory=self.campaign_root / "autotune" / family.name,
            run_trial=self.trial_runner(family),
            excluded_candidates=rejected,
            available_precision_profiles=(
                ["fp8", "bf16"]
                if "fp8"
                in set(
                    measured_role_dtype_map(
                        self.precision_role_plans[family.name]
                    ).values()
                )
                else ["bf16"]
            ),
            precision_role_plan=self.precision_role_plans[family.name],
        )

    def topology_race(
        self,
        profiles: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        placements = [
            str(item) for item in self.config.raw["autotune"]["topology_placements"]
        ]
        if not placements or any(
            item not in {"contiguous", "interleaved"} for item in placements
        ):
            raise RuntimeError("Topology race has no supported placement candidates")
        if all(
            profile.get("topology_placement") in placements
            and profile.get("topology_race", {}).get("inventory_fingerprint")
            == self.fingerprint
            for profile in profiles.values()
        ):
            return profiles
        race_rows: list[dict[str, Any]] = []
        for placement in placements:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for family in self.config.families:
                    selected = profiles[family.name]["selected"]
                    candidate = Candidate(
                        micro_batch_size=int(selected["micro_batch_size"]),
                        grad_accum_steps=int(selected["grad_accum_steps"]),
                        precision_profile=str(selected["precision_profile"]),
                        compile_mode=str(selected["compile_mode"]),
                        dispatch_overlap=bool(selected["dispatch_overlap"]),
                        ngram_table_mode=str(selected["ngram_table_mode"]),
                        learning_rate=None,
                    )
                    output = (
                        self.campaign_root
                        / "autotune"
                        / "topology"
                        / placement
                        / f"{family.name}.json"
                    )
                    futures[family.name] = (
                        family,
                        candidate,
                        output,
                        executor.submit(
                            self.trial_runner(family, placement=placement),
                            candidate,
                            output,
                            False,
                        ),
                    )
                family_rows: dict[str, Any] = {}
                placement_ok = True
                combined_throughput = 0.0
                for name, (family, candidate, output, future) in futures.items():
                    report = future.result()
                    bounds = load_tuning_bounds(
                        family.manifest,
                        default_maximum_hbm_fraction=float(
                            self.config.raw["autotune"][
                                "default_maximum_hbm_fraction"
                            ]
                        ),
                    )
                    passed, reasons, throughput = validate_performance_report(
                        report,
                        candidate=candidate,
                        bounds=bounds,
                        hbm_bytes=self.hbm_bytes,
                    )
                    if (
                        report.get("precision_role_plan_sha256")
                        != profiles[name].get("precision_role_plan_sha256")
                    ):
                        passed = False
                        reasons = [
                            *reasons,
                            "topology race changed the precision role plan",
                        ]
                    family_rows[name] = {
                        "passed": passed,
                        "reasons": reasons,
                        "tokens_per_second": throughput,
                        "report": str(output),
                        "report_sha256": file_sha256(output),
                    }
                    placement_ok = placement_ok and passed
                    combined_throughput += throughput
                race_rows.append(
                    {
                        "placement": placement,
                        "passed": placement_ok,
                        "combined_tokens_per_second": combined_throughput,
                        "families": family_rows,
                    }
                )
        passing = [row for row in race_rows if row["passed"]]
        if not passing:
            raise RuntimeError("No simultaneous Praxis/Logos topology placement passed")
        selected_row = max(
            passing,
            key=lambda row: (
                float(row["combined_tokens_per_second"]),
                1 if row["placement"] == "contiguous" else 0,
            ),
        )
        race = {
            "schema": "metis.portage-topology-race/v1",
            "created_at": utc_now(),
            "inventory_fingerprint": self.fingerprint,
            "selected": selected_row["placement"],
            "rows": race_rows,
        }
        race["race_sha256"] = json_sha256(
            race,
            omit=("race_sha256",),
        )
        atomic_write_json(
            self.campaign_root / "autotune" / "topology" / "race.json",
            race,
        )
        for family in self.config.families:
            profile = profiles[family.name]
            profile["topology_placement"] = selected_row["placement"]
            profile["topology_race"] = {
                "inventory_fingerprint": self.fingerprint,
                "race_sha256": race["race_sha256"],
            }
            profile["profile_sha256"] = json_sha256(
                profile, omit=("profile_sha256",)
            )
            atomic_write_json(
                self.campaign_root / "autotune" / family.name / "profile.json",
                profile,
            )
        return profiles

    def node_snapshot(self, label: str) -> None:
        output = self.campaign_root / "telemetry" / "nodes"
        argv = [
            "srun",
            "--nodes",
            "128",
            "--ntasks",
            "128",
            "--ntasks-per-node",
            "1",
            "--exact",
            "--distribution=block:block",
            "python3",
            "-m",
            "metis_portage.snapshot",
            "--output",
            str(output),
            "--label",
            label,
        ]
        returncode = self._run(
            argv,
            log_path=self.campaign_root / "logs" / f"snapshot-{label}.log",
            environment=self._environment(self.config.families[0]),
            timeout=600,
        )
        snapshots = list((output / label).glob("*.json"))
        if returncode != 0 or len(snapshots) != 128:
            raise RuntimeError(
                f"Node telemetry snapshot {label} covered {len(snapshots)}/128 nodes"
            )
        if self.config.raw["site"]["require_cxi_counters"]:
            unreadable = [
                path for path in snapshots if not read_json(path).get("cxi", {}).get("ok")
            ]
            if unreadable:
                raise RuntimeError(
                    f"CXI counters unreadable on {len(unreadable)} allocated nodes"
                )

    def start_monitor(self) -> subprocess.Popen[str]:
        output = self.campaign_root / "telemetry" / "continuous"
        output.mkdir(parents=True, exist_ok=True)
        argv = [
            "srun",
            "--nodes",
            "128",
            "--ntasks",
            "128",
            "--ntasks-per-node",
            "1",
            "--cpus-per-task",
            "1",
            "--exact",
            "--distribution=block:block",
            "--cpu-bind=cores",
            "python3",
            "-m",
            "metis_portage.monitor",
            "--config",
            str(self.config.path),
            "--output",
            str(output),
        ]
        log_path = self.campaign_root / "logs" / "continuous-telemetry.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            argv,
            cwd=self.config.repository,
            env=self._environment(self.config.families[0]),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        with self.process_lock:
            self.processes.append(process)
        return process

    def stop_monitor(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            try:
                os.kill(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
        with self.process_lock:
            if process in self.processes:
                self.processes.remove(process)
        files = list((self.campaign_root / "telemetry" / "continuous").glob("*.jsonl"))
        if returncode not in {0, -signal.SIGTERM, 128 + signal.SIGTERM}:
            raise RuntimeError(f"Continuous telemetry step failed with {returncode}")
        if len(files) != 128 or any(path.stat().st_size == 0 for path in files):
            raise RuntimeError(
                f"Continuous telemetry covered {len(files)}/128 allocated nodes"
            )

    def train(self, family: FamilyTopology, profile: dict[str, Any]) -> int:
        profile_path = self.campaign_root / "autotune" / family.name / "profile.json"
        output = (
            self.campaign_root
            / self.config.raw["training"]["output_subdirectory"]
            / family.name
        )
        command = self._base_trainer_args(
            family,
            output=output,
            profile=profile_path,
            stage=str(self.config.raw["training"]["campaign_stage"]),
        )
        argv = [
            *self._family_srun_prefix(
                family,
                placement=str(profile.get("topology_placement", "contiguous")),
            ),
            "python3",
            "-m",
            "metis_portage.exec_train",
            "--tunable-directory",
            str(self.campaign_root / "autotune" / family.name / "kernels"),
            "--tuning",
            "0",
            "--",
            *command,
        ]
        return self._run(
            argv,
            log_path=self._training_log_path(family.name),
            environment=self._environment(family),
            timeout=None,
        )

    def reject_profiles(
        self,
        profiles: dict[str, dict[str, Any]],
        families: list[str],
        *,
        reason: str,
    ) -> None:
        for name in families:
            directory = self.campaign_root / "autotune" / name
            rejected_path = directory / "rejected-candidates.json"
            payload = (
                read_json(rejected_path)
                if rejected_path.is_file()
                else {
                    "schema": "metis.portage-rejected-candidates/v1",
                    "family": name,
                    "candidates": [],
                }
            )
            selected = dict(profiles[name]["selected"])
            selected["learning_rate"] = None
            normalized = {
                key: selected[key]
                for key in (
                    "micro_batch_size",
                    "grad_accum_steps",
                    "precision_profile",
                    "compile_mode",
                    "dispatch_overlap",
                    "ngram_table_mode",
                    "learning_rate",
                )
            }
            if normalized not in payload["candidates"]:
                payload["candidates"].append(normalized)
            payload["last_reason"] = reason
            payload["updated_at"] = utc_now()
            payload["payload_sha256"] = json_sha256(
                payload, omit=("payload_sha256",)
            )
            atomic_write_json(rejected_path, payload)
            profile_path = directory / "profile.json"
            if profile_path.is_file():
                profile_path.replace(
                    directory / f"profile.rejected-{int(time.time())}.json"
                )

    def classify_failure(self, returncodes: dict[str, int]) -> tuple[str, list[str]]:
        failed = [name for name, code in returncodes.items() if code not in {0, REQUEUE_EXIT_CODE}]
        if not failed:
            return "checkpoint", []
        texts: dict[str, str] = {}
        for name in failed:
            path = self._training_log_path(name)
            try:
                texts[name] = path.read_text(encoding="utf-8", errors="replace")[-2_000_000:]
            except OSError:
                texts[name] = ""
        oom_patterns = (
            "out of memory",
            "hip out of memory",
            "memory allocation",
            "status=253",
        )
        if all(
            returncodes[name] == 253
            or any(pattern in texts[name].lower() for pattern in oom_patterns)
            for name in failed
        ):
            return "measured_oom", failed
        transient_patterns = [
            str(item).lower()
            for item in self.config.raw["autonomy"]["recognized_transient_patterns"]
        ]
        if all(
            any(pattern in texts[name].lower() for pattern in transient_patterns)
            for name in failed
        ):
            return "transient_runtime", failed
        return "trainer_failure", failed

    def validate_training_telemetry(
        self,
        profiles: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        required = {
            "tokens_per_second",
            "estimated_train_flops",
            "step_time_s",
            "mfu",
            "all_to_all_bytes",
            "all_to_all_seconds",
            "overflow_drop_tokens",
            "expert_load_cv",
            "loss",
            "global_token_cursor",
        }
        families: dict[str, Any] = {}
        for family in self.config.families:
            directory = (
                self.campaign_root
                / self.config.raw["training"]["output_subdirectory"]
                / family.name
                / "telemetry"
            )
            paths = sorted(directory.glob("rank-*.jsonl"))
            if len(paths) != family.world_size:
                raise RuntimeError(
                    f"{family.name} telemetry covers {len(paths)}/{family.world_size} ranks"
                )
            last_rows: list[dict[str, Any]] = []
            files: list[dict[str, Any]] = []
            for path in paths:
                last = ""
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            last = line
                if not last:
                    raise RuntimeError(f"Empty trainer telemetry: {path}")
                row = json.loads(last)
                missing = sorted(required - row.keys())
                if missing:
                    raise RuntimeError(
                        f"Trainer telemetry {path.name} is missing {missing}"
                    )
                for key in required - {"global_token_cursor"}:
                    value = row[key]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise RuntimeError(
                            f"Trainer telemetry {path.name} has invalid {key}"
                        )
                if (
                    float(row["tokens_per_second"]) <= 0
                    or float(row["step_time_s"]) <= 0
                    or float(row["estimated_train_flops"]) <= 0
                    or not 0 <= float(row["mfu"]) <= 1.25
                    or float(row["all_to_all_bytes"]) <= 0
                    or float(row["all_to_all_seconds"]) <= 0
                    or int(row["overflow_drop_tokens"]) != 0
                    or int(row["global_token_cursor"]) < 1_000_000_000_000
                ):
                    raise RuntimeError(
                        f"Trainer telemetry completion gate failed: {path.name}"
                    )
                last_rows.append(row)
                files.append(
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "last_row_sha256": json_sha256(row),
                    }
                )
            throughputs = [
                float(row["tokens_per_second"]) for row in last_rows
            ]
            global_throughput = sum(throughputs) / len(throughputs)
            maximum_relative_deviation = max(
                abs(value - global_throughput)
                / max(abs(global_throughput), 1.0e-12)
                for value in throughputs
            )
            if maximum_relative_deviation > 0.10:
                raise RuntimeError(
                    f"{family.name} ranks disagree on global tokens_per_second: "
                    f"mean={global_throughput}, "
                    f"maximum_relative_deviation={maximum_relative_deviation}"
                )
            families[family.name] = {
                "rank_count": len(paths),
                "mfu_mean": sum(float(row["mfu"]) for row in last_rows)
                / len(last_rows),
                # Every rank reports the same global-token numerator. Summing
                # it would inflate throughput by world size.
                "tokens_per_second_global_mean": global_throughput,
                "tokens_per_second_maximum_relative_deviation": (
                    maximum_relative_deviation
                ),
                "all_to_all_bytes_sum": sum(
                    float(row["all_to_all_bytes"]) for row in last_rows
                ),
                "all_to_all_seconds_mean": sum(
                    float(row["all_to_all_seconds"]) for row in last_rows
                )
                / len(last_rows),
                "maximum_expert_load_cv": max(
                    float(row["expert_load_cv"]) for row in last_rows
                ),
                "minimum_global_token_cursor": min(
                    int(row["global_token_cursor"]) for row in last_rows
                ),
                "profile_sha256": profiles[family.name]["profile_sha256"],
                "files": files,
            }
        summary = {
            "schema": "metis.portage-training-telemetry-summary/v1",
            "created_at": utc_now(),
            "families": families,
            "overflow_drop_tokens": 0,
        }
        summary["summary_sha256"] = json_sha256(summary)
        atomic_write_json(
            self.campaign_root / "telemetry" / "training-summary.json",
            summary,
        )
        return summary

    def run(self) -> int:
        if os.environ.get("SLURM_NNODES") and int(os.environ["SLURM_NNODES"]) != 128:
            raise RuntimeError("Family supervisor requires one 128-node allocation")
        signal.signal(signal.SIGUSR1, self.request_checkpoint)
        signal.signal(signal.SIGTERM, self.request_checkpoint)
        startup_phase = "audit"
        try:
            for family in self.config.families:
                self.audit(family)
                if self.signal_requested.is_set():
                    self.write_requeue_marker(
                        returncodes={},
                        require_checkpoint=False,
                        classification=f"signal_during_{startup_phase}",
                    )
                    return REQUEUE_EXIT_CODE
            startup_phase = "autotune"
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_profiles = {
                    family.name: executor.submit(self.tune, family)
                    for family in self.config.families
                }
                profiles = {
                    name: future.result()
                    for name, future in future_profiles.items()
                }
            if self.signal_requested.is_set():
                self.write_requeue_marker(
                    returncodes={},
                    require_checkpoint=False,
                    classification=f"signal_during_{startup_phase}",
                )
                return REQUEUE_EXIT_CODE
            startup_phase = "topology_race"
            profiles = self.topology_race(profiles)
        except BaseException:
            if self.signal_requested.is_set():
                self.write_requeue_marker(
                    returncodes={},
                    require_checkpoint=False,
                    classification=f"signal_during_{startup_phase}",
                )
                return REQUEUE_EXIT_CODE
            raise
        if self.signal_requested.is_set():
            self.write_requeue_marker(
                returncodes={},
                require_checkpoint=False,
                classification=f"signal_during_{startup_phase}",
            )
            return REQUEUE_EXIT_CODE
        self.node_snapshot("before-training")
        monitor = self.start_monitor()
        started = utc_now()
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    family.name: executor.submit(
                        self.train_with_deferred_materialization,
                        family,
                        profiles[family.name],
                    )
                    for family in self.config.families
                }
                reverse = {future: name for name, future in futures.items()}
                pending = set(reverse)
                returncodes: dict[str, int] = {}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    saw_failure = False
                    for future in done:
                        name = reverse[future]
                        returncodes[name] = future.result()
                        if returncodes[name] not in {0}:
                            saw_failure = True
                    if saw_failure and pending:
                        self.checkpoint_running()
                        done_rest, pending = wait(pending)
                        for future in done_rest:
                            returncodes[reverse[future]] = future.result()
        finally:
            self.stop_monitor(monitor)
        resume_codes_safe = all(
            code in {0, REQUEUE_EXIT_CODE} for code in returncodes.values()
        )
        if resume_codes_safe and (
            self.signal_requested.is_set()
            or any(code == REQUEUE_EXIT_CODE for code in returncodes.values())
        ):
            self.write_requeue_marker(
                returncodes=returncodes,
                require_checkpoint=True,
                classification="checkpoint_signal",
            )
            return REQUEUE_EXIT_CODE
        if any(code != 0 for code in returncodes.values()):
            classification, failed_families = self.classify_failure(returncodes)
            restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
            maximum_restarts = int(
                self.config.raw["autonomy"]["maximum_automatic_restarts"]
            )
            if (
                classification in {"measured_oom", "transient_runtime"}
                and restart_count < maximum_restarts
            ):
                try:
                    checkpoint_states = self.checkpoint_states(
                        require_checkpoint=False
                    )
                    migrations: dict[str, Any] = {}
                    stage_batch_migrations: dict[str, Any] = {}
                    if classification == "measured_oom":
                        family_by_name = {
                            family.name: family
                            for family in self.config.families
                        }
                        for name in failed_families:
                            family = family_by_name[name]
                            stage_migration = (
                                self.revise_posttraining_batch_after_oom(
                                    family=family,
                                    restart_count=restart_count,
                                )
                            )
                            if stage_migration is not None:
                                stage_batch_migrations[name] = stage_migration
                                continue
                            if (
                                self._output_for(family)
                                / "BASE_PRETRAINING_RECEIPT.json"
                            ).is_file():
                                raise RuntimeError(
                                    f"{name} OOM occurred after base pretraining "
                                    "without an exact stage OOM request"
                                )
                            revised, migration = self.revise_profile_after_oom(
                                family=family,
                                profile=profiles[name],
                                checkpoint=checkpoint_states[name],
                            )
                            profiles[name] = revised
                            if migration is not None:
                                migrations[name] = {
                                    "receipt_sha256": migration[
                                        "receipt_sha256"
                                    ],
                                    "old_profile_sha256": migration[
                                        "old_profile_sha256"
                                    ],
                                    "new_profile_sha256": migration[
                                        "new_profile_sha256"
                                    ],
                                }
                    self.write_requeue_marker(
                        returncodes=returncodes,
                        require_checkpoint=False,
                        classification=classification,
                        extra={
                            "restart_count": restart_count,
                            "maximum_restarts": maximum_restarts,
                            "checkpoints": checkpoint_states,
                            "profile_migrations": migrations,
                            "posttraining_batch_migrations": (
                                stage_batch_migrations
                            ),
                        },
                    )
                    return REQUEUE_EXIT_CODE
                except Exception as exc:
                    classification = (
                        f"{classification}_unsafe_recovery: "
                        f"{type(exc).__name__}: {exc}"
                    )
            failure = {
                "schema": "metis.portage-family-failure/v1",
                "created_at": utc_now(),
                "returncodes": returncodes,
                "classification": classification,
                "restart_count": restart_count,
                "automatic_restart_safe": False,
            }
            failure["failure_sha256"] = json_sha256(failure)
            atomic_write_json(self.campaign_root / "failure.json", failure)
            return max(returncodes.values())
        telemetry_summary = self.validate_training_telemetry(profiles)
        self.node_snapshot("after-training")
        completion = {
            "schema": "metis.portage-campaign-complete/v1",
            "started_at": started,
            "completed_at": utc_now(),
            "families": {
                family.name: {
                    "output": str(
                        self.campaign_root
                        / self.config.raw["training"]["output_subdirectory"]
                        / family.name
                    ),
                    "profile_sha256": profiles[family.name]["profile_sha256"],
                }
                for family in self.config.families
            },
            "release_marker_sha256": self.release_marker["marker_sha256"],
            "telemetry_summary_sha256": telemetry_summary["summary_sha256"],
            "git_commit": self.git_commit,
        }
        completion["completion_sha256"] = json_sha256(completion)
        atomic_write_json(self.campaign_root / "COMPLETE.json", completion)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autotune and supervise simultaneous Praxis/Logos training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    config = load_portage_config(args.config)
    root = Path(args.campaign_root).expanduser().resolve()
    try:
        root.relative_to(config.state_root)
    except ValueError as exc:
        raise RuntimeError("Campaign root escapes the configured state root") from exc
    return FamilySupervisor(config, root).run()


if __name__ == "__main__":
    raise SystemExit(main())
