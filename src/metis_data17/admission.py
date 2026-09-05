"""Source expansion follows real, complete canaries, not download order."""

from __future__ import annotations

import fcntl
import re
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from .common import ObjectSpec, digest_json, read_receipt, under_root, utc_now, write_receipt
from .dedup_locks import require_distributed_locks


def claim(path: Path) -> BinaryIO | None:
    require_distributed_locks(path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def admit_source(
    root: Path, spec: ObjectSpec, normalized: Mapping[str, Any],
    screened: Mapping[str, Mapping[str, Any]], *, generation: str, minimum_acceptance: float,
) -> dict[str, Any]:
    if type(minimum_acceptance) not in (int, float) or not 0 < minimum_acceptance <= 1:
        raise ValueError("Source acceptance must be a fraction in (0, 1]")
    if not isinstance(generation, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", generation) is None:
        raise ValueError("A safe source-canary generation identity is required")
    if (
        normalized.get("status") != "NORMALIZED" or normalized.get("reblock_complete") is not True
        or normalized.get("object_id") != spec.object_id or normalized.get("source_id") != spec.source_id
    ):
        raise RuntimeError("Source admission requires the object's sealed EOF inventory")
    expected = {chunk["chunk_id"] for chunk in normalized["chunks"]}
    if len(expected) != len(normalized["chunks"]) or set(screened) != expected:
        raise RuntimeError("Source admission requires exact coverage of the sealed object")
    intake_only = bool(screened) and all(value["status"] == "SCREENED_FOR_INTAKE" for value in screened.values())
    expected_status = "SCREENED_FOR_INTAKE" if intake_only else "ELIGIBLE"
    if any(value["status"] != expected_status for value in screened.values()):
        raise RuntimeError("Pending screening cannot admit a source")
    covered = sum(value["input_documents"] for value in screened.values())
    if covered != normalized["normalized_documents"]:
        raise RuntimeError("Screened documents do not cover normalization exactly once")
    accepted = sum(value["accepted_documents"] if intake_only else value["eligible_documents"]
                   for value in screened.values())
    total = normalized["input_documents"]
    if not 0 <= accepted <= covered <= total:
        raise RuntimeError("Invalid canary retention accounting")
    normalization_seal = digest_json(normalized)
    eligible_receipts = []
    for chunk_id in sorted(screened):
        summary = screened[chunk_id]
        path = under_root(root, summary["receipt_path"])
        stage = read_receipt(path)
        if (
            stage.get("schema") != ("metis17.intake-screened-chunk/v1" if intake_only else "metis17.prepared-chunk/v1")
            or stage.get("status") != expected_status
            or stage.get("eligible") is not (not intake_only)
            or stage.get("training_ready") is not (not intake_only)
            or stage.get("object_complete") is not True
            or stage.get("object_id") != spec.object_id or stage.get("source_id") != spec.source_id
            or stage.get("chunk_id") != chunk_id
            or stage.get("input_documents") != summary["input_documents"]
            or stage.get("eligible_documents") != summary["eligible_documents"]
            or (intake_only and (
                stage.get("accepted_documents") != summary["accepted_documents"]
                or stage.get("deferred_gates") != ["source_quality_selection_pending"]
                or stage.get("inputs", {}).get("purpose") != "acquisition_compliance_only"
            ))
            or stage.get("object_completion") != {
                "path": normalized["receipt_path"], "receipt_sha256": normalization_seal,
            }
        ):
            raise RuntimeError("Invalid eligible canary receipt or object-completion proof")
        eligible_receipts.append({"path": summary["receipt_path"], "receipt_sha256": digest_json(stage)})
    group = str(spec.policy.get("admission_group", spec.source_id))
    ratio = accepted / total if total else 0.0
    result = {
        "schema": "metis17.source-admission/v1", "admission_group": group,
        "generation": generation, "source_id": spec.source_id,
        "object_id": spec.object_id, "input_documents": total,
        "eligible_documents": 0 if intake_only else accepted,
        "screened_documents": accepted, "acceptance_fraction": ratio,
        "admission_basis": "compliance_screening_quality_deferred" if intake_only else "fully_eligible",
        "minimum_acceptance": minimum_acceptance,
        "normalization_receipt": normalized["receipt_path"],
        "normalization_receipt_sha256": normalization_seal,
        ("intake_receipts" if intake_only else "eligible_receipts"): eligible_receipts,
        "status": "admitted" if accepted and ratio >= minimum_acceptance else "retention_review_required",
        "created_at": utc_now(),
    }
    canary_path = root / "admissions" / "canaries" / generation / f"{spec.object_id}.json"
    write_receipt(canary_path, result)
    if result["status"] == "admitted":
        path = root / "admissions" / f"{digest_json(group)}.json"
        lock = claim(root / "locks" / "source-admission" / f"{digest_json(group)}.flock")
        if lock is not None:
            try:
                if path.exists():
                    previous = read_receipt(path)
                    if previous["admission_group"] != group:
                        raise RuntimeError("Source-admission identity changed")
                else:
                    write_receipt(path, result)
            finally:
                lock.close()
    return result
