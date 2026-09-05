"""Compliance screening for raw-web intake, without granting training eligibility."""

from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from . import prep
from .common import digest_json, read_receipt, sha256_file, utc_now, write_receipt
from .prep_policy import load_eligibility_policy


_CODE_SHA256 = sha256_file(Path(__file__))


def screen_intake_chunk(
    base_chunk_path: Path, output_dir: Path, config: Mapping[str, Any],
) -> dict[str, Any]:
    spec, ready, _ = prep._load_base_chunk(base_chunk_path, config)
    root, output, chunk_bytes, batch_size, _ = prep._settings(spec, output_dir, config)
    policy = load_eligibility_policy(spec, config)
    if policy.pending != ("source_quality_selection_pending",):
        raise ValueError("Intake-only screening requires exactly the deferred source-quality gate")
    inputs = {
        "base_chunk_receipt_sha256": digest_json(ready), "policy": policy.descriptor,
        "code_sha256": _CODE_SHA256, "purpose": "acquisition_compliance_only",
    }
    fingerprint = digest_json(inputs)
    namespace = output / "intake-chunks" / ready["chunk_id"]
    namespace.mkdir(parents=True, exist_ok=True)
    with prep._object_lock(namespace / ".prepare.lock"), prep._storage_quota(
        config, f"intake-screening:{ready['chunk_id']}", namespace,
    ) as quota:
        destination = namespace / "screened" / fingerprint
        filtered_path = destination / "FILTERED_INTAKE.json"
        if filtered_path.exists():
            filtered = read_receipt(filtered_path)
            prep._validate_stage(root, {**filtered, "chunks": filtered["screened_chunks"]}, spec)
        else:
            destination, staging = prep._generation(namespace, "screened", fingerprint, quota)
            try:
                normalized = {
                    "chunks": [ready["chunk"]],
                    "normalized_documents": ready["chunk"]["records"],
                    "input_documents": ready["chunk"]["records"],
                    "input_rows": ready["chunk"]["source_rows"],
                    "rejected": {}, "quarantined": {},
                }
                # Defer only learned/final source selection. All declared
                # licensing, opt-out, hygiene, privacy and benchmark gates run.
                screening_policy = replace(policy, pending=())
                outcome = prep._filter_output(
                    spec, normalized, root, destination, staging,
                    chunk_bytes, batch_size, screening_policy, quota,
                )
                screened = outcome.pop("chunks")
                filtered = {
                    **outcome, "schema": "metis17.intake-screened-chunk/v1",
                    "status": "FILTERED_FOR_INTAKE", "eligible": False, "training_ready": False,
                    "source_id": spec.source_id, "object_id": spec.object_id,
                    "chunk_id": ready["chunk_id"], "inputs": inputs,
                    "input_documents": ready["chunk"]["records"],
                    "screened_documents": outcome["accepted_documents"],
                    "eligible_documents": 0, "chunks": [], "screened_chunks": screened,
                    "deferred_gates": ["source_quality_selection_pending"],
                    "created_at": utc_now(),
                }
                write_receipt(staging / filtered_path.name, filtered)
                os.replace(staging, destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        completion = prep._object_completion(spec, ready, root)
        receipt_path = destination / ("INTAKE_READY.json" if completion else "INTAKE_PENDING.json")
        result = {
            **filtered,
            "status": "SCREENED_FOR_INTAKE" if completion else "SCREENED_INTAKE_PENDING_OBJECT_COMPLETION",
            "object_complete": completion is not None, "object_completion": completion,
            "receipt_path": str(receipt_path.relative_to(root)),
        }
        if receipt_path.exists() and read_receipt(receipt_path) != result:
            raise RuntimeError("Immutable intake-screening receipt changed")
        write_receipt(receipt_path, result)
        return result
