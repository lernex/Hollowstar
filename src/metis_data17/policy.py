from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from metis_data.config import load_yaml
from metis_data.datatrove_blocks import (
    _canonical_json_sha256,
    _contamination_manifest_sha256,
    load_contamination_index,
)
from metis_data.freshweb import snapshot_common_crawl_opt_out

from .acquisition import file_lock
from .common import digest_json, read_receipt, sha256_file, utc_now, write_receipt


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"Existing policy artifact differs from its source: {destination}")
        return
    temporary = destination.with_name(destination.name + ".part")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != expected_sha256:
        raise RuntimeError(f"Copied policy artifact failed integrity: {source.name}")
    temporary.replace(destination)
    destination.chmod(0o444)


def import_policy(
    root: Path,
    source_directory: Path,
    *,
    registry_path: Path | None = None,
    refresh_opt_out: bool = True,
) -> dict[str, Any]:
    source_index = source_directory / "index.json"
    original = json.loads(source_index.read_text())
    recorded_registry = Path(original["inputs"]["registry"]["path"])
    registry = registry_path or recorded_registry
    tuning = {
        "minimum_short_matching_ngrams": 0,
        "minimum_code_skeleton_matching_ngrams": 0,
    }
    identity = digest_json({
        "source_manifest_sha256": original["manifest_sha256"],
        "overrides": tuning,
    })
    output = root / "policy" / identity
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(root / "locks" / "policy-import.lock"):
        # Structural index parameters and benchmark content stay unchanged;
        # only already-supported query thresholds are overridden for 1.7.
        load_contamination_index(source_index, benchmark_registry_path=registry)
        for record in original["array_artifacts"].values():
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe source contamination-array path")
            _copy_verified(source_directory / relative, output / relative, record["sha256"])
        for key in ("holdouts", "holdout_report"):
            record = original["inputs"][key]
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Unsafe source holdout path")
            _copy_verified(source_directory / relative, output / relative, record["sha256"])
        copied_registry = output / "eval-holdouts.yaml"
        _copy_verified(registry, copied_registry, original["inputs"]["registry"]["sha256"])
        derived = json.loads(json.dumps(original))
        derived.update(tuning)
        inputs = derived["inputs"]
        inputs["registry"]["path"] = str(copied_registry)
        inputs["registry"]["repository_relative_path"] = None
        inputs["policy_contract"].update(tuning)
        inputs["bundle_sha256"] = _canonical_json_sha256({
            key: value for key, value in inputs.items() if key != "bundle_sha256"
        })
        derived["inputs_sha256"] = _canonical_json_sha256(inputs)
        derived["derivation"] = {
            "source_manifest_sha256": original["manifest_sha256"],
            "source_index_sha256": sha256_file(source_index),
            "overrides": tuning,
            "arrays_changed": False,
        }
        derived["manifest_sha256"] = _contamination_manifest_sha256(derived)
        target_index = output / "index.json"
        if target_index.exists() and json.loads(target_index.read_text()) != derived:
            raise RuntimeError("Immutable derived contamination index changed")
        temporary = output / "index.json.part"
        temporary.write_text(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
        temporary.replace(target_index)
        load_contamination_index(target_index, benchmark_registry_path=copied_registry)
        if refresh_opt_out:
            opt_out = snapshot_common_crawl_opt_out(root / "policy" / "opt-out")
        else:
            raise ValueError("Live 1.7 policy publication requires a current opt-out snapshot")
        result = {
            "schema": "metis17.policy-ready/v1",
            "created_at": utc_now(),
            "decontamination_index": str(target_index),
            "benchmark_registry": str(copied_registry),
            "index_manifest_sha256": derived["manifest_sha256"],
            "holdout_registry_sha256": original["inputs"]["registry"]["sha256"],
            "opt_out_snapshot": opt_out["path"],
            "opt_out_sha256": opt_out["sha256"],
            "opt_out_unparsed_entries": opt_out["unparsed_entries"],
            "policy_overrides": tuning,
        }
        write_receipt(root / "policy" / "CURRENT.json", result)
        return result


def policy_config(root: Path) -> dict[str, Any]:
    path = root / "policy" / "CURRENT.json"
    if not path.exists():
        return {
            "decontamination_index": None,
            "benchmark_registry": None,
            "opt_out_snapshot": None,
            "policy_ready": False,
        }
    result = read_receipt(path)
    snapshot = Path(result["opt_out_snapshot"])
    if not snapshot.is_relative_to(root.resolve()) or sha256_file(snapshot) != result["opt_out_sha256"]:
        raise RuntimeError("Frozen opt-out artifact is missing or changed")
    registry = Path(result["benchmark_registry"])
    if sha256_file(registry) != result["holdout_registry_sha256"]:
        raise RuntimeError("Frozen benchmark registry changed")
    return {
        "decontamination_index": result["decontamination_index"],
        "benchmark_registry": result["benchmark_registry"],
        "opt_out_snapshot": result["opt_out_snapshot"],
        "policy_ready": True,
    }
