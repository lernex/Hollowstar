from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from metis_data.datatrove_blocks import (
    _canonical_json_sha256,
    _contamination_manifest_sha256,
    load_contamination_index,
)
from .acquisition import file_lock
from .common import digest_json, read_receipt, sha256_file, under_root, utc_now, write_receipt
from .optout17 import (
    OptOut17Error,
    load_opt_out_snapshot17,
    snapshot_common_crawl_opt_out17 as snapshot_common_crawl_opt_out,
)

DECONTAMINATION_THRESHOLDS = (
    "minimum_matching_ngrams", "minimum_short_matching_ngrams",
    "minimum_code_matching_ngrams", "minimum_code_skeleton_matching_ngrams",
    "match_fraction", "contiguous_run_minimum",
)


def _strict_opt_out_state(root: Path, path: str | Path, expected_sha256: str) -> dict[str, Any]:
    snapshot = under_root(root, str(path))
    try:
        parsed = load_opt_out_snapshot17(snapshot, expected_sha256)
    except (OptOut17Error, OSError) as exc:
        raise RuntimeError(f"Frozen 1.7 opt-out policy failed validation: {exc}") from None
    audit = parsed.audit()
    return {
        "opt_out_snapshot": str(snapshot),
        "opt_out_sha256": parsed.snapshot_sha256,
        "opt_out_unparsed_entries": parsed.unparsed_entries,
        "opt_out_parser_version": audit["parser_version"],
        "opt_out_parser_sha256": audit["parser_sha256"],
        "opt_out_effective_rules_sha256": audit["effective_rules_sha256"],
        "opt_out_audit": audit,
    }


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
    if not refresh_opt_out:
        raise ValueError("Live 1.7 policy publication requires a current opt-out snapshot")
    root = root.expanduser().resolve()
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
    with file_lock(root / "locks" / "policy-import.lock"):
        opt_out = snapshot_common_crawl_opt_out(root / "policy" / "opt-out")
        opt_out_state = _strict_opt_out_state(root, opt_out["path"], opt_out["sha256"])
        output.mkdir(parents=True, exist_ok=True)
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
        if target_index.exists():
            if json.loads(target_index.read_text()) != derived:
                raise RuntimeError("Immutable derived contamination index changed")
        else:
            temporary = output / "index.json.part"
            temporary.write_text(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
            temporary.replace(target_index)
        load_contamination_index(target_index, benchmark_registry_path=copied_registry)
        result = {
            "schema": "metis17.policy-ready/v1",
            "created_at": utc_now(),
            "decontamination_index": str(target_index),
            "benchmark_registry": str(copied_registry),
            "index_manifest_sha256": derived["manifest_sha256"],
            "holdout_registry_sha256": original["inputs"]["registry"]["sha256"],
            **opt_out_state,
            "policy_overrides": tuning,
        }
        write_receipt(root / "policy" / "CURRENT.json", result)
        return result


def policy_config(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    path = root / "policy" / "CURRENT.json"
    if not path.exists():
        return {
            "decontamination_index": None,
            "benchmark_registry": None,
            "opt_out_snapshot": None,
            "policy_ready": False,
        }
    result = read_receipt(path)
    opt_out_state = _strict_opt_out_state(root, result["opt_out_snapshot"], result["opt_out_sha256"])
    registry = Path(result["benchmark_registry"])
    if sha256_file(registry) != result["holdout_registry_sha256"]:
        raise RuntimeError("Frozen benchmark registry changed")
    index = json.loads(Path(result["decontamination_index"]).read_text())
    if index.get("manifest_sha256") != result["index_manifest_sha256"]:
        raise RuntimeError("Frozen decontamination index identity changed")
    effective = {key: index[key] for key in DECONTAMINATION_THRESHOLDS}
    run_path = root / "RUN.json"
    if run_path.exists():
        declared = read_receipt(run_path)["config"].get("decontamination", {})
        disagreements = [
            key for key, value in declared.items()
            if key not in effective or type(value) is not type(effective[key]) or value != effective[key]
        ]
        if disagreements:
            raise RuntimeError(
                "Declared decontamination settings disagree with the sealed policy: "
                + ", ".join(sorted(disagreements))
            )
    return {
        "decontamination_index": result["decontamination_index"],
        "benchmark_registry": result["benchmark_registry"],
        **opt_out_state,
        "published_opt_out_unparsed_entries": result.get("opt_out_unparsed_entries"),
        "decontamination_effective": effective,
        "policy_ready": True,
    }
