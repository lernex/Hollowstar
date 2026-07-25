from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRETRAINING_CONTRACT = REPOSITORY_ROOT / "configs" / "metis16" / "pretraining.yaml"


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in {resolved}")
    payload["_path"] = str(resolved)
    return payload


def _require_keys(mapping: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(mapping))
    if missing:
        raise RuntimeError(f"{label} is missing required keys: {', '.join(missing)}")


def _require_positive_ints(values: Any, label: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"{label} must be a non-empty list")
    cooked = [int(value) for value in values]
    if any(value <= 0 for value in cooked) or len(set(cooked)) != len(cooked):
        raise RuntimeError(f"{label} must contain unique positive integers")
    return cooked


def validate_family_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "metis.model-family/v1":
        raise RuntimeError("Family manifest must use schema metis.model-family/v1")
    family = str(payload.get("family", "")).lower()
    if family not in {"praxis", "logos"}:
        raise RuntimeError("Family manifest family must be praxis or logos")
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise RuntimeError("Family manifest is missing architecture")
    geometry = {**dict(architecture), **dict(payload)}
    _require_keys(
        geometry,
        {
            "vocab_size",
            "sequence_length",
            "d_model",
            "n_layers",
            "latent_dim",
            "n_routed_experts",
            "n_shared_experts",
            "expert_intermediate_dim",
            "max_passes",
        },
        "architecture",
    )
    if int(geometry["vocab_size"]) != 65_536:
        raise RuntimeError("Metis-1.6 family manifests must use exactly 65,536 token IDs")
    if int(geometry["sequence_length"]) != 4_096:
        raise RuntimeError("Metis-1.6 base pretraining sequence length must be 4,096")
    if int(geometry["max_passes"]) != 5:
        raise RuntimeError("Metis-1.6 max passes must be five")
    expected_geometry = {
        "praxis": (2_048, 12, 1_024, 128, 1, 512),
        "logos": (2_560, 20, 1_024, 192, 1, 768),
    }[family]
    observed_geometry = (
        int(geometry["d_model"]),
        int(geometry["n_layers"]),
        int(geometry["latent_dim"]),
        int(geometry["n_routed_experts"]),
        int(geometry["n_shared_experts"]),
        int(geometry["expert_intermediate_dim"]),
    )
    if observed_geometry != expected_geometry:
        raise RuntimeError(
            f"{family} executable geometry is stale: {observed_geometry}"
        )

    topology = payload.get("topology")
    if not isinstance(topology, Mapping):
        raise RuntimeError("Family manifest is missing topology")
    expected = {
        "praxis": (128, 128, 1),
        "logos": (384, 192, 2),
    }[family]
    observed = (
        int(topology.get("world_size", 0)),
        int(topology.get("expert_parallel_size", 0)),
        int(topology.get("expert_replica_count", 0)),
    )
    if observed != expected:
        raise RuntimeError(
            f"{family} topology must be world/EP/replicas={expected}, observed {observed}"
        )

    autotune = payload.get("autotune")
    if not isinstance(autotune, Mapping):
        raise RuntimeError("Family manifest must declare bounded autotune candidates")
    bounds = autotune.get("bounds")
    gates = autotune.get("gates")
    if not isinstance(bounds, Mapping) or not isinstance(gates, Mapping):
        raise RuntimeError("autotune.bounds and autotune.gates must be mappings")
    micro_batches = _require_positive_ints(bounds.get("micro_batch_sizes"), "micro_batch_sizes")
    if micro_batches != sorted(micro_batches, reverse=True):
        raise RuntimeError("micro_batch_sizes must be ordered from largest to smallest")
    _require_positive_ints(bounds.get("grad_accum_steps"), "grad_accum_steps")
    learning_rates = bounds.get("learning_rates")
    if (
        not isinstance(learning_rates, list)
        or not learning_rates
        or any(float(value) <= 0.0 for value in learning_rates)
    ):
        raise RuntimeError("learning_rates must be an explicit non-empty positive list")
    global_batch = bounds.get("global_token_batch")
    if not isinstance(global_batch, Mapping):
        raise RuntimeError("global_token_batch must be a mapping")
    minimum = int(global_batch.get("min", 0))
    maximum = int(global_batch.get("max", 0))
    target = int(global_batch.get("target", 0))
    if not 0 < minimum <= target <= maximum:
        raise RuntimeError("global_token_batch must satisfy 0 < min <= target <= max")
    if not set(bounds.get("precision_profiles", [])) <= {"fp8", "bf16"}:
        raise RuntimeError("precision_profiles may only contain fp8 and bf16")
    if "bf16" not in bounds.get("precision_profiles", []):
        raise RuntimeError("A BF16 numerical reference profile is mandatory")
    overlap_values = {
        ("on" if value is True else "off" if value is False else str(value).lower())
        for value in bounds.get("dispatch_overlap", [])
    }
    if not overlap_values or not overlap_values <= {"on", "off"}:
        raise RuntimeError("dispatch_overlap may only contain on and off")
    ngram_table_modes = bounds.get("ngram_table_modes", [])
    if (
        not isinstance(ngram_table_modes, list)
        or not ngram_table_modes
        or not set(str(value) for value in ngram_table_modes)
        <= {"replicated", "row_sharded"}
    ):
        raise RuntimeError(
            "ngram_table_modes must contain bounded replicated/row_sharded candidates"
        )
    if not set(bounds.get("compile_modes", [])) <= {
        "none",
        "default",
        "reduce-overhead",
        "max-autotune",
    }:
        raise RuntimeError("compile_modes contains an unsupported torch.compile mode")
    _require_keys(
        gates,
        {
            "max_hbm_fraction",
            "max_fp8_loss_relative_error",
            "max_ngram_layout_loss_relative_error",
            "max_update_to_weight_ratio",
            "max_grad_norm",
        },
        "autotune.gates",
    )
    if not 0.0 < float(gates["max_hbm_fraction"]) < 1.0:
        raise RuntimeError("max_hbm_fraction must be between zero and one")
    return dict(payload)


def load_family_manifest(path: str | Path) -> dict[str, Any]:
    return validate_family_manifest(load_yaml(path))


@dataclass(frozen=True)
class AutotuneSelection:
    family: str
    micro_batch: int
    grad_accum: int
    learning_rate: float
    precision_profile: str
    compile_mode: str
    dispatch_overlap: bool
    ngram_table_mode: str
    profile_sha256: str
    environment_sha256: str
    release_marker_sha256: str
    precision_role_plan_sha256: str = ""
    precision_role_inventory_sha256: str = ""
    measured_precision_role_map: tuple[tuple[str, str], ...] = ()

    @property
    def measured_role_dtypes(self) -> dict[str, str]:
        return dict(self.measured_precision_role_map)

    def execution_role_dtypes(self) -> dict[str, str]:
        if self.precision_profile == "bf16":
            return {
                role: "bf16" for role, _dtype in self.measured_precision_role_map
            }
        return self.measured_role_dtypes


def load_autotune_selection(
    path: str | Path,
    *,
    family_manifest: Mapping[str, Any],
    expected_environment_sha256: str | None = None,
) -> AutotuneSelection:
    from .model_config import Metis16Config
    from .precision_plan import (
        measured_role_dtype_map,
        validate_precision_role_plan,
    )

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema") != "metis.portage-autotune/v1":
        raise RuntimeError("Autotune profile must use schema metis.portage-autotune/v1")
    unsigned = {key: value for key, value in payload.items() if key != "profile_sha256"}
    observed_hash = canonical_json_sha256(unsigned)
    if payload.get("profile_sha256") != observed_hash:
        raise RuntimeError("Autotune profile self-hash is invalid")
    family = str(payload.get("family", "")).lower()
    if family != str(family_manifest["family"]).lower():
        raise RuntimeError("Autotune profile family does not match model manifest")
    environment_sha = str(
        payload.get("environment_sha256", payload.get("inventory_fingerprint", ""))
    )
    if expected_environment_sha256 and environment_sha != expected_environment_sha256:
        raise RuntimeError("Autotune profile was measured on a different Portage environment")

    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise RuntimeError("Autotune profile has no selected candidate")
    bounds = family_manifest["autotune"]["bounds"]
    micro_batch = int(selected.get("micro_batch", selected.get("micro_batch_size", 0)))
    grad_accum = int(selected.get("grad_accum", selected.get("grad_accum_steps", 0)))
    learning_rate = float(selected.get("learning_rate", 0.0))
    precision_profile = str(selected.get("precision_profile", ""))
    compile_mode = str(selected.get("compile_mode", ""))
    overlap_raw = selected.get("dispatch_overlap")
    dispatch_overlap = (
        overlap_raw
        if isinstance(overlap_raw, bool)
        else str(overlap_raw).lower() == "on"
    )
    ngram_table_mode = str(selected.get("ngram_table_mode", ""))
    if micro_batch not in [int(value) for value in bounds["micro_batch_sizes"]]:
        raise RuntimeError("Autotune profile selected an out-of-bounds micro batch")
    if grad_accum not in [int(value) for value in bounds["grad_accum_steps"]]:
        raise RuntimeError("Autotune profile selected out-of-bounds gradient accumulation")
    if learning_rate not in [float(value) for value in bounds["learning_rates"]]:
        raise RuntimeError("Autotune profile selected an out-of-bounds learning rate")
    if precision_profile not in bounds["precision_profiles"]:
        raise RuntimeError("Autotune profile selected an out-of-bounds precision")
    allowed_compile_modes = [
        "eager" if str(value) == "none" else str(value)
        for value in bounds["compile_modes"]
    ]
    if compile_mode not in allowed_compile_modes:
        raise RuntimeError("Autotune profile selected an out-of-bounds compile mode")
    if ("on" if dispatch_overlap else "off") not in bounds["dispatch_overlap"]:
        raise RuntimeError("Autotune profile selected out-of-bounds dispatch overlap")
    if ngram_table_mode not in bounds["ngram_table_modes"]:
        raise RuntimeError("Autotune profile selected out-of-bounds N-gram table mode")
    raw_plan = payload.get("precision_role_plan")
    if not isinstance(raw_plan, Mapping):
        raise RuntimeError("Autotune profile has no sealed precision role plan")
    config_payload = {
        key: value for key, value in family_manifest.items() if key != "_path"
    }
    model_config = Metis16Config.from_mapping(config_payload)
    role_plan = validate_precision_role_plan(raw_plan, config=model_config)
    role_plan_sha256 = str(role_plan["plan_sha256"])
    inventory_sha256 = str(role_plan["inventory_sha256"])
    measured_map = measured_role_dtype_map(role_plan)
    if (
        payload.get("precision_role_plan_sha256") != role_plan_sha256
        or payload.get("precision_role_inventory_sha256") != inventory_sha256
        or payload.get("measured_precision_role_map") != measured_map
    ):
        raise RuntimeError("Autotune profile precision role plan binding is stale")
    if precision_profile == "fp8" and "fp8" not in set(measured_map.values()):
        raise RuntimeError("Autotune profile selected FP8 but its measured role map has no FP8 role")
    return AutotuneSelection(
        family=family,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        precision_profile=precision_profile,
        compile_mode=compile_mode,
        dispatch_overlap=dispatch_overlap,
        ngram_table_mode=ngram_table_mode,
        profile_sha256=observed_hash,
        environment_sha256=environment_sha,
        release_marker_sha256=str(payload.get("release_marker_sha256", "")),
        precision_role_plan_sha256=role_plan_sha256,
        precision_role_inventory_sha256=inventory_sha256,
        measured_precision_role_map=tuple(sorted(measured_map.items())),
    )


def _safe_relative_artifact(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise RuntimeError(f"{label} must be a safe relative path")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the release root") from exc
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or is a symlink: {path}")
    return path


def validate_release_marker(
    release_root: str | Path,
    marker_path: str | Path,
    *,
    contract_path: str | Path = PRETRAINING_CONTRACT,
) -> dict[str, Any]:
    """Validate the cheap binding emitted by the distributed deep release audit.

    The marker never replaces the initial distributed byte/hash audit. It avoids
    re-reading roughly two terabytes before both jobs while still binding the
    launch to the exact RELEASE.json, pretraining contract, and SHARDS.jsonl.
    """

    root = Path(release_root).expanduser().resolve()
    marker_file = Path(marker_path).expanduser().resolve()
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    marker_schema = marker.get("schema")
    if marker_schema not in {
        "metis.release-verification-marker/v1",
        "metis.portage-release-verification/v1",
    }:
        raise RuntimeError("Unexpected release verification marker schema")
    unsigned = {key: value for key, value in marker.items() if key != "marker_sha256"}
    if marker.get("marker_sha256") != canonical_json_sha256(unsigned):
        raise RuntimeError("Release verification marker failed its self-hash")
    if marker_schema == "metis.release-verification-marker/v1" and marker.get("ok") is not True:
        raise RuntimeError("Release verification marker is not successful")

    release_path = root / "RELEASE.json"
    if not release_path.is_file():
        raise RuntimeError(f"Release descriptor is missing: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release_unsigned = {key: value for key, value in release.items() if key != "release_sha256"}
    if release.get("release_sha256") != canonical_json_sha256(release_unsigned):
        raise RuntimeError("RELEASE.json failed its self-hash check")
    artifacts = release.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("RELEASE.json has no artifact map")
    shard_manifest = _safe_relative_artifact(
        root, artifacts.get("shard_manifest"), "shard manifest"
    )
    actual = {
        "release_root": str(root),
        "release_json_sha256": sha256_file(release_path),
        "contract_sha256": sha256_file(contract_path),
        "shard_manifest_sha256": sha256_file(shard_manifest),
    }
    marker_root = str(Path(str(marker.get("release_root", ""))).expanduser().resolve())
    if marker_root != actual["release_root"]:
        raise RuntimeError("Release marker points at a different release root")
    aliases = {
        "release_json_sha256": ("release_json_sha256", "release_sha256"),
        "contract_sha256": (
            "training_contract_sha256",
            "contract_sha256",
            "pretraining_contract_sha256",
        ),
        "shard_manifest_sha256": ("shard_manifest_sha256",),
    }
    for field, names in aliases.items():
        observed = next((marker.get(name) for name in names if marker.get(name)), None)
        if observed != actual[field]:
            raise RuntimeError(f"Release marker binding is stale for {field}")
    if marker_schema == "metis.portage-release-verification/v1":
        shard_rows = [
            json.loads(line)
            for line in shard_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        inventory_sha = canonical_json_sha256(
            [
                {
                    "task_index": int(row["task_index"]),
                    "phase": str(row["phase"]),
                    "binary": str(row["binary"]),
                    "binary_sha256": str(row["binary_sha256"]),
                    "index": str(row["index"]),
                    "index_sha256": str(row["index_sha256"]),
                    "tokens": int(row["tokens"]),
                }
                for row in sorted(shard_rows, key=lambda item: int(item["task_index"]))
            ]
        )
        if marker.get("release_self_sha256") != release.get("release_sha256"):
            raise RuntimeError("Release marker self-hash binding is stale")
        if marker.get("shard_inventory_sha256") != inventory_sha:
            raise RuntimeError("Release marker shard inventory binding is stale")
        if int(marker.get("shard_count", -1)) != len(shard_rows):
            raise RuntimeError("Release marker shard count is stale")
        if int(marker.get("total_tokens", -1)) != 1_000_000_000_000:
            raise RuntimeError("Release marker does not attest exactly 1T tokens")
        if marker.get("phase_tokens") != {
            "phase_a": 700_000_000_000,
            "phase_b": 250_000_000_000,
            "phase_c": 50_000_000_000,
        }:
            raise RuntimeError("Release marker phase totals are not exact")
        receipt_hashes = marker.get("task_receipt_sha256s")
        if (
            not isinstance(receipt_hashes, (list, dict))
            or not receipt_hashes
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in (
                    receipt_hashes.values()
                    if isinstance(receipt_hashes, dict)
                    else receipt_hashes
                )
            )
        ):
            raise RuntimeError("Release marker does not bind distributed task receipts")
        if int(marker.get("world_size", 0)) <= 0:
            raise RuntimeError("Release marker has no distributed audit world size")
    elif marker.get("deep_hash_verified") is not True:
        raise RuntimeError("Release marker does not attest a distributed deep hash audit")
    return marker


def require_release_verification(
    release_root: str | Path,
    *,
    contract_path: str | Path = PRETRAINING_CONTRACT,
    marker_path: str | Path | None = None,
    allow_direct_deep_validation: bool = False,
) -> dict[str, Any]:
    marker_path = marker_path or os.environ.get("METIS_RELEASE_VERIFICATION_MARKER")
    if marker_path:
        validate_release_marker(release_root, marker_path, contract_path=contract_path)
        # The descriptor is cheap to parse after its binding has been checked.
        release = json.loads((Path(release_root).resolve() / "RELEASE.json").read_text(encoding="utf-8"))
        return {
            "ok": True,
            "release": release.get("release"),
            "release_root": str(Path(release_root).resolve()),
            "marker": str(Path(marker_path).resolve()),
            "marker_verified": True,
        }
    if not allow_direct_deep_validation:
        raise RuntimeError(
            "Production training requires METIS_RELEASE_VERIFICATION_MARKER from the "
            "distributed deep audit; refusing to re-hash or trust the release implicitly"
        )
    from metis_data.training_contract import validate_training_release

    return validate_training_release(release_root, contract_path)
