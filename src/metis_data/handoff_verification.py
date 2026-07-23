from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .handoff import HANDOFF_SCHEMA, _json_digest, _sha256_file
from .state import StateStore, utc_now


ARTIFACT_MARKER_SCHEMA = "metis.handoff-artifact-verification/v2"
FINAL_MARKER_SCHEMA = "metis.handoff-deep-verification/v2"
FINAL_MARKER_NAME = "HANDOFF_VERIFIED.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _current_handoff(profile: dict[str, Any], state: StateStore) -> tuple[dict[str, Any], Path, Path]:
    handoff = state.read("ACQUISITION_READY.json")
    if not isinstance(handoff, dict):
        raise RuntimeError("ACQUISITION_READY.json is missing; finish acquisition on login2 first")
    if handoff.get("schema") != HANDOFF_SCHEMA:
        raise RuntimeError(f"Unsupported acquisition handoff schema: {handoff.get('schema')}")
    handoff_sha256 = str(handoff.get("handoff_sha256") or "")
    unsigned = {key: value for key, value in handoff.items() if key != "handoff_sha256"}
    if not _SHA256.fullmatch(handoff_sha256) or handoff_sha256 != _json_digest(unsigned):
        raise RuntimeError("ACQUISITION_READY.json failed its self-hash check")

    root = Path(profile["storage"]["lustre_root"]).expanduser().resolve()
    recorded_root = Path(str(handoff.get("lustre_root", ""))).expanduser().resolve()
    if (
        recorded_root != root
        and not profile.get("gates", {}).get("allow_relocated_lustre_root", False)
    ):
        raise RuntimeError(
            f"Rhea sees a different Lustre path ({root}) than login2 recorded ({recorded_root})"
        )
    return handoff, root, recorded_root


def _descriptor(
    *,
    role: str,
    path: Any,
    size: Any,
    sha256: Any,
    kind: Any = None,
    source_id: Any = None,
) -> dict[str, Any]:
    normalized_path = str(path or "")
    normalized_sha256 = str(sha256 or "").lower()
    if not normalized_path or Path(normalized_path).is_absolute():
        raise RuntimeError(f"Frozen handoff artifact has an invalid relative path: {normalized_path!r}")
    if int(size) < 0:
        raise RuntimeError(f"Frozen handoff artifact has an invalid size: {normalized_path}")
    if not _SHA256.fullmatch(normalized_sha256):
        raise RuntimeError(f"Frozen handoff artifact has no valid SHA-256: {normalized_path}")
    payload = {
        "role": role,
        "path": normalized_path,
        "size": int(size),
        "sha256": normalized_sha256,
        "kind": str(kind or "file"),
        "source_id": None if source_id is None else str(source_id),
    }
    payload["artifact_id"] = _json_digest(payload)
    return payload


def frozen_handoff_artifacts(
    profile: dict[str, Any],
    state: StateStore,
) -> list[dict[str, Any]]:
    """Return the complete immutable byte-verification inventory.

    This function validates only the small handoff document. It deliberately
    does not read artifact contents, so it is safe to call before submitting
    the Rhea array even when the acquisition is multiple terabytes.
    """

    handoff, _, _ = _current_handoff(profile, state)
    acquisition = handoff.get("artifacts")
    if not isinstance(acquisition, list):
        raise RuntimeError("Acquisition handoff artifact inventory is invalid")
    if int(handoff.get("artifact_count", -1)) != len(acquisition):
        raise RuntimeError("Acquisition handoff artifact count is inconsistent")

    records: list[dict[str, Any]] = []
    for item in acquisition:
        if not isinstance(item, dict):
            raise RuntimeError("Acquisition handoff contains an invalid artifact record")
        records.append(
            _descriptor(
                role="acquisition",
                path=item.get("path"),
                size=item.get("size", -1),
                sha256=item.get("sha256"),
                kind=item.get("kind"),
                source_id=item.get("source_id"),
            )
        )

    holdouts = handoff.get("holdouts")
    if not isinstance(holdouts, dict):
        raise RuntimeError("Acquisition handoff is missing evaluation holdouts")
    records.extend(
        [
            _descriptor(
                role="evaluation_holdouts",
                path=holdouts.get("path"),
                size=holdouts.get("size", -1),
                sha256=holdouts.get("sha256"),
                kind="holdout_jsonl",
            ),
            _descriptor(
                role="evaluation_holdout_report",
                path=holdouts.get("report_path"),
                size=holdouts.get("report_size", -1),
                sha256=holdouts.get("report_sha256"),
                kind="holdout_provenance",
            ),
        ]
    )

    common_crawl = handoff.get("common_crawl_opt_out")
    if common_crawl is not None:
        if not isinstance(common_crawl, dict):
            raise RuntimeError("Common Crawl opt-out inventory is invalid")
        artifacts = common_crawl.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "snapshot",
            "rules",
            "metadata",
            "latest",
        }:
            raise RuntimeError("Common Crawl opt-out handoff artifacts are incomplete")
        for name, item in sorted(artifacts.items()):
            if not isinstance(item, dict):
                raise RuntimeError(f"Common Crawl opt-out {name} artifact is invalid")
            records.append(
                _descriptor(
                    role=f"common_crawl_opt_out:{name}",
                    path=item.get("path"),
                    size=item.get("size", -1),
                    sha256=item.get("sha256"),
                    kind="policy_artifact",
                )
            )

    records.sort(key=lambda row: (row["path"], row["role"], row["artifact_id"]))
    seen_paths: dict[str, dict[str, Any]] = {}
    for record in records:
        prior = seen_paths.get(record["path"])
        if prior is not None:
            raise RuntimeError(
                "Frozen handoff contains a duplicate physical artifact path: "
                f"{record['path']} ({prior['role']} and {record['role']})"
            )
        seen_paths[record["path"]] = record
    return records


def _artifact_path(root: Path, descriptor: dict[str, Any]) -> Path:
    path = (root / str(descriptor["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Frozen handoff artifact escapes the Rhea data root: {path}") from exc
    return path


def _marker_path(state: StateStore, index: int, descriptor: dict[str, Any]) -> Path:
    return state.path(
        "completed",
        "handoff_signature",
        f"task-{index:08d}-{descriptor['artifact_id'][:16]}.json",
    )


def _validate_artifact_marker(
    marker: dict[str, Any],
    *,
    handoff_sha256: str,
    index: int,
    descriptor: dict[str, Any],
    path: Path,
    check_stat: bool,
) -> dict[str, Any]:
    if marker.get("schema") != ARTIFACT_MARKER_SCHEMA:
        raise RuntimeError(f"Deep-verification marker {index} has an unsupported schema")
    marker_sha256 = str(marker.get("marker_sha256") or "")
    unsigned = {key: value for key, value in marker.items() if key != "marker_sha256"}
    if marker_sha256 != _json_digest(unsigned):
        raise RuntimeError(f"Deep-verification marker {index} failed its self-hash check")
    expected = {
        "handoff_sha256": handoff_sha256,
        "artifact_index": index,
        "artifact_id": descriptor["artifact_id"],
        "path": descriptor["path"],
        "size": descriptor["size"],
        "sha256": descriptor["sha256"],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RuntimeError(
                f"Deep-verification marker {index} is not bound to the current handoff artifact"
            )
    if check_stat:
        if not path.is_file():
            raise RuntimeError(f"Deep-verified artifact is now missing: {path}")
        stat = path.stat()
        if (
            stat.st_size != int(marker.get("size", -1))
            or stat.st_mtime_ns != int(marker.get("observed_mtime_ns", -1))
            or stat.st_ino != int(marker.get("observed_inode", -1))
            or stat.st_ctime_ns != int(marker.get("observed_ctime_ns", -1))
        ):
            raise RuntimeError(f"Artifact changed after deep verification: {path}")
    return marker


def verify_handoff_artifact(
    profile: dict[str, Any],
    state: StateStore,
    artifact_index: int,
) -> dict[str, Any]:
    """Hash one frozen artifact and write one immutable completion marker."""

    handoff, root, _ = _current_handoff(profile, state)
    descriptors = frozen_handoff_artifacts(profile, state)
    if artifact_index < 0 or artifact_index >= len(descriptors):
        raise ValueError(f"Unknown handoff artifact index {artifact_index}")
    descriptor = descriptors[artifact_index]
    path = _artifact_path(root, descriptor)
    marker_path = _marker_path(state, artifact_index, descriptor)
    task_id = f"task-{artifact_index:08d}-{descriptor['artifact_id'][:16]}"

    def existing_marker() -> dict[str, Any] | None:
        if not marker_path.is_file():
            return None
        marker = state.read(
            "completed",
            "handoff_signature",
            marker_path.name,
        )
        return _validate_artifact_marker(
            marker,
            handoff_sha256=str(handoff["handoff_sha256"]),
            index=artifact_index,
            descriptor=descriptor,
            path=path,
            check_stat=True,
        )

    existing = existing_marker()
    if existing is not None:
        return {**existing, "resumed": True}

    with state.task_lock("handoff_signature", task_id):
        existing = existing_marker()
        if existing is not None:
            return {**existing, "resumed": True}
        if not path.is_file():
            raise RuntimeError(f"Frozen handoff artifact is missing: {path}")
        before = path.stat()
        if before.st_size != descriptor["size"]:
            raise RuntimeError(f"Frozen handoff artifact size changed: {path}")
        actual_sha256 = _sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise RuntimeError(f"Frozen handoff artifact changed while it was being hashed: {path}")
        if actual_sha256 != descriptor["sha256"]:
            raise RuntimeError(f"Frozen handoff artifact hash changed: {path}")
        marker: dict[str, Any] = {
            "schema": ARTIFACT_MARKER_SCHEMA,
            "handoff_sha256": handoff["handoff_sha256"],
            "artifact_index": artifact_index,
            "artifact_id": descriptor["artifact_id"],
            "role": descriptor["role"],
            "path": descriptor["path"],
            "size": descriptor["size"],
            "sha256": descriptor["sha256"],
            "observed_mtime_ns": after.st_mtime_ns,
            "observed_inode": after.st_ino,
            "observed_ctime_ns": after.st_ctime_ns,
            "verified_at": utc_now(),
        }
        marker["marker_sha256"] = _json_digest(marker)
        if marker_path.exists():
            raise RuntimeError(f"Refusing to overwrite immutable deep-verification marker: {marker_path}")
        state.write(
            "completed",
            "handoff_signature",
            marker_path.name,
            payload=marker,
        )
        return marker


def _marker_inventory(
    profile: dict[str, Any],
    state: StateStore,
    *,
    require_all: bool,
) -> tuple[list[dict[str, Any]], list[int]]:
    handoff, root, _ = _current_handoff(profile, state)
    descriptors = frozen_handoff_artifacts(profile, state)
    markers: list[dict[str, Any]] = []
    missing: list[int] = []
    expected_names: set[str] = set()
    for index, descriptor in enumerate(descriptors):
        marker_path = _marker_path(state, index, descriptor)
        expected_names.add(marker_path.name)
        if not marker_path.is_file():
            missing.append(index)
            continue
        marker = state.read("completed", "handoff_signature", marker_path.name)
        markers.append(
            _validate_artifact_marker(
                marker,
                handoff_sha256=str(handoff["handoff_sha256"]),
                index=index,
                descriptor=descriptor,
                path=_artifact_path(root, descriptor),
                check_stat=True,
            )
        )
    marker_root = state.path("completed", "handoff_signature")
    unexpected = (
        sorted(path.name for path in marker_root.glob("*.json") if path.name not in expected_names)
        if marker_root.is_dir()
        else []
    )
    if unexpected:
        raise RuntimeError(
            "Unexpected immutable handoff-verification marker(s): " + ", ".join(unexpected)
        )
    if require_all and missing:
        raise RuntimeError(
            f"Deep handoff verification is incomplete: {len(missing)} artifact marker(s) are missing"
        )
    return markers, missing


def _marker_manifest_sha256(markers: list[dict[str, Any]]) -> str:
    return _json_digest(
        [
            {
                "artifact_index": marker["artifact_index"],
                "artifact_id": marker["artifact_id"],
                "marker_sha256": marker["marker_sha256"],
            }
            for marker in sorted(markers, key=lambda item: int(item["artifact_index"]))
        ]
    )


def _validate_final_marker(
    marker: dict[str, Any],
    *,
    handoff_sha256: str,
    artifact_count: int,
    artifact_bytes: int,
    marker_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if marker.get("schema") != FINAL_MARKER_SCHEMA:
        raise RuntimeError("HANDOFF_VERIFIED.json has an unsupported schema")
    verification_sha256 = str(marker.get("verification_sha256") or "")
    unsigned = {key: value for key, value in marker.items() if key != "verification_sha256"}
    if verification_sha256 != _json_digest(unsigned):
        raise RuntimeError("HANDOFF_VERIFIED.json failed its self-hash check")
    if (
        marker.get("handoff_sha256") != handoff_sha256
        or int(marker.get("artifact_count", -1)) != artifact_count
        or int(marker.get("artifact_bytes", -1)) != artifact_bytes
    ):
        raise RuntimeError("HANDOFF_VERIFIED.json is not bound to the current acquisition handoff")
    if (
        marker_manifest_sha256 is not None
        and marker.get("marker_manifest_sha256") != marker_manifest_sha256
    ):
        raise RuntimeError("HANDOFF_VERIFIED.json does not match the artifact-marker inventory")
    return marker


def reduce_handoff_verification(
    profile: dict[str, Any],
    state: StateStore,
) -> dict[str, Any]:
    """Reduce all immutable array markers into the normalization release gate."""

    handoff, _, _ = _current_handoff(profile, state)
    descriptors = frozen_handoff_artifacts(profile, state)
    markers, _ = _marker_inventory(profile, state, require_all=True)
    marker_manifest_sha256 = _marker_manifest_sha256(markers)
    artifact_bytes = sum(int(item["size"]) for item in descriptors)
    final_path = state.path(FINAL_MARKER_NAME)
    if final_path.is_file():
        return _validate_final_marker(
            state.read(FINAL_MARKER_NAME),
            handoff_sha256=str(handoff["handoff_sha256"]),
            artifact_count=len(descriptors),
            artifact_bytes=artifact_bytes,
            marker_manifest_sha256=marker_manifest_sha256,
        )
    payload: dict[str, Any] = {
        "schema": FINAL_MARKER_SCHEMA,
        "handoff_sha256": handoff["handoff_sha256"],
        "artifact_count": len(descriptors),
        "artifact_bytes": artifact_bytes,
        "marker_manifest_sha256": marker_manifest_sha256,
        "verified_at": utc_now(),
    }
    payload["verification_sha256"] = _json_digest(payload)
    if final_path.exists():
        raise RuntimeError(f"Refusing to overwrite immutable deep-verification marker: {final_path}")
    state.write(FINAL_MARKER_NAME, payload=payload)
    state.complete("handoff_verify", "task-000000", payload)
    return payload


def handoff_verification_plan(
    profile: dict[str, Any],
    state: StateStore,
) -> dict[str, Any]:
    """Return missing array work without reading artifact contents."""

    handoff, _, _ = _current_handoff(profile, state)
    descriptors = frozen_handoff_artifacts(profile, state)
    markers, missing = _marker_inventory(profile, state, require_all=False)
    final = state.read(FINAL_MARKER_NAME)
    complete = False
    if final is not None:
        if missing:
            raise RuntimeError("HANDOFF_VERIFIED.json exists but artifact markers are incomplete")
        _validate_final_marker(
            final,
            handoff_sha256=str(handoff["handoff_sha256"]),
            artifact_count=len(descriptors),
            artifact_bytes=sum(int(item["size"]) for item in descriptors),
            marker_manifest_sha256=_marker_manifest_sha256(markers),
        )
        complete = True
    return {
        "handoff_sha256": handoff["handoff_sha256"],
        "artifact_count": len(descriptors),
        "artifact_bytes": sum(int(item["size"]) for item in descriptors),
        "missing_indices": missing,
        "complete": complete,
    }


def require_deep_handoff_verified(
    profile: dict[str, Any],
    state: StateStore,
) -> dict[str, Any]:
    """Cheap per-normalization-task gate; the reducer did the expensive work."""

    handoff, _, _ = _current_handoff(profile, state)
    descriptors = frozen_handoff_artifacts(profile, state)
    markers, _ = _marker_inventory(profile, state, require_all=True)
    marker = state.read(FINAL_MARKER_NAME)
    if marker is None:
        raise RuntimeError(
            "Deep acquisition-handoff verification is incomplete; normalization is blocked"
        )
    return _validate_final_marker(
        marker,
        handoff_sha256=str(handoff["handoff_sha256"]),
        artifact_count=len(descriptors),
        artifact_bytes=sum(int(item["size"]) for item in descriptors),
        marker_manifest_sha256=_marker_manifest_sha256(markers),
    )


def require_verified_build_input(
    profile: dict[str, Any],
    state: StateStore,
    file_record: dict[str, Any],
) -> dict[str, Any]:
    """Bind one normalization task to its current deep-verification marker."""

    require_deep_handoff_verified(profile, state)
    handoff, root, _ = _current_handoff(profile, state)
    relative = str(file_record.get("relative_path") or "")
    digest = str(file_record.get("sha256") or "")
    size = int(file_record.get("size", -1))
    descriptors = frozen_handoff_artifacts(profile, state)
    matches = [
        (index, descriptor)
        for index, descriptor in enumerate(descriptors)
        if descriptor["role"] == "acquisition"
        and descriptor["path"] == relative
        and descriptor["sha256"] == digest
        and int(descriptor["size"]) == size
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Build input is not uniquely bound to the frozen handoff: {relative!r}"
        )
    index, descriptor = matches[0]
    path = _artifact_path(root, descriptor)
    marker_path = _marker_path(state, index, descriptor)
    if not marker_path.is_file():
        raise RuntimeError(f"Deep-verification marker is missing for build input: {relative}")
    marker = state.read("completed", "handoff_signature", marker_path.name)
    _validate_artifact_marker(
        marker,
        handoff_sha256=str(handoff["handoff_sha256"]),
        index=index,
        descriptor=descriptor,
        path=path,
        check_stat=True,
    )
    return {
        "path": str(path),
        "size": size,
        "sha256": digest,
        "mtime_ns": int(marker["observed_mtime_ns"]),
        "inode": int(marker["observed_inode"]),
        "ctime_ns": int(marker["observed_ctime_ns"]),
        "artifact_id": descriptor["artifact_id"],
        "marker_sha256": marker["marker_sha256"],
    }


def verify_build_input_after_read(integrity: dict[str, Any]) -> None:
    """Refuse to publish normalized output if its source moved or changed."""

    path = Path(str(integrity["path"]))
    if not path.is_file():
        raise RuntimeError(f"Build input disappeared during normalization: {path}")
    before = path.stat()
    actual_sha256 = _sha256_file(path)
    after = path.stat()
    expected_stat = (
        int(integrity["size"]),
        int(integrity["mtime_ns"]),
        int(integrity["inode"]),
        int(integrity["ctime_ns"]),
    )
    if (
        (before.st_size, before.st_mtime_ns, before.st_ino, before.st_ctime_ns)
        != expected_stat
        or (after.st_size, after.st_mtime_ns, after.st_ino, after.st_ctime_ns)
        != expected_stat
        or actual_sha256 != integrity["sha256"]
    ):
        raise RuntimeError(f"Build input changed while normalization owned it: {path}")
