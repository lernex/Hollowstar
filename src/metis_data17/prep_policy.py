"""Eligibility policy over saved canonical records, never over remote sources.

Common Crawl always requires a valid opt-out snapshot and checks published
document URLs. Missing URLs retain explicit UNKNOWN coverage unless the source
policy sets ``require_opt_out_url: true``; identifiers are not invented URLs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from metis_data.config import load_yaml
from metis_data.datatrove_blocks import load_contamination_index
from metis_data.decontaminate import benchmark_genealogy_match
from metis_data.freshweb import OptOutPolicy
from metis_data.quality import evaluate_quality

from .common import ObjectSpec, canonical_json, digest_json, sha256_file
from .optout17 import OptOut17Error, OptOutPolicy17, parse_opt_out_registry17 as parse_opt_out_registry
from .prep_readers import PreparationError, published_http_url


ELIGIBILITY_VERSION = "metis17.object-eligibility/v1"


@dataclass(frozen=True)
class EligibilityPolicy:
    descriptor: dict[str, Any]
    profiles: dict[str, Any]
    profile_name: str
    registry: dict[str, Any] | None
    index: Any
    opt_out: OptOutPolicy | None
    pending: tuple[str, ...]


class _RuntimePolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PolicyRuntime:
    profiles: dict[str, Any]
    registry: dict[str, Any] | None
    index: Any
    opt_out: OptOutPolicy | None
    descriptor: dict[str, Any]
    pending: tuple[str, ...]
    stamps: tuple[tuple[Any, ...], ...]


_PREPARED_RUNTIMES: dict[tuple[str | None, ...], _PolicyRuntime] = {}
_RUNTIME_OWNER_PID: int | None = None


def _file_stamp(path: Path) -> tuple[Any, ...]:
    stat = path.stat()
    return str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _artifact_stamps(path: Path, payload: Mapping[str, Any]) -> tuple[Any, ...]:
    paths = [path]
    for record in payload.get("array_artifacts", {}).values():
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe contamination artifact path")
        paths.append(path.parent / relative)
    for name in ("holdouts", "holdout_report"):
        record = payload.get("inputs", {}).get(name, {})
        if record.get("path"):
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe contamination input path")
            paths.append(path.parent / relative)
    return tuple(_file_stamp(path) for path in paths)


@lru_cache(maxsize=4)
def _cached_index(path: str, registry_path: str, stamp: tuple[Any, ...]) -> Any:
    return load_contamination_index(path, benchmark_registry_path=registry_path)


@lru_cache(maxsize=8)
def _cached_yaml(path: str, digest: str) -> dict[str, Any]:
    result = load_yaml(Path(path))
    if sha256_file(Path(path)) != digest:
        raise _RuntimePolicyError("policy_yaml_changed_while_loading")
    return result


@lru_cache(maxsize=4)
def _cached_opt_out(path: str, digest: str) -> OptOutPolicy17:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Opt-out snapshot changed while loading")
    return parse_opt_out_registry(payload)


def _runtime_key(config: Mapping[str, Any]) -> tuple[str | None, ...]:
    return tuple(str(Path(config[key]).expanduser().resolve()) if config.get(key) else None
                 for key in ("quality_profiles_path", "benchmark_registry",
                             "decontamination_index", "opt_out_snapshot"))


def _read_runtime(config: Mapping[str, Any], *, include_opt_out: bool) -> _PolicyRuntime:
    path_value = config.get("quality_profiles_path")
    if not path_value:
        raise _RuntimePolicyError("explicit_quality_profile_required")
    profile_path = Path(path_value).expanduser().resolve()
    profiles = _cached_yaml(str(profile_path), sha256_file(profile_path))
    if not isinstance(profiles.get("defaults", {}), dict) or not isinstance(profiles.get("profiles"), dict):
        raise _RuntimePolicyError("invalid_quality_profiles_schema")
    pending: list[str] = []
    descriptor: dict[str, Any] = {}
    stamps: list[tuple[Any, ...]] = [_file_stamp(profile_path)]
    registry = None
    index = None
    registry_path = Path(config["benchmark_registry"]).expanduser().resolve() if config.get("benchmark_registry") else None
    index_path = Path(config["decontamination_index"]).expanduser().resolve() if config.get("decontamination_index") else None
    if registry_path is None or not registry_path.is_file():
        pending.append("benchmark_registry_unavailable")
    else:
        descriptor["benchmark_registry_sha256"] = sha256_file(registry_path)
        registry = _cached_yaml(str(registry_path), descriptor["benchmark_registry_sha256"])
        if registry.get("schema") != "metis.contamination-registry/v2" or not registry.get("benchmarks"):
            raise _RuntimePolicyError("invalid_benchmark_registry")
        stamps.append(_file_stamp(registry_path))
    if index_path is None or not index_path.is_file():
        pending.append("decontamination_index_unavailable")
    elif registry_path is not None and registry is not None:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            descriptor["decontamination_index_sha256"] = sha256_file(index_path)
            artifact_stamps = _artifact_stamps(index_path, payload)
            index = _cached_index(
                str(index_path), str(registry_path),
                (descriptor["decontamination_index_sha256"], descriptor["benchmark_registry_sha256"],
                 *artifact_stamps),
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            raise _RuntimePolicyError("invalid_or_unsealed_decontamination_index") from None
        if (
            index.ngram_size != 13 or index.minimum_matching_ngrams < 1
            or index.minimum_code_matching_ngrams < 1 or index.contiguous_run_minimum < 1
            or index.minimum_short_matching_ngrams != 0
            or index.minimum_code_skeleton_matching_ngrams != 0
        ):
            raise _RuntimePolicyError("decontamination_policy_not_metis17")
        descriptor["decontamination_manifest_sha256"] = payload["manifest_sha256"]
        stamps.extend(artifact_stamps)
    opt_out = None
    if include_opt_out:
        snapshot = Path(config["opt_out_snapshot"]).expanduser().resolve() if config.get("opt_out_snapshot") else None
        if snapshot is None or not snapshot.is_file():
            pending.append("common_crawl_opt_out_snapshot_unavailable")
        else:
            try:
                digest = sha256_file(snapshot)
                opt_out = _cached_opt_out(str(snapshot), digest)
            except OptOut17Error as exc:
                reason = ("empty_or_incomplete_common_crawl_opt_out_snapshot"
                          if exc.reason == "empty_registry" else "invalid_common_crawl_opt_out_snapshot")
                raise _RuntimePolicyError(reason) from None
            except (OSError, UnicodeError, ValueError, RuntimeError):
                raise _RuntimePolicyError("invalid_common_crawl_opt_out_snapshot") from None
            if not opt_out.input_entries or opt_out.unparsed_entries:
                raise _RuntimePolicyError("empty_or_incomplete_common_crawl_opt_out_snapshot")
            descriptor["opt_out_snapshot_sha256"] = digest
            audit = opt_out.audit()
            descriptor["opt_out_parser_version"] = audit["parser_version"]
            descriptor["opt_out_parser_sha256"] = audit["parser_sha256"]
            descriptor["opt_out_effective_rules_sha256"] = audit["effective_rules_sha256"]
            descriptor["opt_out_audit_sha256"] = digest_json(audit)
            stamps.append(_file_stamp(snapshot))
            # cached_property otherwise materializes a separate URL-rule map
            # in every forked worker on its first document.
            _ = opt_out._rules_by_host
    return _PolicyRuntime(profiles, registry, index, opt_out, descriptor, tuple(pending),
                          tuple(dict.fromkeys(stamps)))


def prepare_runtime(config: Mapping[str, Any], *, require_ready: bool = False) -> None:
    """Verify/cache immutable policies in the coordinator before a Linux fork.

    Missing policies stay pending in the inherited snapshot. Call this again
    in the coordinator and replace the pool when those policies become ready.
    No acquired object, normalization output, or corpus inventory is opened.
    """
    global _RUNTIME_OWNER_PID
    if _RUNTIME_OWNER_PID is not None and _RUNTIME_OWNER_PID != os.getpid():
        raise _RuntimePolicyError("prepare_runtime_must_run_before_fork")
    if type(require_ready) is not bool:
        raise ValueError("require_ready must be a boolean")
    runtime = _read_runtime(config, include_opt_out=True)
    if require_ready and runtime.pending:
        raise _RuntimePolicyError("preparation_policies_pending:" + ",".join(runtime.pending))
    _PREPARED_RUNTIMES[_runtime_key(config)] = runtime
    _RUNTIME_OWNER_PID = os.getpid()


def _runtime(config: Mapping[str, Any], *, include_opt_out: bool) -> _PolicyRuntime:
    runtime = _PREPARED_RUNTIMES.get(_runtime_key(config))
    if runtime is None:
        if _RUNTIME_OWNER_PID is not None and _RUNTIME_OWNER_PID != os.getpid():
            raise _RuntimePolicyError("runtime_configuration_not_preloaded_before_fork")
        return _read_runtime(config, include_opt_out=include_opt_out)
    for expected in runtime.stamps:
        try:
            current = _file_stamp(Path(expected[0]))
        except OSError:
            raise _RuntimePolicyError("preloaded_policy_artifact_changed") from None
        if current != expected:
            raise _RuntimePolicyError("preloaded_policy_artifact_changed")
    return runtime


def load_eligibility_policy(spec: ObjectSpec, config: Mapping[str, Any]) -> EligibilityPolicy:
    profile_name = spec.policy.get("quality_profile")
    if not isinstance(profile_name, str):
        raise PreparationError(spec, "explicit_quality_profile_required")
    try:
        runtime = _runtime(config, include_opt_out=bool(spec.policy.get("common_crawl_derived")))
    except _RuntimePolicyError as exc:
        raise PreparationError(spec, str(exc)) from None
    profiles = runtime.profiles
    if profile_name not in profiles["profiles"]:
        raise PreparationError(spec, "unknown_quality_profile")
    selected = profiles["profiles"][profile_name]
    if not isinstance(selected, dict):
        raise PreparationError(spec, "invalid_quality_profile")
    effective = {**profiles.get("defaults", {}), **selected}
    pending = list(runtime.pending)
    descriptor = dict(runtime.descriptor)
    descriptor.update({
        "version": ELIGIBILITY_VERSION,
        "quality_profile": profile_name,
        "quality_profile_sha256": digest_json(effective),
        "source_policy": {
            key: spec.policy.get(key) for key in (
                "license_mode", "collection_license", "allowed_licenses", "common_crawl_derived",
                "generated", "require_opt_out_url", "quality_fail_closed", "allowed_languages",
            )
        },
        "priority": spec.priority,
    })
    license_mode = spec.policy.get("license_mode")
    if license_mode not in {"compilation", "per_record"}:
        raise PreparationError(spec, "explicit_license_mode_required")
    if license_mode == "compilation" and not str(spec.policy.get("collection_license") or "").strip():
        raise PreparationError(spec, "compilation_license_required")
    allowed_licenses = spec.policy.get("allowed_licenses")
    if allowed_licenses is not None and (
        not isinstance(allowed_licenses, list) or not allowed_licenses
        or any(not isinstance(value, str) or not value.strip() for value in allowed_licenses)
    ):
        raise PreparationError(spec, "invalid_allowed_licenses")
    if type(spec.policy.get("common_crawl_derived", False)) is not bool:
        raise PreparationError(spec, "invalid_common_crawl_policy")
    for key in ("generated", "require_opt_out_url", "quality_fail_closed"):
        if key in spec.policy and type(spec.policy[key]) is not bool:
            raise PreparationError(spec, "invalid_boolean_source_policy")
    allowed_languages = spec.policy.get("allowed_languages")
    if allowed_languages is not None and (
        not isinstance(allowed_languages, list) or not allowed_languages
        or any(not isinstance(value, str) or not value for value in allowed_languages)
    ):
        raise PreparationError(spec, "invalid_allowed_languages")
    opt_out = runtime.opt_out
    if spec.policy.get("common_crawl_derived"):
        requires_url = spec.policy.get("require_opt_out_url", False)
        descriptor["opt_out_requires_document_url"] = requires_url
        descriptor["opt_out_missing_url_action"] = "quarantine" if requires_url else "retain_unknown"
    if not spec.policy.get("common_crawl_derived"):
        pending = [reason for reason in pending if reason != "common_crawl_opt_out_snapshot_unavailable"]
        for key in tuple(descriptor):
            if key.startswith("opt_out_"):
                del descriptor[key]
        opt_out = None
    if spec.policy.get("metadata", {}).get("quality_selection_pending") is True:
        pending.append("source_quality_selection_pending")
    descriptor["pending_reasons"] = pending
    return EligibilityPolicy(descriptor, profiles, profile_name, runtime.registry, runtime.index,
                             opt_out, tuple(pending))


def _license_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _urls(value: Any) -> list[str]:
    keys = {"url", "original_url", "canonical_url", "declared_canonical_url", "source_url", "warc_target_uri"}
    found: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for name, nested in node.items():
                if name == "source_defaults":
                    continue
                if name in keys and (url := published_http_url(nested)) is not None:
                    found.append(url)
                elif isinstance(nested, (Mapping, list)):
                    visit(nested)
        elif isinstance(node, list):
            for nested in node:
                visit(nested)

    upstream = value.get("upstream") if isinstance(value, Mapping) else None
    if isinstance(upstream, Mapping):
        visit(upstream)
        for container in ("meta", "metadata"):
            if container in upstream:
                visit(value.get(container))
        for evidence in value.get("normalization_evidence", []):
            if (
                isinstance(evidence, Mapping) and evidence.get("field") == "canonical_url"
                and evidence.get("method") == "upstream_field"
            ):
                canonical = published_http_url(value.get("canonical_url"))
                if canonical is not None:
                    found.append(canonical)
    else:
        visit(value)
    return list(dict.fromkeys(found))


def decide_eligibility(
    record: Mapping[str, Any], spec: ObjectSpec, policy: EligibilityPolicy,
) -> tuple[dict[str, Any] | None, str | None, bool, dict[str, int]]:
    """Return record/reason/quarantine flag and non-sensitive policy counters."""

    if policy.pending or policy.registry is None or policy.index is None:
        raise PreparationError(spec, "eligibility_policy_is_pending")
    metadata = json.loads(record["metadata_json"])
    row = metadata.get("row_index", "-")
    if metadata.get("admission_block"):
        return None, str(metadata["admission_block"]), True, {}
    benchmark = benchmark_genealogy_match(metadata, policy.registry)
    if benchmark:
        return None, "benchmark_genealogy", False, {}
    if metadata.get("publisher_machine_learning_opt_out") is True:
        return None, "publisher_machine_learning_opt_out", False, {}
    license_value = metadata.get("license")
    if spec.policy["license_mode"] == "per_record":
        if not license_value:
            return None, "missing_per_record_license", True, {}
        allowed = spec.policy.get("allowed_licenses")
        if allowed is not None:
            allowed_keys = {_license_key(value) for value in allowed}
            licenses = str(license_value).split(",")
            if any(_license_key(value) not in allowed_keys for value in licenses):
                return None, "per_record_license_not_allowed", False, {}
        metadata["license_evidence_scope"] = "per_record"
    else:
        metadata["collection_license"] = spec.policy["collection_license"]
        metadata["license_evidence_scope"] = "compilation_not_per_record"
    counters: dict[str, int] = {}
    if policy.opt_out is not None:
        urls = _urls(metadata)
        for url in urls:
            candidate = policy.opt_out.reason(url)
            if candidate in {"common_crawl_opt_out_domain", "common_crawl_opt_out_url"}:
                return None, f"final_{candidate}", False, {}
        if not urls:
            # A collection licence or WARC identifier cannot identify which
            # publisher's opt-out rules apply to a generated derivative.
            counters["opt_out_no_published_url"] = 1
            if spec.policy.get("require_opt_out_url", False):
                return None, "missing_opt_out_url", True, counters
        metadata["opt_out_application"] = {
            "snapshot_sha256": policy.opt_out.snapshot_sha256,
            "urls_checked": len(urls),
            "coverage": "published_urls" if urls else "no_published_url",
            "status": "NO_MATCH" if urls else "UNKNOWN",
            "url_required": spec.policy.get("require_opt_out_url", False),
        }
    allowed_languages = spec.policy.get("allowed_languages")
    if allowed_languages is not None and record["language"] not in allowed_languages:
        return None, "language_not_in_source_policy", False, counters
    try:
        decision = evaluate_quality(
            str(record["text"]), profile_name=policy.profile_name, metadata=metadata,
            profiles=policy.profiles, fail_closed=bool(spec.policy.get("quality_fail_closed", True)),
        )
    except (ValueError, TypeError, OverflowError):
        raise PreparationError(spec, "invalid_quality_evidence_or_threshold", row) from None
    if not decision.keep:
        return None, f"quality_{decision.reason}", False, counters
    reason = policy.index.reason(str(record["text"]))
    if reason:
        return None, reason, False, counters
    metadata["eligibility"] = {
        "policy_sha256": digest_json(policy.descriptor),
        "quality_profile": policy.profile_name,
        "benchmark_genealogy": "no_registered_lineage_match",
        "decontamination": "passed",
    }
    result = dict(record)
    result["priority"] = spec.priority
    result["metadata_json"] = canonical_json(metadata)
    return result, None, False, counters
