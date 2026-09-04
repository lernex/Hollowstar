from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models

from metis_data.ngram_canonical import (
    CANONICALIZATION_ALGORITHM,
    CANONICALIZATION_REFERENCE,
    CANONICAL_IDS_BINARY,
    CANONICAL_IDS_MANIFEST,
    CANONICAL_IDS_SCHEMA,
)
from metis_portage.posttraining_builder import (
    BUILD_SPEC_SCHEMA,
    DEEP_RECEIPT_SCHEMA,
    _sealed_base_tokenizer,
    build_posttraining_release,
    posttraining_build_template,
)
from metis_portage.posttraining_release import (
    SEALED_SCHEMA,
    inspect_posttraining_release_index,
    verify_posttraining_release_distributed,
)
from metis_portage.config import load_portage_config
from metis_portage.util import file_sha256, json_sha256


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sealed(
    root: Path,
    *,
    directory: str,
    schema: str,
    tokenizer_sha256: str | None,
) -> Path:
    artifact = root / directory
    artifact.mkdir(parents=True, exist_ok=True)
    payload_path = artifact / "payload.bin"
    payload_path.write_bytes(f"{directory}-payload".encode())
    manifest = {
        "envelope_schema": SEALED_SCHEMA,
        "schema": schema,
        "complete": True,
        "files": [
            {
                "path": payload_path.name,
                "bytes": payload_path.stat().st_size,
                "sha256": file_sha256(payload_path),
            }
        ],
        "metadata": {},
    }
    if tokenizer_sha256 is not None:
        manifest["tokenizer_sha256"] = tokenizer_sha256
    manifest["manifest_sha256"] = json_sha256(manifest)
    path = artifact / "MANIFEST.json"
    _write_json(path, manifest)
    return path


def _base_release(root: Path) -> Path:
    release = root / "releases" / "metis-1.6-data-r1"
    tokenizer_root = release / "tokenizer"
    tokenizer_root.mkdir(parents=True)
    tokenizer = tokenizer_root / "tokenizer.json"
    Tokenizer(
        models.WordLevel(
            vocab={
                f"token_{token_id:05d}": token_id
                for token_id in range(65_536)
            },
            unk_token=None,
        )
    ).save(str(tokenizer))
    tokenizer_sha = file_sha256(tokenizer)
    canonical_ids = tokenizer_root / CANONICAL_IDS_BINARY
    np.arange(65_536, dtype="<u2").tofile(canonical_ids)
    canonical_ids_sha = file_sha256(canonical_ids)
    canonical = {
        "schema": CANONICAL_IDS_SCHEMA,
        "created_at": "2026-07-24T00:00:00Z",
        "algorithm": CANONICALIZATION_ALGORITHM,
        "algorithm_reference": CANONICALIZATION_REFERENCE,
        "normalization_steps": [
            "NFKC",
            "NFD",
            "StripAccents",
            "Lowercase",
            "CollapseAsciiWhitespace",
            "PreserveSingleSpace",
            "Strip",
        ],
        "replacement_character_fallback": "raw_tokenizer_vocab_token",
        "empty_normalization_fallback": "decoded_token",
        "tokenizer_sha256": tokenizer_sha,
        "vocabulary_size": 65_536,
        "entry_count": 65_536,
        "canonical_vocabulary_size": 65_536,
        "minimum_canonical_id": 0,
        "maximum_canonical_id": 65_535,
        "canonical_ids_contiguous": True,
        "dtype": "uint16",
        "endianness": "little",
        "binary": CANONICAL_IDS_BINARY,
        "binary_size_bytes": canonical_ids.stat().st_size,
        "binary_sha256": canonical_ids_sha,
    }
    canonical["manifest_sha256"] = json_sha256(canonical)
    canonical_path = tokenizer_root / CANONICAL_IDS_MANIFEST
    _write_json(canonical_path, canonical)
    descriptor = {
        "schema": "metis.data-release/v2",
        "release": "metis-1.6-data-r1",
        "target_tokens": 1_000_000_000_000,
        "phase_tokens": {
            "phase_a": 700_000_000_000,
            "phase_b": 250_000_000_000,
            "phase_c": 50_000_000_000,
        },
        "token_dtype": "uint16",
        "token_endianness": "little",
        "tokenizer_sha256": tokenizer_sha,
        "ngram_canonical_map_manifest_sha256": file_sha256(canonical_path),
        "ngram_canonical_map_self_sha256": canonical["manifest_sha256"],
        "ngram_canonical_ids_sha256": canonical_ids_sha,
        "tokenizer_contract": {
            "tokenizer_sha256": tokenizer_sha,
            "ngram_canonical_map_self_sha256": canonical["manifest_sha256"],
            "ngram_canonical_ids_sha256": canonical_ids_sha,
        },
        "verification": {"ok": True},
        "artifacts": {
            "tokenizer": f"tokenizer/{tokenizer.name}",
            "ngram_canonical_map": f"tokenizer/{canonical_path.name}",
            "ngram_canonical_ids": f"tokenizer/{canonical_ids.name}",
        },
    }
    descriptor["release_sha256"] = json_sha256(descriptor)
    _write_json(release / "RELEASE.json", descriptor)
    return release


def _fixture(tmp_path: Path) -> tuple[SimpleNamespace, Path, Path]:
    lustre = tmp_path / "lustre"
    base_release = _base_release(lustre)
    release_root = lustre / "releases" / "metis-1.6-posttraining-r1"
    release_root.mkdir(parents=True)
    contract = {
        "schema": "metis.posttraining-pipeline/v1",
        "stages": [
            {
                "id": "static",
                "enabled": True,
                "requirements": [
                    {
                        "name": "data",
                        "env": "STATIC_DATA",
                        "schema": "metis.sft-data/v1",
                        "tokenizer_bound": True,
                    }
                ],
            },
            {
                "id": "on_policy",
                "enabled": True,
                "requirements": [
                    {
                        "name": "rollouts",
                        "env": "ROLLOUTS",
                        "schema": "metis.rlvr-data/v1",
                        "tokenizer_bound": True,
                        "family_bound": True,
                        "checkpoint_bound": True,
                    }
                ],
            },
        ],
    }
    contract_path = tmp_path / "posttraining.yaml"
    import yaml

    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
    config = SimpleNamespace(
        posttraining_release_index=release_root / "RELEASE_INDEX.json",
        posttraining_contract=contract_path,
        lustre_root=lustre.resolve(),
        release_root=base_release,
        families=(
            SimpleNamespace(name="praxis"),
            SimpleNamespace(name="logos"),
        ),
    )

    tokenizer_record, _tokenizer, _payload = _sealed_base_tokenizer(
        config=config,
        release_root=release_root,
    )
    tokenizer_sha = tokenizer_record["manifest_sha256"]
    static_data = _sealed(
        release_root,
        directory="static",
        schema="metis.sft-data/v1",
        tokenizer_sha256=tokenizer_sha,
    )
    executable = release_root / "hooks" / "generate.py"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
    executable.chmod(0o750)

    family_requirements = {
        "static": {
            "data": {
                "state": "sealed",
                "manifest": static_data.relative_to(release_root).as_posix(),
            }
        },
        "on_policy": {
            "rollouts": {
                "state": "deferred",
                "manifest": "generated/FAMILY-rollouts.json",
                "generation_hook": {
                    "executable": executable.relative_to(release_root).as_posix(),
                    "args": ["--mode", "rollouts"],
                    "timeout_seconds": 60,
                    "execution": {"protocol": "distributed_family_v1"},
                    "receipt": "generated/FAMILY-rollouts-receipt.json",
                    "rank_receipts": "generated/FAMILY-rank-receipts",
                },
            }
        },
    }
    families = {}
    for family in ("praxis", "logos"):
        encoded = json.loads(json.dumps(family_requirements))
        encoded["on_policy"]["rollouts"]["manifest"] = encoded["on_policy"][
            "rollouts"
        ]["manifest"].replace("FAMILY", family)
        encoded["on_policy"]["rollouts"]["generation_hook"]["receipt"] = encoded[
            "on_policy"
        ]["rollouts"]["generation_hook"]["receipt"].replace("FAMILY", family)
        encoded["on_policy"]["rollouts"]["generation_hook"]["rank_receipts"] = encoded[
            "on_policy"
        ]["rollouts"]["generation_hook"]["rank_receipts"].replace("FAMILY", family)
        families[family] = {"requirements": encoded}
    spec = {
        "schema": BUILD_SPEC_SCHEMA,
        "posttraining_contract_sha256": file_sha256(contract_path),
        "families": families,
    }
    spec_path = tmp_path / "build.json"
    _write_json(spec_path, spec)
    return config, spec_path, release_root


def test_builder_emits_valid_umbrella_indexes_and_deep_receipt(
    tmp_path: Path,
) -> None:
    config, spec, root = _fixture(tmp_path)
    result = build_posttraining_release(
        config=config,
        spec_path=spec,
        workers=2,
    )
    assert result["ok"] is True
    report = inspect_posttraining_release_index(config)
    assert report["ok"] is True
    deep = json.loads((root / "DEEP_VERIFICATION.json").read_text())
    assert deep["schema"] == DEEP_RECEIPT_SCHEMA
    assert deep["complete"] is True
    assert deep["receipt_sha256"] == json_sha256(
        deep, omit=("receipt_sha256",)
    )
    assert deep["file_count"] == len(deep["files"])
    assert {row["kind"] for row in deep["files"]} == {
        "manifest",
        "payload",
        "generation_executable",
    }


def test_builder_rejects_missing_contract_requirement(tmp_path: Path) -> None:
    config, spec_path, _root = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    del spec["families"]["logos"]["requirements"]["static"]["data"]
    _write_json(spec_path, spec)
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_builder_deep_verification_rejects_payload_corruption(
    tmp_path: Path,
) -> None:
    config, spec_path, root = _fixture(tmp_path)
    (root / "static" / "payload.bin").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch|SHA-256 mismatch"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_builder_requires_checkpoint_bound_inputs_to_be_deferred(
    tmp_path: Path,
) -> None:
    config, spec_path, root = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    tokenizer_sha = json.loads(
        (root / "tokenizer" / "MANIFEST.json").read_text()
    )["manifest_sha256"]
    sealed_rollouts = _sealed(
        root,
        directory="rollouts",
        schema="metis.rlvr-data/v1",
        tokenizer_sha256=tokenizer_sha,
    )
    spec["families"]["praxis"]["requirements"]["on_policy"]["rollouts"] = {
        "state": "sealed",
        "manifest": sealed_rollouts.relative_to(root).as_posix(),
    }
    _write_json(spec_path, spec)
    with pytest.raises(RuntimeError, match="checkpoint-bound.*deferred"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_builder_requires_static_inputs_to_be_presealed(
    tmp_path: Path,
) -> None:
    config, spec_path, _root = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    deferred = json.loads(
        json.dumps(
            spec["families"]["praxis"]["requirements"]["on_policy"][
                "rollouts"
            ]
        )
    )
    spec["families"]["praxis"]["requirements"]["static"]["data"] = deferred
    _write_json(spec_path, spec)
    with pytest.raises(RuntimeError, match="not checkpoint-bound.*sealed"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_builder_rejects_non_executable_generator(tmp_path: Path) -> None:
    config, spec_path, root = _fixture(tmp_path)
    executable = root / "hooks" / "generate.py"
    executable.chmod(0o640)
    assert not os.access(executable, os.X_OK)
    with pytest.raises(RuntimeError, match="not executable"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_builder_rejects_rank_receipt_path_that_is_a_file(tmp_path: Path) -> None:
    config, spec_path, root = _fixture(tmp_path)
    invalid = root / "generated" / "praxis-rank-receipts"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="absent or an existing directory"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_builder_rejects_a_manually_claimed_tokenizer(tmp_path: Path) -> None:
    config, spec_path, _root = _fixture(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec["tokenizer_manifest"] = {"manifest": "tokenizer/MANIFEST.json"}
    _write_json(spec_path, spec)
    with pytest.raises(RuntimeError, match="derived directly"):
        build_posttraining_release(config=config, spec_path=spec_path)


def test_preflight_rejects_base_tokenizer_byte_drift(tmp_path: Path) -> None:
    config, spec_path, _root = _fixture(tmp_path)
    build_posttraining_release(config=config, spec_path=spec_path)
    tokenizer = config.release_root / "tokenizer" / "tokenizer.json"
    tokenizer.write_bytes(b"x" * tokenizer.stat().st_size)
    report = inspect_posttraining_release_index(config)
    assert report["ok"] is False
    assert any(
        "tokenizer" in error.lower()
        for error in report["errors"]
    )


def test_preflight_rejects_forged_deferred_static_requirement(
    tmp_path: Path,
) -> None:
    config, spec_path, root = _fixture(tmp_path)
    build_posttraining_release(config=config, spec_path=spec_path)
    index_path = root / "PRAXIS_RELEASE_INDEX.json"
    index = json.loads(index_path.read_text())
    forged = json.loads(
        json.dumps(index["requirements"]["on_policy"]["rollouts"])
    )
    forged["schema"] = "metis.sft-data/v1"
    index["requirements"]["static"]["data"] = forged
    index["index_sha256"] = json_sha256(
        index, omit=("index_sha256",)
    )
    _write_json(index_path, index)
    umbrella_path = root / "RELEASE_INDEX.json"
    umbrella = json.loads(umbrella_path.read_text())
    umbrella["families"]["praxis"]["sha256"] = file_sha256(index_path)
    umbrella["families"]["praxis"]["index_sha256"] = index["index_sha256"]
    umbrella["umbrella_sha256"] = json_sha256(
        umbrella, omit=("umbrella_sha256",)
    )
    _write_json(umbrella_path, umbrella)
    report = inspect_posttraining_release_index(config)
    assert report["ok"] is False
    assert any(
        "not checkpoint-bound" in error and "must be sealed" in error
        for error in report["errors"]
    )


def test_distributed_audit_hashes_static_payload_after_bounded_preflight(
    tmp_path: Path,
) -> None:
    config, spec_path, root = _fixture(tmp_path)
    build_posttraining_release(config=config, spec_path=spec_path)
    preflight = inspect_posttraining_release_index(config)
    assert preflight["ok"] is True
    preflight_path = tmp_path / "posttraining-preflight.json"
    _write_json(preflight_path, preflight)
    context = SimpleNamespace(
        rank=0,
        world_size=1,
        initialized=False,
        is_root=True,
    )
    marker = verify_posttraining_release_distributed(
        preflight_path=preflight_path,
        output_path=tmp_path / "posttraining-verification.json",
        receipt_directory=tmp_path / "posttraining-rank-receipts",
        context=context,
    )
    assert marker is not None
    assert marker["ok"] is True
    assert marker["file_count"] == preflight["deep_verification"]["file_count"]

    payload = root / "static" / "payload.bin"
    payload.write_bytes(b"x" * payload.stat().st_size)
    assert inspect_posttraining_release_index(config)["ok"] is True
    with pytest.raises(RuntimeError, match="payload changed"):
        verify_posttraining_release_distributed(
            preflight_path=preflight_path,
            output_path=tmp_path / "tampered-verification.json",
            receipt_directory=tmp_path / "tampered-rank-receipts",
            context=context,
        )


def test_build_template_covers_every_requirement_and_uses_deferred_hooks(
    tmp_path: Path,
) -> None:
    config, _spec_path, _root = _fixture(tmp_path)
    template = posttraining_build_template(config)
    assert set(template["families"]) == {"praxis", "logos"}
    assert "tokenizer_manifest" not in template
    for family in template["families"].values():
        requirements = family["requirements"]
        assert set(requirements) == {"static", "on_policy"}
        assert requirements["static"]["data"]["state"] == "sealed"
        deferred = requirements["on_policy"]["rollouts"]
        assert deferred["state"] == "deferred"
        hook = deferred["generation_hook"]
        assert hook["execution"] == {"protocol": "distributed_family_v1"}
        assert hook["rank_receipts"].endswith("/rank-receipts")


def test_production_template_installs_one_pinned_adapter_runner(
    tmp_path: Path,
) -> None:
    lustre = tmp_path / "lustre" / "vollmerc" / "metis-1.6"
    _base_release(lustre)
    config = load_portage_config(
        environment={
            "METIS_LUSTRE_ROOT": str(lustre),
            "METIS_DATA_RELEASE": str(
                lustre / "releases" / "metis-1.6-data-r1"
            ),
            "METIS_PORTAGE_STATE_ROOT": str(lustre / "training"),
        }
    )
    template = posttraining_build_template(config)
    expected = {
        "hybrid_mode_gspo": (
            "hybrid_mode_gspo",
            "hybrid_mode_verifier",
        ),
        "specialist_reasoning": (
            "specialist_reasoning",
            "stem_verifier",
        ),
        "opd_consolidation": (
            "opd_consolidation",
            "opd_generation_adapter",
        ),
        "evaluation": ("evaluation", "evaluation_suite"),
        "publish_gate": ("evaluation", "evaluation_suite"),
    }
    executables: set[str] = set()
    for family in template["families"].values():
        for stage_id, (adapter_stage, adapter_requirement) in expected.items():
            deferred = next(
                row
                for row in family["requirements"][stage_id].values()
                if row["state"] == "deferred"
            )
            hook = deferred["generation_hook"]
            executables.add(hook["executable"])
            assert hook["args"] == [
                "--adapter-stage",
                adapter_stage,
                "--adapter-requirement",
                adapter_requirement,
            ]
    assert executables == {"hooks/metis16-posttraining-materialize"}
    executable = (
        config.posttraining_release_index.parent
        / "hooks"
        / "metis16-posttraining-materialize"
    )
    assert executable.is_file()
    assert os.access(executable, os.X_OK)
