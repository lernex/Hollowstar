from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from metis_portage.util import atomic_write_json, file_sha256, json_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
HOOK = REPOSITORY / "ops" / "metis16-posttraining-materialize"


def test_builtin_generation_hook_runs_only_a_sealed_adapter(
    tmp_path: Path,
) -> None:
    root = tmp_path / "release"
    adapter_root = root / "adapter"
    adapter_root.mkdir(parents=True)
    adapter_executable = adapter_root / "adapter.py"
    adapter_executable.write_text(
        """#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path

def digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

output = Path(os.environ["METIS_OUTPUT_MANIFEST"])
output.parent.mkdir(parents=True, exist_ok=True)
payload = output.parent / "payload.bin"
payload.write_bytes(b"sealed generated payload")
manifest = {
    "envelope_schema": "metis.sealed-artifact/v1",
    "schema": "metis.rlvr-data/v2",
    "complete": True,
    "metadata": {
        "family": os.environ["METIS_FAMILY"],
        "generated_from_checkpoint_sha256": os.environ[
            "METIS_PARENT_CHECKPOINT_SHA256"
        ],
    },
    "files": [{
        "path": payload.name,
        "bytes": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }],
}
manifest["manifest_sha256"] = digest(manifest)
temporary = output.with_suffix(".partial")
temporary.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
temporary.replace(output)
""",
        encoding="utf-8",
    )
    adapter_executable.chmod(0o750)
    adapter_contract = {
        "schema": "metis.generation-adapter/v1",
        "executable": adapter_executable.name,
        "executable_sha256": file_sha256(adapter_executable),
        "args": [],
        "stages": ["hybrid_mode_gspo"],
        "requirements": ["hybrid_mode_rl_data"],
        "output_envelope_schema": "metis.sealed-artifact/v1",
    }
    adapter_contract["adapter_sha256"] = json_sha256(adapter_contract)
    adapter_manifest = {
        "envelope_schema": "metis.sealed-artifact/v1",
        "schema": "metis.verifier-bundle/v1",
        "complete": True,
        "metadata": {
            "generation_adapter_present": True,
            "generation_adapter": adapter_contract,
        },
        "files": [
            {
                "path": adapter_executable.name,
                "bytes": adapter_executable.stat().st_size,
                "sha256": file_sha256(adapter_executable),
            }
        ],
    }
    adapter_manifest["manifest_sha256"] = json_sha256(adapter_manifest)
    adapter_manifest_path = adapter_root / "MANIFEST.json"
    atomic_write_json(adapter_manifest_path, adapter_manifest)

    target_record = {
        "schema": "metis.rlvr-data/v2",
        "state": "deferred",
        "manifest": "generated/MANIFEST.json",
        "generation_hook": {},
    }
    family_index = {
        "schema": "metis.posttraining-release-index/v1",
        "family": "praxis",
        "requirements": {
            "hybrid_mode_gspo": {
                "hybrid_mode_verifier": {
                    "schema": "metis.verifier-bundle/v1",
                    "state": "sealed",
                    "path": adapter_manifest_path.relative_to(root).as_posix(),
                    "sha256": file_sha256(adapter_manifest_path),
                    "manifest_sha256": adapter_manifest["manifest_sha256"],
                },
                "hybrid_mode_rl_data": target_record,
            }
        },
    }
    family_index["index_sha256"] = json_sha256(family_index)
    index_path = root / "PRAXIS_RELEASE_INDEX.json"
    atomic_write_json(index_path, family_index)

    output = root / "generated" / "MANIFEST.json"
    reducer = root / "generated" / "GENERATION_RECEIPT.json"
    ranks = root / "generated" / "rank-receipts"
    deep = root / "DEEP_VERIFICATION.json"
    deep_payload = {
        "schema": "metis.posttraining-release-deep-verification/v1",
        "complete": True,
    }
    deep_payload["receipt_sha256"] = json_sha256(deep_payload)
    atomic_write_json(deep, deep_payload)
    request = {
        "schema": "metis.deferred-materialization-request/v1",
        "family": "praxis",
        "stage": "hybrid_mode_gspo",
        "requirement": "hybrid_mode_rl_data",
        "requirement_schema": "metis.rlvr-data/v2",
        "parent_checkpoint_sha256": "a" * 64,
        "stage_bindings": {"fixture": True},
        "release_index_path": str(index_path),
        "release_index_file_sha256": file_sha256(index_path),
        "release_index_sha256": family_index["index_sha256"],
        "record_sha256": json_sha256(target_record),
        "deep_verification": {
            "path": str(deep),
            "file_sha256": file_sha256(deep),
            "receipt_sha256": deep_payload["receipt_sha256"],
        },
        "hook": {
            "executable": str(HOOK),
            "executable_sha256": file_sha256(HOOK),
            "args": [
                "--adapter-stage",
                "hybrid_mode_gspo",
                "--adapter-requirement",
                "hybrid_mode_verifier",
            ],
            "timeout_seconds": 60,
            "output_manifest": str(output),
            "reducer_receipt": str(reducer),
            "rank_receipts": str(ranks),
            "execution": {
                "protocol": "rank0_only_v1",
                "nodes": 1,
                "tasks": 1,
                "gpus_per_task": 0,
            },
            "world_size": 1,
        },
    }
    request["request_sha256"] = json_sha256(request)
    request_path = tmp_path / "request.json"
    atomic_write_json(request_path, request)

    environment = {
        **os.environ,
        "METIS_GENERATION_REQUEST": str(request_path),
        "METIS_GENERATION_REQUEST_SHA256": request["request_sha256"],
        "METIS_GENERATION_REQUEST_FILE_SHA256": file_sha256(request_path),
        "METIS_FAMILY": "praxis",
        "METIS_STAGE_ID": "hybrid_mode_gspo",
        "METIS_REQUIREMENT_NAME": "hybrid_mode_rl_data",
        "METIS_PARENT_CHECKPOINT_SHA256": "a" * 64,
        "METIS_OUTPUT_MANIFEST": str(output),
        "METIS_GENERATION_RECEIPT": str(reducer),
        "METIS_GENERATION_RANK_RECEIPT_DIRECTORY": str(ranks),
        "METIS_GENERATION_WORLD_SIZE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--adapter-stage",
            "hybrid_mode_gspo",
            "--adapter-requirement",
            "hybrid_mode_verifier",
            "--reducer-timeout-seconds",
            "5",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(reducer.read_text(encoding="utf-8"))
    assert receipt["schema"] == "metis.generation-hook-receipt/v2"
    assert receipt["adapter_contract_sha256"] == adapter_contract[
        "adapter_sha256"
    ]
    assert receipt["receipt_sha256"] == json_sha256(
        receipt, omit=("receipt_sha256",)
    )
    assert len(receipt["rank_receipts"]) == 1
