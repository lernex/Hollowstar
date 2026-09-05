"""Run bounded, genuine source canaries independently of downstream reducers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .acquisition import CapacityPending, receipt_path
from .admission import admit_source
from .cli import code_root, load_run
from .common import ObjectSpec, RawReceipt, canonical_json, digest_json, read_receipt
from .policy import policy_config
from .prep import prepare_chunk, prepare_runtime, reblock_object


def prepare_canaries(
    root: Path, object_ids: list[str], *, maximum_raw_bytes: int = 16_000_000,
) -> list[dict[str, Any]]:
    if (
        type(maximum_raw_bytes) is not int or maximum_raw_bytes < 1
        or not object_ids or len(object_ids) != len(set(object_ids))
    ):
        raise ValueError("Canaries need unique object IDs and a positive total byte ceiling")
    if any(not isinstance(object_id, str) or re.fullmatch(r"[0-9a-f]{64}", object_id) is None
           for object_id in object_ids):
        raise ValueError("Canaries must name exact SHA-256 object identities")
    run = load_run(root)
    policy = policy_config(root)
    if not policy["policy_ready"]:
        raise RuntimeError("Canaries require verified eligibility policies")
    records = [read_receipt(receipt_path(root, object_id)) for object_id in object_ids]
    if sum(RawReceipt.from_dict(record).byte_count for record in records) > maximum_raw_bytes:
        raise CapacityPending("Requested canaries exceed the bounded raw-byte allowance")
    config = {
        **run["config"]["prep"], **policy, "root": str(root),
        "quality_profiles_path": str(code_root() / run["config"]["prep"]["quality_profiles_path"]),
        "enforce_storage_budget": True,
    }
    prepare_runtime(config, require_ready=True)
    results = []
    for record in records:
        spec, raw = ObjectSpec.from_dict(record["spec"]), RawReceipt.from_dict(record)
        output = root / "canaries" / spec.object_id
        normalized = reblock_object(spec, raw, output / "reblock", config)
        screened = {}
        for chunk in normalized["chunks"]:
            value = prepare_chunk(root / chunk["ready_receipt"], output / "prepared", config)
            screened[value["chunk_id"]] = value
        result = admit_source(
            root, spec, normalized, screened,
            generation=f"canary-{digest_json(normalized['inputs'])}",
            minimum_acceptance=float(config["source_minimum_acceptance"]),
        )
        results.append({key: value for key, value in result.items() if key != "eligible_receipts"})
        print(canonical_json(results[-1]), flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metis17-canary")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--object", action="append", required=True)
    parser.add_argument("--maximum-raw-bytes", type=int, default=16_000_000)
    args = parser.parse_args(argv)
    prepare_canaries(args.root.expanduser().resolve(), args.object, maximum_raw_bytes=args.maximum_raw_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
