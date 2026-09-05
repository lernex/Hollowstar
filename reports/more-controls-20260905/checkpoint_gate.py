"""Reject a corrected control before full training if its experts did not learn."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path

import torch


WORLD_SIZES = {
    "moe-k4": 20,
    "moe-k8": 20,
    "random-k": 80,
    "random-depth": 80,
    "mor-fixed-k": 80,
}
GATE_STEP = 100
TOTAL_STEPS = 25429


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def inspect(run_root: Path, row: str, source_revision: str) -> dict:
    run = json.loads((run_root / "run.json").read_text())
    identity = run["run_identity"]
    require(identity["row"] == row, "Wrong control row")
    require(identity["source_revision"] == source_revision, "Wrong source revision")
    require(identity["seed"] == 16062026, "Wrong initialization seed")
    require(run["total_steps"] == TOTAL_STEPS, "Truncated training horizon")
    require(run["global_batch_tokens"] == 1966080, "Changed global batch")
    require(run["world_size"] == WORLD_SIZES[row], "Changed data-parallel geometry")
    require(run["schedule"]["base_learning_rate"] == 0.00026, "Changed learning rate")
    require(run["schedule"]["total_steps"] == TOTAL_STEPS, "Changed LR horizon")
    require(
        run["optimizer"].get("require_routed_expert_gradients") is True,
        "In-trainer expert-gradient gate is disabled",
    )
    require(
        run["model"]["activation_recompute_policy"] == "none",
        "Changed recompute lane",
    )
    require(
        identity["release"] == {
            "release_sha256": "763948d8fa1e24ad5615257d3e141ec420275d868295e6826090ecffb8cae959",
            "shard_manifest_sha256": "85c973ca54dda9a464c48529259605f3e453905827a40596ba44b8f894210c73",
        },
        "Changed sealed data release",
    )
    require(run["start_step"] == 0, "Gate did not start from fresh initialization")
    summary = json.loads((run_root / "summary.json").read_text())
    require(summary["steps"] == GATE_STEP, "Gate did not reach its execution bound")
    require(summary["tokens"] == GATE_STEP * 1966080, "Gate token count mismatch")
    require(math.isfinite(summary["final_loss"]), "Non-finite pilot loss")
    rank_gradients = []
    for rank in range(WORLD_SIZES[row]):
        initial = None
        with (run_root / "telemetry" / f"rank-{rank:05d}.jsonl").open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["step"] == 0:
                    initial = record["telemetry"]
                    break
        require(initial is not None, f"Missing initial gradient evidence on rank {rank}")
        for field in ("expected", "present", "nonzero"):
            require(
                initial[f"routed_expert_gradient_parameters_{field}"] == 16,
                f"Incomplete {field} expert gradients on rank {rank}",
            )
        norm = initial["routed_expert_gradient_norm"]
        require(
            math.isfinite(norm) and norm > 0,
            f"Invalid expert gradient norm on rank {rank}",
        )
        rank_gradients.append({"rank": rank, "nonzero_expert_chunks": 16, "norm": norm})
    checkpoint = run_root / "checkpoints" / f"step-{GATE_STEP:07d}"
    state_path = checkpoint / "state.pt"
    state = load(state_path)
    require(
        state["run_identity_sha256"] == run["run_identity_sha256"],
        "Checkpoint identity mismatch",
    )
    require(state["step"] == GATE_STEP, "Incorrect resume cursor")
    require(state["step_semantics"] == "next_unexecuted", "Ambiguous resume cursor")
    require(state["total_steps"] == TOTAL_STEPS, "Checkpoint horizon mismatch")
    shards = state["optimizer_shards"]
    require(len(shards) == WORLD_SIZES[row], "Missing optimizer shard")
    require(
        {item["rank"] for item in shards} == set(range(WORLD_SIZES[row])),
        "Missing or duplicate optimizer ownership rank",
    )
    for item in shards:
        require(
            (checkpoint / item["path"]).stat().st_size == item["bytes"],
            f"Optimizer shard size mismatch: {item['path']}",
        )
    first = load(checkpoint / "optimizer-rank-00000.pt")
    owners = first["optimizer"]["owners"]
    del first
    names = [
        name for name in state["model"]
        if ".moe.local_experts." in name and ".weight_chunks." in name
    ]
    require(len(names) == 16, "Unexpected routed-expert parameter inventory")
    evidence = []
    for owner in sorted({owners[name] for name in names}):
        payload = load(checkpoint / f"optimizer-rank-{owner:05d}.pt")
        optimizer = payload["optimizer"]
        require(optimizer["rank"] == owner, "Optimizer shard rank mismatch")
        require(optimizer["owners"] == owners, "Optimizer ownership mismatch")
        values = optimizer["bundle"]["dense"]["state"]
        for name in names:
            if owners[name] != owner:
                continue
            shape = tuple(state["model"][name].shape)
            candidates = [
                value for value in values.values()
                if "master_param" in value
                and tuple(value["master_param"].shape) == shape
            ]
            require(len(candidates) == 1, f"Ambiguous optimizer mapping: {name}")
            value = candidates[0]
            momentum = value["momentum_buffer"]
            count = int(torch.count_nonzero(momentum))
            require(count > 0, f"Routed expert has no learned momentum: {name}")
            if "momentum_scale" in value:
                require(
                    bool(torch.isfinite(value["momentum_scale"]).all())
                    and bool((value["momentum_scale"] > 0).any()),
                    f"Non-finite or entirely zero optimizer scale: {name}",
                )
            evidence.append({
                "parameter": name,
                "owner_rank": owner,
                "parameter_shape": list(shape),
                "momentum_elements": momentum.numel(),
                "nonzero_momentum_elements": count,
            })
        del values, optimizer, payload
        gc.collect()
    digest = hashlib.sha256()
    with state_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "schema": "more.corrected-control-expert-gate/v1",
        "passed": True,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row": row,
        "source_revision": source_revision,
        "run_identity_sha256": run["run_identity_sha256"],
        "campaign_identity_sha256": run["campaign_identity_sha256"],
        "checkpoint_step": GATE_STEP,
        "checkpoint": str(checkpoint),
        "checkpoint_state_sha256": digest.hexdigest(),
        "total_steps": TOTAL_STEPS,
        "optimizer_shards_present_and_size_matched": len(shards),
        "routed_expert_chunks": len(evidence),
        "chunks_with_nonzero_momentum": len(evidence),
        "expert_momentum": evidence,
        "initial_rank_gradients": rank_gradients,
        "pilot_summary": summary,
    }


def verify_existing(run_root: Path, row: str, source_revision: str, gate_path: Path) -> dict:
    gate = json.loads(gate_path.read_text())
    run = json.loads((run_root / "run.json").read_text())
    require(gate["schema"] == "more.corrected-control-expert-gate/v1", "Wrong gate schema")
    require(gate["passed"] is True, "Existing gate did not pass")
    require(gate["row"] == row == run["run_identity"]["row"], "Gate row mismatch")
    require(
        gate["source_revision"] == source_revision == run["run_identity"]["source_revision"],
        "Gate source revision mismatch",
    )
    require(
        gate["run_identity_sha256"] == run["run_identity_sha256"],
        "Gate no longer matches the resumed experiment",
    )
    require(gate["checkpoint_step"] == GATE_STEP, "Wrong gate checkpoint")
    require(gate["total_steps"] == run["total_steps"] == TOTAL_STEPS, "Changed horizon")
    evidence = gate["expert_momentum"]
    require(len(evidence) == 16, "Incomplete expert gate evidence")
    require(
        len({entry["parameter"] for entry in evidence}) == 16,
        "Duplicate expert gate evidence",
    )
    require(
        all(entry["nonzero_momentum_elements"] > 0 for entry in evidence),
        "Existing gate contains an untrained expert",
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--row", required=True, choices=tuple(WORLD_SIZES))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    gate_path = args.run_root / "operational" / "expert-training-gate.json"
    if args.verify_existing:
        gate = verify_existing(
            args.run_root, args.row, args.source_revision, gate_path
        )
    else:
        gate = inspect(args.run_root, args.row, args.source_revision)
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        with gate_path.open("x") as handle:
            json.dump(gate, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({
        "row": args.row,
        "passed": gate["passed"],
        "checkpoint_step": gate["checkpoint_step"],
        "trained_expert_chunks": gate["chunks_with_nonzero_momentum"],
        "gate_path": str(gate_path),
    }), flush=True)


if __name__ == "__main__":
    main()
