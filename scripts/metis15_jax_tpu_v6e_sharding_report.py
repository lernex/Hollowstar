#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _spec_contains(spec: object, text: str) -> bool:
    return text in repr(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report and validate Metis-1.5 JAX v6e sharding contracts.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "configs/metis15_manifest.json")
    parser.add_argument("--require-runtime", action="store_true", help="Require an 8-device runtime placement proof.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    from metis_mamba.jax_metis import (
        JaxMetisTrainConfig,
        count_params,
        create_v6e_expert_mesh,
        init_optimizer_state,
        init_params,
        load_manifest_config,
        make_repeated_batch,
        parameter_partition_specs,
        shard_batch_for_v6e,
        shard_optimizer_state_for_v6e,
        shard_params_for_v6e,
        stack_microbatches,
        tiny_config,
    )

    failures: list[str] = []
    report: dict[str, object] = {
        "manifest": str(args.manifest),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "device_count": jax.device_count(),
        "devices": [f"{device.platform}:{device.id}" for device in jax.devices()],
    }

    key = jax.random.PRNGKey(20260602)
    pre_cfg, pre_train = load_manifest_config(args.manifest, stage="pretrain")
    cpt_cfg, cpt_train = load_manifest_config(args.manifest, stage="continued_pretrain")
    pre_shapes = jax.eval_shape(lambda rng: init_params(rng, pre_cfg), key)
    cpt_shapes = jax.eval_shape(lambda rng: init_params(rng, cpt_cfg), key)
    specs = parameter_partition_specs(cpt_shapes)
    abstract = {
        "param_count": count_params(pre_shapes),
        "pretrain_capacity": pre_cfg.capacity_for_batch(pre_train.local_batch_size),
        "continued_pretrain_capacity": cpt_cfg.capacity_for_batch(cpt_train.local_batch_size),
        "experts_per_chip": pre_cfg.moe_num_experts if pre_cfg.moe_expert_parallel_size == 1 else pre_cfg.experts_per_rank,
        "promoted_parallelism": "data_parallel" if pre_cfg.moe_expert_parallel_size == 1 else "expert_parallel",
        "layer0_expert_w1_spec": repr(specs["layers"][0]["expert_w1"]),
        "layer0_router_spec": repr(specs["layers"][0]["router"]),
        "layer0_router_bias_spec": repr(specs["layers"][0]["router_bias"]),
        "mor_router_w1_spec": repr(specs["mor_router"]["w1"]),
        "embed_spec": repr(specs["embed"]),
    }
    report["abstract_full_shape"] = abstract

    if abstract["param_count"] != 898_051_168:
        failures.append(f"unexpected full-shape param count: {abstract['param_count']}")
    if pre_cfg.moe_expert_parallel_size == 1:
        if abstract["experts_per_chip"] != 32:
            failures.append(f"expected 32 local experts/chip, got {abstract['experts_per_chip']}")
    else:
        if abstract["experts_per_chip"] != 4:
            failures.append(f"expected 4 sharded experts/chip, got {abstract['experts_per_chip']}")
        if not _spec_contains(specs["layers"][0]["expert_w1"], "expert"):
            failures.append("routed expert weights are not expert-axis sharded.")
        if not _spec_contains(specs["layers"][0]["router"], "expert"):
            failures.append("layer router output dimension is not expert-axis sharded.")
        if not _spec_contains(specs["layers"][0]["router_bias"], "expert"):
            failures.append("layer router bias is not expert-axis sharded.")
    if _spec_contains(specs["mor_router"]["w1"], "expert") or _spec_contains(specs["mor_router"]["w2"], "expert"):
        failures.append("MoR control router must stay replicated.")
    if _spec_contains(specs["embed"], "expert"):
        failures.append("embedding table must stay replicated.")

    runtime: dict[str, object] = {"available": jax.device_count() == 8}
    if jax.device_count() == 8:
        mesh = create_v6e_expert_mesh()
        tiny = tiny_config(mor=False)
        tiny = tiny.__class__(**{**tiny.__dict__, "expert_execution": "shard_map"})
        params = init_params(key, tiny)
        sharded_params = shard_params_for_v6e(params, mesh)
        opt_state = shard_optimizer_state_for_v6e(init_optimizer_state(sharded_params), sharded_params, mesh)
        micro = make_repeated_batch(batch_size=2, block_size=tiny.block_size, vocab_size=tiny.vocab_size)
        batch = {
            name: jnp.asarray(value)
            for name, value in stack_microbatches([micro, micro]).items()
        }
        sharded_batch = shard_batch_for_v6e(batch, mesh)
        runtime.update(
            {
                "mesh": repr(mesh),
                "param_expert_w1_sharding": repr(sharded_params["layers"][0]["expert_w1"].sharding),
                "param_router_sharding": repr(sharded_params["layers"][0]["router"].sharding),
                "optimizer_step_sharding": repr(opt_state.step.sharding),
                "batch_input_ids_sharding": repr(sharded_batch["input_ids"].sharding),
                "batch_input_ids_shape": tuple(int(dim) for dim in sharded_batch["input_ids"].shape),
            }
        )
        if "NamedSharding" not in runtime["param_expert_w1_sharding"]:
            failures.append("runtime expert weights did not receive NamedSharding.")
        if "NamedSharding" not in runtime["optimizer_step_sharding"]:
            failures.append("runtime optimizer step did not receive replicated NamedSharding.")
        if "NamedSharding" not in runtime["batch_input_ids_sharding"]:
            failures.append("runtime batch did not receive replicated NamedSharding.")
    elif args.require_runtime:
        failures.append(f"runtime sharding proof requires exactly 8 devices; saw {jax.device_count()}.")
    report["runtime_tiny"] = runtime

    print("Metis-1.5 JAX TPU v6e sharding report")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("metis15_jax_tpu_sharding_report_ok")


if __name__ == "__main__":
    main()
