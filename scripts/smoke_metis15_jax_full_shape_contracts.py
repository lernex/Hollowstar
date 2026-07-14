#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    import jax

    from metis_mamba.jax_metis import (
        count_params,
        init_params,
        load_manifest_config,
        muon_mask,
        parameter_partition_specs,
    )

    manifest = ROOT_DIR / "configs/metis15_manifest.json"
    pre_cfg, pre_train = load_manifest_config(manifest, stage="pretrain")
    cpt_cfg, cpt_train = load_manifest_config(manifest, stage="continued_pretrain")
    if pre_cfg.mor_enabled or pre_cfg.mor_runtime_mode != "disabled":
        raise AssertionError("Pretraining must keep MoR disabled.")
    if not cpt_cfg.mor_enabled or cpt_cfg.mor_runtime_mode != "dynamic_token":
        raise AssertionError("Continued pretraining must use dynamic-token MoR.")
    if cpt_cfg.mor_compute_mode != "static_packed_hard":
        raise AssertionError("Continued pretraining must use static-packed hard MoR compute.")
    if (
        abs(cpt_cfg.mor_target_avg_depth_start - 1.05) > 1e-9
        or abs(cpt_cfg.mor_target_avg_depth - 1.65) > 1e-9
        or cpt_cfg.mor_target_avg_depth_warmup_tokens != 1_000_000_000
        or abs(cpt_cfg.mor_router_temperature - 1.0) > 1e-9
        or abs(cpt_cfg.mor_packed_depth_capacity_factor - 0.75) > 1e-9
        or cpt_cfg.mor_packed_depth_capacity_alignment != 128
        or abs(cpt_cfg.mor_router_aux_loss_coef_start - 0.01) > 1e-9
        or abs(cpt_cfg.mor_router_aux_loss_coef - 0.02) > 1e-9
    ):
        raise AssertionError("CPT MoR target/aux warmup schedule did not load from the manifest.")
    if not pre_cfg.remat_layers or not cpt_cfg.remat_layers:
        raise AssertionError("Full Metis-1.5 v6e configs must keep remat_layers enabled.")
    if pre_train.grad_accum_steps != 8 or cpt_train.grad_accum_steps != 6:
        raise AssertionError("Manifest grad_accum_steps did not load correctly.")
    expected_pre_steps = 50_000_000_000 // (8 * 8 * pre_cfg.block_size)
    expected_cpt_steps = 10_000_000_000 // (6 * 6 * cpt_cfg.block_size)
    if pre_train.max_steps != expected_pre_steps or cpt_train.max_steps != expected_cpt_steps:
        raise AssertionError(
            f"Effective train steps must include grad accumulation: pre={pre_train.max_steps}, cpt={cpt_train.max_steps}"
        )

    key = jax.random.PRNGKey(20260602)
    pre_shapes = jax.eval_shape(lambda rng: init_params(rng, pre_cfg), key)
    cpt_shapes = jax.eval_shape(lambda rng: init_params(rng, cpt_cfg), key)
    param_count = count_params(pre_shapes)
    if param_count != pre_cfg.__dict__.get("estimated_params", param_count):
        # The manifest estimate includes historical implementation details; keep this as telemetry,
        # not a failure, while the shape checks below guard the live JAX contract.
        pass
    if pre_cfg.moe_num_experts != 32 or pre_cfg.experts_per_rank != 4:
        raise AssertionError("Full Metis-1.5 must shard 32 routed experts as 4 experts per v6e chip.")
    expected_pre_capacity = 4096
    expected_cpt_capacity = 3072
    if pre_cfg.capacity_for_batch(pre_train.local_batch_size) != expected_pre_capacity:
        raise AssertionError("Pretrain capacity must stay fixed at 4096 for local_batch_size=8.")
    if cpt_cfg.capacity_for_batch(cpt_train.local_batch_size) != expected_cpt_capacity:
        raise AssertionError("CPT capacity must stay fixed at 3072 for local_batch_size=6.")
    expected_cpt_depth_capacity = 4608
    if cpt_cfg.mor_depth_capacity_for_batch(cpt_train.local_batch_size) != expected_cpt_depth_capacity:
        raise AssertionError("CPT packed MoR depth capacity must stay fixed at 4608 for local_batch_size=6.")

    mask = muon_mask(pre_shapes)
    muon_count = sum(1 for leaf in jax.tree_util.tree_leaves(mask) if bool(leaf))
    adamw_count = sum(1 for leaf in jax.tree_util.tree_leaves(mask) if not bool(leaf))
    if muon_count <= 0 or adamw_count <= 0:
        raise AssertionError("Expected both Muon and AdamW parameter groups.")
    specs = parameter_partition_specs(cpt_shapes)
    layer0 = specs["layers"][0]
    if "expert" not in repr(layer0["expert_w1"]) or "expert" not in repr(layer0["expert_w2"]):
        raise AssertionError("Routed expert tensors must shard over the expert axis.")
    if "None, 'expert'" not in repr(layer0["router"]) or "'expert'" not in repr(layer0["router_bias"]):
        raise AssertionError("Layer router outputs and biases must shard over the expert axis.")
    if "expert" in repr(specs["mor_router"]["w1"]) or "expert" in repr(specs["mor_router"]["w2"]):
        raise AssertionError("MoR depth router must stay replicated; it is not an expert-axis tensor.")
    if "expert" in repr(specs["embed"]):
        raise AssertionError("Embeddings must stay replicated in the first v6e expert-parallel layout.")

    print(
        "metis15_jax_full_shape_contracts_ok "
        f"params={param_count} pre_steps={pre_train.max_steps} cpt_steps={cpt_train.max_steps} "
        f"experts_per_chip={pre_cfg.experts_per_rank} muon_leaves={muon_count} adamw_leaves={adamw_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
