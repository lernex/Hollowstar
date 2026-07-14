#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import tempfile


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local fixed-shape JAX contract tests for Metis-1.5.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mor", action="store_true", help="Also prove the dynamic-token MoR fixed-depth path.")
    args = parser.parse_args()

    from metis_mamba.jax_metis import (
        JaxMetisTrainConfig,
        JaxMemmapTokenData,
        JaxSamplerState,
        count_params,
        forward,
        init_optimizer_state,
        init_params,
        make_jit_train_step,
        make_repeated_batch,
        manifest_fingerprint,
        muon_mask,
        parameter_partition_specs,
        qk_clip_with_optimizer_state,
        restore_training_checkpoint,
        save_training_checkpoint,
        stack_microbatches,
        tiny_config,
    )
    import jax
    import jax.numpy as jnp

    cfg = tiny_config(mor=args.mor)
    cfg.validate(local_batch_size=2)
    key = jax.random.PRNGKey(20260602)
    params = init_params(key, cfg)
    mask = muon_mask(params)
    opt_state = init_optimizer_state(params)
    train_cfg = JaxMetisTrainConfig(
        stage="continued_pretrain" if args.mor else "pretrain",
        local_batch_size=2,
        grad_accum_steps=2,
        learning_rate=3e-3,
        weight_decay=0.01,
        muon_ns_steps=2,
        max_steps=args.steps,
    )
    micro_np = make_repeated_batch(batch_size=2, block_size=cfg.block_size, vocab_size=cfg.vocab_size)
    batch_np = stack_microbatches([micro_np, micro_np])
    batch = {key: jnp.asarray(value) for key, value in batch_np.items()}

    loss, metrics = forward(params, batch["input_ids"][0], cfg, labels=batch["labels"][0])
    if not bool(jnp.isfinite(loss)):
        raise AssertionError("Initial JAX loss must be finite.")
    if args.mor:
        temp_params = dict(params)
        temp_params["mor_router"] = dict(params["mor_router"])
        temp_params["mor_router"]["w1"] = jnp.zeros_like(params["mor_router"]["w1"])
        temp_params["mor_router"]["b1"] = jnp.ones_like(params["mor_router"]["b1"])
        temp_params["mor_router"]["w2"] = jnp.zeros_like(params["mor_router"]["w2"])
        temp_params["mor_router"]["b2"] = jnp.linspace(-2.0, 2.0, cfg.mor_max_depth).astype(params["mor_router"]["b2"].dtype)
        cold_cfg = cfg.__class__(**{**cfg.__dict__, "mor_router_temperature": 0.5})
        warm_cfg = cfg.__class__(**{**cfg.__dict__, "mor_router_temperature": 2.0})
        _, cold_metrics = forward(temp_params, batch["input_ids"][0], cold_cfg, labels=batch["labels"][0])
        _, warm_metrics = forward(temp_params, batch["input_ids"][0], warm_cfg, labels=batch["labels"][0])
        if float(jax.device_get(warm_metrics["mor_entropy"])) <= float(jax.device_get(cold_metrics["mor_entropy"])):
            raise AssertionError("MoR router temperature did not increase depth-router entropy.")
        depth1_params = dict(params)
        depth1_params["mor_router"] = dict(params["mor_router"])
        depth1_params["mor_router"]["w1"] = jnp.zeros_like(params["mor_router"]["w1"])
        depth1_params["mor_router"]["b1"] = jnp.ones_like(params["mor_router"]["b1"])
        depth1_params["mor_router"]["w2"] = jnp.zeros_like(params["mor_router"]["w2"])
        depth1_params["mor_router"]["b2"] = jnp.asarray([3.0, 0.0, -3.0], dtype=params["mor_router"]["b2"].dtype)
        _, depth1_metrics = forward(depth1_params, batch["input_ids"][0], cfg, labels=batch["labels"][0])
        depth1_total = cfg.n_layer * batch["input_ids"][0].size * cfg.moe_top_k
        if int(jax.device_get(depth1_metrics["total_assignments"])) != depth1_total:
            raise AssertionError("Hard packed MoR did not let depth-1 tokens skip recursive MoE assignments.")
        if float(jax.device_get(depth1_metrics["mor_packed_active_tokens"])) != 0.0:
            raise AssertionError("Depth-1 hard routing should not activate packed recursive depth buffers.")
        depth3_params = dict(params)
        depth3_params["mor_router"] = dict(params["mor_router"])
        depth3_params["mor_router"]["w1"] = jnp.zeros_like(params["mor_router"]["w1"])
        depth3_params["mor_router"]["b1"] = jnp.ones_like(params["mor_router"]["b1"])
        depth3_params["mor_router"]["w2"] = jnp.zeros_like(params["mor_router"]["w2"])
        depth3_params["mor_router"]["b2"] = jnp.asarray([-3.0, 0.0, 3.0], dtype=params["mor_router"]["b2"].dtype)
        _, depth3_metrics = forward(depth3_params, batch["input_ids"][0], cfg, labels=batch["labels"][0])
        dense_max_total = depth1_total * cfg.mor_max_depth
        if int(jax.device_get(depth3_metrics["total_assignments"])) >= dense_max_total:
            raise AssertionError("Hard packed recursive MoR should stay below the old dense max-depth assignment count.")
        if float(jax.device_get(depth3_metrics["mor_packed_active_tokens"])) <= 0.0:
            raise AssertionError("Depth-3 hard routing did not use packed recursive depth buffers.")
    expected_total = cfg.n_layer * batch["input_ids"][0].size * cfg.moe_top_k
    if not args.mor and int(metrics["total_assignments"]) != expected_total:
        raise AssertionError(f"Unexpected total assignments: {metrics['total_assignments']} vs {expected_total}")
    if float(metrics["expert_drop_frac"]) > 0.25:
        raise AssertionError(f"Tiny fixed-capacity path dropped too many assignments: {metrics['expert_drop_frac']}")

    train_step = make_jit_train_step(cfg, train_cfg, mask)
    losses: list[float] = []
    mor_targets: list[float] = []
    mor_coefs: list[float] = []
    for _ in range(args.steps):
        params, opt_state, step_metrics = train_step(params, opt_state, batch)
        losses.append(float(jax.device_get(step_metrics["loss"])))
        mor_targets.append(float(jax.device_get(step_metrics["mor_target_depth"])))
        mor_coefs.append(float(jax.device_get(step_metrics["mor_aux_coef"])))
        if int(jax.device_get(step_metrics["grad_accum_steps"])) != 2:
            raise AssertionError("JAX train step did not report the expected grad_accum_steps.")
        if not all(jnp.isfinite(value).all() for value in jax.tree_util.tree_leaves(step_metrics)):
            raise AssertionError("Nonfinite JAX training metric.")

    if losses[-1] >= losses[0] * 0.99:
        raise AssertionError(f"JAX tiny loss did not improve: start={losses[0]:.6f} end={losses[-1]:.6f}")
    if args.mor:
        if mor_targets[-1] <= mor_targets[0] or mor_coefs[-1] <= mor_coefs[0]:
            raise AssertionError(
                "CPT MoR target/aux schedule did not increase during the tiny fixed-shape proof."
            )
    elif max(abs(value) for value in mor_coefs) > 1e-9:
        raise AssertionError("Pretraining MoR aux coefficient must stay disabled.")

    muon_leaves = [leaf for leaf in jax.tree_util.tree_leaves(mask) if bool(getattr(leaf, "shape", ()) == ())]
    muon_count = sum(1 for leaf in jax.tree_util.tree_leaves(mask) if bool(leaf))
    adamw_count = sum(1 for leaf in jax.tree_util.tree_leaves(mask) if not bool(leaf))
    if muon_count <= 0 or adamw_count <= 0 or not muon_leaves:
        raise AssertionError("Expected both Muon and AdamW leaves in JAX optimizer mask.")
    specs = parameter_partition_specs(params)
    if "expert" not in repr(specs["layers"][0]["expert_w1"]):
        raise AssertionError("Routed expert weights must be sharded over the expert axis.")
    if "expert" not in repr(specs["layers"][0]["router"]):
        raise AssertionError("Router output dimension must be sharded over the expert axis.")
    if "expert" in repr(specs["embed"]):
        raise AssertionError("Embeddings should stay replicated in the first v6e layout.")
    if args.mor and ("expert" in repr(specs["mor_router"]["w1"]) or "expert" in repr(specs["mor_router"]["w2"])):
        raise AssertionError("MoR depth router must stay replicated; only routed MoE experts use the expert axis.")
    clip_cfg = cfg.__class__(**{**cfg.__dict__, "qk_clip_threshold": 1e-8})
    before_momentum = opt_state.muon_momentum["layers"][0]["q"]
    before_norm = jnp.linalg.norm(before_momentum.astype(jnp.float32))
    clipped_params, clipped_opt_state, clip_metrics = qk_clip_with_optimizer_state(params, opt_state, clip_cfg)
    if int(clip_metrics["qk_clip_scaled_layers"]) <= 0 or float(clip_metrics["qk_clip_min_scale"]) >= 1.0:
        raise AssertionError("Forced QK clip did not scale any layer.")
    if float(before_norm) > 0:
        expected_norm = before_norm * float(clip_metrics["qk_clip_min_scale"])
        after_norm = jnp.linalg.norm(clipped_opt_state.muon_momentum["layers"][0]["q"].astype(jnp.float32))
        if not bool(jnp.isclose(after_norm, expected_norm, rtol=2e-3, atol=1e-7)):
            raise AssertionError("QK clip did not mirror the scale into Muon momentum state.")
    if jnp.linalg.norm(clipped_params["layers"][0]["q"].astype(jnp.float32)) >= jnp.linalg.norm(params["layers"][0]["q"].astype(jnp.float32)):
        raise AssertionError("Forced QK clip did not reduce Q parameter norm.")

    print(
        "metis15_jax_contracts_ok "
        f"mor={int(args.mor)} params={count_params(params)} start_loss={losses[0]:.6f} "
        f"end_loss={losses[-1]:.6f} muon_leaves={muon_count} adamw_leaves={adamw_count}",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        tokens = (jnp.arange(256, dtype=jnp.int32) % cfg.vocab_size).astype(jnp.uint16)
        import numpy as np

        np.asarray(tokens).tofile(data_dir / "train.bin")
        (data_dir / "meta.json").write_text(
            json.dumps({"dtype": "uint16", "vocab_size": cfg.vocab_size, "train_tokens": 256}, indent=2),
            encoding="utf-8",
        )
        loader = JaxMemmapTokenData(data_dir, split="train", batch_size=2, block_size=cfg.block_size)
        first = loader.next_batch()
        state_after_first = loader.state
        expected_second = loader.next_batch()
        loader.load_state(state_after_first)
        replayed_second = loader.next_batch()
        if not np.array_equal(expected_second["input_ids"], replayed_second["input_ids"]):
            raise AssertionError("Sampler resume did not reproduce the next input batch.")
        if state_after_first.tokens_emitted != 2 * cfg.block_size:
            raise AssertionError("Sampler tokens_emitted did not advance by batch tokens.")
        checkpoint_dir = Path(tmp) / "ckpt"
        manifest_hash = "tiny-manifest"
        save_training_checkpoint(
            checkpoint_dir,
            params=params,
            opt_state=opt_state,
            sampler_state=loader.state,
            step=args.steps,
            manifest_hash=manifest_hash,
            metrics={"loss": losses[-1]},
        )
        restored_params, restored_opt_state, restored_sampler_state, metadata = restore_training_checkpoint(
            checkpoint_dir,
            target_params=params,
            target_opt_state=opt_state,
            expected_manifest_hash=manifest_hash,
        )
        if metadata["step"] != args.steps:
            raise AssertionError("Checkpoint step metadata was not restored.")
        if restored_sampler_state.data_fingerprint != loader.state.data_fingerprint:
            raise AssertionError("Checkpoint sampler fingerprint was not restored.")
        if count_params(restored_params) != count_params(params) or int(restored_opt_state.step) != int(opt_state.step):
            raise AssertionError("Checkpoint params/optimizer state did not restore correctly.")
        print("metis15_jax_data_resume_ok", flush=True)


if __name__ == "__main__":
    main()
