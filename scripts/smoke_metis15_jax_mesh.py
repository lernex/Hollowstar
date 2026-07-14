#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="8-device shard_map proof for Metis-1.5 JAX experts.")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np

    if jax.device_count() != 8:
        raise SystemExit(
            "This smoke expects 8 visible devices. Run with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8 on CPU, or on v6e-8."
        )

    from metis_mamba.jax_metis import (
        JaxMetisTrainConfig,
        JaxSamplerState,
        count_params,
        create_v6e_expert_mesh,
        forward,
        init_optimizer_state,
        init_params,
        make_jit_train_step,
        make_repeated_batch,
        optimizer_matrix_mask,
        restore_training_checkpoint,
        save_training_checkpoint,
        shard_batch_for_v6e,
        shard_optimizer_state_for_v6e,
        shard_params_for_v6e,
        stack_microbatches,
        tiny_config,
    )

    cfg_ref = tiny_config(mor=False)
    cfg_shard = cfg_ref.__class__(**{**cfg_ref.__dict__, "expert_execution": "shard_map"})
    key = jax.random.PRNGKey(20260602)
    params = init_params(key, cfg_ref)
    micro = {
        name: jnp.asarray(value)
        for name, value in make_repeated_batch(
            batch_size=2,
            block_size=cfg_ref.block_size,
            vocab_size=cfg_ref.vocab_size,
        ).items()
    }
    batch = {name: jnp.asarray(value) for name, value in stack_microbatches([{k: np.asarray(v) for k, v in micro.items()}] * 2).items()}
    mesh = create_v6e_expert_mesh()
    batch = shard_batch_for_v6e(batch, mesh)
    if "NamedSharding" not in repr(batch["input_ids"].sharding):
        raise AssertionError("Input batch did not get explicit replicated mesh sharding.")
    sharded_params = shard_params_for_v6e(params, mesh)
    ref_loss, ref_metrics = forward(params, micro["input_ids"], cfg_ref, labels=micro["labels"])
    shard_loss, shard_metrics = forward(
        sharded_params,
        micro["input_ids"],
        cfg_shard,
        labels=micro["labels"],
        expert_mesh=mesh,
    )
    ref_loss = float(jax.device_get(ref_loss))
    shard_loss = float(jax.device_get(shard_loss))
    if abs(ref_loss - shard_loss) > 5e-4:
        raise AssertionError(f"shard_map loss diverged from reference: {shard_loss} vs {ref_loss}")
    if int(jax.device_get(ref_metrics["valid_assignments"])) != int(jax.device_get(shard_metrics["valid_assignments"])):
        raise AssertionError("shard_map valid assignment count differs from reference.")

    train_cfg = JaxMetisTrainConfig(
        local_batch_size=2,
        grad_accum_steps=2,
        learning_rate=3e-3,
        weight_decay=0.01,
        optimizer="adamuon",
        muon_ns_steps=2,
    )
    opt_state = init_optimizer_state(sharded_params)
    opt_state = shard_optimizer_state_for_v6e(opt_state, sharded_params, mesh)
    mask = optimizer_matrix_mask(sharded_params, train_cfg.optimizer)
    step_fn = make_jit_train_step(cfg_shard, train_cfg, mask, expert_mesh=mesh)
    losses: list[float] = []
    for _ in range(args.steps):
        sharded_params, opt_state, metrics = step_fn(sharded_params, opt_state, batch)
        losses.append(float(jax.device_get(metrics["loss"])))
    if not all(jnp.isfinite(jnp.asarray(losses))):
        raise AssertionError("Nonfinite loss in shard_map train step.")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "mesh_ckpt"
        sampler = JaxSamplerState(
            split="synthetic",
            cursor=args.steps,
            epoch=0,
            tokens_emitted=args.steps * 2 * 2 * cfg_ref.block_size,
            data_fingerprint="synthetic:mesh-smoke",
        )
        save_training_checkpoint(
            ckpt_dir,
            params=sharded_params,
            opt_state=opt_state,
            sampler_state=sampler,
            step=args.steps,
            manifest_hash="mesh-smoke",
        )
        restored_params, restored_opt, restored_sampler, metadata = restore_training_checkpoint(
            ckpt_dir,
            target_params=sharded_params,
            target_opt_state=opt_state,
            expected_manifest_hash="mesh-smoke",
        )
        if metadata["step"] != args.steps or restored_sampler.tokens_emitted != sampler.tokens_emitted:
            raise AssertionError("Sharded Orbax checkpoint metadata/sampler state did not restore.")
        if count_params(restored_params) != count_params(sharded_params) or int(restored_opt.step) != int(opt_state.step):
            raise AssertionError("Sharded Orbax checkpoint params/optimizer state did not restore.")
        if "NamedSharding" not in repr(restored_params["layers"][0]["expert_w1"].sharding):
            raise AssertionError("Restored expert weights did not keep named sharding.")
        restored_opt = shard_optimizer_state_for_v6e(restored_opt, restored_params, mesh)
        if "NamedSharding" not in repr(restored_opt.step.sharding):
            raise AssertionError("Restored optimizer step did not become a replicated mesh scalar.")
    print(
        "metis15_jax_mesh_ok "
        f"devices={jax.device_count()} ref_loss={ref_loss:.6f} shard_loss={shard_loss:.6f} "
        f"train_start={losses[0]:.6f} train_end={losses[-1]:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
