#!/usr/bin/env python3
"""Regression proofs for the 2026-06 Metis-1.5 JAX training fixes.

Covers, on CPU (with 4 forced host devices for the pmap proof):
  1. Loader label convention: labels == input_ids (no pre-shift double-shift).
  2. RoPE: relative-position invariance of rotated q.k dot products.
  3. fp32 master weights: params actually move at production-scale LR, while a
     bf16-stored control moves far fewer entries (sub-ULP rounding).
  4. Aux-loss-free balance bias: overloaded expert bias is pushed down.
  5. LR schedule: warmup then cosine decay to the floor.
  6. QK-clip folded into train_step (works under jit and data-parallel pmap).
  7. pmap data-parallel step over replicated+donated state: replicas stay in
     sync and the tiny loss improves.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np


def main() -> None:
    from metis_mamba.jax_metis import (
        JaxMemmapTokenData,
        JaxMetisTrainConfig,
        _apply_rope,
        _rope_cos_sin,
        init_optimizer_state,
        init_params,
        make_jit_train_step,
        make_pmap_data_parallel_train_step,
        make_repeated_batch,
        optimizer_matrix_mask,
        put_sharded_for_pmap,
        replicate_for_pmap,
        scheduled_learning_rate,
        stack_microbatches,
        tiny_config,
        train_step,
    )
    import jax
    import jax.numpy as jnp

    # ------------------------------------------------------------------ 1. loader convention
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        tokens = (np.arange(4096) % 61).astype(np.uint16)
        tokens.tofile(data_dir / "train.bin")
        (data_dir / "meta.json").write_text(
            json.dumps({"dtype": "uint16", "vocab_size": 64, "train_tokens": int(tokens.size)}),
            encoding="utf-8",
        )
        loader = JaxMemmapTokenData(data_dir, split="train", batch_size=2, block_size=16)
        batch = loader.next_batch()
        if not np.array_equal(batch["input_ids"], batch["labels"]):
            raise AssertionError(
                "Loader must return labels == input_ids; forward() shifts internally. "
                "Pre-shifted labels double-shift into predict-2-ahead training."
            )
    print("label_convention_ok", flush=True)

    # ------------------------------------------------------------------ 2. RoPE relative invariance
    head_dim = 8
    key = jax.random.PRNGKey(0)
    qk_key, kk_key = jax.random.split(key)
    q_vec = jax.random.normal(qk_key, (1, 1, head_dim), dtype=jnp.float32)
    k_vec = jax.random.normal(kk_key, (1, 1, head_dim), dtype=jnp.float32)

    def rotated_dot(q_pos: int, k_pos: int) -> float:
        cos_q, sin_q = _rope_cos_sin(jnp.asarray([q_pos]), head_dim, 10000.0)
        cos_k, sin_k = _rope_cos_sin(jnp.asarray([k_pos]), head_dim, 10000.0)
        q_rot = _apply_rope(q_vec, cos_q[:, None, :], sin_q[:, None, :])
        k_rot = _apply_rope(k_vec, cos_k[:, None, :], sin_k[:, None, :])
        return float(jnp.sum(q_rot * k_rot))

    base = rotated_dot(3, 1)
    shifted = rotated_dot(10, 8)
    different = rotated_dot(9, 1)
    if abs(base - shifted) > 1e-4:
        raise AssertionError(f"RoPE q.k must depend only on relative offset: {base} vs {shifted}")
    if abs(base - different) < 1e-4:
        raise AssertionError("RoPE q.k must differ across different relative offsets.")
    norm_before = float(jnp.linalg.norm(q_vec))
    cos_q, sin_q = _rope_cos_sin(jnp.asarray([7]), head_dim, 10000.0)
    norm_after = float(jnp.linalg.norm(_apply_rope(q_vec, cos_q[:, None, :], sin_q[:, None, :])))
    if abs(norm_before - norm_after) > 1e-4:
        raise AssertionError("RoPE must be norm-preserving (a pure rotation).")
    print("rope_relative_invariance_ok", flush=True)

    # ------------------------------------------------------------------ 3. fp32 master weights move at real LR
    def moved_fraction(weight_dtype: str) -> float:
        cfg = tiny_config()
        cfg = cfg.__class__(**{**cfg.__dict__, "weight_dtype": weight_dtype})
        cfg.validate(local_batch_size=2)
        train_cfg = JaxMetisTrainConfig(
            local_batch_size=2,
            grad_accum_steps=1,
            learning_rate=1.5e-4,  # production-scale LR: the regime where bf16 storage rounds updates away
            weight_decay=0.0,
            optimizer="adamuon",
            muon_ns_steps=2,
            qk_clip_enabled=False,
            max_steps=10,
        )
        params = init_params(jax.random.PRNGKey(1), cfg)
        mask = optimizer_matrix_mask(params, train_cfg.optimizer)
        opt_state = init_optimizer_state(params)
        batch = {k: jnp.asarray(v) for k, v in make_repeated_batch(batch_size=2, block_size=cfg.block_size, vocab_size=cfg.vocab_size).items()}
        before = np.asarray(jax.device_get(params["embed"]), dtype=np.float64)
        new_params, _, _ = train_step(params, opt_state, batch, cfg, train_cfg, mask)
        after = np.asarray(jax.device_get(new_params["embed"]), dtype=np.float64)
        return float(np.mean(before != after))

    frac_fp32 = moved_fraction("float32")
    frac_bf16 = moved_fraction("bfloat16")
    if frac_fp32 < 0.9:
        raise AssertionError(f"fp32-stored embeddings should move almost everywhere at lr=1.5e-4; moved {frac_fp32:.3f}")
    if frac_bf16 >= frac_fp32:
        raise AssertionError(
            f"bf16-stored weights should lose updates to rounding vs fp32 (bf16 {frac_bf16:.3f} vs fp32 {frac_fp32:.3f})"
        )
    print(f"fp32_master_weights_ok fp32_moved={frac_fp32:.3f} bf16_moved={frac_bf16:.3f}", flush=True)

    # ------------------------------------------------------------------ 4. balance bias pushes against overload
    cfg = tiny_config()
    cfg.validate(local_batch_size=2)
    train_cfg = JaxMetisTrainConfig(
        local_batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        optimizer="adamuon",
        muon_ns_steps=2,
        qk_clip_enabled=False,
        max_steps=10,
    )
    params = init_params(jax.random.PRNGKey(2), cfg)
    layer0 = dict(params["layers"][0])
    forced_router = np.zeros((cfg.d_model, cfg.moe_num_experts), dtype=np.float32)
    forced_router[:, 0] = 1.0  # every token loves expert 0
    layer0["router"] = jnp.asarray(forced_router)
    params["layers"] = (layer0,) + tuple(params["layers"][1:])
    mask = optimizer_matrix_mask(params, train_cfg.optimizer)
    opt_state = init_optimizer_state(params)
    batch = {k: jnp.asarray(v) for k, v in make_repeated_batch(batch_size=2, block_size=cfg.block_size, vocab_size=cfg.vocab_size).items()}
    for _ in range(3):
        params, opt_state, metrics = train_step(params, opt_state, batch, cfg, train_cfg, mask)
    bias = np.asarray(jax.device_get(params["layers"][0]["router_bias"]))
    loads = np.asarray(jax.device_get(metrics["expert_load_per_layer"]))[0]
    if loads[0] <= 1.0 / cfg.moe_num_experts:
        raise AssertionError("Test setup failed: expert 0 should be overloaded.")
    if not (bias[0] < 0.0 and bias[0] == bias.min()):
        raise AssertionError(f"Balance bias must push the overloaded expert down; got {bias}")
    if float(np.max(bias[1:])) <= 0.0:
        raise AssertionError(f"Underloaded experts should receive positive bias; got {bias}")
    print(f"balance_bias_ok bias0={bias[0]:.4f} max_other={float(np.max(bias[1:])):.4f}", flush=True)

    # ------------------------------------------------------------------ 5. LR schedule shape
    sched_cfg = JaxMetisTrainConfig(
        learning_rate=1e-3,
        warmup_steps=10,
        max_steps=100,
        lr_schedule="warmup_cosine",
        lr_min_ratio=0.1,
    )
    lr_1 = float(scheduled_learning_rate(sched_cfg, jnp.asarray(1)))
    lr_warm = float(scheduled_learning_rate(sched_cfg, jnp.asarray(10)))
    lr_mid = float(scheduled_learning_rate(sched_cfg, jnp.asarray(55)))
    lr_end = float(scheduled_learning_rate(sched_cfg, jnp.asarray(100)))
    if not (abs(lr_1 - 1e-4) < 1e-9 and abs(lr_warm - 1e-3) < 1e-9):
        raise AssertionError(f"Warmup is off: lr(1)={lr_1}, lr(10)={lr_warm}")
    if not (lr_end < lr_mid < lr_warm):
        raise AssertionError(f"Cosine decay is off: {lr_warm} -> {lr_mid} -> {lr_end}")
    if abs(lr_end - 1e-4) > 1e-8:
        raise AssertionError(f"Final LR should hit the floor ratio: {lr_end}")
    constant_cfg = JaxMetisTrainConfig(learning_rate=1e-3, lr_schedule="constant")
    if abs(float(scheduled_learning_rate(constant_cfg, jnp.asarray(1))) - 1e-3) > 1e-9:
        raise AssertionError("Constant schedule must return base LR.")
    print("lr_schedule_ok", flush=True)

    # ------------------------------------------------------------------ 6. QK-clip inside train_step
    cfg = tiny_config()
    clip_cfg = cfg.__class__(**{**cfg.__dict__, "qk_clip_threshold": 1e-8})
    clip_cfg.validate(local_batch_size=2)
    train_cfg = JaxMetisTrainConfig(
        local_batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-4,
        optimizer="adamuon",
        muon_ns_steps=2,
        qk_clip_enabled=True,
        qk_clip_interval=1,
        qk_clip_warmup_steps=0,
        max_steps=10,
    )
    params = init_params(jax.random.PRNGKey(3), clip_cfg)
    mask = optimizer_matrix_mask(params, train_cfg.optimizer)
    opt_state = init_optimizer_state(params)
    q_norm_before = float(jnp.linalg.norm(params["layers"][0]["q"].astype(jnp.float32)))
    params, opt_state, metrics = train_step(params, opt_state, batch, clip_cfg, train_cfg, mask)
    q_norm_after = float(jnp.linalg.norm(params["layers"][0]["q"].astype(jnp.float32)))
    if float(jax.device_get(metrics["qk_clip_scaled_layers"])) < 1.0:
        raise AssertionError("In-step QK clip did not report scaled layers under a forced threshold.")
    if q_norm_after > 0.1 * q_norm_before:
        raise AssertionError("In-step QK clip did not shrink Q weights under a forced threshold.")
    print("qk_clip_in_step_ok", flush=True)

    # ------------------------------------------------------------------ 7. pmap data-parallel replicated state
    device_count = jax.device_count()
    if device_count < 2:
        raise AssertionError("Expected forced multi-device CPU run (XLA_FLAGS host platform device count).")
    cfg = tiny_config()
    cfg = cfg.__class__(**{**cfg.__dict__, "expert_execution": "data_parallel"})
    cfg.validate(local_batch_size=device_count)
    train_cfg = JaxMetisTrainConfig(
        local_batch_size=device_count,
        grad_accum_steps=2,
        grad_accum_impl="scan",
        learning_rate=3e-3,
        weight_decay=0.01,
        optimizer="adamuon",
        muon_ns_steps=2,
        max_steps=12,
    )
    params = init_params(jax.random.PRNGKey(4), cfg)
    mask = optimizer_matrix_mask(params, train_cfg.optimizer)
    opt_state = init_optimizer_state(params)
    devices = jax.devices()
    params = replicate_for_pmap(params, devices)
    opt_state = replicate_for_pmap(opt_state, devices)
    pmap_step = make_pmap_data_parallel_train_step(cfg, train_cfg, mask)
    micro = make_repeated_batch(batch_size=device_count, block_size=cfg.block_size, vocab_size=cfg.vocab_size)
    batch_np = stack_microbatches([micro, micro])

    def split_for_pmap(value: np.ndarray):
        accum, global_batch, seq = value.shape
        per_device = global_batch // device_count
        arr = value.reshape(accum, device_count, per_device, seq).transpose(1, 0, 2, 3)
        return put_sharded_for_pmap([arr[d] for d in range(device_count)], devices)

    batch = {k: split_for_pmap(np.asarray(v)) for k, v in batch_np.items()}
    pmap_losses = []
    for _ in range(train_cfg.max_steps):
        params, opt_state, metrics = pmap_step(params, opt_state, batch)
        pmap_losses.append(float(np.asarray(jax.device_get(metrics["loss"]))[0]))
    if not all(np.isfinite(pmap_losses)):
        raise AssertionError(f"pmap losses must be finite: {pmap_losses}")
    if pmap_losses[-1] >= pmap_losses[0] * 0.99:
        raise AssertionError(f"pmap tiny loss did not improve: {pmap_losses[0]:.4f} -> {pmap_losses[-1]:.4f}")
    embed = np.asarray(jax.device_get(params["embed"]))
    for replica in range(1, device_count):
        if not np.allclose(embed[0], embed[replica]):
            raise AssertionError("Replicated params diverged across data-parallel devices.")
    step_values = np.asarray(jax.device_get(opt_state.step)).reshape(-1)
    if not np.all(step_values == train_cfg.max_steps):
        raise AssertionError(f"Optimizer step must advance identically on all replicas: {step_values}")
    print(
        f"pmap_data_parallel_ok devices={device_count} start={pmap_losses[0]:.4f} end={pmap_losses[-1]:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------ 8. jit path still healthy end-to-end
    cfg = tiny_config()
    cfg.validate(local_batch_size=2)
    train_cfg = JaxMetisTrainConfig(
        local_batch_size=2,
        grad_accum_steps=2,
        grad_accum_impl="scan",
        learning_rate=3e-3,
        weight_decay=0.01,
        optimizer="adamuon",
        muon_ns_steps=2,
        max_steps=12,
    )
    params = init_params(jax.random.PRNGKey(5), cfg)
    mask = optimizer_matrix_mask(params, train_cfg.optimizer)
    opt_state = init_optimizer_state(params)
    jit_step = make_jit_train_step(cfg, train_cfg, mask)
    micro = make_repeated_batch(batch_size=2, block_size=cfg.block_size, vocab_size=cfg.vocab_size)
    batch = {k: jnp.asarray(v) for k, v in stack_microbatches([micro, micro]).items()}
    jit_losses = []
    for _ in range(train_cfg.max_steps):
        params, opt_state, metrics = jit_step(params, opt_state, batch)
        jit_losses.append(float(jax.device_get(metrics["loss"])))
    if jit_losses[-1] >= jit_losses[0] * 0.99:
        raise AssertionError(f"jit tiny loss did not improve: {jit_losses[0]:.4f} -> {jit_losses[-1]:.4f}")
    print(f"jit_train_ok start={jit_losses[0]:.4f} end={jit_losses[-1]:.4f}", flush=True)

    print("metis15_jax_training_fixes_ok", flush=True)


if __name__ == "__main__":
    main()
