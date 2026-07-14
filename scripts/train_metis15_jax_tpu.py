#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Metis-1.5 JAX TPU v6e fixed-shape trainer.")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "configs/metis15_manifest.json")
    parser.add_argument("--stage", choices=["pretrain", "continued_pretrain"], default="pretrain")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--local-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--expert-capacity-factor", type=float, default=None)
    parser.add_argument("--remat-layers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--weight-dtype", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--checkpoint-backend", choices=["orbax", "npz"], default="orbax")
    parser.add_argument("--gcs-checkpoint-dir", default=None)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--expert-execution", choices=["reference", "shard_map"], default="reference")
    parser.add_argument("--tiny-config", action="store_true", help="Use the tiny local proof model instead of full Metis-1.5.")
    parser.add_argument("--allow-cpu-full-model", action="store_true")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "checkpoints/metis15_jax_tpu_v6e")
    args = parser.parse_args()

    from metis_mamba.jax_metis import (
        count_params,
        create_v6e_expert_mesh,
        JaxMetisTrainConfig,
        JaxMemmapTokenData,
        JaxSamplerState,
        init_optimizer_state,
        init_params,
        load_manifest_config,
        make_jit_train_step,
        make_repeated_batch,
        manifest_fingerprint,
        muon_mask,
        qk_clip_with_optimizer_state,
        restore_training_checkpoint,
        save_training_checkpoint,
        shard_batch_for_v6e,
        shard_optimizer_state_for_v6e,
        shard_params_for_v6e,
        stack_microbatches,
        tiny_config,
    )
    import jax
    import jax.numpy as jnp

    if args.gcs_checkpoint_dir:
        destination = str(args.gcs_checkpoint_dir)
        if not destination.startswith("gs://"):
            raise SystemExit("--gcs-checkpoint-dir must start with gs://")
        if shutil.which("gcloud") is None:
            raise SystemExit("--gcs-checkpoint-dir requires the gcloud CLI on PATH.")

    def sync_to_gcs() -> None:
        if not args.gcs_checkpoint_dir:
            return
        destination = str(args.gcs_checkpoint_dir).rstrip("/")
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", str(args.out_dir), destination],
            check=True,
        )

    if args.tiny_config:
        model_cfg = tiny_config(mor=args.stage == "continued_pretrain")
        train_cfg = JaxMetisTrainConfig(
            stage=args.stage,
            local_batch_size=args.local_batch_size or 2,
            grad_accum_steps=args.grad_accum_steps or 1,
            learning_rate=3e-3,
            weight_decay=0.01,
            muon_ns_steps=2,
            max_steps=args.max_steps or 10,
            checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else 1,
        )
    else:
        model_cfg, train_cfg = load_manifest_config(args.manifest, stage=args.stage)
    if args.local_batch_size is not None:
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "local_batch_size": args.local_batch_size})
    if args.grad_accum_steps is not None:
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "grad_accum_steps": args.grad_accum_steps})
    if args.max_steps is not None:
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "max_steps": args.max_steps})
    model_overrides = {"expert_execution": args.expert_execution}
    if args.block_size is not None:
        if args.block_size <= 1:
            raise SystemExit("--block-size must be greater than 1.")
        model_overrides["block_size"] = args.block_size
    if args.expert_capacity_factor is not None:
        if args.expert_capacity_factor <= 0:
            raise SystemExit("--expert-capacity-factor must be positive.")
        model_overrides["expert_capacity_factor"] = args.expert_capacity_factor
    if args.remat_layers is not None:
        model_overrides["remat_layers"] = bool(args.remat_layers)
    if args.dtype is not None:
        model_overrides["dtype"] = args.dtype
    if args.weight_dtype is not None:
        model_overrides["weight_dtype"] = args.weight_dtype
    model_cfg = model_cfg.__class__(**{**model_cfg.__dict__, **model_overrides})
    model_cfg.validate(local_batch_size=train_cfg.local_batch_size)
    if not args.tiny_config and jax.default_backend() == "cpu" and not args.allow_cpu_full_model:
        raise SystemExit(
            "Refusing to initialize full Metis-1.5 on CPU. Use --tiny-config for local proof "
            "or --allow-cpu-full-model for an intentional memory-heavy debug run."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = manifest_fingerprint(args.manifest)
    key = jax.random.PRNGKey(args.seed)
    params = init_params(key, model_cfg)
    expert_mesh = None
    if args.expert_execution == "shard_map":
        expert_mesh = create_v6e_expert_mesh()
        params = shard_params_for_v6e(params, expert_mesh)
    opt_state = init_optimizer_state(params)
    if expert_mesh is not None:
        opt_state = shard_optimizer_state_for_v6e(opt_state, params, expert_mesh)
    mask = muon_mask(params)
    train_step = make_jit_train_step(model_cfg, train_cfg, mask, expert_mesh=expert_mesh)
    data_loader = None
    sampler_state = JaxSamplerState(
        split="synthetic",
        cursor=0,
        epoch=0,
        tokens_emitted=0,
        data_fingerprint=f"synthetic:{args.seed}:{train_cfg.local_batch_size}:{model_cfg.block_size}",
    )
    if args.synthetic_data:
        static_batch_np = make_repeated_batch(
            batch_size=train_cfg.local_batch_size,
            block_size=model_cfg.block_size,
            vocab_size=model_cfg.vocab_size,
        )
    else:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required unless --synthetic-data is set.")
        data_loader = JaxMemmapTokenData(
            args.data_dir,
            split=args.split,
            batch_size=train_cfg.local_batch_size,
            block_size=model_cfg.block_size,
        )
        static_batch_np = None

    start_step = 0
    resume_dir = args.resume_dir or (args.out_dir / "latest")
    if args.resume and resume_dir.is_dir():
        params, opt_state, restored_sampler, metadata = restore_training_checkpoint(
            resume_dir,
            target_params=params,
            target_opt_state=opt_state,
            expected_manifest_hash=manifest_hash,
        )
        start_step = int(metadata.get("step", 0))
        if data_loader is not None:
            data_loader.load_state(restored_sampler)
            sampler_state = data_loader.state
        else:
            sampler_state = restored_sampler
        if expert_mesh is not None:
            params = shard_params_for_v6e(params, expert_mesh)
            opt_state = shard_optimizer_state_for_v6e(opt_state, params, expert_mesh)
        print(f"Resumed JAX checkpoint from {resume_dir} at step {start_step}", flush=True)

    print("Launching Metis-1.5 JAX train on Google Cloud TPU v6e")
    print(
        f"  stage={args.stage} devices={jax.device_count()} local_batch={train_cfg.local_batch_size} "
        f"block={model_cfg.block_size} capacity={model_cfg.capacity_for_batch(train_cfg.local_batch_size)} "
        f"capacity_factor={model_cfg.expert_capacity_factor:g} remat={int(model_cfg.remat_layers)} "
        f"dtype={model_cfg.dtype} weight_dtype={model_cfg.weight_dtype} expert_execution={args.expert_execution}"
    )
    print(
        f"  model layers={model_cfg.n_layer} d_model={model_cfg.d_model} experts={model_cfg.moe_num_experts} "
        f"top_k={model_cfg.moe_top_k} latent={model_cfg.moe_routed_latent_size} params={count_params(params):,}"
    )
    print("  synthetic_data=1" if args.synthetic_data else f"  data_dir={args.data_dir} split={args.split}")

    last = time.perf_counter()
    losses: list[float] = []
    final_metrics = None
    for step in range(start_step + 1, train_cfg.max_steps + 1):
        if data_loader is None:
            batch_np = (
                stack_microbatches([static_batch_np] * train_cfg.grad_accum_steps)
                if train_cfg.grad_accum_steps > 1
                else static_batch_np
            )
            sampler_state = JaxSamplerState(
                split="synthetic",
                cursor=step,
                epoch=0,
                tokens_emitted=step * train_cfg.grad_accum_steps * train_cfg.local_batch_size * model_cfg.block_size,
                data_fingerprint=f"synthetic:{args.seed}:{train_cfg.local_batch_size}:{model_cfg.block_size}",
            )
        else:
            batch_np = (
                stack_microbatches([data_loader.next_batch() for _ in range(train_cfg.grad_accum_steps)])
                if train_cfg.grad_accum_steps > 1
                else data_loader.next_batch()
            )
            sampler_state = data_loader.state
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        if expert_mesh is not None:
            batch = shard_batch_for_v6e(batch, expert_mesh)
        params, opt_state, metrics = train_step(params, opt_state, batch)
        final_metrics = metrics
        if train_cfg.qk_clip_enabled:
            params, opt_state, clip_metrics = qk_clip_with_optimizer_state(params, opt_state, model_cfg)
        else:
            clip_metrics = {"qk_clip_max_logit": 0.0, "qk_clip_min_scale": 1.0, "qk_clip_scaled_layers": 0}
        jax.block_until_ready(metrics["loss"])
        now = time.perf_counter()
        step_s = now - last
        last = now
        loss = float(jax.device_get(metrics["loss"]))
        losses.append(loss)
        tokens = train_cfg.grad_accum_steps * train_cfg.local_batch_size * model_cfg.block_size
        tok_s = tokens / max(step_s, 1e-9)
        print(
            f"step {step:6d} | loss {loss:.6f} | lm {float(jax.device_get(metrics['lm_loss'])):.6f} "
            f"| moe_aux {float(jax.device_get(metrics['moe_aux_loss'])):.6f} "
            f"| mor_aux {float(jax.device_get(metrics['mor_aux_loss'])):.6f} "
            f"| mean_depth {float(jax.device_get(metrics['mean_depth'])):.3f} "
            f"| mor_target {float(jax.device_get(metrics['mor_target_depth'])):.3f} "
            f"| mor_coef {float(jax.device_get(metrics['mor_aux_coef'])):.6f} "
            f"| valid_assign {int(jax.device_get(metrics['valid_assignments']))} "
            f"| total_assign {int(jax.device_get(metrics['total_assignments']))} "
            f"| drop {float(jax.device_get(metrics['expert_drop_frac'])):.4f} "
            f"| accum {int(jax.device_get(metrics['grad_accum_steps']))} "
            f"| tok/s {tok_s:,.0f} | step_s {step_s:.4f} "
            f"| qk_max {clip_metrics['qk_clip_max_logit']:.3f} qk_scale {clip_metrics['qk_clip_min_scale']:.6f} "
            f"qk_scaled_layers {int(clip_metrics['qk_clip_scaled_layers'])} "
            f"| mor_pack_active {int(jax.device_get(metrics['mor_packed_active_tokens']))} "
            f"mor_pack_valid {int(jax.device_get(metrics['mor_packed_valid_tokens']))} "
            f"mor_pack_overflow {float(jax.device_get(metrics['mor_packed_overflow_frac'])):.4f}",
            flush=True,
        )
        checkpoint_interval = (
            args.checkpoint_interval
            if args.checkpoint_interval is not None
            else train_cfg.checkpoint_interval
        )
        if not args.skip_checkpoint and checkpoint_interval > 0 and step % checkpoint_interval == 0:
            save_training_checkpoint(
                args.out_dir / "latest",
                params=params,
                opt_state=opt_state,
                sampler_state=sampler_state,
                step=step,
                manifest_hash=manifest_hash,
                metrics={
                    "loss": loss,
                    "lm_loss": float(jax.device_get(metrics["lm_loss"])),
                    "expert_drop_frac": float(jax.device_get(metrics["expert_drop_frac"])),
                    "tokens_seen": int(sampler_state.tokens_emitted),
                },
                backend=args.checkpoint_backend,
            )
            sync_to_gcs()
    summary = {
        "stage": args.stage,
        "steps": len(losses),
        "start_step": start_step,
        "final_step": train_cfg.max_steps,
        "start_loss": losses[0] if losses else None,
        "end_loss": losses[-1] if losses else None,
        "device_count": jax.device_count(),
        "capacity": model_cfg.capacity_for_batch(train_cfg.local_batch_size),
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "tokens_seen": int(sampler_state.tokens_emitted),
        "mor_target_depth": (
            float(jax.device_get(final_metrics["mor_target_depth"]))
            if final_metrics is not None
            else None
        ),
        "mor_aux_coef": (
            float(jax.device_get(final_metrics["mor_aux_coef"]))
            if final_metrics is not None
            else None
        ),
        "mor_packed_overflow_frac": (
            float(jax.device_get(final_metrics["mor_packed_overflow_frac"]))
            if final_metrics is not None
            else None
        ),
    }
    (args.out_dir / "jax_train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.skip_checkpoint:
        save_training_checkpoint(
            args.out_dir / "latest",
            params=params,
            opt_state=opt_state,
            sampler_state=sampler_state,
            step=train_cfg.max_steps,
            manifest_hash=manifest_hash,
            metrics=summary,
            backend=args.checkpoint_backend,
        )
        sync_to_gcs()


if __name__ == "__main__":
    main()
