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
    parser.add_argument("--grad-accum-impl", choices=["loop", "scan"], default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--expert-capacity-factor", type=float, default=None)
    parser.add_argument(
        "--attention-backend",
        choices=["jax_causal_attention_reference", "pallas_flash_attention"],
        default=None,
    )
    parser.add_argument("--moe-backend", choices=["jax_static_sort_pack", "pallas_megablox_gmm"], default=None)
    parser.add_argument("--optimizer", choices=["adamuon", "muon_adamw"], default="adamuon")
    parser.add_argument("--adamuon-matrix-policy", choices=["all", "no_embed_head"], default="all")
    parser.add_argument("--muon-ns-steps", type=int, default=None)
    parser.add_argument("--qk-clip-interval", type=int, default=None)
    parser.add_argument("--qk-clip-warmup-steps", type=int, default=None)
    parser.add_argument("--remat-layers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--remat-attention", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--weight-dtype", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--ce-logits-dtype", choices=["float32", "bfloat16", "model"], default=None)
    parser.add_argument("--ce-loss-impl", choices=["standard", "vocab_parallel"], default=None)
    parser.add_argument("--attention-scores-dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--grad-allreduce-dtype", choices=["float32", "bfloat16"], default=None)
    parser.add_argument("--moe-dispatch-impl", choices=["argsort", "cumsum"], default=None)
    parser.add_argument("--moe-balance-bias-update-rate", type=float, default=None)
    parser.add_argument("--moe-aux-loss-coef", type=float, default=None)
    parser.add_argument("--moe-router-score", choices=["sigmoid", "softmax"], default=None)
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-shuffle-seed", type=int, default=20260611)
    parser.add_argument("--sft-data-dir", type=Path, default=None,
                        help="Use the fixed-shape masked SFT loader (JaxSftData) on this dir instead of "
                        "the pretrain memmap loader. Labels carry -100 prompt masks.")
    parser.add_argument("--base-lr", type=float, default=None, help="Override peak learning rate (e.g. SFT).")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Override warmup ratio (recomputes warmup steps).")
    parser.add_argument(
        "--reset-sampler",
        action="store_true",
        help="Restore params/optimizer from the checkpoint but start the data sampler fresh "
        "(required once when switching an existing run to --data-shuffle).",
    )
    parser.add_argument(
        "--init-from-checkpoint",
        type=Path,
        default=None,
        help="Start a FRESH training phase initialized from a prior checkpoint's PARAMS only "
        "(fresh optimizer state, step 0, fresh sampler, fresh LR schedule). Used to begin "
        "dense CPT from the PT-final weights. Distinct from --resume, which continues a phase.",
    )
    parser.add_argument(
        "--force-dense",
        action="store_true",
        help="Force MoR off for this run regardless of the stage's manifest setting "
        "(runs continued_pretrain as a dense midtrain phase).",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--checkpoint-backend", choices=["orbax", "npz"], default="orbax")
    parser.add_argument("--gcs-checkpoint-dir", default=None)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument(
        "--expert-execution",
        choices=["reference", "shard_map", "pmap_data", "data_parallel"],
        default="data_parallel",
    )
    parser.add_argument("--batch-sharding", choices=["replicated", "data"], default="replicated")
    parser.add_argument("--mesh-shape", choices=["1x8", "2x4"], default="1x8")
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
        make_pmap_data_parallel_train_step,
        make_repeated_batch,
        manifest_fingerprint,
        mesh_axis_size,
        optimizer_matrix_mask,
        put_sharded_for_pmap,
        replicate_for_pmap,
        JaxSftData,
        restore_training_checkpoint,
        restore_params_only,
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
        # Best-effort: checkpoint durability is local-first; a failed upload must
        # warn, retry, and never kill the training run.
        if not args.gcs_checkpoint_dir:
            return
        destination = str(args.gcs_checkpoint_dir).rstrip("/")
        # rsync needs storage.buckets.get on the bucket (granted to the TPU
        # service account); it is idempotent and skips unchanged files.
        for attempt in range(1, 4):
            # Mirror semantics: delete superseded checkpoint files in GCS,
            # otherwise every orbax save accumulates (~14GB per checkpoint).
            result = subprocess.run(
                [
                    "gcloud", "storage", "rsync", "-r",
                    "--delete-unmatched-destination-objects",
                    str(args.out_dir), destination,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
            print(
                f"WARNING: GCS checkpoint sync attempt {attempt}/3 failed: "
                f"{(result.stderr or result.stdout).strip()[-300:]}",
                flush=True,
            )
            time.sleep(10)
        print(
            "WARNING: GCS checkpoint sync failed after 3 attempts; continuing training "
            "(local checkpoint intact).",
            flush=True,
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
    if args.grad_accum_impl is not None:
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "grad_accum_impl": args.grad_accum_impl})
    train_cfg = train_cfg.__class__(
        **{
            **train_cfg.__dict__,
            "optimizer": args.optimizer,
            "adamuon_matrix_policy": args.adamuon_matrix_policy,
        }
    )
    if args.max_steps is not None:
        step_overrides = {"max_steps": args.max_steps}
        if train_cfg.warmup_ratio > 0:
            step_overrides["warmup_steps"] = max(1, int(round(train_cfg.warmup_ratio * args.max_steps)))
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, **step_overrides})
    if args.log_interval is not None:
        if args.log_interval < 1:
            raise SystemExit("--log-interval must be >= 1.")
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "log_interval": args.log_interval})
    if args.base_lr is not None:
        if args.base_lr <= 0:
            raise SystemExit("--base-lr must be positive.")
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "learning_rate": args.base_lr})
    if args.warmup_ratio is not None:
        if not (0.0 <= args.warmup_ratio < 1.0):
            raise SystemExit("--warmup-ratio must be in [0, 1).")
        train_cfg = train_cfg.__class__(**{
            **train_cfg.__dict__, "warmup_ratio": args.warmup_ratio,
            "warmup_steps": max(1, int(round(args.warmup_ratio * train_cfg.max_steps)))})
    if args.muon_ns_steps is not None:
        if args.muon_ns_steps <= 0:
            raise SystemExit("--muon-ns-steps must be positive.")
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "muon_ns_steps": args.muon_ns_steps})
    if args.qk_clip_interval is not None:
        if args.qk_clip_interval < 1:
            raise SystemExit("--qk-clip-interval must be >= 1.")
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "qk_clip_interval": args.qk_clip_interval})
    if args.qk_clip_warmup_steps is not None:
        if args.qk_clip_warmup_steps < 0:
            raise SystemExit("--qk-clip-warmup-steps must be >= 0.")
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "qk_clip_warmup_steps": args.qk_clip_warmup_steps})
    model_overrides = {"expert_execution": args.expert_execution}
    if args.block_size is not None:
        if args.block_size <= 1:
            raise SystemExit("--block-size must be greater than 1.")
        model_overrides["block_size"] = args.block_size
    if args.expert_capacity_factor is not None:
        if args.expert_capacity_factor <= 0:
            raise SystemExit("--expert-capacity-factor must be positive.")
        model_overrides["expert_capacity_factor"] = args.expert_capacity_factor
    if args.attention_backend is not None:
        model_overrides["attention_backend"] = args.attention_backend
    if args.moe_backend is not None:
        model_overrides["moe_backend"] = args.moe_backend
    if args.remat_layers is not None:
        model_overrides["remat_layers"] = bool(args.remat_layers)
    if args.remat_attention is not None:
        model_overrides["remat_attention"] = bool(args.remat_attention)
    if args.dtype is not None:
        model_overrides["dtype"] = args.dtype
    if args.weight_dtype is not None:
        model_overrides["weight_dtype"] = args.weight_dtype
    if args.ce_logits_dtype is not None:
        model_overrides["ce_logits_dtype"] = args.ce_logits_dtype
    if args.ce_loss_impl is not None:
        model_overrides["ce_loss_impl"] = args.ce_loss_impl
    if args.attention_scores_dtype is not None:
        model_overrides["attention_scores_dtype"] = args.attention_scores_dtype
    if args.moe_dispatch_impl is not None:
        model_overrides["moe_dispatch_impl"] = args.moe_dispatch_impl
    if args.moe_balance_bias_update_rate is not None:
        if args.moe_balance_bias_update_rate < 0:
            raise SystemExit("--moe-balance-bias-update-rate cannot be negative.")
        model_overrides["moe_balance_bias_update_rate"] = args.moe_balance_bias_update_rate
    if args.moe_aux_loss_coef is not None:
        if args.moe_aux_loss_coef < 0:
            raise SystemExit("--moe-aux-loss-coef cannot be negative.")
        model_overrides["moe_aux_loss_coef"] = args.moe_aux_loss_coef
    if args.moe_router_score is not None:
        model_overrides["moe_router_score"] = args.moe_router_score
    if args.force_dense:
        # Run any stage as dense pretraining: MoR fully disabled. Keeps the
        # stage's other hyperparameters (LR, token budget) from the manifest.
        model_overrides["training_mode"] = "static_dense_pretrain"
        model_overrides["mor_enabled"] = False
        model_overrides["mor_runtime_mode"] = "disabled"
        model_overrides["mor_compute_mode"] = "soft_fixed_depth"
    if args.grad_allreduce_dtype is not None:
        train_cfg = train_cfg.__class__(**{**train_cfg.__dict__, "grad_allreduce_dtype": args.grad_allreduce_dtype})
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
    param_count = count_params(params)
    expert_mesh = None
    if args.expert_execution == "shard_map":
        expert_mesh = create_v6e_expert_mesh(mesh_shape=args.mesh_shape)
        params = shard_params_for_v6e(params, expert_mesh)
    opt_state = init_optimizer_state(params)
    if expert_mesh is not None:
        opt_state = shard_optimizer_state_for_v6e(opt_state, params, expert_mesh)
    mask = optimizer_matrix_mask(
        params,
        train_cfg.optimizer,
        adamuon_matrix_policy=train_cfg.adamuon_matrix_policy,
    )
    is_data_parallel = args.expert_execution in {"pmap_data", "data_parallel"}
    dp_devices = jax.devices() if is_data_parallel else None
    if is_data_parallel:
        if train_cfg.local_batch_size % len(dp_devices) != 0:
            raise SystemExit(
                f"--local-batch-size must be divisible by device count ({len(dp_devices)}) for data_parallel."
            )
        train_step = make_pmap_data_parallel_train_step(model_cfg, train_cfg, mask)
    else:
        if args.batch_sharding == "data" and args.expert_execution != "shard_map":
            raise SystemExit("--batch-sharding=data currently requires --expert-execution=shard_map.")
        data_shard_size = (
            mesh_axis_size(expert_mesh, "data", default=jax.device_count())
            if expert_mesh is not None
            else jax.device_count()
        )
        if args.batch_sharding == "data" and train_cfg.local_batch_size % data_shard_size != 0:
            raise SystemExit(
                f"--local-batch-size must be divisible by data shard size ({data_shard_size}) for data batch sharding."
            )
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
    elif args.sft_data_dir is not None:
        data_loader = JaxSftData(
            args.sft_data_dir,
            split=args.split,
            batch_size=train_cfg.local_batch_size,
            block_size=model_cfg.block_size,
            shuffle=bool(args.data_shuffle),
            shuffle_seed=int(args.data_shuffle_seed),
        )
        static_batch_np = None
    else:
        if args.data_dir is None:
            raise SystemExit("--data-dir, --sft-data-dir, or --synthetic-data is required.")
        data_loader = JaxMemmapTokenData(
            args.data_dir,
            split=args.split,
            batch_size=train_cfg.local_batch_size,
            block_size=model_cfg.block_size,
            shuffle=bool(args.data_shuffle),
            shuffle_seed=int(args.data_shuffle_seed),
        )
        static_batch_np = None

    start_step = 0
    if args.init_from_checkpoint is not None:
        # Fresh phase from prior params only. Optimizer state, step counter, LR
        # schedule, and sampler all start clean — this is a new training phase
        # (e.g. dense CPT), not a resume. Manifest-hash check is intentionally
        # skipped: we are deliberately loading weights from a different stage.
        init_dir = args.init_from_checkpoint
        if not init_dir.is_dir():
            raise SystemExit(f"--init-from-checkpoint dir not found: {init_dir}")
        # Params ONLY — never load the checkpoint's optimizer onto the device, or
        # it transiently doubles optimizer HBM (loaded + fresh) and OOMs the program.
        params, init_meta = restore_params_only(init_dir, target_params=params, target_opt_state=opt_state)
        opt_state = init_optimizer_state(params)  # fresh moments + step 0
        if expert_mesh is not None:
            params = shard_params_for_v6e(params, expert_mesh)
            opt_state = shard_optimizer_state_for_v6e(opt_state, params, expert_mesh)
        if data_loader is not None:
            sampler_state = data_loader.state
        print(
            f"Initialized fresh phase from {init_dir} "
            f"(prior step {int(init_meta.get('step', 0))}); optimizer/step/sampler reset.",
            flush=True,
        )
    resume_dir = args.resume_dir or (args.out_dir / "latest")
    if args.init_from_checkpoint is None and args.resume and resume_dir.is_dir():
        params, opt_state, restored_sampler, metadata = restore_training_checkpoint(
            resume_dir,
            target_params=params,
            target_opt_state=opt_state,
            expected_manifest_hash=manifest_hash,
        )
        start_step = int(metadata.get("step", 0))
        if data_loader is not None:
            if args.reset_sampler:
                print("Sampler state reset requested; starting fresh data pass (params/optimizer restored).", flush=True)
                sampler_state = data_loader.state
            else:
                data_loader.load_state(restored_sampler)
                sampler_state = data_loader.state
        else:
            sampler_state = restored_sampler
        if expert_mesh is not None:
            params = shard_params_for_v6e(params, expert_mesh)
            opt_state = shard_optimizer_state_for_v6e(opt_state, params, expert_mesh)
        print(f"Resumed JAX checkpoint from {resume_dir} at step {start_step}", flush=True)

    if is_data_parallel:
        # Replicate once; the pmapped step keeps state on-device (donated buffers).
        params = replicate_for_pmap(params, dp_devices)
        opt_state = replicate_for_pmap(opt_state, dp_devices)

    print("Launching Metis-1.5 JAX train on Google Cloud TPU v6e")
    capacity_batch = train_cfg.local_batch_size
    if is_data_parallel:
        capacity_batch = train_cfg.local_batch_size // jax.device_count()
    print(
        f"  stage={args.stage} devices={jax.device_count()} local_batch={train_cfg.local_batch_size} "
        f"per_device_batch={capacity_batch} optimizer={train_cfg.optimizer} "
        f"adamuon_matrix_policy={train_cfg.adamuon_matrix_policy} "
        f"block={model_cfg.block_size} attention_backend={model_cfg.attention_backend} "
        f"moe_backend={model_cfg.moe_backend} "
        f"capacity={model_cfg.capacity_for_batch(capacity_batch)} "
        f"capacity_factor={model_cfg.expert_capacity_factor:g} remat={int(model_cfg.remat_layers)} "
        f"attention_remat={int(model_cfg.remat_attention)} "
        f"dtype={model_cfg.dtype} weight_dtype={model_cfg.weight_dtype} "
        f"ce_logits_dtype={model_cfg.ce_logits_dtype} ce_loss_impl={model_cfg.ce_loss_impl} "
        f"expert_execution={args.expert_execution} "
        f"batch_sharding={args.batch_sharding} mesh_shape={args.mesh_shape} muon_ns_steps={train_cfg.muon_ns_steps} "
        f"grad_accum_impl={train_cfg.grad_accum_impl} "
        f"qk_clip_interval={train_cfg.qk_clip_interval} qk_clip_warmup_steps={train_cfg.qk_clip_warmup_steps}"
    )
    print(
        f"  model layers={model_cfg.n_layer} d_model={model_cfg.d_model} experts={model_cfg.moe_num_experts} "
        f"top_k={model_cfg.moe_top_k} latent={model_cfg.moe_routed_latent_size} params={param_count:,}"
    )
    print("  synthetic_data=1" if args.synthetic_data else f"  data_dir={args.data_dir} split={args.split}")

    losses: list[float] = []
    final_metrics = None

    def first_replica_scalar(value) -> float:
        return float(np.asarray(jax.device_get(value)).reshape(-1)[0])

    def first_replica_int(value) -> int:
        return int(np.asarray(jax.device_get(value)).reshape(-1)[0])

    def unreplicate_tree(tree):
        if not is_data_parallel:
            return tree
        return jax.tree_util.tree_map(lambda leaf: leaf[0], tree)

    def split_for_pmap(value: np.ndarray):
        # Shard on host (numpy views) and place each slice directly on its
        # device; never stage the whole batch through device 0.
        device_count = len(dp_devices)
        if value.ndim == 3:
            accum, global_batch, seq = value.shape
            per_device = global_batch // device_count
            arr = value.reshape(accum, device_count, per_device, seq).transpose(1, 0, 2, 3)
        else:
            global_batch, seq = value.shape
            per_device = global_batch // device_count
            arr = value.reshape(device_count, per_device, seq)
        return put_sharded_for_pmap([arr[d] for d in range(device_count)], dp_devices)

    checkpoint_interval = (
        args.checkpoint_interval
        if args.checkpoint_interval is not None
        else train_cfg.checkpoint_interval
    )
    tokens_per_step = train_cfg.grad_accum_steps * train_cfg.local_batch_size * model_cfg.block_size
    last_log_time = time.perf_counter()
    last_log_step = start_step

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
        if expert_mesh is not None:
            batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
            batch = shard_batch_for_v6e(batch, expert_mesh, batch_sharding=args.batch_sharding)
        elif is_data_parallel:
            batch = {key: split_for_pmap(np.asarray(value)) for key, value in batch_np.items()}
        else:
            batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        params, opt_state, metrics = train_step(params, opt_state, batch)
        final_metrics = metrics
        # The step is dispatched asynchronously; only synchronize with the device
        # on logging/checkpoint boundaries so compute, host data prep, and
        # metric transfers overlap instead of serializing every step.
        should_checkpoint = not args.skip_checkpoint and checkpoint_interval > 0 and step % checkpoint_interval == 0
        should_log = (
            step == start_step + 1
            or step % train_cfg.log_interval == 0
            or step == train_cfg.max_steps
            or should_checkpoint
        )
        if not should_log:
            continue
        jax.block_until_ready(metrics["loss"])
        now = time.perf_counter()
        window_steps = max(1, step - last_log_step)
        step_s = (now - last_log_time) / window_steps
        last_log_time = now
        last_log_step = step
        loss = first_replica_scalar(metrics["loss"])
        losses.append(loss)
        tok_s = tokens_per_step / max(step_s, 1e-9)
        print(
            f"step {step:6d} | loss {loss:.6f} | lm {first_replica_scalar(metrics['lm_loss']):.6f} "
            f"| moe_aux {first_replica_scalar(metrics['moe_aux_loss']):.6f} "
            f"| mor_aux {first_replica_scalar(metrics['mor_aux_loss']):.6f} "
            f"| mean_depth {first_replica_scalar(metrics['mean_depth']):.3f} "
            f"| mor_target {first_replica_scalar(metrics['mor_target_depth']):.3f} "
            f"| mor_coef {first_replica_scalar(metrics['mor_aux_coef']):.6f} "
            f"| valid_assign {first_replica_int(metrics['valid_assignments'])} "
            f"| total_assign {first_replica_int(metrics['total_assignments'])} "
            f"| drop {first_replica_scalar(metrics['expert_drop_frac']):.4f} "
            f"| accum {first_replica_int(metrics['grad_accum_steps'])} "
            f"| tok/s {tok_s:,.0f} | step_s {step_s:.4f} "
            f"| qk_max {first_replica_scalar(metrics['qk_clip_max_logit']):.3f} "
            f"qk_scale {first_replica_scalar(metrics['qk_clip_min_scale']):.6f} "
            f"qk_scaled_layers {first_replica_int(metrics['qk_clip_scaled_layers'])} "
            f"| mor_pack_active {first_replica_int(metrics['mor_packed_active_tokens'])} "
            f"mor_pack_valid {first_replica_int(metrics['mor_packed_valid_tokens'])} "
            f"mor_pack_overflow {first_replica_scalar(metrics['mor_packed_overflow_frac']):.4f} "
            f"| lr {first_replica_scalar(metrics['learning_rate']):.6e}",
            flush=True,
        )
        if should_checkpoint:
            save_training_checkpoint(
                args.out_dir / "latest",
                params=unreplicate_tree(params),
                opt_state=unreplicate_tree(opt_state),
                sampler_state=sampler_state,
                step=step,
                manifest_hash=manifest_hash,
                metrics={
                    "loss": loss,
                    "lm_loss": first_replica_scalar(metrics["lm_loss"]),
                    "expert_drop_frac": first_replica_scalar(metrics["expert_drop_frac"]),
                    "tokens_seen": int(sampler_state.tokens_emitted),
                },
                backend=args.checkpoint_backend,
            )
            sync_to_gcs()
    summary = {
        "stage": args.stage,
        "steps": max(0, train_cfg.max_steps - start_step),
        "logged_steps": len(losses),
        "log_interval": train_cfg.log_interval,
        "start_step": start_step,
        "final_step": train_cfg.max_steps,
        "start_loss": losses[0] if losses else None,
        "end_loss": losses[-1] if losses else None,
        "device_count": jax.device_count(),
        "optimizer": train_cfg.optimizer,
        "expert_execution": args.expert_execution,
        "capacity": model_cfg.capacity_for_batch(train_cfg.local_batch_size),
        "capacity_per_device": model_cfg.capacity_for_batch(capacity_batch),
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "tokens_seen": int(sampler_state.tokens_emitted),
        "mor_target_depth": (
            first_replica_scalar(final_metrics["mor_target_depth"])
            if final_metrics is not None
            else None
        ),
        "mor_aux_coef": (
            first_replica_scalar(final_metrics["mor_aux_coef"])
            if final_metrics is not None
            else None
        ),
        "mor_packed_overflow_frac": (
            first_replica_scalar(final_metrics["mor_packed_overflow_frac"])
            if final_metrics is not None
            else None
        ),
    }
    (args.out_dir / "jax_train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.skip_checkpoint:
        save_training_checkpoint(
            args.out_dir / "latest",
            params=unreplicate_tree(params),
            opt_state=unreplicate_tree(opt_state),
            sampler_state=sampler_state,
            step=train_cfg.max_steps,
            manifest_hash=manifest_hash,
            metrics=summary,
            backend=args.checkpoint_backend,
        )
        sync_to_gcs()


if __name__ == "__main__":
    main()
