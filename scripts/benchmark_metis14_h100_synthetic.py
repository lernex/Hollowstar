from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch

from metis_mamba import MetisMambaConfig, build_model, parse_torch_dtype
from metis_mamba.fp8 import build_fp8_recipe


def nvtx_range(name: str):
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic dense Metis-1.4 throughput sanity benchmark for H100.")
    parser.add_argument("--manifest", default="configs/metis14_h100_dense_manifest.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--te-fused-mlp", action="store_true")
    parser.add_argument("--lm-loss-impl", choices=["standard", "liger_fused_linear_ce"], default=None)
    parser.add_argument("--attention-backend", choices=["auto", "flash_attention_3", "sdpa", "eager"], default=None)
    parser.add_argument("--disable-triton-swiglu", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.disable_triton_swiglu:
        os.environ["METIS_DISABLE_TRITON_SWIGLU"] = "1"

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    manifest = json.loads(Path(args.manifest).read_text())
    config = MetisMambaConfig.from_dict(manifest["model"])
    config.training_mode = "static_dense_pretrain"
    config.mor_enabled = False
    config.mor_train_router = False
    config.mor_runtime_mode = "disabled"
    config.mor_max_depth = 1
    config.mor_target_avg_depth = 1.0
    config.mor_router_aux_loss_coef = 0.0
    config.attention_mask_mode = "causal_none"
    config.disable_depth_stack = True
    config.disable_token_packing = True
    config.disable_token_scatter = True
    if args.lm_loss_impl is not None:
        config.lm_loss_impl = args.lm_loss_impl
    if args.attention_backend is not None:
        config.attention_backend = args.attention_backend
    if args.fp8:
        config.low_precision_mode = "fp8"
    config.te_fused_mlp = bool(args.te_fused_mlp)
    config.validate()

    device = torch.device("cuda")
    recipe = None
    if args.fp8:
        fp8_manifest = manifest.get("hardware", {}).get("fp8", {})
        recipe = build_fp8_recipe(
            format_name=fp8_manifest.get("format", "HYBRID"),
            margin=int(fp8_manifest.get("margin", 0)),
            amax_history_len=int(fp8_manifest.get("amax_history_len", 16)),
            amax_compute_algo=fp8_manifest.get("amax_compute_algo", "max"),
            fp8_dpa=bool(config.fp8_dpa),
            fp8_mha=bool(config.fp8_mha),
        )

    dtype = parse_torch_dtype(args.dtype)
    model = build_model(config, use_fp8=args.fp8, fp8_recipe=recipe).to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

    tokens_per_step = args.batch_size * config.block_size * args.grad_accum_steps
    samples: list[float] = []
    print(
        f"model_params={config.estimate_params():,} batch={args.batch_size} block={config.block_size} "
        f"grad_accum={args.grad_accum_steps} tokens_per_step={tokens_per_step} "
        f"fp8={args.fp8} loss={config.lm_loss_impl} attention={config.attention_backend} "
        f"te_fused_mlp={config.te_fused_mlp} "
        f"triton_swiglu={os.environ.get('METIS_DISABLE_TRITON_SWIGLU', '1') not in {'1', 'true', 'yes', 'on'}}",
        flush=True,
    )
    for step in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = None
        for micro in range(args.grad_accum_steps):
            x = torch.randint(
                0,
                config.vocab_size,
                (args.batch_size, config.block_size),
                device=device,
                dtype=torch.long,
            )
            with nvtx_range("metis14_dense_microbatch"):
                out = model(x, labels=x, is_first_microbatch=(micro == 0), return_logits=False)
                loss = out.loss / args.grad_accum_steps
                loss.backward()
            total_loss = loss.detach() if total_loss is None else total_loss + loss.detach()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup_steps:
            samples.append(elapsed)
        tok_s = tokens_per_step / max(elapsed, 1e-9)
        if step == args.warmup_steps or step == args.steps - 1:
            tflops = (6.0 * config.estimate_params() * tok_s) / 1e12
            print(
                f"step={step} step_s={elapsed:.6f} tok_s={tok_s:,.0f} "
                f"total_tflops={tflops:.1f} loss={float(total_loss):.6f}",
                flush=True,
            )
        if not torch.isfinite(total_loss):
            print(f"nonfinite_loss step={step} loss={float(total_loss)}", flush=True)
            break

    if samples:
        mean_s = sum(samples) / len(samples)
        mean_tok_s = tokens_per_step / max(mean_s, 1e-9)
        mean_tflops = (6.0 * config.estimate_params() * mean_tok_s) / 1e12
        print(
            f"summary mean_step_s={mean_s:.6f} mean_tok_s={mean_tok_s:,.0f} "
            f"mean_total_tflops={mean_tflops:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
