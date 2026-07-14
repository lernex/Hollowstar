from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch

from metis_mamba.model import MetisGroupedHeadExperts


def nvtx_range(name: str):
    if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the isolated Metis-1.5 grouped expert MLP path.")
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--rows-per-expert", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=384)
    parser.add_argument("--intermediate-size", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--swiglu-impl", choices=["torch", "liger", "triton", "compiled"], default="liger")
    parser.add_argument("--disable-triton-swiglu", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.disable_triton_swiglu:
        os.environ["METIS_DISABLE_TRITON_SWIGLU"] = "1"
    else:
        os.environ.setdefault("METIS_DISABLE_TRITON_SWIGLU", "0")
    os.environ["METIS_SWIGLU_IMPL"] = args.swiglu_impl

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    total_rows = args.num_experts * args.rows_per_expert
    splits = torch.full((args.num_experts,), args.rows_per_expert, device=device, dtype=torch.int32)

    experts = MetisGroupedHeadExperts(
        args.num_experts,
        args.head_dim,
        args.intermediate_size,
        bias=False,
        use_fp8=False,
        init_std=0.02,
        backend="torch_grouped",
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(experts.parameters(), lr=1e-4, fused=True)

    samples: list[float] = []
    for step in range(args.steps):
        x = torch.randn(total_rows, args.head_dim, device=device, dtype=dtype)
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with nvtx_range("expert_grouped_mlp"):
            y = experts(x, splits, is_first_microbatch=True)
            loss = (y.float().square().mean())
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup_steps:
            samples.append(elapsed)
        if step == args.warmup_steps or step == args.steps - 1:
            rows_per_s = total_rows / max(elapsed, 1e-9)
            print(
                f"step={step} step_s={elapsed:.6f} rows_s={rows_per_s:,.0f} "
                f"loss={loss.item():.6f} swiglu_impl={args.swiglu_impl}",
                flush=True,
            )

    mean_s = sum(samples) / max(len(samples), 1)
    print(
        f"summary rows={total_rows} rows_per_expert={args.rows_per_expert} dtype={args.dtype} "
        f"mean_step_s={mean_s:.6f} mean_rows_s={total_rows / max(mean_s, 1e-9):,.0f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
