from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from metis_mamba.moe_kernels import fused_swiglu


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Triton fused SwiGLU forward/backward against PyTorch.")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=1280)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    gate_up = torch.randn(args.rows, 2 * args.hidden_size, device=device, dtype=dtype, requires_grad=True)
    ref_input = gate_up.detach().clone().requires_grad_(True)
    grad = torch.randn(args.rows, args.hidden_size, device=device, dtype=dtype)

    out = fused_swiglu(gate_up, args.hidden_size)
    gate, up = ref_input.split(args.hidden_size, dim=-1)
    ref = F.silu(gate) * up

    out.backward(grad)
    ref.backward(grad)
    torch.cuda.synchronize()

    out_err = (out.float() - ref.float()).abs().max().item()
    grad_err = (gate_up.grad.float() - ref_input.grad.float()).abs().max().item()
    print(
        f"rows={args.rows} hidden={args.hidden_size} dtype={args.dtype} "
        f"METIS_DISABLE_TRITON_SWIGLU={os.environ.get('METIS_DISABLE_TRITON_SWIGLU', '0')} "
        f"forward_max_abs={out_err:.6e} backward_max_abs={grad_err:.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
