from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
import time
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metis_mamba.fp8 import build_fp8_recipe, fp8_autocast_context
from metis_mamba.model import MetisGroupedHeadExperts


SHAPES = [
    ("qkv", 24576, 1536, 2560),
    ("attn_o", 24576, 1536, 1536),
    ("routed_down", 98304, 384, 384),
    ("expert_gate_up", 12288, 384, 2560),
    ("expert_down", 12288, 1280, 384),
    ("shared_gate_up", 98304, 384, 2560),
    ("shared_down", 98304, 1280, 384),
    ("lm_head", 24576, 1536, 32768),
]


def make_recipe(
    name: str,
    *,
    nvfp4_disable_rht: bool,
    nvfp4_disable_2d_quantization: bool,
    nvfp4_disable_stochastic_rounding: bool,
):
    if name == "bf16":
        return None
    if name == "fp8":
        return build_fp8_recipe()
    import transformer_engine.common.recipe as recipe

    if name == "nvfp4":
        return recipe.NVFP4BlockScaling(
            disable_rht=nvfp4_disable_rht,
            disable_2d_quantization=nvfp4_disable_2d_quantization,
            disable_stochastic_rounding=nvfp4_disable_stochastic_rounding,
        )
    if name == "mxfp8":
        return recipe.MXFP8BlockScaling()
    if name == "fp8_block":
        return recipe.Float8BlockScaling()
    raise ValueError(f"Unknown recipe: {name}")


def recipe_context(recipe_obj):
    if recipe_obj is None:
        return nullcontext()
    return fp8_autocast_context(enabled=True, recipe=recipe_obj)


def run_once(layer, x, recipe_obj):
    layer.zero_grad(set_to_none=True)
    x.grad = None
    with recipe_context(recipe_obj):
        y = layer(x)
        loss = y.float().square().mean()
    loss.backward()
    return y


def run_grouped_moe_once(experts: MetisGroupedHeadExperts, x: torch.Tensor, m_splits: list[int]):
    experts.zero_grad(set_to_none=True)
    x.grad = None
    y = experts(x, m_splits, is_first_microbatch=True)
    loss = y.float().square().mean()
    loss.backward()
    return y


def run_liger_once(loss_fn, weight: torch.Tensor, hidden: torch.Tensor, targets: torch.Tensor):
    weight.grad = None
    hidden.grad = None
    loss = loss_fn(weight, hidden, targets)
    loss.backward()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test exact Metis-1.5 GEMM shapes on RTX PRO 6000 Blackwell.")
    parser.add_argument("--recipes", default="fp8_block,nvfp4,bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--allow-non-sm120", action="store_true")
    parser.add_argument("--skip-lm-head", action="store_true")
    parser.add_argument("--skip-grouped-moe", action="store_true")
    parser.add_argument("--skip-liger", action="store_true")
    parser.add_argument("--nvfp4-disable-rht", action="store_true")
    parser.add_argument("--nvfp4-disable-2d-quantization", action="store_true")
    parser.add_argument("--nvfp4-disable-stochastic-rounding", action="store_true")
    parser.add_argument("--grouped-moe-rows-per-expert", type=int, default=12288)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Blackwell kernel smoke test.")

    import transformer_engine.pytorch as te
    import transformer_engine.common.recipe as recipe_mod

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    print("torch", torch.__version__)
    print("cuda", torch.version.cuda)
    print("gpu", torch.cuda.get_device_name(device))
    print("capability", capability)
    print("has NVFP4", hasattr(recipe_mod, "NVFP4BlockScaling"))
    print("has MXFP8", hasattr(recipe_mod, "MXFP8BlockScaling"))
    if capability != (12, 0) and not args.allow_non_sm120:
        raise RuntimeError(f"Expected RTX PRO 6000 / sm120 capability (12, 0), got {capability}.")

    shapes = [shape for shape in SHAPES if not (args.skip_lm_head and shape[0] == "lm_head")]
    for recipe_name in [item.strip().lower() for item in args.recipes.split(",") if item.strip()]:
        recipe_obj = make_recipe(
            recipe_name,
            nvfp4_disable_rht=args.nvfp4_disable_rht,
            nvfp4_disable_2d_quantization=args.nvfp4_disable_2d_quantization,
            nvfp4_disable_stochastic_rounding=args.nvfp4_disable_stochastic_rounding,
        )
        print(f"\n=== recipe={recipe_name} ===", flush=True)
        for shape_name, m, k, n in shapes:
            print(f"Testing {shape_name}: [{m}, {k}] x [{k}, {n}]", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            layer = te.Linear(k, n, bias=False).to(device=device, dtype=torch.bfloat16)
            x = torch.randn(m, k, device=device, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(args.warmup):
                run_once(layer, x, recipe_obj)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                y = run_once(layer, x, recipe_obj)
            torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-9)
            avg_ms = (elapsed / max(args.iters, 1)) * 1000.0
            # Forward + dgrad + wgrad is roughly 6*m*k*n FLOPs.
            tflops = (6.0 * m * k * n) / ((avg_ms / 1000.0) * 1e12)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"ok shape={tuple(y.shape)} avg_ms={avg_ms:.2f} approx_tflops={tflops:.2f} peak_gib={peak_gib:.2f}")

        if not args.skip_grouped_moe:
            rows_per_expert = args.grouped_moe_rows_per_expert
            m_splits = [rows_per_expert] * 32
            total_rows = sum(m_splits)
            use_low_precision_grouped = recipe_name != "bf16"
            print(
                f"Testing grouped_moe_experts: 32 x [{rows_per_expert}, 384] -> [{rows_per_expert}, 384]",
                flush=True,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            experts = MetisGroupedHeadExperts(
                32,
                384,
                1280,
                bias=False,
                use_fp8=use_low_precision_grouped,
                init_std=0.02,
                precision_kwargs={
                    "low_precision_allowed": True,
                    "local_low_precision_recipe": recipe_obj if use_low_precision_grouped else None,
                },
            ).to(device=device, dtype=torch.bfloat16)
            x = torch.randn(total_rows, 384, device=device, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(args.warmup):
                run_grouped_moe_once(experts, x, m_splits)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                y = run_grouped_moe_once(experts, x, m_splits)
            torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-9)
            avg_ms = (elapsed / max(args.iters, 1)) * 1000.0
            gate_up_flops = 6.0 * total_rows * 384 * 2560
            down_flops = 6.0 * total_rows * 1280 * 384
            tflops = (gate_up_flops + down_flops) / ((avg_ms / 1000.0) * 1e12)
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            uses_te = experts.gate_up_proj.uses_transformer_engine and experts.down_proj.uses_transformer_engine
            print(
                f"ok shape={tuple(y.shape)} uses_te_grouped={int(uses_te)} "
                f"avg_ms={avg_ms:.2f} approx_tflops={tflops:.2f} peak_gib={peak_gib:.2f}",
                flush=True,
            )

    if not args.skip_liger:
        print("\n=== liger_fused_linear_ce ===", flush=True)
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        except ImportError as exc:
            print(f"skip liger_fused_linear_ce: {exc}", flush=True)
        else:
            vocab_size = 32768
            hidden_size = 1536
            rows = 24576
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
            weight = torch.randn(
                vocab_size,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            hidden = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
            targets = torch.randint(0, vocab_size, (rows,), device=device, dtype=torch.long)
            for _ in range(args.warmup):
                run_liger_once(loss_fn, weight, hidden, targets)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iters):
                loss = run_liger_once(loss_fn, weight, hidden, targets)
            torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - start, 1e-9)
            avg_ms = (elapsed / max(args.iters, 1)) * 1000.0
            peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"ok loss={float(loss.detach()):.6f} avg_ms={avg_ms:.2f} peak_gib={peak_gib:.2f}", flush=True)


if __name__ == "__main__":
    main()
