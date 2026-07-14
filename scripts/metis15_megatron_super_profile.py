#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any


MEGATRON_RECOMMENDED_FLAGS = {
    "expert_model_parallel_size": 8,
    "moe_token_dispatcher_type": "flex",
    "moe_flex_dispatcher_backend": "deepep",
    "overlap_moe_expert_parallel_comm": True,
    "delay_wgrad_compute": True,
    "moe_grouped_gemm": True,
    "moe_router_fusion": True,
    "moe_permute_fusion": True,
    "memory_efficient_permutation": True,
    "overlap_grad_reduce": True,
    "overlap_param_gather": True,
    "use_distributed_optimizer": True,
    "bf16": True,
    "fp16": False,
}


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def build_profile(manifest: dict[str, Any], *, check_imports: bool) -> dict[str, Any]:
    model = manifest.get("model") or {}
    hardware = manifest.get("hardware") or {}
    native = {
        "trainer": "native_metis_torch_distributed",
        "world_size": hardware.get("world_size", model.get("moe_expert_parallel_size", 1)),
        "precision": hardware.get("fallback_precision", model.get("torch_dtype", "bf16")),
        "moe_num_experts": model.get("moe_num_experts"),
        "moe_top_k": model.get("moe_top_k"),
        "moe_routed_latent_size": model.get("moe_routed_latent_size"),
        "moe_expert_parallel_size": model.get("moe_expert_parallel_size", 1),
        "moe_backend": model.get("moe_backend"),
        "moe_dispatch_mode": model.get("moe_dispatch_mode"),
        "moe_memory_efficient_permutation": model.get("moe_memory_efficient_permutation", False),
        "moe_token_dispatcher_type": model.get("moe_token_dispatcher_type", "alltoall"),
        "moe_flex_dispatcher_backend": model.get("moe_flex_dispatcher_backend", "none"),
    }
    bridge = {
        "trainer": "megatron_bridge_reference_port",
        "status": "requires_model_port",
        "why": (
            "DeepEP/HybridEP, EP overlap, delayed expert wgrad, router fusion, Megatron distributed "
            "optimizer, and Bridge checkpointing live in the Megatron-Core/Bridge trainer, not in "
            "the native Metis module graph."
        ),
        "recommended_flags": dict(MEGATRON_RECOMMENDED_FLAGS),
        "metis_shape_to_port": {
            "hidden_size": model.get("d_model"),
            "num_layers": model.get("n_layer"),
            "num_attention_heads": model.get("n_heads"),
            "num_query_groups": model.get("n_kv_heads"),
            "seq_length": model.get("block_size"),
            "num_moe_experts": model.get("moe_num_experts"),
            "moe_router_topk": model.get("moe_top_k"),
            "moe_ffn_hidden_size": model.get("moe_expert_intermediate_size"),
            "moe_latent_size": model.get("moe_routed_latent_size"),
            "activation": model.get("moe_activation"),
            "router_score": model.get("moe_router_score"),
            "balance_strategy": model.get("moe_balance_strategy"),
        },
    }
    profile: dict[str, Any] = {
        "model": manifest.get("name", model.get("name", "Metis-1.5")),
        "native_metis": native,
        "megatron_bridge": bridge,
    }
    if check_imports:
        profile["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "modules": {
                "torch": module_available("torch"),
                "transformer_engine": module_available("transformer_engine"),
                "megatron.core": module_available("megatron.core"),
                "megatron.bridge": module_available("megatron.bridge"),
                "deepep": module_available("deepep"),
                "hybridep": module_available("hybridep"),
            },
        }
        if profile["environment"]["modules"]["torch"]:
            import torch

            profile["environment"]["torch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "nccl_available": torch.distributed.is_nccl_available()
                if torch.distributed.is_available()
                else False,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit the native Metis vs Megatron-Bridge/Nemotron-Super MoE optimization profile.",
    )
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--check-imports", action="store_true", help="Probe installed Megatron/DeepEP stack.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    profile = build_profile(load_manifest(manifest_path), check_imports=args.check_imports)
    text = json.dumps(profile, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
