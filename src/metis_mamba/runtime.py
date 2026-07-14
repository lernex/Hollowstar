from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model

from .config import MetisMambaConfig
from .checkpoint_compat import filter_state_dict_for_model
from .fp8 import build_fp8_recipe
from .hybrid_runtime import load_hybrid_exported_model
from .model import MetisMoRLMHeadModel, MetisMoRRewardModel


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype:
    mapping = {
        None: torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def build_model(
    config: MetisMambaConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    use_fp8: bool = False,
    fp8_recipe=None,
    fp8_group=None,
):
    config.validate()
    if use_fp8 and fp8_recipe is None:
        fp8_recipe = build_fp8_recipe()
    model = MetisMoRLMHeadModel(
        config,
        use_fp8=use_fp8,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
    )
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    model.config = config
    model.model_family = config.model_type
    return model


def build_reward_model(
    config: MetisMambaConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    use_fp8: bool = False,
    fp8_recipe=None,
    fp8_group=None,
):
    config.validate()
    if use_fp8 and fp8_recipe is None:
        fp8_recipe = build_fp8_recipe()
    model = MetisMoRRewardModel(
        config,
        use_fp8=use_fp8,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
    )
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    model.config = config
    return model


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MetisMambaConfig.from_dict(checkpoint["model_config"])
    model = build_model(config)
    filtered_state, _conversions = filter_state_dict_for_model(model, checkpoint["model_state_dict"])
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    allowed_missing = {"lm_head.impl.weight", "lm_head.weight"} if config.tie_embeddings else set()
    real_missing = sorted(name for name in missing if name not in allowed_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint load result: "
            f"missing={real_missing}, unexpected={sorted(unexpected)}, conversions={len(_conversions)}"
        )
    if config.tie_embeddings:
        model.tie_weights()
    model.to(device)
    model.eval()
    return model


def load_exported_model(
    model_dir: str | Path,
    device: torch.device,
):
    model_dir = Path(model_dir)
    raw_config = json.loads((model_dir / "config.json").read_text())
    if raw_config.get("model_type") == "metis_mamba2_hybrid":
        return load_hybrid_exported_model(model_dir, device)
    config = MetisMambaConfig.from_dict(raw_config)
    model = build_model(config)
    missing, unexpected = load_safetensors_model(model, str(model_dir / "model.safetensors"), device="cpu")
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected export load result for {model_dir}: missing={missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model


def export_checkpoint_to_dir(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_only_filename: str = "model.safetensors",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MetisMambaConfig.from_dict(checkpoint["model_config"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dtype = parse_torch_dtype(config.torch_dtype)
    model = build_model(config, device="cpu", dtype=export_dtype)
    state_dict = {
        name: tensor.detach().to(dtype=export_dtype).cpu()
        for name, tensor in checkpoint["model_state_dict"].items()
    }
    filtered_state, _conversions = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    allowed_missing = {"lm_head.impl.weight", "lm_head.weight"} if config.tie_embeddings else set()
    real_missing = sorted(name for name in missing if name not in allowed_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            "Unexpected checkpoint export load result: "
            f"missing={real_missing}, unexpected={sorted(unexpected)}, conversions={len(_conversions)}"
        )
    if config.tie_embeddings:
        model.tie_weights()
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    save_safetensors_model(model, str(output_dir / model_only_filename))
    return {
        "config": config.to_dict(),
        "model_path": str(output_dir / model_only_filename),
        "conversions": _conversions,
    }


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
    return torch.tensor([prompt_ids], dtype=torch.long, device=device)


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    model.eval()
    input_ids = encode_prompt(tokenizer, prompt, device)
    eos_token_id = tokenizer.token_to_id("<eos>")
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :].float()
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
        if input_ids.shape[1] > model.config.block_size:
            input_ids = input_ids[:, -model.config.block_size :]

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def cosine_lr(step: int, *, max_steps: int, warmup_steps: int, min_lr_factor: float = 0.0) -> float:
    if max_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(step + 1, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_factor + (1.0 - min_lr_factor) * cosine
