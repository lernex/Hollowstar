from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from .config import MetisMambaConfig


def _require_mamba():
    try:
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            "mamba-ssm is required for Metis-1.3 training/inference. "
            "Install the GPU training dependencies first."
        ) from exc
    return MambaConfig, MambaLMHeadModel


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
):
    config.validate()
    MambaConfig, MambaLMHeadModel = _require_mamba()
    mamba_config = MambaConfig(**config.to_mamba_config_dict())
    model = MambaLMHeadModel(
        mamba_config,
        device=device,
        dtype=dtype,
    )
    model.config = config
    model.model_family = config.model_type
    return model


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = MetisMambaConfig.from_dict(checkpoint["model_config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_exported_model(
    model_dir: str | Path,
    device: torch.device,
):
    model_dir = Path(model_dir)
    config = MetisMambaConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
    model = build_model(config)
    state_dict = load_safetensors(str(model_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def export_checkpoint_to_dir(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_only_filename: str = "model.safetensors",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = MetisMambaConfig.from_dict(checkpoint["model_config"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().to(dtype=parse_torch_dtype(config.torch_dtype)).cpu()
        for name, tensor in checkpoint["model_state_dict"].items()
    }
    save_safetensors(state_dict, str(output_dir / model_only_filename))
    return {
        "config": config.to_dict(),
        "model_path": str(output_dir / model_only_filename),
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
