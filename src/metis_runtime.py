from __future__ import annotations

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from metis_mamba import generate_completion as generate_mamba_completion
from metis_mamba import load_checkpoint_model as load_mamba_checkpoint_model
from metis_mamba import load_exported_model as load_mamba_exported_model
from tinylm.inference import (
    build_bounded_chat_prompt,
    build_chat_prompt,
    choose_device,
    extract_assistant_reply,
    generate_completion as generate_gpt_completion,
    list_best_checkpoints,
    load_model as load_legacy_model,
    looks_degenerate,
)


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def _is_mamba_export(path: Path) -> bool:
    if not path.is_dir():
        return False
    config_path = path / "config.json"
    model_path = path / "model.safetensors"
    if not config_path.exists() or not model_path.exists():
        return False
    payload = json.loads(config_path.read_text())
    return payload.get("model_type") == "metis_mamba2_hybrid"


def _is_mamba_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".pt":
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return checkpoint.get("model_family") == "metis_mamba2_hybrid"


def load_model(checkpoint_path: str | Path, device: torch.device):
    path = Path(checkpoint_path)
    if _is_mamba_export(path):
        return load_mamba_exported_model(path, device)
    if _is_mamba_checkpoint(path):
        return load_mamba_checkpoint_model(path, device)
    return load_legacy_model(path, device)


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
    if getattr(model, "model_family", None) == "metis_mamba2_hybrid":
        return generate_mamba_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    return generate_gpt_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


__all__ = [
    "build_bounded_chat_prompt",
    "build_chat_prompt",
    "choose_device",
    "extract_assistant_reply",
    "generate_completion",
    "list_best_checkpoints",
    "load_model",
    "load_tokenizer",
    "looks_degenerate",
]
