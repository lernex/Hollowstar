from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

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
    model_type = payload.get("model_type", "")
    architecture = payload.get("architecture", "")
    return model_type.startswith("metis_") or architecture.startswith("metis_")


def _is_mamba_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".pt":
        return False
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    model_family = str(checkpoint.get("model_family", ""))
    return model_family.startswith("metis_")


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
    if str(getattr(model, "model_family", "")).startswith("metis_"):
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


@torch.no_grad()
def generate_completion_stream(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> Iterator[dict[str, object]]:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_token_id = tokenizer.token_to_id("<eos>")

    decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    yield {"type": "start", "raw_text": decoded}

    block_size = int(getattr(model.config, "block_size", input_ids.shape[1]))
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -block_size:]
        output = model(idx_cond)
        logits = output[0] if isinstance(output, tuple) else output.logits
        logits = logits[:, -1, :].float()
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
        next_decoded = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        delta = next_decoded[len(decoded) :]
        decoded = next_decoded
        yield {
            "type": "token",
            "token_id": int(next_token.item()),
            "delta": delta,
            "raw_text": decoded,
        }
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    yield {"type": "done", "raw_text": decoded, "reply": extract_assistant_reply(decoded)}


__all__ = [
    "build_bounded_chat_prompt",
    "build_chat_prompt",
    "choose_device",
    "extract_assistant_reply",
    "generate_completion",
    "generate_completion_stream",
    "list_best_checkpoints",
    "load_model",
    "load_tokenizer",
    "looks_degenerate",
]
