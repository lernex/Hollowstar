from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer

from .model import GPTConfig, GPTLanguageModel


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_path))


def load_model(checkpoint_path: str | Path, device: torch.device) -> GPTLanguageModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**checkpoint["model_config"])
    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def encode_prompt(tokenizer: Tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
    return torch.tensor([prompt_ids], dtype=torch.long, device=device)


@torch.no_grad()
def generate_completion(
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    input_ids = encode_prompt(tokenizer, prompt, device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def list_best_checkpoints(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    checkpoints = sorted(root.glob("*/best.pt"))
    return [path.resolve() for path in checkpoints]


def build_chat_prompt(
    turns: Iterable[dict[str, str]],
    system_prompt: str | None = None,
) -> str:
    parts: list[str] = []
    if system_prompt and system_prompt.strip():
        parts.append(f"System: {system_prompt.strip()}")
    for turn in turns:
        role = turn["role"].strip().capitalize()
        content = turn["content"].strip()
        parts.append(f"{role}: {content}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def build_bounded_chat_prompt(
    tokenizer: Tokenizer,
    block_size: int,
    turns: Iterable[dict[str, str]],
    system_prompt: str | None = None,
    reply_budget: int = 48,
) -> str:
    turn_lines = []
    for turn in turns:
        role = turn["role"].strip().capitalize()
        content = turn["content"].strip()
        if content:
            turn_lines.append(f"{role}: {content}")

    system_line = f"System: {system_prompt.strip()}" if system_prompt and system_prompt.strip() else None
    budget = max(24, block_size - reply_budget)
    selected: list[str] = []

    for line in reversed(turn_lines):
        trial = [line] + selected
        parts = [system_line] if system_line else []
        parts.extend(trial)
        candidate = "\n".join(parts + ["Assistant: "])
        token_count = len(tokenizer.encode(candidate, add_special_tokens=False).ids) + 1
        if token_count <= budget or not selected:
            selected = trial
        else:
            break

    parts = [system_line] if system_line else []
    parts.extend(selected)
    return "\n".join(parts + ["Assistant: "])


def extract_assistant_reply(full_text: str) -> str:
    marker = "Assistant:"
    if marker in full_text:
        full_text = full_text.split(marker)[-1]
    for stop in ("\nUser:", "\nSystem:", "\nAssistant:"):
        if stop in full_text:
            full_text = full_text.split(stop)[0]
    full_text = re.sub(r"^[\s:;,.!?\-\"'`]+", "", full_text)
    return full_text.strip()


def looks_degenerate(reply: str) -> bool:
    sample = reply.strip()
    if not sample:
        return True
    head = sample[:80]
    if not any(char.isalpha() for char in head):
        return True
    if len(set(head)) <= 4 and len(head) >= 16:
        return True
    punctuation = sum(char in ":;,.!?-_ \n" for char in head)
    return punctuation / max(len(head), 1) > 0.72
