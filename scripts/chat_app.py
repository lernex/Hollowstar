from __future__ import annotations

import argparse
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from tinylm import (
    build_bounded_chat_prompt,
    choose_device,
    extract_assistant_reply,
    generate_completion,
    list_best_checkpoints,
    looks_degenerate,
    load_model,
    load_tokenizer,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DEFAULT_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "tokenizer" / "tokenizer.json"
METIS_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "metis_tokenizer" / "tokenizer.json"
METIS11_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "metis11_tokenizer" / "tokenizer.json"
DEFAULT_SYSTEM_PROMPT = "You are a concise, friendly assistant. Answer clearly and directly."

MODEL_PRESETS = {
    "metis_base": {
        "name": "Metis 1.0",
        "tagline": "Reasoning-oriented base",
        "description": "A fresh 10M educational-web base model aimed at cleaner explanation and more deliberate answers.",
        "default_temperature": 0.55,
        "tokenizer_path": METIS_TOKENIZER_PATH,
    },
    "metis_think": {
        "name": "Metis 1.0 Think",
        "tagline": "Short-form reasoning tune",
        "description": "Metis with a compact reasoning tune for showing its work on short questions without drowning in giant traces.",
        "default_temperature": 0.25,
        "tokenizer_path": METIS_TOKENIZER_PATH,
    },
    "metis11_base": {
        "name": "Metis 1.1",
        "tagline": "50M base",
        "description": "A much roomier Metis base aimed at more stable question-answering and less brittle post-training.",
        "default_temperature": 0.45,
        "tokenizer_path": METIS11_TOKENIZER_PATH,
    },
    "metis11_chat": {
        "name": "Metis 1.1 Chat",
        "tagline": "Conversational tune",
        "description": "Metis 1.1 after a SmolTalk-based SFT stage so it behaves more naturally in back-and-forth conversation.",
        "default_temperature": 0.35,
        "tokenizer_path": METIS11_TOKENIZER_PATH,
    },
    "metis11_think": {
        "name": "Metis 1.1 Think",
        "tagline": "Compact reasoning tune",
        "description": "The full Metis 1.1 recipe with a conversational stage followed by concise OpenThoughts reasoning traces.",
        "default_temperature": 0.2,
        "tokenizer_path": METIS11_TOKENIZER_PATH,
    },
    "metis100_base": {
        "name": "Metis 100",
        "tagline": "100M experimental base",
        "description": "A large-for-local experimental Metis run that trades a much longer training time for noticeably more capacity.",
        "default_temperature": 0.4,
        "tokenizer_path": METIS_TOKENIZER_PATH,
    },
    "fast": {
        "name": "Little Lantern 1.0",
        "tagline": "Base pretrain",
        "description": "The original TinyStories-trained model. More narrative than helpful, but great for seeing the raw base model.",
        "default_temperature": 0.85,
        "tokenizer_path": DEFAULT_TOKENIZER_PATH,
    },
    "chat_fast": {
        "name": "Little Lantern 1.0 Instruct",
        "tagline": "Experimental instruct tune",
        "description": "A chat-tuned experiment on top of Little Lantern 1.0. Interesting to probe, but this run can still collapse into formatting or whitespace.",
        "default_temperature": 0.35,
        "tokenizer_path": DEFAULT_TOKENIZER_PATH,
    },
    "standard": {
        "name": "Little Lantern 1.1",
        "tagline": "Longer base run",
        "description": "A roomier base checkpoint for future comparisons once you train it.",
        "default_temperature": 0.8,
        "tokenizer_path": DEFAULT_TOKENIZER_PATH,
    },
    "chat_sft": {
        "name": "Little Lantern 1.1 Instruct",
        "tagline": "Chat-tuned",
        "description": "Instruction-tuned variant of the longer base run.",
        "default_temperature": 0.35,
        "tokenizer_path": DEFAULT_TOKENIZER_PATH,
    },
}

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "web" / "templates"),
    static_folder=str(ROOT_DIR / "web" / "static"),
)

_state_lock = threading.Lock()
_cached = {
    "checkpoint": None,
    "device": None,
    "model": None,
    "tokenizer": None,
}


def checkpoint_choices() -> list[dict[str, str]]:
    results = []
    seen = set()
    for path in list_best_checkpoints(CHECKPOINTS_DIR):
        slug = path.parent.name
        resolved = str(path)
        if slug in MODEL_PRESETS:
            preset = MODEL_PRESETS[slug]
            results.append(
                {
                    "label": preset["name"],
                    "value": resolved,
                    "slug": slug,
                    "tagline": preset["tagline"],
                    "description": preset["description"],
                    "default_temperature": preset["default_temperature"],
                    "tokenizer_path": str(preset["tokenizer_path"]),
                }
            )
            seen.add(slug)

    for path in list_best_checkpoints(CHECKPOINTS_DIR):
        slug = path.parent.name
        resolved = str(path)
        if slug in seen or slug in {"benchmark", "smoke", "chat_sft_smoke"}:
            continue
        try:
            label = str(path.relative_to(ROOT_DIR))
        except ValueError:
            label = str(path)
        results.append(
            {
                "label": f"Experimental / {slug}",
                "value": resolved,
                "slug": slug,
                "tagline": "Experimental",
                "description": label,
                "default_temperature": 0.8,
                "tokenizer_path": str(DEFAULT_TOKENIZER_PATH),
            }
        )
    return results


def pick_default_checkpoint() -> str | None:
    preferred = [
        CHECKPOINTS_DIR / "metis11_think" / "best.pt",
        CHECKPOINTS_DIR / "metis11_chat" / "best.pt",
        CHECKPOINTS_DIR / "metis_think" / "best.pt",
        CHECKPOINTS_DIR / "metis11_base" / "best.pt",
        CHECKPOINTS_DIR / "metis100_base" / "best.pt",
        CHECKPOINTS_DIR / "metis_base" / "best.pt",
        CHECKPOINTS_DIR / "fast" / "best.pt",
        CHECKPOINTS_DIR / "chat_fast" / "best.pt",
        CHECKPOINTS_DIR / "chat_sft" / "best.pt",
        CHECKPOINTS_DIR / "standard" / "best.pt",
    ]
    for path in preferred:
        if path.exists():
            return str(path.resolve())
    options = checkpoint_choices()
    return options[0]["value"] if options else None


def tokenizer_path_for_checkpoint(checkpoint_path: str) -> Path:
    meta = checkpoint_metadata(checkpoint_path)
    if meta and meta.get("tokenizer_path"):
        return Path(meta["tokenizer_path"])
    return DEFAULT_TOKENIZER_PATH


def get_runtime(checkpoint_path: str, device_name: str | None):
    checkpoint_path = str(Path(checkpoint_path).resolve())
    device = choose_device(device_name)
    tokenizer_path = str(tokenizer_path_for_checkpoint(checkpoint_path).resolve())
    with _state_lock:
        if (
            _cached["checkpoint"] == checkpoint_path
            and _cached["device"] == str(device)
            and _cached.get("tokenizer_path") == tokenizer_path
            and _cached["model"] is not None
            and _cached["tokenizer"] is not None
        ):
            return _cached["model"], _cached["tokenizer"], device

        tokenizer = load_tokenizer(Path(tokenizer_path))
        model = load_model(checkpoint_path, device)
        _cached.update(
            {
                "checkpoint": checkpoint_path,
                "device": str(device),
                "tokenizer_path": tokenizer_path,
                "model": model,
                "tokenizer": tokenizer,
            }
        )
        return model, tokenizer, device


def checkpoint_metadata(checkpoint_path: str | None) -> dict | None:
    if checkpoint_path is None:
        return None
    for item in checkpoint_choices():
        if item["value"] == checkpoint_path:
            return item
    return None


@app.get("/")
def index():
    return render_template(
        "index.html",
        checkpoints=checkpoint_choices(),
        default_checkpoint=pick_default_checkpoint(),
        default_checkpoint_meta=checkpoint_metadata(pick_default_checkpoint()),
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
    )


@app.get("/api/checkpoints")
def api_checkpoints():
    return jsonify({"checkpoints": checkpoint_choices(), "default": pick_default_checkpoint()})


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True)
    checkpoint = payload.get("checkpoint") or pick_default_checkpoint()
    if not checkpoint:
        return jsonify({"error": "No checkpoints found. Train or fine-tune a model first."}), 400

    turns = payload.get("messages", [])
    if not turns:
        return jsonify({"error": "No conversation history provided."}), 400

    system_prompt = payload.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
    max_new_tokens = int(payload.get("max_new_tokens", 100))
    temperature = float(payload.get("temperature", 0.8))
    top_k = int(payload.get("top_k", 40))
    device_name = payload.get("device")

    model, tokenizer, device = get_runtime(checkpoint, device_name)
    prompt = build_bounded_chat_prompt(
        tokenizer=tokenizer,
        block_size=model.config.block_size,
        turns=turns,
        system_prompt=system_prompt,
        reply_budget=min(max_new_tokens, 64),
    )
    full_text = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    reply = extract_assistant_reply(full_text)
    if looks_degenerate(reply):
        full_text = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=min(0.25, temperature),
            top_k=min(top_k, 20),
        )
        reply = extract_assistant_reply(full_text)
    if looks_degenerate(reply):
        meta = checkpoint_metadata(checkpoint)
        label = meta["label"] if meta else "This checkpoint"
        reply = (
            f"{label} is collapsing into whitespace or formatting tokens on this prompt. "
            "Try a stronger Metis checkpoint for now, or retrain this run with a less aggressive truncation strategy."
        )
    return jsonify(
        {
            "reply": reply,
            "checkpoint": checkpoint,
            "checkpoint_meta": checkpoint_metadata(checkpoint),
            "device": str(device),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser chat UI for your tiny model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
