from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metis_runtime import (
    build_bounded_chat_prompt,
    choose_device,
    extract_assistant_reply,
    generate_completion,
    generate_completion_stream,
    list_best_checkpoints,
    looks_degenerate,
    load_model,
    load_tokenizer,
)


CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
DEFAULT_RELEASES_ROOT = ROOT_DIR / "releases"
DEFAULT_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "tokenizer" / "tokenizer.json"
METIS_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "metis_tokenizer" / "tokenizer.json"
METIS11_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "metis11_tokenizer" / "tokenizer.json"
METIS15_TOKENIZER_PATH = ROOT_DIR / "artifacts" / "metis15_hf_assets" / "tokenizer.json"
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
    "metis13_base": {
        "name": "Metis 1.3 Base",
        "tagline": "201M Mamba2 hybrid base",
        "description": "The Hugging Face Metis-1.3 base release loaded locally for raw pretrained-model probing.",
        "default_temperature": 0.45,
        "tokenizer_path": ROOT_DIR / "releases" / "metis13" / "base" / "tokenizer.json",
    },
    "metis15_base": {
        "name": "Metis 1.5 Base",
        "tagline": "1B sparse MoE base",
        "description": "The Metis-1.5 single LatentMoE base with 1K context and a BF16-first pretraining plus midtraining plan.",
        "default_temperature": 0.35,
        "tokenizer_path": METIS15_TOKENIZER_PATH,
    },
    "metis15_think": {
        "name": "Metis 1.5 Think",
        "tagline": "Unified chat and reasoning tune",
        "description": "The single post-trained Metis-1.5 release, with chat ability and verifier-backed reasoning consolidated into the think model.",
        "default_temperature": 0.2,
        "tokenizer_path": METIS15_TOKENIZER_PATH,
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
_remote_state = {
    "api_base": None,
    "checkpoints": None,
    "default": None,
}


def _normalize_api_base(api_base: str | None) -> str | None:
    if not api_base:
        return None
    return api_base.rstrip("/")


def _configured_releases_root() -> Path | None:
    raw = app.config.get("RELEASES_ROOT")
    if not raw:
        return None
    return Path(raw)


def _api_base() -> str | None:
    return _normalize_api_base(app.config.get("REMOTE_API_BASE"))


def _http_json(method: str, url: str, payload: dict | None = None) -> dict:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote API {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Remote API unreachable: {exc.reason}") from exc


def _http_stream(method: str, url: str, payload: dict | None = None):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if line.strip():
                    yield line
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        yield _json_line({"type": "error", "error": f"Remote API {exc.code}: {detail}"})
    except urllib.error.URLError as exc:
        yield _json_line({"type": "error", "error": f"Remote API unreachable: {exc.reason}"})


def _json_line(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _refresh_remote_checkpoints() -> tuple[list[dict[str, str]], str | None]:
    base = _api_base()
    if not base:
        return [], None
    payload = _http_json("GET", f"{base}/api/checkpoints")
    checkpoints = payload.get("checkpoints", [])
    default = payload.get("default")
    _remote_state.update({"api_base": base, "checkpoints": checkpoints, "default": default})
    return checkpoints, default


def release_choices() -> list[dict[str, str]]:
    releases_root = _configured_releases_root()
    if releases_root is None or not releases_root.exists():
        return []

    results = []
    release_map = {
        "metis13": {
            "base": "metis13_base",
        },
        "metis15": {
            "base": "metis15_base",
            "think": "metis15_think",
        },
    }
    family_roots = [releases_root] if (releases_root / "base").exists() else sorted(
        path for path in releases_root.iterdir() if path.is_dir()
    )
    for family_root in family_roots:
        family = family_root.name
        for dirname, preset_key in release_map.get(family, {}).items():
            release_dir = family_root / dirname
            if not release_dir.exists():
                continue
            preset = MODEL_PRESETS[preset_key]
            results.append(
                {
                    "label": preset["name"],
                    "value": str(release_dir.resolve()),
                    "slug": f"release_{family}_{dirname}",
                    "tagline": f'{preset["tagline"]} / HF export',
                    "description": preset["description"],
                    "default_temperature": preset["default_temperature"],
                    "tokenizer_path": str((release_dir / "tokenizer.json").resolve()),
                }
            )
    return results


def checkpoint_choices() -> list[dict[str, str]]:
    if _api_base():
        checkpoints, _ = _refresh_remote_checkpoints()
        return checkpoints

    results = []
    seen = set()
    for item in release_choices():
        results.append(item)
        seen.add(item["slug"])

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
    if _api_base():
        _, default = _refresh_remote_checkpoints()
        return default

    release_candidates = [Path(item["value"]) for item in release_choices()]
    preferred = [
        *release_candidates,
        CHECKPOINTS_DIR / "metis15_think" / "best.pt",
        CHECKPOINTS_DIR / "metis15_base" / "best.pt",
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
    )


@app.get("/api/checkpoints")
def api_checkpoints():
    return jsonify({"checkpoints": checkpoint_choices(), "default": pick_default_checkpoint()})


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True)
    if _api_base():
        return jsonify(_http_json("POST", f"{_api_base()}/api/chat", payload))

    checkpoint = payload.get("checkpoint") or pick_default_checkpoint()
    if not checkpoint:
        return jsonify({"error": "No checkpoints found. Train or fine-tune a model first."}), 400

    turns = payload.get("messages", [])
    if not turns:
        return jsonify({"error": "No conversation history provided."}), 400

    max_new_tokens = int(payload.get("max_new_tokens", 100))
    temperature = float(payload.get("temperature", 0.8))
    top_k = int(payload.get("top_k", 40))
    device_name = payload.get("device")

    model, tokenizer, device = get_runtime(checkpoint, device_name)
    prompt = build_bounded_chat_prompt(
        tokenizer=tokenizer,
        block_size=model.config.block_size,
        turns=turns,
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
            "raw_text": full_text,
            "checkpoint": checkpoint,
            "checkpoint_meta": checkpoint_metadata(checkpoint),
            "device": str(device),
        }
    )


@app.post("/api/chat/stream")
def api_chat_stream():
    payload = request.get_json(force=True)
    if _api_base():
        return Response(
            stream_with_context(_http_stream("POST", f"{_api_base()}/api/chat/stream", payload)),
            mimetype="application/x-ndjson",
        )

    checkpoint = payload.get("checkpoint") or pick_default_checkpoint()
    if not checkpoint:
        return jsonify({"error": "No checkpoints found. Train or fine-tune a model first."}), 400

    turns = payload.get("messages", [])
    if not turns:
        return jsonify({"error": "No conversation history provided."}), 400

    max_new_tokens = int(payload.get("max_new_tokens", 100))
    temperature = float(payload.get("temperature", 0.8))
    top_k = int(payload.get("top_k", 40))
    device_name = payload.get("device")

    model, tokenizer, device = get_runtime(checkpoint, device_name)
    prompt = build_bounded_chat_prompt(
        tokenizer=tokenizer,
        block_size=model.config.block_size,
        turns=turns,
        reply_budget=min(max_new_tokens, 64),
    )
    meta = checkpoint_metadata(checkpoint)

    def generate():
        yield _json_line(
            {
                "type": "meta",
                "checkpoint": checkpoint,
                "checkpoint_meta": meta,
                "device": str(device),
                "prompt": prompt,
            }
        )
        try:
            for event in generate_completion_stream(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                if event.get("type") == "done" and looks_degenerate(str(event.get("reply", ""))):
                    event = dict(event)
                    label = meta["label"] if meta else "This checkpoint"
                    event["reply"] = (
                        f"{label} is collapsing into whitespace or formatting tokens on this prompt. "
                        "Try a stronger Metis checkpoint for now, or retrain this run with a less aggressive truncation strategy."
                    )
                yield _json_line(event)
        except Exception as exc:
            yield _json_line({"type": "error", "error": str(exc)})

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local browser chat UI for your tiny model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--remote-api-base", default=None, help="Proxy all model calls to a remote chat_app API.")
    parser.add_argument(
        "--releases-root",
        default=str(DEFAULT_RELEASES_ROOT),
        help="Directory containing exported release folders like base/chat/think.",
    )
    args = parser.parse_args()
    app.config["REMOTE_API_BASE"] = _normalize_api_base(args.remote_api_base)
    app.config["RELEASES_ROOT"] = args.releases_root
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
