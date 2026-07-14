#!/usr/bin/env python3
"""Browser playground for `Lernex/Metis-1.5-think`, served by the JAX CPU runtime.

The existing `scripts/chat_app.py` UI is wired to the PyTorch `metis_runtime`
(and imports torch at module load), which can't host the JAX-native 1.5 weights.
This is a self-contained Flask server that reuses the SAME `web/` templates and
static assets but generates with the JAX model loaded from the HF safetensors
export — exactly the path validated in `scripts/metis15_local_chat.py`.

Runs entirely on CPU (JAX_PLATFORMS=cpu). No torch, no metis_runtime.

Usage:
  tmp/jaxcpu-venv/bin/python scripts/metis15_web_chat.py          # http://127.0.0.1:7861
  tmp/jaxcpu-venv/bin/python scripts/metis15_web_chat.py --port 8000
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Reuse build_from_safetensors() (loader) and the decode primitives.
_LC_PATH = REPO / "scripts" / "metis15_local_chat.py"
_spec = importlib.util.spec_from_file_location("metis15_local_chat", _LC_PATH)
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)
gen = lc.gen  # the metis15_jax_cpu_generate module (decode cache, sampler, BOS/EOS)

app = Flask(
    __name__,
    template_folder=str(REPO / "web" / "templates"),
    static_folder=str(REPO / "web" / "static"),
)

# Filled once at startup by load().
_RT: dict = {"cfg": None, "params": None, "tok": None, "get_decode": None}

# Single "checkpoint" entry the UI dropdown shows.
CKPT_VALUE = "metis15_think_jax"
CHECKPOINT = {
    "value": CKPT_VALUE,
    "label": "Metis 1.5 Think",
    "slug": "metis15_think",
    "tagline": "898M MoE (340M active) / JAX",
    "description": "The single post-trained Metis-1.5 release — chat + reasoning, run locally "
                   "on CPU from the Hugging Face safetensors export.",
    "default_temperature": 0.2,
}


def load(model_dir: Path, manifest: Path) -> None:
    weights = model_dir / "model.safetensors"
    tok_path = model_dir / "tokenizer.json"
    if not weights.exists():
        raise SystemExit(f"missing {weights} — download Lernex/Metis-1.5-think first")
    t0 = time.time()
    print(f"[load] {weights} (CPU, fp32, dense MoE) ...", file=sys.stderr, flush=True)
    cfg, params, tok = lc.build_from_safetensors(manifest, weights, tok_path)
    _RT.update(cfg=cfg, params=params, tok=tok, get_decode=gen.make_decode_cache(cfg))
    print(f"[load] done in {time.time()-t0:.1f}s | {cfg.n_layer}L d{cfg.d_model} "
          f"{cfg.moe_num_experts}E top-{cfg.moe_top_k}", file=sys.stderr, flush=True)


def build_prompt(turns: list[dict]) -> str:
    """messages [{role,content}] -> 'User:/Assistant:' template ending in 'Assistant: '."""
    lines = []
    for t in turns:
        role = "User" if t.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {t.get('content','')}")
    return "\n".join(lines) + "\nAssistant: "


def stream_tokens(prompt: str, *, max_new: int, temp: float, top_k: int,
                  top_p: float, rep_penalty: float, seed: int):
    """Mirror gen.generate_cached but yield decoded text deltas as they arrive."""
    cfg, params, tok = _RT["cfg"], _RT["params"], _RT["tok"]
    rng = np.random.default_rng(seed)
    ids = [gen.BOS_ID] + tok.encode(prompt, add_special_tokens=False).ids
    need = len(ids) + max_new
    max_len = next((b for b in gen._BUCKETS if b >= need), 1024)
    max_len = min(max_len, cfg.block_size, 1024)
    if len(ids) >= max_len:
        ids = ids[: max_len - 1]
    decode, _ = _RT["get_decode"](max_len)
    kcache, vcache = gen._new_caches(cfg, max_len)

    def step(tid, pos):
        return decode(params, jnp.asarray([[tid]], jnp.int32), jnp.asarray(pos, jnp.int32),
                      kcache, vcache)

    last_logits = None
    for pos, tid in enumerate(ids):
        last_logits, kcache, vcache = step(tid, pos)

    out_ids: list[int] = []
    printed = ""
    stop = ["\nUser:", "\nUser :"]
    cur = len(ids)
    for _ in range(max_new):
        if cur >= max_len:
            break
        nxt = gen.sample_next(np.asarray(last_logits), temp=temp, top_k=top_k, top_p=top_p,
                              rng=rng, recent=out_ids, rep_penalty=rep_penalty)
        if nxt == gen.EOS_ID:
            break
        out_ids.append(nxt)
        text = tok.decode(out_ids)
        if text != printed:
            yield text[len(printed):]
            printed = text
        if any(s in text for s in stop):
            break
        last_logits, kcache, vcache = step(nxt, cur)
        cur += 1


def _line(payload: dict) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


@app.get("/")
def index():
    return render_template("index.html", checkpoints=[CHECKPOINT],
                           default_checkpoint=CKPT_VALUE, default_checkpoint_meta=CHECKPOINT)


@app.get("/api/checkpoints")
def api_checkpoints():
    return jsonify({"checkpoints": [CHECKPOINT], "default": CKPT_VALUE})


def _params(payload: dict):
    return dict(
        max_new=int(payload.get("max_new_tokens", 256)),
        temp=float(payload.get("temperature", 0.2)),
        top_k=int(payload.get("top_k", 40)),
        top_p=0.95,
        rep_penalty=1.15,
    )


@app.post("/api/chat/stream")
def api_chat_stream():
    payload = request.get_json(force=True)
    turns = payload.get("messages", [])
    if not turns:
        return jsonify({"error": "No conversation history provided."}), 400
    prompt = build_prompt(turns)
    kw = _params(payload)

    def generate():
        yield _line({"type": "meta", "checkpoint": CKPT_VALUE,
                     "checkpoint_meta": CHECKPOINT, "device": "cpu", "prompt": prompt})
        yield _line({"type": "start", "raw_text": ""})
        full = ""
        try:
            for delta in stream_tokens(prompt, seed=int(time.time()) & 0xFFFF, **kw):
                full += delta
                yield _line({"type": "token", "delta": delta})
            reply = full
            for s in ("\nUser:", "\nUser :"):
                if s in reply:
                    reply = reply.split(s)[0]
            yield _line({"type": "done", "raw_text": full, "reply": reply.strip() or "(empty reply)"})
        except Exception as exc:  # noqa: BLE001
            yield _line({"type": "error", "error": str(exc)})

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(force=True)
    turns = payload.get("messages", [])
    if not turns:
        return jsonify({"error": "No conversation history provided."}), 400
    prompt = build_prompt(turns)
    kw = _params(payload)
    full = "".join(stream_tokens(prompt, seed=int(time.time()) & 0xFFFF, **kw))
    reply = full
    for s in ("\nUser:", "\nUser :"):
        if s in reply:
            reply = reply.split(s)[0]
    return jsonify({"reply": reply.strip() or "(empty reply)", "raw_text": full,
                    "checkpoint": CKPT_VALUE, "checkpoint_meta": CHECKPOINT, "device": "cpu"})


def main() -> None:
    ap = argparse.ArgumentParser(description="JAX-backed browser playground for Metis-1.5-think")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--model-dir", default=str(REPO / "models" / "Metis-1.5-think"))
    ap.add_argument("--manifest", default=str(REPO / "configs/metis15_manifest.json"))
    args = ap.parse_args()
    load(Path(args.model_dir), Path(args.manifest))
    print(f"\n  Metis-1.5-think playground → http://{args.host}:{args.port}\n",
          file=sys.stderr, flush=True)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
