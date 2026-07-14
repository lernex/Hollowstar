#!/usr/bin/env python3
"""Run the Hugging Face `Lernex/Metis-1.5-think` release locally on a laptop.

The HF repo ships `model.safetensors` in the `metis_jax_safetensors_v1` format —
the JAX param pytree flattened with '.'-joined paths (embed, final_norm.scale,
layers.N.q, layers.N.expert_w1, ...). That is exactly the tree `jm.init_params`
builds, so loading is a pure *unflatten*; we then reuse the proven KV-cache chat
decode from `metis15_jax_cpu_generate.py` verbatim.

Runs entirely on the host CPU (JAX_PLATFORMS=cpu), fp32 dense — no TPU/GPU needed.

Usage:
  P=tmp/jaxcpu-venv/bin/python
  $P scripts/metis15_local_chat.py                      # interactive chat REPL
  $P scripts/metis15_local_chat.py --prompt "Explain photosynthesis in two sentences."
  $P scripts/metis15_local_chat.py --model-dir models/Metis-1.5-think
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import dataclasses
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from safetensors import safe_open  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from metis_mamba import jax_metis as jm  # noqa: E402

# Import the decode/chat machinery from the sibling CPU-generate script so the
# generation path here is byte-for-byte identical to the validated one.
_GEN_PATH = REPO / "scripts" / "metis15_jax_cpu_generate.py"
_spec = importlib.util.spec_from_file_location("metis15_jax_cpu_generate", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def unflatten_params(flat: dict) -> dict:
    """{'layers.3.q': arr, 'embed': arr, ...} -> nested params pytree."""
    params: dict = {}
    layers: dict[int, dict] = {}
    for key, arr in flat.items():
        parts = key.split(".")
        if parts[0] == "layers":
            idx = int(parts[1])
            node = layers.setdefault(idx, {})
            for p in parts[2:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = arr
        elif len(parts) == 1:
            params[parts[0]] = arr
        else:
            node = params
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = arr
    params["layers"] = tuple(layers[i] for i in sorted(layers))
    return params


def build_from_safetensors(manifest: Path, weights: Path, tok_path: Path):
    """Mirror gen.build() but restore weights from the HF safetensors export."""
    model_cfg, _ = jm.load_manifest_config(manifest, stage="pretrain")
    cfg = dataclasses.replace(
        model_cfg,
        dtype="float32",            # CPU has no fast bf16
        ce_logits_dtype="float32",
        attention_scores_dtype="float32",
        mor_enabled=False,
        remat_layers=False,
        expert_execution="reference",   # single-device dense MoE, no mesh
    )

    # Read every tensor as fp32 (weights are stored bf16; jax handles bf16->fp32).
    flat: dict[str, np.ndarray] = {}
    with safe_open(str(weights), framework="flax") as f:
        for k in f.keys():
            flat[k] = np.asarray(f.get_tensor(k).astype(jnp.float32))
    params = unflatten_params(flat)

    # Sanity-check the restored tree against a fresh init: same structure + shapes.
    ref = jm.init_params(jax.random.PRNGKey(0), cfg)
    ref_leaves = jax.tree_util.tree_structure(ref)
    got_leaves = jax.tree_util.tree_structure(params)
    if ref_leaves != got_leaves:
        raise SystemExit(
            "param tree mismatch vs init_params — export/config skew.\n"
            f"  expected: {ref_leaves}\n  got:      {got_leaves}"
        )
    for rp, gp in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(params)):
        if tuple(rp.shape) != tuple(gp.shape):
            raise SystemExit(f"shape mismatch: init={rp.shape} vs file={gp.shape}")

    params = jax.device_put(params, jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    tok = Tokenizer.from_file(str(tok_path))
    return cfg, params, tok


def main() -> None:
    ap = argparse.ArgumentParser(description="Local CPU chat for Lernex/Metis-1.5-think (HF safetensors)")
    ap.add_argument("--model-dir", default=str(REPO / "models" / "Metis-1.5-think"),
                    help="dir containing model.safetensors + tokenizer.json + config.json")
    ap.add_argument("--manifest", default=str(REPO / "configs/metis15_manifest.json"))
    ap.add_argument("--prompt", default=None, help="one-shot message; omit for interactive chat")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=1024, help="fixed jit window (<= model_max 1024)")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-turn", action="store_true",
                    help="each message is independent — no conversation history")
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    weights = mdir / "model.safetensors"
    tok_path = mdir / "tokenizer.json"
    if not weights.exists():
        raise SystemExit(f"missing {weights} — download it from huggingface.co/Lernex/Metis-1.5-think")

    cap = min(args.max_len, 1024)
    t0 = time.time()
    print(f"[load] {weights}  (CPU, fp32, dense MoE) ...", file=sys.stderr, flush=True)
    cfg, params, tok = build_from_safetensors(Path(args.manifest), weights, tok_path)
    get_decode = gen.make_decode_cache(cfg)
    print(f"[load] done in {time.time()-t0:.1f}s | {cfg.n_layer}L d{cfg.d_model} "
          f"{cfg.moe_num_experts}E top-{cfg.moe_top_k} | kv-cache decode",
          file=sys.stderr, flush=True)

    def reply(prompt: str, seed: int) -> str:
        return gen.generate_cached(
            get_decode, params, tok, prompt, cfg, max_new=args.max_new, cap=cap,
            temp=args.temp, top_k=args.top_k, top_p=args.top_p,
            rep_penalty=args.rep_penalty, seed=seed, stream=True,
            stop=["\nUser:", "\nUser :"],
        )

    if args.prompt is not None:
        print("Assistant: ", end="", flush=True)
        reply(f"User: {args.prompt}\nAssistant: ", args.seed + 1)
        print()
        return

    mode = "single-turn" if args.single_turn else "multi-turn"
    print(f"Metis-1.5-think — chat [{mode}]. Type a message; Ctrl-C / Ctrl-D to exit.\n",
          file=sys.stderr, flush=True)
    history: list[tuple[str, str]] = []
    turn = 0
    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not user.strip():
            continue
        turn += 1
        if args.single_turn:
            prompt = f"User: {user}\nAssistant: "
        else:
            history.append(("user", user))
            prompt = "".join(f"{'User' if r == 'user' else 'Assistant'}: {c}\n"
                             for r, c in history) + "Assistant: "
        print("Assistant: ", end="", flush=True)
        ts = time.time()
        resp = reply(prompt, args.seed + turn)
        if not args.single_turn:
            history.append(("assistant", resp.strip()))
        dt = time.time() - ts
        ntok = len(tok.encode(resp, add_special_tokens=False).ids)
        print(f"\n[{ntok} tok in {dt:.1f}s = {ntok/max(dt,1e-6):.1f} tok/s]\n",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
