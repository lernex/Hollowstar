#!/usr/bin/env python3
"""Interactive / one-shot text generation from a Metis-1.5 JAX (orbax) checkpoint,
running ENTIRELY on the host CPU.

Why CPU: the TPU is owned exclusively by the live training job and sits near the
HBM edge — a second JAX process can neither share the chips nor spare the memory.
The host CPU is separate silicon, so this never touches the TPU and is safe to run
alongside training. JAX_PLATFORMS=cpu is set before importing jax to guarantee it.

The PT base model is a *base LM* (no SFT): it continues text, it does not chat or
follow instructions. Feed it a stem ("The three laws of motion are") not a question.

Usage (on the VM):
  P=~/metis/.venv-jax-tpu/bin/python
  $P ~/metis/scripts/metis15_jax_cpu_generate.py --prompt "The capital of France is"
  $P ~/metis/scripts/metis15_jax_cpu_generate.py            # interactive REPL
  $P ~/metis/scripts/metis15_jax_cpu_generate.py --ckpt ~/metis/checkpoints/cpt-jax-v6e/latest
"""
from __future__ import annotations

import os

# Force CPU backend BEFORE importing jax — never grabs libtpu, never sees the TPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from metis_mamba import jax_metis as jm  # noqa: E402

BOS_ID, EOS_ID = 1, 2  # from tokenizer_meta.json: <pad>0 <bos>1 <eos>2 <unk>3


def build(manifest: Path, ckpt: Path, tok_path: Path):
    model_cfg, _ = jm.load_manifest_config(manifest, stage="pretrain")
    # bf16 compute on TPU (fast); fp32 on CPU (no fast bf16 there). Dense, no remat.
    on_tpu = any(d.platform == "tpu" for d in jax.devices())
    dt = "bfloat16" if on_tpu else "float32"
    cfg = dataclasses.replace(
        model_cfg,
        dtype=dt,
        ce_logits_dtype="float32",
        attention_scores_dtype=dt,
        mor_enabled=False,
        remat_layers=False,
    )
    key = jax.random.PRNGKey(0)
    target_params = jm.init_params(key, cfg)
    target_opt = jm.init_optimizer_state(target_params)
    params, meta = jm.restore_params_only(
        ckpt, target_params=target_params, target_opt_state=target_opt
    )
    # restore_params_only returns CPU-committed arrays; round-trip through host numpy
    # (uncommitted) then place on the active device — sidesteps a jax-0.6.2 re-device_put
    # recursion bug and puts the weights on the TPU for fast decode.
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    params = jax.device_put(params, jax.sharding.SingleDeviceSharding(jax.devices()[0]))
    tok = Tokenizer.from_file(str(tok_path))
    return cfg, params, tok, meta


# Coarse window buckets: size the jit'd window close to what a generation needs
# (less wasted compute over pad) while compiling only a handful of programs.
_BUCKETS = (64, 128, 192, 256, 384, 512, 768, 1024)


def make_forward_cache(cfg):
    """Lazily compile one fixed-shape forward per window bucket, cached.

    Fixed shape => compiles once per bucket (not once per step). capacity=lmax =>
    dropless MoE (pad tokens never starve real ones); causal masking means the
    right-pad never affects the real positions whose logits we read.
    """
    cache: dict[int, object] = {}

    def get(need: int, cap: int):
        lmax = next((b for b in _BUCKETS if b >= need), 1024)
        lmax = min(lmax, cap, 1024)
        if lmax not in cache:
            print(f"[compile] window={lmax} ...", file=sys.stderr, flush=True)

            @jax.jit
            def fwd(params, ids, _lmax=lmax):
                logits, _ = jm.forward(params, ids, cfg, capacity=_lmax)
                return logits

            cache[lmax] = fwd
        return cache[lmax], lmax

    return get


# ---------------------------------------------------------------------------
# KV-cache incremental decode.  One token in -> updated cache + next logits.
# Attention is the only reimplemented piece (vs jm._causal_attention); norms,
# RoPE, the MoE FFN and the output head are the exact training functions, so a
# decoded token is bit-for-bit the same as that token inside a full forward()
# (verified by the parity test). The expensive per-position MoE/projections now
# run once per *new* token (O(1)/step) instead of over the whole prefix every
# step (O(prefix)/step) — that is the O(n^2) -> O(n) win.
# ---------------------------------------------------------------------------
def _attn_step_cached(attn_in, layer, cfg, kc_i, vc_i, cos, sin, pos):
    nh, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q = jnp.einsum("btd,dh->bth", attn_in, layer["q"]).reshape(1, 1, nh, hd)
    k = jnp.einsum("btd,dh->bth", attn_in, layer["k"]).reshape(1, 1, nkv, hd)
    v = jnp.einsum("btd,dh->bth", attn_in, layer["v"]).reshape(1, 1, nkv, hd)
    q = jm._apply_rope(q, cos[None, :, None, :], sin[None, :, None, :])
    k = jm._apply_rope(k, cos[None, :, None, :], sin[None, :, None, :])
    # Append this token's K/V at absolute slot `pos`.
    kc_i = lax.dynamic_update_slice(kc_i, k[0].astype(kc_i.dtype), (pos, 0, 0))
    vc_i = lax.dynamic_update_slice(vc_i, v[0].astype(vc_i.dtype), (pos, 0, 0))
    group = nh // nkv
    k_rep = jnp.repeat(kc_i, group, axis=1)  # [max_len, nh, hd]
    v_rep = jnp.repeat(vc_i, group, axis=1)
    q0 = q[0, 0]  # [nh, hd]
    scores = jnp.einsum("hd,mhd->hm", q0, k_rep, preferred_element_type=jnp.float32)
    scores = scores / math.sqrt(float(hd))
    key_pos = jnp.arange(kc_i.shape[0], dtype=jnp.int32)
    scores = jnp.where(key_pos[None, :] <= pos, scores, jnp.float32(-1e9))  # causal over abs pos
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hm,mhd->hd", probs, v_rep).reshape(1, 1, nh * hd).astype(attn_in.dtype)
    attn_out = jnp.einsum("btx,xd->btd", out, layer["o"])
    return attn_out, kc_i, vc_i


def make_decode_cache(cfg):
    """get(max_len) -> (decode_fn, max_len). decode_fn jits once per max_len.

    decode_fn(params, token[1,1], pos[], kcache, vcache) ->
        (next_token_logits[vocab], kcache, vcache)
    with kcache/vcache shaped [n_layer, max_len, n_kv_heads, head_dim].
    """
    cache: dict[int, object] = {}

    def get(max_len: int):
        max_len = min(max_len, 1024)
        if max_len not in cache:
            print(f"[compile] kv-decode window={max_len} ...", file=sys.stderr, flush=True)

            @jax.jit
            def decode(params, token, pos, kcache, vcache):
                params = jm._cast_params_for_compute(params, cfg)
                x = params["embed"][token].astype(cfg.activation_dtype)  # [1,1,d]
                cos, sin = jm._rope_cos_sin(pos.reshape(1), cfg.head_dim, cfg.rope_theta)
                new_k, new_v = [], []
                for i, layer in enumerate(params["layers"]):
                    attn_in = jm.rms_norm(x, layer["attn_norm"]["scale"])
                    attn_out, kc_i, vc_i = _attn_step_cached(
                        attn_in, layer, cfg, kcache[i], vcache[i], cos, sin, pos
                    )
                    x = x + attn_out.astype(x.dtype)
                    moe_in = jm.rms_norm(x, layer["moe_norm"]["scale"])
                    moe_out, _ = jm.latent_moe(moe_in, layer, cfg, capacity=1, expert_mesh=None)
                    x = x + moe_out.astype(x.dtype)
                    new_k.append(kc_i)
                    new_v.append(vc_i)
                x = jm.rms_norm(x, params["final_norm"]["scale"])
                if cfg.tie_embeddings:
                    logits = jnp.einsum("btd,vd->btv", x, params["embed"])
                else:
                    logits = jnp.einsum("btd,dv->btv", x, params["lm_head"])
                return logits[0, 0].astype(jnp.float32), jnp.stack(new_k, 0), jnp.stack(new_v, 0)

            cache[max_len] = decode
        return cache[max_len], max_len

    return get


def _new_caches(cfg, max_len):
    shape = (cfg.n_layer, max_len, cfg.n_kv_heads, cfg.head_dim)
    return jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32)


def sample_next(logits: np.ndarray, *, temp: float, top_k: int, top_p: float,
                rng: np.random.Generator, recent: list[int], rep_penalty: float) -> int:
    logits = logits.astype(np.float32).copy()
    if rep_penalty and rep_penalty != 1.0 and recent:
        idx = np.array(sorted(set(recent)), dtype=np.int64)
        pos = logits[idx] > 0
        logits[idx[pos]] /= rep_penalty
        logits[idx[~pos]] *= rep_penalty
    if temp <= 0.0:
        return int(logits.argmax())
    logits /= temp
    if top_k and top_k < logits.shape[-1]:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p and top_p < 1.0:
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        cut = np.searchsorted(csum, top_p) + 1
        keep = order[:cut]
        mask = np.ones_like(probs, dtype=bool)
        mask[keep] = False
        probs[mask] = 0.0
        probs /= probs.sum()
    return int(rng.choice(probs.shape[-1], p=probs))


def generate(get_fwd, params, tok, prompt: str, *, max_new: int, cap: int, temp: float,
             top_k: int, top_p: float, rep_penalty: float, seed: int, stream: bool) -> str:
    rng = np.random.default_rng(seed)
    # add_special_tokens=False: the tokenizer template otherwise appends <eos>, and
    # reading logits *after* <eos> predicts a NEW document (prompt-independent!).
    # We add exactly one <bos> ourselves.
    ids = [BOS_ID] + tok.encode(prompt, add_special_tokens=False).ids
    fwd, lmax = get_fwd(len(ids) + max_new, cap)
    if len(ids) >= lmax:
        ids = ids[: lmax - 1]
    start = len(ids)
    buf = np.zeros((1, lmax), dtype=np.int32)
    printed = ""
    for _ in range(max_new):
        cur = len(ids)
        if cur >= lmax:
            break
        buf[0, :cur] = ids
        logits = np.asarray(fwd(params, jnp.asarray(buf)))[0, cur - 1]
        nxt = sample_next(logits, temp=temp, top_k=top_k, top_p=top_p, rng=rng,
                          recent=ids[start:], rep_penalty=rep_penalty)
        if nxt == EOS_ID:
            break
        ids.append(nxt)
        if stream:
            text = tok.decode(ids[start:])
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
    return tok.decode(ids[start:])


def generate_cached(get_decode, params, tok, prompt: str, cfg, *, max_new: int, cap: int,
                    temp: float, top_k: int, top_p: float, rep_penalty: float, seed: int,
                    stream: bool, stop: "list[str] | None" = None) -> str:
    rng = np.random.default_rng(seed)
    ids = [BOS_ID] + tok.encode(prompt, add_special_tokens=False).ids
    need = len(ids) + max_new
    max_len = next((b for b in _BUCKETS if b >= need), 1024)
    max_len = min(max_len, cap, 1024)
    if len(ids) >= max_len:
        ids = ids[: max_len - 1]
    decode, _ = get_decode(max_len)
    kcache, vcache = _new_caches(cfg, max_len)

    def step(tid, pos):
        return decode(params, jnp.asarray([[tid]], jnp.int32), jnp.asarray(pos, jnp.int32),
                      kcache, vcache)

    # Prefill: stream the prompt through the cache one token at a time.
    last_logits = None
    for pos, tid in enumerate(ids):
        last_logits, kcache, vcache = step(tid, pos)

    out_ids: list[int] = []
    printed = ""
    cur = len(ids)
    for _ in range(max_new):
        if cur >= max_len:
            break
        nxt = sample_next(np.asarray(last_logits), temp=temp, top_k=top_k, top_p=top_p,
                          rng=rng, recent=out_ids, rep_penalty=rep_penalty)
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        if stream:
            text = tok.decode(out_ids)
            sys.stdout.write(text[len(printed):])
            sys.stdout.flush()
            printed = text
        if stop and any(s in tok.decode(out_ids) for s in stop):
            break
        last_logits, kcache, vcache = step(nxt, cur)
        cur += 1
    out = tok.decode(out_ids)
    if stop:  # trim anything from the earliest stop sequence onward
        cuts = [out.find(s) for s in stop if s in out]
        if cuts:
            out = out[: min(cuts)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU generation from a Metis-1.5 JAX checkpoint")
    ap.add_argument("--manifest", default=str(REPO / "configs/metis15_manifest.json"))
    ap.add_argument("--ckpt", default=str(REPO / "checkpoints/base-jax-v6e/latest"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/tokenizer.json"))
    ap.add_argument("--prompt", default=None, help="one-shot prompt; omit for interactive REPL")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512, help="fixed jit window (<= model_max 1024)")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--rep-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-kv-cache", action="store_true",
                    help="use the slower full-forward path instead of the KV cache")
    ap.add_argument("--chat", action="store_true",
                    help="instruction-tuned chat mode (applies the User:/Assistant: template); "
                         "defaults --ckpt to the SFT/think checkpoint")
    ap.add_argument("--single-turn", action="store_true",
                    help="each message is independent — no conversation history, fresh cache each turn")
    args = ap.parse_args()

    if args.chat and args.ckpt == str(REPO / "checkpoints/base-jax-v6e/latest"):
        # identity-patched think checkpoint (falls back to pre-patch SFT if absent)
        patched = REPO / "checkpoints/sft-identity-jax-v6e/latest"
        args.ckpt = str(patched if patched.exists() else REPO / "checkpoints/sft-jax-v6e/latest")
    cap = min(args.max_len, 1024)
    t0 = time.time()
    print(f"[load] {Path(args.ckpt)}  (CPU, fp32, dense)  ...", file=sys.stderr, flush=True)
    cfg, params, tok, meta = build(Path(args.manifest), Path(args.ckpt), Path(args.tokenizer))
    step = meta.get("metrics", {}).get("step") or meta.get("step")
    end_loss = meta.get("metrics", {}).get("end_loss")
    mode = "full-forward" if args.no_kv_cache else "kv-cache"
    get_fwd = make_forward_cache(cfg)
    get_decode = make_decode_cache(cfg)
    print(f"[load] done in {time.time()-t0:.1f}s | ckpt step={step} end_loss={end_loss} "
          f"| {cfg.n_layer}L d{cfg.d_model} {cfg.moe_num_experts}E | decode={mode}",
          file=sys.stderr, flush=True)

    if args.no_kv_cache:
        gen = lambda p: generate(  # noqa: E731
            get_fwd, params, tok, p, max_new=args.max_new, cap=cap, temp=args.temp,
            top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep_penalty,
            seed=args.seed, stream=True,
        )
    else:
        gen = lambda p: generate_cached(  # noqa: E731
            get_decode, params, tok, p, cfg, max_new=args.max_new, cap=cap, temp=args.temp,
            top_k=args.top_k, top_p=args.top_p, rep_penalty=args.rep_penalty,
            seed=args.seed, stream=True,
        )

    if args.chat:
        mode = "single-turn, fresh each message" if args.single_turn else "multi-turn"
        print(f"Metis-1.5-think — chat [{mode}]. Type a message; Ctrl-C / Ctrl-D to exit.\n",
              file=sys.stderr, flush=True)
        history: "list[tuple[str, str]]" = []
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
                prompt = "".join(f"{'User' if r == 'user' else 'Assistant'}: {c}\n" for r, c in history) + "Assistant: "
            print("Assistant: ", end="", flush=True)
            ts = time.time()
            resp = generate_cached(get_decode, params, tok, prompt, cfg, max_new=args.max_new,
                                   cap=cap, temp=args.temp, top_k=args.top_k, top_p=args.top_p,
                                   rep_penalty=args.rep_penalty, seed=args.seed + turn,
                                   stream=True, stop=["\nUser:", "\nUser :"])
            if not args.single_turn:
                history.append(("assistant", resp.strip()))
            dt = time.time() - ts
            ntok = len(tok.encode(resp, add_special_tokens=False).ids)
            print(f"\n[{ntok} tok in {dt:.1f}s = {ntok/max(dt,1e-6):.1f} tok/s]\n",
                  file=sys.stderr, flush=True)
        return

    if args.prompt is not None:
        print(args.prompt, end="", flush=True)
        gen(args.prompt)
        print()
        return

    print("Interactive base-LM continuation. Type a stem; Ctrl-C / empty line + Ctrl-D to exit.\n"
          "(This is a BASE model — it continues text, it does not answer questions.)\n",
          file=sys.stderr, flush=True)
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if not prompt.strip():
            continue
        ts = time.time()
        print(prompt, end="", flush=True)
        out = gen(prompt)
        ntok = len(tok.encode(out).ids)
        dt = time.time() - ts
        print(f"\n[{ntok} tok in {dt:.1f}s = {ntok/max(dt,1e-6):.1f} tok/s]\n",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
