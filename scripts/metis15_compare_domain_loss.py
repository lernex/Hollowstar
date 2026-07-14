#!/usr/bin/env python3
"""Compare two Metis-1.5 JAX checkpoints by per-token loss on domain corpora.

CPU-only (safe alongside training). The test for "is CPT doing anything":
continued-pretraining on math/STEM should LOWER the loss on math/STEM text
relative to the PT checkpoint, and lower it MORE than it changes general-prose
loss. Loss here is mean next-token NLL (nats/token) — lower is better. Computed
from the logits directly (labels=None path) so it never touches the labeled-CE
branch.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import metis15_jax_cpu_generate as g  # noqa: E402
from metis_mamba import jax_metis as jm  # noqa: E402

MATH = [
    "To differentiate f(x) = x^3 + 2x^2 - 5x + 1 we apply the power rule term by "
    "term, which gives f'(x) = 3x^2 + 4x - 5. Setting the derivative equal to zero "
    "locates the critical points of the function.",
    "Theorem (Lagrange). Let G be a finite group and let H be a subgroup of G. Then "
    "the order of H divides the order of G. The proof partitions G into cosets of H, "
    "each having the same cardinality as H.",
    "Consider a particle of mass m under a constant force F. By Newton's second law "
    "the acceleration is a = F/m. Integrating with respect to time gives the velocity "
    "v(t) = v0 + a t, and integrating once more yields the position.",
    "Binary search runs in O(log n) time because each comparison halves the size of "
    "the remaining interval. This logarithmic behaviour makes it far more efficient "
    "than a linear scan over a large sorted array.",
    "The ideal gas law relates pressure, volume and temperature through PV = nRT, "
    "where n is the number of moles and R is the universal gas constant. Rearranging "
    "lets one solve for any single variable.",
    "Let A be an n by n matrix. The determinant of A is nonzero if and only if A is "
    "invertible, which is equivalent to the columns of A being linearly independent "
    "and the system Ax = 0 having only the trivial solution.",
]

GENERAL = [
    "Yesterday afternoon I walked down to the corner store to pick up some milk and a "
    "loaf of bread. The owner waved hello and we talked for a few minutes about how "
    "cold it had been all week.",
    "The concert last night was absolutely amazing. The band played for almost three "
    "hours, and when they came back for the encore the whole crowd was singing along "
    "at the top of their lungs.",
    "She pushed open the heavy wooden door and stepped into the quiet hallway. Dust "
    "floated in the thin beams of afternoon light, and somewhere upstairs a floorboard "
    "creaked under its own weight.",
    "Thanks so much for getting back to me. I really appreciate you taking the time, "
    "and the plan you suggested sounds great. Let me know what works for your schedule "
    "and we can set something up.",
    "The recipe calls for two cups of flour, a teaspoon of baking soda and a pinch of "
    "salt. Mix the dry ingredients first, then slowly fold in the butter and sugar "
    "until the batter is smooth.",
]


def _step(meta: dict) -> object:
    return meta.get("metrics", {}).get("step") or meta.get("step")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-token loss comparison of two checkpoints")
    ap.add_argument("--manifest", default=str(REPO / "configs/metis15_manifest.json"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/tokenizer.json"))
    ap.add_argument("--pt", default=str(REPO / "checkpoints/base-jax-v6e/latest"))
    ap.add_argument("--cpt", default=str(REPO / "checkpoints/cpt-jax-v6e/latest"))
    ap.add_argument("--pad-len", type=int, default=160)
    args = ap.parse_args()

    print("[load] PT + tokenizer ...", file=sys.stderr, flush=True)
    cfg, params_pt, tok, meta_pt = g.build(Path(args.manifest), Path(args.pt), Path(args.tokenizer))
    print("[load] CPT ...", file=sys.stderr, flush=True)
    key = jax.random.PRNGKey(0)
    tgt = jm.init_params(key, cfg)
    opt = jm.init_optimizer_state(tgt)
    params_cpt, meta_cpt = jm.restore_params_only(Path(args.cpt), target_params=tgt, target_opt_state=opt)

    L = args.pad_len
    get_fwd = g.make_forward_cache(cfg)
    fwd, _ = get_fwd(L, L)

    def nll(params, text: str) -> float:
        ids = ([1] + tok.encode(text, add_special_tokens=False).ids)[:L]
        buf = np.zeros((1, L), np.int32)
        buf[0, : len(ids)] = ids
        lg = np.asarray(fwd(params, jnp.asarray(buf))[0, : len(ids)]).astype(np.float64)
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        tok_nll = -lp[np.arange(len(ids) - 1), np.array(ids[1:])]
        return float(tok_nll.mean())

    print(f"\nPT step={_step(meta_pt)}   CPT step={_step(meta_cpt)}   (CPT total target 19073)\n")
    for name, corpus in [("MATH / STEM  (CPT target domain)", MATH), ("GENERAL PROSE  (control)", GENERAL)]:
        print(f"=== {name} — loss in nats/token, lower=better ===")
        a_all, b_all = [], []
        for t in corpus:
            a, b = nll(params_pt, t), nll(params_cpt, t)
            a_all.append(a)
            b_all.append(b)
            flag = "  <-- CPT better" if b < a else ("  <-- PT better" if a < b else "")
            print(f"  PT {a:5.3f}   CPT {b:5.3f}   delta {a-b:+.3f}{flag}   | {t[:46]}...")
        print(f"  MEAN  PT {st.mean(a_all):5.3f}   CPT {st.mean(b_all):5.3f}   "
              f"improvement {st.mean(a_all)-st.mean(b_all):+.3f}\n")


if __name__ == "__main__":
    main()
