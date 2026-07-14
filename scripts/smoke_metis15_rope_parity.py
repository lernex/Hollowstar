#!/usr/bin/env python3
"""RoPE convention parity proofs across the Metis lanes.

Checks, against a pure-numpy NeoX half-split reference:
  1. The JAX lane (`jax_metis._apply_rope`) matches the reference exactly.
  2. The reference is a pure rotation (norm-preserving) and q.k depends only
     on the relative offset.
  3. If torch is importable: model.py's `apply_rotary_pos_emb(convention="neox")`
     matches the reference, while the `legacy_metis14` convention demonstrably
     differs (the historical mixed-pairing behavior is preserved, gated, and
     non-default for Metis-1.5).

The torch leg is skipped (with a notice) on hosts without torch; run it on the
GPU box before exporting JAX-trained Metis-1.5 weights through the torch lane.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

HEAD_DIM = 64
THETA = 10000.0


def reference_tables(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    freqs = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def reference_apply(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    cos, sin = reference_tables(positions)
    half = HEAD_DIM // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rotated * sin


def main() -> None:
    rng = np.random.default_rng(20260609)
    positions = np.asarray([0, 1, 7, 100, 1023])
    x = rng.standard_normal((positions.size, HEAD_DIM))

    # ---------------------------------------------------------------- invariants
    out = reference_apply(x, positions)
    norms_in = np.linalg.norm(x, axis=-1)
    norms_out = np.linalg.norm(out, axis=-1)
    if not np.allclose(norms_in, norms_out, atol=1e-9):
        raise AssertionError("NeoX reference must be norm-preserving.")
    q = rng.standard_normal(HEAD_DIM)
    k = rng.standard_normal(HEAD_DIM)

    def ref_dot(q_pos: int, k_pos: int) -> float:
        q_rot = reference_apply(q[None, :], np.asarray([q_pos]))[0]
        k_rot = reference_apply(k[None, :], np.asarray([k_pos]))[0]
        return float(q_rot @ k_rot)

    if abs(ref_dot(5, 2) - ref_dot(105, 102)) > 1e-9:
        raise AssertionError("NeoX reference q.k must depend only on relative offset.")
    print("rope_reference_invariants_ok", flush=True)

    # ---------------------------------------------------------------- JAX parity
    from metis_mamba.jax_metis import _apply_rope, _rope_cos_sin
    import jax.numpy as jnp

    cos_j, sin_j = _rope_cos_sin(jnp.asarray(positions), HEAD_DIM, THETA)
    jax_out = np.asarray(
        _apply_rope(jnp.asarray(x, dtype=jnp.float32)[:, None, :], cos_j[:, None, :], sin_j[:, None, :])
    )[:, 0, :]
    if not np.allclose(jax_out, out, atol=1e-4):
        raise AssertionError(f"JAX RoPE diverges from NeoX reference: max err {np.max(np.abs(jax_out - out))}")
    print("rope_jax_parity_ok", flush=True)

    # ---------------------------------------------------------------- torch parity (optional leg)
    try:
        import torch
    except ModuleNotFoundError:
        print("rope_torch_parity_skipped (torch not installed on this host)", flush=True)
        print("metis15_rope_parity_ok torch=skipped", flush=True)
        return

    from metis_mamba.model import MetisRotaryEmbedding, apply_rotary_pos_emb

    rotary = MetisRotaryEmbedding(HEAD_DIM, base=THETA)
    position_ids = torch.tensor(positions, dtype=torch.long)[None, :]
    cos_t, sin_t = rotary(position_ids, dtype=torch.float64)
    # model.py applies rope to [batch, heads, seq, head_dim]; build [1, 1, S, D].
    x_t = torch.tensor(x, dtype=torch.float64)[None, None, :, :]
    neox_out = apply_rotary_pos_emb(x_t, cos_t, sin_t, convention="neox")[0, 0].numpy()
    if not np.allclose(neox_out, out, atol=1e-9):
        raise AssertionError(
            f"torch neox RoPE diverges from reference: max err {np.max(np.abs(neox_out - out))}"
        )
    legacy_out = apply_rotary_pos_emb(x_t, cos_t, sin_t, convention="legacy_metis14")[0, 0].numpy()
    if np.allclose(legacy_out, out, atol=1e-6):
        raise AssertionError("legacy_metis14 must differ from neox (it preserves the historical behavior).")
    from metis_mamba.config import MetisMambaConfig

    if MetisMambaConfig().rope_convention != "legacy_metis14":
        raise AssertionError("Default rope_convention must stay legacy_metis14 for Metis-1.4 checkpoint compat.")
    print("rope_torch_parity_ok", flush=True)
    print("metis15_rope_parity_ok torch=checked", flush=True)


if __name__ == "__main__":
    main()
