from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


try:  # Keep import errors readable in local environments before JAX is installed.
    import jax
    import jax.numpy as jnp
    from jax import lax
except Exception as exc:  # pragma: no cover - exercised by preflight when deps are absent.
    jax = None
    jnp = None
    lax = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None


Array = Any
Params = dict[str, Any]


@dataclass(frozen=True)
class JaxMetisConfig:
    vocab_size: int = 32768
    block_size: int = 1024
    d_model: int = 1536
    n_layer: int = 19
    n_heads: int = 24
    n_kv_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 4096
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    dtype: str = "bfloat16"
    weight_dtype: str = "float32"
    training_mode: str = "static_dense_pretrain"
    mor_enabled: bool = False
    mor_train_router: bool = False
    mor_runtime_mode: str = "disabled"
    mor_compute_mode: str = "soft_fixed_depth"
    mor_max_depth: int = 3
    mor_router_hidden_dim: int = 384
    mor_router_temperature: float = 1.0
    mor_packed_depth_capacity_factor: float = 0.75
    mor_packed_depth_capacity_alignment: int = 128
    mor_target_avg_depth_start: float = 1.0
    mor_target_avg_depth: float = 1.0
    mor_target_avg_depth_warmup_tokens: int = 0
    mor_router_aux_loss_coef_start: float = 0.0
    mor_router_aux_loss_coef: float = 0.0
    mor_router_entropy_coef: float = 0.0
    mor_router_z_loss_coef: float = 0.0
    moe_num_experts: int = 32
    moe_top_k: int = 4
    moe_shared_experts: int = 1
    moe_expert_intermediate_size: int = 1024
    moe_router_latent_size: int = 512
    moe_routed_latent_size: int = 512
    moe_activation: str = "squared_relu"
    moe_router_temperature: float = 1.0
    moe_aux_loss_coef: float = 1e-4
    moe_router_score: str = "sigmoid"
    moe_single_latent_router_input: str = "hidden"
    moe_balance_bias_update_rate: float = 1e-3
    moe_balance_bias_clamp: float = 5.0
    moe_expert_parallel_size: int = 8
    expert_capacity_factor: float = 4.0
    expert_capacity_alignment: int = 128
    qk_clip_threshold: float = 100.0
    qk_clip_alpha: float = 0.5
    remat_layers: bool = True
    expert_execution: str = "reference"

    @property
    def activation_dtype(self):
        _require_jax()
        if self.dtype == "bfloat16":
            return jnp.bfloat16
        if self.dtype == "float32":
            return jnp.float32
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    @property
    def param_dtype(self):
        _require_jax()
        if self.weight_dtype == "bfloat16":
            return jnp.bfloat16
        if self.weight_dtype == "float32":
            return jnp.float32
        raise ValueError(f"Unsupported weight_dtype: {self.weight_dtype}")

    @property
    def experts_per_rank(self) -> int:
        return self.moe_num_experts // self.moe_expert_parallel_size

    @property
    def assignments_per_batch(self) -> int:
        return self.block_size * self.moe_top_k

    def capacity_for_batch(self, local_batch_size: int) -> int:
        return self.capacity_for_tokens(local_batch_size * self.block_size)

    def capacity_for_tokens(self, n_tokens: int) -> int:
        raw = math.ceil(
            n_tokens * self.moe_top_k * self.expert_capacity_factor / self.moe_num_experts
        )
        align = max(1, int(self.expert_capacity_alignment))
        return int(math.ceil(raw / align) * align)

    def mor_depth_capacity_for_batch(self, local_batch_size: int) -> int:
        n_tokens = local_batch_size * self.block_size
        raw = math.ceil(n_tokens * self.mor_packed_depth_capacity_factor)
        align = max(1, int(self.mor_packed_depth_capacity_alignment))
        return min(n_tokens, int(math.ceil(raw / align) * align))

    def validate(self, *, local_batch_size: int = 1) -> None:
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.moe_num_experts % self.moe_expert_parallel_size != 0:
            raise ValueError("moe_num_experts must be divisible by moe_expert_parallel_size.")
        if self.moe_expert_parallel_size != 8:
            raise ValueError("Metis-1.5 JAX TPU lane is shaped for 8-way v6e expert parallelism.")
        if self.moe_top_k <= 0 or self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts].")
        if self.moe_activation != "squared_relu":
            raise ValueError("JAX LatentMoE lane currently implements squared_relu routed experts.")
        if self.moe_router_score not in {"sigmoid", "softmax"}:
            raise ValueError("moe_router_score must be sigmoid or softmax.")
        if self.training_mode == "static_dense_pretrain" and (self.mor_enabled or self.mor_runtime_mode != "disabled"):
            raise ValueError("Base pretraining must keep MoR disabled.")
        if self.training_mode == "dynamic_token_mor" and not self.mor_enabled:
            raise ValueError("Continued pretraining dynamic_token_mor requires mor_enabled.")
        if self.mor_compute_mode not in {"soft_fixed_depth", "static_packed_hard"}:
            raise ValueError("mor_compute_mode must be soft_fixed_depth or static_packed_hard.")
        if not self.mor_enabled and self.mor_compute_mode != "soft_fixed_depth":
            raise ValueError("Disabled MoR must use soft_fixed_depth as the inert compute mode.")
        if self.mor_max_depth <= 0:
            raise ValueError("mor_max_depth must be positive.")
        if self.mor_router_temperature <= 0:
            raise ValueError("mor_router_temperature must be positive.")
        if self.mor_packed_depth_capacity_factor <= 0:
            raise ValueError("mor_packed_depth_capacity_factor must be positive.")
        if self.mor_packed_depth_capacity_alignment <= 0:
            raise ValueError("mor_packed_depth_capacity_alignment must be positive.")
        if self.mor_target_avg_depth_start <= 0 or self.mor_target_avg_depth <= 0:
            raise ValueError("MoR target depths must be positive.")
        if self.mor_target_avg_depth_start > self.mor_max_depth or self.mor_target_avg_depth > self.mor_max_depth:
            raise ValueError("MoR target depths must not exceed mor_max_depth.")
        if self.mor_target_avg_depth_warmup_tokens < 0:
            raise ValueError("mor_target_avg_depth_warmup_tokens must be nonnegative.")
        if self.mor_router_aux_loss_coef_start < 0 or self.mor_router_aux_loss_coef < 0:
            raise ValueError("MoR router aux loss coefficients must be nonnegative.")
        if self.capacity_for_batch(local_batch_size) <= 0:
            raise ValueError("Expert capacity must be positive.")
        if self.mor_depth_capacity_for_batch(local_batch_size) <= 0:
            raise ValueError("Packed MoR depth capacity must be positive.")
        if self.expert_execution not in {"reference", "shard_map"}:
            raise ValueError("expert_execution must be reference or shard_map.")


@dataclass(frozen=True)
class JaxMetisTrainConfig:
    stage: str = "pretrain"
    local_batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adamw_eps: float = 1e-8
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    muon_lr_scale: float = 1.0
    muon_scale_mode: str = "match_rms_adamw"
    muon_nesterov: bool = True
    warmup_steps: int = 100
    max_steps: int = 1000
    checkpoint_interval: int = 100
    qk_clip_enabled: bool = True


@dataclass(frozen=True)
class JaxSamplerState:
    split: str
    cursor: int
    epoch: int
    tokens_emitted: int
    data_fingerprint: str


def _require_jax() -> None:
    if _JAX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "JAX is required for the Metis-1.5 JAX TPU lane. "
            "Install requirements-jax-tpu-train.txt first."
        ) from _JAX_IMPORT_ERROR


def load_manifest_config(path: str | Path, *, stage: str = "pretrain") -> tuple[JaxMetisConfig, JaxMetisTrainConfig]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = dict(payload["model"])
    hardware = dict(payload.get("hardware", {}))
    stability = dict(hardware.get("stability", {}))
    stage_payload = dict(payload[stage])
    mor = dict(stage_payload.get("mor", {}))

    cfg = JaxMetisConfig(
        vocab_size=int(model["vocab_size"]),
        block_size=int(model["block_size"]),
        d_model=int(model["d_model"]),
        n_layer=int(model["n_layer"]),
        n_heads=int(model["n_heads"]),
        n_kv_heads=int(model["n_kv_heads"]),
        head_dim=int(model["head_dim"]),
        intermediate_size=int(model["intermediate_size"]),
        tie_embeddings=bool(model["tie_embeddings"]),
        initializer_range=float(model["initializer_range"]),
        training_mode=str(stage_payload["training_mode"]),
        mor_enabled=bool(mor.get("enabled", model.get("mor_enabled", False))),
        mor_train_router=bool(mor.get("train_router", model.get("mor_train_router", False))),
        mor_runtime_mode=str(mor.get("runtime_mode", model.get("mor_runtime_mode", "disabled"))),
        mor_compute_mode=str(mor.get("compute_mode", model.get("mor_compute_mode", "soft_fixed_depth"))),
        mor_max_depth=int(mor.get("max_depth", model.get("mor_max_depth", 3))),
        mor_router_hidden_dim=int(mor.get("router_hidden_dim", model.get("mor_router_hidden_dim", 384))),
        mor_router_temperature=float(mor.get("router_temperature", model.get("mor_router_temperature", 1.0))),
        mor_packed_depth_capacity_factor=float(mor.get("packed_depth_capacity_factor", 0.75)),
        mor_packed_depth_capacity_alignment=int(mor.get("packed_depth_capacity_alignment", 128)),
        mor_target_avg_depth_start=float(
            mor.get(
                "target_avg_depth_start",
                mor.get("target_avg_depth", model.get("mor_target_avg_depth", 1.0)),
            )
        ),
        mor_target_avg_depth=float(
            mor.get(
                "target_avg_depth_end",
                mor.get("target_avg_depth", model.get("mor_target_avg_depth", 1.0)),
            )
        ),
        mor_target_avg_depth_warmup_tokens=int(mor.get("target_avg_depth_warmup_tokens", 0)),
        mor_router_aux_loss_coef_start=float(
            mor.get(
                "router_aux_loss_coef_start",
                mor.get("router_aux_loss_coef", model.get("mor_router_aux_loss_coef", 0.0)),
            )
        ),
        mor_router_aux_loss_coef=float(
            mor.get(
                "router_aux_loss_coef_end",
                mor.get("router_aux_loss_coef", model.get("mor_router_aux_loss_coef", 0.0)),
            )
        ),
        mor_router_entropy_coef=float(mor.get("router_entropy_coef", model.get("mor_router_entropy_coef", 0.0))),
        mor_router_z_loss_coef=float(mor.get("router_z_loss_coef", model.get("mor_router_z_loss_coef", 0.0))),
        moe_num_experts=int(model["moe_num_experts"]),
        moe_top_k=int(model["moe_top_k"]),
        moe_shared_experts=int(model["moe_shared_experts"]),
        moe_expert_intermediate_size=int(model["moe_expert_intermediate_size"]),
        moe_router_latent_size=int(model["moe_router_latent_size"]),
        moe_routed_latent_size=int(model["moe_routed_latent_size"]),
        moe_activation=str(model["moe_activation"]),
        moe_router_temperature=float(model["moe_router_temperature"]),
        moe_aux_loss_coef=float(model["moe_aux_loss_coef"]),
        moe_router_score=str(model["moe_router_score"]),
        moe_single_latent_router_input=str(model["moe_single_latent_router_input"]),
        moe_balance_bias_update_rate=float(model["moe_balance_bias_update_rate"]),
        moe_balance_bias_clamp=float(model["moe_balance_bias_clamp"]),
        moe_expert_parallel_size=int(model["moe_expert_parallel_size"]),
        expert_capacity_factor=float(stability.get("expert_capacity_factor", 4.0)),
        expert_capacity_alignment=int(model.get("moe_capacity_alignment", 128)),
        qk_clip_threshold=float(stability.get("qk_clip", {}).get("threshold", 100.0)),
        qk_clip_alpha=float(stability.get("qk_clip", {}).get("alpha", 0.5)),
        remat_layers=bool(hardware.get("remat_layers", model.get("remat_layers", True))),
    )
    train_cfg = JaxMetisTrainConfig(
        stage=stage,
        local_batch_size=int(stage_payload.get("local_batch_size", 1)),
        grad_accum_steps=int(stage_payload.get("grad_accum_steps", 1)),
        learning_rate=float(stage_payload["base_lr"]),
        weight_decay=float(stage_payload["weight_decay"]),
        beta1=float(stage_payload["optimizer_beta1"]),
        beta2=float(stage_payload["optimizer_beta2"]),
        max_steps=max(
            1,
            int(stage_payload["target_train_tokens"])
            // max(
                1,
                int(stage_payload.get("local_batch_size", 1))
                * int(stage_payload.get("grad_accum_steps", 1))
                * cfg.block_size,
            ),
        ),
        checkpoint_interval=int(stage_payload.get("checkpoint_interval", 100)),
    )
    cfg.validate(local_batch_size=train_cfg.local_batch_size)
    return cfg, train_cfg


def tiny_config(*, mor: bool = False) -> JaxMetisConfig:
    mode = "dynamic_token_mor" if mor else "static_dense_pretrain"
    return JaxMetisConfig(
        vocab_size=64,
        block_size=16,
        d_model=32,
        n_layer=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        training_mode=mode,
        mor_enabled=mor,
        mor_train_router=mor,
        mor_runtime_mode="dynamic_token" if mor else "disabled",
        mor_compute_mode="static_packed_hard" if mor else "soft_fixed_depth",
        mor_max_depth=3,
        mor_router_hidden_dim=16,
        mor_router_temperature=1.0,
        mor_packed_depth_capacity_factor=0.75,
        mor_packed_depth_capacity_alignment=8,
        mor_target_avg_depth_start=1.05 if mor else 1.0,
        mor_target_avg_depth=1.65 if mor else 1.0,
        mor_target_avg_depth_warmup_tokens=256 if mor else 0,
        mor_router_aux_loss_coef_start=0.01 if mor else 0.0,
        mor_router_aux_loss_coef=0.02 if mor else 0.0,
        mor_router_entropy_coef=0.001 if mor else 0.0,
        mor_router_z_loss_coef=0.0001 if mor else 0.0,
        moe_num_experts=8,
        moe_top_k=2,
        moe_shared_experts=1,
        moe_expert_intermediate_size=32,
        moe_router_latent_size=16,
        moe_routed_latent_size=16,
        moe_expert_parallel_size=8,
        expert_capacity_factor=2.0,
        expert_capacity_alignment=1,
        remat_layers=False,
    )


def _split(key, n: int):
    _require_jax()
    return jax.random.split(key, n)


def _normal(key, shape: tuple[int, ...], cfg: JaxMetisConfig, scale: float | None = None):
    scale = cfg.initializer_range if scale is None else scale
    return (jax.random.normal(key, shape, dtype=jnp.float32) * scale).astype(cfg.param_dtype)


def init_params(key: Array, cfg: JaxMetisConfig) -> Params:
    _require_jax()
    cfg.validate(local_batch_size=1)
    keys = list(_split(key, 8 + cfg.n_layer * 16))
    params: Params = {
        "embed": _normal(keys.pop(), (cfg.vocab_size, cfg.d_model), cfg),
        "final_norm": {"scale": jnp.ones((cfg.d_model,), dtype=cfg.param_dtype)},
    }
    if not cfg.tie_embeddings:
        params["lm_head"] = _normal(keys.pop(), (cfg.d_model, cfg.vocab_size), cfg)
    if cfg.mor_enabled:
        params["mor_router"] = {
            "w1": _normal(keys.pop(), (cfg.d_model, cfg.mor_router_hidden_dim), cfg),
            "b1": jnp.zeros((cfg.mor_router_hidden_dim,), dtype=cfg.param_dtype),
            "w2": _normal(keys.pop(), (cfg.mor_router_hidden_dim, cfg.mor_max_depth), cfg),
            "b2": jnp.zeros((cfg.mor_max_depth,), dtype=cfg.param_dtype),
        }
    layers = []
    for _ in range(cfg.n_layer):
        layer = {
            "attn_norm": {"scale": jnp.ones((cfg.d_model,), dtype=cfg.param_dtype)},
            "q": _normal(keys.pop(), (cfg.d_model, cfg.n_heads * cfg.head_dim), cfg),
            "k": _normal(keys.pop(), (cfg.d_model, cfg.n_kv_heads * cfg.head_dim), cfg),
            "v": _normal(keys.pop(), (cfg.d_model, cfg.n_kv_heads * cfg.head_dim), cfg),
            "o": _normal(keys.pop(), (cfg.n_heads * cfg.head_dim, cfg.d_model), cfg),
            "moe_norm": {"scale": jnp.ones((cfg.d_model,), dtype=cfg.param_dtype)},
            "latent_down": _normal(keys.pop(), (cfg.d_model, cfg.moe_routed_latent_size), cfg),
            "latent_up": _normal(keys.pop(), (cfg.moe_routed_latent_size, cfg.d_model), cfg),
            "router": _normal(keys.pop(), (cfg.d_model, cfg.moe_num_experts), cfg),
            "router_bias": jnp.zeros((cfg.moe_num_experts,), dtype=jnp.float32),
            "expert_w1": _normal(
                keys.pop(),
                (cfg.moe_num_experts, cfg.moe_routed_latent_size, cfg.moe_expert_intermediate_size),
                cfg,
            ),
            "expert_w2": _normal(
                keys.pop(),
                (cfg.moe_num_experts, cfg.moe_expert_intermediate_size, cfg.moe_routed_latent_size),
                cfg,
            ),
            "shared_w1": _normal(keys.pop(), (cfg.d_model, cfg.moe_expert_intermediate_size), cfg),
            "shared_w2": _normal(keys.pop(), (cfg.moe_expert_intermediate_size, cfg.d_model), cfg),
        }
        layers.append(layer)
    params["layers"] = tuple(layers)
    return params


def rms_norm(x: Array, scale: Array, eps: float = 1e-6) -> Array:
    return x * lax.rsqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + eps) * scale


def _causal_attention(x: Array, layer: Mapping[str, Any], cfg: JaxMetisConfig) -> Array:
    bsz, seq, _ = x.shape
    q = jnp.einsum("btd,dh->bth", x, layer["q"]).reshape(bsz, seq, cfg.n_heads, cfg.head_dim)
    k = jnp.einsum("btd,dh->bth", x, layer["k"]).reshape(bsz, seq, cfg.n_kv_heads, cfg.head_dim)
    v = jnp.einsum("btd,dh->bth", x, layer["v"]).reshape(bsz, seq, cfg.n_kv_heads, cfg.head_dim)
    group = cfg.n_heads // cfg.n_kv_heads
    k = jnp.repeat(k, group, axis=2)
    v = jnp.repeat(v, group, axis=2)
    q = jnp.swapaxes(q, 1, 2).astype(jnp.float32)
    k = jnp.swapaxes(k, 1, 2).astype(jnp.float32)
    v = jnp.swapaxes(v, 1, 2)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(float(cfg.head_dim))
    causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    logits = jnp.where(causal[None, None, :, :], logits, jnp.array(-1e9, dtype=jnp.float32))
    probs = jax.nn.softmax(logits, axis=-1).astype(cfg.activation_dtype)
    out = jnp.einsum("bhqk,bhkd->bqhd", probs, v).reshape(bsz, seq, cfg.n_heads * cfg.head_dim)
    return jnp.einsum("btd,dh->bth", out, layer["o"])


def _causal_attention_packed_queries(
    x: Array,
    layer: Mapping[str, Any],
    cfg: JaxMetisConfig,
    packed_tokens: Array,
    packed_valid: Array,
) -> Array:
    bsz, seq, _ = x.shape
    capacity = int(packed_tokens.shape[0])
    q_all = jnp.einsum("btd,dh->bth", x, layer["q"]).reshape(bsz, seq, cfg.n_heads, cfg.head_dim)
    k_all = jnp.einsum("btd,dh->bth", x, layer["k"]).reshape(bsz, seq, cfg.n_kv_heads, cfg.head_dim)
    v_all = jnp.einsum("btd,dh->bth", x, layer["v"]).reshape(bsz, seq, cfg.n_kv_heads, cfg.head_dim)
    group = cfg.n_heads // cfg.n_kv_heads
    k_all = jnp.repeat(k_all, group, axis=2)
    v_all = jnp.repeat(v_all, group, axis=2)
    batch_ids = (packed_tokens // seq).astype(jnp.int32)
    pos_ids = (packed_tokens % seq).astype(jnp.int32)
    q = q_all.reshape(bsz * seq, cfg.n_heads, cfg.head_dim)[packed_tokens].astype(jnp.float32)
    k = jnp.swapaxes(k_all, 1, 2)[batch_ids].astype(jnp.float32)
    v = jnp.swapaxes(v_all, 1, 2)[batch_ids]
    logits = jnp.einsum("chd,chkd->chk", q, k) / math.sqrt(float(cfg.head_dim))
    key_pos = jnp.arange(seq, dtype=jnp.int32)
    causal = key_pos[None, :] <= pos_ids[:, None]
    valid = packed_valid.astype(bool)
    logits = jnp.where(causal[:, None, :] & valid[:, None, None], logits, jnp.array(-1e9, dtype=jnp.float32))
    probs = jax.nn.softmax(logits, axis=-1).astype(cfg.activation_dtype)
    out = jnp.einsum("chk,chkd->chd", probs, v).reshape(capacity, cfg.n_heads * cfg.head_dim)
    out = jnp.einsum("cd,dh->ch", out, layer["o"])
    return jnp.where(valid[:, None], out, jnp.zeros_like(out))


def _topk_route_scores(logits: Array, cfg: JaxMetisConfig) -> tuple[Array, Array, Array]:
    if cfg.moe_router_score == "sigmoid":
        scores = jax.nn.sigmoid(logits / cfg.moe_router_temperature)
        top_scores, top_idx = lax.top_k(scores, cfg.moe_top_k)
        top_weights = top_scores / jnp.clip(jnp.sum(top_scores, axis=-1, keepdims=True), min=1e-6)
    else:
        scores = jax.nn.softmax(logits / cfg.moe_router_temperature, axis=-1)
        top_scores, top_idx = lax.top_k(scores, cfg.moe_top_k)
        top_weights = top_scores / jnp.clip(jnp.sum(top_scores, axis=-1, keepdims=True), min=1e-6)
    return top_idx.astype(jnp.int32), top_weights.astype(jnp.float32), scores.astype(jnp.float32)


def _pack_assignments(
    latent: Array,
    top_idx: Array,
    top_weights: Array,
    *,
    capacity: int,
    num_experts: int,
    token_mask: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    n_tokens, latent_dim = latent.shape
    arange_assign = jnp.arange(n_tokens * top_idx.shape[-1], dtype=jnp.int32)
    token_ids = jnp.repeat(jnp.arange(n_tokens, dtype=jnp.int32), top_idx.shape[-1])
    expert_ids = top_idx.reshape(-1).astype(jnp.int32)
    weights = top_weights.reshape(-1).astype(jnp.float32)
    if token_mask is None:
        assignment_valid = jnp.ones((n_tokens * top_idx.shape[-1],), dtype=bool)
    else:
        assignment_valid = jnp.repeat(token_mask.reshape(-1).astype(bool), top_idx.shape[-1])
    sort_experts = jnp.where(assignment_valid, expert_ids, jnp.asarray(num_experts, dtype=jnp.int32))
    keys = sort_experts * (n_tokens * top_idx.shape[-1]) + arange_assign
    order = jnp.argsort(keys)
    sorted_experts = sort_experts[order]
    sorted_tokens = token_ids[order]
    sorted_weights = weights[order]
    sorted_valid = assignment_valid[order]
    experts = jnp.arange(num_experts, dtype=jnp.int32)
    starts = jnp.searchsorted(sorted_experts, experts, side="left")
    ends = jnp.searchsorted(sorted_experts, experts, side="right")
    slots = starts[:, None] + jnp.arange(capacity, dtype=jnp.int32)[None, :]
    valid = slots < ends[:, None]
    safe_slots = jnp.minimum(slots, max(n_tokens * top_idx.shape[-1] - 1, 0))
    valid = valid & sorted_valid[safe_slots]
    packed_tokens = sorted_tokens[safe_slots]
    packed_weights = jnp.where(valid, sorted_weights[safe_slots], 0.0)
    expert_in = latent[packed_tokens]
    expert_in = jnp.where(valid[..., None], expert_in, jnp.zeros_like(expert_in))
    valid_assignments = jnp.sum(valid.astype(jnp.int32))
    return expert_in, packed_tokens, packed_weights, valid, valid_assignments


def _squared_relu_experts(expert_in: Array, w1: Array, w2: Array) -> Array:
    hidden = jnp.einsum("ecl,elh->ech", expert_in, w1)
    hidden = jnp.square(jnp.maximum(hidden, jnp.array(0.0, dtype=hidden.dtype)))
    return jnp.einsum("ech,ehl->ecl", hidden, w2)


def sharded_squared_relu_experts(expert_in: Array, w1: Array, w2: Array, mesh: Any) -> Array:
    """Run routed experts over an explicit expert-axis mesh.

    The global expert dimension is sharded across the mesh. On v6e-8 with 32 experts,
    each TPU chip owns exactly four routed experts. The function body sees only its
    local expert slice, which avoids the Python ragged-dispatch pattern that hurt the
    Trainium path.
    """

    _require_jax()
    from jax.sharding import PartitionSpec as P

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("expert", None, None), P("expert", None, None), P("expert", None, None)),
        out_specs=P("expert", None, None),
        check_vma=False,
    )
    def expert_kernel(local_in, local_w1, local_w2):
        return _squared_relu_experts(local_in, local_w1, local_w2)

    return expert_kernel(expert_in, w1, w2)


def latent_moe(
    x: Array,
    layer: Mapping[str, Any],
    cfg: JaxMetisConfig,
    *,
    capacity: int,
    expert_mesh: Any | None = None,
    token_mask: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    bsz, seq, _ = x.shape
    n_tokens = bsz * seq
    flat_hidden = x.reshape(n_tokens, cfg.d_model)
    latent = jnp.einsum("nd,dl->nl", flat_hidden, layer["latent_down"])
    router_input = flat_hidden if cfg.moe_single_latent_router_input == "hidden" else latent
    logits = jnp.einsum("nd,de->ne", router_input, layer["router"]) + layer["router_bias"]
    top_idx, top_weights, scores = _topk_route_scores(logits, cfg)
    expert_in, packed_tokens, packed_weights, valid, valid_assignments = _pack_assignments(
        latent,
        top_idx,
        top_weights,
        capacity=capacity,
        num_experts=cfg.moe_num_experts,
        token_mask=token_mask,
    )
    if cfg.expert_execution == "shard_map":
        if expert_mesh is None:
            raise ValueError("expert_mesh is required when expert_execution='shard_map'.")
        expert_out = sharded_squared_relu_experts(expert_in, layer["expert_w1"], layer["expert_w2"], expert_mesh)
    else:
        expert_out = _squared_relu_experts(expert_in, layer["expert_w1"], layer["expert_w2"])
    weighted = expert_out * packed_weights[..., None].astype(expert_out.dtype)
    weighted = jnp.where(valid[..., None], weighted, jnp.zeros_like(weighted))
    routed_latent = jnp.zeros((n_tokens, cfg.moe_routed_latent_size), dtype=expert_out.dtype)
    routed_latent = routed_latent.at[packed_tokens.reshape(-1)].add(weighted.reshape(-1, cfg.moe_routed_latent_size))
    routed = jnp.einsum("nl,ld->nd", routed_latent, layer["latent_up"]).reshape(bsz, seq, cfg.d_model)
    shared = jnp.einsum("btd,dh->bth", x, layer["shared_w1"])
    shared = jnp.square(jnp.maximum(shared, jnp.array(0.0, dtype=shared.dtype)))
    shared = jnp.einsum("bth,hd->btd", shared, layer["shared_w2"])
    if token_mask is None:
        token_weights = jnp.ones((n_tokens,), dtype=jnp.float32)
    else:
        token_weights = token_mask.reshape(-1).astype(jnp.float32)
    active_tokens = jnp.sum(token_weights)
    load = (
        jnp.sum(jax.nn.one_hot(top_idx.reshape(-1), cfg.moe_num_experts) * jnp.repeat(token_weights, cfg.moe_top_k)[:, None], axis=0)
        / jnp.clip(active_tokens * cfg.moe_top_k, min=1.0)
    )
    importance = jnp.sum(scores * token_weights[:, None], axis=0) / jnp.clip(active_tokens, min=1.0)
    aux = cfg.moe_aux_loss_coef * cfg.moe_num_experts * jnp.sum(load * importance)
    total_assignments = active_tokens * float(cfg.moe_top_k)
    metrics = {
        "moe_aux_loss": aux.astype(jnp.float32),
        "valid_assignments": valid_assignments.astype(jnp.float32),
        "total_assignments": total_assignments.astype(jnp.float32),
        "expert_drop_frac": 1.0 - valid_assignments.astype(jnp.float32) / jnp.clip(total_assignments, min=1.0),
        "router_entropy": -jnp.sum(token_weights * jnp.sum(scores * jnp.log(jnp.clip(scores, min=1e-6)), axis=-1))
        / jnp.clip(active_tokens, min=1.0),
    }
    return routed + shared, metrics


def _decoder_layer(
    x: Array,
    layer: Mapping[str, Any],
    cfg: JaxMetisConfig,
    *,
    capacity: int,
    expert_mesh: Any | None = None,
) -> tuple[Array, dict[str, Array]]:
    attn_in = rms_norm(x, layer["attn_norm"]["scale"])
    x = x + _causal_attention(attn_in, layer, cfg).astype(x.dtype)
    moe_in = rms_norm(x, layer["moe_norm"]["scale"])
    moe_out, metrics = latent_moe(moe_in, layer, cfg, capacity=capacity, expert_mesh=expert_mesh)
    x = x + moe_out.astype(x.dtype)
    return x, metrics


def _pack_active_tokens(x: Array, active_mask: Array, capacity: int) -> tuple[Array, Array, Array, Array, Array]:
    n_tokens = int(x.shape[0] * x.shape[1])
    flat = x.reshape(n_tokens, x.shape[-1])
    active = active_mask.reshape(-1).astype(bool)
    token_ids = jnp.arange(n_tokens, dtype=jnp.int32)
    active_count = jnp.sum(active.astype(jnp.int32))
    keys = jnp.where(active, token_ids, token_ids + n_tokens)
    order = jnp.argsort(keys)
    slots = jnp.arange(capacity, dtype=jnp.int32)
    safe_slots = jnp.minimum(slots, max(n_tokens - 1, 0))
    packed_tokens = order[safe_slots]
    valid = slots < active_count
    packed = flat[packed_tokens]
    packed = jnp.where(valid[:, None], packed, jnp.zeros_like(packed))
    valid_count = jnp.sum(valid.astype(jnp.int32))
    return packed, packed_tokens, valid, active_count, valid_count


def _decoder_layer_packed_queries(
    x: Array,
    layer: Mapping[str, Any],
    cfg: JaxMetisConfig,
    *,
    active_mask: Array,
    depth_capacity: int,
    expert_capacity: int,
    expert_mesh: Any | None = None,
    active_gate: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    packed_x, packed_tokens, packed_valid, active_count, valid_count = _pack_active_tokens(
        x, active_mask, depth_capacity
    )
    attn_in = rms_norm(x, layer["attn_norm"]["scale"])
    packed_after_attn = packed_x + _causal_attention_packed_queries(
        attn_in,
        layer,
        cfg,
        packed_tokens,
        packed_valid,
    ).astype(x.dtype)
    packed_moe_in = rms_norm(packed_after_attn, layer["moe_norm"]["scale"])
    moe_out, metrics = latent_moe(
        packed_moe_in[None, :, :],
        layer,
        cfg,
        capacity=expert_capacity,
        expert_mesh=expert_mesh,
        token_mask=packed_valid[None, :],
    )
    packed_after_moe = packed_after_attn + moe_out[0].astype(x.dtype)
    flat = x.reshape(-1, cfg.d_model)
    if active_gate is None:
        packed_gate = packed_valid.astype(x.dtype)
    else:
        packed_gate = active_gate.reshape(-1)[packed_tokens].astype(x.dtype)
        packed_gate = jnp.where(packed_valid, packed_gate, jnp.zeros_like(packed_gate))
    delta = jnp.where(
        packed_valid[:, None],
        (packed_after_moe - flat[packed_tokens]) * packed_gate[:, None],
        jnp.zeros_like(packed_after_moe),
    )
    out = flat.at[packed_tokens].add(delta).reshape(x.shape)
    metrics = {
        **metrics,
        "mor_packed_active_tokens": active_count.astype(jnp.float32),
        "mor_packed_valid_tokens": valid_count.astype(jnp.float32),
    }
    metrics["mor_packed_overflow_frac"] = jnp.where(
        metrics["mor_packed_active_tokens"] > 0,
        1.0 - metrics["mor_packed_valid_tokens"] / jnp.clip(metrics["mor_packed_active_tokens"], min=1.0),
        jnp.array(0.0, dtype=jnp.float32),
    )
    return out, metrics


def mor_schedule_for_tokens(cfg: JaxMetisConfig, tokens_seen: Array | int | float | None = None) -> dict[str, Array]:
    _require_jax()
    tokens = jnp.asarray(0.0 if tokens_seen is None else tokens_seen, dtype=jnp.float32)
    warmup = float(max(0, int(cfg.mor_target_avg_depth_warmup_tokens)))
    if warmup <= 0:
        ratio = jnp.array(1.0, dtype=jnp.float32)
    else:
        ratio = jnp.clip(tokens / jnp.asarray(warmup, dtype=jnp.float32), 0.0, 1.0)
    target_start = jnp.asarray(cfg.mor_target_avg_depth_start, dtype=jnp.float32)
    target_end = jnp.asarray(cfg.mor_target_avg_depth, dtype=jnp.float32)
    coef_start = jnp.asarray(cfg.mor_router_aux_loss_coef_start, dtype=jnp.float32)
    coef_end = jnp.asarray(cfg.mor_router_aux_loss_coef, dtype=jnp.float32)
    return {
        "mor_schedule_ratio": ratio.astype(jnp.float32),
        "mor_target_depth": (target_start + ratio * (target_end - target_start)).astype(jnp.float32),
        "mor_aux_coef": (coef_start + ratio * (coef_end - coef_start)).astype(jnp.float32),
    }


def _mor_depth_probs(
    x: Array,
    params: Params,
    cfg: JaxMetisConfig,
    *,
    tokens_seen: Array | int | float | None = None,
) -> tuple[Array, dict[str, Array]]:
    router = params["mor_router"]
    h = jax.nn.silu(jnp.einsum("btd,dh->bth", x, router["w1"]) + router["b1"])
    logits = jnp.einsum("bth,hk->btk", h, router["w2"]) + router["b2"]
    probs = jax.nn.softmax(logits / jnp.asarray(cfg.mor_router_temperature, dtype=logits.dtype), axis=-1)
    depths = jnp.arange(1, cfg.mor_max_depth + 1, dtype=jnp.float32)
    mean_depth = jnp.mean(jnp.sum(probs * depths, axis=-1))
    schedule = mor_schedule_for_tokens(cfg, tokens_seen)
    target = schedule["mor_target_depth"]
    aux = schedule["mor_aux_coef"] * jnp.square(mean_depth - target)
    entropy = -jnp.mean(jnp.sum(probs * jnp.log(jnp.clip(probs, min=1e-6)), axis=-1))
    z_loss = jnp.mean(jnp.square(jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)))
    aux = aux - cfg.mor_router_entropy_coef * entropy + cfg.mor_router_z_loss_coef * z_loss
    return probs.astype(jnp.float32), {
        "mor_aux_loss": aux.astype(jnp.float32),
        "mean_depth": mean_depth.astype(jnp.float32),
        "mor_entropy": entropy.astype(jnp.float32),
        **schedule,
    }


def forward(
    params: Params,
    input_ids: Array,
    cfg: JaxMetisConfig,
    *,
    labels: Array | None = None,
    capacity: int | None = None,
    expert_mesh: Any | None = None,
    tokens_seen: Array | int | float | None = None,
) -> tuple[Array, dict[str, Array]]:
    _require_jax()
    capacity = cfg.capacity_for_batch(int(input_ids.shape[0])) if capacity is None else int(capacity)
    x = params["embed"][input_ids].astype(cfg.activation_dtype)
    metrics_acc: dict[str, Array] = {
        "moe_aux_loss": jnp.array(0.0, dtype=jnp.float32),
        "valid_assignments": jnp.array(0.0, dtype=jnp.float32),
        "total_assignments": jnp.array(0.0, dtype=jnp.float32),
        "expert_drop_frac": jnp.array(0.0, dtype=jnp.float32),
        "router_entropy": jnp.array(0.0, dtype=jnp.float32),
        "mor_aux_loss": jnp.array(0.0, dtype=jnp.float32),
        "mean_depth": jnp.array(1.0, dtype=jnp.float32),
        "mor_entropy": jnp.array(0.0, dtype=jnp.float32),
        "mor_schedule_ratio": jnp.array(0.0, dtype=jnp.float32),
        "mor_target_depth": jnp.array(1.0, dtype=jnp.float32),
        "mor_aux_coef": jnp.array(0.0, dtype=jnp.float32),
        "mor_expected_depth": jnp.array(1.0, dtype=jnp.float32),
        "mor_packed_active_tokens": jnp.array(0.0, dtype=jnp.float32),
        "mor_packed_valid_tokens": jnp.array(0.0, dtype=jnp.float32),
        "mor_packed_overflow_frac": jnp.array(0.0, dtype=jnp.float32),
    }
    mor_probs = None
    mor_hard_depth = None
    if cfg.mor_enabled:
        mor_probs, mor_metrics = _mor_depth_probs(x, params, cfg, tokens_seen=tokens_seen)
        metrics_acc.update(mor_metrics)
        metrics_acc["mor_expected_depth"] = mor_metrics["mean_depth"]
        mor_hard_depth = jnp.argmax(mor_probs, axis=-1).astype(jnp.int32) + 1
        if cfg.mor_compute_mode == "static_packed_hard":
            metrics_acc["mean_depth"] = jnp.mean(mor_hard_depth.astype(jnp.float32))

    def apply_layer(hidden: Array, layer_params: Mapping[str, Any]) -> tuple[Array, dict[str, Array]]:
        return _decoder_layer(hidden, layer_params, cfg, capacity=capacity, expert_mesh=expert_mesh)

    depth_capacity = cfg.mor_depth_capacity_for_batch(int(input_ids.shape[0]))
    depth_expert_capacity = cfg.capacity_for_tokens(depth_capacity)

    def apply_packed_layer(
        hidden: Array,
        layer_params: Mapping[str, Any],
        active_mask: Array,
        active_gate: Array,
    ) -> tuple[Array, dict[str, Array]]:
        return _decoder_layer_packed_queries(
            hidden,
            layer_params,
            cfg,
            active_mask=active_mask,
            active_gate=active_gate,
            depth_capacity=depth_capacity,
            expert_capacity=depth_expert_capacity,
            expert_mesh=expert_mesh,
        )

    layer_call = jax.checkpoint(apply_layer) if cfg.remat_layers else apply_layer
    packed_layer_call = jax.checkpoint(apply_packed_layer) if cfg.remat_layers else apply_packed_layer
    for layer in params["layers"]:
        if cfg.mor_enabled and cfg.mor_runtime_mode == "dynamic_token" and cfg.mor_compute_mode == "soft_fixed_depth":
            layer_input = x
            current = x
            depth_metrics: dict[str, Array] | None = None
            for depth_idx in range(cfg.mor_max_depth):
                current, layer_metrics = layer_call(current, layer)
                if depth_metrics is None:
                    depth_metrics = layer_metrics
                else:
                    depth_metrics = {
                        key: depth_metrics[key] + layer_metrics[key]
                        for key in ("moe_aux_loss", "valid_assignments", "total_assignments", "router_entropy")
                    }
                    depth_metrics["expert_drop_frac"] = (
                        1.0
                        - depth_metrics["valid_assignments"]
                        / jnp.clip(depth_metrics["total_assignments"], min=1.0)
                    )
                gate = mor_probs[..., depth_idx : depth_idx + 1].astype(current.dtype)
                x = x + gate * (current - layer_input)
            metrics = depth_metrics if depth_metrics is not None else layer_metrics
        elif cfg.mor_enabled and cfg.mor_runtime_mode == "dynamic_token" and cfg.mor_compute_mode == "static_packed_hard":
            if mor_hard_depth is None:
                raise ValueError("Hard packed MoR requires depth router outputs.")
            x, metrics = layer_call(x, layer)
            for depth_idx in range(1, cfg.mor_max_depth):
                active_mask = mor_hard_depth > depth_idx
                active_prob = jnp.sum(mor_probs[..., depth_idx:], axis=-1)
                active_gate = active_mask.astype(active_prob.dtype) + active_prob - lax.stop_gradient(active_prob)
                x, packed_metrics = packed_layer_call(x, layer, active_mask, active_gate)
                metrics = {
                    key: metrics[key] + packed_metrics[key]
                    for key in ("moe_aux_loss", "valid_assignments", "total_assignments", "router_entropy")
                }
                metrics["expert_drop_frac"] = (
                    1.0
                    - metrics["valid_assignments"]
                    / jnp.clip(metrics["total_assignments"], min=1.0)
                )
                for key in ("mor_packed_active_tokens", "mor_packed_valid_tokens"):
                    metrics_acc[key] = metrics_acc[key] + packed_metrics[key]
        else:
            x, metrics = layer_call(x, layer)
        for key in ("moe_aux_loss", "valid_assignments", "total_assignments", "router_entropy"):
            metrics_acc[key] = metrics_acc[key] + metrics[key]
    metrics_acc["mor_packed_overflow_frac"] = jnp.where(
        metrics_acc["mor_packed_active_tokens"] > 0,
        1.0
        - metrics_acc["mor_packed_valid_tokens"]
        / jnp.clip(metrics_acc["mor_packed_active_tokens"], min=1.0),
        jnp.array(0.0, dtype=jnp.float32),
    )
    x = rms_norm(x, params["final_norm"]["scale"])
    if cfg.tie_embeddings:
        logits = jnp.einsum("btd,vd->btv", x.astype(jnp.float32), params["embed"].astype(jnp.float32))
    else:
        logits = jnp.einsum("btd,dv->btv", x.astype(jnp.float32), params["lm_head"].astype(jnp.float32))
    if labels is None:
        return logits, metrics_acc
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, shift_labels[..., None], axis=-1).squeeze(-1)
    lm_loss = jnp.mean(nll)
    total_loss = lm_loss + metrics_acc["moe_aux_loss"] + metrics_acc["mor_aux_loss"]
    metrics_acc["lm_loss"] = lm_loss.astype(jnp.float32)
    metrics_acc["loss"] = total_loss.astype(jnp.float32)
    metrics_acc["expert_drop_frac"] = 1.0 - metrics_acc["valid_assignments"] / jnp.clip(metrics_acc["total_assignments"], min=1.0)
    return total_loss.astype(jnp.float32), metrics_acc


def _flatten_paths(tree: Any, prefix: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _flatten_paths(value, prefix + (str(key),))
    elif isinstance(tree, (tuple, list)):
        for index, value in enumerate(tree):
            yield from _flatten_paths(value, prefix + (str(index),))
    else:
        yield prefix, tree


def muon_mask(params: Params) -> Params:
    def classify(path: tuple[str, ...], value: Any) -> bool:
        name = "/".join(path)
        if not hasattr(value, "ndim") or value.ndim != 2:
            return False
        if any(part in name for part in ("embed", "lm_head", "router", "norm", "expert_w")):
            return False
        return any(part in name for part in ("q", "k", "v", "o", "latent_down", "latent_up", "shared_w"))

    flat = {path: classify(path, value) for path, value in _flatten_paths(params)}
    return _unflatten_like(params, flat)


def _unflatten_like(tree: Any, values: dict[tuple[str, ...], Any], prefix: tuple[str, ...] = ()) -> Any:
    if isinstance(tree, Mapping):
        return {key: _unflatten_like(value, values, prefix + (str(key),)) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_unflatten_like(value, values, prefix + (str(index),)) for index, value in enumerate(tree))
    if isinstance(tree, list):
        return [_unflatten_like(value, values, prefix + (str(index),)) for index, value in enumerate(tree)]
    return values[prefix]


@dataclass
class OptimState:
    adam_m: Params
    adam_v: Params
    muon_momentum: Params
    step: Array

    def tree_flatten(self):
        return (self.adam_m, self.adam_v, self.muon_momentum, self.step), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        adam_m, adam_v, muon_momentum, step = children
        return cls(adam_m=adam_m, adam_v=adam_v, muon_momentum=muon_momentum, step=step)


if jax is not None:
    jax.tree_util.register_pytree_node_class(OptimState)


@dataclass(frozen=True)
class _OptimizerLeafUpdate:
    param: Array
    adam_m: Array
    adam_v: Array
    muon_momentum: Array


def init_optimizer_state(params: Params) -> OptimState:
    _require_jax()
    zeros = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), params)
    return OptimState(adam_m=zeros, adam_v=zeros, muon_momentum=zeros, step=jnp.array(0, dtype=jnp.int32))


def _zeropower_via_newton_schulz5(update: Array, *, steps: int) -> Array:
    x = update.astype(jnp.float32)
    norm = jnp.linalg.norm(x)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / jnp.clip(norm, min=1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = (a * x) + ((b * xx_t + c * (xx_t @ xx_t)) @ x)
    if transpose:
        x = x.T
    return x


def _muon_update_scale(shape: tuple[int, ...], mode: str) -> float:
    rows, cols = int(shape[-2]), int(shape[-1])
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(float(max(rows, cols, 1)))
    if mode == "original":
        return math.sqrt(max(1.0, float(rows) / float(max(cols, 1))))
    raise ValueError("muon_scale_mode must be original or match_rms_adamw.")


def apply_optimizer(
    params: Params,
    grads: Params,
    state: OptimState,
    mask: Params,
    cfg: JaxMetisTrainConfig,
) -> tuple[Params, OptimState]:
    _require_jax()
    step = state.step + 1
    beta1_t = cfg.beta1
    beta2_t = cfg.beta2

    def update_one(param, grad, adam_m, adam_v, muon_mom, is_muon):
        grad_f = grad.astype(jnp.float32)
        if is_muon:
            mom = cfg.muon_beta * muon_mom + grad_f
            update = grad_f + cfg.muon_beta * mom if cfg.muon_nesterov else mom
            update = _zeropower_via_newton_schulz5(update, steps=cfg.muon_ns_steps)
            update = update * (_muon_update_scale(param.shape, cfg.muon_scale_mode) * cfg.muon_lr_scale)
            decayed = param.astype(jnp.float32) * (1.0 - cfg.learning_rate * cfg.weight_decay)
            new_param = decayed - cfg.learning_rate * update
            return _OptimizerLeafUpdate(new_param.astype(param.dtype), adam_m, adam_v, mom)
        new_m = beta1_t * adam_m + (1.0 - beta1_t) * grad_f
        new_v = beta2_t * adam_v + (1.0 - beta2_t) * jnp.square(grad_f)
        bias1 = 1.0 - beta1_t**step.astype(jnp.float32)
        bias2 = 1.0 - beta2_t**step.astype(jnp.float32)
        update = (new_m / jnp.clip(bias1, min=1e-16)) / (jnp.sqrt(new_v / jnp.clip(bias2, min=1e-16)) + cfg.adamw_eps)
        decayed = param.astype(jnp.float32) * (1.0 - cfg.learning_rate * cfg.weight_decay)
        new_param = decayed - cfg.learning_rate * update
        return _OptimizerLeafUpdate(new_param.astype(param.dtype), new_m, new_v, muon_mom)

    map_args = (params, grads, state.adam_m, state.adam_v, state.muon_momentum, mask)
    updates = jax.tree_util.tree_map(lambda *items: update_one(*items), *map_args)
    is_update_leaf = lambda value: isinstance(value, _OptimizerLeafUpdate)
    new_params = jax.tree_util.tree_map(lambda item: item.param, updates, is_leaf=is_update_leaf)
    new_adam_m = jax.tree_util.tree_map(lambda item: item.adam_m, updates, is_leaf=is_update_leaf)
    new_adam_v = jax.tree_util.tree_map(lambda item: item.adam_v, updates, is_leaf=is_update_leaf)
    new_muon_m = jax.tree_util.tree_map(lambda item: item.muon_momentum, updates, is_leaf=is_update_leaf)
    return new_params, OptimState(new_adam_m, new_adam_v, new_muon_m, step)


def _scale_tree(tree: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda value: value * scale, tree)


def _add_tree(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def _average_accum_metrics(metrics_sum: dict[str, Array], accum_steps: int) -> dict[str, Array]:
    denom = float(max(1, accum_steps))
    metrics = dict(metrics_sum)
    for key in (
        "loss",
        "lm_loss",
        "moe_aux_loss",
        "mor_aux_loss",
        "router_entropy",
        "mean_depth",
        "mor_entropy",
        "mor_schedule_ratio",
        "mor_target_depth",
        "mor_aux_coef",
        "mor_expected_depth",
        "mor_packed_overflow_frac",
    ):
        if key in metrics:
            metrics[key] = metrics[key] / denom
    if "valid_assignments" in metrics and "total_assignments" in metrics:
        metrics["expert_drop_frac"] = 1.0 - metrics["valid_assignments"] / jnp.clip(metrics["total_assignments"], min=1.0)
    if "mor_packed_active_tokens" in metrics and "mor_packed_valid_tokens" in metrics:
        metrics["mor_packed_overflow_frac"] = jnp.where(
            metrics["mor_packed_active_tokens"] > 0,
            1.0 - metrics["mor_packed_valid_tokens"] / jnp.clip(metrics["mor_packed_active_tokens"], min=1.0),
            jnp.array(0.0, dtype=jnp.float32),
        )
    return metrics


def train_step(
    params: Params,
    opt_state: OptimState,
    batch: Mapping[str, Array],
    model_cfg: JaxMetisConfig,
    train_cfg: JaxMetisTrainConfig,
    mask: Params,
    expert_mesh: Any | None = None,
) -> tuple[Params, OptimState, dict[str, Array]]:
    _require_jax()
    tokens_per_step = train_cfg.grad_accum_steps * train_cfg.local_batch_size * model_cfg.block_size
    tokens_seen = jnp.asarray(opt_state.step, dtype=jnp.float32) * jnp.asarray(tokens_per_step, dtype=jnp.float32)

    def loss_fn(p, microbatch):
        loss, metrics = forward(
            p,
            microbatch["input_ids"],
            model_cfg,
            labels=microbatch["labels"],
            expert_mesh=expert_mesh,
            tokens_seen=tokens_seen,
        )
        return loss, metrics

    accum_steps = int(batch["input_ids"].shape[0]) if batch["input_ids"].ndim == 3 else 1
    grad_sum = None
    metrics_sum = None
    for micro_idx in range(accum_steps):
        microbatch = (
            {key: value[micro_idx] for key, value in batch.items()}
            if accum_steps > 1
            else batch
        )
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, microbatch)
        metrics = {**metrics, "loss": loss}
        grad_sum = grads if grad_sum is None else _add_tree(grad_sum, grads)
        metrics_sum = metrics if metrics_sum is None else {key: metrics_sum[key] + metrics[key] for key in metrics_sum}
    grads = _scale_tree(grad_sum, 1.0 / float(accum_steps))
    metrics = _average_accum_metrics(metrics_sum, accum_steps)
    params, opt_state = apply_optimizer(params, grads, opt_state, mask, train_cfg)
    metrics["grad_accum_steps"] = jnp.array(accum_steps, dtype=jnp.float32)
    return params, opt_state, metrics


def make_repeated_batch(*, batch_size: int, block_size: int, vocab_size: int) -> dict[str, np.ndarray]:
    row = np.arange(block_size, dtype=np.int32)
    batch = np.stack([(row * (i + 3) + (i * 7)) % vocab_size for i in range(batch_size)], axis=0)
    return {"input_ids": batch, "labels": batch.copy()}


def stack_microbatches(microbatches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not microbatches:
        raise ValueError("At least one microbatch is required.")
    return {
        key: np.stack([microbatch[key] for microbatch in microbatches], axis=0)
        for key in microbatches[0]
    }


def manifest_fingerprint(path: str | Path) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _data_fingerprint(path: Path, meta: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode("utf-8"))
    h.update(str(path.stat().st_size).encode("utf-8"))
    h.update(json.dumps(dict(meta), sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


class JaxMemmapTokenData:
    """Deterministic fixed-window memmap loader with explicit resume state."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str,
        batch_size: int,
        block_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.chunk_len = self.block_size + 1
        self.dp_rank = int(dp_rank)
        self.dp_world_size = max(1, int(dp_world_size))
        self.infinite = bool(infinite)
        meta_path = self.data_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing data meta file: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = np.dtype(self.meta.get("dtype", "uint16"))
        bin_path = self.data_dir / f"{self.split}.bin"
        if not bin_path.is_file():
            raise FileNotFoundError(f"Missing memmap split: {bin_path}")
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        if len(self.data) < self.chunk_len:
            raise ValueError(f"{bin_path} has {len(self.data)} tokens; need at least {self.chunk_len}.")
        self._initial_cursor = self.chunk_len * self.dp_rank
        self._stride = self.chunk_len * self.dp_world_size
        self._cursor = self._initial_cursor
        self._epoch = 0
        self._tokens_emitted = 0
        self._fingerprint = _data_fingerprint(bin_path, self.meta)

    @property
    def state(self) -> JaxSamplerState:
        return JaxSamplerState(
            split=self.split,
            cursor=int(self._cursor),
            epoch=int(self._epoch),
            tokens_emitted=int(self._tokens_emitted),
            data_fingerprint=self._fingerprint,
        )

    def load_state(self, state: JaxSamplerState | Mapping[str, Any]) -> None:
        if isinstance(state, Mapping):
            state = JaxSamplerState(
                split=str(state["split"]),
                cursor=int(state["cursor"]),
                epoch=int(state["epoch"]),
                tokens_emitted=int(state.get("tokens_emitted", 0)),
                data_fingerprint=str(state["data_fingerprint"]),
            )
        if state.split != self.split:
            raise ValueError(f"Sampler split mismatch: {state.split!r} vs {self.split!r}.")
        if state.data_fingerprint != self._fingerprint:
            raise ValueError("Sampler data fingerprint mismatch; refusing replay-prone resume.")
        if state.cursor < 0 or state.cursor >= len(self.data) + self.chunk_len:
            raise ValueError(f"Sampler cursor is out of range: {state.cursor}.")
        self._cursor = int(state.cursor)
        self._epoch = int(state.epoch)
        self._tokens_emitted = int(state.tokens_emitted)

    def next_batch(self) -> dict[str, np.ndarray]:
        windows: list[np.ndarray] = []
        while len(windows) < self.batch_size:
            if self._cursor + self.chunk_len > len(self.data):
                if not self.infinite:
                    raise StopIteration
                self._epoch += 1
                self._cursor = self._initial_cursor
            window = np.asarray(self.data[self._cursor : self._cursor + self.chunk_len], dtype=np.int32).copy()
            self._cursor += self._stride
            windows.append(window)
        stacked = np.stack(windows, axis=0)
        self._tokens_emitted += self.batch_size * self.block_size
        return {"input_ids": stacked[:, :-1], "labels": stacked[:, 1:]}


def _save_tree_npz(path: Path, tree: Any) -> None:
    arrays = {
        "__".join(path_parts): np.asarray(jax.device_get(value))
        for path_parts, value in _flatten_paths(tree)
        if hasattr(value, "shape")
    }
    np.savez(path, **arrays)


def _restore_tree_npz(path: Path, target: Any) -> Any:
    loaded = np.load(path)
    values = {
        tuple(key.split("__")): jnp.asarray(loaded[key])
        for key in loaded.files
    }
    restored = _unflatten_like(target, values)

    def match_target_sharding(value, target_value):
        sharding = getattr(target_value, "sharding", None)
        if sharding is None:
            return value
        return jax.device_put(value, sharding)

    return jax.tree_util.tree_map(match_target_sharding, restored, target)


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    params: Params,
    opt_state: OptimState,
    sampler_state: JaxSamplerState,
    step: int,
    manifest_hash: str,
    metrics: Mapping[str, Any] | None = None,
    backend: str = "orbax",
) -> None:
    _require_jax()
    if backend not in {"orbax", "npz"}:
        raise ValueError("checkpoint backend must be orbax or npz.")
    path = Path(checkpoint_dir).resolve()
    if backend == "orbax":
        import shutil
        import orbax.checkpoint as ocp

        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        state = {
            "params": params,
            "optimizer": {
                "adam_m": opt_state.adam_m,
                "adam_v": opt_state.adam_v,
                "muon_momentum": opt_state.muon_momentum,
                "step": opt_state.step,
            },
        }
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path / "state", state, force=True)
        checkpointer.wait_until_finished()
        write_dir = path
    else:
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            import shutil

            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        _save_tree_npz(tmp / "params.npz", params)
        _save_tree_npz(tmp / "adam_m.npz", opt_state.adam_m)
        _save_tree_npz(tmp / "adam_v.npz", opt_state.adam_v)
        _save_tree_npz(tmp / "muon_momentum.npz", opt_state.muon_momentum)
        write_dir = tmp
    metadata = {
        "format": f"metis_jax_{backend}_v1",
        "step": int(step),
        "optimizer_step": int(jax.device_get(opt_state.step)),
        "manifest_hash": str(manifest_hash),
        "sampler_state": asdict(sampler_state),
        "metrics": dict(metrics or {}),
    }
    (write_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if backend == "npz":
        if path.exists():
            import shutil

            shutil.rmtree(path)
        write_dir.rename(path)


def restore_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    target_params: Params,
    target_opt_state: OptimState,
    expected_manifest_hash: str | None = None,
) -> tuple[Params, OptimState, JaxSamplerState, dict[str, Any]]:
    _require_jax()
    path = Path(checkpoint_dir).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if expected_manifest_hash and metadata.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("Checkpoint manifest hash mismatch; refusing unsafe resume.")
    checkpoint_format = str(metadata.get("format", "metis_jax_npz_v1"))
    if checkpoint_format == "metis_jax_orbax_v1":
        import orbax.checkpoint as ocp

        target = {
            "params": target_params,
            "optimizer": {
                "adam_m": target_opt_state.adam_m,
                "adam_v": target_opt_state.adam_v,
                "muon_momentum": target_opt_state.muon_momentum,
                "step": target_opt_state.step,
            },
        }
        restored = ocp.StandardCheckpointer().restore(path / "state", target=target)
        params = restored["params"]
        optimizer = restored["optimizer"]
        opt_state = OptimState(
            adam_m=optimizer["adam_m"],
            adam_v=optimizer["adam_v"],
            muon_momentum=optimizer["muon_momentum"],
            step=jnp.asarray(optimizer["step"], dtype=jnp.int32),
        )
    elif checkpoint_format == "metis_jax_npz_v1":
        params = _restore_tree_npz(path / "params.npz", target_params)
        adam_m = _restore_tree_npz(path / "adam_m.npz", target_opt_state.adam_m)
        adam_v = _restore_tree_npz(path / "adam_v.npz", target_opt_state.adam_v)
        muon_momentum = _restore_tree_npz(path / "muon_momentum.npz", target_opt_state.muon_momentum)
        opt_state = OptimState(
            adam_m=adam_m,
            adam_v=adam_v,
            muon_momentum=muon_momentum,
            step=jnp.asarray(metadata.get("optimizer_step", metadata.get("step", 0)), dtype=jnp.int32),
        )
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    sampler_payload = metadata["sampler_state"]
    sampler_state = JaxSamplerState(
        split=str(sampler_payload["split"]),
        cursor=int(sampler_payload["cursor"]),
        epoch=int(sampler_payload["epoch"]),
        tokens_emitted=int(sampler_payload.get("tokens_emitted", 0)),
        data_fingerprint=str(sampler_payload["data_fingerprint"]),
    )
    return params, opt_state, sampler_state, metadata


def make_jit_train_step(
    model_cfg: JaxMetisConfig,
    train_cfg: JaxMetisTrainConfig,
    mask: Params,
    *,
    expert_mesh: Any | None = None,
):
    _require_jax()
    return jax.jit(
        lambda params, opt_state, batch: train_step(
            params,
            opt_state,
            batch,
            model_cfg,
            train_cfg,
            mask,
            expert_mesh=expert_mesh,
        ),
    )


def create_v6e_expert_mesh(devices: Any | None = None):
    _require_jax()
    devices = np.asarray(jax.devices() if devices is None else devices)
    if devices.size != 8:
        raise ValueError(f"Metis-1.5 v6e mesh expects exactly 8 devices; got {devices.size}.")
    from jax.sharding import Mesh

    return Mesh(devices.reshape((8,)), ("expert",))


def parameter_partition_specs(params: Params) -> Params:
    _require_jax()
    from jax.sharding import PartitionSpec as P

    def spec_for(path: tuple[str, ...], value: Any):
        name = "/".join(path)
        leaf_name = path[-1] if path else ""
        if leaf_name.startswith("expert_w") and getattr(value, "ndim", 0) == 3:
            return P("expert", None, None)
        if leaf_name == "router" and getattr(value, "ndim", 0) == 2 and value.shape[-1] % 8 == 0:
            return P(None, "expert")
        if leaf_name == "router_bias" and getattr(value, "ndim", 0) == 1:
            return P("expert")
        return P()

    flat = {path: spec_for(path, value) for path, value in _flatten_paths(params)}
    return _unflatten_like(params, flat)


def shard_params_for_v6e(params: Params, mesh: Any | None = None) -> Params:
    _require_jax()
    from jax.sharding import NamedSharding

    mesh = create_v6e_expert_mesh() if mesh is None else mesh
    specs = parameter_partition_specs(params)
    return jax.tree_util.tree_map(lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec)), params, specs)


def shard_optimizer_state_for_v6e(opt_state: OptimState, params: Params, mesh: Any | None = None) -> OptimState:
    _require_jax()
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = create_v6e_expert_mesh() if mesh is None else mesh

    def match_param(value, param):
        sharding = getattr(param, "sharding", None)
        if sharding is None:
            return value
        return jax.device_put(value, sharding)

    step = jax.device_put(opt_state.step, NamedSharding(mesh, P()))
    return OptimState(
        adam_m=jax.tree_util.tree_map(match_param, opt_state.adam_m, params),
        adam_v=jax.tree_util.tree_map(match_param, opt_state.adam_v, params),
        muon_momentum=jax.tree_util.tree_map(match_param, opt_state.muon_momentum, params),
        step=step,
    )


def shard_batch_for_v6e(batch: Mapping[str, Array], mesh: Any | None = None) -> dict[str, Array]:
    _require_jax()
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = create_v6e_expert_mesh() if mesh is None else mesh
    replicated = NamedSharding(mesh, P())
    return {key: jax.device_put(value, replicated) for key, value in batch.items()}


def count_params(params: Params) -> int:
    return int(sum(np.prod(value.shape) for _path, value in _flatten_paths(params) if hasattr(value, "shape")))


def _qk_clip_params(params: Params, cfg: JaxMetisConfig) -> tuple[Params, tuple[Array, ...], dict[str, float]]:
    _require_jax()

    def clip_layer(layer):
        q_norm = jnp.linalg.norm(layer["q"].astype(jnp.float32), axis=0, keepdims=True)
        k_norm = jnp.linalg.norm(layer["k"].astype(jnp.float32), axis=0, keepdims=True)
        max_logit = jnp.max(q_norm) * jnp.max(k_norm)
        scale = jnp.minimum(1.0, (cfg.qk_clip_threshold / jnp.clip(max_logit, min=1e-6)) ** cfg.qk_clip_alpha)
        return {**layer, "q": (layer["q"] * scale).astype(layer["q"].dtype), "k": (layer["k"] * scale).astype(layer["k"].dtype)}, max_logit, scale

    new_layers = []
    max_logits = []
    scales = []
    for layer in params["layers"]:
        clipped, max_logit, scale = clip_layer(layer)
        new_layers.append(clipped)
        max_logits.append(max_logit)
        scales.append(scale)
    new_params = {**params, "layers": tuple(new_layers)}
    return new_params, tuple(scales), {
        "qk_clip_max_logit": float(jax.device_get(jnp.max(jnp.stack(max_logits)))),
        "qk_clip_min_scale": float(jax.device_get(jnp.min(jnp.stack(scales)))),
        "qk_clip_scaled_layers": int(jax.device_get(jnp.sum(jnp.stack(scales) < 1.0))),
    }


def qk_clip(params: Params, cfg: JaxMetisConfig) -> tuple[Params, dict[str, float]]:
    new_params, _scales, metrics = _qk_clip_params(params, cfg)
    return new_params, metrics


def qk_clip_with_optimizer_state(
    params: Params,
    opt_state: OptimState,
    cfg: JaxMetisConfig,
) -> tuple[Params, OptimState, dict[str, float]]:
    """Apply QK clipping and mirror the same transform into optimizer state.

    The old TPU lane taught us not to mutate model weights while leaving FP32
    master/state tensors semantically behind. JAX keeps params as the optimizer
    source of truth here, but Muon momentum and any AdamW moments for Q/K should
    still follow the same scale transform.
    """

    new_params, scales, metrics = _qk_clip_params(params, cfg)

    def sync_layer_state(layer_state, scale, *, square: bool = False):
        state_scale = jnp.square(scale) if square else scale
        return {
            **layer_state,
            "q": (layer_state["q"] * state_scale).astype(layer_state["q"].dtype),
            "k": (layer_state["k"] * state_scale).astype(layer_state["k"].dtype),
        }

    new_adam_m_layers = []
    new_adam_v_layers = []
    new_muon_layers = []
    for layer_m, layer_v, layer_muon, scale in zip(
        opt_state.adam_m["layers"],
        opt_state.adam_v["layers"],
        opt_state.muon_momentum["layers"],
        scales,
    ):
        new_adam_m_layers.append(sync_layer_state(layer_m, scale, square=False))
        new_adam_v_layers.append(sync_layer_state(layer_v, scale, square=True))
        new_muon_layers.append(sync_layer_state(layer_muon, scale, square=False))
    new_opt_state = OptimState(
        adam_m={**opt_state.adam_m, "layers": tuple(new_adam_m_layers)},
        adam_v={**opt_state.adam_v, "layers": tuple(new_adam_v_layers)},
        muon_momentum={**opt_state.muon_momentum, "layers": tuple(new_muon_layers)},
        step=opt_state.step,
    )
    return new_params, new_opt_state, metrics


def with_stage(cfg: JaxMetisConfig, stage: str) -> JaxMetisConfig:
    if stage == "pretrain":
        return replace(
            cfg,
            training_mode="static_dense_pretrain",
            mor_enabled=False,
            mor_train_router=False,
            mor_runtime_mode="disabled",
            mor_compute_mode="soft_fixed_depth",
        )
    if stage == "continued_pretrain":
        return replace(
            cfg,
            training_mode="dynamic_token_mor",
            mor_enabled=True,
            mor_train_router=True,
            mor_runtime_mode="dynamic_token",
            mor_compute_mode="static_packed_hard",
        )
    raise ValueError("stage must be pretrain or continued_pretrain.")
