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
    rope_theta: float = 10000.0
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
    moe_backend: str = "jax_static_sort_pack"
    # "argsort": sort-based packing (reference). "cumsum": one-hot prefix-sum
    # slot assignment with identical drop semantics — removes the per-layer
    # argsort+searchsorted, the main non-GEMM dispatch cost on TPU.
    moe_dispatch_impl: str = "argsort"
    moe_balance_bias_update_rate: float = 1e-3
    moe_balance_bias_clamp: float = 5.0
    moe_expert_parallel_size: int = 8
    expert_capacity_factor: float = 4.0
    expert_capacity_alignment: int = 128
    qk_clip_threshold: float = 100.0
    qk_clip_alpha: float = 0.5
    remat_layers: bool = True
    remat_attention: bool = False
    expert_execution: str = "reference"
    attention_backend: str = "jax_causal_attention_reference"
    # "float32": QK^T accumulates and materializes fp32 scores (most precise).
    # "bfloat16": scores stored bf16 (fp32 MXU accumulation inside the dot),
    # softmax max/sum reductions still computed in fp32 — halves the dominant
    # attention HBM stream at seq 1024.
    attention_scores_dtype: str = "float32"
    ce_logits_dtype: str = "float32"
    ce_loss_impl: str = "standard"

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
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embeddings.")
        if self.moe_num_experts % self.moe_expert_parallel_size != 0:
            raise ValueError("moe_num_experts must be divisible by moe_expert_parallel_size.")
        if self.moe_expert_parallel_size not in {1, 8}:
            raise ValueError("Metis-1.5 JAX TPU lane supports full data parallelism or 8-way expert sharding.")
        if self.moe_top_k <= 0 or self.moe_top_k > self.moe_num_experts:
            raise ValueError("moe_top_k must be in [1, moe_num_experts].")
        if self.moe_activation != "squared_relu":
            raise ValueError("JAX LatentMoE lane currently implements squared_relu routed experts.")
        if self.moe_router_score not in {"sigmoid", "softmax"}:
            raise ValueError("moe_router_score must be sigmoid or softmax.")
        if self.moe_backend not in {"jax_static_sort_pack", "pallas_megablox_gmm"}:
            raise ValueError("moe_backend must be jax_static_sort_pack or pallas_megablox_gmm.")
        if self.moe_dispatch_impl not in {"argsort", "cumsum"}:
            raise ValueError("moe_dispatch_impl must be argsort or cumsum.")
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
        if self.expert_execution not in {"reference", "shard_map", "pmap_data", "data_parallel"}:
            raise ValueError("expert_execution must be reference, shard_map, pmap_data, or data_parallel.")
        if self.attention_backend not in {"jax_causal_attention_reference", "pallas_flash_attention"}:
            raise ValueError("attention_backend must be jax_causal_attention_reference or pallas_flash_attention.")
        if self.attention_scores_dtype not in {"float32", "bfloat16"}:
            raise ValueError("attention_scores_dtype must be float32 or bfloat16.")
        if self.ce_logits_dtype not in {"float32", "bfloat16", "model"}:
            raise ValueError("ce_logits_dtype must be float32, bfloat16, or model.")
        if self.ce_loss_impl not in {"standard", "vocab_parallel"}:
            raise ValueError("ce_loss_impl must be standard or vocab_parallel.")


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
    optimizer: str = "adamuon"
    adamuon_matrix_policy: str = "all"
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    muon_lr_scale: float = 1.0
    muon_scale_mode: str = "match_rms_adamw"
    muon_nesterov: bool = True
    warmup_steps: int = 100
    warmup_ratio: float = 0.0
    lr_schedule: str = "constant"
    lr_min_ratio: float = 0.1
    max_steps: int = 1000
    log_interval: int = 1
    checkpoint_interval: int = 100
    qk_clip_enabled: bool = True
    qk_clip_interval: int = 1
    qk_clip_warmup_steps: int = 0
    grad_accum_impl: str = "loop"
    # Dtype for the cross-device gradient all-reduce. bfloat16 halves the
    # 2-bytes-per-param ICI transfer per step; the optimizer still consumes
    # fp32 (cast back after the mean).
    grad_allreduce_dtype: str = "float32"


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
        rope_theta=float(model.get("rope_theta", 10000.0)),
        tie_embeddings=bool(model["tie_embeddings"]),
        initializer_range=float(model["initializer_range"]),
        dtype=str(model.get("jax_dtype", model.get("torch_dtype", "bfloat16"))),
        weight_dtype=str(model.get("weight_dtype", hardware.get("weight_dtype", "bfloat16"))),
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
        moe_backend=str(model.get("moe_backend", "jax_static_sort_pack")),
        moe_balance_bias_update_rate=float(model["moe_balance_bias_update_rate"]),
        moe_balance_bias_clamp=float(model["moe_balance_bias_clamp"]),
        moe_expert_parallel_size=int(model["moe_expert_parallel_size"]),
        expert_capacity_factor=float(stability.get("expert_capacity_factor", 4.0)),
        expert_capacity_alignment=int(model.get("moe_capacity_alignment", 128)),
        qk_clip_threshold=float(stability.get("qk_clip", {}).get("threshold", 100.0)),
        qk_clip_alpha=float(stability.get("qk_clip", {}).get("alpha", 0.5)),
        remat_layers=bool(hardware.get("remat_layers", model.get("remat_layers", True))),
        remat_attention=bool(hardware.get("remat_attention", model.get("remat_attention", False))),
        expert_execution=str(model.get("expert_execution", hardware.get("expert_execution", "data_parallel"))),
        attention_backend=str(model.get("attention_backend", "jax_causal_attention_reference")),
        ce_logits_dtype=str(stability.get("ce_logits_dtype", "float32")),
        ce_loss_impl=str(stability.get("ce_loss_impl", "standard")),
    )
    qk_clip_cfg = stability.get("qk_clip", {}) if isinstance(stability.get("qk_clip", {}), Mapping) else {}
    optimizer_payload = dict(payload.get("optimizer", {}))
    optimizer_name = str(optimizer_payload.get("name", "adamuon")).lower().replace("-", "_")
    if optimizer_name in {"hybrid_muon_adamw", "muon_adamw"}:
        optimizer_name = "muon_adamw"
    elif optimizer_name != "adamuon":
        optimizer_name = "adamuon"
    max_steps = max(
        1,
        int(stage_payload["target_train_tokens"])
        // max(
            1,
            int(stage_payload.get("local_batch_size", 1))
            * int(stage_payload.get("grad_accum_steps", 1))
            * cfg.block_size,
        ),
    )
    warmup_ratio = float(stage_payload.get("warmup_ratio", 0.0))
    train_cfg = JaxMetisTrainConfig(
        stage=stage,
        local_batch_size=int(stage_payload.get("local_batch_size", 1)),
        grad_accum_steps=int(stage_payload.get("grad_accum_steps", 1)),
        learning_rate=float(stage_payload["base_lr"]),
        weight_decay=float(stage_payload["weight_decay"]),
        beta1=float(stage_payload["optimizer_beta1"]),
        beta2=float(stage_payload["optimizer_beta2"]),
        adamw_eps=float(optimizer_payload.get("adamw_eps", 1e-8)),
        optimizer=optimizer_name,
        muon_beta=float(optimizer_payload.get("muon_beta", 0.95)),
        muon_ns_steps=int(optimizer_payload.get("muon_ns_steps", 5)),
        muon_lr_scale=float(optimizer_payload.get("muon_lr_scale", 1.0)),
        muon_scale_mode=str(optimizer_payload.get("muon_scale_mode", "match_rms_adamw")),
        muon_nesterov=bool(optimizer_payload.get("muon_nesterov", True)),
        warmup_ratio=warmup_ratio,
        warmup_steps=max(1, int(round(warmup_ratio * max_steps))),
        lr_schedule=str(stage_payload.get("lr_schedule", "warmup_cosine")),
        lr_min_ratio=float(stage_payload.get("lr_min_ratio", 0.1)),
        max_steps=max_steps,
        log_interval=max(1, int(stage_payload.get("log_interval", 20))),
        checkpoint_interval=int(stage_payload.get("checkpoint_interval", 100)),
        qk_clip_enabled=bool(qk_clip_cfg.get("enabled", True)),
        qk_clip_interval=int(qk_clip_cfg.get("interval", 1)),
        qk_clip_warmup_steps=int(qk_clip_cfg.get("warmup_steps", 0)),
        grad_accum_impl=str(stage_payload.get("grad_accum_impl", hardware.get("grad_accum_impl", "scan"))),
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
    # Statistics in fp32, output cast back to the input dtype so downstream
    # matmuls stay on the bf16 MXU path instead of silently promoting to fp32.
    x_f = x.astype(jnp.float32)
    normed = x_f * lax.rsqrt(jnp.mean(jnp.square(x_f), axis=-1, keepdims=True) + eps)
    return (normed * scale.astype(jnp.float32)).astype(x.dtype)


def _rope_cos_sin(positions: Array, head_dim: int, theta: float) -> tuple[Array, Array]:
    """NeoX/Llama half-split rotary tables: channel pair (i, i + head_dim/2) shares one frequency."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim)))
    freqs = positions.astype(jnp.float32)[..., None] * inv_freq
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x: Array) -> Array:
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    # Rotation in fp32, result back in the input dtype for MXU-friendly attention matmuls.
    x_f = x.astype(jnp.float32)
    return (x_f * cos + _rotate_half(x_f) * sin).astype(x.dtype)


def _causal_attention(x: Array, layer: Mapping[str, Any], cfg: JaxMetisConfig) -> Array:
    bsz, seq, _ = x.shape
    q = jnp.einsum("btd,dh->bth", x, layer["q"]).reshape(bsz, seq, cfg.n_heads, cfg.head_dim)
    k = jnp.einsum("btd,dh->bth", x, layer["k"]).reshape(bsz, seq, cfg.n_kv_heads, cfg.head_dim)
    v = jnp.einsum("btd,dh->bth", x, layer["v"]).reshape(bsz, seq, cfg.n_kv_heads, cfg.head_dim)
    cos, sin = _rope_cos_sin(jnp.arange(seq, dtype=jnp.int32), cfg.head_dim, cfg.rope_theta)
    q = _apply_rope(q, cos[None, :, None, :], sin[None, :, None, :])
    k = _apply_rope(k, cos[None, :, None, :], sin[None, :, None, :])
    group = cfg.n_heads // cfg.n_kv_heads
    k = jnp.repeat(k, group, axis=2)
    v = jnp.repeat(v, group, axis=2)
    if cfg.attention_backend == "pallas_flash_attention":
        try:
            from jax.experimental.pallas.ops.tpu import flash_attention as pallas_flash_attention
        except Exception as exc:  # pragma: no cover - depends on TPU JAX extras.
            raise RuntimeError("pallas_flash_attention backend requires jax.experimental.pallas.ops.tpu.flash_attention.") from exc

        q_flash = jnp.swapaxes(q, 1, 2)
        k_flash = jnp.swapaxes(k, 1, 2)
        v_flash = jnp.swapaxes(v, 1, 2)
        out = pallas_flash_attention.flash_attention(
            q_flash,
            k_flash,
            v_flash,
            causal=True,
            sm_scale=1.0 / math.sqrt(float(cfg.head_dim)),
        )
        out = jnp.swapaxes(out, 1, 2).reshape(bsz, seq, cfg.n_heads * cfg.head_dim)
        return jnp.einsum("btd,dh->bth", out, layer["o"])
    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
    if cfg.attention_scores_dtype == "bfloat16":
        # Scores stored bf16 (MXU still accumulates the dot in fp32 internally).
        # Softmax: exact bf16 max, fp32 sum reduction, bf16 normalize — no
        # [B,H,T,T] fp32 buffer ever materializes, halving attention HBM traffic.
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.asarray(
            math.sqrt(float(cfg.head_dim)), dtype=q.dtype
        )
        logits = jnp.where(causal[None, None, :, :], logits, jnp.asarray(-1e9, dtype=logits.dtype))
        score_max = jnp.max(logits, axis=-1, keepdims=True)
        unnorm = jnp.exp(logits - score_max)
        denom = jnp.sum(unnorm, axis=-1, keepdims=True, dtype=jnp.float32)
        probs = (unnorm * (1.0 / denom).astype(unnorm.dtype)).astype(cfg.activation_dtype)
    else:
        # bf16 operands with fp32 accumulation: full MXU rate, fp32 softmax numerics.
        logits = jnp.einsum(
            "bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32
        ) / math.sqrt(float(cfg.head_dim))
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
    cos, sin = _rope_cos_sin(jnp.arange(seq, dtype=jnp.int32), cfg.head_dim, cfg.rope_theta)
    k_all = _apply_rope(k_all, cos[None, :, None, :], sin[None, :, None, :])
    group = cfg.n_heads // cfg.n_kv_heads
    k_all = jnp.repeat(k_all, group, axis=2)
    v_all = jnp.repeat(v_all, group, axis=2)
    batch_ids = (packed_tokens // seq).astype(jnp.int32)
    pos_ids = (packed_tokens % seq).astype(jnp.int32)
    q = q_all.reshape(bsz * seq, cfg.n_heads, cfg.head_dim)[packed_tokens]
    q = _apply_rope(q, cos[pos_ids][:, None, :], sin[pos_ids][:, None, :])
    k = jnp.swapaxes(k_all, 1, 2)[batch_ids]
    v = jnp.swapaxes(v_all, 1, 2)[batch_ids]
    logits = jnp.einsum(
        "chd,chkd->chk", q, k, preferred_element_type=jnp.float32
    ) / math.sqrt(float(cfg.head_dim))
    key_pos = jnp.arange(seq, dtype=jnp.int32)
    causal = key_pos[None, :] <= pos_ids[:, None]
    valid = packed_valid.astype(bool)
    logits = jnp.where(causal[:, None, :] & valid[:, None, None], logits, jnp.array(-1e9, dtype=jnp.float32))
    probs = jax.nn.softmax(logits, axis=-1).astype(cfg.activation_dtype)
    out = jnp.einsum("chk,chkd->chd", probs, v).reshape(capacity, cfg.n_heads * cfg.head_dim)
    out = jnp.einsum("cd,dh->ch", out, layer["o"])
    return jnp.where(valid[:, None], out, jnp.zeros_like(out))


def _topk_route_scores(
    logits: Array, balance_bias: Array | None, cfg: JaxMetisConfig
) -> tuple[Array, Array, Array]:
    """Top-k routing with aux-loss-free balance bias.

    The balance bias only steers *selection* (which experts win top-k); the
    combine gates always come from the unbiased scores, DeepSeek-V3 style.
    The bias receives no gradients — it is updated out-of-band from observed
    expert load in train_step.
    """
    logits = logits.astype(jnp.float32)
    if cfg.moe_router_score == "sigmoid":
        scores = jax.nn.sigmoid(logits / cfg.moe_router_temperature)
    else:
        scores = jax.nn.softmax(logits / cfg.moe_router_temperature, axis=-1)
    if balance_bias is not None:
        select_scores = scores + lax.stop_gradient(balance_bias.astype(jnp.float32))[None, :]
    else:
        select_scores = scores
    _, top_idx = lax.top_k(select_scores, cfg.moe_top_k)
    top_scores = jnp.take_along_axis(scores, top_idx, axis=-1)
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


def _pack_assignments_cumsum(
    latent: Array,
    top_idx: Array,
    top_weights: Array,
    *,
    capacity: int,
    num_experts: int,
    token_mask: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Sort-free packing: slot = exclusive prefix count of each expert's assignments.

    Token-major assignment order is preserved, so drop semantics match the
    argsort implementation exactly; only the argsort/searchsorted machinery is
    replaced by a cumsum and direct scatters.
    """
    n_tokens, latent_dim = latent.shape
    k = int(top_idx.shape[-1])
    token_ids = jnp.repeat(jnp.arange(n_tokens, dtype=jnp.int32), k)
    expert_ids = top_idx.reshape(-1).astype(jnp.int32)
    weights = top_weights.reshape(-1).astype(jnp.float32)
    if token_mask is None:
        assignment_valid = jnp.ones((n_tokens * k,), dtype=bool)
    else:
        assignment_valid = jnp.repeat(token_mask.reshape(-1).astype(bool), k)
    onehot = jax.nn.one_hot(expert_ids, num_experts, dtype=jnp.int32) * assignment_valid[:, None].astype(jnp.int32)
    slot = jnp.sum((jnp.cumsum(onehot, axis=0) - onehot) * onehot, axis=-1)
    valid = assignment_valid & (slot < capacity)
    overflow = num_experts * capacity  # parking slot for dropped assignments
    dest = jnp.where(valid, expert_ids * capacity + slot, overflow)
    expert_in = jnp.zeros((num_experts * capacity + 1, latent_dim), dtype=latent.dtype)
    expert_in = expert_in.at[dest].set(jnp.where(valid[:, None], latent[token_ids], jnp.zeros_like(latent[token_ids])))
    packed_tokens = jnp.zeros((num_experts * capacity + 1,), dtype=jnp.int32).at[dest].set(
        jnp.where(valid, token_ids, 0)
    )
    packed_weights = jnp.zeros((num_experts * capacity + 1,), dtype=jnp.float32).at[dest].set(
        jnp.where(valid, weights, 0.0)
    )
    valid_grid = jnp.zeros((num_experts * capacity + 1,), dtype=bool).at[dest].set(valid)
    expert_in = expert_in[:-1].reshape(num_experts, capacity, latent_dim)
    packed_tokens = packed_tokens[:-1].reshape(num_experts, capacity)
    packed_weights = packed_weights[:-1].reshape(num_experts, capacity)
    valid_grid = valid_grid[:-1].reshape(num_experts, capacity)
    valid_assignments = jnp.sum(valid.astype(jnp.int32))
    return expert_in, packed_tokens, packed_weights, valid_grid, valid_assignments


def _squared_relu_experts(expert_in: Array, w1: Array, w2: Array) -> Array:
    hidden = jnp.einsum("ecl,elh->ech", expert_in, w1)
    hidden = jnp.square(jnp.maximum(hidden, jnp.array(0.0, dtype=hidden.dtype)))
    return jnp.einsum("ech,ehl->ecl", hidden, w2)


def _megablox_squared_relu_experts(expert_in: Array, w1: Array, w2: Array) -> Array:
    try:
        from jax.experimental.pallas.ops.tpu.megablox import ops as megablox_ops
    except Exception as exc:  # pragma: no cover - depends on TPU JAX extras.
        raise RuntimeError("pallas_megablox_gmm MoE backend requires jax.experimental.pallas.ops.tpu.megablox.") from exc

    num_experts, capacity, latent_dim = expert_in.shape
    flat_in = expert_in.reshape(num_experts * capacity, latent_dim)
    group_sizes = jnp.full((num_experts,), capacity, dtype=jnp.int32)
    hidden = megablox_ops.gmm(
        flat_in,
        w1,
        group_sizes,
        preferred_element_type=expert_in.dtype,
        tiling=(128, 128, 128),
    )
    hidden = jnp.square(jnp.maximum(hidden, jnp.array(0.0, dtype=hidden.dtype)))
    out = megablox_ops.gmm(
        hidden,
        w2,
        group_sizes,
        preferred_element_type=expert_in.dtype,
        tiling=(128, 128, 128),
    )
    return out.reshape(num_experts, capacity, latent_dim)


def _megablox_routed_experts(
    latent: Array,
    top_idx: Array,
    top_weights: Array,
    w1: Array,
    w2: Array,
    *,
    num_experts: int,
) -> tuple[Array, Array, Array, Array]:
    try:
        from jax.experimental.pallas.ops.tpu.megablox import ops as megablox_ops
    except Exception as exc:  # pragma: no cover - depends on TPU JAX extras.
        raise RuntimeError("pallas_megablox_gmm MoE backend requires jax.experimental.pallas.ops.tpu.megablox.") from exc

    n_tokens = latent.shape[0]
    assignments = n_tokens * top_idx.shape[-1]
    if assignments % 128 != 0:
        raise ValueError("pallas_megablox_gmm requires routed assignment count to be divisible by 128.")
    arange_assign = jnp.arange(assignments, dtype=jnp.int32)
    token_ids = jnp.repeat(jnp.arange(n_tokens, dtype=jnp.int32), top_idx.shape[-1])
    expert_ids = top_idx.reshape(-1).astype(jnp.int32)
    weights = top_weights.reshape(-1).astype(jnp.float32)
    keys = expert_ids * assignments + arange_assign
    order = jnp.argsort(keys)
    sorted_experts = expert_ids[order]
    sorted_tokens = token_ids[order]
    sorted_weights = weights[order]
    group_sizes = jnp.sum(
        jax.nn.one_hot(sorted_experts, num_experts, dtype=jnp.int32),
        axis=0,
    ).astype(jnp.int32)
    expert_in = latent[sorted_tokens]
    hidden = megablox_ops.gmm(
        expert_in,
        w1,
        group_sizes,
        preferred_element_type=expert_in.dtype,
        tiling=(128, 128, 128),
    )
    hidden = jnp.square(jnp.maximum(hidden, jnp.array(0.0, dtype=hidden.dtype)))
    expert_out = megablox_ops.gmm(
        hidden,
        w2,
        group_sizes,
        preferred_element_type=expert_in.dtype,
        tiling=(128, 128, 128),
    )
    valid_assignments = jnp.asarray(assignments, dtype=jnp.int32)
    return expert_out, sorted_tokens, sorted_weights, valid_assignments


def sharded_squared_relu_experts(expert_in: Array, w1: Array, w2: Array, mesh: Any) -> Array:
    """Run routed experts over an explicit expert-axis mesh.

    The global expert dimension is sharded across the mesh. On v6e-8 with 32 experts,
    each TPU chip owns exactly four routed experts. The function body sees only its
    local expert slice, which avoids the Python ragged-dispatch pattern that hurt the
    Trainium path.
    """

    _require_jax()
    from jax.sharding import PartitionSpec as P

    has_data_axis = "data" in getattr(mesh, "axis_names", ())
    expert_in_spec = P("expert", "data", None) if has_data_axis else P("expert", None, None)

    @jax.shard_map(
        mesh=mesh,
        in_specs=(expert_in_spec, P("expert", None, None), P("expert", None, None)),
        out_specs=expert_in_spec,
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
    # Routing decisions in fp32; tiny [n_tokens, n_experts] GEMM, negligible cost.
    logits = jnp.einsum("nd,de->ne", router_input, layer["router"], preferred_element_type=jnp.float32)
    top_idx, top_weights, scores = _topk_route_scores(logits, layer["router_bias"], cfg)
    if cfg.moe_backend == "pallas_megablox_gmm" and cfg.expert_execution != "shard_map" and token_mask is None:
        expert_out, packed_tokens, packed_weights, valid_assignments = _megablox_routed_experts(
            latent,
            top_idx,
            top_weights,
            layer["expert_w1"],
            layer["expert_w2"],
            num_experts=cfg.moe_num_experts,
        )
        weighted = expert_out * packed_weights[:, None].astype(expert_out.dtype)
        routed_latent = jnp.zeros((n_tokens, cfg.moe_routed_latent_size), dtype=expert_out.dtype)
        routed_latent = routed_latent.at[packed_tokens].add(weighted)
    else:
        pack_fn = _pack_assignments_cumsum if cfg.moe_dispatch_impl == "cumsum" else _pack_assignments
        expert_in, packed_tokens, packed_weights, valid, valid_assignments = pack_fn(
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
        elif cfg.moe_backend == "pallas_megablox_gmm":
            expert_out = _megablox_squared_relu_experts(expert_in, layer["expert_w1"], layer["expert_w2"])
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
        "expert_load": load.astype(jnp.float32),
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
    if cfg.remat_attention:
        attn_out = jax.checkpoint(lambda y, layer_params: _causal_attention(y, layer_params, cfg))(attn_in, layer)
    else:
        attn_out = _causal_attention(attn_in, layer, cfg)
    x = x + attn_out.astype(x.dtype)
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
    if cfg.remat_attention:
        packed_attn = jax.checkpoint(
            lambda y, layer_params, tokens, valid: _causal_attention_packed_queries(y, layer_params, cfg, tokens, valid)
        )(attn_in, layer, packed_tokens, packed_valid)
    else:
        packed_attn = _causal_attention_packed_queries(
            attn_in,
            layer,
            cfg,
            packed_tokens,
            packed_valid,
        )
    packed_after_attn = packed_x + packed_attn.astype(x.dtype)
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


def vocab_parallel_tied_ce_loss(
    hidden: Array,
    embed: Array,
    labels: Array,
    *,
    mesh: Any,
    ce_dtype: Any,
) -> Array:
    _require_jax()
    from jax.sharding import PartitionSpec as P

    mesh_size = int(np.asarray(mesh.devices).size)
    vocab_size = int(embed.shape[0])
    if vocab_size % mesh_size != 0:
        raise ValueError(f"vocab_size={vocab_size} must be divisible by mesh size {mesh_size}.")
    vocab_per_shard = vocab_size // mesh_size

    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )
    def ce_kernel(local_hidden, full_embed, local_labels):
        axis_idx = lax.axis_index("expert")
        vocab_start = axis_idx * vocab_per_shard
        embed_shard = lax.dynamic_slice_in_dim(
            full_embed.astype(ce_dtype),
            vocab_start,
            vocab_per_shard,
            axis=0,
        )
        valid = local_labels != CE_IGNORE_INDEX
        masked_labels = jnp.where(valid, local_labels, 0)
        local_logits = jnp.einsum("btd,vd->btv", local_hidden.astype(ce_dtype), embed_shard)
        local_logits_f = local_logits.astype(jnp.float32)
        local_max = jnp.max(local_logits_f, axis=-1)
        global_max = lax.pmax(lax.stop_gradient(local_max), "expert")
        local_exp_sum = jnp.sum(jnp.exp(local_logits_f - global_max[..., None]), axis=-1)
        global_exp_sum = lax.psum(local_exp_sum, "expert")
        local_label = masked_labels - vocab_start
        in_shard = (local_label >= 0) & (local_label < vocab_per_shard)
        safe_label = jnp.clip(local_label, 0, vocab_per_shard - 1)
        local_target = jnp.take_along_axis(local_logits_f, safe_label[..., None], axis=-1).squeeze(-1)
        target = lax.psum(jnp.where(in_shard, local_target, jnp.zeros_like(local_target)), "expert")
        nll = jnp.log(jnp.clip(global_exp_sum, min=1e-30)) + global_max - target
        valid_f = valid.astype(jnp.float32)
        return (jnp.sum(nll * valid_f) / jnp.clip(jnp.sum(valid_f), min=1.0)).astype(jnp.float32)

    return ce_kernel(hidden, embed, labels)


CE_IGNORE_INDEX = -100  # matches the torch lane's F.cross_entropy(ignore_index=-100)


def full_vocab_ce_loss(hidden: Array, output_weight: Array, labels: Array, *, tied: bool, ce_dtype: Any) -> Array:
    """Next-token CE with torch-compatible ignore_index=-100 masking.

    Pretraining labels contain no -100s, so the masked mean reduces to the
    plain mean (bit-identical denominators). SFT/DPO labels mask prompt, BOS,
    and padding positions with -100 exactly like the torch lane.
    """
    valid = labels != CE_IGNORE_INDEX
    safe_labels = jnp.where(valid, labels, 0)
    if tied:
        logits = jnp.einsum("btd,vd->btv", hidden.astype(ce_dtype), output_weight.astype(ce_dtype))
    else:
        logits = jnp.einsum("btd,dv->btv", hidden.astype(ce_dtype), output_weight.astype(ce_dtype))
    if logits.dtype == jnp.float32:
        max_logits = jnp.max(logits, axis=-1)
        exp_sum = jnp.sum(jnp.exp(logits - max_logits[..., None]), axis=-1)
        target = jnp.take_along_axis(logits, safe_labels[..., None], axis=-1).squeeze(-1)
    else:
        # Streaming softmax stats: never materialize a fp32 [B, T, V] tensor.
        # max is exact on the stored dtype; exp stays in the stored dtype
        # (values <= 1); only the running sum accumulates in fp32.
        max_logits_low = jnp.max(logits, axis=-1)
        exp_sum = jnp.sum(
            jnp.exp(logits - max_logits_low[..., None]), axis=-1, dtype=jnp.float32
        )
        max_logits = max_logits_low.astype(jnp.float32)
        target = jnp.take_along_axis(logits, safe_labels[..., None], axis=-1).squeeze(-1).astype(jnp.float32)
    nll = jnp.log(jnp.clip(exp_sum, min=1e-30)) + max_logits - target
    valid_f = valid.astype(jnp.float32)
    return (jnp.sum(nll * valid_f) / jnp.clip(jnp.sum(valid_f), min=1.0)).astype(jnp.float32)


def _cast_params_for_compute(params: Params, cfg: JaxMetisConfig) -> Params:
    """Cast stored (fp32 master) weights to the activation dtype for compute.

    The stored tree stays fp32 so tiny optimizer updates survive; the cast
    happens inside the differentiated function, so gradients flow back in
    fp32 via the astype transpose. router_bias is exempt: it is an fp32
    non-gradient balance bias consumed directly by the router.
    """
    target = cfg.activation_dtype

    def cast(path: tuple[str, ...], value: Any) -> Any:
        if path and path[-1] == "router_bias":
            return value
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != target:
            return value.astype(target)
        return value

    flat = {path: cast(path, value) for path, value in _flatten_paths(params)}
    return _unflatten_like(params, flat)


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
    """Forward pass. NOTE: `labels` must equal `input_ids` (same window, unshifted);
    the next-token shift happens internally. Do not pass pre-shifted labels."""
    _require_jax()
    capacity = cfg.capacity_for_batch(int(input_ids.shape[0])) if capacity is None else int(capacity)
    params = _cast_params_for_compute(params, cfg)
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
    uses_dynamic_mor = cfg.mor_enabled and cfg.mor_runtime_mode == "dynamic_token"
    layer_moe_calls = cfg.mor_max_depth if uses_dynamic_mor else 1
    expert_loads: list[Array] = []
    for layer in params["layers"]:
        if uses_dynamic_mor and cfg.mor_compute_mode == "soft_fixed_depth":
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
                        for key in (
                            "moe_aux_loss",
                            "valid_assignments",
                            "total_assignments",
                            "router_entropy",
                            "expert_load",
                        )
                    }
                    depth_metrics["expert_drop_frac"] = (
                        1.0
                        - depth_metrics["valid_assignments"]
                        / jnp.clip(depth_metrics["total_assignments"], min=1.0)
                    )
                gate = mor_probs[..., depth_idx : depth_idx + 1].astype(current.dtype)
                x = x + gate * (current - layer_input)
            metrics = depth_metrics if depth_metrics is not None else layer_metrics
        elif uses_dynamic_mor and cfg.mor_compute_mode == "static_packed_hard":
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
                    for key in (
                        "moe_aux_loss",
                        "valid_assignments",
                        "total_assignments",
                        "router_entropy",
                        "expert_load",
                    )
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
        expert_loads.append(metrics["expert_load"] / float(layer_moe_calls))
        for key in ("moe_aux_loss", "valid_assignments", "total_assignments", "router_entropy"):
            metrics_acc[key] = metrics_acc[key] + metrics[key]
    metrics_acc["expert_load_per_layer"] = jnp.stack(expert_loads, axis=0)
    metrics_acc["mor_packed_overflow_frac"] = jnp.where(
        metrics_acc["mor_packed_active_tokens"] > 0,
        1.0
        - metrics_acc["mor_packed_valid_tokens"]
        / jnp.clip(metrics_acc["mor_packed_active_tokens"], min=1.0),
        jnp.array(0.0, dtype=jnp.float32),
    )
    x = rms_norm(x, params["final_norm"]["scale"])
    if cfg.ce_logits_dtype == "float32":
        ce_dtype = jnp.float32
    elif cfg.ce_logits_dtype == "bfloat16":
        ce_dtype = jnp.bfloat16
    else:
        ce_dtype = cfg.activation_dtype
    if labels is not None and cfg.ce_loss_impl == "vocab_parallel":
        if not cfg.tie_embeddings:
            raise ValueError("vocab_parallel CE currently requires tied embeddings.")
        if expert_mesh is None:
            raise ValueError("vocab_parallel CE requires the expert mesh.")
        shift_labels = labels[:, 1:]
        lm_loss = vocab_parallel_tied_ce_loss(
            x[:, :-1, :],
            params["embed"],
            shift_labels,
            mesh=expert_mesh,
            ce_dtype=ce_dtype,
        )
        total_loss = lm_loss + metrics_acc["moe_aux_loss"] + metrics_acc["mor_aux_loss"]
        metrics_acc["lm_loss"] = lm_loss.astype(jnp.float32)
        metrics_acc["loss"] = total_loss.astype(jnp.float32)
        metrics_acc["expert_drop_frac"] = 1.0 - metrics_acc["valid_assignments"] / jnp.clip(metrics_acc["total_assignments"], min=1.0)
        return total_loss.astype(jnp.float32), metrics_acc
    if labels is None:
        if cfg.tie_embeddings:
            logits = jnp.einsum("btd,vd->btv", x.astype(ce_dtype), params["embed"].astype(ce_dtype))
        else:
            logits = jnp.einsum("btd,dv->btv", x.astype(ce_dtype), params["lm_head"].astype(ce_dtype))
        return logits, metrics_acc
    shift_labels = labels[:, 1:]
    if cfg.tie_embeddings:
        lm_loss = full_vocab_ce_loss(x[:, :-1, :], params["embed"], shift_labels, tied=True, ce_dtype=ce_dtype)
    else:
        lm_loss = full_vocab_ce_loss(x[:, :-1, :], params["lm_head"], shift_labels, tied=False, ce_dtype=ce_dtype)
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
    if getattr(x, "ndim", 0) < 2:
        raise ValueError("Newton-Schulz orthogonalization requires at least a matrix.")
    norm = jnp.linalg.norm(x, axis=(-2, -1), keepdims=True)
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = jnp.swapaxes(x, -2, -1)
    x = x / jnp.clip(norm, min=1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ jnp.swapaxes(x, -2, -1)
        x = (a * x) + ((b * xx_t + c * (xx_t @ xx_t)) @ x)
    if transpose:
        x = jnp.swapaxes(x, -2, -1)
    return x


def _muon_update_scale(shape: tuple[int, ...], mode: str) -> float:
    rows, cols = int(shape[-2]), int(shape[-1])
    if mode == "match_rms_adamw":
        return 0.2 * math.sqrt(float(max(rows, cols, 1)))
    if mode == "original":
        return math.sqrt(max(1.0, float(rows) / float(max(cols, 1))))
    raise ValueError("muon_scale_mode must be original or match_rms_adamw.")


def scheduled_learning_rate(cfg: JaxMetisTrainConfig, step: Array) -> Array:
    """Learning rate for a (1-indexed) optimizer step: linear warmup then cosine decay."""
    if cfg.lr_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError("lr_schedule must be constant or warmup_cosine.")
    base = jnp.asarray(cfg.learning_rate, dtype=jnp.float32)
    if cfg.lr_schedule == "constant":
        return base
    s = step.astype(jnp.float32)
    warmup = jnp.asarray(float(max(1, cfg.warmup_steps)), dtype=jnp.float32)
    warm_factor = jnp.minimum(1.0, s / warmup)
    total = jnp.asarray(float(max(1, cfg.max_steps)), dtype=jnp.float32)
    progress = jnp.clip((s - warmup) / jnp.maximum(1.0, total - warmup), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    floor = jnp.asarray(cfg.lr_min_ratio, dtype=jnp.float32)
    return base * warm_factor * (floor + (1.0 - floor) * cosine)


def apply_optimizer(
    params: Params,
    grads: Params,
    state: OptimState,
    mask: Params,
    cfg: JaxMetisTrainConfig,
) -> tuple[Params, OptimState, Array]:
    _require_jax()
    if cfg.optimizer not in {"adamuon", "muon_adamw"}:
        raise ValueError("optimizer must be adamuon or muon_adamw.")
    step = state.step + 1
    lr = scheduled_learning_rate(cfg, step)
    beta1_t = cfg.beta1
    beta2_t = cfg.beta2

    def update_one(param, grad, adam_m, adam_v, muon_mom, is_muon):
        grad_f = grad.astype(jnp.float32)
        if cfg.optimizer == "adamuon" and is_muon and getattr(param, "ndim", 0) >= 2:
            mom = cfg.muon_beta * muon_mom + grad_f
            direction = grad_f + cfg.muon_beta * mom if cfg.muon_nesterov else mom
            # Newton-Schulz approximates the *matrix* sign (msign = U V^T). Do not
            # take an elementwise sign first — that destroys the singular structure
            # and degenerates the update into signSGD.
            direction = _zeropower_via_newton_schulz5(direction, steps=cfg.muon_ns_steps)
            new_v = cfg.muon_beta * adam_v + (1.0 - cfg.muon_beta) * jnp.square(direction)
            update = direction / (jnp.sqrt(new_v) + cfg.adamw_eps)
            rows, cols = int(param.shape[-2]), int(param.shape[-1])
            scale = 0.2 * math.sqrt(float(max(1, rows) * max(1, cols))) / (
                jnp.linalg.norm(update, axis=(-2, -1), keepdims=True) + cfg.adamw_eps
            )
            update = update * scale * cfg.muon_lr_scale
            decayed = param.astype(jnp.float32) * (1.0 - lr * cfg.weight_decay)
            new_param = decayed - lr * update
            return _OptimizerLeafUpdate(new_param.astype(param.dtype), adam_m, new_v, mom)
        if cfg.optimizer == "adamuon":
            new_m = beta1_t * adam_m + (1.0 - beta1_t) * grad_f
            new_v = beta2_t * adam_v + (1.0 - beta2_t) * jnp.square(grad_f)
            bias1 = 1.0 - beta1_t**step.astype(jnp.float32)
            bias2 = 1.0 - beta2_t**step.astype(jnp.float32)
            update = (new_m / jnp.clip(bias1, min=1e-16)) / (
                jnp.sqrt(new_v / jnp.clip(bias2, min=1e-16)) + cfg.adamw_eps
            )
            new_param = param.astype(jnp.float32) - lr * update
            return _OptimizerLeafUpdate(new_param.astype(param.dtype), new_m, new_v, muon_mom)
        if is_muon:
            mom = cfg.muon_beta * muon_mom + grad_f
            update = grad_f + cfg.muon_beta * mom if cfg.muon_nesterov else mom
            update = _zeropower_via_newton_schulz5(update, steps=cfg.muon_ns_steps)
            update = update * (_muon_update_scale(param.shape, cfg.muon_scale_mode) * cfg.muon_lr_scale)
            decayed = param.astype(jnp.float32) * (1.0 - lr * cfg.weight_decay)
            new_param = decayed - lr * update
            return _OptimizerLeafUpdate(new_param.astype(param.dtype), adam_m, adam_v, mom)
        new_m = beta1_t * adam_m + (1.0 - beta1_t) * grad_f
        new_v = beta2_t * adam_v + (1.0 - beta2_t) * jnp.square(grad_f)
        bias1 = 1.0 - beta1_t**step.astype(jnp.float32)
        bias2 = 1.0 - beta2_t**step.astype(jnp.float32)
        update = (new_m / jnp.clip(bias1, min=1e-16)) / (jnp.sqrt(new_v / jnp.clip(bias2, min=1e-16)) + cfg.adamw_eps)
        decayed = param.astype(jnp.float32) * (1.0 - lr * cfg.weight_decay)
        new_param = decayed - lr * update
        return _OptimizerLeafUpdate(new_param.astype(param.dtype), new_m, new_v, muon_mom)

    map_args = (params, grads, state.adam_m, state.adam_v, state.muon_momentum, mask)
    updates = jax.tree_util.tree_map(lambda *items: update_one(*items), *map_args)
    is_update_leaf = lambda value: isinstance(value, _OptimizerLeafUpdate)
    new_params = jax.tree_util.tree_map(lambda item: item.param, updates, is_leaf=is_update_leaf)
    new_adam_m = jax.tree_util.tree_map(lambda item: item.adam_m, updates, is_leaf=is_update_leaf)
    new_adam_v = jax.tree_util.tree_map(lambda item: item.adam_v, updates, is_leaf=is_update_leaf)
    new_muon_m = jax.tree_util.tree_map(lambda item: item.muon_momentum, updates, is_leaf=is_update_leaf)
    return new_params, OptimState(new_adam_m, new_adam_v, new_muon_m, step), lr


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
        "expert_load_per_layer",
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
    data_axis_name: str | None = None,
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
    if train_cfg.grad_accum_impl not in {"loop", "scan"}:
        raise ValueError("grad_accum_impl must be loop or scan.")
    if accum_steps > 1 and train_cfg.grad_accum_impl == "scan":
        metric_keys = (
            "loss",
            "lm_loss",
            "moe_aux_loss",
            "mor_aux_loss",
            "valid_assignments",
            "total_assignments",
            "expert_drop_frac",
            "router_entropy",
            "mean_depth",
            "mor_entropy",
            "mor_schedule_ratio",
            "mor_target_depth",
            "mor_aux_coef",
            "mor_expected_depth",
            "mor_packed_active_tokens",
            "mor_packed_valid_tokens",
            "mor_packed_overflow_frac",
            "expert_load_per_layer",
        )
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, params)
        zero_metrics = {key: jnp.array(0.0, dtype=jnp.float32) for key in metric_keys}
        zero_metrics["expert_load_per_layer"] = jnp.zeros(
            (model_cfg.n_layer, model_cfg.moe_num_experts), dtype=jnp.float32
        )

        def scan_micro(carry, microbatch):
            grad_acc, metrics_acc = carry
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, microbatch)
            metrics = {**metrics, "loss": loss}
            grad_acc = _add_tree(grad_acc, grads)
            metrics_acc = {key: metrics_acc[key] + metrics[key] for key in metric_keys}
            return (grad_acc, metrics_acc), None

        (grad_sum, metrics_sum), _ = lax.scan(scan_micro, (zero_grads, zero_metrics), batch)
    else:
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
    if data_axis_name is not None:
        if train_cfg.grad_allreduce_dtype == "bfloat16":
            grads = jax.tree_util.tree_map(
                lambda value: lax.pmean(value.astype(jnp.bfloat16), data_axis_name).astype(jnp.float32),
                grads,
            )
        elif train_cfg.grad_allreduce_dtype == "float32":
            grads = jax.tree_util.tree_map(lambda value: lax.pmean(value, data_axis_name), grads)
        else:
            raise ValueError("grad_allreduce_dtype must be float32 or bfloat16.")
        sum_keys = {
            "valid_assignments",
            "total_assignments",
            "mor_packed_active_tokens",
            "mor_packed_valid_tokens",
        }
        reduced_metrics = {}
        for key, value in metrics.items():
            reduced_metrics[key] = (
                lax.psum(value, data_axis_name)
                if key in sum_keys
                else lax.pmean(value, data_axis_name)
            )
        metrics = reduced_metrics
        if "valid_assignments" in metrics and "total_assignments" in metrics:
            metrics["expert_drop_frac"] = 1.0 - metrics["valid_assignments"] / jnp.clip(metrics["total_assignments"], min=1.0)
        if "mor_packed_active_tokens" in metrics and "mor_packed_valid_tokens" in metrics:
            metrics["mor_packed_overflow_frac"] = jnp.where(
                metrics["mor_packed_active_tokens"] > 0,
                1.0
                - metrics["mor_packed_valid_tokens"]
                / jnp.clip(metrics["mor_packed_active_tokens"], min=1.0),
                jnp.array(0.0, dtype=jnp.float32),
            )
    params, opt_state, lr = apply_optimizer(params, grads, opt_state, mask, train_cfg)
    metrics["learning_rate"] = lr.astype(jnp.float32)
    if model_cfg.moe_balance_bias_update_rate > 0.0:
        # Aux-loss-free balancing (DeepSeek-V3): nudge each layer's selection bias
        # toward uniform expert load with a non-gradient sign update. Loads are
        # already averaged over grad-accum microbatches and pmean'd across the
        # data-parallel axis, so every replica applies an identical update.
        loads = metrics["expert_load_per_layer"]
        target = 1.0 / float(model_cfg.moe_num_experts)
        rate = float(model_cfg.moe_balance_bias_update_rate)
        clamp = float(model_cfg.moe_balance_bias_clamp)
        new_layers = []
        for layer_idx, layer in enumerate(params["layers"]):
            bias = layer["router_bias"]
            error = jnp.asarray(target, dtype=jnp.float32) - loads[layer_idx]
            new_bias = jnp.clip(bias + rate * jnp.sign(error).astype(bias.dtype), -clamp, clamp)
            new_layers.append({**layer, "router_bias": new_bias})
        params = {**params, "layers": tuple(new_layers)}
    if train_cfg.qk_clip_enabled:
        interval = max(1, int(train_cfg.qk_clip_interval))
        gate = (opt_state.step >= train_cfg.qk_clip_warmup_steps) & (
            jnp.mod(opt_state.step, interval) == 0
        )
        params, scales, qk_metrics = _qk_clip_transform(params, model_cfg, gate=gate)
        opt_state = _qk_clip_sync_state(opt_state, scales)
        metrics.update(qk_metrics)
    else:
        metrics["qk_clip_max_logit"] = jnp.array(0.0, dtype=jnp.float32)
        metrics["qk_clip_min_scale"] = jnp.array(1.0, dtype=jnp.float32)
        metrics["qk_clip_scaled_layers"] = jnp.array(0.0, dtype=jnp.float32)
    metrics["grad_accum_steps"] = jnp.array(accum_steps, dtype=jnp.float32)
    return params, opt_state, metrics


class JaxSftData:
    """Fixed-shape SFT loader: reads prep's {split}_input_ids.bin (uint16 [N,block])
    + {split}_labels.bin (int32 [N,block], -100 on prompt/pad). Labels are already
    masked, so forward() consumes them directly. Shuffle + resume mirror
    JaxMemmapTokenData so the orchestrator's resume path is identical."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str,
        batch_size: int,
        block_size: int,
        shuffle: bool = True,
        shuffle_seed: int = 20260613,
        infinite: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.infinite = bool(infinite)
        meta = json.loads((self.data_dir / "meta.json").read_text(encoding="utf-8"))
        block = int(meta.get("block", block_size))
        if block != block_size:
            raise ValueError(f"SFT data block {block} != model block_size {block_size}.")
        ids_path = self.data_dir / f"{self.split}_input_ids.bin"
        lab_path = self.data_dir / f"{self.split}_labels.bin"
        if not ids_path.is_file() or not lab_path.is_file():
            raise FileNotFoundError(f"Missing SFT split files: {ids_path} / {lab_path}")
        self._ids = np.memmap(ids_path, dtype=np.uint16, mode="r").reshape(-1, block)
        self._labels = np.memmap(lab_path, dtype=np.int32, mode="r").reshape(-1, block)
        if len(self._ids) != len(self._labels) or len(self._ids) == 0:
            raise ValueError("SFT input_ids/labels row count mismatch or empty.")
        self._n = len(self._ids)
        self._cursor = 0
        self._epoch = 0
        self._tokens_emitted = 0
        self._perm: np.ndarray | None = None
        self._fingerprint = _data_fingerprint(ids_path, meta) + (f":shuffle:{self.shuffle_seed}" if self.shuffle else "")

    def _epoch_perm(self) -> np.ndarray:
        if self._perm is None:
            if self.shuffle:
                self._perm = np.random.default_rng(self.shuffle_seed + self._epoch).permutation(self._n)
            else:
                self._perm = np.arange(self._n)
        return self._perm

    @property
    def state(self) -> JaxSamplerState:
        return JaxSamplerState(self.split, int(self._cursor), int(self._epoch),
                               int(self._tokens_emitted), self._fingerprint)

    def load_state(self, state: JaxSamplerState | Mapping[str, Any]) -> None:
        if isinstance(state, Mapping):
            state = JaxSamplerState(str(state["split"]), int(state["cursor"]), int(state["epoch"]),
                                    int(state.get("tokens_emitted", 0)), str(state["data_fingerprint"]))
        if state.data_fingerprint != self._fingerprint:
            raise ValueError("SFT sampler fingerprint mismatch; refusing replay-prone resume.")
        if int(state.epoch) != self._epoch:
            self._perm = None
        self._cursor = int(state.cursor) % self._n
        self._epoch = int(state.epoch)
        self._tokens_emitted = int(state.tokens_emitted)

    def next_batch(self) -> dict[str, np.ndarray]:
        rows = []
        while len(rows) < self.batch_size:
            if self._cursor >= self._n:
                if not self.infinite:
                    raise StopIteration
                self._epoch += 1
                self._cursor = 0
                self._perm = None
            rows.append(int(self._epoch_perm()[self._cursor]))
            self._cursor += 1
        idx = np.asarray(rows, dtype=np.int64)
        self._tokens_emitted += self.batch_size * self.block_size
        return {
            "input_ids": np.asarray(self._ids[idx], dtype=np.int32),
            "labels": np.asarray(self._labels[idx], dtype=np.int32),
        }


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
        shuffle: bool = False,
        shuffle_seed: int = 20260611,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.block_size = int(block_size)
        self.chunk_len = self.block_size + 1
        self.dp_rank = int(dp_rank)
        self.dp_world_size = max(1, int(dp_world_size))
        self.infinite = bool(infinite)
        self.shuffle = bool(shuffle)
        self.shuffle_seed = int(shuffle_seed)
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
        self._n_windows = len(self.data) // self.chunk_len
        self._initial_cursor = self.chunk_len * self.dp_rank
        self._stride = self.chunk_len * self.dp_world_size
        self._cursor = self._initial_cursor
        self._epoch = 0
        self._tokens_emitted = 0
        self._perm: np.ndarray | None = None
        base_fingerprint = _data_fingerprint(bin_path, self.meta)
        if self.shuffle:
            # Window-shuffled sampling decorrelates batches from corpus
            # neighborhoods (domain-clustered batches overload the same routed
            # experts simultaneously, which capacity/bias cannot absorb).
            # cursor = position in the seeded per-epoch permutation, so resume
            # stays deterministic.
            self._fingerprint = f"{base_fingerprint}:shuffle:{self.shuffle_seed}"
        else:
            self._fingerprint = base_fingerprint

    def _epoch_perm(self) -> np.ndarray:
        if self._perm is None:
            rng = np.random.default_rng(self.shuffle_seed + self._epoch)
            self._perm = rng.permutation(self._n_windows)
        return self._perm

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
            base = self._fingerprint.split(":shuffle:")[0]
            if self.shuffle and state.data_fingerprint == base:
                # One-way transition: checkpoint written by the sequential
                # sampler, resuming under shuffled sampling. Start a fresh
                # shuffled pass; params/optimizer are unaffected.
                print(
                    "Sampler transitioning sequential -> shuffled; starting fresh shuffled pass.",
                    flush=True,
                )
                return
            raise ValueError("Sampler data fingerprint mismatch; refusing replay-prone resume.")
        if state.cursor < 0 or state.cursor >= len(self.data) + self.chunk_len:
            raise ValueError(f"Sampler cursor is out of range: {state.cursor}.")
        self._cursor = int(state.cursor)
        if int(state.epoch) != self._epoch:
            self._perm = None
        self._epoch = int(state.epoch)
        self._tokens_emitted = int(state.tokens_emitted)

    def next_batch(self) -> dict[str, np.ndarray]:
        """Return a fixed-shape batch with `labels == input_ids`.

        forward() performs the next-token shift internally (hidden[:, :-1]
        against labels[:, 1:]), so the loader must NOT pre-shift. Returning
        pre-shifted labels here double-shifts and silently trains the model
        to predict two tokens ahead — that bug plateaued real-data loss
        around ~7 while synthetic smoke runs (which already used unshifted
        labels) looked healthy.
        """
        windows: list[np.ndarray] = []
        if self.shuffle:
            while len(windows) < self.batch_size:
                if self._cursor >= self._n_windows:
                    if not self.infinite:
                        raise StopIteration
                    self._epoch += 1
                    self._cursor = self.dp_rank
                    self._perm = None
                start = int(self._epoch_perm()[self._cursor]) * self.chunk_len
                window = np.asarray(self.data[start : start + self.chunk_len], dtype=np.int32).copy()
                self._cursor += self.dp_world_size
                windows.append(window)
        else:
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
        window_ids = stacked[:, :-1]
        return {"input_ids": window_ids, "labels": window_ids.copy()}


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


def restore_params_only(
    checkpoint_dir: str | Path, *, target_params: Params, target_opt_state: "OptimState"
) -> tuple[Params, dict[str, Any]]:
    """Restore params for a fresh phase WITHOUT putting the saved optimizer in HBM.

    Loading the checkpoint's optimizer into HBM and then allocating a fresh one
    transiently doubles optimizer memory (~2x10.8G fp32) and OOMs the training
    program — the CPT init-from-checkpoint bug. We load the full checkpoint to
    HOST RAM (plentiful) and return only params; the caller replicates just the
    params to the device and builds a fresh optimizer. HBM never sees the saved
    optimizer. Returns params as host arrays (replicate_for_pmap places them).
    """
    _require_jax()
    path = Path(checkpoint_dir).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    checkpoint_format = str(metadata.get("format", "metis_jax_npz_v1"))
    if checkpoint_format == "metis_jax_orbax_v1":
        import orbax.checkpoint as ocp
        from jax.sharding import SingleDeviceSharding

        host = SingleDeviceSharding(jax.devices("cpu")[0])
        to_host = lambda tree: jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=host), tree
        )
        target = {
            "params": to_host(target_params),
            "optimizer": {
                "adam_m": to_host(target_opt_state.adam_m),
                "adam_v": to_host(target_opt_state.adam_v),
                "muon_momentum": to_host(target_opt_state.muon_momentum),
                "step": jax.ShapeDtypeStruct(
                    target_opt_state.step.shape, target_opt_state.step.dtype, sharding=host
                ),
            },
        }
        restored = ocp.StandardCheckpointer().restore(path / "state", target=target)
        params = restored["params"]  # host arrays; restored optimizer (also host) is discarded
    elif checkpoint_format == "metis_jax_npz_v1":
        params = _restore_tree_npz(path / "params.npz", target_params)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")
    return params, metadata


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


def _pmap_axis0_sharding(devices: Any, ndim: int):
    """Sharding for a [device_count, ...] stacked array, axis 0 split across devices.

    Built on NamedSharding (stable from jax 0.4 through 0.10+) because
    device_put_replicated / device_put_sharded / PmapSharding were removed in
    newer JAX releases.
    """
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(devices), ("data",))
    return NamedSharding(mesh, PartitionSpec("data", *([None] * (ndim - 1))))


def replicate_for_pmap(tree: Any, devices: Any) -> Any:
    """Replicate a pytree across devices with a leading device axis, once.

    The result feeds jax.pmap with in_axes=0 and stays on-device across steps;
    do NOT re-replicate inside the step loop.
    """
    _require_jax()
    device_count = len(devices)

    def rep(value):
        host = np.asarray(jax.device_get(value))
        stacked = np.broadcast_to(host, (device_count,) + host.shape)
        return jax.device_put(stacked, _pmap_axis0_sharding(devices, stacked.ndim))

    return jax.tree_util.tree_map(rep, tree)


def put_sharded_for_pmap(per_device: list[np.ndarray], devices: Any) -> Array:
    """Place per-device numpy shards directly onto their devices (axis 0 = device)."""
    _require_jax()
    stacked = np.stack([np.ascontiguousarray(np.asarray(shard)) for shard in per_device], axis=0)
    return jax.device_put(stacked, _pmap_axis0_sharding(devices, stacked.ndim))


def make_pmap_data_parallel_train_step(
    model_cfg: JaxMetisConfig,
    train_cfg: JaxMetisTrainConfig,
    mask: Params,
):
    """Data-parallel train step over replicated state.

    params/opt_state must be replicated across devices once up front
    (replicate_for_pmap) and then stay on-device for the whole run.
    The previous in_axes/out_axes=None pattern re-broadcast the full ~12GB
    params+optimizer tree from host/device0 to every chip on every step and
    gathered it back afterwards — that transfer, not compute, was the
    throughput ceiling. Donation lets XLA reuse the state buffers in place.
    """
    _require_jax()
    return jax.pmap(
        lambda params, opt_state, batch: train_step(
            params,
            opt_state,
            batch,
            model_cfg,
            train_cfg,
            mask,
            data_axis_name="data",
        ),
        axis_name="data",
        in_axes=(0, 0, 0),
        out_axes=(0, 0, 0),
        donate_argnums=(0, 1),
    )


def optimizer_matrix_mask(
    params: Params,
    optimizer: str = "adamuon",
    *,
    adamuon_matrix_policy: str = "all",
) -> Params:
    if optimizer == "adamuon":
        if adamuon_matrix_policy not in {"all", "no_embed_head"}:
            raise ValueError("adamuon_matrix_policy must be all or no_embed_head.")

        def classify(path: tuple[str, ...], value: Any) -> bool:
            if not (hasattr(value, "ndim") and value.ndim >= 2):
                return False
            if adamuon_matrix_policy == "no_embed_head" and any(part in path for part in ("embed", "lm_head")):
                return False
            return True

        flat = {path: classify(path, value) for path, value in _flatten_paths(params)}
        return _unflatten_like(params, flat)
    if optimizer == "muon_adamw":
        return muon_mask(params)
    raise ValueError("optimizer must be adamuon or muon_adamw.")


def create_v6e_expert_mesh(devices: Any | None = None, mesh_shape: str = "1x8"):
    _require_jax()
    devices = np.asarray(jax.devices() if devices is None else devices)
    if devices.size != 8:
        raise ValueError(f"Metis-1.5 v6e mesh expects exactly 8 devices; got {devices.size}.")
    from jax.sharding import Mesh

    if mesh_shape == "1x8":
        return Mesh(devices.reshape((8,)), ("expert",))
    if mesh_shape == "2x4":
        return Mesh(devices.reshape((2, 4)), ("data", "expert"))
    raise ValueError("mesh_shape must be 1x8 or 2x4.")


def mesh_axis_size(mesh: Any, axis_name: str, default: int | None = None) -> int:
    names = tuple(getattr(mesh, "axis_names", ()))
    if axis_name not in names:
        if default is None:
            raise ValueError(f"Mesh has no axis named {axis_name!r}.")
        return int(default)
    return int(np.asarray(mesh.devices).shape[names.index(axis_name)])


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


def shard_batch_for_v6e(
    batch: Mapping[str, Array],
    mesh: Any | None = None,
    *,
    batch_sharding: str = "replicated",
) -> dict[str, Array]:
    _require_jax()
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = create_v6e_expert_mesh() if mesh is None else mesh
    if batch_sharding == "replicated":
        replicated = NamedSharding(mesh, P())
        return {key: jax.device_put(value, replicated) for key, value in batch.items()}
    if batch_sharding != "data":
        raise ValueError("batch_sharding must be replicated or data.")
    data_axis = "data" if "data" in getattr(mesh, "axis_names", ()) else "expert"
    mesh_size = mesh_axis_size(mesh, data_axis)

    def put_data_sharded(value):
        if value.ndim == 3:
            if value.shape[1] % mesh_size != 0:
                raise ValueError(f"Batch dimension {value.shape[1]} must be divisible by mesh size {mesh_size}.")
            spec = P(None, data_axis, None)
        elif value.ndim == 2:
            if value.shape[0] % mesh_size != 0:
                raise ValueError(f"Batch dimension {value.shape[0]} must be divisible by mesh size {mesh_size}.")
            spec = P(data_axis, None)
        else:
            spec = P()
        return jax.device_put(value, NamedSharding(mesh, spec))

    return {key: put_data_sharded(value) for key, value in batch.items()}


def count_params(params: Params) -> int:
    return int(sum(np.prod(value.shape) for _path, value in _flatten_paths(params) if hasattr(value, "shape")))


def _qk_clip_transform(
    params: Params,
    cfg: JaxMetisConfig,
    *,
    gate: Array | None = None,
) -> tuple[Params, tuple[Array, ...], dict[str, Array]]:
    """Pure (trace-safe) QK clip. `gate` is an optional traced bool: when False the
    transform is the identity, so it can run inside the jitted/pmapped train step."""
    _require_jax()

    def clip_layer(layer):
        q_norm = jnp.linalg.norm(layer["q"].astype(jnp.float32), axis=0, keepdims=True)
        k_norm = jnp.linalg.norm(layer["k"].astype(jnp.float32), axis=0, keepdims=True)
        max_logit = jnp.max(q_norm) * jnp.max(k_norm)
        scale = jnp.minimum(1.0, (cfg.qk_clip_threshold / jnp.clip(max_logit, min=1e-6)) ** cfg.qk_clip_alpha)
        if gate is not None:
            scale = jnp.where(gate, scale, jnp.ones_like(scale))
        return (
            {**layer, "q": (layer["q"] * scale).astype(layer["q"].dtype), "k": (layer["k"] * scale).astype(layer["k"].dtype)},
            max_logit,
            scale,
        )

    new_layers = []
    max_logits = []
    scales = []
    for layer in params["layers"]:
        clipped, max_logit, scale = clip_layer(layer)
        new_layers.append(clipped)
        max_logits.append(max_logit)
        scales.append(scale)
    new_params = {**params, "layers": tuple(new_layers)}
    metrics = {
        "qk_clip_max_logit": jnp.max(jnp.stack(max_logits)).astype(jnp.float32),
        "qk_clip_min_scale": jnp.min(jnp.stack(scales)).astype(jnp.float32),
        "qk_clip_scaled_layers": jnp.sum(jnp.stack(scales) < 1.0).astype(jnp.float32),
    }
    return new_params, tuple(scales), metrics


def _qk_clip_sync_state(opt_state: OptimState, scales: tuple[Array, ...]) -> OptimState:
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
    return OptimState(
        adam_m={**opt_state.adam_m, "layers": tuple(new_adam_m_layers)},
        adam_v={**opt_state.adam_v, "layers": tuple(new_adam_v_layers)},
        muon_momentum={**opt_state.muon_momentum, "layers": tuple(new_muon_layers)},
        step=opt_state.step,
    )


def _qk_clip_params(params: Params, cfg: JaxMetisConfig) -> tuple[Params, tuple[Array, ...], dict[str, float]]:
    new_params, scales, metrics = _qk_clip_transform(params, cfg)
    host_metrics = {
        "qk_clip_max_logit": float(jax.device_get(metrics["qk_clip_max_logit"])),
        "qk_clip_min_scale": float(jax.device_get(metrics["qk_clip_min_scale"])),
        "qk_clip_scaled_layers": int(jax.device_get(metrics["qk_clip_scaled_layers"])),
    }
    return new_params, scales, host_metrics


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
    new_opt_state = _qk_clip_sync_state(opt_state, scales)
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
