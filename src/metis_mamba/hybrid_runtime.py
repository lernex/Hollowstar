from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torch import nn


DEFAULT_ATTN_LAYER_IDX = [3, 7, 11, 15, 19, 23, 27]


@dataclass
class MetisHybridConfig:
    name: str = "Metis-1.3"
    model_type: str = "metis_mamba2_hybrid"
    architecture: str = "mamba2_hybrid_decoder"
    vocab_size: int = 8192
    block_size: int = 4096
    d_model: int = 1152
    n_layer: int = 28
    n_heads: int = 18
    n_kv_heads: int = 6
    head_dim: int = 64
    attn_layer_idx: list[int] = field(default_factory=lambda: list(DEFAULT_ATTN_LAYER_IDX))
    attn_d_conv: int = 4
    attn_rotary_emb_dim: int = 0
    ssm_layer: str = "Mamba2"
    ssm_d_state: int = 64
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    tie_embeddings: bool = True
    rms_norm: bool = True
    residual_in_fp32: bool = False
    fused_add_norm: bool = False
    pad_vocab_size_multiple: int = 16
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetisHybridConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    @property
    def padded_vocab_size(self) -> int:
        if self.vocab_size % self.pad_vocab_size_multiple == 0:
            return self.vocab_size
        return self.vocab_size + self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)


@dataclass
class HybridCausalLMOutput:
    logits: torch.Tensor


class HybridRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        rstd = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rstd * self.weight.float()).to(dtype)


class HybridGatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float = 1e-5, group_size: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.group_size = group_size

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float() * F.silu(gate.float())
        if self.group_size and self.group_size < x_float.shape[-1]:
            original_shape = x_float.shape
            x_group = x_float.reshape(*original_shape[:-1], -1, self.group_size)
            rstd = torch.rsqrt(x_group.square().mean(dim=-1, keepdim=True) + self.eps)
            x_float = (x_group * rstd).reshape(original_shape)
        else:
            rstd = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
            x_float = x_float * rstd
        return (x_float * self.weight.float()).to(dtype)


class HybridMamba2(nn.Module):
    def __init__(self, config: MetisHybridConfig, *, layer_idx: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.d_state = config.ssm_d_state
        self.d_conv = config.ssm_d_conv
        self.expand = config.ssm_expand
        self.d_inner = self.expand * self.d_model
        self.headdim = config.head_dim
        self.d_ssm = self.d_inner
        self.ngroups = 1
        self.nheads = self.d_ssm // self.headdim
        self.activation = "silu"

        factory_kwargs = {"dtype": dtype}
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=conv_dim,
            bias=True,
            **factory_kwargs,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
        self.A_log = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = HybridGatedRMSNorm(self.d_ssm, group_size=self.d_ssm // self.ngroups)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        zxbcdt = self.in_proj(hidden_states)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xbc, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        xbc = self.conv1d(xbc.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
        xbc = F.silu(xbc)
        x, b_vec, c_vec = torch.split(
            xbc,
            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        x = x.reshape(batch_size, seq_len, self.nheads, self.headdim).float()
        b_vec = b_vec.reshape(batch_size, seq_len, self.ngroups, self.d_state).float()
        c_vec = c_vec.reshape(batch_size, seq_len, self.ngroups, self.d_state).float()
        dt = F.softplus(dt.float() + self.dt_bias.float())
        a = -torch.exp(self.A_log.float())
        d = self.D.float()

        state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        y_steps: list[torch.Tensor] = []
        heads_per_group = self.nheads // self.ngroups
        head_groups = torch.arange(self.nheads, device=hidden_states.device) // heads_per_group
        for index in range(seq_len):
            dt_t = dt[:, index]
            x_t = x[:, index]
            b_t = b_vec[:, index].index_select(1, head_groups)
            c_t = c_vec[:, index].index_select(1, head_groups)
            state = state * torch.exp(dt_t[:, :, None, None] * a[None, :, None, None])
            state = state + dt_t[:, :, None, None] * x_t[:, :, :, None] * b_t[:, :, None, :]
            y_t = torch.einsum("bhpn,bhn->bhp", state, c_t)
            y_t = y_t + d[None, :, None] * x_t
            y_steps.append(y_t.reshape(batch_size, self.d_ssm))

        y = torch.stack(y_steps, dim=1).to(dtype)
        y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        return self.out_proj(y)


class HybridMHA(nn.Module):
    def __init__(self, config: MetisHybridConfig, *, layer_idx: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_conv = config.attn_d_conv
        self.softmax_scale = self.head_dim ** -0.5
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_kv_heads)
        factory_kwargs = {"dtype": dtype}
        self.in_proj = nn.Linear(config.d_model, qkv_dim, bias=False, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            qkv_dim,
            qkv_dim,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=qkv_dim,
            bias=True,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.in_proj(hidden_states)
        qkv = self.conv1d(qkv.transpose(1, 2)).transpose(1, 2)[:, :seq_len]
        q, kv = torch.split(
            qkv,
            [self.num_heads * self.head_dim, 2 * self.num_kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.view(batch_size, seq_len, 2, self.num_kv_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = torch.repeat_interleave(k, repeats=repeats, dim=1)
        v = torch.repeat_interleave(v, repeats=repeats, dim=1)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.softmax_scale)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(context)


class HybridBlock(nn.Module):
    def __init__(self, config: MetisHybridConfig, *, layer_idx: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.norm = HybridRMSNorm(config.d_model)
        if layer_idx in set(config.attn_layer_idx):
            self.mixer = HybridMHA(config, layer_idx=layer_idx, dtype=dtype)
        else:
            self.mixer = HybridMamba2(config, layer_idx=layer_idx, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states + residual if residual is not None else hidden_states
        hidden_states = self.norm(residual)
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class HybridBackbone(nn.Module):
    def __init__(self, config: MetisHybridConfig, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.padded_vocab_size, config.d_model, dtype=dtype)
        self.layers = nn.ModuleList([HybridBlock(config, layer_idx=index, dtype=dtype) for index in range(config.n_layer)])
        self.norm_f = HybridRMSNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = hidden_states + residual if residual is not None else hidden_states
        return self.norm_f(residual)


class MetisMamba2HybridLMHeadModel(nn.Module):
    def __init__(self, config: MetisHybridConfig, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.config = config
        self.model_family = config.model_type
        self.backbone = HybridBackbone(config, dtype=dtype)
        self.lm_head = nn.Linear(config.d_model, config.padded_vocab_size, bias=False, dtype=dtype)
        if config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> HybridCausalLMOutput:
        hidden_states = self.backbone(input_ids)
        return HybridCausalLMOutput(logits=self.lm_head(hidden_states))


def _load_config(model_dir: Path) -> MetisHybridConfig:
    return MetisHybridConfig.from_dict(json.loads((model_dir / "config.json").read_text()))


def load_hybrid_exported_model(model_dir: str | Path, device: torch.device) -> MetisMamba2HybridLMHeadModel:
    model_dir = Path(model_dir)
    config = _load_config(model_dir)
    model = MetisMamba2HybridLMHeadModel(config, dtype=torch.float32)
    state_dict = load_safetensors(str(model_dir / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"} if config.tie_embeddings else set()
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Unexpected Metis-1.3 load result: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model

