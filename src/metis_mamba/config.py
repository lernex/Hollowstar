from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_ATTN_LAYER_IDX = [3, 7, 11, 15, 19, 23, 27]


@dataclass
class MetisMambaConfig:
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
    def from_dict(cls, payload: dict[str, Any]) -> "MetisMambaConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        cooked = {key: value for key, value in payload.items() if key in allowed}
        return cls(**cooked)

    def validate(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.d_model <= 0 or self.n_layer <= 0:
            raise ValueError("d_model and n_layer must be positive.")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive.")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model.")
        if self.ssm_layer not in {"Mamba1", "Mamba2"}:
            raise ValueError("ssm_layer must be Mamba1 or Mamba2.")
        if self.ssm_expand <= 0 or self.ssm_d_state <= 0 or self.ssm_d_conv <= 0:
            raise ValueError("Mamba SSM settings must be positive.")
        if any(index < 0 or index >= self.n_layer for index in self.attn_layer_idx):
            raise ValueError("attn_layer_idx contains an out-of-range layer index.")

    @property
    def padded_vocab_size(self) -> int:
        if self.vocab_size % self.pad_vocab_size_multiple == 0:
            return self.vocab_size
        return self.vocab_size + (
            self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)
        )

    @property
    def attention_layer_count(self) -> int:
        return len(self.attn_layer_idx)

    @property
    def mamba_layer_count(self) -> int:
        return self.n_layer - self.attention_layer_count

    @property
    def ssm_cfg(self) -> dict[str, Any]:
        return {
            "layer": self.ssm_layer,
            "d_state": self.ssm_d_state,
            "d_conv": self.ssm_d_conv,
            "expand": self.ssm_expand,
        }

    @property
    def attn_cfg(self) -> dict[str, Any]:
        return {
            "causal": True,
            "d_conv": self.attn_d_conv,
            "head_dim": self.head_dim,
            "num_heads": self.n_heads,
            "num_heads_kv": self.n_kv_heads,
            "out_proj_bias": False,
            "qkv_proj_bias": False,
            "rotary_emb_dim": self.attn_rotary_emb_dim,
        }

    def estimate_params(self) -> int:
        embed_params = self.padded_vocab_size * self.d_model
        mamba_block = 3 * self.ssm_expand * self.d_model * self.d_model
        qkv_dim = self.head_dim * (self.n_heads + 2 * self.n_kv_heads)
        attn_out_dim = self.head_dim * self.n_heads
        attention_block = self.d_model * qkv_dim + attn_out_dim * self.d_model
        norm_params = 2 * self.d_model * self.n_layer + self.d_model
        total = (
            embed_params
            + (self.mamba_layer_count * mamba_block)
            + (self.attention_layer_count * attention_block)
            + norm_params
        )
        return int(total)

    def to_mamba_config_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "d_model": self.d_model,
            "d_intermediate": 0,
            "n_layer": self.n_layer,
            "vocab_size": self.vocab_size,
            "ssm_cfg": self.ssm_cfg,
            "attn_layer_idx": list(self.attn_layer_idx),
            "attn_cfg": self.attn_cfg,
            "rms_norm": self.rms_norm,
            "residual_in_fp32": self.residual_in_fp32,
            "fused_add_norm": self.fused_add_norm,
            "pad_vocab_size_multiple": self.pad_vocab_size_multiple,
            "tie_embeddings": self.tie_embeddings,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimated_params"] = self.estimate_params()
        payload["ssm_cfg"] = self.ssm_cfg
        payload["attn_cfg"] = self.attn_cfg
        return payload
