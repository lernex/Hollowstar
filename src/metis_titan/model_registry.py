from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
    compute_ffn_hidden_dim,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec


METIS12_DIM = 768
METIS12_HEADS = 12
METIS12_KV_HEADS = 12
METIS12_LAYERS = 18
METIS12_VOCAB = 8192
METIS12_MULTIPLE_OF = 256
METIS12_FFN_MULTIPLIER = 0.6875
METIS12_MAX_SEQ = 1024
METIS12_ROPE_THETA = 500000

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    n_kv_heads: int | None = None,
    attn_backend: str = "sdpa",
) -> list[TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers: list[TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                ),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    fuse_qkv=False,
                    mask_type=mask_type,
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _metis12(attn_backend: str = "sdpa") -> Llama3Model.Config:
    hidden_dim = compute_ffn_hidden_dim(
        METIS12_DIM,
        multiple_of=METIS12_MULTIPLE_OF,
        ffn_dim_multiplier=METIS12_FFN_MULTIPLIER,
    )
    return Llama3Model.Config(
        dim=METIS12_DIM,
        vocab_size=METIS12_VOCAB,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=METIS12_VOCAB,
            embedding_dim=METIS12_DIM,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=METIS12_DIM, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=METIS12_DIM,
            out_features=METIS12_VOCAB,
            param_init=_output_linear_init(METIS12_DIM),
        ),
        rope=RoPE.Config(
            dim=METIS12_DIM // METIS12_HEADS,
            max_seq_len=METIS12_MAX_SEQ,
            theta=METIS12_ROPE_THETA,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_layers(
            n_layers=METIS12_LAYERS,
            dim=METIS12_DIM,
            n_heads=METIS12_HEADS,
            n_kv_heads=METIS12_KV_HEADS,
            hidden_dim=hidden_dim,
            attn_backend=attn_backend,
        ),
    )


_FLAVORS = {
    "metis12": _metis12,
}


def model_registry(flavor: str, attn_backend: str = "sdpa") -> ModelSpec:
    if flavor not in _FLAVORS:
        raise KeyError(f"Unknown Metis TorchTitan flavor: {flavor}")

    return ModelSpec(
        name="metis_titan",
        flavor=flavor,
        model=_FLAVORS[flavor](attn_backend=attn_backend),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )

