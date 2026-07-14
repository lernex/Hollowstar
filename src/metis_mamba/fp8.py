from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache
from inspect import signature

from torch import nn


@lru_cache(maxsize=1)
def _load_transformer_engine():
    import transformer_engine.pytorch as te  # type: ignore
    import transformer_engine.common.recipe as recipe_mod  # type: ignore

    return te, recipe_mod


def transformer_engine_is_available() -> bool:
    try:
        _load_transformer_engine()
    except Exception:
        return False
    return True


def transformer_engine_supports_nvfp4() -> bool:
    try:
        _te, recipe_mod = _load_transformer_engine()
        return hasattr(recipe_mod, "NVFP4BlockScaling")
    except Exception:
        return False


def transformer_engine_supports_mxfp8() -> bool:
    try:
        _te, recipe_mod = _load_transformer_engine()
        return hasattr(recipe_mod, "MXFP8BlockScaling")
    except Exception:
        return False


def transformer_engine_supports_fp8_block_scaling() -> bool:
    try:
        _te, recipe_mod = _load_transformer_engine()
        return hasattr(recipe_mod, "Float8BlockScaling")
    except Exception:
        return False


def _current_cuda_capability() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return tuple(int(part) for part in torch.cuda.get_device_capability())  # type: ignore[return-value]
    except Exception:
        return None


def transformer_engine_runtime_supports_nvfp4(
    *,
    disable_rht: bool = False,
    disable_2d_quantization: bool = False,
    disable_stochastic_rounding: bool = False,
) -> bool:
    if not transformer_engine_supports_nvfp4():
        return False
    capability = _current_cuda_capability()
    # TE exposes NVFP4 on RTX PRO 6000 / SM120, but the default production
    # recipe uses SM100-oriented RHT/2D/SR kernels that fail exact Metis GEMMs.
    # The reduced recipe has been smoke-tested on the Metis shapes.
    if capability is not None and capability >= (12, 0):
        return bool(disable_rht and disable_2d_quantization and disable_stochastic_rounding)
    return True


def transformer_engine_runtime_supports_mxfp8() -> bool:
    if not transformer_engine_supports_mxfp8():
        return False
    capability = _current_cuda_capability()
    # TE 2.15 raises "MXFP8 ... is not supported on 12.0+ architectures yet"
    # before dispatching GEMMs on RTX PRO 6000 / SM120.
    if capability is not None and capability >= (12, 0):
        return False
    return True


def transformer_engine_runtime_supports_fp8_block_scaling() -> bool:
    if not transformer_engine_supports_fp8_block_scaling():
        return False
    try:
        import transformer_engine.pytorch.quantization as quantization  # type: ignore

        check_fn = getattr(quantization, "check_fp8_block_scaling_support", None)
        if check_fn is not None:
            supported, _reason = check_fn()
            return bool(supported)
    except Exception:
        pass
    capability = _current_cuda_capability()
    if capability is None or capability < (9, 0):
        return False
    try:
        import torch

        cuda_version = float(torch.version.cuda or 0)
    except Exception:
        return False
    return cuda_version >= 12.9


def build_fp8_recipe(
    *,
    format_name: str = "HYBRID",
    margin: int = 0,
    amax_history_len: int = 16,
    amax_compute_algo: str = "max",
    fp8_dpa: bool = False,
    fp8_mha: bool = False,
):
    _te, recipe_mod = _load_transformer_engine()
    DelayedScaling = recipe_mod.DelayedScaling
    Format = recipe_mod.Format
    fp8_format = getattr(Format, format_name)
    kwargs = {
        "margin": margin,
        "fp8_format": fp8_format,
        "amax_history_len": amax_history_len,
        "amax_compute_algo": amax_compute_algo,
    }
    try:
        recipe_params = signature(DelayedScaling).parameters
    except (TypeError, ValueError):
        recipe_params = {}
    if fp8_dpa and "fp8_dpa" in recipe_params:
        kwargs["fp8_dpa"] = True
    if fp8_mha and "fp8_mha" in recipe_params:
        kwargs["fp8_mha"] = True
    return DelayedScaling(**kwargs)


def build_nvfp4_recipe(
    *,
    disable_rht: bool = False,
    disable_2d_quantization: bool = False,
    disable_stochastic_rounding: bool = False,
):
    _te, recipe_mod = _load_transformer_engine()
    recipe_cls = getattr(recipe_mod, "NVFP4BlockScaling", None)
    if recipe_cls is None:
        raise RuntimeError(
            "Transformer Engine does not expose NVFP4BlockScaling. "
            "Install a Blackwell-capable Transformer Engine build (2.14+ recommended)."
        )
    kwargs = {
        "disable_rht": disable_rht,
        "disable_2d_quantization": disable_2d_quantization,
        "disable_stochastic_rounding": disable_stochastic_rounding,
    }
    try:
        recipe_params = signature(recipe_cls).parameters
    except (TypeError, ValueError):
        recipe_params = {}
    if recipe_params:
        kwargs = {key: value for key, value in kwargs.items() if key in recipe_params}
    return recipe_cls(**kwargs)


def build_mxfp8_recipe(*, format_name: str = "E4M3"):
    _te, recipe_mod = _load_transformer_engine()
    recipe_cls = getattr(recipe_mod, "MXFP8BlockScaling", None)
    if recipe_cls is None:
        raise RuntimeError(
            "Transformer Engine does not expose MXFP8BlockScaling. "
            "Install a Blackwell-capable Transformer Engine build (2.14+ recommended)."
        )
    kwargs = {}
    try:
        recipe_params = signature(recipe_cls).parameters
    except (TypeError, ValueError):
        recipe_params = {}
    if "fp8_format" in recipe_params:
        kwargs["fp8_format"] = getattr(recipe_mod.Format, format_name)
    return recipe_cls(**kwargs)


def build_fp8_block_recipe():
    _te, recipe_mod = _load_transformer_engine()
    recipe_cls = getattr(recipe_mod, "Float8BlockScaling", None)
    if recipe_cls is None:
        raise RuntimeError(
            "Transformer Engine does not expose Float8BlockScaling. "
            "Use a CUDA 12.9+/TE build for block-scaled FP8 fallback surfaces."
        )
    return recipe_cls()


def fp8_autocast_context(
    *,
    enabled: bool,
    recipe=None,
    fp8_group=None,
):
    if not enabled:
        return nullcontext()
    te, _recipe_mod = _load_transformer_engine()
    if hasattr(te, "autocast"):
        kwargs = {"enabled": True, "recipe": recipe}
        try:
            params = signature(te.autocast).parameters
            if "fp8_group" in params:
                kwargs["fp8_group"] = fp8_group
        except (TypeError, ValueError):
            pass
        return te.autocast(**kwargs)
    return te.fp8_autocast(enabled=True, fp8_recipe=recipe, fp8_group=fp8_group)


def fp8_disabled_context():
    try:
        te, _recipe_mod = _load_transformer_engine()
    except Exception:
        return nullcontext()
    if hasattr(te, "autocast"):
        return te.autocast(enabled=False)
    return te.fp8_autocast(enabled=False)


def build_linear(
    *,
    in_features: int,
    out_features: int,
    bias: bool,
    use_fp8: bool,
    low_precision_allowed: bool = True,
):
    if use_fp8 and low_precision_allowed and supports_fp8_linear(in_features, out_features):
        te, _recipe_mod = _load_transformer_engine()
        return te.Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def build_grouped_linear(
    *,
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias: bool,
    use_fp8: bool,
    low_precision_allowed: bool = True,
    init_method=None,
):
    if (
        low_precision_allowed
        and transformer_engine_is_available()
        and (not use_fp8 or supports_fp8_linear(in_features, out_features))
    ):
        te, _recipe_mod = _load_transformer_engine()
        grouped_linear = getattr(te, "GroupedLinear", None)
        if grouped_linear is not None:
            kwargs = {
                "num_gemms": num_gemms,
                "bias": bias,
                "device": "cpu",
            }
            if init_method is not None:
                kwargs["init_method"] = init_method
            try:
                return grouped_linear(in_features, out_features, **kwargs)
            except TypeError:
                # Older TE nightlies briefly accepted num_gemms positionally.
                kwargs.pop("num_gemms", None)
                return grouped_linear(num_gemms, in_features, out_features, **kwargs)
    return None


def build_rmsnorm(
    *,
    hidden_size: int,
    eps: float,
    use_fp8: bool,
):
    if use_fp8:
        te, _recipe_mod = _load_transformer_engine()
        return te.RMSNorm(hidden_size, eps=eps)
    return None


def build_layernorm_mlp(
    *,
    hidden_size: int,
    ffn_hidden_size: int,
    eps: float,
    bias: bool,
    use_fp8: bool,
):
    if use_fp8:
        te, _recipe_mod = _load_transformer_engine()
        return te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=eps,
            bias=bias,
            normalization="RMSNorm",
            activation="swiglu",
            device="cpu",
        )
    return None


def build_swiglu_activation(*, use_fp8: bool):
    if not use_fp8:
        return None
    try:
        te, _recipe_mod = _load_transformer_engine()
        ops = getattr(te, "ops", None)
        if ops is not None and hasattr(ops, "SwiGLU"):
            return ops.SwiGLU()
    except Exception:
        return None
    return None


def build_dot_product_attention(
    *,
    num_attention_heads: int,
    num_gqa_groups: int,
    head_dim: int,
    attention_dropout: float,
    softmax_scale: float,
    use_fp8: bool,
):
    if not use_fp8:
        return None
    te, _recipe_mod = _load_transformer_engine()
    dot_product_attention = getattr(te, "DotProductAttention", None)
    if dot_product_attention is None:
        return None

    kwargs = {
        "num_gqa_groups": num_gqa_groups,
        "attention_dropout": attention_dropout,
        "attn_mask_type": "causal",
        "qkv_format": "bshd",
        "softmax_scale": softmax_scale,
    }
    try:
        params = signature(dot_product_attention).parameters
    except (TypeError, ValueError):
        params = {}
    has_var_kwargs = any(param.kind == param.VAR_KEYWORD for param in params.values())
    if params and not has_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in params}
        if num_gqa_groups != num_attention_heads and "num_gqa_groups" not in kwargs:
            return None

    try:
        return dot_product_attention(num_attention_heads, head_dim, **kwargs)
    except TypeError:
        return None


def is_transformer_engine_module(module: nn.Module) -> bool:
    return module.__class__.__module__.startswith("transformer_engine.")


def supports_fp8_linear(in_features: int, out_features: int) -> bool:
    return (in_features % 16 == 0) and (out_features % 16 == 0)
