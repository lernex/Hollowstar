from .config import MetisMambaConfig

try:
    from .model import MetisMoRLMHeadModel
    from .runtime import (
        build_reward_model,
        build_model,
        cosine_lr,
        encode_prompt,
        export_checkpoint_to_dir,
        generate_completion,
        load_checkpoint_model,
        load_exported_model,
        parse_torch_dtype,
    )
except ModuleNotFoundError as exc:  # Allows JAX-only TPU environments without torch installed.
    if exc.name != "torch":
        raise
    MetisMoRLMHeadModel = None
    build_reward_model = None
    build_model = None
    cosine_lr = None
    encode_prompt = None
    export_checkpoint_to_dir = None
    generate_completion = None
    load_checkpoint_model = None
    load_exported_model = None
    parse_torch_dtype = None

__all__ = [
    "MetisMambaConfig",
    "MetisMoRLMHeadModel",
    "build_reward_model",
    "build_model",
    "cosine_lr",
    "encode_prompt",
    "export_checkpoint_to_dir",
    "generate_completion",
    "load_checkpoint_model",
    "load_exported_model",
    "parse_torch_dtype",
]
