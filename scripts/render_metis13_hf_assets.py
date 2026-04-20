from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer


CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}\n{% elif message['role'] == 'user' %}User: {{ message['content'] }}\n{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}\n{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"""


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def load_tokenizer_meta(tokenizer_dir: Path) -> dict:
    meta_path = tokenizer_dir / "tokenizer_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render local HF-style assets for Metis-1.3.")
    parser.add_argument("--manifest", default="configs/metis13_manifest.json")
    parser.add_argument("--tokenizer-dir", default="artifacts/metis13_hf_assets")
    parser.add_argument("--output-dir", default="artifacts/metis13_hf_assets")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if tokenizer_path.resolve() != (output_dir / "tokenizer.json").resolve():
        shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    meta = load_tokenizer_meta(tokenizer_dir)
    special_ids = meta.get("special_tokens", {})

    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    unk_token = "<unk>"

    model = manifest["model"]
    config = {
        "architectures": ["MetisMambaLMHeadModel"],
        "model_type": model["model_type"],
        "name": manifest["name"],
        "architecture": model["architecture"],
        "vocab_size": model["vocab_size"],
        "block_size": model["block_size"],
        "d_model": model["d_model"],
        "n_layer": model["n_layer"],
        "n_heads": model["n_heads"],
        "n_kv_heads": model["n_kv_heads"],
        "head_dim": model["head_dim"],
        "attn_layer_idx": model["attn_layer_idx"],
        "attn_d_conv": model["attn_d_conv"],
        "attn_rotary_emb_dim": model["attn_rotary_emb_dim"],
        "ssm_layer": model["ssm_layer"],
        "ssm_d_state": model["ssm_d_state"],
        "ssm_d_conv": model["ssm_d_conv"],
        "ssm_expand": model["ssm_expand"],
        "ssm_cfg": {
            "layer": model["ssm_layer"],
            "d_state": model["ssm_d_state"],
            "d_conv": model["ssm_d_conv"],
            "expand": model["ssm_expand"]
        },
        "attn_cfg": {
            "causal": True,
            "d_conv": model["attn_d_conv"],
            "head_dim": model["head_dim"],
            "num_heads": model["n_heads"],
            "num_heads_kv": model["n_kv_heads"],
            "qkv_proj_bias": False,
            "out_proj_bias": False,
            "rotary_emb_dim": model["attn_rotary_emb_dim"]
        },
        "bos_token_id": special_ids.get(bos_token, tokenizer.token_to_id(bos_token)),
        "eos_token_id": special_ids.get(eos_token, tokenizer.token_to_id(eos_token)),
        "pad_token_id": special_ids.get(pad_token, tokenizer.token_to_id(pad_token)),
        "unk_token_id": special_ids.get(unk_token, tokenizer.token_to_id(unk_token)),
        "rms_norm": model["rms_norm"],
        "residual_in_fp32": model["residual_in_fp32"],
        "fused_add_norm": model["fused_add_norm"],
        "pad_vocab_size_multiple": model["pad_vocab_size_multiple"],
        "tie_embeddings": model["tie_embeddings"],
        "torch_dtype": model["torch_dtype"],
        "estimated_params": model["estimated_params"]
    }
    generation_config = {
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "pad_token_id": config["pad_token_id"],
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 256
    }
    tokenizer_config = {
        "add_bos_token": True,
        "add_eos_token": True,
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token,
        "clean_up_tokenization_spaces": False,
        "model_max_length": model["block_size"],
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": CHAT_TEMPLATE
    }
    special_tokens_map = {
        "bos_token": bos_token,
        "eos_token": eos_token,
        "unk_token": unk_token,
        "pad_token": pad_token
    }

    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2) + "\n")
    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    (output_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2) + "\n")

    summary = {
        "name": manifest["name"],
        "output_dir": str(output_dir),
        "tokenizer_path": str(output_dir / "tokenizer.json"),
        "config_path": str(output_dir / "config.json")
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
