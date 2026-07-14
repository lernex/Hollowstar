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
    parser = argparse.ArgumentParser(description="Render local HF-style assets for the current Metis line.")
    parser.add_argument("--manifest", default="configs/metis15_manifest.json")
    parser.add_argument("--tokenizer-dir", default="artifacts/metis15_hf_assets")
    parser.add_argument("--output-dir", default="artifacts/metis15_hf_assets")
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
    tokenizer_vocab_size = tokenizer.get_vocab_size()

    manifest_tokenizer_vocab_size = int(manifest.get("tokenizer", {}).get("vocab_size", 0))
    manifest_model_vocab_size = int(manifest["model"]["vocab_size"])
    if manifest_tokenizer_vocab_size and manifest_tokenizer_vocab_size != manifest_model_vocab_size:
        raise ValueError(
            "Manifest tokenizer.vocab_size does not match model.vocab_size: "
            f"{manifest_tokenizer_vocab_size} != {manifest_model_vocab_size}"
        )
    if tokenizer_vocab_size != manifest_model_vocab_size:
        raise ValueError(
            "Trained tokenizer vocab size does not match the Metis manifest: "
            f"{tokenizer_vocab_size} != {manifest_model_vocab_size}. "
            "Retrain the tokenizer with the manifest vocab size before rendering/uploading assets."
        )

    bos_token = "<bos>"
    eos_token = "<eos>"
    pad_token = "<pad>"
    unk_token = "<unk>"

    model = manifest["model"]
    architecture_name = "MetisMoRLMHeadModel"
    config = {
        "architectures": [architecture_name],
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
        "intermediate_size": model["intermediate_size"],
        "hidden_act": model["hidden_act"],
        "attn_cfg": model.get("attn_cfg", {}),
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
    for key in [
        "attention_bias",
        "mlp_bias",
        "attention_dropout",
        "rope_theta",
        "attention_backend",
        "fp8_pad_multiple",
        "mor_max_depth",
        "mor_router_hidden_dim",
        "mor_router_temperature",
        "mor_router_aux_loss_coef",
        "mor_target_avg_depth",
        "effective_layer_count",
        "target_effective_layer_count",
    ]:
        if key in model:
            config[key] = model[key]
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
        "vocab_size": tokenizer_vocab_size,
        "output_dir": str(output_dir),
        "tokenizer_path": str(output_dir / "tokenizer.json"),
        "config_path": str(output_dir / "config.json")
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
