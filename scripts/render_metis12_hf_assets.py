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
    parser = argparse.ArgumentParser(description="Render local HF assets for Metis-1.2.")
    parser.add_argument("--manifest", default="configs/metis12_manifest.json")
    parser.add_argument("--tokenizer-dir", default="artifacts/metis12_hf_assets")
    parser.add_argument("--output-dir", default="artifacts/metis12_hf_assets")
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

    config = {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "bos_token_id": special_ids.get(bos_token, tokenizer.token_to_id(bos_token)),
        "eos_token_id": special_ids.get(eos_token, tokenizer.token_to_id(eos_token)),
        "hidden_act": "silu",
        "hidden_size": manifest["model"]["n_embd"],
        "initializer_range": 0.02,
        "intermediate_size": manifest["model"]["ffn_hidden_size"],
        "max_position_embeddings": manifest["model"]["block_size"],
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": manifest["model"]["n_head"],
        "num_hidden_layers": manifest["model"]["n_layer"],
        "num_key_value_heads": manifest["model"]["n_kv_head"],
        "pad_token_id": special_ids.get(pad_token, tokenizer.token_to_id(pad_token)),
        "pretraining_tp": 1,
        "rms_norm_eps": manifest["model"]["rms_norm_eps"],
        "rope_theta": manifest["model"]["rope_theta"],
        "tie_word_embeddings": manifest["model"]["tie_word_embeddings"],
        "torch_dtype": "bfloat16",
        "transformers_version": "4.57.0",
        "use_cache": True,
        "vocab_size": manifest["model"]["vocab_size"]
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
        "model_max_length": manifest["model"]["block_size"],
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
    (output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2) + "\n"
    )
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n"
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2) + "\n"
    )

    summary = {
        "name": manifest["name"],
        "output_dir": str(output_dir),
        "tokenizer_path": str(output_dir / "tokenizer.json"),
        "config_path": str(output_dir / "config.json"),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

