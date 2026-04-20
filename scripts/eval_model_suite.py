from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from metis_runtime import (
    choose_device as runtime_choose_device,
    extract_assistant_reply,
    generate_completion,
    load_model,
    load_tokenizer,
    looks_degenerate,
)


@dataclass
class EvalResult:
    label: str
    prompt_id: str
    group: str
    prompt: str
    response: str
    degenerate: bool


class LegacyRunner:
    def __init__(self, path: Path, device: torch.device):
        tokenizer_path = path.parent.parent / "artifacts" / "tokenizer" / "tokenizer.json"
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.model = load_model(path, device)
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        text = generate_completion(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            device=self.device,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
        )
        if text.startswith(prompt):
            text = text[len(prompt) :]
        if "Assistant:" in prompt:
            text = extract_assistant_reply(f"{prompt}{text}")
        return text.strip()


class HfRunner:
    def __init__(self, path: Path, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model_kwargs = {"torch_dtype": torch_dtype}
        if device.type == "cuda":
            model_kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device.type != "cuda":
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            self.model.to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if "Assistant:" in prompt:
            text = extract_assistant_reply(f"{prompt}{text}")
        return text.strip()


def choose_device(requested: str | None) -> torch.device:
    return runtime_choose_device(requested)


def load_runner(model_path: Path, device: torch.device):
    if model_path.is_dir() and (model_path / "config.json").exists():
        config = json.loads((model_path / "config.json").read_text())
        if config.get("model_type") == "metis_mamba2_hybrid":
            return CustomRunner(model_path, device)
        return HfRunner(model_path, device)
    if model_path.is_file() and model_path.suffix == ".pt":
        checkpoint = torch.load(model_path, map_location="cpu")
        if checkpoint.get("model_family") == "metis_mamba2_hybrid":
            return CustomRunner(model_path, device)
        return LegacyRunner(model_path, device)
    raise ValueError(f"Unsupported model path for evaluation: {model_path}")


class CustomRunner:
    def __init__(self, path: Path, device: torch.device):
        if path.is_dir():
            tokenizer_path = path / "tokenizer.json"
        else:
            tokenizer_path = path.parent.parent / "artifacts" / "metis13_hf_assets" / "tokenizer.json"
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.model = load_model(path, device)
        self.device = device

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        text = generate_completion(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            device=self.device,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
        )
        if text.startswith(prompt):
            text = text[len(prompt) :]
        if "Assistant:" in prompt:
            text = extract_assistant_reply(f"{prompt}{text}")
        return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Metis prompt suite against one or more checkpoints.")
    parser.add_argument("--suite", default="configs/metis12_eval_prompts.json")
    parser.add_argument("--model", action="append", required=True, help="label=/path/to/model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    suite = json.loads(Path(args.suite).read_text())
    device = choose_device(args.device)

    results: list[EvalResult] = []
    for model_arg in args.model:
        if "=" not in model_arg:
            raise ValueError("--model entries must look like label=/absolute/or/relative/path")
        label, raw_path = model_arg.split("=", 1)
        runner = load_runner(Path(raw_path), device)
        for prompt in suite["prompts"]:
            response = runner.generate(prompt["prompt"], int(prompt.get("max_new_tokens", 120)))
            results.append(
                EvalResult(
                    label=label,
                    prompt_id=prompt["id"],
                    group=prompt["group"],
                    prompt=prompt["prompt"],
                    response=response,
                    degenerate=looks_degenerate(response),
                )
            )

    summary = {
        "models": sorted({result.label for result in results}),
        "degenerate_responses": sum(1 for result in results if result.degenerate),
        "results": [result.__dict__ for result in results],
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
