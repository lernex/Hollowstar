from __future__ import annotations

import argparse
from pathlib import Path

from metis_runtime import choose_device, generate_completion, load_model, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/default/best.pt")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(Path(args.checkpoint), device)
    print(
        generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
