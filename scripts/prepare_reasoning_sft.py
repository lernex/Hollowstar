from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm


THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>",
    re.DOTALL,
)
SOLUTION_RE = re.compile(
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def extract_reasoning(answer: str) -> tuple[str, str]:
    thought_match = THOUGHT_RE.search(answer)
    solution_match = SOLUTION_RE.search(answer)
    if thought_match or solution_match:
        thought = thought_match.group(1).strip() if thought_match else ""
        solution = solution_match.group(1).strip() if solution_match else answer.strip()
        return collapse_whitespace(thought), collapse_whitespace(solution)

    think_match = THINK_RE.search(answer)
    if think_match:
        thought = think_match.group(1).strip()
        solution = THINK_RE.sub("", answer).strip()
        return collapse_whitespace(thought), collapse_whitespace(solution)

    thought = ""
    solution = answer.strip()
    return collapse_whitespace(thought), collapse_whitespace(solution)


def format_example(
    system_prompt: str,
    user_text: str,
    thought: str,
    solution: str,
    assistant_format: str,
) -> list[dict[str, str]]:
    if assistant_format == "think_tags" and thought:
        assistant = f"<think>\n{thought}\n</think>\n\n{solution}".strip()
    else:
        assistant = f"Thought:\n{thought}\n\nSolution:\n{solution}".strip()

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant},
        ]
    )
    return messages


def encode_messages(tokenizer: Tokenizer, messages: list[dict[str, str]], max_length: int) -> dict | None:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    input_ids: list[int] = []
    labels: list[int] = []
    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    saw_assistant = False
    last_role = None
    for message in messages:
        role = message["role"].strip().lower()
        prefix = role.capitalize()
        text = f"{prefix}: {message['content'].strip()}\n"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        input_ids.extend(ids)
        if role == "assistant":
            labels.extend(ids)
            saw_assistant = True
        else:
            labels.extend([-100] * len(ids))
        last_role = role

    if eos_id is not None:
        input_ids.append(eos_id)
        labels.append(eos_id if last_role == "assistant" else -100)

    if len(input_ids) > max_length:
        return None
    if not saw_assistant:
        return None
    return {"input_ids": input_ids, "labels": labels}


def split_name(index: int, val_ratio: float) -> str:
    return "val" if index % max(int(round(1 / max(val_ratio, 1e-6))), 2) == 0 else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare short-form reasoning SFT data from OpenThoughts.")
    parser.add_argument("--dataset-name", default="open-thoughts/OpenThoughts-114k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-path", default="artifacts/tokenizer/tokenizer.json")
    parser.add_argument("--output-dir", default="data/metis_reasoning")
    parser.add_argument("--max-examples", type=int, default=12000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-user-chars", type=int, default=500)
    parser.add_argument("--max-thought-chars", type=int, default=700)
    parser.add_argument("--max-solution-chars", type=int, default=300)
    parser.add_argument("--assistant-format", choices=["sections", "think_tags"], default="sections")
    parser.add_argument(
        "--system-prompt",
        default="You are Metis, a tiny reasoning assistant. Think briefly but clearly, then give the final answer.",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=True)

    train_examples: list[dict] = []
    val_examples: list[dict] = []
    skipped = 0
    consumed = 0

    progress = tqdm(total=args.max_examples, desc="Collecting reasoning examples")
    for row in dataset:
        if len(train_examples) + len(val_examples) >= args.max_examples:
            break
        consumed += 1

        conversations = row.get("conversations") or []
        if len(conversations) < 2:
            skipped += 1
            continue

        user_turn = conversations[0].get("value", "").strip()
        assistant_turn = conversations[1].get("value", "").strip()
        if not user_turn or not assistant_turn:
            skipped += 1
            continue

        user_turn = user_turn[: args.max_user_chars].strip()
        thought, solution = extract_reasoning(assistant_turn)
        if not solution:
            skipped += 1
            continue

        thought = thought[: args.max_thought_chars].strip()
        solution = solution[: args.max_solution_chars].strip()
        if not thought:
            thought = "I break the problem into smaller steps and check the important constraints."

        messages = format_example(
            args.system_prompt,
            user_turn,
            thought,
            solution,
            args.assistant_format,
        )
        encoded = encode_messages(tokenizer, messages, args.max_length)
        if encoded is None:
            skipped += 1
            continue

        target = val_examples if split_name(consumed, args.val_ratio) == "val" else train_examples
        target.append(encoded)
        progress.update(1)
        accepted = len(train_examples) + len(val_examples)
        if accepted == 1 or accepted % 2000 == 0:
            print(
                f"Collected reasoning examples: {accepted}/{args.max_examples} | "
                f"train={len(train_examples)} val={len(val_examples)} skipped={skipped}",
                flush=True,
            )

    progress.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_examples, output_dir / "train.pt")
    torch.save(val_examples, output_dir / "val.pt")

    meta = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "tokenizer_path": args.tokenizer_path,
        "max_examples": args.max_examples,
        "val_ratio": args.val_ratio,
        "max_length": args.max_length,
        "max_user_chars": args.max_user_chars,
        "max_thought_chars": args.max_thought_chars,
        "max_solution_chars": args.max_solution_chars,
        "assistant_format": args.assistant_format,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "skipped": skipped,
        "system_prompt": args.system_prompt,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
