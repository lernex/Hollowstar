from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable, Iterator

from datasets import load_dataset


THOUGHT_RE = re.compile(
    r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>",
    re.DOTALL,
)
SOLUTION_RE = re.compile(
    r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
CHATML_RE = re.compile(r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>", re.DOTALL)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_name: str
    split: str
    weight: float
    format: str
    dataset_config: str | None = None
    streaming: bool = True
    keep_think: bool = False
    use_custom_instructions: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=raw["name"],
            dataset_name=raw["dataset_name"],
            split=raw["split"],
            weight=float(raw["weight"]),
            format=raw["format"],
            dataset_config=raw.get("dataset_config"),
            streaming=bool(raw.get("streaming", True)),
            keep_think=bool(raw.get("keep_think", False)),
            use_custom_instructions=bool(raw.get("use_custom_instructions", True)),
        )


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def split_name(payload: str, val_ratio: float) -> str:
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def build_schedule(weights: list[float], total_examples: int, seed: int) -> list[int]:
    weight_sum = sum(weights)
    normalized = [weight / weight_sum for weight in weights]
    raw_counts = [weight * total_examples for weight in normalized]
    counts = [int(count) for count in raw_counts]
    remainder = total_examples - sum(counts)
    ranked = sorted(
        range(len(raw_counts)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1

    schedule: list[int] = []
    for index, count in enumerate(counts):
        schedule.extend([index] * count)
    rng = Random(seed)
    rng.shuffle(schedule)
    return schedule


def normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


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

    return "", collapse_whitespace(answer)


def shorten_reasoning(
    assistant: str,
    *,
    keep_think: bool,
    max_think_chars: int,
    max_answer_chars: int,
    max_assistant_chars: int,
) -> str:
    thought, answer = extract_reasoning(assistant)
    if not keep_think:
        return collapse_whitespace(answer[:max_assistant_chars])

    thought = collapse_whitespace(thought[:max_think_chars])
    answer = collapse_whitespace(answer[:max_answer_chars])
    if thought:
        return f"<think>\n{thought}\n</think>\n\n{answer}".strip()
    return answer


def build_user_prompt(
    history: list[dict[str, str]],
    *,
    custom_instructions: str,
    max_history_turns: int,
    max_user_chars: int,
) -> str:
    if not history or history[-1]["role"] != "user":
        return ""

    current_user = collapse_whitespace(history[-1]["content"])
    prior = history[:-1]
    if not prior and not custom_instructions.strip():
        return current_user[:max_user_chars]

    context_lines: list[str] = []
    if custom_instructions.strip():
        context_lines.append(f"Follow these instructions while replying: {collapse_whitespace(custom_instructions)}")

    if prior:
        retained = prior[-(2 * max_history_turns) :]
        context_lines.append("Conversation so far:")
        for message in retained:
            content = collapse_whitespace(message["content"])
            if message["role"] == "assistant":
                _, content = extract_reasoning(content)
            context_lines.append(
                f"{message['role'].capitalize()}: {content}"
            )

    context_lines.append(f"User: {current_user}")
    return "\n".join(context_lines)[:max_user_chars].strip()


def iter_pairs_from_messages(
    messages: list[dict[str, str]],
    *,
    custom_instructions: str,
    source_name: str,
    keep_think: bool,
    max_history_turns: int,
    max_user_chars: int,
    max_assistant_chars: int,
    max_think_chars: int,
    max_answer_chars: int,
) -> Iterator[dict[str, Any]]:
    normalized: list[dict[str, str]] = []
    for raw_message in messages:
        role = normalize_role(raw_message["role"])
        content = raw_message["content"].strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})

    history: list[dict[str, str]] = []
    for message in normalized:
        history.append(message)
        if message["role"] != "assistant":
            continue
        if len(history) < 2 or history[-2]["role"] != "user":
            continue

        user_prompt = build_user_prompt(
            history[:-1],
            custom_instructions=custom_instructions,
            max_history_turns=max_history_turns,
            max_user_chars=max_user_chars,
        )
        assistant = shorten_reasoning(
            message["content"],
            keep_think=keep_think,
            max_think_chars=max_think_chars,
            max_answer_chars=max_answer_chars,
            max_assistant_chars=max_assistant_chars,
        )
        if not user_prompt or not assistant:
            continue

        yield {
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant},
            ],
            "source": source_name,
        }


def parse_chatml_text(raw_text: str) -> tuple[str, list[dict[str, str]]]:
    custom_instructions = ""
    messages: list[dict[str, str]] = []
    for role, content in CHATML_RE.findall(raw_text):
        normalized_role = normalize_role(role)
        content = content.strip()
        if normalized_role == "system":
            custom_instructions = content
            continue
        if normalized_role in {"user", "assistant"}:
            messages.append({"role": normalized_role, "content": content})
    return custom_instructions, messages


def iter_source_examples(
    spec: SourceSpec,
    *,
    max_history_turns: int,
    max_user_chars: int,
    max_assistant_chars: int,
    max_think_chars: int,
    max_answer_chars: int,
) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(
        spec.dataset_name,
        name=spec.dataset_config,
        split=spec.split,
        streaming=spec.streaming,
    )

    for row in dataset:
        if spec.format == "smoltalk_messages":
            messages = row.get("messages") or []
            chat_kwargs = row.get("chat_template_kwargs") or {}
            custom_instructions = chat_kwargs.get("custom_instructions", "") if spec.use_custom_instructions else ""
            yield from iter_pairs_from_messages(
                messages,
                custom_instructions=custom_instructions,
                source_name=spec.name,
                keep_think=spec.keep_think,
                max_history_turns=max_history_turns,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_think_chars=max_think_chars,
                max_answer_chars=max_answer_chars,
            )
            continue

        if spec.format == "open_thoughts_conversations":
            conversations = row.get("conversations") or []
            messages = [
                {
                    "role": normalize_role(item.get("from", "")),
                    "content": item.get("value", ""),
                }
                for item in conversations
            ]
            yield from iter_pairs_from_messages(
                messages,
                custom_instructions=row.get("system", "") if spec.use_custom_instructions else "",
                source_name=spec.name,
                keep_think=spec.keep_think,
                max_history_turns=max_history_turns,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_think_chars=max_think_chars,
                max_answer_chars=max_answer_chars,
            )
            continue

        if spec.format == "chatml_text_think":
            custom_instructions, messages = parse_chatml_text(row.get("text", ""))
            yield from iter_pairs_from_messages(
                messages,
                custom_instructions=custom_instructions if spec.use_custom_instructions else "",
                source_name=spec.name,
                keep_think=spec.keep_think,
                max_history_turns=max_history_turns,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_think_chars=max_think_chars,
                max_answer_chars=max_answer_chars,
            )
            continue

        raise ValueError(f"Unsupported Metis SFT format: {spec.format}")


def load_mixture(path: Path) -> tuple[int, int, list[SourceSpec]]:
    payload = json.loads(path.read_text())
    seed = int(payload.get("seed", 42))
    max_history_turns = int(payload.get("max_history_turns", 2))
    sources = [SourceSpec.from_dict(item) for item in payload["sources"]]
    return seed, max_history_turns, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Metis-1.2 chat or reasoning JSONL SFT data.")
    parser.add_argument("--mixture-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-examples", type=int, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-user-chars", type=int, default=1400)
    parser.add_argument("--max-assistant-chars", type=int, default=900)
    parser.add_argument("--max-think-chars", type=int, default=700)
    parser.add_argument("--max-answer-chars", type=int, default=360)
    parser.add_argument("--progress-interval", type=int, default=2000)
    args = parser.parse_args()

    seed, max_history_turns, sources = load_mixture(Path(args.mixture_config))
    schedule = build_schedule(
        [source.weight for source in sources],
        args.total_examples,
        seed,
    )
    iterators = [
        iter(
            iter_source_examples(
                source,
                max_history_turns=max_history_turns,
                max_user_chars=args.max_user_chars,
                max_assistant_chars=args.max_assistant_chars,
                max_think_chars=args.max_think_chars,
                max_answer_chars=args.max_answer_chars,
            )
        )
        for source in sources
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    train_tmp_path = output_dir / "train.jsonl.tmp"
    val_tmp_path = output_dir / "val.jsonl.tmp"

    accepted = 0
    duplicates = 0
    exhausted = Counter()
    split_counts = Counter()
    source_counts = Counter()
    seen_hashes: set[str] = set()

    try:
        with train_tmp_path.open("w") as train_handle, val_tmp_path.open("w") as val_handle:
            for source_index in schedule:
                if accepted >= args.total_examples:
                    break

                iterator = iterators[source_index]
                try:
                    example = next(iterator)
                except StopIteration:
                    exhausted[sources[source_index].name] += 1
                    continue

                payload = json.dumps(example["messages"], ensure_ascii=False, sort_keys=True)
                digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(digest)

                target_split = split_name(payload, args.val_ratio)
                handle = val_handle if target_split == "val" else train_handle
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")

                accepted += 1
                split_counts[target_split] += 1
                source_counts[example["source"]] += 1

                if accepted == 1 or accepted % args.progress_interval == 0:
                    print(
                        f"Prepared SFT examples: {accepted}/{args.total_examples} | "
                        f"train={split_counts['train']} val={split_counts['val']} | "
                        f"latest_source={example['source']} duplicates={duplicates}",
                        flush=True,
                    )

        train_tmp_path.replace(train_path)
        val_tmp_path.replace(val_path)
    finally:
        if train_tmp_path.exists():
            train_tmp_path.unlink()
        if val_tmp_path.exists():
            val_tmp_path.unlink()

    meta = {
        "mixture_config": args.mixture_config,
        "total_examples_requested": args.total_examples,
        "train_examples": split_counts["train"],
        "val_examples": split_counts["val"],
        "duplicates": duplicates,
        "exhausted_sources": dict(exhausted),
        "source_counts": dict(source_counts),
        "max_history_turns": max_history_turns,
        "max_user_chars": args.max_user_chars,
        "max_assistant_chars": args.max_assistant_chars,
        "max_think_chars": args.max_think_chars,
        "max_answer_chars": args.max_answer_chars,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
